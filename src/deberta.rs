//! DeBERTa-v2 encoder (disentangled attention with log-bucketed relative
//! positions), matching `transformers.models.deberta_v2` in eval mode.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::DebertaConfig;

struct Layer {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    attn_dense: Linear,
    attn_norm: LayerNorm,
    intermediate: Linear,
    output_dense: Linear,
    output_norm: LayerNorm,
    /// `share_att_key` positional projections of the (fixed) relative
    /// embeddings, `[1, num_heads, 2*att_span, head_dim]`. These don't depend
    /// on the input, so they're computed once at load time instead of on
    /// every forward call.
    pos_q: Tensor,
    pos_k: Tensor,
}

pub struct DebertaV2 {
    word_embeddings: Embedding,
    embeddings_norm: LayerNorm,
    layers: Vec<Layer>,
    num_heads: usize,
    head_dim: usize,
    att_span: i64,
    bucket_size: i64,
    max_position: i64,
    device: Device,
    dtype: DType,
    metal_attention_optimizations: bool,
    #[cfg(feature = "metal")]
    fused_attention: Option<crate::metal_attention::RelativeAttentionKernel>,
}

/// `make_log_bucket_position` for a single relative distance.
fn log_bucket_position(rel: i64, bucket_size: i64, max_position: i64) -> i64 {
    let mid = bucket_size / 2;
    let abs_pos = if rel < mid && rel > -mid {
        mid - 1
    } else {
        rel.abs()
    };
    if abs_pos <= mid {
        return rel;
    }
    // Float32 arithmetic, as in the reference implementation.
    let log_pos = ((abs_pos as f32 / mid as f32).ln()
        / ((max_position - 1) as f32 / mid as f32).ln()
        * (mid - 1) as f32)
        .ceil()
        + mid as f32;
    (log_pos * rel.signum() as f32) as i64
}

fn split_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let (b, t, _) = x.dims3()?;
    x.reshape((b, t, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

impl DebertaV2 {
    pub fn load(vb: VarBuilder, cfg: &DebertaConfig) -> Result<Self> {
        Self::load_with_attention_optimizations(vb, cfg, true)
    }

    /// Attention allocation/index optimizations affect Metal only. Passing false
    /// restores the original encoder path, independently of weight loading.
    pub fn load_with_attention_optimizations(
        vb: VarBuilder,
        cfg: &DebertaConfig,
        enabled: bool,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let eps = cfg.layer_norm_eps;
        let word_embeddings = embedding(cfg.vocab_size, h, vb.pp("embeddings.word_embeddings"))?;
        let embeddings_norm = layer_norm(h, eps, vb.pp("embeddings.LayerNorm"))?;
        let enc = vb.pp("encoder");
        let att_span = cfg.pos_ebd_size();
        let rel_weight = enc.get((2 * att_span as usize, h), "rel_embeddings.weight")?;
        let rel_norm = layer_norm(h, eps, enc.pp("LayerNorm"))?;
        let rel_embeddings = rel_norm.forward(&rel_weight)?;
        let rel = rel_embeddings.unsqueeze(0)?;
        let num_heads = cfg.num_attention_heads;
        let head_dim = h / num_heads;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l = enc.pp(format!("layer.{i}"));
            let query_proj = linear(h, h, l.pp("attention.self.query_proj"))?;
            let key_proj = linear(h, h, l.pp("attention.self.key_proj"))?;
            // share_att_key: positional projections reuse the content projections
            // and don't depend on the input, so they're fixed once the weights load.
            let pos_q = split_heads(&query_proj.forward(&rel)?, num_heads, head_dim)?;
            let pos_k = split_heads(&key_proj.forward(&rel)?, num_heads, head_dim)?;
            layers.push(Layer {
                query_proj,
                key_proj,
                value_proj: linear(h, h, l.pp("attention.self.value_proj"))?,
                attn_dense: linear(h, h, l.pp("attention.output.dense"))?,
                attn_norm: layer_norm(h, eps, l.pp("attention.output.LayerNorm"))?,
                intermediate: linear(h, cfg.intermediate_size, l.pp("intermediate.dense"))?,
                output_dense: linear(cfg.intermediate_size, h, l.pp("output.dense"))?,
                output_norm: layer_norm(h, eps, l.pp("output.LayerNorm"))?,
                pos_q,
                pos_k,
            });
        }
        Ok(Self {
            word_embeddings,
            embeddings_norm,
            layers,
            num_heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
            att_span,
            bucket_size: cfg.position_buckets,
            max_position: cfg.max_relative_positions(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
            metal_attention_optimizations: enabled
                && vb.dtype() == DType::F32
                && matches!(vb.device(), Device::Metal(_)),
            #[cfg(feature = "metal")]
            fused_attention: match vb.device() {
                Device::Metal(device) if enabled && vb.dtype() == DType::F32 => Some(
                    crate::metal_attention::RelativeAttentionKernel::new(device)?,
                ),
                _ => None,
            },
        })
    }

    /// c2p and p2c gather indices, each `[1, 1, T, T]`.
    fn position_indices(&self, t: usize) -> Result<(Tensor, Tensor)> {
        let (c2p, p2c) = position_index_vectors(
            t,
            self.att_span,
            self.bucket_size,
            self.max_position,
            self.metal_attention_optimizations,
        );
        let c2p = Tensor::from_vec(c2p, (1, 1, t, t), &self.device)?;
        let p2c = Tensor::from_vec(p2c, (1, 1, t, t), &self.device)?;
        Ok((c2p, p2c))
    }

    /// `input_ids` / `attention_mask`: `[B, T]` (u32). Returns `[B, T, H]`.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let mask = attention_mask.to_dtype(self.dtype)?;
        let mut hidden = self
            .embeddings_norm
            .forward(&self.word_embeddings.forward(input_ids)?)?;
        hidden = hidden.broadcast_mul(&mask.unsqueeze(2)?)?;

        let (c2p_idx, p2c_idx) = self.position_indices(t)?;
        let pair_mask = mask
            .unsqueeze(1)?
            .broadcast_mul(&mask.unsqueeze(2)?)?
            .unsqueeze(1)?
            .to_dtype(DType::U8)?;
        // The fused F32 kernel reads base indices/masks directly. Other paths
        // retain the original expanded gathers, floor, and arithmetic kernels.
        let dense_attention = if self.metal_attention_optimizations {
            None
        } else {
            Some((
                c2p_idx.expand((b, self.num_heads, t, t))?.contiguous()?,
                p2c_idx.expand((b, self.num_heads, t, t))?.contiguous()?,
                pair_mask.expand((b, self.num_heads, t, t))?,
                attention_floor(self.dtype, (b, self.num_heads, t, t), &self.device, false)?,
            ))
        };

        let scale = ((self.head_dim * 3) as f64).sqrt();
        for layer in &self.layers {
            let q = split_heads(
                &layer.query_proj.forward(&hidden)?,
                self.num_heads,
                self.head_dim,
            )?;
            let k = split_heads(
                &layer.key_proj.forward(&hidden)?,
                self.num_heads,
                self.head_dim,
            )?;
            let v = split_heads(
                &layer.value_proj.forward(&hidden)?,
                self.num_heads,
                self.head_dim,
            )?;

            let content = q.matmul(&k.t()?)?;
            #[cfg(feature = "metal")]
            let fused_scores = if let Some(kernel) = &self.fused_attention {
                Some(kernel.forward(
                    &content,
                    &q.broadcast_matmul(&layer.pos_k.t()?)?,
                    &k.broadcast_matmul(&layer.pos_q.t()?)?,
                    &c2p_idx,
                    &p2c_idx,
                    &pair_mask,
                    (1.0 / scale) as f32,
                )?)
            } else {
                None
            };
            #[cfg(not(feature = "metal"))]
            let fused_scores: Option<Tensor> = None;
            let scores = if let Some(scores) = fused_scores {
                scores
            } else {
                let (c2p_idx, p2c_idx, pair_mask, floor) =
                    dense_attention.as_ref().ok_or_else(|| {
                        candle_core::Error::Msg("missing original attention operands".into())
                    })?;
                let c2p = q.broadcast_matmul(&layer.pos_k.t()?)?.gather(c2p_idx, 3)?;
                let p2c = k
                    .broadcast_matmul(&layer.pos_q.t()?)?
                    .gather(p2c_idx, 3)?
                    .transpose(2, 3)?;
                let scores = ((content + c2p)? + p2c)?.affine(1.0 / scale, 0.0)?;
                pair_mask.where_cond(&scores, floor)?
            };
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let context = probs.matmul(&v)?.transpose(1, 2)?.reshape((
                b,
                t,
                self.num_heads * self.head_dim,
            ))?;

            let attn_out = layer
                .attn_norm
                .forward(&(layer.attn_dense.forward(&context)? + &hidden)?)?;
            let inter = layer.intermediate.forward(&attn_out)?.gelu_erf()?;
            hidden = layer
                .output_norm
                .forward(&(layer.output_dense.forward(&inter)? + &attn_out)?)?;
        }
        Ok(hidden)
    }
}

/// c2p/p2c values preserve the original F32 bucket calculation exactly. The
/// optimized route evaluates it only for 2T-1 distances, rather than T*T pairs.
fn position_index_vectors(
    t: usize,
    span: i64,
    bucket_size: i64,
    max_position: i64,
    optimized: bool,
) -> (Vec<u32>, Vec<u32>) {
    let index = |mut rel: i64| {
        if bucket_size > 0 && max_position > 0 {
            rel = log_bucket_position(rel, bucket_size, max_position);
        }
        (
            (rel + span).clamp(0, 2 * span - 1) as u32,
            (-rel + span).clamp(0, 2 * span - 1) as u32,
        )
    };
    let mut c2p = Vec::with_capacity(t * t);
    let mut p2c = Vec::with_capacity(t * t);
    if optimized && t > 0 {
        let radius = t as i64 - 1;
        let distances: Vec<_> = (-radius..=radius).map(index).collect();
        for i in 0..t {
            for j in 0..t {
                let (c, p) = distances[i + t - 1 - j];
                c2p.push(c);
                p2c.push(p);
            }
        }
    } else {
        for i in 0..t as i64 {
            for j in 0..t as i64 {
                let (c, p) = index(i - j);
                c2p.push(c);
                p2c.push(p);
            }
        }
    }
    (c2p, p2c)
}

fn attention_floor(
    dtype: DType,
    shape: (usize, usize, usize, usize),
    device: &Device,
    broadcast: bool,
) -> Result<Tensor> {
    let min_value = match dtype {
        DType::F16 => f16_min(),
        DType::BF16 => -3.389_531_4e38,
        DType::F64 => f64::MIN,
        _ => f32::MIN as f64,
    };
    if broadcast {
        // Metal where_cond accepts zero strides. A single scalar replaces the
        // full B*H*T*T sentinel allocation and its GPU fill, without changing
        // the sentinel value, masking rule, or softmax arithmetic.
        let scalar = if dtype == DType::F64 {
            Tensor::new(min_value, device)?
        } else {
            Tensor::new(min_value as f32, device)?.to_dtype(dtype)?
        };
        scalar.broadcast_as(shape)
    } else if dtype == DType::F64 {
        Tensor::full(min_value, shape, device)
    } else {
        // Metal has no F64 storage: build non-F64 sentinels in F32 then cast.
        Tensor::full(min_value as f32, shape, device)?.to_dtype(dtype)
    }
}

fn f16_min() -> f64 {
    -65504.0
}

#[cfg(test)]
mod tests {
    use super::{attention_floor, log_bucket_position, position_index_vectors};
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn buckets_are_identity_near_zero() {
        for r in -128..=128 {
            assert_eq!(log_bucket_position(r, 256, 512), r);
        }
    }

    #[test]
    fn buckets_are_log_scaled_far_away() {
        assert_eq!(log_bucket_position(511, 256, 512), 255);
        assert_eq!(log_bucket_position(-511, 256, 512), -255);
        assert!(log_bucket_position(2000, 256, 512) > 255);
    }

    #[test]
    fn distinct_distance_buckets_match_original_for_all_pairs() {
        for t in [0, 1, 2, 17, 129, 257, 513] {
            for (span, buckets, maximum) in [(256, 256, 512), (32, 0, 32), (8, 8, 64)] {
                let original = position_index_vectors(t, span, buckets, maximum, false);
                let optimized = position_index_vectors(t, span, buckets, maximum, true);
                assert_eq!(
                    original, optimized,
                    "T={t}, buckets={buckets}, maximum={maximum}"
                );
            }
        }
    }

    fn check_padded_strided_mask(device: &Device) -> candle_core::Result<()> {
        // B=2, H=3, T=4: both padded columns and entirely masked rows, with
        // distinct masks in each batch. Attention logits are also strided.
        let shape = (2, 3, 4, 4);
        let mask = Tensor::from_vec(vec![1f32, 1., 0., 0., 1., 0., 1., 0.], (2, 4), device)?;
        let pair_mask = mask
            .unsqueeze(1)?
            .broadcast_mul(&mask.unsqueeze(2)?)?
            .unsqueeze(1)?
            .to_dtype(DType::U8)?
            .expand(shape)?;
        let scores = Tensor::from_vec(
            (0..96).map(|i| (i as f32 - 48.) / 7.).collect::<Vec<_>>(),
            (2, 4, 3, 4),
            device,
        )?
        .transpose(1, 2)?;
        assert!(!scores.is_contiguous());
        for dtype in [DType::F32, DType::F16] {
            let scores = scores.to_dtype(dtype)?;
            let full = attention_floor(dtype, shape, device, false)?;
            let scalar = attention_floor(dtype, shape, device, true)?;
            assert_eq!(scalar.stride(), &[0, 0, 0, 0]);
            let original = pair_mask.where_cond(&scores, &full)?;
            let optimized = pair_mask.where_cond(&scores, &scalar)?;
            let original_values = original
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let optimized_values = optimized
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(original_values, optimized_values);
            let mask_values = pair_mask.contiguous()?.flatten_all()?.to_vec1::<u8>()?;
            let floor = if dtype == DType::F16 {
                -65504.
            } else {
                f32::MIN
            };
            for (&condition, &value) in mask_values.iter().zip(&optimized_values) {
                if condition == 0 {
                    assert_eq!(value, floor);
                }
            }
            let actual = candle_nn::ops::softmax_last_dim(&optimized)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (input, probabilities) in
                optimized_values.chunks_exact(4).zip(actual.chunks_exact(4))
            {
                let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp: Vec<f32> = input.iter().map(|x| (x - max).exp()).collect();
                let denominator: f32 = exp.iter().sum();
                for (&p, &e) in probabilities.iter().zip(&exp) {
                    assert!(p.is_finite());
                    assert!(
                        (p - e / denominator).abs()
                            <= if dtype == DType::F16 { 0.001 } else { 1e-6 }
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn broadcast_floor_preserves_padded_attention_with_strided_logits() -> candle_core::Result<()> {
        check_padded_strided_mask(&Device::Cpu)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires an isolated Metal GPU validation window"]
    fn metal_broadcast_floor_preserves_padded_strided_attention() -> candle_core::Result<()> {
        check_padded_strided_mask(&Device::new_metal(0)?)
    }
}
