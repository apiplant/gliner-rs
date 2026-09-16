//! DeBERTa-v2 encoder (disentangled attention with log-bucketed relative
//! positions), matching `transformers.models.deberta_v2` in eval mode.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::EncoderConfig;

struct Layer {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    attn_dense: Linear,
    attn_norm: LayerNorm,
    intermediate: Linear,
    output_dense: Linear,
    output_norm: LayerNorm,
}

pub struct DebertaV2 {
    word_embeddings: Embedding,
    embeddings_norm: LayerNorm,
    layers: Vec<Layer>,
    rel_embeddings: Tensor,
    num_heads: usize,
    head_dim: usize,
    att_span: i64,
    bucket_size: i64,
    max_position: i64,
    device: Device,
    dtype: DType,
}

/// `make_log_bucket_position` for a single relative distance.
fn log_bucket_position(rel: i64, bucket_size: i64, max_position: i64) -> i64 {
    let mid = bucket_size / 2;
    let abs_pos = if rel < mid && rel > -mid { mid - 1 } else { rel.abs() };
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

impl DebertaV2 {
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let eps = cfg.layer_norm_eps;
        let word_embeddings = embedding(cfg.vocab_size, h, vb.pp("embeddings.word_embeddings"))?;
        let embeddings_norm = layer_norm(h, eps, vb.pp("embeddings.LayerNorm"))?;
        let enc = vb.pp("encoder");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l = enc.pp(format!("layer.{i}"));
            layers.push(Layer {
                query_proj: linear(h, h, l.pp("attention.self.query_proj"))?,
                key_proj: linear(h, h, l.pp("attention.self.key_proj"))?,
                value_proj: linear(h, h, l.pp("attention.self.value_proj"))?,
                attn_dense: linear(h, h, l.pp("attention.output.dense"))?,
                attn_norm: layer_norm(h, eps, l.pp("attention.output.LayerNorm"))?,
                intermediate: linear(h, cfg.intermediate_size, l.pp("intermediate.dense"))?,
                output_dense: linear(cfg.intermediate_size, h, l.pp("output.dense"))?,
                output_norm: layer_norm(h, eps, l.pp("output.LayerNorm"))?,
            });
        }
        let att_span = cfg.pos_ebd_size();
        let rel_weight = enc.get((2 * att_span as usize, h), "rel_embeddings.weight")?;
        let rel_norm = layer_norm(h, eps, enc.pp("LayerNorm"))?;
        let rel_embeddings = rel_norm.forward(&rel_weight)?;
        Ok(Self {
            word_embeddings,
            embeddings_norm,
            layers,
            rel_embeddings,
            num_heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
            att_span,
            bucket_size: cfg.position_buckets,
            max_position: cfg.max_relative_positions(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// c2p and p2c gather indices, each `[1, 1, T, T]`.
    fn position_indices(&self, t: usize) -> Result<(Tensor, Tensor)> {
        let span = self.att_span;
        let mut c2p = Vec::with_capacity(t * t);
        let mut p2c = Vec::with_capacity(t * t);
        for i in 0..t as i64 {
            for j in 0..t as i64 {
                let mut rel = i - j;
                if self.bucket_size > 0 && self.max_position > 0 {
                    rel = log_bucket_position(rel, self.bucket_size, self.max_position);
                }
                c2p.push((rel + span).clamp(0, 2 * span - 1) as u32);
                p2c.push((-rel + span).clamp(0, 2 * span - 1) as u32);
            }
        }
        let c2p = Tensor::from_vec(c2p, (1, 1, t, t), &self.device)?;
        let p2c = Tensor::from_vec(p2c, (1, 1, t, t), &self.device)?;
        Ok((c2p, p2c))
    }

    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        x.reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// `input_ids` / `attention_mask`: `[B, T]` (u32). Returns `[B, T, H]`.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let mask = attention_mask.to_dtype(self.dtype)?;
        let mut hidden = self.embeddings_norm.forward(&self.word_embeddings.forward(input_ids)?)?;
        hidden = hidden.broadcast_mul(&mask.unsqueeze(2)?)?;

        let (c2p_idx, p2c_idx) = self.position_indices(t)?;
        let c2p_idx = c2p_idx.expand((b, self.num_heads, t, t))?.contiguous()?;
        let p2c_idx = p2c_idx.expand((b, self.num_heads, t, t))?.contiguous()?;

        let pair_mask = mask
            .unsqueeze(1)?
            .broadcast_mul(&mask.unsqueeze(2)?)?
            .unsqueeze(1)?
            .to_dtype(DType::U8)?
            .expand((b, self.num_heads, t, t))?;
        let min_value = match self.dtype {
            DType::F16 => f16_min(),
            DType::BF16 => -3.389_531_4e38,
            DType::F64 => f64::MIN,
            _ => f32::MIN as f64,
        };
        let floor = Tensor::full(min_value, (b, self.num_heads, t, t), &self.device)?
            .to_dtype(self.dtype)?;

        let rel = self.rel_embeddings.unsqueeze(0)?;
        let scale = ((self.head_dim * 3) as f64).sqrt();
        for layer in &self.layers {
            let q = self.split_heads(&layer.query_proj.forward(&hidden)?)?;
            let k = self.split_heads(&layer.key_proj.forward(&hidden)?)?;
            let v = self.split_heads(&layer.value_proj.forward(&hidden)?)?;
            // share_att_key: positional projections reuse the content projections.
            let pos_q = self.split_heads(&layer.query_proj.forward(&rel)?)?;
            let pos_k = self.split_heads(&layer.key_proj.forward(&rel)?)?;

            let content = q.matmul(&k.t()?)?;
            let c2p = q
                .broadcast_matmul(&pos_k.t()?)?
                .gather(&c2p_idx, 3)?;
            let p2c = k
                .broadcast_matmul(&pos_q.t()?)?
                .gather(&p2c_idx, 3)?
                .transpose(2, 3)?;
            let scores = ((content + c2p)? + p2c)?.affine(1.0 / scale, 0.0)?;
            let scores = pair_mask.where_cond(&scores, &floor)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let context = probs
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, t, self.num_heads * self.head_dim))?;

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

fn f16_min() -> f64 {
    -65504.0
}

#[cfg(test)]
mod tests {
    use super::log_bucket_position;

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
}
