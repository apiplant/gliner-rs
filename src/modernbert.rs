//! ModernBERT encoder (alternating global/local RoPE attention, GeGLU MLP,
//! bias-free norms), matching `transformers.ModernBertModel.forward(...).last_hidden_state`
//! in eval mode. Used by the `fastino/GLiNER2.5-*-Decide*` "Decide" checkpoints
//! (the older boundary-architecture checkpoints use [`crate::deberta::DebertaV2`]).

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub norm_eps: f64,
    pub local_attention: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeParams,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub full_attention: RopeEntry,
    pub sliding_attention: RopeEntry,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeEntry {
    pub rope_theta: f64,
}

/// ModernBERT's norms are bias-free; an explicit zero bias keeps them on
/// candle's fused LayerNorm path instead of the slower bias-free fallback.
fn layer_norm_no_bias(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = Tensor::zeros(size, weight.dtype(), weight.device())?;
    Ok(LayerNorm::new(weight, bias, eps))
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// `candle_nn::Linear::forward` issues a batched GEMM on rank-3 input; flatten
/// the leading dims first so it collapses into one large GEMM instead, like
/// torch's `F.linear`.
fn linear_flat(l: &Linear, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() <= 2 {
        return l.forward(x);
    }
    let in_dim = dims[dims.len() - 1];
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let out = l.forward(&x.reshape((rows, in_dim))?)?;
    let out_dim = out.dim(D::Minus1)?;
    let mut shape = dims[..dims.len() - 1].to_vec();
    shape.push(out_dim);
    out.reshape(shape)
}

struct MlpGeglu {
    wi: Linear,
    wo: Linear,
}

impl MlpGeglu {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        let wi = linear_no_bias(hidden, intermediate * 2, vb.pp("Wi"))?;
        let wo = linear_no_bias(intermediate, hidden, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = linear_flat(&self.wi, x)?;
        let last = x.dim(D::Minus1)?;
        let half = last / 2;
        let gate = x.narrow(D::Minus1, 0, half)?.gelu_erf()?;
        let up = x.narrow(D::Minus1, half, half)?;
        linear_flat(&self.wo, &(gate * up)?)
    }
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(hidden: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let wqkv = linear_no_bias(hidden, hidden * 3, vb.pp("Wqkv"))?;
        let wo = linear_no_bias(hidden, hidden, vb.pp("Wo"))?;
        Ok(Self { wqkv, wo, n_heads, head_dim: hidden / n_heads })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let qkv = linear_flat(&self.wqkv, x)?.reshape((b, s, 3, self.n_heads, self.head_dim))?;

        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?; // [b,h,s,d]
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        let q = (q * (1f64 / (self.head_dim as f64).sqrt()))?;
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?; // [b,h,s,s]
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?; // [b,h,s,d]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, self.n_heads * self.head_dim))?;
        linear_flat(&self.wo, &out)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let half = last / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [b,h,s,d], cos/sin: [1,1,s,d]
    let a = x.broadcast_mul(cos)?;
    let b = rotate_half(x)?.broadcast_mul(sin)?;
    a + b
}

/// Matches HF ModernBert's `ModernBertRotaryEmbedding.forward`: freqs/cos/sin
/// are always computed in F32 and only cast to the compute dtype at the end,
/// right before being multiplied against Q/K.
fn rope_cos_sin(theta: f64, head_dim: usize, seq_len: usize, compute_dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half).map(|i| 1f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32)).collect();
    let inv_freq = Tensor::from_vec(inv_freq, half, device)?; // [half]
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, seq_len, device)?; // [s]
    let freqs = positions.reshape((seq_len, 1))?.broadcast_mul(&inv_freq.reshape((1, half))?)?; // [s, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [s, head_dim]
    let cos = emb.cos()?.reshape((1, 1, seq_len, head_dim))?.to_dtype(compute_dtype)?;
    let sin = emb.sin()?.reshape((1, 1, seq_len, head_dim))?.to_dtype(compute_dtype)?;
    Ok((cos, sin))
}

struct Layer {
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: MlpGeglu,
    is_global: bool,
}

pub struct ModernBert {
    tok_embeddings: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    config: ModernBertConfig,
    device: Device,
}

impl ModernBert {
    pub fn load(vb: VarBuilder, cfg: &ModernBertConfig) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let vb_emb = vb.pp("embeddings");
        let tok_weight = vb_emb.pp("tok_embeddings").get((cfg.vocab_size, hidden), "weight")?;
        let tok_embeddings = Embedding::new(tok_weight, hidden);
        let emb_norm = layer_norm_no_bias(hidden, cfg.norm_eps, vb_emb.pp("norm"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb.pp("layers").pp(i);
            let is_global = cfg.layer_types[i] == "full_attention";
            // Layer 0 skips its attention norm (its input is already the
            // embedding norm's output), matching HF's `ModernBertEncoderLayer`.
            let attn_norm = if i == 0 { None } else { Some(layer_norm_no_bias(hidden, cfg.norm_eps, vb_l.pp("attn_norm"))?) };
            let attn = Attention::new(hidden, cfg.num_attention_heads, vb_l.pp("attn"))?;
            let mlp_norm = layer_norm_no_bias(hidden, cfg.norm_eps, vb_l.pp("mlp_norm"))?;
            let mlp = MlpGeglu::new(hidden, cfg.intermediate_size, vb_l.pp("mlp"))?;
            layers.push(Layer { attn_norm, attn, mlp_norm, mlp, is_global });
        }
        let final_norm = layer_norm_no_bias(hidden, cfg.norm_eps, vb.pp("final_norm"))?;

        Ok(Self { tok_embeddings, emb_norm, layers, final_norm, config: cfg.clone(), device: vb.device().clone() })
    }

    /// `input_ids` / `attention_mask`: `[B, T]` (u32/1-0). Returns `[B, T, H]`.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_b, s) = input_ids.dims2()?;
        let mut h = self.tok_embeddings.forward(input_ids)?;
        h = self.emb_norm.forward(&h)?;
        let compute_dtype = h.dtype();

        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        let (cos_g, sin_g) = rope_cos_sin(self.config.rope_parameters.full_attention.rope_theta, head_dim, s, compute_dtype, &self.device)?;
        let (cos_l, sin_l) = rope_cos_sin(self.config.rope_parameters.sliding_attention.rope_theta, head_dim, s, compute_dtype, &self.device)?;

        let pad_mask = build_padding_mask(attention_mask)?.to_dtype(compute_dtype)?; // [b,1,1,s] additive
        let local_window = self.config.local_attention / 2;
        let sliding_mask = build_sliding_mask(s, local_window, &self.device)?.to_dtype(compute_dtype)?; // [1,1,s,s] additive
        let global_mask = pad_mask.broadcast_add(&Tensor::zeros((1, 1, s, s), compute_dtype, &self.device)?)?;
        let local_mask = pad_mask.broadcast_add(&sliding_mask)?;

        for layer in &self.layers {
            let residual = h.clone();
            let normed = match &layer.attn_norm {
                Some(n) => n.forward(&h)?,
                None => h.clone(),
            };
            let (cos, sin, mask) = if layer.is_global { (&cos_g, &sin_g, &global_mask) } else { (&cos_l, &sin_l, &local_mask) };
            let attn_out = layer.attn.forward(&normed, cos, sin, mask)?;
            h = (residual + attn_out)?;

            let residual = h.clone();
            let normed = layer.mlp_norm.forward(&h)?;
            let mlp_out = layer.mlp.forward(&normed)?;
            h = (residual + mlp_out)?;
        }
        self.final_norm.forward(&h)
    }
}

fn build_padding_mask(attention_mask: &Tensor) -> Result<Tensor> {
    // attention_mask: [b,s] -> additive mask [b,1,1,s]: 0 for keep, -inf for pad
    let (b, s) = attention_mask.dims2()?;
    let m = attention_mask.to_dtype(DType::F32)?;
    let inv = ((m * -1f64)? + 1f64)?; // 1 where pad
    let neg_inf = (inv * -6.0e4_f64)?;
    neg_inf.reshape((b, 1, 1, s))
}

fn build_sliding_mask(seq_len: usize, window: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let diff = if i > j { i - j } else { j - i };
            if diff > window {
                data[i * seq_len + j] = -6.0e4;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)
}
