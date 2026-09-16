//! Boundary head (shared candidate pool path), classifier and relation scorer.
//!
//! Dense projections run as candle tensors; the discrete top-k / dedup pool
//! logic runs on CPU vectors, which are tiny (`[Q, L+1]` and a 32x32 pool).
//! All head math is float32, like the reference implementation.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use crate::config::BoundaryHeadConfig;

/// Finite masking sentinel (`gliner2.models.boundary.constants.MASK_LOGIT`).
pub const MASK_LOGIT: f32 = -1.0e4;
const TORCH_LN_EPS: f64 = 1e-5;

fn tensor_to_vec2(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    t.to_dtype(DType::F32)?.to_vec2::<f32>()
}

fn u32_tensor(values: impl IntoIterator<Item = usize>, device: &Device) -> Result<Tensor> {
    let v: Vec<u32> = values.into_iter().map(|x| x as u32).collect();
    let n = v.len();
    Tensor::from_vec(v, n, device)
}

/// `torch.argmax`: first maximal index.
fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// Indices sorted by descending score; ties keep ascending index (stable).
fn argsort_desc(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx
}

// ---------------------------------------------------------------------------
// Boundary encoder
// ---------------------------------------------------------------------------

struct BoundaryAttentionBlock {
    norm: LayerNorm,
    qkv: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl BoundaryAttentionBlock {
    fn forward(&self, states: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let n = states.dim(0)?;
        let qkv = self
            .qkv
            .forward(&self.norm.forward(states)?)?
            .reshape((n, 3, self.heads, self.head_dim))?;
        let part = |i: usize| -> Result<Tensor> {
            qkv.narrow(1, i, 1)?.squeeze(1)?.transpose(0, 1)?.contiguous()
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let scores = q
            .matmul(&k.t()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?
            .broadcast_add(bias)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((n, self.heads * self.head_dim))?;
        states + self.output.forward(&attended)?
    }
}

struct ResidualSwiGlu {
    norm: LayerNorm,
    input_projection: Linear,
    output_projection: Linear,
    hidden: usize,
}

impl ResidualSwiGlu {
    fn forward(&self, states: &Tensor) -> Result<Tensor> {
        let projected = self.input_projection.forward(&self.norm.forward(states)?)?;
        let value = projected.narrow(D::Minus1, 0, self.hidden)?;
        let gate = projected.narrow(D::Minus1, self.hidden, self.hidden)?;
        states + self.output_projection.forward(&(value * gate.silu()?)?)?
    }
}

struct BoundaryEncoder {
    left_projection: Linear,
    right_projection: Linear,
    output_projection: Linear,
    layer_norm: LayerNorm,
    attention_blocks: Vec<BoundaryAttentionBlock>,
    refinement_blocks: Vec<ResidualSwiGlu>,
    bos_state: Tensor,
    eos_state: Tensor,
    window: usize,
}

impl BoundaryEncoder {
    fn load(vb: VarBuilder, hidden: usize, cfg: &BoundaryHeadConfig) -> Result<Self> {
        let d = cfg.boundary_dim;
        let mut attention_blocks = Vec::new();
        for i in 0..cfg.boundary_attention_layers {
            let b = vb.pp(format!("attention_blocks.{i}"));
            attention_blocks.push(BoundaryAttentionBlock {
                norm: layer_norm(d, TORCH_LN_EPS, b.pp("norm"))?,
                qkv: linear(d, 3 * d, b.pp("qkv_projection"))?,
                output: linear(d, d, b.pp("output_projection"))?,
                heads: cfg.boundary_attention_heads,
                head_dim: d / cfg.boundary_attention_heads,
            });
        }
        let ffn_hidden = ((d as f64) * cfg.boundary_ffn_multiplier).floor().max(1.0) as usize;
        let mut refinement_blocks = Vec::new();
        for i in 0..cfg.boundary_refinement_layers {
            let b = vb.pp(format!("refinement_blocks.{i}"));
            refinement_blocks.push(ResidualSwiGlu {
                norm: layer_norm(d, TORCH_LN_EPS, b.pp("norm"))?,
                input_projection: linear(d, 2 * ffn_hidden, b.pp("input_projection"))?,
                output_projection: linear(ffn_hidden, d, b.pp("output_projection"))?,
                hidden: ffn_hidden,
            });
        }
        Ok(Self {
            left_projection: linear(hidden, d, vb.pp("left_projection"))?,
            right_projection: linear(hidden, d, vb.pp("right_projection"))?,
            output_projection: linear(2 * d, d, vb.pp("output_projection"))?,
            layer_norm: layer_norm(d, TORCH_LN_EPS, vb.pp("layer_norm"))?,
            attention_blocks,
            refinement_blocks,
            bos_state: vb.get(hidden, "bos_state")?,
            eos_state: vb.get(hidden, "eos_state")?,
            window: cfg.boundary_attention_window,
        })
    }

    /// `text`: `[n, H]` -> boundary states `[n + 1, d]`.
    fn forward(&self, text: &Tensor) -> Result<Tensor> {
        let left = Tensor::cat(&[&self.bos_state.unsqueeze(0)?, text], 0)?;
        let right = Tensor::cat(&[text, &self.eos_state.unsqueeze(0)?], 0)?;
        let joined = Tensor::cat(
            &[&self.left_projection.forward(&left)?, &self.right_projection.forward(&right)?],
            1,
        )?;
        let mut states = self.layer_norm.forward(&self.output_projection.forward(&joined)?)?;
        if !self.attention_blocks.is_empty() {
            let n = states.dim(0)?;
            let mut bias = vec![0f32; n * n];
            if self.window > 0 {
                for i in 0..n {
                    for j in 0..n {
                        if i.abs_diff(j) > self.window {
                            bias[i * n + j] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let bias = Tensor::from_vec(bias, (n, n), states.device())?;
            for block in &self.attention_blocks {
                states = block.forward(&states, &bias)?;
            }
        }
        for block in &self.refinement_blocks {
            states = block.forward(&states)?;
        }
        Ok(states)
    }
}

// ---------------------------------------------------------------------------
// Query-conditioned marginals
// ---------------------------------------------------------------------------

struct BoundaryQueryHead {
    start_boundary: Linear,
    start_query: Linear,
    end_boundary: Linear,
    end_query: Linear,
    inside_text: Linear,
    inside_query: Linear,
    scale: f64,
}

struct Marginals {
    start: Vec<Vec<f32>>,         // [Q, N]
    end: Vec<Vec<f32>>,           // [Q, N]
    inside_prefix: Vec<Vec<f32>>, // [Q, N] (centered cumulative sums)
    inside_mean: Vec<f32>,        // [Q]
}

impl BoundaryQueryHead {
    fn load(vb: VarBuilder, hidden: usize, d: usize) -> Result<Self> {
        Ok(Self {
            start_boundary: linear(d, d, vb.pp("start_boundary_projection"))?,
            start_query: linear(hidden, d, vb.pp("start_query_projection"))?,
            end_boundary: linear(d, d, vb.pp("end_boundary_projection"))?,
            end_query: linear(hidden, d, vb.pp("end_query_projection"))?,
            inside_text: linear(hidden, d, vb.pp("inside_text_projection"))?,
            inside_query: linear(hidden, d, vb.pp("inside_query_projection"))?,
            scale: 1.0 / (d as f64).sqrt(),
        })
    }

    fn forward(&self, boundary: &Tensor, text: &Tensor, query: &Tensor) -> Result<Marginals> {
        let score = |q: &Linear, k: &Linear, keys: &Tensor| -> Result<Vec<Vec<f32>>> {
            let logits = q
                .forward(query)?
                .matmul(&k.forward(keys)?.t()?)?
                .affine(self.scale, 0.0)?;
            tensor_to_vec2(&logits)
        };
        let start = score(&self.start_query, &self.start_boundary, boundary)?;
        let end = score(&self.end_query, &self.end_boundary, boundary)?;
        let inside = score(&self.inside_query, &self.inside_text, text)?;

        let mut inside_prefix = Vec::with_capacity(inside.len());
        let mut inside_mean = Vec::with_capacity(inside.len());
        for row in &inside {
            let count = row.len().max(1) as f32;
            let mean = row.iter().sum::<f32>() / count;
            let mut prefix = Vec::with_capacity(row.len() + 1);
            let mut acc = 0f32;
            prefix.push(0.0);
            for &x in row {
                acc += x - mean;
                prefix.push(acc);
            }
            inside_prefix.push(prefix);
            inside_mean.push(mean);
        }
        Ok(Marginals { start, end, inside_prefix, inside_mean })
    }
}

// ---------------------------------------------------------------------------
// Shared document candidate pool
// ---------------------------------------------------------------------------

struct DocumentCandidatePool {
    start_projection: Linear,
    end_projection: Linear,
    boundary_dim: usize,
    top_k: usize,
    pool_size: usize,
    min_pool_per_query: usize,
}

/// A retained candidate span `[start, end)` over word boundaries.
#[derive(Debug, Clone, Copy)]
struct PooledCandidate {
    start: usize,
    end: usize,
    compat: f32,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl DocumentCandidatePool {
    fn build(&self, boundary: &Tensor, marginals: &Marginals) -> Result<Vec<PooledCandidate>> {
        let n = boundary.dim(0)?;
        let q = marginals.start.len();
        let start_all = tensor_to_vec2(&self.start_projection.forward(boundary)?)?;
        let end_all = tensor_to_vec2(&self.end_projection.forward(boundary)?)?;
        let inv_sqrt_d = 1.0 / (self.boundary_dim as f32).sqrt();

        let union = |rows: &[Vec<f32>]| -> Vec<f32> {
            (0..n)
                .map(|i| rows.iter().map(|r| r[i]).fold(MASK_LOGIT, f32::max))
                .collect()
        };
        let union_start = union(&marginals.start);
        let union_end = union(&marginals.end);

        let k = self.top_k.min(n);
        let starts: Vec<usize> = argsort_desc(&union_start).into_iter().take(k).collect();
        let ends: Vec<usize> = argsort_desc(&union_end).into_iter().take(k).collect();

        // Cartesian pairs in (start rank, end rank) order.
        let mut pair_s = Vec::with_capacity(k * k);
        let mut pair_e = Vec::with_capacity(k * k);
        let mut pair_valid = Vec::with_capacity(k * k);
        let mut compat = Vec::with_capacity(k * k);
        for &s in &starts {
            for &e in &ends {
                pair_s.push(s);
                pair_e.push(e);
                pair_valid.push(e > s);
                compat.push(dot(&start_all[s], &end_all[e]) * inv_sqrt_d);
            }
        }
        let pairs = pair_s.len();

        let mut keys: Vec<usize> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut valid: Vec<bool> = Vec::new();

        // Per-query quota, ranked above every ordinary global score.
        let quota = self.min_pool_per_query.min(pairs);
        if quota > 0 {
            for qi in 0..q {
                let per_query: Vec<f32> = (0..pairs)
                    .map(|p| {
                        if pair_valid[p] {
                            marginals.start[qi][pair_s[p]] + marginals.end[qi][pair_e[p]] + compat[p]
                        } else {
                            MASK_LOGIT
                        }
                    })
                    .collect();
                for (rank, p) in argsort_desc(&per_query).into_iter().take(quota).enumerate() {
                    keys.push(pair_s[p] * n + pair_e[p]);
                    valid.push(pair_valid[p]);
                    scores.push(-MASK_LOGIT * 0.5 + (quota - rank) as f32);
                }
            }
        }
        for p in 0..pairs {
            keys.push(pair_s[p] * n + pair_e[p]);
            valid.push(pair_valid[p]);
            scores.push(compat[p] + union_start[pair_s[p]] + union_end[pair_e[p]]);
        }

        let selected = deduplicate_pool(&keys, &scores, &valid, self.pool_size, n);
        Ok(selected
            .into_iter()
            .map(|key| {
                let (start, end) = (key / n, key % n);
                PooledCandidate { start, end, compat: dot(&start_all[start], &end_all[end]) * inv_sqrt_d }
            })
            .collect())
    }
}

/// `_deduplicate_pool`: keep the best-scored occurrence of every key and return
/// up to `capacity` valid keys ordered by score (ties by ascending key).
fn deduplicate_pool(keys: &[usize], scores: &[f32], valid: &[bool], capacity: usize, n: usize) -> Vec<usize> {
    let invalid_key = n * n;
    let keys: Vec<usize> = keys.iter().zip(valid).map(|(&k, &v)| if v { k } else { invalid_key }).collect();
    let scores: Vec<f32> = scores.iter().zip(valid).map(|(&s, &v)| if v { s } else { MASK_LOGIT }).collect();
    // Score-descending, then stably key-ascending: the first row of each key
    // group is its best-scored occurrence.
    let mut order = argsort_desc(&scores);
    order.sort_by_key(|&i| keys[i]);
    let keep_flags: Vec<bool> = order
        .iter()
        .enumerate()
        .map(|(pos, &i)| valid[i] && (pos == 0 || keys[i] != keys[order[pos - 1]]))
        .collect();
    let kept_scores: Vec<f32> = order
        .iter()
        .zip(&keep_flags)
        .map(|(&i, &keep)| if keep { scores[i] } else { MASK_LOGIT })
        .collect();
    argsort_desc(&kept_scores)
        .into_iter()
        .take(capacity)
        .filter(|&pos| keep_flags[pos])
        .map(|pos| keys[order[pos]])
        .collect()
}

// ---------------------------------------------------------------------------
// Shared pool scorer
// ---------------------------------------------------------------------------

struct SpanContentPooler {
    value_projection: Linear,
    layer_norm: LayerNorm,
}

struct SharedPoolScorer {
    start_projection: Linear,
    end_projection: Linear,
    length_projection: Linear,
    prior_projection: Linear,
    content: Option<(SpanContentPooler, Linear)>,
    candidate_norm: LayerNorm,
    query_projection: Linear,
    film: Linear,
    film_hidden: Linear,
    film_out: Linear,
    pair_dim: usize,
}

impl SharedPoolScorer {
    fn load(vb: VarBuilder, hidden: usize, cfg: &BoundaryHeadConfig) -> Result<Self> {
        let d = cfg.boundary_dim;
        let p = cfg.pair_dim;
        let content = if cfg.enable_span_content {
            let pooler = SpanContentPooler {
                value_projection: linear(hidden, cfg.content_dim, vb.pp("content_pooler.value_projection"))?,
                layer_norm: layer_norm(cfg.content_dim, TORCH_LN_EPS, vb.pp("content_pooler.layer_norm"))?,
            };
            Some((pooler, linear(cfg.content_dim, p, vb.pp("content_projection"))?))
        } else {
            None
        };
        Ok(Self {
            start_projection: linear(d, p, vb.pp("start_projection"))?,
            end_projection: linear(d, p, vb.pp("end_projection"))?,
            length_projection: linear(3, p, vb.pp("length_projection"))?,
            prior_projection: linear(1, p, vb.pp("prior_projection"))?,
            content,
            candidate_norm: layer_norm(p, TORCH_LN_EPS, vb.pp("candidate_norm"))?,
            query_projection: linear(hidden, p, vb.pp("query_projection"))?,
            film: linear(p, 2 * p, vb.pp("film"))?,
            film_hidden: linear(p, 64, vb.pp("film_output.0"))?,
            film_out: linear(64, 1, vb.pp("film_output.3"))?,
            pair_dim: p,
        })
    }

    /// Returns the learned part of the pair score, `[C, Q]`.
    fn forward(
        &self,
        boundary: &Tensor,
        text: &Tensor,
        query: &Tensor,
        candidates: &[PooledCandidate],
    ) -> Result<Vec<Vec<f32>>> {
        let device = boundary.device();
        let c = candidates.len();
        let n_words = text.dim(0)?;
        let s_idx = u32_tensor(candidates.iter().map(|x| x.start), device)?;
        let e_idx = u32_tensor(candidates.iter().map(|x| x.end), device)?;

        let start_rep = self.start_projection.forward(boundary)?.index_select(&s_idx, 0)?;
        let end_rep = self.end_projection.forward(boundary)?.index_select(&e_idx, 0)?;
        let tl = n_words.max(1) as f32;
        let mut length_features = Vec::with_capacity(c * 3);
        let mut lengths = Vec::with_capacity(c);
        for cand in candidates {
            let length = (cand.end.saturating_sub(cand.start)).max(1) as f32;
            length_features.extend([length.ln_1p(), length / tl, 1.0 / length.sqrt()]);
            lengths.push(length);
        }
        let length_features = Tensor::from_vec(length_features, (c, 3), device)?;
        let prior = Tensor::from_vec(candidates.iter().map(|x| x.compat).collect::<Vec<_>>(), (c, 1), device)?;

        let mut candidate = (((start_rep + end_rep)? + self.length_projection.forward(&length_features)?)?
            + self.prior_projection.forward(&prior)?)?;

        if let Some((pooler, projection)) = &self.content {
            let values = pooler.value_projection.forward(text)?;
            let zeros = Tensor::zeros((1, values.dim(1)?), DType::F32, device)?;
            let prefix = Tensor::cat(&[&zeros, &values.cumsum(0)?], 0)?;
            let lengths = Tensor::from_vec(lengths, (c, 1), device)?;
            let pooled = (prefix.index_select(&e_idx, 0)? - prefix.index_select(&s_idx, 0)?)?
                .broadcast_div(&lengths)?;
            let pooled = pooler.layer_norm.forward(&pooled)?;
            candidate = (candidate + projection.forward(&pooled)?)?;
        }
        let candidate = self.candidate_norm.forward(&candidate)?;

        let query = self.query_projection.forward(query)?; // [Q, P]
        let base = candidate
            .matmul(&query.t()?)?
            .affine(1.0 / (self.pair_dim as f64).sqrt(), 0.0)?; // [C, Q]

        let film = self.film.forward(&query)?;
        let gamma = film.narrow(1, 0, self.pair_dim)?;
        let beta = film.narrow(1, self.pair_dim, self.pair_dim)?;
        let conditioned = candidate
            .unsqueeze(1)?
            .broadcast_mul(&(gamma + 1.0)?.unsqueeze(0)?)?
            .broadcast_add(&beta.unsqueeze(0)?)?; // [C, Q, P]
        let film_score = self
            .film_out
            .forward(&self.film_hidden.forward(&conditioned)?.gelu_erf()?)?
            .squeeze(2)?;
        tensor_to_vec2(&(base + film_score)?)
    }
}

// ---------------------------------------------------------------------------
// Per-query pair reranker (explicit span scoring)
// ---------------------------------------------------------------------------

/// Rotate `states` (interleaved even/odd pairs) by boundary position, like
/// `RotaryBoundaryEmbedding` (float32 angles).
fn rotate(states: &mut [f32], position: usize, base: f32) {
    let dim = states.len();
    for i in 0..dim / 2 {
        let inv_freq = 1.0 / base.powf((2 * i) as f32 / dim as f32);
        let angle = position as f32 * inv_freq;
        let (cos, sin) = (angle.cos(), angle.sin());
        let (even, odd) = (states[2 * i], states[2 * i + 1]);
        states[2 * i] = even * cos - odd * sin;
        states[2 * i + 1] = even * sin + odd * cos;
    }
}

/// `sigmoid(gate)` repeated element-wise twice (`repeat_interleave(2)`).
fn interleaved_gate(logits: &[f32]) -> Vec<f32> {
    logits.iter().flat_map(|&x| {
        let g = 1.0 / (1.0 + (-x).exp());
        [g, g]
    }).collect()
}

/// `SparseBoundaryProposer.score_explicit_pairs` weights (marginal-free prior).
struct ExplicitProposer {
    start_pair_projection: Linear,
    end_key_projection: Linear,
    start_query_projection: Linear,
}

/// `SparseBoundaryPairScorer` (rotary, multi-head compat, difference, content,
/// query-conditioned inside weight, length features).
struct PairScorer {
    start_endpoint_projection: Linear,
    end_endpoint_projection: Linear,
    query_gate: Linear,
    length_query_projection: Linear,
    inside_weight: Linear,
    endpoint_difference_projection: Linear,
    content_value_projection: Linear,
    content_layer_norm: LayerNorm,
    content_query_projection: Linear,
    content_bias: Linear,
    compat_mix: Linear,
    compat_heads: usize,
    pair_dim: usize,
}

impl PairScorer {
    fn load(vb: VarBuilder, hidden: usize, cfg: &BoundaryHeadConfig) -> Result<Self> {
        let (d, p, c) = (cfg.boundary_dim, cfg.pair_dim, cfg.content_dim);
        Ok(Self {
            start_endpoint_projection: linear(d, p, vb.pp("start_endpoint_projection"))?,
            end_endpoint_projection: linear(d, p, vb.pp("end_endpoint_projection"))?,
            query_gate: linear(hidden, p / 2, vb.pp("query_gate"))?,
            length_query_projection: linear(hidden, 3, vb.pp("length_query_projection"))?,
            inside_weight: linear(hidden, 1, vb.pp("inside_weight"))?,
            endpoint_difference_projection: linear(2 * p, 1, vb.pp("endpoint_difference_projection"))?,
            content_value_projection: linear(hidden, c, vb.pp("content_pooler.value_projection"))?,
            content_layer_norm: layer_norm(c, TORCH_LN_EPS, vb.pp("content_pooler.layer_norm"))?,
            content_query_projection: linear(hidden, c, vb.pp("content_query_projection"))?,
            content_bias: linear(c, 1, vb.pp("content_bias"))?,
            compat_mix: linear(cfg.multihead_pair_compat_heads, 1, vb.pp("compat_mix"))?,
            compat_heads: cfg.multihead_pair_compat_heads,
            pair_dim: p,
        })
    }
}

// ---------------------------------------------------------------------------
// Boundary head
// ---------------------------------------------------------------------------

/// Candidate spans shared by every query plus their per-query logits.
#[derive(Debug, Clone)]
pub struct CandidateScores {
    /// Half-open token spans `[start, end)` over the (prefix + text) words.
    pub spans: Vec<(usize, usize)>,
    /// `[Q][C]` pair logits.
    pub pair_logits: Vec<Vec<f32>>,
    /// `[Q]` abstention logits, when the checkpoint has a null head.
    pub null_logits: Option<Vec<f32>>,
}

/// Boundary head outputs plus the intermediate state needed by record decoding
/// and explicit span scoring.
pub struct BoundaryForward {
    pub scores: CandidateScores,
    boundary: Tensor,
    marginals: Marginals,
}

pub struct BoundaryHead {
    boundary_encoder: BoundaryEncoder,
    query_head: BoundaryQueryHead,
    pool: DocumentCandidatePool,
    scorer: SharedPoolScorer,
    null_projection: Option<Linear>,
    candidate_encoder: Option<Linear>,
    proposer: ExplicitProposer,
    pair_scorer: PairScorer,
    use_inside_evidence: bool,
    boundary_dim: usize,
    rotary_base: f32,
}

impl BoundaryHead {
    pub fn load(vb: VarBuilder, hidden: usize, cfg: &BoundaryHeadConfig) -> Result<Self> {
        let d = cfg.boundary_dim;
        Ok(Self {
            boundary_encoder: BoundaryEncoder::load(vb.pp("boundary_encoder"), hidden, cfg)?,
            query_head: BoundaryQueryHead::load(vb.pp("boundary_query_head"), hidden, d)?,
            pool: DocumentCandidatePool {
                start_projection: linear(d, d, vb.pp("shared_pool_builder.start_projection"))?,
                end_projection: linear(d, d, vb.pp("shared_pool_builder.end_projection"))?,
                boundary_dim: d,
                top_k: cfg.pool_boundary_top_k,
                pool_size: cfg.pool_size,
                min_pool_per_query: cfg.min_pool_per_query,
            },
            scorer: SharedPoolScorer::load(vb.pp("shared_pool_scorer"), hidden, cfg)?,
            null_projection: if cfg.enable_abstention {
                Some(linear(hidden, 1, vb.pp("null_projection"))?)
            } else {
                None
            },
            candidate_encoder: if cfg.enable_records {
                Some(linear(2 * d, hidden, vb.pp("candidate_encoder"))?)
            } else {
                None
            },
            proposer: ExplicitProposer {
                start_pair_projection: linear(d, d, vb.pp("boundary_proposer.start_pair_projection"))?,
                end_key_projection: linear(d, d, vb.pp("boundary_proposer.end_key_projection"))?,
                start_query_projection: linear(hidden, d / 2, vb.pp("boundary_proposer.start_query_projection"))?,
            },
            pair_scorer: PairScorer::load(vb.pp("pair_scorer"), hidden, cfg)?,
            use_inside_evidence: cfg.use_inside_evidence,
            boundary_dim: d,
            rotary_base: cfg.rotary_base,
        })
    }

    /// `text`: `[n, H]` word states (n >= 1), `query`: `[Q, H]` (Q >= 1).
    pub fn forward(&self, text: &Tensor, query: &Tensor) -> Result<BoundaryForward> {
        let boundary = self.boundary_encoder.forward(text)?;
        let marginals = self.query_head.forward(&boundary, text, query)?;
        let pooled = self.pool.build(&boundary, &marginals)?;
        let q = marginals.start.len();

        let learned = if pooled.is_empty() {
            Vec::new()
        } else {
            self.scorer.forward(&boundary, text, query, &pooled)?
        };
        let mut pair_logits = vec![Vec::with_capacity(pooled.len()); q];
        for (ci, cand) in pooled.iter().enumerate() {
            let (s, e) = (cand.start, cand.end);
            for (qi, row) in pair_logits.iter_mut().enumerate() {
                let score = learned[ci][qi]
                    + marginals.start[qi][s]
                    + marginals.end[qi][e]
                    + self.inside_term(&marginals, qi, s, e);
                row.push(score);
            }
        }

        let null_logits = match &self.null_projection {
            Some(proj) => Some(proj.forward(query)?.squeeze(1)?.to_dtype(DType::F32)?.to_vec1::<f32>()?),
            None => None,
        };
        Ok(BoundaryForward {
            scores: CandidateScores {
                spans: pooled.iter().map(|c| (c.start, c.end)).collect(),
                pair_logits,
                null_logits,
            },
            boundary,
            marginals,
        })
    }

    fn inside_term(&self, marginals: &Marginals, qi: usize, s: usize, e: usize) -> f32 {
        if !self.use_inside_evidence {
            return 0.0;
        }
        let prefix = &marginals.inside_prefix[qi];
        let interval = prefix[e] - prefix[s] + marginals.inside_mean[qi] * (e - s) as f32;
        interval / ((e.saturating_sub(s)).max(1) as f32).sqrt()
    }

    /// Record-head candidate states `[C, H]` (`candidate_encoder(cat(start, end))`).
    pub fn candidate_states(&self, fwd: &BoundaryForward) -> Result<Option<Tensor>> {
        let Some(encoder) = &self.candidate_encoder else { return Ok(None) };
        let spans = &fwd.scores.spans;
        if spans.is_empty() {
            return Ok(None);
        }
        let device = fwd.boundary.device();
        let starts = fwd.boundary.index_select(&u32_tensor(spans.iter().map(|s| s.0), device)?, 0)?;
        let ends = fwd.boundary.index_select(&u32_tensor(spans.iter().map(|s| s.1), device)?, 0)?;
        Ok(Some(encoder.forward(&Tensor::cat(&[&starts, &ends], 1)?)?))
    }

    /// `BoundaryHead.score_explicit_spans` for one query: pair logits for
    /// caller-provided half-open spans, bypassing sparse proposal.
    pub fn score_explicit_spans(
        &self,
        fwd: &BoundaryForward,
        text: &Tensor,
        query: &Tensor,
        query_index: usize,
        spans: &[(usize, usize)],
    ) -> Result<Vec<f32>> {
        if spans.is_empty() {
            return Ok(Vec::new());
        }
        let device = fwd.boundary.device();
        let n_boundaries = fwd.boundary.dim(0)?;
        let n_words = text.dim(0)?;
        let c = spans.len();
        let q_row = query.get(query_index)?.unsqueeze(0)?; // [1, H]
        let clamp = |x: usize| x.min(n_boundaries - 1);
        let s_idx = u32_tensor(spans.iter().map(|s| clamp(s.0)), device)?;
        let e_idx = u32_tensor(spans.iter().map(|s| clamp(s.1)), device)?;
        let b_start = fwd.boundary.index_select(&s_idx, 0)?;
        let b_end = fwd.boundary.index_select(&e_idx, 0)?;

        let rotated = |proj: &Linear, states: &Tensor, positions: &mut dyn Iterator<Item = usize>| -> Result<Vec<Vec<f32>>> {
            let mut rows = tensor_to_vec2(&proj.forward(states)?)?;
            for (row, pos) in rows.iter_mut().zip(positions) {
                rotate(row, pos, self.rotary_base);
            }
            Ok(rows)
        };
        let row_vec = |t: Tensor| -> Result<Vec<f32>> { t.squeeze(0)?.to_vec1::<f32>() };

        // Proposal prior (marginal-free compatibility).
        let prop = &self.proposer;
        let ps = rotated(&prop.start_pair_projection, &b_start, &mut spans.iter().map(|s| clamp(s.0)))?;
        let pe = rotated(&prop.end_key_projection, &b_end, &mut spans.iter().map(|s| clamp(s.1)))?;
        let prop_gate = interleaved_gate(&row_vec(prop.start_query_projection.forward(&q_row)?)?);
        let inv_sqrt_d = 1.0 / (self.boundary_dim as f32).sqrt();

        // Reranker endpoints.
        let ps_ = &self.pair_scorer;
        let ss = rotated(&ps_.start_endpoint_projection, &b_start, &mut spans.iter().map(|s| clamp(s.0)))?;
        let se = rotated(&ps_.end_endpoint_projection, &b_end, &mut spans.iter().map(|s| clamp(s.1)))?;
        let gate = interleaved_gate(&row_vec(ps_.query_gate.forward(&q_row)?)?);
        let p = ps_.pair_dim;
        let head_width = p / ps_.compat_heads;

        let mut per_head = Vec::with_capacity(c * ps_.compat_heads);
        let mut difference = Vec::with_capacity(c * 2 * p);
        let mut prior = Vec::with_capacity(c);
        for i in 0..c {
            for h in 0..ps_.compat_heads {
                let lo = h * head_width;
                per_head.push((lo..lo + head_width).map(|k| ss[i][k] * gate[k] * se[i][k]).sum::<f32>());
            }
            difference.extend((0..p).map(|k| ss[i][k] - se[i][k]));
            difference.extend((0..p).map(|k| (ss[i][k] - se[i][k]).abs()));
            prior.push((0..self.boundary_dim).map(|k| ps[i][k] * prop_gate[k] * pe[i][k]).sum::<f32>() * inv_sqrt_d);
        }
        let per_head = Tensor::from_vec(per_head, (c, ps_.compat_heads), device)?;
        let compat = ps_.compat_mix.forward(&per_head)?.squeeze(1)?.to_vec1::<f32>()?;
        let difference = Tensor::from_vec(difference, (c, 2 * p), device)?;
        let difference = ps_.endpoint_difference_projection.forward(&difference)?.squeeze(1)?.to_vec1::<f32>()?;

        // Span content.
        let values = ps_.content_value_projection.forward(text)?;
        let zeros = Tensor::zeros((1, values.dim(1)?), DType::F32, device)?;
        let prefix = Tensor::cat(&[&zeros, &values.cumsum(0)?], 0)?;
        let lengths: Vec<f32> = spans.iter().map(|s| s.1.saturating_sub(s.0).max(1) as f32).collect();
        let length_t = Tensor::from_vec(lengths.clone(), (c, 1), device)?;
        let content = (prefix.index_select(&e_idx, 0)? - prefix.index_select(&s_idx, 0)?)?.broadcast_div(&length_t)?;
        let content = ps_.content_layer_norm.forward(&content)?;
        let content_bias = ps_.content_bias.forward(&content)?.squeeze(1)?.to_vec1::<f32>()?;
        let content = tensor_to_vec2(&content)?;
        let coefficient = row_vec(ps_.content_query_projection.forward(&q_row)?)?;
        let content_scale = 1.0 / (coefficient.len() as f32).sqrt();

        let inside_weight = row_vec(ps_.inside_weight.forward(&q_row)?)?[0];
        let length_coeff = row_vec(ps_.length_query_projection.forward(&q_row)?)?;
        let tl = n_words.max(1) as f32;
        let m = &fwd.marginals;
        let inv_sqrt_p = 1.0 / (p as f32).sqrt();

        Ok((0..c)
            .map(|i| {
                let (s, e) = (clamp(spans[i].0), clamp(spans[i].1));
                let mut score = compat[i] * inv_sqrt_p + difference[i];
                score += m.start[query_index][s] + m.end[query_index][e] + prior[i];
                score += content[i].iter().zip(&coefficient).map(|(a, b)| a * b).sum::<f32>() * content_scale;
                score += content_bias[i];
                if self.use_inside_evidence {
                    let pre = &m.inside_prefix[query_index];
                    let interval = pre[e] - pre[s] + m.inside_mean[query_index] * (e as f32 - s as f32);
                    score += inside_weight * (interval / lengths[i].sqrt());
                }
                let len = lengths[i];
                let feats = [len.ln_1p(), len / tl, 1.0 / len.sqrt()];
                score += feats.iter().zip(&length_coeff).map(|(a, b)| a * b).sum::<f32>();
                score
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Classification MLP
// ---------------------------------------------------------------------------

pub struct Classifier {
    hidden: Linear,
    output: Linear,
}

impl Classifier {
    pub fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            hidden: linear(hidden, 2 * hidden, vb.pp("0"))?,
            output: linear(2 * hidden, 1, vb.pp("3"))?,
        })
    }

    /// Same MLP shape, but the span architecture's `create_mlp` was built with
    /// `dropout=0`, so there's no `Dropout` submodule between the two
    /// `Linear`s and the output layer is index `2`, not `3`.
    pub fn load_span(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            hidden: linear(hidden, 2 * hidden, vb.pp("0"))?,
            output: linear(2 * hidden, 1, vb.pp("2"))?,
        })
    }

    /// `states`: `[K, H]` label-marker states -> `K` logits.
    pub fn forward(&self, states: &Tensor) -> Result<Vec<f32>> {
        self.output
            .forward(&self.hidden.forward(states)?.relu()?)?
            .squeeze(1)?
            .to_vec1::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Span (legacy) architecture heads
// ---------------------------------------------------------------------------
//
// Ported from `gliner2.layers` (`CountLSTM`, `SpanMarkerV0`) and
// `gliner2.models.span.model.SpanExtractorModel`. Only what `extract_entities`
// needs: `count_pred` gates whether there's anything to extract, `count_embed`
// (`CountLSTM` only — see `SpanConfig::counting_layer`) and `span_rep` produce
// the per-span logits.

/// `Linear -> ReLU -> Linear` (no final activation), the MLP shape shared by
/// `count_pred`, `count_embed.projector` and the span-rep projections.
struct TwoLayerMlp {
    hidden: Linear,
    output: Linear,
}

impl TwoLayerMlp {
    fn load(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize, idx1: &str, idx2: &str) -> Result<Self> {
        Ok(Self { hidden: linear(in_dim, hidden_dim, vb.pp(idx1))?, output: linear(hidden_dim, out_dim, vb.pp(idx2))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.output.forward(&self.hidden.forward(x)?.relu()?)
    }
}

/// `count_pred`: predicts how many instances to extract from the `[P]` state.
pub struct CountPred(TwoLayerMlp);

impl CountPred {
    pub fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self(TwoLayerMlp::load(vb, hidden, 2 * hidden, 20, "0", "2")?))
    }

    /// `prompt_state`: `[H]` -> the predicted count (argmax over 20 classes, capped at 19).
    pub fn predict(&self, prompt_state: &Tensor) -> Result<usize> {
        let logits = self.0.forward(&prompt_state.unsqueeze(0)?)?.squeeze(0)?.to_vec1::<f32>()?;
        Ok(argmax(&logits))
    }
}

/// `count_embed` (`CountLSTM` only). Unrolling the reference GRU past step 0
/// only feeds later "count slots" that `extract_entities` never reads (see
/// `_extract_entities` in the Python reference, which always indexes count
/// step 0): step 0 depends only on the field embeddings and `pos_embedding`
/// row 0, not on how many steps the caller asks for, so it's enough to run
/// exactly one GRU step here instead of reproducing the full unroll.
pub struct CountLstmStep0 {
    pos0: Tensor,   // `pos_embedding.weight[0]`, `[H]`
    w_ih: Tensor,   // `[3H, H]`
    w_hh: Tensor,   // `[3H, H]`
    b_ih: Tensor,   // `[3H]`
    b_hh: Tensor,   // `[3H]`
    projector: TwoLayerMlp,
    hidden: usize,
}

impl CountLstmStep0 {
    pub fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        let pos_embedding = vb.pp("pos_embedding").get((20, hidden), "weight")?;
        let pos0 = pos_embedding.narrow(0, 0, 1)?.squeeze(0)?;
        let gru = vb.pp("gru");
        Ok(Self {
            pos0,
            w_ih: gru.get((3 * hidden, hidden), "weight_ih_l0")?,
            w_hh: gru.get((3 * hidden, hidden), "weight_hh_l0")?,
            b_ih: gru.get(3 * hidden, "bias_ih_l0")?,
            b_hh: gru.get(3 * hidden, "bias_hh_l0")?,
            projector: TwoLayerMlp::load(vb.pp("projector"), 2 * hidden, 4 * hidden, hidden, "0", "2")?,
            hidden,
        })
    }

    /// `field_states`: `[M, H]` per-label `[E]`/`[C]`/`[R]` states -> `[M, H]`
    /// count-aware projections (`struct_proj` at count step 0).
    pub fn forward(&self, field_states: &Tensor) -> Result<Tensor> {
        let m = field_states.dim(0)?;
        let pos_seq = self.pos0.unsqueeze(0)?.broadcast_as((m, self.hidden))?;
        let gi = (pos_seq.matmul(&self.w_ih.t()?)? + self.b_ih.broadcast_as((m, 3 * self.hidden))?)?;
        let gh = (field_states.matmul(&self.w_hh.t()?)? + self.b_hh.broadcast_as((m, 3 * self.hidden))?)?;
        let chunk = |t: &Tensor, i: usize| -> Result<Tensor> { t.narrow(1, i * self.hidden, self.hidden) };
        let (i_r, i_z, i_n) = (chunk(&gi, 0)?, chunk(&gi, 1)?, chunk(&gi, 2)?);
        let (h_r, h_z, h_n) = (chunk(&gh, 0)?, chunk(&gh, 1)?, chunk(&gh, 2)?);
        let r = candle_nn::ops::sigmoid(&(i_r + h_r)?)?;
        let z = candle_nn::ops::sigmoid(&(i_z + h_z)?)?;
        let n = (i_n + (r * h_n)?)?.tanh()?;
        let one_minus_z = z.affine(-1.0, 1.0)?;
        let h1 = ((one_minus_z * n)? + (z * field_states)?)?;
        self.projector.forward(&Tensor::cat(&[&h1, field_states], 1)?)
    }
}

/// `span_rep` (`SpanMarkerV0`): per-`(start, width)` span representations
/// from start/end token projections.
pub struct SpanRep {
    project_start: TwoLayerMlp,
    project_end: TwoLayerMlp,
    out_project: TwoLayerMlp,
    max_width: usize,
}

impl SpanRep {
    pub fn load(vb: VarBuilder, hidden: usize, max_width: usize) -> Result<Self> {
        let vb = vb.pp("span_rep_layer");
        Ok(Self {
            project_start: TwoLayerMlp::load(vb.pp("project_start"), hidden, 4 * hidden, hidden, "0", "3")?,
            project_end: TwoLayerMlp::load(vb.pp("project_end"), hidden, 4 * hidden, hidden, "0", "3")?,
            out_project: TwoLayerMlp::load(vb.pp("out_project"), 2 * hidden, 4 * hidden, hidden, "0", "3")?,
            max_width,
        })
    }

    /// `token_states`: `[L, H]` text-word states. Returns span representations
    /// `[N, H]` for every valid `(start, width)` with `start + width < L`, in
    /// the same order as the returned `(start, width)` list.
    pub fn compute(&self, token_states: &Tensor, device: &Device) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let l = token_states.dim(0)?;
        let hidden = token_states.dim(1)?;
        let start_rep = self.project_start.forward(token_states)?;
        let end_rep = self.project_end.forward(token_states)?;

        let mut pairs = Vec::new();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for s in 0..l {
            for w in 0..self.max_width {
                let e = s + w;
                if e < l {
                    pairs.push((s, w));
                    starts.push(s as u32);
                    ends.push(e as u32);
                }
            }
        }
        if pairs.is_empty() {
            return Ok((Tensor::zeros((0, hidden), start_rep.dtype(), device)?, pairs));
        }
        let s_idx = Tensor::from_vec(starts, pairs.len(), device)?;
        let e_idx = Tensor::from_vec(ends, pairs.len(), device)?;
        let start_sel = start_rep.index_select(&s_idx, 0)?;
        let end_sel = end_rep.index_select(&e_idx, 0)?;
        let cat = Tensor::cat(&[&start_sel, &end_sel], 1)?.relu()?;
        Ok((self.out_project.forward(&cat)?, pairs))
    }
}

// ---------------------------------------------------------------------------
// Relation scorer
// ---------------------------------------------------------------------------

/// One proposed relation pair (word spans are half-open).
#[derive(Debug, Clone, Copy)]
pub struct RelationPair {
    pub relation: usize,
    pub head: (usize, usize),
    pub tail: (usize, usize),
}

pub struct RelationScorer {
    mlp_hidden: Linear,
    mlp_out: Linear,
    biaffine: Option<(Linear, Linear, Linear, Linear)>,
    hidden: usize,
}

impl RelationScorer {
    pub fn load(vb: VarBuilder, hidden: usize, cfg: &BoundaryHeadConfig) -> Result<Self> {
        let rel_dim = if cfg.directional_relation_states { 2 * hidden } else { hidden };
        let biaffine = if cfg.relation_biaffine_content {
            Some((
                linear(hidden, hidden, vb.pp("head_content_projection"))?,
                linear(hidden, hidden, vb.pp("tail_content_projection"))?,
                linear(rel_dim, hidden, vb.pp("relation_content_gate"))?,
                linear(2 * hidden + rel_dim, 1, vb.pp("content_linear"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            mlp_hidden: linear(4 * hidden + rel_dim + 2, hidden, vb.pp("mlp.0"))?,
            mlp_out: linear(hidden, 1, vb.pp("mlp.3"))?,
            biaffine,
            hidden,
        })
    }

    /// `text`: `[n, H]`, `relation_states`: `[R, rel_dim]`. Returns one logit per pair.
    pub fn forward(&self, text: &Tensor, relation_states: &Tensor, pairs: &[RelationPair]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let device = text.device();
        let n = text.dim(0)?;
        let p = pairs.len();
        let last = n.saturating_sub(1);
        let gather = |pos: Vec<usize>| -> Result<Tensor> {
            text.index_select(&u32_tensor(pos.into_iter().map(|x| x.min(last)), device)?, 0)
        };
        let h_start = gather(pairs.iter().map(|x| x.head.0).collect())?;
        let h_end = gather(pairs.iter().map(|x| x.head.1.saturating_sub(1)).collect())?;
        let t_start = gather(pairs.iter().map(|x| x.tail.0).collect())?;
        let t_end = gather(pairs.iter().map(|x| x.tail.1.saturating_sub(1)).collect())?;
        let rel = relation_states.index_select(&u32_tensor(pairs.iter().map(|x| x.relation), device)?, 0)?;

        let mut positional = Vec::with_capacity(2 * p);
        for pair in pairs {
            let delta = pair.tail.0 as f32 - pair.head.0 as f32;
            // torch.sign: 0 for 0.
            positional.push(if delta == 0.0 { 0.0 } else { delta.signum() });
            positional.push(delta.abs() / n.max(1) as f32);
        }
        let positional = Tensor::from_vec(positional, (p, 2), device)?;
        let feats = Tensor::cat(&[&h_start, &h_end, &t_start, &t_end, &rel, &positional], 1)?;
        let mut score = self
            .mlp_out
            .forward(&self.mlp_hidden.forward(&feats)?.gelu_erf()?)?
            .squeeze(1)?;

        if let Some((head_proj, tail_proj, gate_proj, content_linear)) = &self.biaffine {
            let zeros = Tensor::zeros((1, self.hidden), DType::F32, device)?;
            let prefix = Tensor::cat(&[&zeros, &text.cumsum(0)?], 0)?;
            let pool = |spans: Vec<(usize, usize)>| -> Result<Tensor> {
                let starts = u32_tensor(spans.iter().map(|s| s.0.min(n)), device)?;
                let ends = u32_tensor(spans.iter().map(|s| s.1.min(n)), device)?;
                let widths: Vec<f32> = spans.iter().map(|s| s.1.saturating_sub(s.0).max(1) as f32).collect();
                let widths = Tensor::from_vec(widths, (spans.len(), 1), device)?;
                (prefix.index_select(&ends, 0)? - prefix.index_select(&starts, 0)?)?.broadcast_div(&widths)
            };
            let head_content = head_proj.forward(&pool(pairs.iter().map(|x| x.head).collect())?)?;
            let tail_content = tail_proj.forward(&pool(pairs.iter().map(|x| x.tail).collect())?)?;
            let gate = candle_nn::ops::sigmoid(&gate_proj.forward(&rel)?)?;
            let biaffine = ((&head_content * gate)? * &tail_content)?
                .sum(1)?
                .affine(1.0 / (self.hidden as f64).sqrt(), 0.0)?;
            let linear_term = content_linear
                .forward(&Tensor::cat(&[&head_content, &tail_content, &rel], 1)?)?
                .squeeze(1)?;
            score = ((score + biaffine)? + linear_term)?;
        }
        score.to_vec1::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_keeps_best_occurrence_and_orders_by_score() {
        // keys: 5 (twice), 3, 7 (invalid)
        let keys = [5, 3, 5, 7];
        let scores = [1.0, 2.0, 3.0, 9.0];
        let valid = [true, true, true, false];
        assert_eq!(deduplicate_pool(&keys, &scores, &valid, 10, 4), vec![5, 3]);
        assert_eq!(deduplicate_pool(&keys, &scores, &valid, 1, 4), vec![5]);
    }

    #[test]
    fn dedup_ties_break_by_key() {
        let keys = [9, 2, 4];
        let scores = [1.0, 1.0, 1.0];
        let valid = [true, true, true];
        assert_eq!(deduplicate_pool(&keys, &scores, &valid, 10, 4), vec![2, 4, 9]);
    }
}
