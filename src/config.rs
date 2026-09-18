//! Checkpoint configuration (`config.json` + `encoder_config/config.json`).

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ExtractorConfig {
    pub architecture: String,
    #[serde(default = "default_pooling")]
    pub token_pooling: String,
    pub boundary_head: BoundaryHeadConfig,
}

fn default_pooling() -> String {
    "first".to_string()
}

fn default_record_instance_queries() -> usize {
    32
}

/// Subset of `BoundaryHeadSettings` that affects inference.
#[derive(Debug, Clone, Deserialize)]
pub struct BoundaryHeadConfig {
    pub boundary_dim: usize,
    pub boundary_attention_heads: usize,
    pub boundary_attention_layers: usize,
    pub boundary_attention_window: usize,
    pub boundary_refinement_layers: usize,
    pub boundary_ffn_multiplier: f64,
    pub pair_dim: usize,
    pub content_dim: usize,
    pub content_soft_max_pool: bool,
    pub enable_span_content: bool,
    pub use_inside_evidence: bool,
    pub candidate_pool: String,
    pub candidate_attention_layers: usize,
    pub query_attention_layers: usize,
    pub pool_boundary_top_k: usize,
    pub pool_size: usize,
    pub min_pool_per_query: usize,
    pub enable_abstention: bool,
    pub abstention_threshold: f32,
    pub adaptive_threshold: bool,
    pub pair_temperature: f32,
    pub classification_temperature: f32,
    pub relation_temperature: f32,
    pub enable_relations: bool,
    pub directional_relation_states: bool,
    pub relation_biaffine_content: bool,
    pub relation_heads_per_type: usize,
    pub relation_tails_per_type: usize,
    pub relation_pair_cap: usize,
    pub relation_argument_proposal_threshold: f32,
    pub overlap_policy: String,
    // Per-query pair reranker (explicit span scoring for choice fields).
    pub enable_rotary_endpoints: bool,
    pub rotary_base: f32,
    pub multihead_pair_compat_heads: usize,
    pub query_conditioned_inside_weight: bool,
    pub endpoint_difference_features: bool,
    pub reranker_endpoint_compat: bool,
    // Record (instance formation) head.
    pub enable_records: bool,
    pub record_dim: usize,
    pub record_temperature: f32,
    #[serde(default = "default_record_instance_queries")]
    pub record_instance_queries: usize,
}

/// DeBERTa-v2 encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default = "minus_one")]
    pub max_relative_positions: i64,
    #[serde(default = "minus_one")]
    pub position_buckets: i64,
    pub vocab_size: usize,
    #[serde(default)]
    pub relative_attention: bool,
    #[serde(default)]
    pub share_att_key: bool,
    #[serde(default)]
    pub pos_att_type: Vec<String>,
    #[serde(default = "yes")]
    pub position_biased_input: bool,
    #[serde(default)]
    pub norm_rel_ebd: String,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub conv_kernel_size: usize,
    #[serde(default)]
    pub type_vocab_size: usize,
}

fn minus_one() -> i64 {
    -1
}

fn yes() -> bool {
    true
}

impl EncoderConfig {
    pub fn max_relative_positions(&self) -> i64 {
        if self.max_relative_positions < 1 {
            self.max_position_embeddings as i64
        } else {
            self.max_relative_positions
        }
    }

    /// Size of one side of the relative-embedding table (`att_span`).
    pub fn pos_ebd_size(&self) -> i64 {
        if self.position_buckets > 0 {
            self.position_buckets
        } else {
            self.max_relative_positions()
        }
    }
}

/// Legacy "span" architecture config (`gliner2-privacy-filter-PII-multi`,
/// `GLiNER2-Guardrails-PII-Multi`, `gliguard-LLMGuardrails-300M`). Distinguished
/// from [`ExtractorConfig`] by the absence of a `boundary_head` key.
#[derive(Debug, Clone, Deserialize)]
pub struct SpanConfig {
    pub max_width: usize,
    pub counting_layer: String,
    #[serde(default = "default_pooling")]
    pub token_pooling: String,
}

pub enum Architecture {
    Boundary,
    Span,
}

/// Peeks `config.json` to tell which architecture a checkpoint uses, without
/// fully parsing either schema.
pub fn sniff_architecture(model_dir: &Path) -> Result<Architecture> {
    let raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    sniff_architecture_str(&raw)
}

/// Same as [`sniff_architecture`] but from an already-read `config.json` string
/// (used when a checkpoint is loaded from in-memory bytes, e.g. in wasm).
pub fn sniff_architecture_str(raw: &str) -> Result<Architecture> {
    let value: serde_json::Value = serde_json::from_str(raw).context("parsing config.json")?;
    if value.get("boundary_head").is_some() {
        Ok(Architecture::Boundary)
    } else if value.get("counting_layer").is_some() {
        Ok(Architecture::Span)
    } else {
        bail!("config.json has neither 'boundary_head' nor 'counting_layer'; unrecognized architecture");
    }
}

pub fn load_span_configs(model_dir: &Path) -> Result<(SpanConfig, EncoderConfig)> {
    let read = |p: &Path| -> Result<String> {
        std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))
    };
    load_span_configs_str(&read(&model_dir.join("config.json"))?, &read(&model_dir.join("encoder_config").join("config.json"))?)
}

/// Same as [`load_span_configs`] but from already-read `config.json` /
/// `encoder_config/config.json` strings.
pub fn load_span_configs_str(config_json: &str, encoder_config_json: &str) -> Result<(SpanConfig, EncoderConfig)> {
    let cfg: SpanConfig = serde_json::from_str(config_json).context("parsing config.json")?;
    let enc: EncoderConfig = serde_json::from_str(encoder_config_json).context("parsing encoder_config/config.json")?;
    if cfg.token_pooling != "first" {
        bail!("only token_pooling='first' is supported, got {:?}", cfg.token_pooling);
    }
    // `counting_layer` gates entity/structure extraction (see `SpanModel::extract`);
    // classification never needs it, so an unsupported layer isn't fatal here.
    validate_encoder(&enc)?;
    Ok((cfg, enc))
}

pub fn load_configs(model_dir: &Path) -> Result<(ExtractorConfig, EncoderConfig)> {
    let read = |p: &Path| -> Result<String> {
        std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))
    };
    load_configs_str(&read(&model_dir.join("config.json"))?, &read(&model_dir.join("encoder_config").join("config.json"))?)
}

/// Same as [`load_configs`] but from already-read `config.json` /
/// `encoder_config/config.json` strings.
pub fn load_configs_str(config_json: &str, encoder_config_json: &str) -> Result<(ExtractorConfig, EncoderConfig)> {
    let cfg: ExtractorConfig = serde_json::from_str(config_json).context("parsing config.json")?;
    let enc: EncoderConfig = serde_json::from_str(encoder_config_json).context("parsing encoder_config/config.json")?;
    validate(&cfg, &enc)?;
    Ok((cfg, enc))
}

fn validate(cfg: &ExtractorConfig, enc: &EncoderConfig) -> Result<()> {
    if cfg.architecture != "boundary" {
        bail!("only the 'boundary' architecture is supported, got {:?}", cfg.architecture);
    }
    if cfg.token_pooling != "first" {
        bail!("only token_pooling='first' is supported, got {:?}", cfg.token_pooling);
    }
    let h = &cfg.boundary_head;
    if h.candidate_pool != "shared" {
        bail!("only candidate_pool='shared' is supported, got {:?}", h.candidate_pool);
    }
    if h.candidate_attention_layers != 0 || h.query_attention_layers != 0 {
        bail!("candidate/query attention layers in the shared pool scorer are not supported");
    }
    if h.content_soft_max_pool {
        bail!("content_soft_max_pool is not supported");
    }
    if h.adaptive_threshold {
        bail!("adaptive_threshold decoding is not supported");
    }
    if !(h.enable_rotary_endpoints
        && h.query_conditioned_inside_weight
        && h.endpoint_difference_features
        && h.reranker_endpoint_compat
        && h.enable_span_content)
    {
        bail!("unsupported pair reranker configuration (expected rotary endpoints, query-conditioned inside weight, endpoint difference and compat features, span content)");
    }
    if h.pair_dim % h.multihead_pair_compat_heads != 0 || h.boundary_dim % 2 != 0 || h.pair_dim % 2 != 0 {
        bail!("pair_dim/boundary_dim incompatible with rotary endpoints or compat heads");
    }
    validate_encoder(enc)
}

fn validate_encoder(enc: &EncoderConfig) -> Result<()> {
    if !enc.relative_attention
        || !enc.share_att_key
        || enc.position_biased_input
        || enc.conv_kernel_size != 0
        || enc.type_vocab_size != 0
    {
        bail!("unsupported DeBERTa-v2 variant (expected relative attention, shared keys, no conv, no absolute positions)");
    }
    let mut att = enc.pos_att_type.clone();
    att.sort();
    if att != ["c2p", "p2c"] {
        bail!("expected pos_att_type [p2c, c2p], got {:?}", enc.pos_att_type);
    }
    if !enc.norm_rel_ebd.contains("layer_norm") {
        bail!("expected norm_rel_ebd=layer_norm");
    }
    if enc.hidden_act != "gelu" {
        bail!("expected hidden_act=gelu, got {:?}", enc.hidden_act);
    }
    Ok(())
}
