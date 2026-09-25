//! End-to-end extractor: load a checkpoint directory and run schema extraction.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;

use crate::chunking::ChunkOptions;
use crate::config::{
    load_configs, load_configs_str, load_span_configs, load_span_configs_str, sniff_architecture, Architecture,
    EncoderConfig, ExtractorConfig, SpanConfig,
};
use crate::deberta::DebertaV2;
use crate::modernbert::ModernBert;
use crate::decode::{
    deduplicate_relation_edges, propose_relation_pairs, resolve_overlaps, OverlapPolicy, RelationEdge,
    RelationProposalSettings, RelationRoles, ScoredSpan,
};
use crate::heads::{BoundaryForward, BoundaryHead, Classifier, CountLstmStep0, CountPred, RelationScorer, SpanRep};
use crate::processor::{PreparedInput, Processor, TaskKind, WordSplitter};
use crate::records::{decode_group, record_local_choice_mentions, Cardinality, RecordField, RecordHead, RecordMode};
use crate::schema::{
    ClassActivation, ClassificationSpec, EntityDtype, FieldDtype, FieldSpec, Schema, StructureMode, StructureSpec,
};

/// A loaded checkpoint: either the current "boundary" architecture
/// (`fastino/gliner2.5-*`) or the legacy "span" architecture (the PII/guardrail
/// checkpoints — `gliner2-privacy-filter-PII-multi`, `GLiNER2-Guardrails-PII-Multi`,
/// `gliguard-LLMGuardrails-300M`). The span path only implements what those
/// checkpoints are used for here: classification and flat entity extraction;
/// relations, structures and record extraction bail with a clear error.
pub enum GLiNER2 {
    Boundary(BoundaryModel),
    Span(SpanModel),
}

impl GLiNER2 {
    pub fn load(model_dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = model_dir.as_ref();
        match sniff_architecture(dir)? {
            Architecture::Boundary => Ok(Self::Boundary(BoundaryModel::load(dir, device, dtype)?)),
            Architecture::Span => Ok(Self::Span(SpanModel::load(dir, device, dtype)?)),
        }
    }

    /// Load a checkpoint from in-memory file contents instead of a directory
    /// on disk (used by the wasm bindings, where files come from `fetch` and
    /// there is no filesystem). `weights` is the raw `model.safetensors`
    /// bytes; the rest are the corresponding checkpoint files' contents.
    pub fn load_from_bytes(
        config_json: &str,
        encoder_config_json: &str,
        tokenizer_bytes: &[u8],
        weights: &[u8],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        match crate::config::sniff_architecture_str(config_json)? {
            Architecture::Boundary => Ok(Self::Boundary(BoundaryModel::load_from_bytes(
                config_json,
                encoder_config_json,
                tokenizer_bytes,
                weights,
                device,
                dtype,
            )?)),
            Architecture::Span => Ok(Self::Span(SpanModel::load_from_bytes(
                config_json,
                encoder_config_json,
                tokenizer_bytes,
                weights,
                device,
                dtype,
            )?)),
        }
    }

    pub fn set_word_splitter(&mut self, splitter: WordSplitter) {
        match self {
            Self::Boundary(m) => m.set_word_splitter(splitter),
            Self::Span(m) => m.set_word_splitter(splitter),
        }
    }

    pub fn word_splitter(&self) -> WordSplitter {
        match self {
            Self::Boundary(m) => m.word_splitter(),
            Self::Span(m) => m.word_splitter(),
        }
    }

    pub fn extract_entities(&self, text: &str, labels: &[&str], opts: &ExtractOptions) -> Result<Value> {
        self.extract(text, &Schema::new().entities(labels.iter().copied()), opts)
    }

    pub fn extract_relations(&self, text: &str, relations: &[&str], opts: &ExtractOptions) -> Result<Value> {
        self.extract(text, &Schema::new().relations(relations.iter().copied()), opts)
    }

    pub fn classify_text(&self, text: &str, tasks: Vec<ClassificationSpec>, opts: &ExtractOptions) -> Result<Value> {
        let schema = tasks.into_iter().fold(Schema::new(), Schema::classification);
        self.extract(text, &schema, opts)
    }

    pub fn classification_probabilities(
        &self,
        text: &str,
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<(String, Vec<(String, f32)>)>> {
        match self {
            Self::Boundary(m) => m.classification_probabilities(text, tasks),
            Self::Span(m) => m.classification_probabilities(text, tasks),
        }
    }

    /// [`Self::classification_probabilities`] for several texts, encoded in
    /// one padded batch pass.
    pub fn classification_probabilities_batch(
        &self,
        texts: &[&str],
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<Vec<(String, Vec<(String, f32)>)>>> {
        match self {
            Self::Boundary(m) => m.classification_probabilities_batch(texts, tasks),
            Self::Span(m) => m.classification_probabilities_batch(texts, tasks),
        }
    }

    pub fn extract_json(&self, text: &str, structures: &[(&str, &[&str])], opts: &ExtractOptions) -> Result<Value> {
        let schema = structures
            .iter()
            .fold(Schema::new(), |schema, (name, fields)| schema.structure(StructureSpec::parse(*name, fields.iter())));
        self.extract(text, &schema, opts)
    }

    pub fn extract(&self, text: &str, schema: &Schema, opts: &ExtractOptions) -> Result<Value> {
        match self {
            Self::Boundary(m) => m.extract(text, schema, opts),
            Self::Span(m) => m.extract(text, schema, opts),
        }
    }

    pub fn extract_entities_batch(&self, texts: &[&str], labels: &[&str], opts: &ExtractOptions) -> Result<Vec<Value>> {
        self.extract_batch(texts, &Schema::new().entities(labels.iter().copied()), opts)
    }

    pub fn extract_relations_batch(&self, texts: &[&str], relations: &[&str], opts: &ExtractOptions) -> Result<Vec<Value>> {
        self.extract_batch(texts, &Schema::new().relations(relations.iter().copied()), opts)
    }

    pub fn classify_text_batch(&self, texts: &[&str], tasks: Vec<ClassificationSpec>, opts: &ExtractOptions) -> Result<Vec<Value>> {
        let schema = tasks.into_iter().fold(Schema::new(), Schema::classification);
        self.extract_batch(texts, &schema, opts)
    }

    pub fn extract_json_batch(
        &self,
        texts: &[&str],
        structures: &[(&str, &[&str])],
        opts: &ExtractOptions,
    ) -> Result<Vec<Value>> {
        let schema = structures
            .iter()
            .fold(Schema::new(), |schema, (name, fields)| schema.structure(StructureSpec::parse(*name, fields.iter())));
        self.extract_batch(texts, &schema, opts)
    }

    /// [`Self::extract`] for several texts against the same `schema`,
    /// encoded in one padded batch pass instead of one encoder call per
    /// text. Each text's result is exactly what [`Self::extract`] would give
    /// it alone; only `texts` may vary across a call, not `schema`.
    pub fn extract_batch(&self, texts: &[&str], schema: &Schema, opts: &ExtractOptions) -> Result<Vec<Value>> {
        match self {
            Self::Boundary(m) => m.extract_batch(texts, schema, opts),
            Self::Span(m) => m.extract_batch(texts, schema, opts),
        }
    }

    /// [`Self::extract_entities`] for documents longer than the encoder's
    /// context: `text` is split into overlapping word chunks (`chunk_opts`),
    /// each chunk is extracted independently, and spans are remapped to
    /// character offsets in the original `text` and merged across overlaps.
    pub fn extract_entities_long(
        &self,
        text: &str,
        labels: &[&str],
        opts: &ExtractOptions,
        chunk_opts: ChunkOptions,
    ) -> Result<Value> {
        self.extract_long(text, &Schema::new().entities(labels.iter().copied()), opts, chunk_opts)
    }

    /// Long-document counterpart of [`Self::extract_relations`]; see
    /// [`Self::extract_entities_long`].
    pub fn extract_relations_long(
        &self,
        text: &str,
        relations: &[&str],
        opts: &ExtractOptions,
        chunk_opts: ChunkOptions,
    ) -> Result<Value> {
        self.extract_long(text, &Schema::new().relations(relations.iter().copied()), opts, chunk_opts)
    }

    /// Long-document counterpart of [`Self::classify_text`]; see
    /// [`Self::extract_entities_long`]. Per-chunk label scores are decided
    /// independently and merged (highest-confidence wins on disagreement),
    /// not aggregated before decoding.
    pub fn classify_text_long(
        &self,
        text: &str,
        tasks: Vec<ClassificationSpec>,
        opts: &ExtractOptions,
        chunk_opts: ChunkOptions,
    ) -> Result<Value> {
        let schema = tasks.into_iter().fold(Schema::new(), Schema::classification);
        self.extract_long(text, &schema, opts, chunk_opts)
    }

    /// Long-document counterpart of [`Self::extract_json`]; see
    /// [`Self::extract_entities_long`].
    pub fn extract_json_long(
        &self,
        text: &str,
        structures: &[(&str, &[&str])],
        opts: &ExtractOptions,
        chunk_opts: ChunkOptions,
    ) -> Result<Value> {
        let schema = structures
            .iter()
            .fold(Schema::new(), |schema, (name, fields)| schema.structure(StructureSpec::parse(*name, fields.iter())));
        self.extract_long(text, &schema, opts, chunk_opts)
    }

    /// Long-document counterpart of [`Self::extract`]; see
    /// [`Self::extract_entities_long`]. `opts.overlap_policy` governs how
    /// duplicate spans found in overlapping chunks are resolved (default
    /// `flat`, matching the reference implementation's chunk-merge default).
    pub fn extract_long(&self, text: &str, schema: &Schema, opts: &ExtractOptions, chunk_opts: ChunkOptions) -> Result<Value> {
        let chunks = crate::chunking::split_text_into_chunks(text, chunk_opts, self.word_splitter())?;
        let scalar_entity_labels: HashSet<String> =
            schema.entities.iter().filter(|e| e.dtype == EntityDtype::Str).map(|e| e.name.clone()).collect();
        let policy = opts.overlap_policy.unwrap_or(OverlapPolicy::Disallow);
        let merge_opts = ExtractOptions { include_confidence: true, include_spans: true, ..opts.clone() };
        crate::chunking::extract_chunked(text, &chunks, &scalar_entity_labels, policy, opts.include_confidence, opts.include_spans, |chunk_text| {
            self.extract(chunk_text, schema, &merge_opts)
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExtractOptions {
    pub threshold: f32,
    pub include_confidence: bool,
    pub include_spans: bool,
    /// Defaults to the checkpoint's `overlap_policy` (flat).
    pub overlap_policy: Option<OverlapPolicy>,
    /// Maximum number of text words; later words are dropped.
    pub max_words: Option<usize>,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            include_confidence: false,
            include_spans: false,
            overlap_policy: None,
            max_words: None,
        }
    }
}

/// Either encoder architecture a boundary-architecture checkpoint can use.
enum Encoder {
    Deberta(DebertaV2),
    ModernBert(ModernBert),
}

impl Encoder {
    fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        match cfg {
            EncoderConfig::Deberta(c) => Ok(Encoder::Deberta(DebertaV2::load(vb, c).context("loading DeBERTa-v2 encoder")?)),
            EncoderConfig::ModernBert(c) => Ok(Encoder::ModernBert(ModernBert::load(vb, c).context("loading ModernBERT encoder")?)),
        }
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Encoder::Deberta(e) => e.forward(input_ids, attention_mask),
            Encoder::ModernBert(e) => e.forward(input_ids, attention_mask),
        }
    }
}

pub struct BoundaryModel {
    encoder: Encoder,
    head: BoundaryHead,
    classifier: Classifier,
    relation_scorer: Option<RelationScorer>,
    record_head: Option<RecordHead>,
    processor: Processor,
    config: ExtractorConfig,
    encoder_config: EncoderConfig,
    device: Device,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(xs: &[f32]) -> usize {
    // torch.argmax returns the first maximal index.
    let mut best = 0;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// `(surface, confidence, char start, char end)`.
type Mention = (String, f32, usize, usize);

/// Word-level query of the boundary head.
struct QuerySlot {
    group: usize,
    field: usize,
}

/// Per-call state shared by the task decoders.
struct Pass<'a> {
    text: &'a str,
    input: &'a PreparedInput,
    slots: Vec<QuerySlot>,
    forward: Option<BoundaryForward>,
    text_states: Option<Tensor>,
    query_states: Option<Tensor>,
    policy: OverlapPolicy,
    opts: &'a ExtractOptions,
}

impl Pass<'_> {
    fn query(&self, group: usize, field: usize) -> Option<usize> {
        self.slots.iter().position(|s| s.group == group && s.field == field)
    }

    fn offset(&self) -> usize {
        self.input.prefix_tokens.len()
    }

    /// Character offsets of a model token span, if it lies inside the text.
    fn char_span(&self, (start, end): (usize, usize)) -> Option<(usize, usize)> {
        let words = &self.input.words;
        let (s, e) = (start.checked_sub(self.offset())?, end.checked_sub(self.offset())?);
        (s < e && e <= words.len()).then(|| (words[s].char_start, words[e - 1].char_end))
    }

    /// Stripped surface text and character offsets of a model token span.
    fn mention(&self, (start, end): (usize, usize)) -> Option<(String, usize, usize)> {
        let words = &self.input.words;
        let (s, e) = (start.checked_sub(self.offset())?, end.checked_sub(self.offset())?);
        if !(s < e && e <= words.len()) {
            return None;
        }
        let surface = self.text[words[s].byte_start..words[e - 1].byte_end].trim();
        (!surface.is_empty()).then(|| (surface.to_string(), words[s].char_start, words[e - 1].char_end))
    }
}

impl BoundaryModel {
    /// Load `config.json`, `encoder_config/config.json`, `tokenizer.json` and
    /// `model.safetensors` from `model_dir`. `dtype` applies to the encoder;
    /// the heads always run in float32.
    pub fn load(model_dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = model_dir.as_ref();
        let (config, encoder_config) = load_configs(dir)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let weights = dir.join("model.safetensors");
        // SAFETY: the checkpoint file must not be modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, device) }
            .with_context(|| format!("mapping {}", weights.display()))?;
        let head_vb = vb.clone().set_dtype(DType::F32);

        Self::from_parts(config, encoder_config, tokenizer, vb, head_vb, device)
    }

    /// Same as [`Self::load`] but from in-memory file contents (see
    /// [`GLiNER2::load_from_bytes`]).
    pub fn load_from_bytes(
        config_json: &str,
        encoder_config_json: &str,
        tokenizer_bytes: &[u8],
        weights: &[u8],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (config, encoder_config) = load_configs_str(config_json, encoder_config_json)?;
        let tokenizer =
            Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let tensors = crate::safetensors32::load_buffer(weights, device).context("loading model.safetensors")?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let head_vb = vb.clone().set_dtype(DType::F32);
        Self::from_parts(config, encoder_config, tokenizer, vb, head_vb, device)
    }

    fn from_parts(
        config: ExtractorConfig,
        encoder_config: EncoderConfig,
        tokenizer: Tokenizer,
        vb: VarBuilder,
        head_vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let hidden = encoder_config.hidden_size();
        let cfg = &config.boundary_head;

        let encoder = Encoder::load(vb.pp("encoder"), &encoder_config).context("loading encoder")?;
        let head = BoundaryHead::load(head_vb.pp("boundary_head"), hidden, cfg).context("loading boundary head")?;
        let classifier = Classifier::load(head_vb.pp("classifier"), hidden).context("loading classifier")?;
        let relation_scorer = if cfg.enable_relations {
            Some(RelationScorer::load(head_vb.pp("relation_scorer"), hidden, cfg).context("loading relation scorer")?)
        } else {
            None
        };
        let record_head = if cfg.enable_records {
            Some(
                RecordHead::load(head_vb.pp("record_decoder"), hidden, cfg.record_dim, cfg.record_instance_queries)
                    .context("loading record head")?,
            )
        } else {
            None
        };
        Ok(Self {
            encoder,
            head,
            classifier,
            relation_scorer,
            record_head,
            processor: Processor::new(tokenizer),
            config,
            encoder_config,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &ExtractorConfig {
        &self.config
    }

    pub fn encoder_config(&self) -> &EncoderConfig {
        &self.encoder_config
    }

    /// Use `WordSplitter::Char` for languages without whitespace-delimited words.
    pub fn set_word_splitter(&mut self, splitter: WordSplitter) {
        self.processor.word_splitter = splitter;
    }

    pub fn word_splitter(&self) -> WordSplitter {
        self.processor.word_splitter
    }

    pub fn processor(&self) -> &Processor {
        &self.processor
    }

    pub fn extract_entities(&self, text: &str, labels: &[&str], opts: &ExtractOptions) -> Result<Value> {
        self.extract(text, &Schema::new().entities(labels.iter().copied()), opts)
    }

    pub fn extract_relations(&self, text: &str, relations: &[&str], opts: &ExtractOptions) -> Result<Value> {
        self.extract(text, &Schema::new().relations(relations.iter().copied()), opts)
    }

    pub fn classify_text(&self, text: &str, tasks: Vec<ClassificationSpec>, opts: &ExtractOptions) -> Result<Value> {
        let schema = tasks.into_iter().fold(Schema::new(), Schema::classification);
        self.extract(text, &schema, opts)
    }

    /// Per-label probabilities for every classification task (softmax for
    /// single-label, sigmoid for multi-label unless `activation` overrides it).
    /// Tasks are returned in order as `(task, [(label, probability)])`.
    pub fn classification_probabilities(
        &self,
        text: &str,
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<(String, Vec<(String, f32)>)>> {
        let schema = tasks.iter().cloned().fold(Schema::new(), Schema::classification);
        let input = self.processor.prepare(text, &schema, None)?;
        let states = self.encode(&input)?;
        self.classification_probabilities_from_states(&input, &states, tasks)
    }

    /// [`Self::classification_probabilities`] for several texts against the
    /// same `tasks`, encoded in one padded batch pass.
    pub fn classification_probabilities_batch(
        &self,
        texts: &[&str],
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<Vec<(String, Vec<(String, f32)>)>>> {
        let schema = tasks.iter().cloned().fold(Schema::new(), Schema::classification);
        let inputs: Vec<PreparedInput> =
            texts.iter().map(|text| self.processor.prepare(text, &schema, None)).collect::<Result<_>>()?;
        let states = self.encode_batch(&inputs)?;
        inputs
            .iter()
            .zip(states.iter())
            .map(|(input, states)| self.classification_probabilities_from_states(input, states, tasks))
            .collect()
    }

    fn classification_probabilities_from_states(
        &self,
        input: &PreparedInput,
        states: &Tensor,
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<(String, Vec<(String, f32)>)>> {
        let temperature = self.config.boundary_head.classification_temperature;
        let mut out = Vec::new();
        for group in input.groups.iter().filter(|g| g.kind == TaskKind::Classifications) {
            let spec = &tasks[group.spec_index];
            let label_states = self.gather_rows(states, &group.marker_positions)?;
            let logits: Vec<f32> =
                self.classifier.forward(&label_states)?.into_iter().map(|x| x / temperature).collect();
            let probs = classification_probs(spec, &logits);
            out.push((spec.task.clone(), spec.labels.iter().cloned().zip(probs).collect()));
        }
        Ok(out)
    }

    /// Structured extraction (`extract_json`). Fields use gliner2's compact
    /// syntax, e.g. `("product", &["name::str", "price::str", "features::list"])`.
    pub fn extract_json(&self, text: &str, structures: &[(&str, &[&str])], opts: &ExtractOptions) -> Result<Value> {
        let schema = structures
            .iter()
            .fold(Schema::new(), |schema, (name, fields)| schema.structure(StructureSpec::parse(*name, fields.iter())));
        self.extract(text, &schema, opts)
    }

    /// Encode the prompt + text and return `[T, H]` float32 token states.
    fn encode(&self, input: &PreparedInput) -> Result<Tensor> {
        let t = input.input_ids.len();
        let ids = Tensor::from_vec(input.input_ids.clone(), (1, t), &self.device)?;
        let mask = Tensor::ones((1, t), DType::U32, &self.device)?;
        Ok(self.encoder.forward(&ids, &mask)?.squeeze(0)?.to_dtype(DType::F32)?)
    }

    /// Encode a batch of prepared inputs in one padded encoder pass (real
    /// tokens attend only to real tokens, so per-example results are exactly
    /// what [`Self::encode`] would give for that text alone); returns one
    /// `[T_i, H]` float32 state tensor per input, in order.
    fn encode_batch(&self, inputs: &[PreparedInput]) -> Result<Vec<Tensor>> {
        let b = inputs.len();
        let max_t = inputs.iter().map(|i| i.input_ids.len()).max().unwrap_or(0);
        let mut ids = vec![0u32; b * max_t];
        let mut mask = vec![0u32; b * max_t];
        for (i, input) in inputs.iter().enumerate() {
            for (j, &id) in input.input_ids.iter().enumerate() {
                ids[i * max_t + j] = id;
                mask[i * max_t + j] = 1;
            }
        }
        let ids = Tensor::from_vec(ids, (b, max_t), &self.device)?;
        let mask = Tensor::from_vec(mask, (b, max_t), &self.device)?;
        let states = self.encoder.forward(&ids, &mask)?.to_dtype(DType::F32)?;
        inputs
            .iter()
            .enumerate()
            .map(|(i, input)| Ok(states.narrow(0, i, 1)?.squeeze(0)?.narrow(0, 0, input.input_ids.len())?.contiguous()?))
            .collect()
    }

    fn gather_rows(&self, states: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let idx: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let idx = Tensor::from_vec(idx, positions.len(), &self.device)?;
        Ok(states.index_select(&idx, 0)?)
    }

    /// Run a full schema extraction for one text. The returned JSON follows
    /// `gliner2`'s formatted results; `start`/`end` are character (code point)
    /// offsets into `text`.
    pub fn extract(&self, text: &str, schema: &Schema, opts: &ExtractOptions) -> Result<Value> {
        let input = self.processor.prepare(text, schema, opts.max_words)?;
        let states = self.encode(&input)?;
        self.extract_from_states(&input, &states, schema, opts)
    }

    /// [`Self::extract`] for several texts against the same `schema`, encoded
    /// in one padded batch pass. Each text's result is identical to what
    /// [`Self::extract`] would give it alone.
    pub fn extract_batch(&self, texts: &[&str], schema: &Schema, opts: &ExtractOptions) -> Result<Vec<Value>> {
        let inputs: Vec<PreparedInput> =
            texts.iter().map(|text| self.processor.prepare(text, schema, opts.max_words)).collect::<Result<_>>()?;
        let states = self.encode_batch(&inputs)?;
        inputs.iter().zip(states.iter()).map(|(input, states)| self.extract_from_states(input, states, schema, opts)).collect()
    }

    fn extract_from_states(&self, input: &PreparedInput, states: &Tensor, schema: &Schema, opts: &ExtractOptions) -> Result<Value> {
        let cfg = &self.config.boundary_head;
        let policy = match opts.overlap_policy {
            Some(p) => p,
            None => cfg.overlap_policy.parse()?,
        };

        let mut slots: Vec<QuerySlot> = Vec::new();
        let mut query_positions: Vec<usize> = Vec::new();
        for (gi, group) in input.groups.iter().enumerate() {
            if group.kind == TaskKind::Classifications {
                continue;
            }
            for (fi, &pos) in group.marker_positions.iter().enumerate() {
                slots.push(QuerySlot { group: gi, field: fi });
                query_positions.push(pos);
            }
        }

        let mut pass = Pass {
            text: &input.text,
            input,
            slots,
            forward: None,
            text_states: None,
            query_states: None,
            policy,
            opts,
        };
        if !pass.slots.is_empty() && !input.word_positions.is_empty() {
            let ts = self.gather_rows(states, &input.word_positions)?;
            let qs = self.gather_rows(states, &query_positions)?;
            pass.forward = Some(self.head.forward(&ts, &qs)?);
            pass.text_states = Some(ts);
            pass.query_states = Some(qs);
        }

        let mut out = Map::new();

        // Structures: record-mode groups first, then legacy aggregates.
        let structure_groups: Vec<usize> =
            (0..input.groups.len()).filter(|&gi| input.groups[gi].kind == TaskKind::Structures).collect();
        let uses_records = |gi: usize| {
            self.record_head.is_some()
                && !matches!(schema.structures[input.groups[gi].spec_index].mode, StructureMode::Legacy)
        };
        for &gi in structure_groups.iter().filter(|&&gi| uses_records(gi)) {
            let spec = &schema.structures[input.groups[gi].spec_index];
            let instances = self.decode_records(&pass, gi, spec)?;
            if !instances.is_empty() {
                out.insert(spec.name.clone(), Value::Array(instances));
            }
        }
        for &gi in structure_groups.iter().filter(|&&gi| !uses_records(gi)) {
            let spec = &schema.structures[input.groups[gi].spec_index];
            if let Some(instance) = self.decode_legacy_structure(&pass, gi, spec)? {
                out.insert(spec.name.clone(), Value::Array(vec![instance]));
            }
        }

        // Entities.
        if let Some(group_index) = input.groups.iter().position(|g| g.kind == TaskKind::Entities) {
            out.insert("entities".to_string(), Value::Object(self.decode_entities(&pass, schema, group_index)?));
        }

        // Classifications.
        for group in input.groups.iter().filter(|g| g.kind == TaskKind::Classifications) {
            let spec = &schema.classifications[group.spec_index];
            if group.marker_positions.is_empty() {
                continue;
            }
            let label_states = self.gather_rows(states, &group.marker_positions)?;
            let logits: Vec<f32> = self
                .classifier
                .forward(&label_states)?
                .into_iter()
                .map(|x| x / cfg.classification_temperature)
                .collect();
            out.insert(spec.task.clone(), classification_value(spec, &logits, opts.include_confidence));
        }

        // Relations.
        if !schema.relations.is_empty() {
            let mut relation_map = Map::new();
            if self.relation_scorer.is_some() && pass.forward.is_some() {
                for (name, values) in self.decode_relations(&pass, schema)? {
                    relation_map.insert(name, Value::Array(values));
                }
            }
            let mut ordered = Map::new();
            for rel in &schema.relations {
                let value = relation_map.remove(&rel.name).unwrap_or_else(|| Value::Array(Vec::new()));
                ordered.entry(rel.name.clone()).or_insert(value);
            }
            out.insert("relation_extraction".to_string(), Value::Object(ordered));
        }

        Ok(Value::Object(out))
    }

    fn pair_probability(&self, logit: f32) -> f32 {
        sigmoid(logit / self.config.boundary_head.pair_temperature)
    }

    fn decode_entities(&self, pass: &Pass, schema: &Schema, group_index: usize) -> Result<Map<String, Value>> {
        let cfg = &self.config.boundary_head;
        let opts = pass.opts;
        // Attribute label fields (`Schema::entity_attributes`) get a query
        // slot but are never emitted as their own entity; they're used below
        // to force-score attribute values at the entities that ARE emitted.
        let mut decoded: Vec<(&str, EntityDtype, Vec<Mention>)> = Vec::new();
        let mut token_spans: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for (field_index, spec) in schema.entities.iter().enumerate() {
            let mut items: Vec<Mention> = Vec::new();
            if let (Some(q), Some(fwd)) = (pass.query(group_index, field_index), &pass.forward) {
                let sc = &fwd.scores;
                let abstained =
                    sc.null_logits.as_ref().is_some_and(|null| sigmoid(null[q]) > cfg.abstention_threshold);
                if !abstained {
                    let threshold = spec.threshold.unwrap_or(opts.threshold);
                    let scored = self.thresholded(sc.spans.as_slice(), &sc.pair_logits[q], threshold);
                    for span in resolve_overlaps(&scored, pass.policy) {
                        if let Some((surface, start, end)) = pass.mention((span.start, span.end)) {
                            if spec.validators.iter().all(|v| v.validate(&surface)) {
                                token_spans.insert((start, end), (span.start, span.end));
                                items.push((surface, span.score, start, end));
                            }
                        }
                    }
                }
            }
            if !spec.is_attribute {
                decoded.push((&spec.name, spec.dtype, items));
            }
        }

        let attributes = self.attach_entity_attributes(pass, schema, group_index, &decoded, &token_spans)?;

        let mut result = Map::new();
        for (name, dtype, items) in decoded {
            let format = |m: &Mention| match attributes.get(&(m.2, m.3)) {
                Some(extra) if !extra.is_empty() => format_attributed_entity(m, opts, extra),
                _ => format_mention(m, opts),
            };
            let value = match dtype {
                EntityDtype::List => dedup_list(items.iter().map(format).collect()),
                EntityDtype::Str => items.first().map_or(Value::Null, format),
            };
            result.entry(name.to_string()).or_insert(value);
        }
        Ok(result)
    }

    /// `_attach_entity_attributes`: force-score configured attribute labels
    /// at every retained entity span via `score_explicit_spans`, bypassing
    /// the normal top-k boundary-candidate pipeline.
    fn attach_entity_attributes(
        &self,
        pass: &Pass,
        schema: &Schema,
        group_index: usize,
        decoded: &[(&str, EntityDtype, Vec<Mention>)],
        token_spans: &HashMap<(usize, usize), (usize, usize)>,
    ) -> Result<HashMap<(usize, usize), Map<String, Value>>> {
        let mut out: HashMap<(usize, usize), Map<String, Value>> = HashMap::new();
        if schema.entity_attribute_groups.is_empty() {
            return Ok(out);
        }
        let (Some(fwd), Some(ts), Some(qs)) = (&pass.forward, &pass.text_states, &pass.query_states) else {
            return Ok(out);
        };

        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (_, _, items) in decoded {
            for &(_, _, start, end) in items {
                if !spans.contains(&(start, end)) {
                    spans.push((start, end));
                }
            }
        }
        if spans.is_empty() {
            return Ok(out);
        }
        let token_span_list: Vec<(usize, usize)> = spans.iter().map(|s| token_spans[s]).collect();

        for (group_name, group) in &schema.entity_attribute_groups {
            let mut label_query: Vec<(&str, usize)> = Vec::new();
            for label in &group.labels {
                let prompt_label = if group.qualify_labels { format!("{group_name}: {label}") } else { label.clone() };
                let Some(field_index) = schema.entities.iter().position(|e| e.name == prompt_label) else { continue };
                let Some(q) = pass.query(group_index, field_index) else { continue };
                label_query.push((label, q));
            }
            if label_query.is_empty() {
                continue;
            }
            let mut logits: Vec<Vec<f32>> = Vec::with_capacity(label_query.len());
            for &(_, q) in &label_query {
                logits.push(
                    self.head
                        .score_explicit_spans(fwd, ts, qs, q, &token_span_list)?
                        .into_iter()
                        .map(|x| x / self.config.boundary_head.pair_temperature)
                        .collect(),
                );
            }

            for &(name, _, ref items) in decoded {
                if group.applies_to.as_ref().is_some_and(|names| !names.iter().any(|n| n == name)) {
                    continue;
                }
                for &(_, _, start, end) in items {
                    let Some(column) = spans.iter().position(|&s| s == (start, end)) else { continue };
                    let values: Vec<f32> = logits.iter().map(|row| row[column]).collect();
                    let value = if group.multi_label {
                        let chosen: Vec<Value> = values
                            .iter()
                            .zip(&label_query)
                            .map(|(&x, &(label, _))| (label, sigmoid(x)))
                            .filter(|&(_, p)| p >= group.threshold)
                            .map(|(label, p)| json!({"label": label, "confidence": p}))
                            .collect();
                        Value::Array(chosen)
                    } else {
                        let probs = softmax(&values);
                        let best = argmax(&probs);
                        json!({"label": label_query[best].0, "confidence": probs[best]})
                    };
                    out.entry((start, end)).or_default().insert(group_name.clone(), value);
                }
            }
        }
        Ok(out)
    }

    fn thresholded(&self, spans: &[(usize, usize)], logits: &[f32], threshold: f32) -> Vec<ScoredSpan> {
        spans
            .iter()
            .zip(logits)
            .map(|(&(start, end), &logit)| ScoredSpan { score: self.pair_probability(logit), start, end })
            .filter(|s| s.score >= threshold)
            .collect()
    }

    /// Record-head instances for one structure (`_decode_records`).
    fn decode_records(&self, pass: &Pass, group_index: usize, spec: &StructureSpec) -> Result<Vec<Value>> {
        let (Some(record_head), Some(fwd), Some(query_states)) = (&self.record_head, &pass.forward, &pass.query_states)
        else {
            return Ok(Vec::new());
        };
        let Some(candidate_states) = self.head.candidate_states(fwd)? else { return Ok(Vec::new()) };
        let cfg = &self.config.boundary_head;
        let threshold = pass.opts.threshold;
        let scores = &fwd.scores;

        let record_mode = match spec.mode {
            StructureMode::Latent => RecordMode::Latent,
            StructureMode::Anchorless => RecordMode::Anchorless,
            StructureMode::Auto | StructureMode::Legacy => RecordMode::Natural,
        };
        let mut fields = Vec::new();
        for (fi, field) in spec.fields.iter().enumerate() {
            let Some(query) = pass.query(group_index, fi) else { return Ok(Vec::new()) };
            let is_anchor = record_mode == RecordMode::Natural
                && match &spec.anchor {
                    Some(name) => &field.name == name,
                    None => fi == 0,
                };
            // `_default_cardinality`: an unset cardinality defaults by dtype,
            // except the anchor field, which is always required.
            let default_cardinality = if is_anchor {
                Cardinality::RequiredOne
            } else {
                match field.dtype {
                    FieldDtype::Str => Cardinality::OptionalOne,
                    FieldDtype::List => Cardinality::ZeroOrMore,
                }
            };
            fields.push(RecordField {
                query,
                cardinality: field.cardinality.unwrap_or(default_cardinality),
                is_anchor,
                exclusive: field.exclusive.unwrap_or(false),
            });
        }
        if fields.is_empty() || (record_mode == RecordMode::Natural && !fields.iter().any(|f| f.is_anchor)) {
            return Ok(Vec::new());
        }
        let field_queries = self.gather_rows(query_states, &fields.iter().map(|f| f.query).collect::<Vec<_>>())?;

        let (object_logits, spans, instance_states) = match record_mode {
            RecordMode::Natural => {
                let anchor_query = fields.iter().find(|f| f.is_anchor).unwrap().query;
                (scores.pair_logits[anchor_query].clone(), Some(scores.spans.clone()), candidate_states.clone())
            }
            RecordMode::Latent => {
                (record_head.latent_seed_scores(&candidate_states)?, Some(scores.spans.clone()), candidate_states.clone())
            }
            RecordMode::Anchorless => {
                let instances = record_head.anchorless_instances(&candidate_states)?;
                let obj = record_head.object_scores(&instances)?;
                (obj, None, instances)
            }
        };
        let assign = record_head.assign_logits(&instance_states, &candidate_states, &field_queries)?;
        let records =
            decode_group(record_mode, &fields, &object_logits, spans.as_deref(), &assign, threshold, cfg.record_temperature);
        let anchor_chars: Vec<Option<(usize, usize)>> =
            records.iter().map(|r| r.anchor.and_then(|a| pass.char_span(scores.spans[a]))).collect();

        let mut instances = Vec::new();
        for (record_index, record) in records.iter().enumerate() {
            let mut instance = Map::new();
            for (fi, (field, rfield)) in spec.fields.iter().zip(&fields).enumerate() {
                let selected = record.fields.get(&fi).map(Vec::as_slice).unwrap_or(&[]);
                let value =
                    self.format_record_field(pass, field, rfield, selected, record_index, &anchor_chars, threshold)?;
                instance.insert(field.name.clone(), value);
            }
            if instance.values().any(|v| !is_empty_value(v)) {
                instances.push(format_struct(instance));
            }
        }
        Ok(instances)
    }

    #[allow(clippy::too_many_arguments)]
    fn format_record_field(
        &self,
        pass: &Pass,
        field: &FieldSpec,
        rfield: &RecordField,
        selected: &[(usize, f32)],
        record_index: usize,
        anchor_chars: &[Option<(usize, usize)>],
        threshold: f32,
    ) -> Result<Value> {
        let fwd = pass.forward.as_ref().expect("record decoding requires a forward pass");
        let opts = pass.opts;
        let is_scalar = rfield.cardinality.is_scalar() || field.dtype == FieldDtype::Str;

        let mut formatted: Vec<Mention> = Vec::new();
        for &(cand, assignment_probability) in selected {
            let Some((surface, start, end)) = pass.mention(fwd.scores.spans[cand]) else { continue };
            let candidate_probability = self.pair_probability(fwd.scores.pair_logits[rfield.query][cand]);
            let probability = candidate_probability.min(assignment_probability);
            if (rfield.cardinality.allows_absent() && candidate_probability < threshold)
                || field.threshold.is_some_and(|t| probability < t)
                || !field.validators.iter().all(|v| v.validate(&surface))
            {
                continue;
            }
            formatted.push((surface, probability, start, end));
        }
        let formatted = finalize_mentions(formatted, is_scalar, pass.policy);

        if let Some(choices) = &field.choices {
            if let Some(Some((anchor_start, anchor_end))) = anchor_chars.get(record_index) {
                let (has_literal, local) = record_local_choice_mentions(pass.text, choices, anchor_chars);
                let mut preferred = local.get(&record_index).cloned().unwrap_or_default();
                if is_scalar && !preferred.is_empty() {
                    let distance = |m: &(String, usize, usize)| {
                        (*anchor_start as i64 - m.2 as i64).max(m.1 as i64 - *anchor_end as i64).max(0)
                    };
                    let mut best = 0;
                    for (i, m) in preferred.iter().enumerate() {
                        if distance(m) < distance(&preferred[best]) {
                            best = i;
                        }
                    }
                    preferred = vec![preferred[best].clone()];
                }
                if !preferred.is_empty() {
                    return self.decode_choice_field(pass, rfield.query, choices, is_scalar, field.threshold, threshold, Some(&preferred), true);
                }
                if has_literal {
                    return Ok(if is_scalar { Value::Null } else { Value::Array(Vec::new()) });
                }
            }
            let matched: Vec<Mention> = formatted
                .iter()
                .filter_map(|(surface, p, s, e)| {
                    let key = surface.to_lowercase();
                    choices.iter().find(|c| c.to_lowercase() == key).map(|c| (c.clone(), *p, *s, *e))
                })
                .collect();
            if !matched.is_empty() {
                return Ok(format_structure_field(&matched, is_scalar, opts));
            }
            return self.decode_choice_field(pass, rfield.query, choices, is_scalar, field.threshold, threshold, None, true);
        }
        Ok(format_structure_field(&formatted, is_scalar, opts))
    }

    /// Aggregate structure without the record head (`_decode_legacy_structures`).
    fn decode_legacy_structure(&self, pass: &Pass, group_index: usize, spec: &StructureSpec) -> Result<Option<Value>> {
        let Some(fwd) = &pass.forward else { return Ok(None) };
        let mut instance = Map::new();
        for (fi, field) in spec.fields.iter().enumerate() {
            let Some(q) = pass.query(group_index, fi) else { continue };
            let is_scalar = field.dtype == FieldDtype::Str;
            let value = if let Some(choices) = &field.choices {
                self.decode_choice_field(pass, q, choices, is_scalar, field.threshold, pass.opts.threshold, None, false)?
            } else {
                let threshold = field.threshold.unwrap_or(pass.opts.threshold);
                let scored = self.thresholded(&fwd.scores.spans, &fwd.scores.pair_logits[q], threshold);
                let mentions: Vec<Mention> = resolve_overlaps(&scored, pass.policy)
                    .into_iter()
                    .filter_map(|s| pass.mention((s.start, s.end)).map(|(t, a, b)| (t, s.score, a, b)))
                    .filter(|(surface, ..)| field.validators.iter().all(|v| v.validate(surface)))
                    .collect();
                format_structure_field(&mentions, is_scalar, pass.opts)
            };
            instance.insert(field.name.clone(), value);
        }
        Ok(instance.values().any(|v| !is_empty_value(v)).then(|| format_struct(instance)))
    }

    /// Score enum values at their schema-prefix positions (`_decode_choice_field`).
    #[allow(clippy::too_many_arguments)]
    fn decode_choice_field(
        &self,
        pass: &Pass,
        query: usize,
        choices: &[String],
        is_scalar: bool,
        configured_threshold: Option<f32>,
        default_threshold: f32,
        preferred: Option<&[(String, usize, usize)]>,
        include_spans: bool,
    ) -> Result<Value> {
        let empty = if is_scalar { Value::Null } else { Value::Array(Vec::new()) };
        let (Some(fwd), Some(ts), Some(qs)) = (&pass.forward, &pass.text_states, &pass.query_states) else {
            return Ok(empty);
        };
        let opts = pass.opts;
        let mut present: Vec<(&str, usize)> = Vec::new();
        for choice in choices {
            if present.iter().any(|(c, _)| *c == choice) {
                continue;
            }
            let lower = choice.to_lowercase();
            if let Some(index) = pass.input.prefix_tokens.iter().position(|t| t.to_lowercase() == lower) {
                present.push((choice, index));
            }
        }
        if present.is_empty() {
            return Ok(empty);
        }
        let spans: Vec<(usize, usize)> = present.iter().map(|&(_, i)| (i, i + 1)).collect();
        let probabilities: Vec<f32> = self
            .head
            .score_explicit_spans(fwd, ts, qs, query, &spans)?
            .into_iter()
            .map(|x| self.pair_probability(x))
            .collect();

        if let Some(preferred) = preferred {
            let preferred: Vec<_> = preferred
                .iter()
                .filter_map(|(choice, s, e)| {
                    present.iter().position(|(c, _)| c == choice).map(|i| (choice, probabilities[i], *s, *e))
                })
                .collect();
            let format = |(choice, p, s, e): &(&String, f32, usize, usize)| {
                if opts.include_confidence || include_spans {
                    let mut value = Map::new();
                    value.insert("text".into(), json!(choice));
                    if opts.include_confidence {
                        value.insert("confidence".into(), json!(p));
                    }
                    if include_spans {
                        value.insert("start".into(), json!(s));
                        value.insert("end".into(), json!(e));
                    }
                    Value::Object(value)
                } else {
                    json!(choice)
                }
            };
            if !preferred.is_empty() {
                return Ok(if is_scalar {
                    format(&preferred[0])
                } else {
                    Value::Array(preferred.iter().map(format).collect())
                });
            }
        }

        let threshold = configured_threshold.unwrap_or(default_threshold);
        let plain = |i: usize| {
            if opts.include_confidence {
                json!({"text": present[i].0, "confidence": probabilities[i]})
            } else {
                json!(present[i].0)
            }
        };
        if !is_scalar {
            return Ok(Value::Array((0..present.len()).filter(|&i| probabilities[i] >= threshold).map(plain).collect()));
        }
        let best = argmax(&probabilities);
        Ok(if probabilities[best] < threshold { Value::Null } else { plain(best) })
    }

    fn decode_relations(&self, pass: &Pass, schema: &Schema) -> Result<Vec<(String, Vec<Value>)>> {
        let cfg = &self.config.boundary_head;
        let (Some(scorer), Some(fwd), Some(text_states), Some(query_states)) =
            (&self.relation_scorer, &pass.forward, &pass.text_states, &pass.query_states)
        else {
            return Ok(Vec::new());
        };
        let scores = &fwd.scores;
        let mut roles = Vec::new();
        let mut relation_state_rows = Vec::new();
        let mut relation_specs = Vec::new();
        for (gi, group) in pass.input.groups.iter().enumerate() {
            if group.kind != TaskKind::Relations {
                continue;
            }
            let (Some(head), Some(tail)) = (pass.query(gi, 0), pass.query(gi, 1)) else { continue };
            let head_state = query_states.get(head)?;
            let tail_state = query_states.get(tail)?;
            let state = if cfg.directional_relation_states {
                Tensor::cat(&[&head_state, &tail_state], 0)?
            } else {
                ((head_state + tail_state)? * 0.5)?
            };
            relation_state_rows.push(state);
            roles.push(RelationRoles { head_query: head, tail_query: tail });
            relation_specs.push(&schema.relations[group.spec_index]);
        }
        if roles.is_empty() {
            return Ok(Vec::new());
        }

        let probs: Vec<Vec<f32>> =
            scores.pair_logits.iter().map(|row| row.iter().map(|&x| sigmoid(x)).collect()).collect();
        let settings = RelationProposalSettings {
            heads_per_relation: cfg.relation_heads_per_type,
            tails_per_relation: cfg.relation_tails_per_type,
            pair_cap: cfg.relation_pair_cap,
            argument_threshold: cfg.relation_argument_proposal_threshold,
        };
        let pairs = propose_relation_pairs(&probs, &scores.spans, &roles, &settings);
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let relation_states = Tensor::stack(&relation_state_rows, 0)?;
        let logits = scorer.forward(text_states, &relation_states, &pairs)?;

        let mut edges: Vec<(String, Vec<RelationEdge>)> = Vec::new();
        for (pair, logit) in pairs.iter().zip(logits) {
            let spec = relation_specs[pair.relation];
            let score = sigmoid(logit / cfg.relation_temperature);
            if score < spec.threshold.unwrap_or(pass.opts.threshold) {
                continue;
            }
            let (Some(head), Some(tail)) = (pass.mention(pair.head), pass.mention(pair.tail)) else { continue };
            let edge = RelationEdge { score, head, tail };
            match edges.iter_mut().find(|(name, _)| *name == spec.name) {
                Some((_, list)) => list.push(edge),
                None => edges.push((spec.name.clone(), vec![edge])),
            }
        }

        Ok(edges
            .into_iter()
            .map(|(name, list)| {
                let values = deduplicate_relation_edges(list)
                    .into_iter()
                    .map(|edge| format_relation(&edge, pass.opts))
                    .collect();
                (name, values)
            })
            .collect())
    }
}

fn is_empty_value(v: &Value) -> bool {
    v.is_null() || v.as_array().is_some_and(Vec::is_empty)
}

/// `finalize_spans` on character offsets: overlap resolution, then first for scalars.
fn finalize_mentions(mentions: Vec<Mention>, is_scalar: bool, policy: OverlapPolicy) -> Vec<Mention> {
    let scored: Vec<ScoredSpan> = mentions.iter().map(|m| ScoredSpan { score: m.1, start: m.2, end: m.3 }).collect();
    let mut resolved: Vec<Mention> = resolve_overlaps(&scored, policy)
        .into_iter()
        .filter_map(|s| mentions.iter().find(|m| m.2 == s.start && m.3 == s.end && m.1 == s.score).cloned())
        .collect();
    if is_scalar {
        resolved.truncate(1);
    }
    resolved
}

fn format_mention((surface, score, start, end): &Mention, opts: &ExtractOptions) -> Value {
    match (opts.include_spans, opts.include_confidence) {
        (true, true) => json!({"text": surface, "confidence": score, "start": start, "end": end}),
        (true, false) => json!({"text": surface, "start": start, "end": end}),
        (false, true) => json!({"text": surface, "confidence": score}),
        (false, false) => json!(surface),
    }
}

/// `_format_attributed_entity`: like `format_mention`, but always an object
/// (`text` plus optional `confidence`/`start`/`end`) with attribute groups
/// merged in, since a schema with attributes forces the object shape.
fn format_attributed_entity((surface, score, start, end): &Mention, opts: &ExtractOptions, extra: &Map<String, Value>) -> Value {
    let mut value = Map::new();
    value.insert("text".to_string(), json!(surface));
    if opts.include_confidence {
        value.insert("confidence".to_string(), json!(score));
    }
    if opts.include_spans {
        value.insert("start".to_string(), json!(start));
        value.insert("end".to_string(), json!(end));
    }
    value.extend(extra.clone());
    Value::Object(value)
}

fn format_structure_field(mentions: &[Mention], is_scalar: bool, opts: &ExtractOptions) -> Value {
    if is_scalar {
        mentions.first().map_or(Value::Null, |m| format_mention(m, opts))
    } else {
        Value::Array(mentions.iter().map(|m| format_mention(m, opts)).collect())
    }
}

/// `_format_entity_dict` / `_format_struct` list dedup: by lowercased text,
/// plus offsets when present.
fn dedup_list(values: Vec<Value>) -> Value {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let (text, span) = match &value {
            Value::String(s) => (s.clone(), None),
            Value::Object(o) => (
                o.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
                o.get("start").zip(o.get("end")).map(|(s, e)| (s.to_string(), e.to_string())),
            ),
            _ => {
                out.push(value);
                continue;
            }
        };
        if !text.is_empty() && seen.insert((text.to_lowercase(), span)) {
            out.push(value);
        }
    }
    Value::Array(out)
}

/// `_format_struct`: dedup list fields, map empty scalars to `null`.
fn format_struct(instance: Map<String, Value>) -> Value {
    Value::Object(
        instance
            .into_iter()
            .map(|(k, v)| {
                let v = match v {
                    Value::Array(items) => dedup_list(items),
                    Value::String(s) if s.is_empty() => Value::Null,
                    other => other,
                };
                (k, v)
            })
            .collect(),
    )
}

fn format_relation(edge: &RelationEdge, opts: &ExtractOptions) -> Value {
    let side = |m: &(String, usize, usize)| match (opts.include_spans, opts.include_confidence) {
        (true, true) => json!({"text": m.0, "start": m.1, "end": m.2, "confidence": edge.score}),
        (true, false) => json!({"text": m.0, "start": m.1, "end": m.2}),
        (false, true) => json!({"text": m.0, "confidence": edge.score}),
        (false, false) => json!(m.0),
    };
    if opts.include_spans || opts.include_confidence {
        json!({"head": side(&edge.head), "tail": side(&edge.tail)})
    } else {
        json!([edge.head.0, edge.tail.0])
    }
}

fn classification_probs(spec: &ClassificationSpec, logits: &[f32]) -> Vec<f32> {
    let use_sigmoid = match spec.activation {
        ClassActivation::Sigmoid => true,
        ClassActivation::Softmax => false,
        ClassActivation::Auto => spec.multi_label,
    };
    if use_sigmoid { logits.iter().map(|&x| sigmoid(x)).collect() } else { softmax(logits) }
}

fn classification_value(spec: &ClassificationSpec, logits: &[f32], include_confidence: bool) -> Value {
    let probs = classification_probs(spec, logits);
    let entry = |i: usize| {
        if include_confidence {
            json!({"label": spec.labels[i], "confidence": probs[i]})
        } else {
            json!(spec.labels[i])
        }
    };
    if spec.multi_label {
        let mut chosen: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] >= spec.cls_threshold).collect();
        if chosen.is_empty() {
            chosen.push(argmax(&probs));
        }
        Value::Array(chosen.into_iter().map(entry).collect())
    } else {
        entry(argmax(&probs))
    }
}

// ---------------------------------------------------------------------------
// Span (legacy) architecture model
// ---------------------------------------------------------------------------

/// Checkpoint using the legacy "span" GLiNER2 architecture (see
/// `heads::{CountPred, CountLstmStep0, SpanRep}`). Supports classification and
/// flat entity extraction only — the PII/guardrail checkpoints never use
/// structures, relations or record extraction, so those bail with a clear
/// error instead of being ported.
pub struct SpanModel {
    encoder: Encoder,
    classifier: Classifier,
    count_pred: Option<CountPred>,
    count_embed: Option<CountLstmStep0>,
    span_rep: Option<SpanRep>,
    processor: Processor,
    #[allow(dead_code)]
    config: SpanConfig,
    #[allow(dead_code)]
    encoder_config: EncoderConfig,
    device: Device,
}

impl SpanModel {
    pub fn load(model_dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let (config, encoder_config) = load_span_configs(model_dir)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let weights = model_dir.join("model.safetensors");
        // SAFETY: the checkpoint file must not be modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], dtype, device) }
            .with_context(|| format!("mapping {}", weights.display()))?;
        let head_vb = vb.clone().set_dtype(DType::F32);
        Self::from_parts(config, encoder_config, tokenizer, vb, head_vb, device)
    }

    /// Same as [`Self::load`] but from in-memory file contents (see
    /// [`GLiNER2::load_from_bytes`]).
    pub fn load_from_bytes(
        config_json: &str,
        encoder_config_json: &str,
        tokenizer_bytes: &[u8],
        weights: &[u8],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (config, encoder_config) = load_span_configs_str(config_json, encoder_config_json)?;
        let tokenizer =
            Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let tensors = crate::safetensors32::load_buffer(weights, device).context("loading model.safetensors")?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let head_vb = vb.clone().set_dtype(DType::F32);
        Self::from_parts(config, encoder_config, tokenizer, vb, head_vb, device)
    }

    fn from_parts(
        config: SpanConfig,
        encoder_config: EncoderConfig,
        tokenizer: Tokenizer,
        vb: VarBuilder,
        head_vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let hidden = encoder_config.hidden_size();

        let encoder = Encoder::load(vb.pp("encoder"), &encoder_config).context("loading encoder")?;
        let classifier = Classifier::load_span(head_vb.pp("classifier"), hidden).context("loading classifier")?;

        // Entity extraction needs `count_pred`/`count_embed`/`span_rep`; only
        // `count_lstm` is ported (see `heads::CountLstmStep0`), so leave them
        // unloaded for other counting layers. Classification still works.
        let (count_pred, count_embed, span_rep) = if config.counting_layer == "count_lstm" {
            (
                Some(CountPred::load(head_vb.pp("count_pred"), hidden).context("loading count_pred")?),
                Some(CountLstmStep0::load(head_vb.pp("count_embed"), hidden).context("loading count_embed")?),
                Some(SpanRep::load(head_vb.pp("span_rep"), hidden, config.max_width).context("loading span_rep")?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            encoder,
            classifier,
            count_pred,
            count_embed,
            span_rep,
            processor: Processor::new(tokenizer),
            config,
            encoder_config,
            device: device.clone(),
        })
    }

    pub fn set_word_splitter(&mut self, splitter: WordSplitter) {
        self.processor.word_splitter = splitter;
    }

    pub fn word_splitter(&self) -> WordSplitter {
        self.processor.word_splitter
    }

    fn encode(&self, input: &PreparedInput) -> Result<Tensor> {
        let t = input.input_ids.len();
        let ids = Tensor::from_vec(input.input_ids.clone(), (1, t), &self.device)?;
        let mask = Tensor::ones((1, t), DType::U32, &self.device)?;
        Ok(self.encoder.forward(&ids, &mask)?.squeeze(0)?.to_dtype(DType::F32)?)
    }

    /// See [`BoundaryModel::encode_batch`].
    fn encode_batch(&self, inputs: &[PreparedInput]) -> Result<Vec<Tensor>> {
        let b = inputs.len();
        let max_t = inputs.iter().map(|i| i.input_ids.len()).max().unwrap_or(0);
        let mut ids = vec![0u32; b * max_t];
        let mut mask = vec![0u32; b * max_t];
        for (i, input) in inputs.iter().enumerate() {
            for (j, &id) in input.input_ids.iter().enumerate() {
                ids[i * max_t + j] = id;
                mask[i * max_t + j] = 1;
            }
        }
        let ids = Tensor::from_vec(ids, (b, max_t), &self.device)?;
        let mask = Tensor::from_vec(mask, (b, max_t), &self.device)?;
        let states = self.encoder.forward(&ids, &mask)?.to_dtype(DType::F32)?;
        inputs
            .iter()
            .enumerate()
            .map(|(i, input)| Ok(states.narrow(0, i, 1)?.squeeze(0)?.narrow(0, 0, input.input_ids.len())?.contiguous()?))
            .collect()
    }

    fn gather_rows(&self, states: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let idx: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let idx = Tensor::from_vec(idx, positions.len(), &self.device)?;
        Ok(states.index_select(&idx, 0)?)
    }

    fn gather_row(&self, states: &Tensor, position: usize) -> Result<Tensor> {
        Ok(self.gather_rows(states, &[position])?.squeeze(0)?)
    }

    pub fn extract(&self, text: &str, schema: &Schema, opts: &ExtractOptions) -> Result<Value> {
        let input = self.processor.prepare(text, schema, opts.max_words)?;
        let states = self.encode(&input)?;
        self.extract_from_states(&input, &states, text, schema, opts)
    }

    /// [`BoundaryModel::extract_batch`], scoped the same way as
    /// [`Self::extract`] (bails if `schema` has structures/relations).
    pub fn extract_batch(&self, texts: &[&str], schema: &Schema, opts: &ExtractOptions) -> Result<Vec<Value>> {
        if !schema.structures.is_empty() || !schema.relations.is_empty() {
            bail!(
                "structures/relations extraction is not implemented for the span architecture \
                 (this checkpoint only supports entities and classification)"
            );
        }
        let inputs: Vec<PreparedInput> =
            texts.iter().map(|text| self.processor.prepare(text, schema, opts.max_words)).collect::<Result<_>>()?;
        let states = self.encode_batch(&inputs)?;
        (0..texts.len())
            .map(|i| self.extract_from_states(&inputs[i], &states[i], texts[i], schema, opts))
            .collect()
    }

    fn extract_from_states(&self, input: &PreparedInput, states: &Tensor, text: &str, schema: &Schema, opts: &ExtractOptions) -> Result<Value> {
        if !schema.structures.is_empty() || !schema.relations.is_empty() {
            bail!(
                "structures/relations extraction is not implemented for the span architecture \
                 (this checkpoint only supports entities and classification)"
            );
        }
        let mut out = Map::new();
        for group in &input.groups {
            if group.kind == TaskKind::Classifications {
                let spec = &schema.classifications[group.spec_index];
                if group.marker_positions.is_empty() {
                    continue;
                }
                let label_states = self.gather_rows(states, &group.marker_positions)?;
                let logits = self.classifier.forward(&label_states)?;
                out.insert(spec.task.clone(), classification_value(spec, &logits, opts.include_confidence));
            }
        }
        if let Some(group) = input.groups.iter().find(|g| g.kind == TaskKind::Entities) {
            out.insert("entities".to_string(), Value::Object(self.decode_entities(input, states, text, schema, group, opts)?));
        }
        Ok(Value::Object(out))
    }

    fn decode_entities(
        &self,
        input: &PreparedInput,
        states: &Tensor,
        text: &str,
        schema: &Schema,
        group: &crate::processor::PromptGroup,
        opts: &ExtractOptions,
    ) -> Result<Map<String, Value>> {
        let empty = || {
            schema
                .entities
                .iter()
                .map(|spec| {
                    let v = match spec.dtype {
                        EntityDtype::List => Value::Array(Vec::new()),
                        EntityDtype::Str => Value::Null,
                    };
                    (spec.name.clone(), v)
                })
                .collect::<Map<String, Value>>()
        };
        let (Some(count_pred), Some(count_embed), Some(span_rep)) = (&self.count_pred, &self.count_embed, &self.span_rep)
        else {
            bail!(
                "this checkpoint's counting_layer ({:?}) isn't supported for entity extraction (only 'count_lstm')",
                self.config.counting_layer
            );
        };
        if group.marker_positions.is_empty() || input.word_positions.is_empty() {
            return Ok(empty());
        }
        let prompt_state = self.gather_row(states, group.prompt_position)?;
        if count_pred.predict(&prompt_state)? == 0 {
            return Ok(empty());
        }

        let field_states = self.gather_rows(states, &group.marker_positions)?;
        let struct_proj = count_embed.forward(&field_states)?; // [M, H]
        let token_states = self.gather_rows(states, &input.word_positions)?; // [L, H]
        let (span_reps, pairs) = span_rep.compute(&token_states, &self.device)?; // [N, H]
        if pairs.is_empty() {
            return Ok(empty());
        }
        let scores = candle_nn::ops::sigmoid(&span_reps.matmul(&struct_proj.t()?)?)?; // [N, M]
        let scores = scores.to_vec2::<f32>()?;
        let offset = input.prefix_tokens.len();

        let mut result = Map::new();
        for (field_index, spec) in schema.entities.iter().enumerate() {
            let threshold = spec.threshold.unwrap_or(opts.threshold);
            let mut candidates: Vec<ScoredSpan> = Vec::new();
            for (n, &(start, width)) in pairs.iter().enumerate() {
                let score = scores[n][field_index];
                if score >= threshold {
                    candidates.push(ScoredSpan { score, start: start + offset, end: start + width + 1 + offset });
                }
            }
            let resolved = match opts.overlap_policy {
                Some(policy) => resolve_overlaps(&candidates, policy),
                // The span architecture's published default: confidence-first
                // greedy suppression, distinct from the shared resolvers above
                // (`finalize_spans` in the Python reference).
                None => greedy_suppress(candidates),
            };
            let mut items: Vec<Mention> = Vec::new();
            for span in resolved {
                if let Some((surface, start, end)) = span_mention(input, text, (span.start, span.end)) {
                    if spec.validators.iter().all(|v| v.validate(&surface)) {
                        items.push((surface, span.score, start, end));
                    }
                }
            }
            let value = match spec.dtype {
                EntityDtype::List => dedup_list(items.iter().map(|m| format_mention(m, opts)).collect()),
                EntityDtype::Str => items.first().map_or(Value::Null, |m| format_mention(m, opts)),
            };
            result.insert(spec.name.clone(), value);
        }
        Ok(result)
    }

    pub fn classification_probabilities(
        &self,
        text: &str,
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<(String, Vec<(String, f32)>)>> {
        let schema = tasks.iter().cloned().fold(Schema::new(), Schema::classification);
        let input = self.processor.prepare(text, &schema, None)?;
        let states = self.encode(&input)?;
        self.classification_probabilities_from_states(&input, &states, tasks)
    }

    /// [`BoundaryModel::classification_probabilities_batch`].
    pub fn classification_probabilities_batch(
        &self,
        texts: &[&str],
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<Vec<(String, Vec<(String, f32)>)>>> {
        let schema = tasks.iter().cloned().fold(Schema::new(), Schema::classification);
        let inputs: Vec<PreparedInput> =
            texts.iter().map(|text| self.processor.prepare(text, &schema, None)).collect::<Result<_>>()?;
        let states = self.encode_batch(&inputs)?;
        inputs
            .iter()
            .zip(states.iter())
            .map(|(input, states)| self.classification_probabilities_from_states(input, states, tasks))
            .collect()
    }

    fn classification_probabilities_from_states(
        &self,
        input: &PreparedInput,
        states: &Tensor,
        tasks: &[ClassificationSpec],
    ) -> Result<Vec<(String, Vec<(String, f32)>)>> {
        let mut out = Vec::new();
        for group in input.groups.iter().filter(|g| g.kind == TaskKind::Classifications) {
            let spec = &tasks[group.spec_index];
            if group.marker_positions.is_empty() {
                continue;
            }
            let label_states = self.gather_rows(states, &group.marker_positions)?;
            let logits = self.classifier.forward(&label_states)?;
            let probs = classification_probs(spec, &logits);
            out.push((spec.task.clone(), spec.labels.iter().cloned().zip(probs).collect()));
        }
        Ok(out)
    }
}

/// Character-offset mention for a half-open word-index span, mirroring
/// `Pass::mention` but against `input.words` directly (span checkpoints never
/// request structures here, so `prefix_tokens` is always empty).
fn span_mention(input: &PreparedInput, text: &str, (start, end): (usize, usize)) -> Option<(String, usize, usize)> {
    let words = &input.words;
    let offset = input.prefix_tokens.len();
    let (s, e) = (start.checked_sub(offset)?, end.checked_sub(offset)?);
    if !(s < e && e <= words.len()) {
        return None;
    }
    let surface = text[words[s].byte_start..words[e - 1].byte_end].trim();
    (!surface.is_empty()).then(|| (surface.to_string(), words[s].char_start, words[e - 1].char_end))
}

/// `finalize_spans` with `overlap_policy=None`: sort by descending score, then
/// greedily keep spans that don't half-open-overlap an already-kept one.
fn greedy_suppress(mut candidates: Vec<ScoredSpan>) -> Vec<ScoredSpan> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<ScoredSpan> = Vec::new();
    'candidates: for cand in candidates {
        for existing in &kept {
            if cand.start < existing.end && existing.start < cand.end {
                continue 'candidates;
            }
        }
        kept.push(cand);
    }
    kept
}
