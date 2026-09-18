//! wasm-bindgen bindings for running GLiNER2 in the browser.
//!
//! Model files (`config.json`, `encoder_config/config.json`, `tokenizer.json`,
//! `model.safetensors`) are fetched by JS (typically from the Hugging Face
//! Hub) and handed to [`WasmModel::load`] as bytes — there is no filesystem
//! here. Everything else (schema building, extraction, classification) goes
//! through JSON strings so the JS side never needs a Rust struct layout.

use candle_core::{DType, Device};
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use crate::cli_schema::{build_schema, CliSchemaArgs};
use crate::schema::{ClassActivation, ClassificationSpec};
use crate::{ExtractOptions, GLiNER2, OverlapPolicy, WordSplitter};

fn to_js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Like [`to_js_err`] but keeps an [`anyhow`] error's whole context chain —
/// the outermost context alone ("loading model.safetensors") says nothing
/// about what actually went wrong.
fn anyhow_to_js_err(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

/// Call once from JS before anything else, to get readable panic messages
/// (from candle shape mismatches etc.) in the browser console.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmModel {
    inner: GLiNER2,
}

/// Options shared by every extraction call. Mirrors `gliner`'s
/// `--threshold`/`--confidence`/`--spans`/`--overlap`/`--char-split` flags.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct WasmOpts {
    threshold: Option<f32>,
    confidence: bool,
    spans: bool,
    overlap: Option<String>,
    char_split: bool,
}

impl WasmOpts {
    fn parse(json: &str) -> Result<Self, JsValue> {
        if json.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(json).map_err(to_js_err)
    }

    fn extract_options(&self) -> Result<ExtractOptions, JsValue> {
        Ok(ExtractOptions {
            threshold: self.threshold.unwrap_or(0.5),
            include_confidence: self.confidence,
            include_spans: self.spans,
            overlap_policy: self.overlap.as_deref().map(str::parse::<OverlapPolicy>).transpose().map_err(to_js_err)?,
            max_words: None,
        })
    }
}

#[wasm_bindgen]
impl WasmModel {
    /// Loads a checkpoint from its four file contents, in float32 on CPU
    /// (the only combination that makes sense in a browser tab).
    ///
    /// `weights` is `model.safetensors`; `tokenizer` is `tokenizer.json`;
    /// `config_json`/`encoder_config_json` are `config.json` and
    /// `encoder_config/config.json`.
    #[wasm_bindgen]
    pub fn load(
        weights: &[u8],
        tokenizer: &[u8],
        config_json: &str,
        encoder_config_json: &str,
    ) -> Result<WasmModel, JsValue> {
        let inner = GLiNER2::load_from_bytes(
            config_json,
            encoder_config_json,
            tokenizer,
            weights,
            &Device::Cpu,
            DType::F32,
        )
        .map_err(anyhow_to_js_err)?;
        Ok(WasmModel { inner })
    }

    /// Use the character-level word splitter (Chinese, Japanese, ...).
    #[wasm_bindgen(js_name = setCharSplit)]
    pub fn set_char_split(&mut self, on: bool) {
        self.inner.set_word_splitter(if on { WordSplitter::Char } else { WordSplitter::Whitespace });
    }

    /// Runs extraction built from the same flag syntax as the `gliner` CLI
    /// binary (`--entities`, `--relations`, `--json`, `--classify`). Powers
    /// the "generic schema" and PII/guardrail demos. Returns pretty JSON.
    ///
    /// `entities`: `label` or `label:description`, one per array entry.
    /// `structures`: `name=field1::str,field2::[a|b],field3::list::description`.
    /// `classify`: `task=label1,label2` (prefix task with `+` for multi-label).
    #[wasm_bindgen(js_name = extractCli)]
    pub fn extract_cli(
        &self,
        text: &str,
        entities: Vec<String>,
        relations: Vec<String>,
        structures: Vec<String>,
        classify: Vec<String>,
        legacy_structures: bool,
        opts_json: &str,
    ) -> Result<String, JsValue> {
        let schema = build_schema(&CliSchemaArgs { entities, relations, json: structures, legacy_structures, classify })
            .map_err(anyhow_to_js_err)?;
        let opts = WasmOpts::parse(opts_json)?.extract_options()?;
        let result = self.inner.extract(text, &schema, &opts).map_err(anyhow_to_js_err)?;
        serde_json::to_string_pretty(&result).map_err(to_js_err)
    }

    /// Full-detail classification: every label's probability for every
    /// task, with per-task multi-label/threshold/activation/prompt/examples
    /// (mirrors every option `gliner-classify`'s interactive shell exposes).
    ///
    /// `tasks_json` is a JSON array of
    /// `{task, labels: [{label, description?}], multiLabel?, threshold?,
    /// activation?: "auto"|"sigmoid"|"softmax", prompt?, examples?: [[input, label]]}`.
    ///
    /// Returns JSON `[{task, labels: [{label, prob}]}]`, labels sorted by
    /// descending probability.
    #[wasm_bindgen(js_name = classify)]
    pub fn classify(&self, text: &str, tasks_json: &str) -> Result<String, JsValue> {
        let raw: Vec<RawTask> = serde_json::from_str(tasks_json).map_err(to_js_err)?;
        if raw.is_empty() {
            return Err(to_js_err("no classification tasks given"));
        }
        let specs: Vec<ClassificationSpec> = raw.into_iter().map(RawTask::into_spec).collect();
        let results = self.inner.classification_probabilities(text, &specs).map_err(anyhow_to_js_err)?;
        let out: Vec<Value> = results
            .into_iter()
            .map(|(task, mut labels)| {
                labels.sort_by(|a, b| b.1.total_cmp(&a.1));
                json!({
                    "task": task,
                    "labels": labels.into_iter().map(|(label, prob)| json!({"label": label, "prob": prob})).collect::<Vec<_>>(),
                })
            })
            .collect();
        serde_json::to_string(&Value::Array(out)).map_err(to_js_err)
    }
}

#[derive(serde::Deserialize)]
struct RawLabel {
    label: String,
    description: Option<String>,
}

#[derive(serde::Deserialize)]
struct RawTask {
    task: String,
    labels: Vec<RawLabel>,
    #[serde(rename = "multiLabel", default)]
    multi_label: bool,
    threshold: Option<f32>,
    activation: Option<String>,
    prompt: Option<String>,
    #[serde(default)]
    examples: Vec<(String, String)>,
}

impl RawTask {
    fn into_spec(self) -> ClassificationSpec {
        let mut spec = ClassificationSpec::new(self.task, self.labels.iter().map(|l| l.label.clone()));
        for label in &self.labels {
            if let Some(desc) = &label.description {
                spec = spec.with_label_description(label.label.clone(), desc.clone());
            }
        }
        if self.multi_label {
            spec = spec.multi_label(self.threshold.unwrap_or(0.5));
        } else if let Some(t) = self.threshold {
            spec.cls_threshold = t;
        }
        spec.activation = match self.activation.as_deref() {
            Some("sigmoid") => ClassActivation::Sigmoid,
            Some("softmax") => ClassActivation::Softmax,
            _ => ClassActivation::Auto,
        };
        spec.prompt = self.prompt;
        spec.examples = self.examples;
        spec
    }
}
