//! Export GLiNER inputs/results for the Swift/Apple port.
//!
//! Usage: cargo run --release --example apple_reference -- MODEL_DIR CASES_JSON
//!        [--inputs-only | --trace] [--metal] [--repeat N] [--timing]

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use gliner_rs::config::{load_configs, EncoderConfig};
use gliner_rs::deberta::DebertaV2;
use gliner_rs::processor::Processor;
use gliner_rs::{
    AttributeGroup, ClassActivation, ClassificationSpec, ExtractOptions, GLiNER2, Schema, StructureMode,
    StructureSpec, WordSplitter,
};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

const MAX_TRACE_TOKENS: usize = 2048;

fn long_text() -> String {
    (0..40)
        .map(|i| format!("In report {i}, analyst Jane Smith of Goldman Sachs said Tesla shipped the Model Y from Berlin while Elon Musk visited Austin."))
        .collect::<Vec<_>>()
        .join(" ")
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value.get(name).unwrap_or(&Value::Null)
}

fn string_array(value: &Value) -> Vec<String> {
    value.as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect()
}

fn schema_from_case(case: &Value) -> Result<Schema> {
    let mut schema = Schema::new();
    if let Value::Object(structures) = field(case, "json") {
        for (name, fields) in structures {
            let fields = fields.as_array().with_context(|| format!("fields for structure '{name}' must be an array"))?;
            let mut spec = StructureSpec::parse(name, fields.iter().filter_map(Value::as_str));
            let mode = field(case, "structure_mode").as_str().unwrap_or_else(|| {
                if field(case, "legacy").as_bool().unwrap_or(false) { "legacy" } else { "auto" }
            });
            spec.mode = match mode {
                "auto" => StructureMode::Auto,
                "legacy" => StructureMode::Legacy,
                "latent" => StructureMode::Latent,
                "anchorless" => StructureMode::Anchorless,
                other => bail!("unsupported structure_mode: {other}"),
            };
            spec.anchor = field(case, "anchor").as_str()
                .or_else(|| field(field(case, "anchors"), name).as_str()).map(str::to_owned);
            schema = schema.structure(spec);
        }
    }
    match field(case, "entities") {
        Value::Array(names) => schema = schema.entities(names.iter().filter_map(Value::as_str)),
        Value::Object(map) => {
            for (name, description) in map {
                let description = description.as_str().with_context(|| format!("description for entity '{name}' must be a string"))?;
                schema = schema.entity_with_description(name, description);
            }
        }
        _ => {}
    }
    if let Value::Array(names) = field(case, "relations") {
        schema = schema.relations(names.iter().filter_map(Value::as_str));
    }
    if let Value::Object(groups) = field(case, "entity_attributes") {
        let groups = groups.iter().map(|(name, value)| {
            let labels = string_array(field(value, "labels"));
            let mut group = AttributeGroup::new(labels);
            group.threshold = field(value, "threshold").as_f64().unwrap_or(0.5) as f32;
            group.multi_label = field(value, "multi_label").as_bool().unwrap_or(false);
            if let Value::Array(names) = field(value, "applies_to") {
                group = group.applies_to(names.iter().filter_map(Value::as_str));
            }
            if field(value, "qualify_labels").as_bool().unwrap_or(false) {
                group = group.qualify_labels();
            }
            (name.clone(), group)
        });
        schema = schema.entity_attributes(groups);
    }
    if let Value::Array(tasks) = field(case, "classifications") {
        for task in tasks {
            let labels = string_array(field(task, "labels"));
            let name = field(task, "task").as_str().context("classification task needs a string 'task'")?;
            let mut spec = ClassificationSpec::new(name, labels);
            spec.multi_label = field(task, "multi_label").as_bool().unwrap_or(false);
            spec.cls_threshold = field(task, "cls_threshold").as_f64().unwrap_or(0.5) as f32;
            spec.prompt = field(task, "prompt").as_str().map(str::to_owned);
            spec.label_descriptions = match field(task, "label_descriptions") {
                Value::Object(descriptions) => descriptions.iter().filter_map(|(label, desc)| {
                    desc.as_str().map(|desc| (label.clone(), desc.to_owned()))
                }).collect(),
                Value::Array(descriptions) => descriptions.iter().filter_map(|item| {
                    Some((field(item, "label").as_str()?.to_owned(), field(item, "description").as_str()?.to_owned()))
                }).collect(),
                _ => Vec::new(),
            };
            spec.examples = match field(task, "examples") {
                Value::Array(examples) => examples.iter().filter_map(|item| {
                    Some((field(item, "input").as_str()?.to_owned(), field(item, "output").as_str()?.to_owned()))
                }).collect(),
                _ => Vec::new(),
            };
            spec.activation = match field(task, "activation").as_str().unwrap_or("auto") {
                "auto" => ClassActivation::Auto,
                "sigmoid" => ClassActivation::Sigmoid,
                "softmax" => ClassActivation::Softmax,
                other => bail!("unsupported classification activation: {other}"),
            };
            schema = schema.classification(spec);
        }
    }
    Ok(schema)
}

fn case_text(case: &Value) -> Result<String> {
    if field(case, "long").as_bool().unwrap_or(false) {
        Ok(long_text())
    } else {
        Ok(field(case, "text").as_str().context("each case needs a string 'text'")?.to_owned())
    }
}

fn case_input(case: &Value, processor: &Processor) -> Result<(String, gliner_rs::processor::PreparedInput)> {
    let text = case_text(case)?;
    let schema = schema_from_case(case)?;
    let prepared = processor.prepare(&text, &schema, None)?;
    Ok((text, prepared))
}

fn prepare_processor(model_dir: &Path) -> Result<Processor> {
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!("loading tokenizer.json: {error}"))?;
    Ok(Processor::new(tokenizer))
}

fn set_splitter(processor: &mut Processor, case: &Value) {
    processor.word_splitter = if field(case, "char_split").as_bool().unwrap_or(false) {
        WordSplitter::Char
    } else {
        WordSplitter::Whitespace
    };
}

fn input_record(case: &Value, processor: &mut Processor) -> Result<Value> {
    set_splitter(processor, case);
    let (_original, prepared) = case_input(case, processor)?;
    let marker_positions: Vec<usize> = prepared.groups.iter().flat_map(|group| group.marker_positions.iter().copied()).collect();
    let words: Vec<Value> = prepared.words.iter().map(|word| json!({
        "text": word.text,
        "char_start": word.char_start,
        "char_end": word.char_end,
        "byte_start": word.byte_start,
        "byte_end": word.byte_end,
    })).collect();
    Ok(json!({
        "text": prepared.text,
        "input_ids": prepared.input_ids,
        "word_positions": prepared.word_positions,
        "marker_positions": marker_positions,
        "prefix_tokens": prepared.prefix_tokens,
        "words": words,
    }))
}

fn load_trace_encoder(model_dir: &Path, device: &Device) -> Result<DebertaV2> {
    let (_, encoder_config) = load_configs(model_dir).context("loading checkpoint configs")?;
    let EncoderConfig::Deberta(config) = encoder_config else {
        bail!("--trace currently supports DeBERTa-v2 encoder checkpoints only");
    };
    let weights = model_dir.join("model.safetensors");
    // SAFETY: the checkpoint must not be modified while its memory map is open.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device) }
        .with_context(|| format!("mapping {}", weights.display()))?;
    DebertaV2::load(vb.pp("encoder"), &config).context("loading DeBERTa-v2 encoder")
}

fn trace_record(case: &Value, processor: &mut Processor, encoder: &DebertaV2, device: &Device) -> Result<Value> {
    set_splitter(processor, case);
    let (_original, prepared) = case_input(case, processor)?;
    let sequence_len = prepared.input_ids.len();
    if sequence_len > MAX_TRACE_TOKENS {
        bail!("--trace only accepts up to {MAX_TRACE_TOKENS} input tokens; case has {sequence_len}");
    }
    let ids = Tensor::from_vec(prepared.input_ids.clone(), (1, sequence_len), device)?;
    let mask = Tensor::ones((1, sequence_len), DType::F32, device)?;
    let states = encoder.forward(&ids, &mask)?.to_dtype(DType::F32)?.contiguous()?;
    let encoder_shape = states.dims().to_vec();
    let encoder_values = states.flatten_all()?.to_vec1::<f32>()?;
    let marker_positions: Vec<usize> = prepared.groups.iter().flat_map(|group| group.marker_positions.iter().copied()).collect();
    Ok(json!({
        "input_ids": prepared.input_ids,
        "word_positions": prepared.word_positions,
        "marker_positions": marker_positions,
        "encoder_shape": encoder_shape,
        "encoder_values": encoder_values,
    }))
}

struct Arguments {
    model_path: String,
    cases_path: String,
    inputs_only: bool,
    trace: bool,
    repeat: usize,
    timing: bool,
    metal: bool,
}

fn parse_args() -> Result<Arguments> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut inputs_only = false;
    let mut trace = false;
    let mut repeat = 1usize;
    let mut timing = false;
    let mut metal = false;
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--inputs-only" => inputs_only = true,
            "--trace" => trace = true,
            "--timing" => timing = true,
            "--metal" => metal = true,
            "--repeat" => {
                i += 1;
                let value = raw.get(i).context("--repeat requires a positive integer")?;
                repeat = value.parse().context("--repeat requires a positive integer")?;
                if repeat == 0 { bail!("--repeat must be greater than zero"); }
            }
            flag if flag.starts_with('-') => bail!("unknown option: {flag}"),
            _ => positional.push(raw[i].clone()),
        }
        i += 1;
    }
    if positional.len() != 2 {
        bail!("usage: apple_reference MODEL_DIR CASES_JSON [--inputs-only | --trace] [--repeat N] [--timing] [--metal]");
    }
    if inputs_only && trace {
        bail!("--inputs-only and --trace cannot be used together");
    }
    if repeat > 1 && (inputs_only || trace) {
        bail!("--repeat N greater than 1 is only supported for inference, not --inputs-only or --trace");
    }
    if timing && (inputs_only || trace) {
        bail!("--timing measures extraction only and cannot be combined with --inputs-only or --trace");
    }
    Ok(Arguments {
        model_path: positional.remove(0),
        cases_path: positional.remove(0),
        inputs_only,
        trace,
        repeat,
        timing,
        metal,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let model_dir = Path::new(&args.model_path);
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&args.cases_path)?)
        .with_context(|| format!("parsing cases file {}", args.cases_path))?;
    if args.inputs_only {
        let mut processor = prepare_processor(model_dir)?;
        for case in &cases {
            println!("{}", input_record(case, &mut processor)?);
        }
        return Ok(());
    }
    let device = if args.metal {
        Device::new_metal(0).context("initializing the requested Metal GPU (no CPU fallback)")?
    } else {
        Device::Cpu
    };
    eprintln!("Backend: {} / F32", if args.metal { "Candle Metal GPU" } else { "Candle CPU" });
    if args.trace {
        let mut processor = prepare_processor(model_dir)?;
        let encoder = load_trace_encoder(model_dir, &device)?;
        for case in &cases {
            println!("{}", trace_record(case, &mut processor, &encoder, &device)?);
        }
        return Ok(());
    }

    let loading = Instant::now();
    let mut model = GLiNER2::load(model_dir, &device, DType::F32)?;
    // Charge queued GPU initialization work to loading, not the first request.
    device.synchronize().context("synchronizing model initialization")?;
    if args.timing {
        eprintln!("Loaded in {:.6} s", loading.elapsed().as_secs_f64());
    }
    let mut stdout = std::io::stdout().lock();
    for (case_index, case) in cases.iter().enumerate() {
        model.set_word_splitter(if field(case, "char_split").as_bool().unwrap_or(false) {
            WordSplitter::Char
        } else {
            WordSplitter::Whitespace
        });
        let text = case_text(case)?;
        let schema = schema_from_case(case)?;
        let opts = ExtractOptions {
            threshold: field(case, "threshold").as_f64().unwrap_or(0.5) as f32,
            include_confidence: true,
            include_spans: true,
            ..Default::default()
        };
        for repetition in 1..=args.repeat {
            let started = Instant::now();
            let result = model.extract(&text, &schema, &opts)?;
            let json = result.to_string();
            use std::io::Write;
            writeln!(stdout, "{json}")?;
            stdout.flush()?;
            if args.timing {
                eprintln!("Case {}/{}, run {}: {:.6} s", case_index + 1, cases.len(), repetition, started.elapsed().as_secs_f64());
            }
        }
    }
    Ok(())
}
