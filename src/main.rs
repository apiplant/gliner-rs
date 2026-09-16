use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{DType, Device};
use clap::Parser;
use gliner_rs::{ClassificationSpec, ExtractOptions, GLiNER2, OverlapPolicy, Schema, StructureMode, StructureSpec, WordSplitter};

/// Run GLiNER2 (boundary architecture) extraction on a text.
///
/// Example:
///   gliner-rs --model ../ --text "Alice works for Acme in Paris." \
///     --entities person,company,location --relations works_for,located_in \
///     --classify "sentiment=positive,negative,neutral" --spans --confidence
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Checkpoint directory (config.json, encoder_config/, tokenizer.json, model.safetensors).
    #[arg(long, default_value = "..")]
    model: PathBuf,

    /// Input text (reads stdin when omitted).
    #[arg(long)]
    text: Option<String>,

    /// Comma-separated entity labels. Use `label:description` for descriptions.
    #[arg(long, value_delimiter = ',')]
    entities: Vec<String>,

    /// Comma-separated relation types.
    #[arg(long, value_delimiter = ',')]
    relations: Vec<String>,

    /// Structure `name=field1::str,field2::[a|b],field3::list::description`; repeatable.
    /// Separate fields with `;` instead of `,` when descriptions contain commas.
    #[arg(long)]
    json: Vec<String>,

    /// Use the legacy aggregate structure decoder instead of the record head.
    #[arg(long)]
    legacy_structures: bool,

    /// Classification task `task=label1,label2`; repeatable. Prefix the task with `+` for multi-label.
    #[arg(long)]
    classify: Vec<String>,

    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Include character offsets.
    #[arg(long)]
    spans: bool,

    /// Include confidences.
    #[arg(long)]
    confidence: bool,

    /// Overlap policy: flat (default), nested, allow, longest.
    #[arg(long)]
    overlap: Option<String>,

    /// Use the character-level word splitter (Chinese, Japanese, ...).
    #[arg(long)]
    char_split: bool,

    /// Run on CUDA device 0 (requires the `cuda` feature).
    #[arg(long)]
    cuda: bool,

    /// Run the encoder in float16.
    #[arg(long)]
    fp16: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if args.fp16 { DType::F16 } else { DType::F32 };

    let text = match args.text {
        Some(t) => t,
        None => std::io::read_to_string(std::io::stdin())?,
    };

    let mut schema = Schema::new();
    for structure in &args.json {
        let Some((name, fields)) = structure.split_once('=') else {
            bail!("--json expects name=field1,field2, got {structure:?}");
        };
        let separator = if fields.contains(';') { ';' } else { ',' };
        let mut spec = StructureSpec::parse(name.trim(), fields.split(separator).map(str::trim));
        if args.legacy_structures {
            spec.mode = StructureMode::Legacy;
        }
        schema = schema.structure(spec);
    }
    for entity in &args.entities {
        schema = match entity.split_once(':') {
            Some((name, desc)) => schema.entity_with_description(name.trim(), desc.trim()),
            None => schema.entities([entity.trim()]),
        };
    }
    schema = schema.relations(args.relations.iter().map(|r| r.trim()));
    for task in &args.classify {
        let Some((name, labels)) = task.split_once('=') else {
            bail!("--classify expects task=label1,label2, got {task:?}");
        };
        let (name, multi) = match name.strip_prefix('+') {
            Some(n) => (n, true),
            None => (name, false),
        };
        let mut spec = ClassificationSpec::new(name.trim(), labels.split(',').map(str::trim));
        spec.multi_label = multi;
        schema = schema.classification(spec);
    }
    if schema.structures.is_empty() && schema.entities.is_empty() && schema.relations.is_empty() && schema.classifications.is_empty() {
        bail!("nothing to do: pass --json, --entities, --relations and/or --classify");
    }

    let started = Instant::now();
    let mut model = GLiNER2::load(&args.model, &device, dtype)?;
    if args.char_split {
        model.set_word_splitter(WordSplitter::Char);
    }
    eprintln!("loaded in {:.2?}", started.elapsed());

    let opts = ExtractOptions {
        threshold: args.threshold,
        include_confidence: args.confidence,
        include_spans: args.spans,
        overlap_policy: args.overlap.as_deref().map(str::parse::<OverlapPolicy>).transpose()?,
        max_words: None,
    };
    let started = Instant::now();
    let result = model.extract(&text, &schema, &opts)?;
    eprintln!("extracted in {:.2?}", started.elapsed());
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
