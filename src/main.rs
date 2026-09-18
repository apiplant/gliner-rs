use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::Parser;
use gliner_rs::cli_schema::{build_schema, CliSchemaArgs};
use gliner_rs::model_path::{self, VariantDef};
use gliner_rs::{ExtractOptions, GLiNER2, OverlapPolicy, WordSplitter};

const VARIANTS: &[VariantDef] = &[
    VariantDef { key: "multi", hf_repo: "fastino/gliner2.5-multi-v1" },
    VariantDef { key: "small", hf_repo: "fastino/gliner2.5-small-v1" },
    VariantDef { key: "base", hf_repo: "fastino/gliner2.5-base-v1" },
];

/// Run GLiNER2 (boundary architecture) extraction on a text.
///
/// Example:
///   gliner --text "Alice works for Acme in Paris." \
///     --entities person,company,location --relations works_for,located_in \
///     --classify "sentiment=positive,negative,neutral" --spans --confidence
#[derive(Parser, Debug)]
#[command(name = "gliner", version)]
struct Args {
    /// Checkpoint directory. Defaults to the `--model-variant` checkpoint in
    /// the gliner-rs cache directory, downloading it there first if needed.
    #[arg(long, env = "GLINER_MODEL")]
    model: Option<PathBuf>,

    /// Which GLiNER2.5 checkpoint to use when `--model` isn't given. Defaults to `multi`.
    #[arg(long, value_enum)]
    model_variant: Option<Variant>,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Variant {
    Multi,
    Small,
    Base,
}

impl Variant {
    fn key(self) -> &'static str {
        match self {
            Variant::Multi => "multi",
            Variant::Small => "small",
            Variant::Base => "base",
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if args.fp16 { DType::F16 } else { DType::F32 };
    let model_path = model_path::resolve(VARIANTS, "multi", args.model.clone(), args.model_variant.map(Variant::key))?;

    let text = match args.text {
        Some(t) => t,
        None => std::io::read_to_string(std::io::stdin())?,
    };

    let schema = build_schema(&CliSchemaArgs {
        entities: args.entities.clone(),
        relations: args.relations.clone(),
        json: args.json.clone(),
        legacy_structures: args.legacy_structures,
        classify: args.classify.clone(),
    })?;

    let started = Instant::now();
    let mut model = GLiNER2::load(&model_path, &device, dtype)?;
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
