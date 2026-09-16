//! PII detection/redaction with GLiNER2's privacy-filter checkpoints.

use std::io::{BufRead, IsTerminal};
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use gliner_rs::model_path::{self, VariantDef};
use gliner_rs::{ExtractOptions, GLiNER2};
use serde_json::{json, Value};

const VARIANTS: &[VariantDef] = &[
    VariantDef {
        key: "privacy",
        default_relative: "models/gliner2-privacy-filter-PII-multi",
        hf_repo: "fastino/gliner2-privacy-filter-PII-multi",
    },
    VariantDef {
        key: "guardrails",
        default_relative: "models/GLiNER2-Guardrails-PII-Multi",
        hf_repo: "fastino/GLiNER2-Guardrails-PII-Multi",
    },
];

/// The privacy-filter checkpoint's full 42-label PII taxonomy.
const DEFAULT_LABELS: &[&str] = &[
    "person",
    "full_name",
    "first_name",
    "middle_name",
    "last_name",
    "date_of_birth",
    "email",
    "phone_number",
    "address",
    "street_address",
    "city",
    "state_or_region",
    "postal_code",
    "country",
    "government_id",
    "national_id_number",
    "passport_number",
    "drivers_license_number",
    "tax_id",
    "tax_number",
    "bank_account",
    "account_number",
    "routing_number",
    "iban",
    "payment_card",
    "card_number",
    "card_expiry",
    "card_cvv",
    "username",
    "ip_address",
    "account_id",
    "sensitive_account_id",
    "password",
    "secret",
    "api_key",
    "access_token",
    "recovery_code",
    "sensitive_date",
    "document_date",
    "expiration_date",
    "transaction_date",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Variant {
    Privacy,
    Guardrails,
}

impl Variant {
    fn key(self) -> &'static str {
        match self {
            Variant::Privacy => "privacy",
            Variant::Guardrails => "guardrails",
        }
    }
}

/// Detect and optionally redact personally identifiable information.
///
/// Texts come from positional arguments, `--file` (one per line), or piped
/// stdin (one per line). Prints one JSON object per line by default; with
/// `--redact`, prints the text with matched spans replaced by `[LABEL]`.
#[derive(Parser, Debug)]
#[command(version, after_help = "Examples:
  gliner-pii \"Email john.smith@acme.com or call +1 415 555 0199.\"
  gliner-pii --redact -f transcripts.txt
  gliner-pii -l email,phone_number,person \"...\"
  echo \"my api key is sk-abc123\" | gliner-pii --labels api_key,secret
  gliner-pii --pretty -f transcripts.txt
  gliner-pii setup /mnt/ai/gliner          # save model paths in one go")]
struct Args {
    /// Texts to scan.
    texts: Vec<String>,

    /// Labels to detect, comma-separated. Defaults to the full PII taxonomy.
    #[arg(short, long)]
    labels: Option<String>,

    /// Detection threshold.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Read texts from a file, one per line.
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Replace detected spans in the text with `[LABEL]` instead of printing JSON.
    #[arg(short, long)]
    redact: bool,

    /// Pretty-print the JSON output (indented, instead of one line per text).
    #[arg(short, long)]
    pretty: bool,

    /// Checkpoint directory. Defaults to `./models/<variant's checkpoint>`,
    /// then a path saved from a previous interactive prompt.
    #[arg(long, env = "GLINER_MODEL")]
    model: Option<PathBuf>,

    /// Which checkpoint to use when `--model` isn't given. Defaults to
    /// `privacy`, or whatever was last used.
    #[arg(long, value_enum)]
    model_variant: Option<Variant>,

    /// Run on CUDA device 0 (requires the `cuda` feature).
    #[arg(long)]
    cuda: bool,

    /// Run the encoder in float16.
    #[arg(long)]
    fp16: bool,

    /// Subcommand.
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Save model paths in one go from a base models directory containing
    /// checkpoint subdirectories (e.g. /mnt/ai/gliner).
    Setup {
        /// Base models directory containing the checkpoint subdirectories.
        base: PathBuf,
    },
}

fn redact(text: &str, entities: &serde_json::Value) -> String {
    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    if let Some(obj) = entities.as_object() {
        for (label, mentions) in obj {
            let Some(list) = mentions.as_array() else { continue };
            for mention in list {
                let (Some(start), Some(end)) =
                    (mention.get("start").and_then(|v| v.as_u64()), mention.get("end").and_then(|v| v.as_u64()))
                else {
                    continue;
                };
                spans.push((start as usize, end as usize, label.to_uppercase()));
            }
        }
    }
    spans.sort_by(|a, b| a.0.cmp(&b.0));

    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut cursor = 0usize;
    for (start, end, label) in spans {
        if start < cursor || end > chars.len() {
            continue; // overlapping or out-of-range span
        }
        out.extend(&chars[cursor..start]);
        out.push_str(&format!("[{label}]"));
        cursor = end;
    }
    out.extend(&chars[cursor..]);
    out
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(Command::Setup { base }) = args.command {
        return model_path::setup("gliner-pii", VARIANTS, &base);
    }

    let mut texts = args.texts.clone();
    if let Some(path) = &args.file {
        let content = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        texts.extend(content.lines().filter(|l| !l.trim().is_empty()).map(str::to_string));
    }
    if texts.is_empty() && !std::io::stdin().is_terminal() {
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            if !line.trim().is_empty() {
                texts.push(line);
            }
        }
    }
    if texts.is_empty() {
        anyhow::bail!("no input: pass text arguments, --file, or pipe lines on stdin");
    }

    let labels: Vec<String> = match &args.labels {
        Some(l) => l.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        None => DEFAULT_LABELS.iter().map(|s| s.to_string()).collect(),
    };
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

    let model_path =
        model_path::resolve("gliner-pii", VARIANTS, "privacy", args.model.clone(), args.model_variant.map(Variant::key))?;

    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if args.fp16 { DType::F16 } else { DType::F32 };
    let model = GLiNER2::load(&model_path, &device, dtype)
        .with_context(|| format!("loading model from {} (set --model or GLINER_MODEL)", model_path.display()))?;

    let opts = ExtractOptions { threshold: args.threshold, include_confidence: true, include_spans: true, ..Default::default() };

    for text in &texts {
        // `extract_entities` returns `{"entities": {label: [...]}}`; unwrap the
        // single top-level key to get the flat per-label map.
        let mut result = model.extract_entities(text, &label_refs, &opts)?;
        let entities = result.get_mut("entities").map(Value::take).unwrap_or(json!({}));
        if args.redact {
            println!("{}", redact(text, &entities));
        } else if args.pretty {
            println!("{}", serde_json::to_string_pretty(&json!({ "text": text, "entities": entities }))?);
        } else {
            println!("{}", json!({ "text": text, "entities": entities }));
        }
    }
    Ok(())
}
