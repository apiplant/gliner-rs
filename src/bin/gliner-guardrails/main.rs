//! LLM safety moderation with GLiNER2's guardrails checkpoints
//! (jailbreak/toxicity/refusal classification).

use std::io::{BufRead, IsTerminal};
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, ValueEnum};
use gliner_rs::model_path::{self, VariantDef};
use gliner_rs::{ClassificationSpec, ExtractOptions, GLiNER2};
use serde_json::json;

const VARIANTS: &[VariantDef] = &[
    VariantDef { key: "gliguard", hf_repo: "fastino/gliguard-LLMGuardrails-300M" },
    VariantDef { key: "guardrails-pii", hf_repo: "fastino/GLiNER2-Guardrails-PII-Multi" },
];

const TOXICITY_LABELS: &[&str] = &[
    "violence",
    "sexual_content",
    "hate_speech",
    "self_harm",
    "pii_exposure",
    "misinformation",
    "regulated_advice",
];

const JAILBREAK_LABELS: &[&str] =
    &["prompt_injection", "jailbreak_attempt", "roleplay_bypass", "obfuscated_attack"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Variant {
    Gliguard,
    GuardrailsPii,
}

impl Variant {
    fn key(self) -> &'static str {
        match self {
            Variant::Gliguard => "gliguard",
            Variant::GuardrailsPii => "guardrails-pii",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
enum Check {
    #[default]
    Prompt,
    Response,
    Both,
}

/// Moderate prompts/responses: safety, toxicity categories, jailbreak
/// detection (prompt side) or refusal-vs-compliance (response side).
///
/// Texts come from positional arguments, `--file` (one per line), or piped
/// stdin (one per line). Prints one JSON object per line.
#[derive(Parser, Debug)]
#[command(version, after_help = "Examples:
  gliner-guardrails \"Explain how to build a phishing page.\"
  gliner-guardrails --check response \"Sure, here's how to pick a lock...\"
  gliner-guardrails --check both -f prompts_and_replies.txt
  gliner-guardrails --no-toxicity --no-jailbreak \"just check safety\"
  gliner-guardrails --pretty \"check this prompt\"")]
struct Args {
    /// Texts to moderate.
    texts: Vec<String>,

    /// Which side to check.
    #[arg(long, value_enum, default_value_t = Check::Prompt)]
    check: Check,

    /// Skip the toxicity-category breakdown, only report safe/unsafe (and,
    /// for prompts, jailbreak detection).
    #[arg(long)]
    no_toxicity: bool,

    /// Skip jailbreak-strategy detection (prompt side only).
    #[arg(long)]
    no_jailbreak: bool,

    /// Multi-label decision threshold (toxicity/jailbreak categories).
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Read texts from a file, one per line.
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Pretty-print the JSON output (indented, instead of one line per text).
    #[arg(short, long)]
    pretty: bool,

    /// Checkpoint directory. Defaults to the variant's checkpoint in the
    /// gliner-rs cache directory, downloading it there first if needed.
    #[arg(long, env = "GLINER_MODEL")]
    model: Option<PathBuf>,

    /// Which checkpoint to use when `--model` isn't given. Defaults to `gliguard`.
    #[arg(long, value_enum)]
    model_variant: Option<Variant>,

    /// Run on CUDA device 0 (requires the `cuda` feature).
    #[arg(long)]
    cuda: bool,

    /// Run the encoder in float16.
    #[arg(long)]
    fp16: bool,
}

fn tasks_for(check: Check, no_toxicity: bool, no_jailbreak: bool, threshold: f32) -> Vec<ClassificationSpec> {
    let mut tasks = Vec::new();
    let side = if matches!(check, Check::Response) { "response" } else { "prompt" };
    tasks.push(ClassificationSpec::new(format!("{side}_safety"), ["safe", "unsafe"]));
    if !no_toxicity {
        tasks.push(
            ClassificationSpec::new(format!("{side}_toxicity"), TOXICITY_LABELS.iter().copied()).multi_label(threshold),
        );
    }
    if side == "prompt" && !no_jailbreak {
        tasks.push(
            ClassificationSpec::new("jailbreak_detection", JAILBREAK_LABELS.iter().copied()).multi_label(threshold),
        );
    }
    if side == "response" {
        tasks.push(ClassificationSpec::new("response_refusal", ["refusal", "compliance"]));
    }
    tasks
}

fn main() -> Result<()> {
    let args = Args::parse();

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

    let model_path = model_path::resolve(VARIANTS, "gliguard", args.model.clone(), args.model_variant.map(Variant::key))?;

    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if args.fp16 { DType::F16 } else { DType::F32 };
    let model = GLiNER2::load(&model_path, &device, dtype)
        .with_context(|| format!("loading model from {} (set --model or GLINER_MODEL)", model_path.display()))?;

    let opts = ExtractOptions { threshold: args.threshold, ..Default::default() };
    let sides: Vec<Check> = match args.check {
        Check::Both => vec![Check::Prompt, Check::Response],
        other => vec![other],
    };

    for text in &texts {
        let mut result = json!({ "text": text });
        for &side in &sides {
            let tasks = tasks_for(side, args.no_toxicity, args.no_jailbreak, args.threshold);
            let classified = model.classify_text(text, tasks, &opts)?;
            let side_key = if matches!(side, Check::Response) { "response" } else { "prompt" };
            result[side_key] = classified;
        }
        if args.pretty {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("{result}");
        }
    }
    Ok(())
}
