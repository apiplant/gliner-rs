//! Quick zero-shot text classification with GLiNER2, one-shot or interactive.

mod repl;
mod session;

use std::io::{BufRead, IsTerminal};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use gliner_rs::model_path::{self, VariantDef};
use gliner_rs::GLiNER2;

use session::{parse_example, Activation, Format, Session, TaskDef};

const VARIANTS: &[VariantDef] = &[
    VariantDef {
        key: "multi",
        default_relative: "models/gliner2.5-multi-v1",
        hf_repo: "fastino/gliner2.5-multi-v1",
    },
    VariantDef {
        key: "small",
        default_relative: "models/gliner2.5-small-v1",
        hf_repo: "fastino/gliner2.5-small-v1",
    },
    VariantDef {
        key: "base",
        default_relative: "models/gliner2.5-base-v1",
        hf_repo: "fastino/gliner2.5-base-v1",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
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

/// Zero-shot text classification with GLiNER2.
///
/// Texts come from positional arguments, `--file` (one per line), or piped
/// stdin (one per line). With no input in a terminal (or with `-i`), an
/// interactive shell starts; type `:help` there.
///
/// Labels may carry descriptions as `label:description`.
#[derive(Parser, Debug)]
#[command(version, after_help = "Examples:
  gliner-classify -l positive,negative,neutral \"I love this phone\"
  gliner-classify -m -l camera,battery,price \"Great camera, awful battery\"
  gliner-classify -t sentiment=positive,negative -t +topics=tech,sports \"...\"
  gliner-classify -l \"spam:Unsolicited ads,ham:Normal mail\" -f emails.txt --format tsv
  cat reviews.txt | gliner-classify -l positive,negative --all --format jsonl
  gliner-classify -i -l positive,negative        # interactive shell
  gliner-classify setup /mnt/ai/gliner     # save model paths in one go")]
struct Args {
    /// Texts to classify.
    texts: Vec<String>,

    /// Labels for a single task, comma-separated (`label` or `label:description`).
    #[arg(short, long)]
    labels: Option<String>,

    /// Name of the `--labels` task.
    #[arg(short = 'n', long, default_value = "label")]
    name: String,

    /// Make the `--labels` task multi-label.
    #[arg(short, long)]
    multi: bool,

    /// Extra task `name=label1,label2`; prefix the name with `+` for multi-label. Repeatable.
    #[arg(short, long = "task")]
    tasks: Vec<String>,

    /// Instruction/prompt appended to the task name (applies to every task).
    #[arg(short, long)]
    prompt: Option<String>,

    /// Few-shot example `input=>label`, used by tasks that have that label. Repeatable.
    #[arg(short, long = "example")]
    examples: Vec<String>,

    /// Multi-label decision threshold.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Score activation (auto: softmax for single-label, sigmoid for multi-label).
    #[arg(long, value_enum, default_value_t = Activation::Auto)]
    activation: Activation,

    /// Read texts from a file, one per line.
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Include the probability of every label.
    #[arg(short, long)]
    all: bool,

    /// Show only the K most probable labels (implies --all).
    #[arg(short = 'k', long)]
    top_k: Option<usize>,

    /// Start the interactive shell.
    #[arg(short, long)]
    interactive: bool,

    /// Checkpoint directory. Defaults to `./models/gliner2.5-<variant>-v1`,
    /// then a path saved from a previous interactive prompt.
    #[arg(long, env = "GLINER_MODEL")]
    model: Option<PathBuf>,

    /// Which GLiNER2.5 checkpoint to use when `--model` isn't given.
    /// Defaults to `multi`, or whatever was last used.
    #[arg(long, value_enum)]
    model_variant: Option<Variant>,

    /// Run on CUDA device 0 (requires the `cuda` feature).
    #[arg(long)]
    cuda: bool,

    /// Run the encoder in float16.
    #[arg(long)]
    fp16: bool,

    /// Print timing information to stderr.
    #[arg(short, long)]
    verbose: bool,

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

fn session_from(args: &Args) -> Result<Session> {
    let mut session = Session {
        prompt: args.prompt.clone(),
        threshold: args.threshold,
        activation: args.activation,
        all: args.all,
        top_k: args.top_k,
        format: args.format,
        ..Session::default()
    };
    if let Some(labels) = &args.labels {
        session.set_task(TaskDef { name: args.name.clone(), multi: args.multi, labels: labels.clone() });
    }
    for task in &args.tasks {
        session.set_task(TaskDef::parse(task)?);
    }
    session.examples = args.examples.iter().map(|e| parse_example(e)).collect::<Result<_>>()?;
    Ok(session)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(Command::Setup { base }) = args.command {
        return model_path::setup("gliner-classify", VARIANTS, &base);
    }
    let session = session_from(&args)?;

    let mut texts = args.texts.clone();
    if let Some(path) = &args.file {
        let content = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        texts.extend(content.lines().filter(|l| !l.trim().is_empty()).map(str::to_string));
    }
    let stdin_is_terminal = std::io::stdin().is_terminal();
    let interactive = args.interactive || (texts.is_empty() && stdin_is_terminal);
    if !interactive {
        // Fail on bad labels before paying for model loading.
        session.specs()?;
        if texts.is_empty() {
            for line in std::io::stdin().lock().lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    texts.push(line);
                }
            }
        }
    }

    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if args.fp16 { DType::F16 } else { DType::F32 };
    let model_path =
        model_path::resolve("gliner-classify", VARIANTS, "multi", args.model.clone(), args.model_variant.map(Variant::key))?;
    let started = Instant::now();
    if interactive {
        eprintln!("loading model from {} ...", model_path.display());
    }
    let model = GLiNER2::load(&model_path, &device, dtype)
        .with_context(|| format!("loading model from {} (set --model or GLINER_MODEL)", model_path.display()))?;
    if args.verbose || interactive {
        eprintln!("model loaded in {:.2?}", started.elapsed());
    }

    if interactive {
        return repl::run(&model, session, &texts);
    }
    let started = Instant::now();
    session.run(&model, &texts)?;
    if args.verbose {
        eprintln!("classified {} text(s) in {:.2?}", texts.len(), started.elapsed());
    }
    Ok(())
}
