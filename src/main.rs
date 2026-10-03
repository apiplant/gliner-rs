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
    VariantDef { key: "decide", hf_repo: "fastino/GLiNER2.5-Decide" },
    VariantDef { key: "decide-multi", hf_repo: "fastino/GLiNER2.5-multi-Decide" },
    VariantDef { key: "decide-1b", hf_repo: "fastino/GLiNER2.5-Decide-1B" },
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
    #[arg(long, global = true, env = "GLINER_MODEL")]
    model: Option<PathBuf>,

    /// Which GLiNER2.5 checkpoint to use when `--model` isn't given. Defaults to `multi`.
    #[arg(long, global = true, value_enum)]
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
    #[arg(long, global = true)]
    char_split: bool,

    /// Run on CUDA device 0 (requires the `cuda` feature).
    #[arg(long, global = true)]
    cuda: bool,

    /// Run the encoder in float16.
    #[arg(long, global = true)]
    fp16: bool,

    /// CPU threads for matmul (candle/rayon). Defaults to min(8, available
    /// parallelism): the model's many small sequential matmuls oversubscribe
    /// and slow down past a handful of threads, so "all cores" is not the
    /// fastest setting. Ignored with --cuda.
    #[arg(long, global = true, env = "GLINER_THREADS")]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Serve schema extraction over HTTP (`POST /v1/extract`, `GET /health`). Uses the global
    /// `--model`/`--model-variant`/`--cuda`/`--fp16`/`--threads`/`--char-split` flags; runs until
    /// Ctrl-C/SIGTERM (an in-flight extraction finishes first).
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// Waiting slots on top of the one in-flight request; beyond that requests get 429.
        #[arg(long, env = "GLINER_MAX_QUEUED", default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
        max_queued: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Variant {
    Multi,
    Small,
    Base,
    Decide,
    DecideMulti,
    Decide1b,
}

impl Variant {
    fn key(self) -> &'static str {
        match self {
            Variant::Multi => "multi",
            Variant::Small => "small",
            Variant::Base => "base",
            Variant::Decide => "decide",
            Variant::DecideMulti => "decide-multi",
            Variant::Decide1b => "decide-1b",
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.cuda {
        // candle sizes its rayon pool from RAYON_NUM_THREADS (read once, lazily, on
        // first tensor op); with the `mkl` feature it instead calls into MKL, whose
        // own internal OpenMP pool reads OMP_NUM_THREADS/MKL_NUM_THREADS. Left unset,
        // both default to all logical CPUs, which oversubscribes badly on this
        // model's many small sequential matmuls; cap them unless already chosen.
        let threads = args.threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(8));
        if std::env::var_os("RAYON_NUM_THREADS").is_none() {
            std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
        }
        if std::env::var_os("OMP_NUM_THREADS").is_none() {
            std::env::set_var("OMP_NUM_THREADS", threads.to_string());
        }
        if std::env::var_os("MKL_NUM_THREADS").is_none() {
            std::env::set_var("MKL_NUM_THREADS", threads.to_string());
        }
    }
    let device = if args.cuda { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if args.fp16 { DType::F16 } else { DType::F32 };
    let model_path = model_path::resolve(VARIANTS, "multi", args.model.clone(), args.model_variant.map(Variant::key))?;

    #[cfg(feature = "server")]
    if let Some(Command::Serve { host, port, max_queued }) = args.command {
        // The reported model identity is what the user supplied: the explicit path, else the variant key.
        let model_name = args.model.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| args.model_variant.map_or("multi", Variant::key).to_string());
        let config = gliner_rs::server::ServerConfig { max_queued: max_queued as usize };
        return ntex::rt::System::new("gliner", ntex::rt::DefaultRuntime).block_on(serve(model_path, device, dtype, args.char_split, host, port, model_name, config));
    }

    #[cfg(not(feature = "server"))]
    if matches!(args.command, Some(Command::Serve { .. })) {
        anyhow::bail!("this gliner binary was built without the `server` feature; rebuild with `--features server`");
    }

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

#[cfg(feature = "server")]
#[allow(clippy::too_many_arguments)]
async fn serve(
    model_path: PathBuf,
    device: Device,
    dtype: DType,
    char_split: bool,
    host: String,
    port: u16,
    model_name: String,
    config: gliner_rs::server::ServerConfig,
) -> Result<()> {
    let started = Instant::now();
    eprintln!("loading checkpoint: {}", model_path.display());
    let mut model = GLiNER2::load(&model_path, &device, dtype)?;
    if char_split {
        model.set_word_splitter(WordSplitter::Char);
    }
    eprintln!("loaded in {:.2?}", started.elapsed());
    let handle = gliner_rs::server::start_server(&host, port, std::sync::Arc::new(model), model_name, config).await?;
    eprintln!("gliner serving '{}' on http://{}:{} (/v1/extract, /health)", handle.model_name, host, handle.port);
    // ntex handles Ctrl-C/SIGTERM: stop accepting, finish in-flight requests, exit.
    handle.wait().await;
    eprintln!("gliner stopped");
    Ok(())
}
