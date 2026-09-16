//! Shared checkpoint-directory resolution for the CLI binaries.
//!
//! For a given app (`gliner-classify`, `gliner-pii`, `gliner-guardrails`, ...)
//! and a set of named variants, resolution order for the chosen variant is:
//! `./models/<variant's default dir>` relative to the current directory, then
//! a path saved in `$XDG_CONFIG_HOME/<app>/config.json` (default `~/.config`),
//! then (if attached to a terminal) an interactive prompt whose answer is
//! saved there for next time. A variant chosen explicitly (not defaulted) is
//! also saved as the app's new default variant.
//!
//! `setup` configures the saved paths in one go: given a base directory that
//! contains checkpoint subdirectories (e.g. `/mnt/ai/gliner`), every
//! known variant found there is saved to the app config.

use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// One checkpoint a CLI can target: a short key, where to look for it by
/// default, and the Hugging Face repo it comes from (shown in prompts).
pub struct VariantDef {
    pub key: &'static str,
    pub default_relative: &'static str,
    pub hf_repo: &'static str,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Config {
    #[serde(default)]
    default_variant: Option<String>,
    #[serde(default)]
    paths: BTreeMap<String, PathBuf>,
}

/// `$XDG_CONFIG_HOME/<app>/config.json`, defaulting to `~/.config`.
fn config_path(app: &str) -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join(app).join("config.json"))
}

fn load_config(app: &str) -> Config {
    config_path(app)
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|contents| serde_json::from_str(&contents).ok())
        .unwrap_or_default()
}

fn save_config(app: &str, config: &Config) -> Result<()> {
    let path = config_path(app).context("no config directory available (set $HOME or $XDG_CONFIG_HOME)")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(&path, json).with_context(|| format!("writing {}", path.display()))
}

fn looks_like_checkpoint(dir: &Path) -> bool {
    dir.is_dir() && (dir.join("config.json").is_file() || dir.join("tokenizer.json").is_file())
}

/// The directory name a variant's checkpoint is expected to have
/// (the last component of its `default_relative` path).
fn expected_dir_name(variant: &VariantDef) -> String {
    Path::new(variant.default_relative)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(variant.key)
        .to_string()
}

fn prompt_for_path(variant: &VariantDef) -> Result<PathBuf> {
    loop {
        eprint!("Checkpoint directory for {} (e.g. a {} download): ", variant.key, variant.hf_repo);
        std::io::stderr().flush().ok();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            bail!("no model path given");
        }
        let entered = line.trim();
        if entered.is_empty() {
            continue;
        }
        let path = PathBuf::from(entered);
        if !path.is_dir() {
            eprintln!("{} is not a directory, try again", path.display());
            continue;
        }
        return Ok(path);
    }
}

/// Resolves the checkpoint directory for `app`. `explicit` is whatever the
/// user passed via `--model`/an env var, if anything; `requested_variant` is
/// the variant key explicitly requested (as opposed to defaulted), if any.
pub fn resolve(
    app: &str,
    variants: &[VariantDef],
    default_variant_key: &str,
    explicit: Option<PathBuf>,
    requested_variant: Option<&str>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }

    let mut config = load_config(app);
    let variant_key = requested_variant
        .or(config.default_variant.as_deref())
        .unwrap_or(default_variant_key)
        .to_string();
    let variant = variants
        .iter()
        .find(|v| v.key == variant_key)
        .with_context(|| format!("unknown model variant {variant_key:?}"))?;

    let record_default = |config: &mut Config| -> Result<()> {
        if requested_variant.is_some() && config.default_variant.as_deref() != Some(variant.key) {
            config.default_variant = Some(variant.key.to_string());
            save_config(app, config)?;
        }
        Ok(())
    };

    let relative = PathBuf::from(variant.default_relative);
    if looks_like_checkpoint(&relative) {
        record_default(&mut config)?;
        return Ok(relative);
    }

    if let Some(saved) = config.paths.get(variant.key) {
        if looks_like_checkpoint(saved) {
            let saved = saved.clone();
            record_default(&mut config)?;
            return Ok(saved);
        }
    }

    let missing = format!(
        "no {} model found: ./{} is empty or missing and no path is configured yet",
        variant.key, variant.default_relative
    );

    if !std::io::stdin().is_terminal() {
        bail!(
            "{}; pass --model, set GLINER_MODEL, or run `{} setup <base_models_path>` (e.g. setup /mnt/ai/gliner) to configure one",
            missing, app
        );
    }

    eprintln!("{}.", missing);
    eprintln!("To fix, either download a checkpoint from {} into ./models/,", variant.hf_repo);
    eprintln!("run `{} setup <base_models_path>` (e.g. setup /mnt/ai/gliner),", app);
    eprintln!("or enter a path to an existing checkpoint directory below.");
    let path = prompt_for_path(variant)?;
    config.paths.insert(variant.key.to_string(), path.clone());
    config.default_variant = Some(variant.key.to_string());
    save_config(app, &config)?;
    Ok(path)
}

/// Configures the app's saved model paths in one go from a base models directory.
///
/// `base` is a directory containing checkpoint subdirectories named like the
/// variants' default directories (e.g. `/mnt/ai/gliner/gliner2.5-multi-v1`);
/// it may hold several models at once. Every known variant found is saved in the
/// app config, so any of them can be used later via `--model-variant`; the
/// preferred one also becomes the default variant.
pub fn setup(app: &str, variants: &[VariantDef], base: &Path) -> Result<()> {
    if !base.is_dir() {
        bail!("{} is not a directory", base.display());
    }
    let mut config = load_config(app);
    let mut saved: Vec<(String, PathBuf)> = Vec::new();

    // Look for checkpoint subdirectories under `base`.
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !looks_like_checkpoint(&path) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(variant) = variants.iter().find(|v| expected_dir_name(v) == name) {
                if !saved.iter().any(|(key, _)| key == variant.key) {
                    saved.push((variant.key.to_string(), path));
                }
            }
        }
    }

    if saved.is_empty() {
        let entries: Vec<String> = std::fs::read_dir(base)
            .with_context(|| format!("reading {}", base.display()))?
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect();
        bail!(
            "no known GLiNER checkpoints found under {} (looked for: {}); found instead: {}",
            base.display(),
            variants.iter().map(expected_dir_name).collect::<Vec<_>>().join(", "),
            if entries.is_empty() { "<nothing>".into() } else { entries.join(", ") }
        );
    }

    // Show variants in the app's preference order (order of `variants`).
    saved.sort_by_key(|(key, _)| variants.iter().position(|v| v.key == key.as_str()).unwrap_or(usize::MAX));
    for (key, path) in &saved {
        config.paths.insert(key.clone(), path.clone());
    }
    if config.default_variant.is_none() {
        config.default_variant = Some(saved[0].0.clone());
    }
    save_config(app, &config)?;
    for (key, path) in &saved {
        println!("{key}: {}", path.display());
    }
    println!(
        "{} model path(s) saved to {}",
        saved.len(),
        config_path(app).map(|p| p.display().to_string()).unwrap_or_default()
    );
    Ok(())
}
