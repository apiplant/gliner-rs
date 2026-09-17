//! Shared checkpoint-directory resolution for the CLI binaries.
//!
//! For a given set of named variants, resolution order for the chosen
//! variant is: an explicit `--model`/`GLINER_MODEL` path, if given, else
//! the variant's checkpoint in the gliner-rs cache directory
//! (`$XDG_CACHE_HOME/gliner-rs`, default `~/.cache/gliner-rs`), downloading
//! it there first if it isn't already present (see [`crate::download`]).

use std::path::PathBuf;

use anyhow::{Context, Result};

/// One checkpoint a CLI can target: a short key and the Hugging Face repo
/// it comes from.
pub struct VariantDef {
    pub key: &'static str,
    pub hf_repo: &'static str,
}

/// Resolves the checkpoint directory for one of `variants`. `explicit` is
/// whatever the user passed via `--model`/an env var, if anything;
/// `requested_variant` is the variant key explicitly requested, if any,
/// else `default_variant_key` is used.
pub fn resolve(
    variants: &[VariantDef],
    default_variant_key: &str,
    explicit: Option<PathBuf>,
    requested_variant: Option<&str>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path);
    }

    let variant_key = requested_variant.unwrap_or(default_variant_key);
    let variant = variants
        .iter()
        .find(|v| v.key == variant_key)
        .with_context(|| format!("unknown model variant {variant_key:?}"))?;

    crate::download::download_variant(variant)
}
