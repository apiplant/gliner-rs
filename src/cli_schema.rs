//! Builds a [`Schema`] from the same flag syntax the `gliner` CLI binary
//! accepts (`--entities`, `--relations`, `--json`, `--classify`), so the
//! wasm bindings can offer an identical "generic schema" mode.

use anyhow::{bail, Result};

use crate::schema::{ClassificationSpec, Schema, StructureMode, StructureSpec};

/// Mirrors `gliner`'s `--entities`/`--relations`/`--json`/`--classify` args.
#[derive(Debug, Clone, Default)]
pub struct CliSchemaArgs {
    /// `label` or `label:description`.
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    /// `name=field1::str,field2::[a|b],field3::list::description` (`;`-separated when fields contain commas).
    pub json: Vec<String>,
    pub legacy_structures: bool,
    /// `task=label1,label2`; prefix the task name with `+` for multi-label.
    pub classify: Vec<String>,
}

pub fn build_schema(args: &CliSchemaArgs) -> Result<Schema> {
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
        bail!("nothing to do: pass json, entities, relations and/or classify");
    }
    Ok(schema)
}
