//! Rust side of `scripts/parity.py`: prints one JSON line per case.

use anyhow::Result;
use candle_core::{DType, Device};
use gliner_rs::{AttributeGroup, ClassificationSpec, ExtractOptions, GLiNER2, Schema, StructureMode, StructureSpec, WordSplitter};
use serde_json::Value;

fn long_text() -> String {
    (0..40)
        .map(|i| format!("In report {i}, analyst Jane Smith of Goldman Sachs said Tesla shipped the Model Y from Berlin while Elon Musk visited Austin."))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = if std::env::var("GLINER_CUDA").is_ok() { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if std::env::var("GLINER_FP16").is_ok() { DType::F16 } else { DType::F32 };
    let mut model = GLiNER2::load(&args[1], &device, dtype)?;
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    for case in cases {
        let text = if case["long"].as_bool().unwrap_or(false) { long_text() } else { case["text"].as_str().unwrap().to_string() };
        model.set_word_splitter(if case["char_split"].as_bool().unwrap_or(false) { WordSplitter::Char } else { WordSplitter::Whitespace });

        let mut schema = Schema::new();
        if let Value::Object(structures) = &case["json"] {
            for (name, fields) in structures {
                let mut spec = StructureSpec::parse(name, fields.as_array().unwrap().iter().map(|v| v.as_str().unwrap()));
                if case["legacy"].as_bool().unwrap_or(false) {
                    spec.mode = StructureMode::Legacy;
                }
                schema = schema.structure(spec);
            }
        }
        match &case["entities"] {
            Value::Array(names) => schema = schema.entities(names.iter().map(|v| v.as_str().unwrap())),
            Value::Object(map) => {
                for (name, desc) in map {
                    schema = schema.entity_with_description(name, desc.as_str().unwrap());
                }
            }
            _ => {}
        }
        if let Value::Array(names) = &case["relations"] {
            schema = schema.relations(names.iter().map(|v| v.as_str().unwrap()));
        }
        if let Value::Object(groups) = &case["entity_attributes"] {
            let groups = groups.iter().map(|(name, g)| {
                let mut group = AttributeGroup::new(g["labels"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()));
                if let Some(threshold) = g["threshold"].as_f64() {
                    group.threshold = threshold as f32;
                }
                group.multi_label = g["multi_label"].as_bool().unwrap_or(false);
                if let Value::Array(names) = &g["applies_to"] {
                    group = group.applies_to(names.iter().map(|v| v.as_str().unwrap()));
                }
                if g["qualify_labels"].as_bool().unwrap_or(false) {
                    group = group.qualify_labels();
                }
                (name.clone(), group)
            });
            schema = schema.entity_attributes(groups);
        }
        if let Value::Array(tasks) = &case["classifications"] {
            for task in tasks {
                let labels = task["labels"].as_array().unwrap().iter().map(|v| v.as_str().unwrap());
                let mut spec = ClassificationSpec::new(task["task"].as_str().unwrap(), labels);
                spec.multi_label = task["multi_label"].as_bool().unwrap_or(false);
                spec.cls_threshold = task["cls_threshold"].as_f64().unwrap_or(0.5) as f32;
                schema = schema.classification(spec);
            }
        }
        let opts = ExtractOptions {
            threshold: case["threshold"].as_f64().unwrap_or(0.5) as f32,
            include_confidence: true,
            include_spans: true,
            ..Default::default()
        };
        println!("{}", model.extract(&text, &schema, &opts)?);
    }
    Ok(())
}
