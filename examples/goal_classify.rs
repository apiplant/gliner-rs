//! Ad-hoc test: can gliner-rs pick the right learning goal (out of ~900) for a sample exercise?
use anyhow::Result;
use candle_core::{DType, Device};
use gliner_rs::model_path::{self, VariantDef};
use gliner_rs::{ClassActivation, ClassificationSpec, GLiNER2};
use serde_json::Value;

const VARIANTS: &[VariantDef] = &[VariantDef { key: "multi", hf_repo: "fastino/gliner2.5-multi-v1" }];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let labels: Vec<&str> = data["labels"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let true_label = data["true_label"].as_str().unwrap();
    let exercise = data["exercise"].as_str().unwrap();

    println!("Loading model ({} labels)...", labels.len());
    let device = if std::env::var("GLINER_CUDA").is_ok() { Device::new_cuda(0)? } else { Device::Cpu };
    let model_path = model_path::resolve(VARIANTS, "multi", None, None)?;
    let dtype = if std::env::var("GLINER_FP16").is_ok() { DType::F16 } else { DType::F32 };
    let model = GLiNER2::load(&model_path, &device, dtype)?;

    // Chunk labels to stay within GPU memory (attention cost is quadratic in the
    // concatenated label sequence); sigmoid activation keeps scores comparable across chunks.
    let chunk_size: usize = std::env::var("GLINER_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let mut probs: Vec<(String, f32)> = Vec::new();
    for (i, chunk) in labels.chunks(chunk_size).enumerate() {
        println!("Classifying chunk {} ({} labels)...", i + 1, chunk.len());
        let mut spec = ClassificationSpec::new("learning_goal", chunk.iter().copied());
        spec.activation = ClassActivation::Sigmoid;
        let results = model.classification_probabilities(exercise, &[spec])?;
        probs.extend(results.into_iter().next().unwrap().1);
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nExercise: {exercise}");
    println!("True label: {true_label}\n");
    println!("Top 10 predictions:");
    for (label, p) in probs.iter().take(10) {
        let marker = if label == true_label { " <-- TRUE" } else { "" };
        println!("  {:.4}  {}{}", p, label, marker);
    }

    let rank = probs.iter().position(|(l, _)| l == true_label).unwrap();
    println!("\nTrue label rank: {} of {}", rank + 1, probs.len());
    Ok(())
}
