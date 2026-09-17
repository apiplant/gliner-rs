//! Long-document chunking and batching over independent texts. Run with a
//! checkpoint directory: `cargo run --release --example long_and_batch --
//! path/to/gliner2.5-multi-v1`.
use anyhow::Result;
use candle_core::{DType, Device};
use gliner_rs::{ChunkOptions, ExtractOptions, GLiNER2};

fn long_document() -> String {
    (0..40)
        .map(|i| format!("In report {i}, analyst Jane Smith of Goldman Sachs said Tesla shipped the Model Y from Berlin."))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<()> {
    let model_dir = std::env::args().nth(1).expect("usage: long_and_batch MODEL_DIR");
    let model = GLiNER2::load(&model_dir, &Device::Cpu, DType::F32)?;
    let opts = ExtractOptions { include_spans: true, include_confidence: true, ..Default::default() };

    // Longer than the encoder's context window: split into overlapping word
    // chunks, extract each, remap spans to offsets in the original document.
    let document = long_document();
    let result = model.extract_entities_long(&document, &["location"], &opts, ChunkOptions::default())?;
    let locations = result["entities"]["location"].as_array().unwrap();
    println!("found {} of 40 'Berlin' mentions across the whole document", locations.len());
    let first = &locations[0];
    let (start, end) = (first["start"].as_u64().unwrap() as usize, first["end"].as_u64().unwrap() as usize);
    println!("first span text {:?} == slice {:?}\n", first["text"], &document[start..end]);

    // Several independent texts against the same schema, in one encoder pass.
    let texts = ["Apple released a new iPhone.", "Nvidia unveiled a new GPU.", "Tesla opened a plant in Berlin."];
    let results = model.extract_entities_batch(&texts, &["company", "product", "location"], &opts)?;
    for (text, result) in texts.iter().zip(&results) {
        println!("{text:?} -> {result}");
    }

    Ok(())
}
