use candle_core::{Device, DType};
use gliner_rs::{GLiNER2, ExtractOptions, Schema};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", "8");
    }
    let path = std::env::var("GLINER_MODEL").unwrap_or_else(|_| "/mnt/extra/ai/gliner/gliner2.5-multi-v1".into());
    let device = if std::env::var_os("GLINER_CUDA").is_some() { Device::new_cuda(0)? } else { Device::Cpu };
    let dtype = if std::env::var_os("GLINER_FP16").is_some() { DType::F16 } else { DType::F32 };
    let model = GLiNER2::load(&path, &device, dtype)?;
    let text = "Alice works for Acme in Paris. She met Bob last Thursday.";
    let opts = ExtractOptions::default();
    let schema = Schema::new()
        .entities(["person", "company", "location"])
        .relations(["works_for", "located_in"]);

    for _ in 0..3 {
        model.extract(text, &schema, &opts)?;
    }

    let n = 20;
    let t0 = Instant::now();
    for _ in 0..n {
        model.extract(text, &schema, &opts)?;
    }
    let elapsed = t0.elapsed();
    println!("rust avg: {:.2} ms", elapsed.as_secs_f64() * 1000.0 / n as f64);
    Ok(())
}
