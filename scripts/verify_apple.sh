#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_variant="${1:-small}"
case "$task_variant" in small|base|multi) ;; *) echo 'Choose small, base or multi' >&2; exit 2;; esac
task_model=".models/gliner2.5-${task_variant}-v1"
task_prefix=".models/${task_variant}"
task_cases="scripts/apple_cases.json"
echo '1/7 — Download e verifica dei file del modello'
python3 scripts/download_apple_model.py --variant "$task_variant"
echo '2/7 — Compilazione e test automatici Swift su Mac'
bash scripts/build_apple.sh test
bash scripts/build_apple.sh macos
task_binary="$task_root/.derivedData/Build/Products/${CONFIGURATION:-Release}/gliner-mlx"
echo '3/7 — Compilazione del riferimento Rust'
cargo +1.99.1 build --release --locked --no-default-features --example apple_reference
task_reference="$task_root/target/release/examples/apple_reference"
echo '4/7 — Confronto esatto degli input'
RAYON_NUM_THREADS=4 "$task_reference" "$task_model" "$task_cases" --inputs-only > "${task_prefix}-inputs-rust.jsonl"
"$task_binary" --model "$task_model" --cases "$task_cases" --inputs-only > "${task_prefix}-inputs-swift.jsonl"
python3 scripts/compare.py "${task_prefix}-inputs-rust.jsonl" "${task_prefix}-inputs-swift.jsonl" 0
echo '5/7 — Confronto di ogni stato numerico dell’encoder F32'
RAYON_NUM_THREADS=4 "$task_reference" "$task_model" "$task_cases" --trace > "${task_prefix}-trace-rust.jsonl"
"$task_binary" --model "$task_model" --cases "$task_cases" --trace > "${task_prefix}-trace-swift.jsonl"
python3 scripts/compare_apple_trace.py "${task_prefix}-trace-rust.jsonl" "${task_prefix}-trace-swift.jsonl"
echo '6/7 — Confronto dei risultati F32 (tolleranza 0.0001; riordini quasi pari segnalati)'
RAYON_NUM_THREADS=4 "$task_reference" "$task_model" "$task_cases" > "${task_prefix}-rust-f32.jsonl"
"$task_binary" --model "$task_model" --cases "$task_cases" > "${task_prefix}-mlx-f32.jsonl"
python3 scripts/compare_apple_results.py "${task_prefix}-rust-f32.jsonl" "${task_prefix}-mlx-f32.jsonl" --allow-near-ties
echo '7/7 — Strutture latent/anchorless, ancore esplicite e classificazione con esempi'
RAYON_NUM_THREADS=4 "$task_reference" "$task_model" scripts/apple_advanced_cases.json > "${task_prefix}-advanced-rust.jsonl"
"$task_binary" --model "$task_model" --cases scripts/apple_advanced_cases.json > "${task_prefix}-advanced-mlx.jsonl"
python3 scripts/compare_apple_results.py "${task_prefix}-advanced-rust.jsonl" "${task_prefix}-advanced-mlx.jsonl" --allow-near-ties
if [ "${2:-}" = '--check-fp16' ]; then
    echo 'Controllo sperimentale F16: questo confronto può fallire perché cambiano le risposte'
    "$task_binary" --model "$task_model" --cases "$task_cases" --fp16 > "${task_prefix}-mlx-f16.jsonl"
    python3 scripts/compare_apple_results.py "${task_prefix}-rust-f32.jsonl" "${task_prefix}-mlx-f16.jsonl" \
        --tolerance 0.02 --allow-near-ties --tie-tolerance 0.02
fi
echo "Verifica completata. Risultati salvati in .models/${task_variant}-*.jsonl"
