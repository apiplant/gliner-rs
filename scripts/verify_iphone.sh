#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [ "$#" -lt 1 ]; then echo 'Usage: bash scripts/verify_iphone.sh DEVICE_ID [small|base|multi] [--trace]' >&2; exit 2; fi
task_device="$1"
task_variant="${2:-small}"
case "$task_variant" in small|base|multi) ;; *) echo 'Choose small, base or multi' >&2; exit 2;; esac
task_prefix=".models/iphone-${task_variant}-verified"
task_model=".models/gliner2.5-${task_variant}-v1"
task_run_id="$(python3 -c 'import uuid; print(uuid.uuid4())')"
echo 'L’app GLiNERDemo deve essere già installata e firmata sul dispositivo.'
python3 scripts/download_apple_model.py --variant "$task_variant"
python3 - <<'PY'
import json
from pathlib import Path
cases = json.loads(Path('scripts/apple_cases.json').read_text()) + json.loads(Path('scripts/apple_advanced_cases.json').read_text())
Path('.models/apple-iphone-cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n')
PY
cargo +1.99.1 build --release --locked --no-default-features --example apple_reference
RAYON_NUM_THREADS=4 target/release/examples/apple_reference "$task_model" .models/apple-iphone-cases.json > "${task_prefix}-rust.jsonl"
echo 'Trasferimento dei pesi e degli esempi nell’app di prova'
xcrun devicectl device copy to --device "$task_device" --source "$task_model" --destination Documents/self-test-model \
  --domain-type appDataContainer --domain-identifier com.example.GLiNERDemo --timeout 120
xcrun devicectl device copy to --device "$task_device" --source .models/apple-iphone-cases.json --destination Documents/apple_cases.json \
  --domain-type appDataContainer --domain-identifier com.example.GLiNERDemo --timeout 30
task_extra=()
if [ "${3:-}" = '--trace' ]; then task_extra+=(--self-test-trace); fi
xcrun devicectl device process launch --device "$task_device" --terminate-existing com.example.GLiNERDemo -- \
  --self-test --self-test-precision float32 --self-test-id "$task_run_id" "${task_extra[@]}"
echo 'Attendo il risultato della nuova esecuzione sul dispositivo'
task_ready=0
rm -f "${task_prefix}-summary.json" "${task_prefix}-summary.json.part"
for task_attempt in $(seq 1 120); do
    if xcrun devicectl device copy from --quiet --device "$task_device" --source Documents/apple-self-test-summary.json \
        --destination "${task_prefix}-summary.json.part" --domain-type appDataContainer --domain-identifier com.example.GLiNERDemo --timeout 10 2>/dev/null; then
        if python3 - "${task_prefix}-summary.json.part" "$task_run_id" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
    valid = data.get('runID') == sys.argv[2]
except (OSError, ValueError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
        then
            mv "${task_prefix}-summary.json.part" "${task_prefix}-summary.json"
            task_ready=1
            break
        fi
    fi
    sleep 2
done
if [ "$task_ready" != 1 ]; then echo 'Il dispositivo non ha prodotto il riepilogo; controllare l’app e il suo stato.' >&2; exit 1; fi
python3 - "$task_prefix" <<'PY'
import json, sys
data = json.load(open(sys.argv[1] + '-summary.json'))
print(json.dumps(data, indent=2))
if data.get('status') != 'passed':
    raise SystemExit('L’esecuzione sull’iPhone è fallita')
PY
xcrun devicectl device copy from --device "$task_device" --source Documents/apple-self-test-results.jsonl \
  --destination "${task_prefix}-results.jsonl" --domain-type appDataContainer --domain-identifier com.example.GLiNERDemo --timeout 30
python3 scripts/compare_apple_results.py "${task_prefix}-rust.jsonl" "${task_prefix}-results.jsonl" --allow-near-ties | tee "${task_prefix}-comparison.json"
if [ "${3:-}" = '--trace' ]; then
    RAYON_NUM_THREADS=4 target/release/examples/apple_reference "$task_model" .models/apple-iphone-cases.json --trace > "${task_prefix}-trace-rust.jsonl"
    xcrun devicectl device copy from --device "$task_device" --source Documents/apple-self-test-encoder.jsonl \
      --destination "${task_prefix}-trace-device.jsonl" --domain-type appDataContainer --domain-identifier com.example.GLiNERDemo --timeout 60
    python3 scripts/compare_apple_trace.py "${task_prefix}-trace-rust.jsonl" "${task_prefix}-trace-device.jsonl" | tee "${task_prefix}-trace-comparison.json"
fi
