#!/usr/bin/env python3
"""Run the installed demo's F32 parity or 5+30 benchmark on a paired iPhone.

Model files must already be present as Documents/self-test-model; use verify_iphone.sh
or devicectl to transfer the checkpoint. Files remain local to the device.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
APP = "com.example.GLiNERDemo"

def run(*args, quiet=False, check=True):
    return subprocess.run(["xcrun", "devicectl", *args], cwd=ROOT, text=True,
                          stdout=subprocess.DEVNULL if quiet else None,
                          stderr=subprocess.DEVNULL if quiet else None, check=check)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("device")
    p.add_argument("--backend", choices=["mlx", "candle"], required=True)
    p.add_argument("--variant", choices=["small", "multi"], required=True)
    p.add_argument("--mode", choices=["parity", "benchmark"], default="parity")
    p.add_argument("--trial", type=int, default=1)
    args = p.parse_args()
    out = ROOT / ".models"
    out.mkdir(exist_ok=True)
    if args.mode == "parity":
        cases = json.loads((ROOT / "scripts/apple_cases.json").read_text()) + json.loads((ROOT / "scripts/apple_advanced_cases.json").read_text())
        name = "apple_cases.json"
    else:
        cases = json.loads((ROOT / "docs/benchmarks/rust-vs-mlx.json").read_text())["requests"]
        name = "apple-benchmark-cases.json"
    source = out / ("iphone-" + name)
    source.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf8")
    common = ["--device", args.device, "--domain-type", "appDataContainer", "--domain-identifier", APP]
    manifest_file = out / "iphone-candle-input-checkpoint.json"
    manifest_file.unlink(missing_ok=True)
    run("device", "copy", "from", *common, "--source", "Documents/self-test-model/checkpoint.json", "--destination", str(manifest_file), "--timeout", "30")
    manifest = json.loads(manifest_file.read_text())
    revisions = {"small": "df5910e44bc4ffdb0d95399a83b0ca4516349aa5", "multi": "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3"}
    if manifest.get("repo") != f"fastino/gliner2.5-{args.variant}-v1" or manifest.get("revision") != revisions[args.variant]:
        raise SystemExit("Device checkpoint does not match requested variant and pinned revision")
    run("device", "copy", "to", *common, "--source", str(source), "--destination", "Documents/" + name, "--timeout", "30")
    run_id = str(uuid.uuid4())
    extra = ["--benchmark"] if args.mode == "benchmark" else []
    run("device", "process", "launch", "--device", args.device, "--terminate-existing", APP, "--", "--self-test", "--self-test-precision", "float32", "--self-test-backend", args.backend, "--self-test-id", run_id, *extra)
    prefix = out / f"iphone-{args.variant}-{args.backend}-{args.mode}-trial{args.trial}"
    temporary = prefix.with_suffix(".summary.part")
    summary = None
    for _ in range(600):
        run("device", "copy", "from", *common, "--source", "Documents/apple-self-test-summary.json", "--destination", str(temporary), "--timeout", "10", quiet=True, check=False)
        try:
            candidate = json.loads(temporary.read_text())
            if candidate.get("runID") == run_id:
                summary = candidate
                break
        except (OSError, ValueError):
            pass
        time.sleep(2)
    if summary is None:
        raise SystemExit(f"Timed out waiting for run {run_id}")
    prefix.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.unlink(missing_ok=True)
    if summary.get("status") != "passed":
        raise SystemExit("Device run failed: " + str(summary.get("error")))
    if summary.get("backend") != args.backend or summary.get("precision") != "float32":
        raise SystemExit("Device summary backend or precision does not match request")
    expected = 28 if args.mode == "parity" else 3
    expected_lines = expected * (35 if args.mode == "benchmark" else 1)
    if summary.get("caseCount") != expected:
        raise SystemExit(f"Unexpected case count: {summary.get('caseCount')}")
    output = prefix.with_suffix(".results.jsonl")
    run("device", "copy", "from", *common, "--source", "Documents/apple-self-test-results.jsonl", "--destination", str(output), "--timeout", "60")
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    if len(lines) != expected_lines:
        raise SystemExit(f"Unexpected result count: {len(lines)} != {expected_lines}")
    if args.mode == "benchmark":
        expected_pairs = {(case, iteration) for case in range(3) for iteration in range(35)}
        actual_pairs = {(row.get("caseIndex"), row.get("iteration")) for row in lines}
        if actual_pairs != expected_pairs or any(row.get("runID") != run_id or row.get("backend") != args.backend or row.get("precision") != "float32" or row.get("warmup") != (row.get("iteration", -1) < 5) for row in lines):
            raise SystemExit("Invalid benchmark metadata or run ID")
    print(json.dumps({"runID": run_id, "backend": args.backend, "variant": args.variant, "mode": args.mode, "caseCount": expected, "resultCount": len(lines), "summary": str(prefix.with_suffix('.summary.json')), "results": str(output)}, indent=2))

if __name__ == "__main__":
    main()
