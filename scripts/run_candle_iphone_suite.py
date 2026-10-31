#!/usr/bin/env python3
"""Run fresh-process F32 parity plus counterbalanced 5+30 trials on iPhone.

Requires a previously signed/installed Candle-enabled demo and the two pinned
checkpoints in .models. Outputs stay under .models; no external services used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
APP = "com.example.GLiNERDemo"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("device")
    p.add_argument("--variant", choices=["small", "multi"], required=True)
    p.add_argument("--reuse-model", action="store_true", help="Reuse existing device checkpoint; each run checks its pinned manifest")
    p.add_argument("--reuse-reference", action="store_true", help="Reuse previously generated CPU references (avoids host inference during isolated benchmarks)")
    p.add_argument("--memory-trace", action="store_true")
    p.add_argument("--load-autoreleasepool", action="store_true")
    p.add_argument("--metal-loader", choices=["stream", "mmap"], default="stream")
    p.add_argument("--metal-original-attention", action="store_true")
    p.add_argument("--tag", default="", help="Distinct artifact tag to preserve the baseline")
    p.add_argument("--reference-binary", type=Path, default=ROOT / "target/release/examples/apple_reference", help="Built CPU reference example")
    args = p.parse_args()
    model = ROOT / ".models" / f"gliner2.5-{args.variant}-v1"
    if not (model / "checkpoint.json").is_file():
        raise SystemExit(f"Missing verified checkpoint in {model}")
    if not args.reference_binary.is_file():
        raise SystemExit(f"Missing CPU reference executable: {args.reference_binary}; build apple_reference first")
    out = ROOT / ".models"
    references = out / "iphone-rust-references"
    references.mkdir(exist_ok=True)
    parity = json.loads((ROOT / "scripts/apple_cases.json").read_text()) + json.loads((ROOT / "scripts/apple_advanced_cases.json").read_text())
    requests = json.loads((ROOT / "docs/benchmarks/rust-vs-mlx.json").read_text())["requests"]
    env = dict(os.environ, RAYON_NUM_THREADS="4")
    def reference(cases, input_path, output_path):
        input_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf8")
        with output_path.open("w", encoding="utf8") as output:
            subprocess.run([str(args.reference_binary), str(model), str(input_path)], cwd=ROOT, env=env, stdout=output, check=True)
    parity_ref = references / f"{args.variant}-parity-cpu-reference.jsonl"
    if args.reuse_reference:
        required = [parity_ref] + [references / f"{args.variant}-{request['name']}-cpu-reference.jsonl" for request in requests]
        if not all(path.is_file() for path in required): raise SystemExit("Missing reusable CPU reference files")
    else:
        reference(parity, references / f"{args.variant}-parity-cases.json", parity_ref)
        for request in requests:
            name = request["name"]
            reference([request], references / f"{args.variant}-{name}-case.json", references / f"{args.variant}-{name}-cpu-reference.jsonl")
    if not args.reuse_model:
        subprocess.run(["xcrun", "devicectl", "device", "copy", "to", "--device", args.device,
                        "--source", str(model), "--destination", "Documents/self-test-model", "--domain-type", "appDataContainer",
                        "--domain-identifier", APP, "--timeout", "240"], cwd=ROOT, check=True)
    extra = (["--memory-trace"] if args.memory_trace else []) + (["--load-autoreleasepool"] if args.load_autoreleasepool else []) + (["--tag", args.tag] if args.tag else []) + ["--metal-loader", args.metal_loader] + (["--metal-original-attention"] if args.metal_original_attention else [])
    for backend in ("mlx", "candle"):
        subprocess.run([sys.executable, "scripts/verify_candle_iphone.py", args.device, "--variant", args.variant,
                        "--backend", backend, "--mode", "parity", "--trial", "1", *extra], cwd=ROOT, check=True)
    for trial, order in enumerate((("mlx", "candle"), ("candle", "mlx"), ("mlx", "candle")), 1):
        for backend in order:
            print(f"{args.variant}: trial {trial}, {backend}", flush=True)
            subprocess.run([sys.executable, "scripts/verify_candle_iphone.py", args.device, "--variant", args.variant,
                            "--backend", backend, "--mode", "benchmark", "--trial", str(trial), *extra], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/analyze_candle_iphone.py", "--variant", args.variant,
                    "--parity-reference", str(parity_ref), "--benchmark-reference-dir", str(references),
                    "--expect-metal-loader", args.metal_loader, "--expect-original-attention", "yes" if args.metal_original_attention else "no", "--expect-load-pool", "on" if args.load_autoreleasepool else "off",
                    *(["--tag", args.tag] if args.tag else [])], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
