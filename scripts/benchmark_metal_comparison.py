#!/usr/bin/env python3
"""Paired Release Candle Metal / MLX Metal measurements on identical checkpoints.

Run only after parity checks, while other local GPU jobs are stopped. Each
trial/request/backend starts a fresh process. Backend order alternates across
pairs. Every output (warmups included) is checked against a fresh CPU reference;
array order stays strict. Raw stdout/stderr, per-process RSS, every timing and
per-trial aggregates are retained. This is a backend comparison, not a language
benchmark. Filesystem and shader caches are not flushed.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

try:
    from scripts import benchmark_apple as apple
except ModuleNotFoundError:
    import benchmark_apple as apple

ROOT = Path(__file__).resolve().parents[1]
RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.MULTILINE)


def stats(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timings must be finite and positive")
    ordered = sorted(values)
    return {"median_ms": statistics.median(values),
            "p95_ms": ordered[math.ceil(len(values) * .95) - 1],
            "min_ms": min(values), "max_ms": max(values), "samples": len(values)}


def pair_order(index):
    return ["candle_metal", "mlx_metal"] if index % 2 == 0 else ["mlx_metal", "candle_metal"]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_process(command, environment, artifacts, stem, timeout):
    # /usr/bin/time reports peak *process* RSS, not a GPU-only allocation budget.
    process = subprocess.run(["/usr/bin/time", "-l", *command], env=environment,
                             capture_output=True, text=True, timeout=timeout)
    stdout_path, stderr_path = artifacts / (stem + ".jsonl"), artifacts / (stem + ".stderr.log")
    stdout_path.write_text(process.stdout)
    stderr_path.write_text(process.stderr)
    if process.returncode:
        raise ValueError(f"Exit {process.returncode}; see {stderr_path}: {process.stderr[-1500:]}")
    rss = RSS_RE.findall(process.stderr)
    return process, {"command": command, "stdout_artifact": str(stdout_path),
                     "stderr_artifact": str(stderr_path),
                     "peak_process_rss_bytes": int(rss[-1]) if rss else None}


def summarize(report):
    rows = []
    for model in report["models"]:
        for request in report["requests"]:
            selected = [x for x in report["runs"] if x["model"] == model["variant"] and x["request"] == request["name"]]
            row = {"model": model["variant"], "request": request["name"], "backends": {}}
            for backend in ("candle_metal", "mlx_metal"):
                runs = [x for x in selected if x["backend"] == backend]
                if not runs:
                    continue
                values = [v for x in runs for v in x["measured_run_ms"]]
                medians = [x["warm_median_ms"] for x in runs]
                row["backends"][backend] = {
                    "pooled": stats(values), "trial_medians_ms": medians,
                    "median_of_trial_medians_ms": statistics.median(medians),
                    "trial_median_range_ms": [min(medians), max(medians)],
                    "load_ms_by_trial": [x["load_ms"] for x in runs],
                    "first_request_ms_by_trial": [x["cold_first_ms"] for x in runs],
                    "peak_process_rss_bytes_by_trial": [x["peak_process_rss_bytes"] for x in runs],
                    "all_outputs_correct": all(x["quality"]["runs_within_reference_tolerance"] == x["repetitions"] for x in runs),
                    "max_confidence_delta": max(x["quality"]["max_confidence_delta"] for x in runs),
                }
            if len(row["backends"]) == 2:
                candle = row["backends"]["candle_metal"]["median_of_trial_medians_ms"]
                mlx = row["backends"]["mlx_metal"]["median_of_trial_medians_ms"]
                row["candle_to_mlx_latency_ratio"] = candle / mlx
                row["mlx_latency_reduction_percent"] = (candle - mlx) / candle * 100
            rows.append(row)
    return rows


def direct_gpu_comparisons(report, tolerance):
    pairs = {}
    for run in report["runs"]:
        pairs.setdefault((run["model"], run["request"], run["trial"]), {})[run["backend"]] = run
    results = []
    for (model, request, trial), runs in pairs.items():
        if len(runs) != 2:
            continue
        outputs = {}
        for backend, run in runs.items():
            outputs[backend] = [json.loads(line) for line in Path(run["stdout_artifact"]).read_text().splitlines() if line.strip()]
        comparison = apple.compare_values(outputs["candle_metal"], outputs["mlx_metal"], tolerance=tolerance)
        results.append({"model": model, "request": request, "trial": trial,
                        "runs_per_backend": len(outputs["candle_metal"]), "strict": not comparison.errors,
                        "max_confidence_delta": comparison.max_confidence_delta,
                        "errors": comparison.errors[:20]})
    return results


def benchmark(args):
    output = Path(args.output).resolve()
    artifacts = output.parent / (output.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    rust, swift = Path(args.rust_binary).resolve(), Path(args.mlx_binary).resolve()
    for binary in (rust, swift):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Executable not found: {binary}")
    requests = apple.load_requests(args.requests)
    report = {
        "schema_version": 1, "status": "running", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": apple.machine_info(), "comparison": "Candle Metal GPU F32 versus MLX Metal GPU F32",
        "method": {"trials": args.trials, "warmup_runs_per_process": args.warmup,
                   "measured_runs_per_process": args.runs, "fresh_process_per_trial_request_backend": True,
                   "backend_order": "Alternating AB/BA across trial/request pairs",
                   "timing_scope": "Preparation/tokenization, inference, output formatting/JSON and synchronous stdout write; model loading separate",
                   "timing_clock": "Rust Instant; Swift DispatchTime uptime (both monotonic)",
                   "aggregate": "Median of independent process medians; pooled p95 also reported",
                   "p95_method": "nearest rank", "confidence_tolerance": args.tolerance,
                   "array_order": "strict", "filesystem_cache_flushed": False, "shader_cache_flushed": False,
                   "rss_method": "/usr/bin/time -l maximum resident set size (bytes on macOS); entire process peak",
                   "mlx_configuration": {"dtype": "float32", "tf32": False, "cache": "auto", "clear": False},
                   "candle_configuration": {"dtype": "F32", "device": "Metal", "rayon_threads": 4},
                   "limits": "Local sequential trials; no thermal/energy instrumentation or statistical confidence intervals"},
        "builds": {"candle": {"binary": str(rust), "sha256": apple.file_sha(rust)},
                   "mlx": {"binary": str(swift), "sha256": apple.file_sha(swift)}},
        "source_sha256": {p: apple.file_sha(ROOT / p) for p in
                          ("Cargo.toml", "Cargo.lock", "examples/apple_reference.rs", "src/deberta.rs",
                           "src/model.rs", "swift/Sources/GLiNERCLI/main.swift")},
        "models": [], "requests": requests, "runs": [], "comparisons": []}
    environment = dict(os.environ, MLX_ENABLE_TF32="0", RAYON_NUM_THREADS="4")
    repetitions = args.warmup + args.runs
    try:
        for variant in args.variants:
            model = (ROOT / ".models" / f"gliner2.5-{variant}-v1").resolve()
            manifest = model / "checkpoint.json"
            report["models"].append({"variant": variant, "checkpoint": json.loads(manifest.read_text()),
                                     "manifest_sha256": apple.file_sha(manifest)})
            references, case_paths = {}, {}
            for request in requests:
                case = [{"text": request["text"], "entities": request["entities"]}]
                path = artifacts / f"{variant}-{request['name']}-case.json"
                write(path, case)
                case_paths[request["name"]] = path
                command = [str(rust), str(model), str(path), "--repeat", "1", "--timing"]
                process, _ = run_process(command, environment, artifacts,
                                         f"{variant}-{request['name']}-cpu-reference", args.timeout)
                references[request["name"]] = apple.parse_run(process.stdout, process.stderr, 1)["outputs"][0]
                # Compare all tokenizer IDs/offsets, not merely output entities.
                prepared = []
                for engine, command in (
                    ("candle", [str(rust), str(model), str(path), "--inputs-only"]),
                    ("mlx", [str(swift), "--model", str(model), "--cases", str(path), "--inputs-only"])):
                    process, _ = run_process(command, environment, artifacts,
                                             f"{variant}-{request['name']}-inputs-{engine}", args.timeout)
                    prepared.append(json.loads(process.stdout))
                if prepared[0] != prepared[1]:
                    raise ValueError(f"Prepared inputs differ for {variant}/{request['name']}")
                if variant == args.variants[0]:
                    request.update(request_sha256=apple.canonical_sha({k: request[k] for k in ("name", "text", "entities")}),
                                   whitespace_word_count=len(request["text"].split()))
                request.setdefault("prepared_tokens_by_model", {})[variant] = len(prepared[0]["input_ids"])
            for trial in range(args.trials):
                for index, request in enumerate(requests):
                    for backend in pair_order(trial * len(requests) + index):
                        name = request["name"]
                        print(f"{variant} trial {trial + 1}/{args.trials}: {name} {backend}", file=sys.stderr, flush=True)
                        if backend == "candle_metal":
                            command = [str(rust), str(model), str(case_paths[name]), "--metal",
                                       "--repeat", str(repetitions), "--timing"]
                        else:
                            config = next(x for x in apple.configurations() if x["name"] == apple.BASELINE)
                            command = apple.command_for(swift, model, config, request, repetitions)
                        process, metadata = run_process(command, environment, artifacts,
                            f"{variant}-{name}-trial{trial + 1}-{backend}", args.timeout)
                        parsed = apple.parse_run(process.stdout, process.stderr, repetitions)
                        if backend == "candle_metal" and "Backend: Candle Metal GPU / F32" not in process.stderr:
                            raise ValueError("Candle failed to explicitly confirm the Metal device")
                        quality = apple.quality_report(references[name], parsed["outputs"], args.tolerance)
                        quality["reference_configuration"] = "fresh_candle_cpu_f32"
                        result = {"model": variant, "request": name, "trial": trial + 1, "backend": backend,
                                  **apple.timing_report(parsed, args.warmup), **metadata, "quality": quality,
                                  "representative_output": parsed["outputs"][0]}
                        report["runs"].append(result)
                        report["comparisons"] = summarize(report)
                        write(output, report)
                        if quality["runs_within_reference_tolerance"] != repetitions:
                            raise ValueError(f"Output parity failed for {variant}/{name}/{backend}: {quality['first_differences']}")
        report["direct_gpu_comparisons"] = direct_gpu_comparisons(report, args.tolerance)
        if not all(x["strict"] for x in report["direct_gpu_comparisons"]):
            raise ValueError("Strict direct GPU-to-GPU comparison failed")
        report.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status="failed", error=str(error))
        write(output, report)
        raise
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-binary", default=str(ROOT / "target/release/examples/apple_reference"))
    parser.add_argument("--mlx-binary", default=str(ROOT / ".derivedData/Build/Products/Release/gliner-mlx"))
    parser.add_argument("--variants", nargs="+", choices=("small", "base", "multi"), default=["small", "multi"])
    parser.add_argument("--requests")
    parser.add_argument("--output", default=str(ROOT / ".models/candle-vs-mlx-mac.json"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if (args.trials < 1 or args.warmup < 1 or args.runs < 1 or
            not math.isfinite(args.tolerance) or args.tolerance < 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("trials/warmup/runs/timeout must be positive and tolerance finite/nonnegative")
    try:
        report = benchmark(args)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"Metal comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "comparisons": report["comparisons"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
