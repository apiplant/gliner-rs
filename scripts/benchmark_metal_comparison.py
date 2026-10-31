#!/usr/bin/env python3
"""Paired Release Candle Metal / MLX Metal measurements on identical checkpoints.

Run only after parity checks, while other local GPU jobs are stopped. Each
trial/request/backend starts a fresh process. Backend order alternates across pairs, or rotates when a baseline is supplied. Every output (warmups included) is checked against a fresh CPU reference;
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
FOOTPRINT_RE = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$", re.MULTILINE)


def stats(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timings must be finite and positive")
    ordered = sorted(values)
    return {"median_ms": statistics.median(values),
            "p95_ms": ordered[math.ceil(len(values) * .95) - 1],
            "min_ms": min(values), "max_ms": max(values), "samples": len(values)}


def pair_order(index, baseline=False):
    if baseline:
        engines = ["candle_metal_baseline", "candle_metal", "mlx_metal"]
        shift = index % len(engines)
        return engines[shift:] + engines[:shift]
    return ["candle_metal", "mlx_metal"] if index % 2 == 0 else ["mlx_metal", "candle_metal"]


def source_revision():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    return {"head": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain"))}


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
    footprint = FOOTPRINT_RE.findall(process.stderr)
    return process, {"command": command, "stdout_artifact": str(stdout_path),
                     "stderr_artifact": str(stderr_path),
                     "peak_process_rss_bytes": int(rss[-1]) if rss else None,
                     "peak_physical_footprint_bytes": int(footprint[-1]) if footprint else None}


def summarize(report):
    rows = []
    for model in report["models"]:
        for request in report["requests"]:
            selected = [x for x in report["runs"] if x["model"] == model["variant"] and x["request"] == request["name"]]
            row = {"model": model["variant"], "request": request["name"], "backends": {}}
            for backend in ("candle_metal", "mlx_metal", "candle_metal_baseline"):
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
                    "peak_physical_footprint_bytes_by_trial": [x.get("peak_physical_footprint_bytes") for x in runs],
                    "all_outputs_correct": all(x["quality"]["runs_within_reference_tolerance"] == x["repetitions"] for x in runs),
                    "max_confidence_delta": max(x["quality"]["max_confidence_delta"] for x in runs),
                }
            if "candle_metal" in row["backends"] and "mlx_metal" in row["backends"]:
                candle = row["backends"]["candle_metal"]["median_of_trial_medians_ms"]
                mlx = row["backends"]["mlx_metal"]["median_of_trial_medians_ms"]
                row["candle_to_mlx_latency_ratio"] = candle / mlx
                row["mlx_latency_reduction_percent"] = (candle - mlx) / candle * 100
            if "candle_metal_baseline" in row["backends"] and "candle_metal" in row["backends"]:
                old, new = row["backends"]["candle_metal_baseline"], row["backends"]["candle_metal"]
                def reduction(before, after):
                    return (before - after) / before * 100 if before else None
                def median_available(values):
                    return statistics.median(values) if all(v is not None for v in values) else None
                old_rss = median_available(old["peak_process_rss_bytes_by_trial"])
                new_rss = median_available(new["peak_process_rss_bytes_by_trial"])
                old_footprint = median_available(old["peak_physical_footprint_bytes_by_trial"])
                new_footprint = median_available(new["peak_physical_footprint_bytes_by_trial"])
                row["candle_before_after"] = {
                    "latency_reduction_percent": reduction(old["median_of_trial_medians_ms"], new["median_of_trial_medians_ms"]),
                    "load_reduction_percent": reduction(statistics.median(old["load_ms_by_trial"]), statistics.median(new["load_ms_by_trial"])),
                    "peak_rss_reduction_percent": reduction(old_rss, new_rss) if old_rss is not None and new_rss is not None else None,
                    "peak_physical_footprint_reduction_percent": reduction(old_footprint, new_footprint) if old_footprint is not None and new_footprint is not None else None,
                }
            rows.append(row)
    return rows


def direct_gpu_comparisons(report, tolerance):
    pairs = {}
    for run in report["runs"]:
        pairs.setdefault((run["model"], run["request"], run["trial"]), {})[run["backend"]] = run
    results = []
    for (model, request, trial), runs in pairs.items():
        outputs = {backend: [json.loads(line) for line in Path(run["stdout_artifact"]).read_text().splitlines() if line.strip()]
                   for backend, run in runs.items()}
        for reference, candidate in (("candle_metal", "mlx_metal"),
                                     ("candle_metal_baseline", "candle_metal")):
            if reference not in outputs or candidate not in outputs:
                continue
            comparison = apple.compare_values(outputs[reference], outputs[candidate], tolerance=tolerance)
            results.append({"model": model, "request": request, "trial": trial,
                            "reference_backend": reference, "candidate_backend": candidate,
                            "runs_per_backend": len(outputs[reference]), "strict": not comparison.errors,
                            "max_confidence_delta": comparison.max_confidence_delta,
                            "errors": comparison.errors[:20]})
    return results


def benchmark(args):
    output = Path(args.output).resolve()
    artifacts = output.parent / (output.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    rust, swift = Path(args.rust_binary).resolve(), Path(args.mlx_binary).resolve()
    baseline_arg = getattr(args, "baseline_rust_binary", None)
    baseline = Path(baseline_arg).resolve() if baseline_arg else None
    for binary in (rust, swift, *([baseline] if baseline else [])):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Executable not found: {binary}")
    requests = apple.load_requests(args.requests)
    report = {
        "schema_version": 1, "status": "running", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": apple.machine_info(), "comparison": "Candle Metal GPU F32 versus MLX Metal GPU F32",
        "method": {"trials": args.trials, "warmup_runs_per_process": args.warmup,
                   "measured_runs_per_process": args.runs, "fresh_process_per_trial_request_backend": True,
                   "backend_order": ("Rotating baseline/optimized/MLX across trial/request pairs" if baseline
                                     else "Alternating AB/BA across trial/request pairs"),
                   "timing_scope": "Preparation/tokenization, inference, output formatting/JSON and synchronous stdout write; model loading separate",
                   "timing_clock": "Rust Instant; Swift DispatchTime uptime (both monotonic)",
                   "aggregate": "Median of independent process medians; pooled p95 also reported",
                   "p95_method": "nearest rank", "confidence_tolerance": args.tolerance,
                   "array_order": "strict", "filesystem_cache_flushed": False, "shader_cache_flushed": False,
                   "rss_method": "/usr/bin/time -l maximum resident set size (bytes on macOS); entire process peak",
                   "physical_footprint_method": "/usr/bin/time -l peak memory footprint; includes CPU and GPU allocations charged to the process; distinct from resident clean file pages",
                   "mlx_configuration": {"dtype": "float32", "tf32": False, "cache": "auto", "clear": False},
                   "candle_configuration": {"dtype": "F32", "device": "Metal", "rayon_threads": 4},
                   "limits": "Local sequential trials; no thermal/energy instrumentation or statistical confidence intervals"},
        "builds": {"candle": {"binary": str(rust), "sha256": apple.file_sha(rust)},
                   "mlx": {"binary": str(swift), "sha256": apple.file_sha(swift)}},
        "source_revision": source_revision(),
        "source_sha256": {str(p.relative_to(ROOT)): apple.file_sha(p) for p in
                          [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "examples/apple_reference.rs",
                           *sorted((ROOT / "src").rglob("*.rs")),
                           *sorted((ROOT / "src").rglob("*.metal")),
                           *sorted((ROOT / "swift/Sources").rglob("*.swift"))]},
        "models": [], "requests": requests, "runs": [], "comparisons": []}
    if baseline:
        report["builds"]["candle_baseline"] = {"binary": str(baseline), "sha256": apple.file_sha(baseline),
                                                "declared_source_ref": getattr(args, "baseline_source_ref", None)}
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
                input_commands = [
                    ("candle", [str(rust), str(model), str(path), "--inputs-only"]),
                    ("mlx", [str(swift), "--model", str(model), "--cases", str(path), "--inputs-only"])]
                if baseline:
                    input_commands.append(("candle-baseline", [str(baseline), str(model), str(path), "--inputs-only"]))
                for engine, command in input_commands:
                    process, _ = run_process(command, environment, artifacts,
                                             f"{variant}-{request['name']}-inputs-{engine}", args.timeout)
                    prepared.append(json.loads(process.stdout))
                if any(value != prepared[0] for value in prepared[1:]):
                    raise ValueError(f"Prepared inputs differ for {variant}/{request['name']}")
                if variant == args.variants[0]:
                    request.update(request_sha256=apple.canonical_sha({k: request[k] for k in ("name", "text", "entities")}),
                                   whitespace_word_count=len(request["text"].split()))
                request.setdefault("prepared_tokens_by_model", {})[variant] = len(prepared[0]["input_ids"])
            for trial in range(args.trials):
                for index, request in enumerate(requests):
                    for backend in pair_order(trial * len(requests) + index, baseline=bool(baseline)):
                        name = request["name"]
                        print(f"{variant} trial {trial + 1}/{args.trials}: {name} {backend}", file=sys.stderr, flush=True)
                        if backend in ("candle_metal", "candle_metal_baseline"):
                            command = [str(baseline if backend == "candle_metal_baseline" else rust), str(model), str(case_paths[name]), "--metal",
                                       "--repeat", str(repetitions), "--timing"]
                        else:
                            config = next(x for x in apple.configurations() if x["name"] == apple.BASELINE)
                            command = apple.command_for(swift, model, config, request, repetitions)
                        process, metadata = run_process(command, environment, artifacts,
                            f"{variant}-{name}-trial{trial + 1}-{backend}", args.timeout)
                        parsed = apple.parse_run(process.stdout, process.stderr, repetitions)
                        if backend in ("candle_metal", "candle_metal_baseline") and "Backend: Candle Metal GPU / F32" not in process.stderr:
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
    parser.add_argument("--baseline-rust-binary", help="Archived pre-optimization Candle reference binary")
    parser.add_argument("--baseline-source-ref", help="Declared source ref for the archived binary (not independently verified)")
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
    if args.baseline_source_ref and not args.baseline_rust_binary:
        parser.error("--baseline-source-ref requires --baseline-rust-binary")
    try:
        report = benchmark(args)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"Metal comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "comparisons": report["comparisons"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
