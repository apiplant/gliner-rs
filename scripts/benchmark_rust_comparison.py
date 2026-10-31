#!/usr/bin/env python3
"""Compare Rust/Candle CPU F32 with Swift/MLX Metal GPU F32 on identical inputs.

This is a BACKEND/DEVICE comparison, not a programming-language speed test.
The Rust executable is the Release apple_reference example built without the
optional MKL/Accelerate/CUDA backends. Each request/thread setting starts a fresh
process. Threads default to 8, then 4; eight matches the Rust CLI's min(8,cores)
default on this Mac. Five warmups include the recorded cold first call, followed
by 30 measured calls. Arrays remain strictly ordered in quality checks.

--refresh-mlx first repeats the three accurate GPU measurements, then runs all
CPU measurements sequentially. Otherwise GPU measurements are imported from the
specified report, with their original timestamp/provenance kept explicit. OS
file and shader caches are NOT flushed, so cold means first call in a process.
No numerical kernels or tolerances are adjusted by this script.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from scripts import benchmark_apple as apple
except ModuleNotFoundError:
    import benchmark_apple as apple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_COMMAND = "cargo +1.99.1 build --release --no-default-features --example apple_reference"


def thread_counts(text):
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("threads must be comma-separated positive integers") from error
    if not values or any(value < 1 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("thread counts must be positive and unique")
    return values


def baseline_from_report(path):
    report = json.loads(Path(path).read_text())
    config = next((item for item in report["configurations"] if item["name"] == apple.BASELINE), None)
    if not config or config["precision"] != "float32" or config["tf32"]:
        raise ValueError("Source report must contain the accurate full-F32, TF32-off baseline")
    requests = report["requests"]
    entries = {entry["request"]: entry for entry in config["results"]}
    if len(entries) != len(requests):
        raise ValueError("Source baseline does not cover all requests")
    for request in requests:
        entry = entries[request["name"]]
        if entry.get("status") != "passed" or "reference_output" not in entry:
            raise ValueError(f"Source baseline is not valid for {request['name']}")
        # Reuse the strict JSON/finite checker; no hidden near-tie allowance.
        finite = apple.compare_values(entry["reference_output"], entry["reference_output"])
        if finite.errors:
            raise ValueError(f"Invalid source baseline for {request['name']}: {'; '.join(finite.errors)}")
        original = {key: request[key] for key in ("name", "text", "entities")}
        if request.get("request_sha256") and apple.canonical_sha(original) != request["request_sha256"]:
            raise ValueError(f"Source request hash mismatch for {request['name']}")
    return report, requests, entries


def link_dependencies(binary):
    try:
        process = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True, timeout=15)
        return process.stdout.splitlines()[1:] if process.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def source_fingerprints():
    names = ["Cargo.toml", "Cargo.lock", "examples/apple_reference.rs", "src/deberta.rs", "src/heads.rs", "src/model.rs"]
    return {name: apple.file_sha(ROOT / name) for name in names if (ROOT / name).is_file()}


def execute(command, environment, repetitions, warmup, artifacts, stem, reference, tolerance, timeout):
    started = time.perf_counter()
    process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=timeout)
    wall = (time.perf_counter() - started) * 1000
    stdout_path, stderr_path = artifacts / (stem + ".jsonl"), artifacts / (stem + ".stderr.log")
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    if process.returncode:
        raise ValueError(f"Process exited {process.returncode}; see {stderr_path}: {process.stderr[-2000:]}")
    parsed = apple.parse_run(process.stdout, process.stderr, repetitions)
    result = apple.timing_report(parsed, warmup)
    result.update({"command": command, "process_wall_ms": wall,
                   "process_overhead_ms": max(0.0, wall - result["load_ms"] - sum(result["all_run_ms"])),
                   "stdout_artifact": str(stdout_path), "stderr_artifact": str(stderr_path),
                   "quality_vs_mlx_accurate": apple.quality_report(reference, parsed["outputs"], tolerance),
                   "representative_output": parsed["outputs"][0]})
    result["status"] = "passed" if result["quality_vs_mlx_accurate"]["runs_within_reference_tolerance"] == repetitions else "quality_deviation"
    return result


def thread_context():
    raw = apple._sysctl("hw.perflevel0.logicalcpu")
    return {
        "controlled_scope": "RAYON_NUM_THREADS: Candle gemm/Rayon matmul, matching the Rust CLI --threads option",
        "other_pinned_variables": ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"],
        "candle_kernel_pool_environment": os.environ.get("CANDLE_NUM_THREADS"),
        "candle_kernel_pool_policy": "CANDLE_NUM_THREADS is inherited; when unset Candle 0.11 defaults its separate barrier-kernel pool to the performance-core count",
        "performance_core_count": int(raw) if raw and raw.isdigit() else None,
    }


def summarize(report):
    gpu = {result["request"]: result for result in report["mlx_accurate"]["results"]}
    rows = []
    for request in report["requests"]:
        name = request["name"]
        candidates = []
        for config in report["rust_configurations"]:
            result = next((value for value in config["results"] if value["request"] == name), None)
            if result and result.get("status") != "failed":
                candidates.append((config, result))
        if not candidates:
            continue
        best_config, best_result = min(candidates, key=lambda item: item[1]["warm_median_ms"])
        default = next((result for config, result in candidates if config["matches_rust_cli_default"]), None)
        gpu_result = gpu[name]
        row = {"request": name, "whitespace_word_count": request["whitespace_word_count"],
               "mlx_gpu_f32_median_ms": gpu_result["warm_median_ms"], "mlx_gpu_f32_p95_ms": gpu_result["warm_p95_ms"],
               "rust_cpu_best_threads": best_config["threads"],
               "rust_cpu_best_median_ms": best_result["warm_median_ms"], "rust_cpu_best_p95_ms": best_result["warm_p95_ms"],
               "rust_best_cpu_to_mlx_gpu_latency_ratio": best_result["warm_median_ms"] / gpu_result["warm_median_ms"] if gpu_result["warm_median_ms"] > 0 else None}
        if default:
            row.update({"rust_cpu_default_threads": next(config["threads"] for config, result in candidates if result is default),
                        "rust_cpu_default_median_ms": default["warm_median_ms"], "rust_cpu_default_p95_ms": default["warm_p95_ms"],
                        "rust_default_cpu_to_mlx_gpu_latency_ratio": default["warm_median_ms"] / gpu_result["warm_median_ms"] if gpu_result["warm_median_ms"] > 0 else None})
        rows.append(row)
    return rows


def run_benchmark(args):
    source_path, destination = Path(args.mlx_report).resolve(), Path(args.output).resolve()
    source, requests, baseline = baseline_from_report(source_path)
    model = Path(args.model or source["model"]).resolve()
    if str(model) != str(Path(source["model"]).resolve()):
        raise ValueError("Model path differs from the GPU source report; use the identical checkpoint")
    binary, gpu_binary = Path(args.rust_binary).resolve(), Path(args.mlx_binary or source["binary"]).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Rust reference executable not found: {binary}")
    if args.refresh_mlx and (not gpu_binary.is_file() or not os.access(gpu_binary, os.X_OK)):
        raise ValueError(f"MLX executable not found: {gpu_binary}")
    if not (model / "model.safetensors").is_file():
        raise ValueError("Checkpoint model.safetensors does not exist")
    device = apple.machine_info()
    for key in ("cpu", "machine", "macos", "physical_memory_bytes"):
        if device.get(key) != source["device"].get(key):
            raise ValueError(f"GPU baseline hardware/OS mismatch for {key}: {device.get(key)!r} vs {source['device'].get(key)!r}")
    default_threads = min(8, device.get("logical_cpu_count") or 8)
    warmup, repetitions = max(1, args.warmup), max(1, args.warmup) + args.runs
    artifacts = destination.parent / (destination.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    linkage = link_dependencies(binary)
    unexpected = [line for line in linkage if any(name in line.lower() for name in ("accelerate.framework", "mkl", "libcuda"))]
    if unexpected:
        raise ValueError("Rust executable unexpectedly links an optional accelerated CPU/CUDA backend: " + "; ".join(unexpected))
    report = {
        "status": "running", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": device,
        "comparison": "Rust/Candle CPU F32 versus Swift/MLX Metal GPU F32; not a language performance comparison",
        "model": str(model), "requests": requests,
        "method": {"warmup_runs_including_cold": warmup, "measured_runs_per_request": args.runs,
                   "fresh_process_per_request_and_thread_setting": True,
                   "cpu_gpu_executed_sequentially": True, "filesystem_cache_flushed": False,
                   "shader_cache_flushed": False, "p95_method": "nearest rank",
                   "timing_scope": "CLI timer: preparation, inference, formatting and stdout serialization",
                   "numeric_tolerance": args.tolerance, "array_order": "strict",
                   "thread_context": thread_context(), "tested_matmul_thread_counts": args.threads,
                   "best_thread_selection": "Lowest median among the explicitly tested Rayon thread counts; not a claim of global optimality"},
        "rust_build": {"binary": str(binary), "binary_sha256": apple.file_sha(binary),
                       "declared_build_command": args.rust_build_command,
                       "declared_backend": "Candle CPU; pure Rust gemm; no MKL/Accelerate/CUDA",
                       "source_sha256": source_fingerprints(), "dynamic_link_dependencies": linkage},
        "mlx_accurate": {"source_report": str(source_path), "source_report_sha256": apple.file_sha(source_path),
                         "source_generated_at_utc": source.get("generated_at_utc"),
                         "source_binary_sha256": source.get("binary_sha256"), "refreshed": args.refresh_mlx,
                         "configuration": {"precision": "float32", "tf32": False, "cache_mib": "auto", "clear_cache": False},
                         "results": []}, "rust_configurations": [],
    }
    manifest = model / "checkpoint.json"
    if manifest.is_file():
        report["checkpoint_manifest"] = json.loads(manifest.read_text())
        report["checkpoint_manifest_sha256"] = apple.file_sha(manifest)
    if args.refresh_mlx:
        report["mlx_accurate"]["binary"] = str(gpu_binary)
        report["mlx_accurate"]["binary_sha256"] = apple.file_sha(gpu_binary)
        config = next(config for config in apple.configurations() if config["name"] == apple.BASELINE)
        for request in requests:
            print(f"MLX accurate GPU: {request['name']} ({repetitions} runs)", file=sys.stderr, flush=True)
            command = apple.command_for(gpu_binary, model, config, request, repetitions)
            try:
                result = execute(command, dict(os.environ, MLX_ENABLE_TF32="0"), repetitions, warmup, artifacts,
                                 "mlx-accurate-" + request["name"], baseline[request["name"]]["reference_output"], args.tolerance, args.timeout)
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                result = {"status": "failed", "error": str(error), "command": command}
                report["mlx_accurate"]["results"].append({"request": request["name"], **result})
                report.update(status="failed", comparisons=[])
                apple.write_report(destination, report)
                return report
            report["mlx_accurate"]["results"].append({"request": request["name"], **result})
            apple.write_report(destination, report)
    else:
        report["mlx_accurate"]["results"] = [baseline[request["name"]] for request in requests]
    for threads in args.threads:
        configuration = {"name": f"rust_cpu_f32_{threads}_threads", "backend": "Candle CPU F32, pure Rust gemm",
                         "threads": threads, "matches_rust_cli_default": threads == default_threads, "results": []}
        report["rust_configurations"].append(configuration)
        for request in requests:
            print(f"Rust CPU {threads} threads: {request['name']} ({repetitions} runs)", file=sys.stderr, flush=True)
            case_path = artifacts / ("request-" + request["name"] + ".json")
            case_path.write_text(json.dumps([{"text": request["text"], "entities": request["entities"],
                                             "include_spans": True, "include_confidence": True}], ensure_ascii=False), encoding="utf-8")
            command = [str(binary), str(model), str(case_path), "--repeat", str(repetitions), "--timing"]
            environment = dict(os.environ)
            for variable in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                environment[variable] = str(threads)
            try:
                # If GPU was refreshed, use its actual fresh output for quality;
                # the refresh itself has already been checked against the source.
                gpu = next(result for result in report["mlx_accurate"]["results"] if result["request"] == request["name"])
                reference = gpu.get("representative_output", baseline[request["name"]]["reference_output"])
                result = execute(command, environment, repetitions, warmup, artifacts,
                                 configuration["name"] + "-" + request["name"], reference, args.tolerance, args.timeout)
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                result = {"status": "failed", "error": str(error), "command": command}
            configuration["results"].append({"request": request["name"], "thread_environment": {key: environment[key] for key in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}, **result})
            report["comparisons"] = summarize(report)
            apple.write_report(destination, report)
    values = report["mlx_accurate"]["results"] + [result for config in report["rust_configurations"] for result in config["results"]]
    report["status"] = "complete" if all(result.get("status") == "passed" for result in values) else "failed"
    report["comparisons"] = summarize(report)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    apple.write_report(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlx-report", default=str(ROOT / ".models/mac-multi-benchmark.json"))
    parser.add_argument("--rust-binary", default=str(ROOT / "target/release/examples/apple_reference"))
    parser.add_argument("--model", help="Must match the GPU report's checkpoint; default: source model path")
    parser.add_argument("--output", "--output-path", default=str(ROOT / ".models/rust-vs-mlx-benchmark.json"))
    parser.add_argument("--threads", type=thread_counts, default=[8, 4])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--refresh-mlx", action="store_true")
    parser.add_argument("--mlx-binary")
    parser.add_argument("--rust-build-command", default=DEFAULT_BUILD_COMMAND, help="Recorded as declared, not inferred from binary")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("runs and timeout must be positive; warmup nonnegative")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    try:
        report = run_benchmark(args)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"Comparison benchmark failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(Path(args.output).resolve()), "status": report["status"],
                      "comparisons": report["comparisons"]}, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
