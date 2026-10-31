#!/usr/bin/env python3
"""Build the self-contained English technical report from recorded benchmark data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

IMPLEMENTATION_COMMIT = "450367eb006a1d2ee088db343ef26a11a18aafd5"
CANDLE_IMPLEMENTATION_REF = "299797591d88245846a610bfc901c7267e0a3de3"
REPOSITORY_URL = "https://github.com/apiplant/gliner-rs"


def refresh_iphone_evidence(root, optimization_tag=None):
    """Export measured device samples to the same chart schema as the Mac run."""
    try:
        from scripts import benchmark_apple as apple
        from scripts.benchmark_metal_comparison import stats, summarize
    except ModuleNotFoundError:
        import benchmark_apple as apple
        from benchmark_metal_comparison import stats, summarize

    requests = apple.load_requests()
    report = {
        "schema_version": 1, "status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "Candle Metal F32 versus MLX Metal F32 in the same Release SwiftUI app",
        "device": {"model": "iPhone 17 Pro", "os": "iOS 27.2", "architecture": "arm64", "physical": True},
        "method": {"trials": 3, "warmup_runs_per_process": 5, "measured_runs_per_process": 30,
                   "fresh_process_per_trial_request_backend": False, "fresh_process_per_trial_backend": True,
                   "requests_per_process": 3, "backend_order": [["mlx", "candle"], ["candle", "mlx"], ["mlx", "candle"]],
                   "timing_scope": "Tokenization/preparation, extraction and backend-boundary conversion; excludes JSONL record serialization/file writing and actor/UI scheduling",
                   "timing_clock": "DispatchTime uptime nanoseconds; stated unit is not a timing-accuracy claim",
                   "aggregate": "Median of independent process medians; pooled p95 also reported",
                   "p95_method": "nearest rank", "confidence_tolerance": 1e-4, "array_order": "strict",
                   "filesystem_cache_flushed": False, "shader_cache_flushed": False,
                   "rss_method": "getrusage(RUSAGE_SELF).ru_maxrss in bytes; each process covers loading and all three requests, so its peak is repeated on each request row",
                   "mlx_configuration": {"dtype": "float32", "tf32": False, "cache_bytes": 20971520, "clear": True},
                   "candle_configuration": {"dtype": "F32", "device": "Metal", "mlx_runtime_initialized": False},
                   "limits": "Both engines linked in one SwiftUI harness; Rust includes JSON request/result bridge overhead; no energy or thermal instrumentation; not a cross-device latency comparison"},
        "models": [], "requests": requests, "runs": [], "comparisons": [], "direct_gpu_comparisons": []}
    provenance_name = f"iphone-candle-provenance-{optimization_tag}.json" if optimization_tag else "iphone-candle-provenance.json"
    provenance = json.loads((root / ".models" / provenance_name).read_text())
    # Device container URLs and signing/account details are unnecessary evidence.
    report["provenance"] = {k: provenance[k] for k in ("build", "sha256", "note")}
    for variant in ("small", "multi"):
        manifest = root / f".models/gliner2.5-{variant}-v1/checkpoint.json"
        report["models"].append({"variant": variant, "checkpoint": json.loads(manifest.read_text()),
                                 "manifest_sha256": apple.file_sha(manifest)})
        aggregate_suffix = f"-{optimization_tag}-optimized" if optimization_tag else ""
        aggregate = json.loads((root / f".models/iphone-{variant}-candle-vs-mlx{aggregate_suffix}.json").read_text())
        paired_outputs = {}
        configurations = [("candle_metal", "candle", None), ("mlx_metal", "mlx", None)]
        if optimization_tag:
            configurations = [("candle_metal_baseline", "candle", "original"),
                              ("candle_metal", "candle", "optimized"), ("mlx_metal", "mlx", "mlx")]
        for backend, engine, configuration in configurations:
            if optimization_tag:
                parity_prefix = root / f".models/iphone-{variant}-{engine}-parity-trial1-{optimization_tag}-{configuration}"
                parity_path = Path(str(parity_prefix) + ".summary.json")
                parity = json.loads(parity_path.read_text())
                if parity.get("status") != "passed" or parity.get("caseCount") != 28 or parity.get("backend") != engine:
                    raise ValueError(f"Invalid long-suite device summary: {parity_path}")
                report.setdefault("long_suite_memory", {}).setdefault(variant, {})[backend] = {
                    "cases": 28, "runs": 1, "run_id": parity["runID"],
                    "peak_process_rss_bytes": parity["peakResidentBytes"],
                    "peak_physical_footprint_bytes": parity["peakPhysicalFootprintBytes"],
                    "memory_points": parity.get("memoryPoints"),
                    "summary_sha256": apple.file_sha(parity_path)}
            for trial in range(1, 4):
                suffix = f"-{optimization_tag}-{configuration}" if optimization_tag else ""
                prefix = root / f".models/iphone-{variant}-{engine}-benchmark-trial{trial}{suffix}"
                summary_path, result_path = Path(str(prefix) + ".summary.json"), Path(str(prefix) + ".results.jsonl")
                summary = json.loads(summary_path.read_text())
                records = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
                if (summary.get("status") != "passed" or summary.get("backend") != engine or
                        summary.get("precision") != "float32" or len(records) != 105 or summary.get("caseCount") != 3):
                    raise ValueError(f"Incorrect device metadata: {prefix}")
                if engine == "mlx" and summary.get("tf32Environment") != "0":
                    raise ValueError("Device MLX run did not explicitly disable TF32")
                if optimization_tag and engine == "candle":
                    expected_loader = "mmap" if configuration == "original" else "stream"
                    if summary.get("candleMetalLoader") != expected_loader or summary.get("candleOriginalAttention") != (configuration == "original"):
                        raise ValueError(f"Wrong Candle optimization settings: {prefix}")
                for case_index, request in enumerate(requests):
                    rows = sorted((r for r in records if r["caseIndex"] == case_index), key=lambda r: r["iteration"])
                    if [r["iteration"] for r in rows] != list(range(35)):
                        raise ValueError(f"Missing device iterations: {prefix}/{case_index}")
                    if any(r.get("runID") != summary["runID"] or r.get("backend") != engine or
                           r.get("precision") != "float32" or r.get("warmup") != (r["iteration"] < 5) for r in rows):
                        raise ValueError(f"Stale or mislabeled device rows: {prefix}")
                    values = [r["seconds"] * 1000 for r in rows]
                    stats(values)  # Reject missing/nonfinite/nonpositive samples before publication.
                    outputs = [r["result"] for r in rows]
                    reference_path = root / f".models/iphone-rust-references/{variant}-{request['name']}-cpu-reference.jsonl"
                    if not reference_path.is_file():
                        reference_path = root / f".models/candle-vs-mlx-mac-artifacts/{variant}-{request['name']}-cpu-reference.jsonl"
                    reference = json.loads(reference_path.read_text())
                    quality = apple.quality_report(reference, outputs, 1e-4)
                    quality["reference_configuration"] = "candle_cpu_f32"
                    if quality["runs_within_reference_tolerance"] != 35:
                        raise ValueError(f"Device output parity failed: {prefix}/{request['name']}")
                    parsed = {"run_ms": values, "load_ms": summary["loadSeconds"] * 1000,
                              "timing_resolution_ms": 0.000001}
                    result = {"model": variant, "request": request["name"], "trial": trial,
                              "backend": backend, **apple.timing_report(parsed, 5),
                              "peak_process_rss_bytes": summary["peakResidentBytes"], "quality": quality,
                              "peak_physical_footprint_bytes": summary.get("peakPhysicalFootprintBytes"),
                              "memory_points": summary.get("memoryPoints"),
                              "summary_sha256": apple.file_sha(summary_path), "results_sha256": apple.file_sha(result_path),
                              "run_id": summary["runID"], "representative_output": outputs[0]}
                    report["runs"].append(result)
                    paired_outputs[(backend, trial, case_index)] = outputs
        for trial in range(1, 4):
            for case_index, request in enumerate(requests):
                pairs = [("candle_metal", "mlx_metal")]
                if optimization_tag:
                    pairs.append(("candle_metal_baseline", "candle_metal"))
                for reference_backend, candidate_backend in pairs:
                    pair = apple.compare_values(paired_outputs[(reference_backend, trial, case_index)],
                                                paired_outputs[(candidate_backend, trial, case_index)], tolerance=1e-4)
                    if pair.errors:
                        raise ValueError(f"Direct GPU-to-GPU parity failed: {variant}/{trial}/{case_index}")
                    report["direct_gpu_comparisons"].append({"model": variant, "request": request["name"],
                        "reference_backend": reference_backend, "candidate_backend": candidate_backend,
                        "trial": trial, "runs_per_backend": 35, "strict": True, "max_confidence_delta": pair.max_confidence_delta})
        report.setdefault("validation", {})[variant] = {"engines": {k: v["parity"] for k,v in aggregate["engines"].items()},
                                                       "direct_gpu": aggregate["direct_backend_comparison"]}
    for request in requests:
        request.update(whitespace_word_count=len(request["text"].split()),
                       request_sha256=apple.canonical_sha({k: request[k] for k in ("name", "text", "entities")}))
    report["comparisons"] = summarize(report)
    if optimization_tag:
        report["method"].update(backend_order=[["original", "optimized", "mlx"], ["optimized", "mlx", "original"], ["mlx", "original", "optimized"]],
            physical_footprint_method="task_vm_info.ledger_phys_footprint_peak, captured outside request timers; CPU and GPU footprint on Apple Silicon",
            optimization_configurations={"candle_metal_baseline": {"loader":"mmap", "attention":"original"},
                                         "candle_metal": {"loader":"stream", "attention":"optimized"}})
    filename = "candle-optimized-iphone.json" if optimization_tag else "candle-vs-mlx-iphone.json"
    destination = root / "docs/benchmarks" / filename
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Device evidence: {destination}")


def build(root, destination):
    comparison = json.loads((root / "docs/benchmarks/rust-vs-mlx.json").read_text())
    profiles = json.loads((root / "docs/benchmarks/mac-m3-pro-multi.json").read_text())
    implementation = json.loads((root / "docs/benchmarks/apple-port-configurations.json").read_text())
    data = {
        "provenance": {
            "implementationCommit": IMPLEMENTATION_COMMIT,
            "candleImplementationRef": CANDLE_IMPLEMENTATION_REF,
            "pullRequest": REPOSITORY_URL + "/pull/1",
        },
        "comparison": comparison,
        "profiles": profiles,
        "implementation": implementation,
        "validation": {
            "note": "Original MLX-only validation and device trace runs; current optimization tests are in optimization.tests.",
            "swiftTests": 32, "pythonTests": 36, "inputCasesPerTokenizer": 92,
            "coreCases": 21, "advancedCases": 7,
            "phoneSmall": {"cases": 28, "strictCases": 28, "encoderValues": 1024128,
                           "maxEncoderError": 0.000072479248046875,
                           "loadSeconds": 0.539281083, "extractionSumSeconds": 2.052727505,
                           "peakResidentBytes": 592855040},
            "phoneMulti": {"cases": 28, "strictCases": 27, "encoderValues": 2592768,
                           "maxEncoderError": 0.00011027045547962189,
                           "maxConfidenceError": 0.000013530254364013672,
                           "loadSeconds": 1.331397791, "extractionSumSeconds": 2.715662457,
                           "peakResidentBytes": 1492697088, "peakMLXBytes": 2673877508},
            "confidenceTolerance": 0.0001, "encoderTolerance": 0.001,
            "nearTieDifference": 0.00000011920928955078125,
        },
        "macDemoSmokeTest": {
            "checkpoint": "multi", "precision": "float32", "cacheMiB": "auto",
            "clearCacheAfterRequest": False, "tf32": False,
            "firstCallIncludingLoadMs": 1251.326,
            "subsequentCallsMs": [57.608, 47.763, 54.531, 41.933],
            "note": "Manually spaced worker timings; excludes rendering/actor queue delay; not a paired CLI benchmark.",
        },
        "example": {
            "text": "Michele vive a Milano e lavora per Acme.",
            "entities": [
                {"text": "Michele", "label": "person", "confidence": 0.9974225759506226},
                {"text": "Milano", "label": "city", "confidence": 0.9989032745361328},
                {"text": "Acme", "label": "organization", "confidence": 0.9804876446723938},
            ],
        },
    }
    evaluation = root / "docs/benchmarks/candle-metal-evaluation.json"
    if evaluation.is_file():
        data["metalComparison"] = json.loads(evaluation.read_text())
        for target in ("mac", "iphone"):
            evidence = root / f"docs/benchmarks/candle-vs-mlx-{target}.json"
            data["metalComparison"][target] = json.loads(evidence.read_text()) if evidence.is_file() else None
    optimization = root / "docs/benchmarks/candle-metal-optimization.json"
    if optimization.is_file():
        data["optimization"] = json.loads(optimization.read_text())
        for target in ("mac", "iphone"):
            data["optimization"][target] = json.loads((root / f"docs/benchmarks/candle-optimized-{target}.json").read_text())
    # Local build paths are diagnostics, not portable reproduction commands.
    # Preserve the committed evidence and hashes; normalize paths only in HTML.
    def portable_paths(value):
        if isinstance(value, str):
            return value.replace(str(root), ".")
        if isinstance(value, list):
            return [portable_paths(item) for item in value]
        if isinstance(value, dict):
            return {key: portable_paths(item) for key, item in value.items()}
        return value
    data = portable_paths(data)
    data["provenance"]["pathNormalization"] = "Repository-root absolute paths are rendered relative in this HTML; committed evidence remains unchanged."
    template = (root / "scripts/templates/apple_report.html").read_text()
    # Keep source links usable when the standalone file is served by the website
    # or downloaded outside the repository. Pin them to the measured implementation.
    template = template.replace('__CANDLE_REF__', CANDLE_IMPLEMENTATION_REF)
    template = template.replace('href="../', f'href="{REPOSITORY_URL}/blob/{IMPLEMENTATION_COMMIT}/')
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    document = template.replace("__REPORT_DATA__", encoded)
    if "__REPORT_DATA__" in document:
        raise ValueError("Report data was not inserted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    print(f"Report: {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh-iphone-evidence", action="store_true", help="Export fresh local device artifacts before building HTML")
    parser.add_argument("--optimization-tag", help="Export the balanced original/optimized/MLX iPhone run with this artifact tag")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.refresh_iphone_evidence:
        refresh_iphone_evidence(repo)
    if args.optimization_tag:
        refresh_iphone_evidence(repo, args.optimization_tag)
    build(repo, args.output or repo / "reports/gliner-apple-report.html")
