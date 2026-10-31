#!/usr/bin/env python3
"""Benchmark the Release Apple CLI with explicit runtime and quality reporting.

Each configuration/request runs in a NEW process because MLX latches TF32.
The default is five warm-up runs (including the reported cold first run), then
30 measured runs. Timings come from the CLI's per-run timer and include its
request preparation, inference, formatting and stdout serialization; subprocess
startup is reported separately. "Cold" means first use in that process, NOT a
flushed filesystem or Metal shader cache. No GPU job runs with --dry-run.

All outputs must be finite JSON objects. Arrays remain strictly ordered in
quality comparisons. Approximate FP16/TF32 differences are reported, including
changed spans/labels/counts, and never described as reference parity. Accurate
configurations must agree within --tolerance; approximate deviations fail only
with --fail-on-approximate-deviation. Raw outputs/stderr are saved alongside the
JSON report. On unsupported GPU architectures the TF32 entry is explicitly
not_applicable rather than presented as an acceleration measurement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

try:
    from scripts.compare_apple_results import compare_values
except ModuleNotFoundError:
    from compare_apple_results import compare_values

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "accurate_f32_auto"
DEFAULT_REQUESTS = [
    {
        "name": "michele_milano",
        "text": "Michele Tagliabue è arrivato a Milano lunedì per incontrare il gruppo di ricerca di Apple. Durante la riunione ha presentato un progetto per organizzare documenti in italiano direttamente sul Mac, senza inviare dati a servizi esterni. Nel pomeriggio Michele ha parlato con Giulia Rossi della prossima dimostrazione a Torino e ha preparato una breve relazione per i colleghi che lavorano nella sede di Roma.",
        "entities": ["person", "company", "location", "product", "date"],
    },
    {
        "name": "apple_ceo",
        "text": "Tim Cook, CEO di Apple, ha incontrato alcuni sviluppatori italiani a Cupertino per discutere delle applicazioni dedicate a iPhone e Mac. Alla presentazione erano presenti Michele Tagliabue e Laura Bianchi, arrivati da Milano con una dimostrazione di ricerca locale nei documenti. Cook ha ricordato che le prestazioni devono essere valutate insieme alla precisione e alla memoria utilizzata. Dopo l'incontro, il gruppo ha visitato San Francisco e ha confrontato le proprie note con i colleghi di Microsoft. Una seconda sessione è prevista a Roma il prossimo mese, con esempi pratici e documenti più lunghi rispetto alla prima prova.",
        "entities": ["person", "company", "location", "product", "date"],
    },
    {
        "name": "technology_report",
        "text": "Nel rapporto preparato a Milano, Michele Tagliabue descrive tre progetti per analizzare testi aziendali senza una connessione a Internet. Il primo utilizza un MacBook Pro di Apple per individuare persone, aziende e città nei verbali delle riunioni. Il secondo riguarda un'applicazione per iPhone che organizza ricevute e contratti, mentre il terzo confronta documenti provenienti dagli uffici di Roma, Torino e Bologna. Giulia Rossi coordina la raccolta degli esempi e Luca Ferri controlla che ogni risultato conservi gli stessi riferimenti al testo originale. Durante una visita a Cupertino, il gruppo ha discusso con Tim Cook delle differenze fra velocità, consumo di memoria e precisione numerica. Nella versione finale del rapporto sono citati anche Microsoft, Amazon e Google, con l'obiettivo di presentare una dimostrazione ripetibile ai colleghi e ai responsabili del progetto.",
        "entities": ["person", "company", "location", "product", "date"],
    },
]
LOAD_RE = re.compile(r"^Loaded in ([0-9]+(?:\.[0-9]+)?) s\s*$", re.MULTILINE)
RUN_RE = re.compile(r"^Case (\d+)/(\d+), run (\d+): ([0-9]+(?:\.[0-9]+)?) s\s*$", re.MULTILINE)
ARCH_RE = re.compile(r"(?:Metal architecture|GPU architecture):\s*(\S+)", re.IGNORECASE)


def canonical_sha(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _sysctl(name):
    try:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def machine_info():
    memory = _sysctl("hw.memsize")
    return {
        "system": platform.system(), "macos": platform.mac_ver()[0] or None,
        "kernel": platform.release(), "machine": platform.machine(),
        "model": _sysctl("hw.model"),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
        "physical_memory_bytes": int(memory) if memory and memory.isdigit() else None,
        "logical_cpu_count": os.cpu_count(),
    }


def tf32_applicability(device, architecture=None):
    """Mirror the pinned MLX 0.33 Metal NAX dispatch, without claiming a speedup."""
    if device.get("system") != "Darwin":
        return {"status": "not_applicable", "reason": "This benchmark targets macOS Metal."}
    os_version = tuple(int(x) for x in (device.get("macos") or "0").split(".") if x.isdigit())
    os_version = (os_version + (0, 0))[:2]
    if os_version < (26, 2):
        return {"status": "not_applicable", "reason": "MLX NAX requires macOS 26.2 or newer."}
    match = re.search(r"(\d{2})([a-z])$", architecture or "", re.IGNORECASE)
    if match:
        generation, variant = int(match[1]), match[2].lower()
        supported = generation >= (18 if variant == "p" else 17)
        return {"status": "applicable" if supported else "not_applicable",
                "reason": "Pinned MLX 0.33 NAX GPU-generation and OS gates.",
                "architecture": architecture, "architecture_generation": generation}
    # The supported binary may omit GPU metadata. Known older Macs still must
    # not be advertised as measuring a TF32 accelerator they do not have.
    match = re.search(r"Apple\s+M([1-4])(?:\s|$)", device.get("cpu") or "")
    if match:
        return {"status": "not_applicable", "reason": f"Apple M{match[1]} has no MLX NAX TF32 route.",
                "architecture": architecture}
    return {"status": "not_measured", "reason": "GPU architecture unavailable; accelerator support not verified.",
            "architecture": architecture}


def configurations(include_cache20=False):
    result = [
        {"name": BASELINE, "profile": "accurate", "precision": "float32", "tf32": False,
         "cache_mib": "auto", "clear_cache": False, "approximate": False},
        {"name": "accurate_f32_no_cache", "profile": "accurate", "precision": "float32", "tf32": False,
         "cache_mib": 0, "clear_cache": True, "approximate": False},
    ]
    if include_cache20:
        result.append({"name": "accurate_f32_20mib", "profile": "accurate", "precision": "float32", "tf32": False,
                       "cache_mib": 20, "clear_cache": False, "approximate": False})
    result += [
        {"name": "fast_fp16_auto", "profile": "fast", "precision": "float16", "tf32": False,
         "cache_mib": "auto", "clear_cache": False, "approximate": True},
        {"name": "low_memory_fp16_20mib", "profile": "low-memory", "precision": "float16", "tf32": False,
         "cache_mib": 20, "clear_cache": True, "approximate": True},
        {"name": "tf32_f32_auto", "profile": "accurate", "precision": "float32", "tf32": True,
         "cache_mib": "auto", "clear_cache": False, "approximate": True},
    ]
    return result


def command_for(binary, model, config, request, repetitions):
    return [str(binary), "--model", str(model), "--text", request["text"],
            "--entities", ",".join(request["entities"]), "--repeat", str(repetitions),
            "--profile", config["profile"], "--fp16" if config["precision"] == "float16" else "--fp32",
            "--tf32" if config["tf32"] else "--no-tf32", "--cache-mb", str(config["cache_mib"]),
            "--clear-cache" if config["clear_cache"] else "--retain-cache"]


def parse_run(stdout, stderr, repetitions):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant {value}")
    outputs = [json.loads(line, parse_constant=reject) for line in stdout.splitlines() if line.strip()]
    if len(outputs) != repetitions:
        raise ValueError(f"Expected {repetitions} JSON outputs, got {len(outputs)}")
    for index, output in enumerate(outputs):
        if type(output) is not dict:
            raise ValueError(f"Output {index} is not a JSON object")
        finite = compare_values(output, output)
        if finite.errors:
            raise ValueError(f"Output {index} invalid: {'; '.join(finite.errors)}")
    loads, runs = LOAD_RE.findall(stderr), RUN_RE.findall(stderr)
    if len(loads) != 1:
        raise ValueError(f"Expected one model-load timing, got {len(loads)}")
    if len(runs) != repetitions:
        raise ValueError(f"Expected {repetitions} per-run timings, got {len(runs)}")
    for index, (case, count, run, _) in enumerate(runs, start=1):
        if (int(case), int(count), int(run)) != (1, 1, index):
            raise ValueError(f"Unexpected per-run timing sequence at run {index}")
    seconds = [float(row[3]) for row in runs]
    load_seconds = float(loads[0])
    if any(not math.isfinite(x) or x < 0 for x in [load_seconds] + seconds):
        raise ValueError("Invalid non-finite or negative CLI timing")
    decimals = [len(row[3].partition(".")[2]) for row in runs]
    architecture = ARCH_RE.search(stderr)
    return {"outputs": outputs, "load_ms": load_seconds * 1000,
            "run_ms": [x * 1000 for x in seconds],
            "timing_resolution_ms": max(1000 * 10 ** (-digits) for digits in decimals),
            "metal_architecture": architecture[1] if architecture else None}


def without_confidence(value):
    if type(value) is dict:
        return {k: without_confidence(v) for k, v in value.items()
                if not (k == "confidence" and type(v) in (int, float))}
    if type(value) is list:
        return [without_confidence(v) for v in value]
    return value


def quality_report(reference, outputs, tolerance):
    exact_hashes = [canonical_sha(output) for output in outputs]
    discrete_reference = without_confidence(reference)
    within = same_discrete = 0
    maximum = 0.0
    differences = []
    for index, output in enumerate(outputs, start=1):
        comparison = compare_values(reference, output, tolerance=tolerance)
        discrete = compare_values(discrete_reference, without_confidence(output), tolerance=0)
        within += not comparison.errors
        same_discrete += not discrete.errors
        if comparison.max_confidence_delta is None:
            maximum = None
        elif maximum is not None:
            maximum = max(maximum, comparison.max_confidence_delta)
        if comparison.errors and len(differences) < 5:
            differences.append({"run": index, "differences": comparison.errors[:20],
                                "discrete_differences": discrete.errors[:20]})
    stable_comparisons = [compare_values(outputs[0], output, tolerance=tolerance) for output in outputs]
    return {
        "reference_configuration": BASELINE, "tolerance": tolerance,
        "reference_output_sha256": canonical_sha(reference),
        "representative_output_sha256": exact_hashes[0],
        "output_sha256_by_run": exact_hashes,
        "distinct_exact_outputs": len(set(exact_hashes)),
        "stable_exact": len(set(exact_hashes)) == 1,
        "stable_within_tolerance": all(not result.errors for result in stable_comparisons),
        "runs_within_reference_tolerance": within,
        "runs_with_identical_discrete_output": same_discrete,
        "compared_runs": len(outputs), "max_confidence_delta": maximum,
        "first_differences": differences,
    }


def timing_report(parsed, warmup):
    warm = parsed["run_ms"][warmup:]
    ordered = sorted(warm)
    return {
        "load_ms": parsed["load_ms"], "cold_first_ms": parsed["run_ms"][0],
        "warm_median_ms": statistics.median(warm),
        "warm_p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "warm_min_ms": min(warm), "warm_max_ms": max(warm),
        "warmup_runs": warmup, "measured_runs": len(warm), "repetitions": len(parsed["run_ms"]),
        "timing_resolution_ms": parsed["timing_resolution_ms"],
        "all_run_ms": parsed["run_ms"], "measured_run_ms": warm,
    }


def load_requests(path=None):
    requests = json.loads(Path(path).read_text()) if path else DEFAULT_REQUESTS
    if type(requests) is not list or not requests:
        raise ValueError("Requests must be a nonempty JSON array")
    seen = set()
    for request in requests:
        if (type(request) is not dict or type(request.get("name")) is not str
                or not request["name"] or request["name"] in seen
                or type(request.get("text")) is not str or not request["text"].strip()
                or type(request.get("entities")) is not list or not request["entities"]
                or any(type(label) is not str or not label or "," in label for label in request["entities"])):
            raise ValueError("Each request needs a unique name, nonempty text and nonempty comma-free entity labels")
        seen.add(request["name"])
    return requests


def run_benchmark(args):
    binary, model, destination = Path(args.binary).resolve(), Path(args.model).resolve(), Path(args.output).resolve()
    requests, device = load_requests(args.requests), machine_info()
    configs = configurations(args.include_cache20)
    warmup, repetitions = max(1, args.warmup), max(1, args.warmup) + args.runs
    report = {
        "status": "planned" if args.dry_run else "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "binary": str(binary), "model": str(model), "device": device,
        "requested_warmup_runs": args.warmup, "effective_warmup_runs": warmup,
        "measured_runs_per_request": args.runs,
        "method": {"new_process_per_configuration_and_request": True, "cold_first_includes_first_warmup": True,
                   "timing_scope": "CLI timer: preparation, inference, formatting and stdout serialization",
                   "p95_method": "nearest rank", "filesystem_cache_flushed": False,
                   "shader_cache_flushed": False, "quality_array_order": "strict"},
        "requests": [{**request, "whitespace_word_count": len(request["text"].split()),
                      "request_sha256": canonical_sha(request)} for request in requests],
        "configurations": [],
    }
    if not args.dry_run:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Executable not found: {binary}; build Release first")
        if not (model / "model.safetensors").is_file():
            raise ValueError(f"Checkpoint model.safetensors not found: {model}")
        help_result = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=30)
        for required_flag in ("--profile", "--fp32", "--no-tf32", "--cache-mb", "--retain-cache"):
            if help_result.returncode or required_flag not in help_result.stdout:
                raise ValueError(f"Binary does not advertise required option {required_flag}; rebuild the current Release CLI")
        report["binary_sha256"] = file_sha(binary)
        manifest = model / "checkpoint.json"
        if manifest.is_file():
            report["checkpoint_manifest"] = json.loads(manifest.read_text())
            report["checkpoint_manifest_sha256"] = file_sha(manifest)
        artifacts = destination.parent / (destination.stem + "-artifacts")
        artifacts.mkdir(parents=True, exist_ok=True)
    reference, baseline_medians = {}, {}
    architecture = None
    accurate_quality_passed = True
    execution_passed = True
    approximate_deviations = []
    for config in configs:
        entry = {**config, "results": []}
        if config["tf32"]:
            applicability = tf32_applicability(device, architecture)
            entry["tf32_applicability"] = applicability
            if applicability["status"] != "applicable":
                entry["status"] = applicability["status"]
                report["configurations"].append(entry)
                continue
        if args.dry_run:
            entry["status"] = "not_measured"
            entry["commands"] = [command_for(binary, model, config, request, repetitions) for request in requests]
            report["configurations"].append(entry)
            continue
        entry["status"] = "measured"
        for request in requests:
            command = command_for(binary, model, config, request, repetitions)
            environment = dict(os.environ, MLX_ENABLE_TF32="1" if config["tf32"] else "0")
            item = {"request": request["name"], "command": command}
            stem = re.sub(r"[^a-zA-Z0-9_-]", "_", config["name"] + "-" + request["name"])
            stdout_path, stderr_path = artifacts / (stem + ".jsonl"), artifacts / (stem + ".stderr.log")
            print(f"{config['name']}: {request['name']} ({repetitions} runs)", file=sys.stderr, flush=True)
            started = time.perf_counter()
            try:
                process = subprocess.run(command, env=environment, capture_output=True, text=True,
                                         timeout=args.timeout)
                item["process_wall_ms"] = (time.perf_counter() - started) * 1000
                stdout_path.write_text(process.stdout, encoding="utf-8")
                stderr_path.write_text(process.stderr, encoding="utf-8")
                item["stdout_artifact"], item["stderr_artifact"] = str(stdout_path), str(stderr_path)
                item["exit_code"] = process.returncode
                if process.returncode:
                    raise ValueError(f"CLI exited {process.returncode}: {process.stderr[-3000:]}")
                parsed = parse_run(process.stdout, process.stderr, repetitions)
                if parsed["metal_architecture"]:
                    architecture = parsed["metal_architecture"]
                    report["device"]["metal_architecture"] = architecture
                item.update(timing_report(parsed, warmup))
                outputs = parsed["outputs"]
                if config["name"] == BASELINE:
                    reference[request["name"]] = outputs[0]
                    item["reference_output"] = outputs[0]
                    baseline_medians[request["name"]] = item["warm_median_ms"]
                if request["name"] not in reference:
                    raise ValueError("Accurate baseline failed for this request; quality cannot be compared")
                baseline_ms = baseline_medians[request["name"]]
                item["warm_latency_ratio_to_accurate"] = item["warm_median_ms"] / baseline_ms if baseline_ms > 0 else None
                item["process_overhead_ms"] = max(0.0, item["process_wall_ms"] - item["load_ms"] - sum(item["all_run_ms"]))
                item["quality"] = quality_report(reference[request["name"]], outputs, args.tolerance)
                deviates = item["quality"]["runs_within_reference_tolerance"] != repetitions
                item["status"] = "quality_deviation" if deviates else "passed"
                if deviates and config["approximate"]:
                    approximate_deviations.append({"configuration": config["name"], "request": request["name"],
                                                   "same_discrete": item["quality"]["runs_with_identical_discrete_output"] == repetitions,
                                                   "max_confidence_delta": item["quality"]["max_confidence_delta"]})
                elif deviates:
                    accurate_quality_passed = False
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                item["status"] = "failed"; item["error"] = str(error)
                entry["status"] = "failed"; execution_passed = False
            entry["results"].append(item)
            # Preserve partial evidence if another request/process fails later.
            report["configurations"] = [e for e in report["configurations"] if e["name"] != entry["name"]] + [entry]
            write_report(destination, report)
        report["configurations"] = [e for e in report["configurations"] if e["name"] != entry["name"]] + [entry]
    report["tf32_applicability"] = tf32_applicability(device, architecture)
    if not args.dry_run:
        report["accurate_quality_passed"] = accurate_quality_passed
        report["execution_passed"] = execution_passed
        report["approximate_quality_deviations"] = approximate_deviations
        report["status"] = "complete" if execution_passed and accurate_quality_passed else "failed"
        if args.fail_on_approximate_deviation and approximate_deviations:
            report["status"] = "failed"
    write_report(destination, report)
    return report


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=str(ROOT / ".derivedData/Build/Products/Release/gliner-mlx"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", "--output-path", default=str(ROOT / ".models/mac-benchmark.json"))
    parser.add_argument("--runs", type=int, default=30, help="Measured runs after warmup (default: 30)")
    parser.add_argument("--warmup", type=int, default=5, help="Discarded runs, including cold first (default: 5; minimum effective 1)")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds allowed per configuration/request process")
    parser.add_argument("--requests", help="Optional JSON array of {name,text,entities}; default: three Italian texts")
    parser.add_argument("--include-cache20", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--fail-on-approximate-deviation", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write command plan without invoking the binary")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        parser.error("runs must be positive; warmup must be nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("timeout must be finite and positive; tolerance finite and nonnegative")
    try:
        report = run_benchmark(args)
    except (OSError, ValueError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(Path(args.output).resolve()), "status": report["status"],
                      "tf32_applicability": report["tf32_applicability"]}, sort_keys=True, allow_nan=False))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
