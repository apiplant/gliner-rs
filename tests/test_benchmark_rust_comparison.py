"""Offline checks for the CPU/GPU comparison; fake binaries never invoke MLX."""
from argparse import ArgumentTypeError, Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import benchmark_apple as apple
from scripts import benchmark_rust_comparison as comparison


class RustComparisonTests(unittest.TestCase):
    def source(self, root, binary):
        device = {"cpu": "Apple M3 Pro", "machine": "arm64", "macos": "27.2", "physical_memory_bytes": 123,
                  "logical_cpu_count": 12, "system": "Darwin"}
        model = root / "model"; model.mkdir(); (model / "model.safetensors").write_bytes(b"fake checkpoint")
        output = {"entities": {"person": [{"text": "Michele", "start": 0, "end": 7, "confidence": 0.9}]}}
        request = {"name": "test", "text": "Michele", "entities": ["person"]}
        request.update(request_sha256=apple.canonical_sha(request), whitespace_word_count=1)
        source = {"model": str(model), "binary": str(binary), "device": device,
                  "requests": [request], "configurations": [{"name": apple.BASELINE, "precision": "float32", "tf32": False,
                  "results": [{"request": "test", "status": "passed", "reference_output": output,
                               "warm_median_ms": 1.0, "warm_p95_ms": 1.2}]}]}
        path = root / "gpu.json"; path.write_text(json.dumps(source))
        return path, device, source

    def test_thread_counts_require_positive_unique_values(self):
        self.assertEqual(comparison.thread_counts("8,4"), [8, 4])
        for text in ("", "0,4", "8,8", "x,4"):
            with self.assertRaises(ArgumentTypeError):
                comparison.thread_counts(text)

    def test_source_rejects_nonfinite_baselines_and_changed_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, source = self.source(root, root / "binary")
            comparison.baseline_from_report(path)
            changed = copy.deepcopy(source); changed["requests"][0]["text"] = "Other"
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                comparison.baseline_from_report(path)
            source["configurations"][0]["results"][0]["reference_output"]["bad"] = float("nan")
            path.write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                comparison.baseline_from_report(path)

    def test_default_and_tuned_thread_results_remain_distinct(self):
        report = {"requests": [{"name": "a", "whitespace_word_count": 64}],
                  "mlx_accurate": {"results": [{"request": "a", "warm_median_ms": 20, "warm_p95_ms": 22}]},
                  "rust_configurations": [
                      {"threads": 8, "matches_rust_cli_default": True, "results": [{"request": "a", "status": "passed", "warm_median_ms": 200, "warm_p95_ms": 210}]},
                      {"threads": 4, "matches_rust_cli_default": False, "results": [{"request": "a", "status": "passed", "warm_median_ms": 180, "warm_p95_ms": 205}]}]}
        result = comparison.summarize(report)[0]
        self.assertEqual((result["rust_cpu_default_threads"], result["rust_cpu_best_threads"]), (8, 4))
        self.assertEqual(result["rust_default_cpu_to_mlx_gpu_latency_ratio"], 10)
        self.assertEqual(result["rust_best_cpu_to_mlx_gpu_latency_ratio"], 9)
        report["mlx_accurate"]["results"][0]["warm_median_ms"] = 0
        self.assertIsNone(comparison.summarize(report)[0]["rust_best_cpu_to_mlx_gpu_latency_ratio"])

    def test_fake_cpu_benchmark_has_fresh_processes_threads_and_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); binary = root / "fake-rust"
            binary.write_text('''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
assert "--timing" in args
threads=int(os.environ["RAYON_NUM_THREADS"])
assert all(os.environ[name]==str(threads) for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"))
case=json.load(open(args[1]))[0]
assert case["text"]=="Michele" and case["entities"]==["person"]
print("Loaded in 0.100000 s",file=sys.stderr)
for i in range(1,int(args[args.index("--repeat")+1])+1):
 print(json.dumps({"entities":{"person":[{"text":"Michele","start":0,"end":7,"confidence":0.9}]}}))
 print(f"Case 1/1, run {i}: {threads/1000:.6f} s",file=sys.stderr)
''')
            binary.chmod(0o755)
            path, device, _ = self.source(root, binary)
            args = Namespace(mlx_report=str(path), output=str(root / "comparison.json"), model=None,
                rust_binary=str(binary), mlx_binary=None, refresh_mlx=False, threads=[8, 4],
                warmup=2, runs=3, tolerance=1e-4, timeout=10, rust_build_command="fake test executable")
            with patch.object(apple, "machine_info", return_value=device), patch.object(comparison, "link_dependencies", return_value=[]), patch.object(comparison, "thread_context", return_value={}):
                report = comparison.run_benchmark(args)
            self.assertEqual(report["status"], "complete")
            self.assertEqual([config["threads"] for config in report["rust_configurations"]], [8, 4])
            self.assertEqual([config["matches_rust_cli_default"] for config in report["rust_configurations"]], [True, False])
            self.assertEqual(report["comparisons"][0]["rust_cpu_best_threads"], 4)
            for config in report["rust_configurations"]:
                result = config["results"][0]
                self.assertEqual((result["repetitions"], result["measured_runs"]), (5, 3))
                self.assertEqual(result["quality_vs_mlx_accurate"]["runs_within_reference_tolerance"], 5)
                self.assertTrue(Path(result["stdout_artifact"]).is_file())
            self.assertIn("not a language performance comparison", report["comparison"])
            self.assertEqual(json.loads(Path(args.output).read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
