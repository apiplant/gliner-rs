"""Offline benchmark checks use a fake CLI; no MLX or GPU inference is run."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import benchmark_apple as benchmark


class AppleBenchmarkTests(unittest.TestCase):
    def logs(self, times=("0.001", "0.002", "0.003")):
        stderr = "Metal architecture: applegpu_g15s\nLoaded in 0.125000 s\n"
        stderr += "\n".join(f"Case 1/1, run {i}: {value} s" for i, value in enumerate(times, 1))
        output = {"entities": {"person": [{"text": "Michele", "start": 0, "end": 7, "confidence": 0.9}]}}
        return "\n".join(json.dumps(output) for _ in times), stderr

    def test_parses_timings_and_reports_real_printed_resolution(self):
        stdout, stderr = self.logs()
        parsed = benchmark.parse_run(stdout, stderr, 3)
        self.assertEqual(parsed["run_ms"], [1, 2, 3])
        self.assertEqual(parsed["load_ms"], 125)
        self.assertEqual(parsed["timing_resolution_ms"], 1)
        report = benchmark.timing_report(parsed, 1)
        self.assertEqual(report["warm_median_ms"], 2.5)
        self.assertEqual(report["warm_p95_ms"], 3)
        self.assertEqual(report["cold_first_ms"], 1)
        stdout, stderr = self.logs(("0.001234", "0.001876"))
        self.assertEqual(benchmark.parse_run(stdout, stderr, 2)["timing_resolution_ms"], 0.001)

    def test_missing_duplicate_and_nonfinite_results_fail(self):
        stdout, stderr = self.logs()
        for a, b in ((stdout.splitlines()[0], stderr), (stdout, stderr.replace("run 2:", "run 1:")),
                     (stdout, stderr.replace("0.002 s", "nan s")),
                     (stdout.replace("0.9", "NaN"), stderr),
                     (stdout.replace("0.9", "1e999"), stderr)):
            with self.assertRaises(ValueError):
                benchmark.parse_run(a, b, 3)

    def test_tf32_skipped_for_m3_and_gated_for_new_hardware(self):
        device = {"system": "Darwin", "macos": "26.2", "cpu": "Apple M3 Pro"}
        self.assertEqual(benchmark.tf32_applicability(device)["status"], "not_applicable")
        self.assertEqual(benchmark.tf32_applicability(device, "applegpu_g15s")["status"], "not_applicable")
        self.assertEqual(benchmark.tf32_applicability(device, "applegpu_g17s")["status"], "applicable")
        self.assertEqual(benchmark.tf32_applicability(device, "applegpu_g17p")["status"], "not_applicable")
        self.assertEqual(benchmark.tf32_applicability(device, "applegpu_g18p")["status"], "applicable")
        device["macos"] = "26.1"
        self.assertEqual(benchmark.tf32_applicability(device, "applegpu_g18p")["status"], "not_applicable")
        device.update(macos="26.2", cpu="Unknown")
        self.assertEqual(benchmark.tf32_applicability(device)["status"], "not_measured")

    def test_quality_reports_discrete_fp16_changes_separately(self):
        baseline = {"entities": {"person": [{"text": "Michele", "start": 0, "end": 7, "confidence": 0.9}]}}
        changed_confidence = copy.deepcopy(baseline)
        changed_confidence["entities"]["person"][0]["confidence"] = 0.85
        quality = benchmark.quality_report(baseline, [changed_confidence], 1e-4)
        self.assertEqual(quality["runs_within_reference_tolerance"], 0)
        self.assertEqual(quality["runs_with_identical_discrete_output"], 1)
        changed_span = copy.deepcopy(changed_confidence)
        changed_span["entities"]["person"][0]["start"] = 1
        quality = benchmark.quality_report(baseline, [baseline, changed_span], 1e-4)
        self.assertEqual(quality["runs_with_identical_discrete_output"], 1)
        self.assertFalse(quality["stable_within_tolerance"])
        self.assertTrue(quality["first_differences"][0]["discrete_differences"])

    def test_command_uses_explicit_independent_switches(self):
        request = benchmark.DEFAULT_REQUESTS[0]
        configs = {config["name"]: config for config in benchmark.configurations(True)}
        cold = benchmark.command_for("binary", "model", configs["accurate_f32_no_cache"], request, 35)
        self.assertIn("--fp32", cold); self.assertIn("--no-tf32", cold); self.assertIn("--clear-cache", cold)
        self.assertEqual(cold[cold.index("--cache-mb") + 1], "0")
        fast = benchmark.command_for("binary", "model", configs["fast_fp16_auto"], request, 35)
        self.assertIn("--fp16", fast); self.assertIn("--retain-cache", fast)
        self.assertEqual(fast[fast.index("--cache-mb") + 1], "auto")
        self.assertEqual(fast[fast.index("--repeat") + 1], "35")

    def test_fake_cli_end_to_end_repeats_quality_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); binary = root / "fake-cli"; model = root / "model"
            model.mkdir(); (model / "model.safetensors").write_bytes(b"test only")
            requests = root / "requests.json"
            requests.write_text(json.dumps([{"name": "test", "text": "Michele", "entities": ["person"]}]))
            binary.write_text('''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
if "--help" in args:
 print("--profile --fp32 --no-tf32 --cache-mb --retain-cache")
 sys.exit(0)
assert os.environ["MLX_ENABLE_TF32"] == ("1" if "--tf32" in args else "0")
print("Metal architecture: applegpu_g15s", file=sys.stderr)
print("Loaded in 0.010000 s", file=sys.stderr)
for i in range(1, int(args[args.index("--repeat")+1])+1):
 start = 1 if "--fp16" in args else 0
 output = {"entities":{"person":[{"text":"Michele","start":start,"end":7,"confidence":0.85 if start else 0.9}]}}
 print(json.dumps(output))
 print(f"Case 1/1, run {i}: {i/1000:.6f} s", file=sys.stderr)
''')
            binary.chmod(0o755)
            args = Namespace(binary=str(binary), model=str(model), output=str(root / "report.json"),
                requests=str(requests), warmup=2, runs=3, include_cache20=False, dry_run=False,
                timeout=10, tolerance=1e-4, fail_on_approximate_deviation=False)
            device = {"system": "Darwin", "macos": "26.2", "cpu": "Apple M3 Pro"}
            with patch.object(benchmark, "machine_info", return_value=device):
                report = benchmark.run_benchmark(args)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["accurate_quality_passed"])
            self.assertEqual(len(report["approximate_quality_deviations"]), 2)
            self.assertTrue(all(not item["same_discrete"] for item in report["approximate_quality_deviations"]))
            self.assertEqual(report["configurations"][-1]["status"], "not_applicable")
            measured = report["configurations"][0]["results"][0]
            self.assertEqual((measured["warmup_runs"], measured["measured_runs"], measured["repetitions"]), (2, 3, 5))
            self.assertEqual(measured["warm_median_ms"], 4)
            self.assertEqual(measured["warm_p95_ms"], 5)
            self.assertTrue(Path(measured["stdout_artifact"]).is_file())
            self.assertTrue(Path(measured["stderr_artifact"]).is_file())
            self.assertEqual(json.loads(Path(args.output).read_text())["status"], "complete")

    def test_dry_run_never_starts_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(binary="/not/real", model="/not/real", output=str(Path(directory) / "plan.json"),
                requests=None, warmup=5, runs=30, include_cache20=True, dry_run=True,
                timeout=10, tolerance=1e-4, fail_on_approximate_deviation=False)
            with patch.object(benchmark, "machine_info", return_value={"system": "Darwin", "cpu": "Apple M3 Pro", "macos": "26.2"}), patch.object(benchmark.subprocess, "run", side_effect=AssertionError("Binary must not execute")):
                report = benchmark.run_benchmark(args)
            self.assertEqual(report["status"], "planned")
            self.assertEqual(len(report["requests"]), 3)
            self.assertTrue(all(60 <= request["whitespace_word_count"] <= 150 for request in report["requests"]))
            self.assertEqual(report["configurations"][0]["commands"][0][-1], "--retain-cache")


if __name__ == "__main__":
    unittest.main()
