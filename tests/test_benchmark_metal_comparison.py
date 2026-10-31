import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import benchmark_metal_comparison as bench


class BenchmarkMetalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(__import__("tempfile").TemporaryDirectory())
        self.root = Path(self.temp)
        self.rust = self.root / "apple_reference"
        self.mlx = self.root / "gliner-mlx"
        for binary in (self.rust, self.mlx):
            binary.write_text("fake executable")
            binary.chmod(0o755)
        self.requests = [
            {"name": f"request_{i}", "text": f"Input {i}.", "entities": ["person"]}
            for i in range(3)
        ]
        for variant in ("small", "base", "multi"):
            directory = self.root / ".models" / f"gliner2.5-{variant}-v1"
            directory.mkdir(parents=True)
            (directory / "checkpoint.json").write_text(json.dumps({"variant": variant}))
        for relative in (
            "Cargo.toml", "Cargo.lock", "examples/apple_reference.rs", "src/deberta.rs",
            "src/model.rs", "swift/Sources/GLiNERCLI/main.swift",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source")
        self.args = SimpleNamespace(
            output=str(self.root / "result.json"), rust_binary=str(self.rust), mlx_binary=str(self.mlx),
            requests=None, variants=["small"], trials=3, warmup=1, runs=2,
            tolerance=1e-4, timeout=1,
        )
        self.commands = []
        self.mode = "ok"

    def fake_subprocess_run(self, argv, env, capture_output, text, timeout):
        command = argv[2:]
        self.commands.append(command)
        metal = "--metal" in command
        inputs_only = "--inputs-only" in command
        swift = "--model" in command
        if inputs_only:
            ids = [1, 2] if not (self.mode == "input-mismatch" and swift) else [1, 3]
            stdout = json.dumps({"input_ids": ids, "word_positions": [1]}) + "\n"
            stderr = ""
        else:
            repetitions = 1
            if "--repeat" in command:
                repetitions = int(command[command.index("--repeat") + 1])
            output = {"entities": {"person": []}}
            if self.mode == "output-mismatch" and metal:
                output = {"entities": {"person": [{"text": "wrong", "start": 0, "end": 5, "confidence": 0.9}]}}
            stdout = "".join(json.dumps(output) + "\n" for _ in range(repetitions))
            stderr = "Loaded in 0.001000 s\n"
            if metal and self.mode != "no-metal-confirmation":
                stderr = "Backend: Candle Metal GPU / F32\n" + stderr
            stderr += "".join(f"Case 1/1, run {i}: 0.002000 s\n" for i in range(1, repetitions + 1))
        stderr += "123 maximum resident set size\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)

    def execute(self, mode="ok"):
        self.mode = mode
        with patch.object(bench, "ROOT", self.root), \
             patch.object(bench.apple, "load_requests", return_value=[dict(x) for x in self.requests]), \
             patch.object(bench.apple, "machine_info", return_value={"machine": "test"}), \
             patch.object(bench.apple, "file_sha", return_value="fake-sha256"), \
             patch.object(bench.subprocess, "run", side_effect=self.fake_subprocess_run):
            return bench.benchmark(self.args)

    def read_report(self):
        return json.loads((self.root / "result.json").read_text())

    def test_output_mismatch_marks_report_failed_without_completion(self):
        with self.assertRaisesRegex(ValueError, "Output parity failed"):
            self.execute("output-mismatch")
        report = self.read_report()
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("completed_at_utc", report)
        self.assertEqual(report["runs"][-1]["backend"], "candle_metal")
        self.assertEqual(report["runs"][-1]["quality"]["runs_within_reference_tolerance"], 0)

    def test_missing_metal_device_confirmation_fails_before_recording_run(self):
        with self.assertRaisesRegex(ValueError, "explicitly confirm the Metal device"):
            self.execute("no-metal-confirmation")
        report = self.read_report()
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("completed_at_utc", report)
        self.assertEqual(report["runs"], [])

    def test_mismatched_prepared_ids_fail_before_any_timed_run(self):
        with self.assertRaisesRegex(ValueError, "Prepared inputs differ"):
            self.execute("input-mismatch")
        report = self.read_report()
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("completed_at_utc", report)
        self.assertEqual(report["runs"], [])
        self.assertFalse(any("--metal" in command for command in self.commands))

    def test_backend_order_alternates_across_three_trials_and_requests(self):
        report = self.execute()
        self.assertEqual(report["status"], "complete")
        self.assertIn("completed_at_utc", report)
        runs = report["runs"]
        self.assertEqual(len(runs), 3 * 3 * 2)
        for trial in range(3):
            for index, request in enumerate(self.requests):
                pair = [run["backend"] for run in runs
                        if run["trial"] == trial + 1 and run["request"] == request["name"]]
                self.assertEqual(pair, bench.pair_order(trial * len(self.requests) + index))


if __name__ == "__main__":
    unittest.main()
