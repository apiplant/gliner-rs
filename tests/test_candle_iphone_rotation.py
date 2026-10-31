"""Offline checks for balanced runs and immutable result tags."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import run_candle_iphone_rotating as runner

class RotationTests(unittest.TestCase):
    def test_latin_square_and_refuse_existing_tag_before_device_actions(self):
        configurations = {'original', 'optimized', 'mlx'}
        self.assertEqual(len(runner.ORDER), 3)
        for row in runner.ORDER:
            self.assertEqual(set(row), configurations)
        for position in range(3):
            self.assertEqual({row[position] for row in runner.ORDER}, configurations)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model = out / 'gliner2.5-small-v1'
            model.mkdir(); (model / 'checkpoint.json').write_text('{}')
            refs = out / 'refs'; refs.mkdir()
            (refs / 'small-parity-cpu-reference.jsonl').write_text('{}\n')
            for request in json.loads((ROOT / 'docs/benchmarks/rust-vs-mlx.json').read_text())['requests']:
                (refs / f"small-{request['name']}-cpu-reference.jsonl").write_text('{}\n')
            (out / 'iphone-small-candle-benchmark-trial2-unique-optimized.results.jsonl').write_text('existing')
            argv = ['runner', 'device', '--variant', 'small', '--tag', 'unique', '--reference-dir', str(refs)]
            with patch.object(runner, 'OUT', out), patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run') as run:
                with self.assertRaisesRegex(SystemExit, 'Tag already exists'):
                    runner.main()
                run.assert_not_called()

if __name__ == '__main__': unittest.main()
