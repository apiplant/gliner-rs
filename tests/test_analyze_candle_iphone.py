"""Offline checks for phone benchmark provenance and output comparisons."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent

class PhoneAnalysisTests(unittest.TestCase):
    def test_valid_trial_and_wrong_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            names = [item['name'] for item in json.loads((ROOT / 'docs/benchmarks/rust-vs-mlx.json').read_text())['requests']]
            value = {'entities': {'person': [{'text': 'Alice', 'start': 0, 'end': 5, 'confidence': 0.9}]}}
            parity = directory / 'parity.jsonl'
            parity.write_text((json.dumps(value) + '\n') * 28)
            for name in names:
                (directory / f'small-{name}-cpu-reference.jsonl').write_text(json.dumps(value) + '\n')
            for engine in ('mlx', 'candle'):
                p = directory / f'iphone-small-{engine}-parity-trial1'
                Path(str(p) + '.results.jsonl').write_text(parity.read_text())
                Path(str(p) + '.summary.json').write_text(json.dumps({'status': 'passed', 'backend': engine, 'precision': 'float32', 'caseCount': 28, 'runID': 'parity', 'loadSeconds': 1, 'peakResidentBytes': 10}))
                for trial in range(1, 4):
                    p = directory / f'iphone-small-{engine}-benchmark-trial{trial}'
                    run_id = f'{engine}-{trial}'
                    Path(str(p) + '.summary.json').write_text(json.dumps({'status':'passed','backend':engine,'precision':'float32','runID':run_id,'loadSeconds':1,'peakResidentBytes':10}))
                    rows = [{'runID':run_id,'backend':engine,'precision':'float32','caseIndex':case,'iteration':i,'warmup':i < 5,'seconds':0.01,'result':value} for case in range(3) for i in range(35)]
                    Path(str(p) + '.results.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            command = [sys.executable, str(ROOT / 'scripts/analyze_candle_iphone.py'), '--variant', 'small', '--artifacts', str(directory), '--parity-reference', str(parity), '--benchmark-reference-dir', str(directory)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((directory / 'iphone-small-candle-vs-mlx.json').read_text())
            self.assertEqual(result['direct_backend_comparison']['benchmark']['strict'], 315)
            target = directory / 'iphone-small-candle-benchmark-trial2.results.jsonl'
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            rows[0]['runID'] = 'stale'
            target.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            invalid = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('invalid run metadata', invalid.stderr)
            rows[0]['runID'] = 'candle-2'
            rows[0]['seconds'] = float('nan')
            target.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            invalid_time = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(invalid_time.returncode, 0)
            self.assertIn('request seconds', invalid_time.stderr)
            rows[0]['seconds'] = 0.01
            rows[0]['result'] = {'entities': {'person': []}}
            target.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            mismatch = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0)

if __name__ == '__main__':
    unittest.main()
