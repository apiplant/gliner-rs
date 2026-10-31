#!/usr/bin/env python3
"""Validate and summarize the raw output of run_candle_iphone_suite.py."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from compare_apple_results import compare_values

ROOT = Path(__file__).resolve().parent.parent

def readlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def compare(left, right, allow_near_ties=False):
    strict = compare_values(left, right, tolerance=1e-4)
    selected = compare_values(left, right, tolerance=1e-4, allow_near_ties=True, tie_tolerance=1e-6) if allow_near_ties else strict
    if selected.errors:
        raise ValueError('; '.join(selected.errors[:4]))
    return not strict.errors, selected.max_confidence_delta, len(selected.reorders)

def positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{field}: expected a positive finite value')
    return value

def p95(values):
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['small', 'multi'], required=True)
    p.add_argument('--artifacts', type=Path, default=ROOT / '.models')
    p.add_argument('--parity-reference', type=Path, required=True)
    p.add_argument('--benchmark-reference-dir', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    requests_file = ROOT / 'docs/benchmarks/rust-vs-mlx.json'
    requests = json.loads(requests_file.read_text())['requests']
    parity_gold = readlines(args.parity_reference)
    if len(parity_gold) != 28 or len(requests) != 3:
        raise SystemExit('Expected 28 parity cases and 3 benchmark requests')
    reference_paths = [args.benchmark_reference_dir / f"{args.variant}-{r['name']}-cpu-reference.jsonl" for r in requests]
    benchmarks = [readlines(path) for path in reference_paths]
    if any(len(v) != 1 for v in benchmarks):
        raise SystemExit('Each CPU benchmark reference must have one result')
    result = {'variant': args.variant, 'precision': 'float32', 'method': {'warmups': 5, 'measured': 30, 'trials': 3,
        'order': [['mlx','candle'],['candle','mlx'],['mlx','candle']], 'fresh_process_per_trial': True,
        'cases_per_process': 3, 'mlx_cache_bytes': 20 * 1024 * 1024, 'mlx_clear_after_request': True,
        'candle_device': 'Metal device 0', 'timing': 'monotonic, request including backend boundary and output conversion'},
        'input_sha256': sha(requests_file), 'parity_reference_sha256': sha(args.parity_reference),
        'benchmark_reference_sha256': {r['name']: sha(path) for r,path in zip(requests,reference_paths)}, 'engines': {}}
    parity_values = {}
    benchmark_values = {}
    for engine in ('mlx','candle'):
        engine_data = {'parity': {}, 'trials': [], 'cases': []}
        parity_prefix = args.artifacts / f'iphone-{args.variant}-{engine}-parity-trial1'
        parity_file = Path(str(parity_prefix) + '.results.jsonl')
        parity_summary = Path(str(parity_prefix) + '.summary.json')
        values = readlines(parity_file)
        parity_meta = json.loads(parity_summary.read_text())
        if parity_meta.get('status') != 'passed' or parity_meta.get('backend') != engine or parity_meta.get('precision') != 'float32' or parity_meta.get('caseCount') != 28:
            raise ValueError(f'{parity_summary}: incorrect parity metadata')
        if len(values) != 28: raise ValueError(f'{parity_file}: expected 28 results')
        positive(parity_meta.get('loadSeconds'), 'parity loadSeconds')
        positive(parity_meta.get('peakResidentBytes'), 'parity peakResidentBytes')
        parity_values[engine] = values
        comparisons = [compare(a,b, allow_near_ties=True) for a,b in zip(parity_gold, values)]
        engine_data['parity'] = {'count': len(values), 'runID': parity_meta.get('runID'), 'load_seconds': parity_meta.get('loadSeconds'), 'peak_resident_bytes': parity_meta.get('peakResidentBytes'), 'strict': sum(x[0] for x in comparisons),
            'near_tie_reorders': sum(x[2] for x in comparisons), 'max_confidence_delta': max(x[1] or 0 for x in comparisons),
            'results_sha256': sha(parity_file), 'summary_sha256': sha(parity_summary)}
        pooled = [[] for _ in requests]
        for trial in range(1,4):
            prefix = args.artifacts / f'iphone-{args.variant}-{engine}-benchmark-trial{trial}'
            results_path = Path(str(prefix) + '.results.jsonl'); summary_path = Path(str(prefix) + '.summary.json')
            rows = readlines(results_path); summary = json.loads(summary_path.read_text())
            benchmark_values[(engine, trial)] = {(row['caseIndex'], row['iteration']): row['result'] for row in rows}
            if summary.get('status') != 'passed' or summary.get('backend') != engine or summary.get('precision') != 'float32' or len(rows) != 105:
                raise ValueError(f'{prefix}: incomplete or incorrect trial')
            positive(summary.get('loadSeconds'), 'loadSeconds')
            positive(summary.get('peakResidentBytes'), 'peakResidentBytes')
            seen = set(); comparisons = []; per_case = [[] for _ in requests]
            for row in rows:
                case, iteration = row['caseIndex'], row['iteration']
                if not isinstance(case,int) or not isinstance(iteration,int) or not 0 <= case < 3 or not 0 <= iteration < 35 or (case,iteration) in seen:
                    raise ValueError(f'{prefix}: invalid case/iteration')
                seen.add((case,iteration))
                if row.get('runID') != summary.get('runID') or row.get('backend') != engine or row.get('precision') != 'float32' or row.get('warmup') != (iteration < 5):
                    raise ValueError(f'{prefix}: invalid run metadata')
                comparisons.append(compare(benchmarks[case][0], row['result']))
                positive(row.get('seconds'), 'request seconds')
                if iteration >= 5:
                    per_case[case].append(row['seconds'] * 1000); pooled[case].append(row['seconds'] * 1000)
            engine_data['trials'].append({'trial': trial, 'runID': summary['runID'], 'load_seconds': summary['loadSeconds'],
                'peak_resident_bytes': summary.get('peakResidentBytes'), 'strict': sum(x[0] for x in comparisons),
                'near_tie_reorders': sum(x[2] for x in comparisons), 'max_confidence_delta': max(x[1] or 0 for x in comparisons),
                'results_sha256': sha(results_path), 'summary_sha256': sha(summary_path),
                'case_medians_ms': [statistics.median(v) for v in per_case]})
        engine_data['cases'] = [{'name': r['name'], 'count': len(v), 'median_ms': statistics.median(v), 'p95_ms': p95(v)} for r,v in zip(requests,pooled)]
        result['engines'][engine] = engine_data
    direct_parity = [compare(a, b) for a,b in zip(parity_values['mlx'], parity_values['candle'])]
    direct_bench = []
    for trial in range(1,4):
        a, b = benchmark_values[('mlx', trial)], benchmark_values[('candle', trial)]
        if a.keys() != b.keys(): raise ValueError(f'trial {trial}: backend case/iteration mismatch')
        direct_bench.extend(compare(a[key], b[key]) for key in sorted(a))
    def direct_summary(values):
        return {'count': len(values), 'strict': sum(x[0] for x in values),
                'near_tie_reorders': sum(x[2] for x in values),
                'max_confidence_delta': max((x[1] or 0 for x in values), default=0)}
    result['direct_backend_comparison'] = {'parity': direct_summary(direct_parity), 'benchmark': direct_summary(direct_bench)}
    output = args.output or args.artifacts / f'iphone-{args.variant}-candle-vs-mlx.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(output), 'engines': {k: {'parity': v['parity'], 'cases': v['cases']} for k,v in result['engines'].items()}}, indent=2))

if __name__ == '__main__':
    main()
