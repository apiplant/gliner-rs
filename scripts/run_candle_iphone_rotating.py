#!/usr/bin/env python3
"""F32 iPhone parity and balanced three-way latency/memory comparison.

Requires installed Candle-enabled demo, pinned model(s), and CPU references.
Each trial uses a fresh process per configuration. Device data remains local.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '.models'
APP = 'com.example.GLiNERDemo'
ORDER = [('original','optimized','mlx'), ('optimized','mlx','original'), ('mlx','original','optimized')]

def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('device')
    parser.add_argument('--variant', choices=['small','multi'], required=True)
    parser.add_argument('--tag', default='rotating', help='Distinct tag; existing artifacts are never overwritten')
    parser.add_argument('--reuse-model', action='store_true')
    parser.add_argument('--reference-dir', type=Path, default=OUT / 'iphone-rust-references')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag): parser.error('Invalid artifact tag')
    requests = json.loads((ROOT/'docs/benchmarks/rust-vs-mlx.json').read_text())['requests']
    parity_ref = args.reference_dir / f'{args.variant}-parity-cpu-reference.jsonl'
    references = [parity_ref] + [args.reference_dir / f"{args.variant}-{r['name']}-cpu-reference.jsonl" for r in requests]
    if not all(p.is_file() for p in references):
        raise SystemExit('Missing CPU reference files; run the normal suite once to generate them')
    model = OUT / f'gliner2.5-{args.variant}-v1'
    if not (model/'checkpoint.json').is_file(): raise SystemExit('Missing local pinned model checkpoint')
    tags = {k: f'{args.tag}-{k}' for k in ('original','optimized','mlx')}
    # Preserve prior runs rather than presenting stale artifacts as new evidence.
    expected = [OUT/f'iphone-{args.variant}-{engine}-{mode}-trial{trial}-{tags[key]}.{suffix}'
                for key,engine in [('original','candle'),('optimized','candle'),('mlx','mlx')]
                for mode,trials in [('parity',[1]),('benchmark',[1,2,3])]
                for trial in trials for suffix in ('summary.json','results.jsonl')]
    expected += [OUT/f'iphone-{args.variant}-mlx-{mode}-trial{trial}-{tags[config]}.{suffix}'
                 for config in ('original','optimized') for mode,trials in [('parity',[1]),('benchmark',[1,2,3])]
                 for trial in trials for suffix in ('summary.json','results.jsonl')]
    expected += [OUT/f'iphone-{args.variant}-candle-vs-mlx-{tags[config]}.json' for config in ('original','optimized')]
    expected += [OUT/f'iphone-{args.variant}-candle-rotating-{args.tag}.json']
    if any(p.exists() for p in expected): raise SystemExit('Tag already exists; choose a new --tag')
    if not args.reuse_model:
        subprocess.run(['xcrun','devicectl','device','copy','to','--device',args.device,'--source',str(model),
                        '--destination','Documents/self-test-model','--domain-type','appDataContainer',
                        '--domain-identifier',APP,'--timeout','240'], cwd=ROOT, check=True)
    def run(config, mode, trial):
        engine = 'mlx' if config == 'mlx' else 'candle'
        command = [sys.executable,'scripts/verify_candle_iphone.py',args.device,'--backend',engine,
                   '--variant',args.variant,'--mode',mode,'--trial',str(trial),'--tag',tags[config], '--memory-trace']
        if config == 'original': command += ['--metal-loader','mmap','--metal-original-attention']
        elif config == 'optimized': command += ['--metal-loader','stream']
        print(f'{args.variant}: {mode} {trial}, {config}',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    for config in ('original','optimized','mlx'): run(config,'parity',1)
    for trial, order in enumerate(ORDER,1):
        for config in order: run(config,'benchmark',trial)
    # Analyzer expects an MLX artifact next to each Candle comparison. These
    # are byte-for-byte copies of the same process/UUID, not extra samples.
    for mode,trials in [('parity',[1]),('benchmark',[1,2,3])]:
        for trial in trials:
            source = OUT/f'iphone-{args.variant}-mlx-{mode}-trial{trial}-{tags["mlx"]}'
            for config in ('original','optimized'):
                dest = OUT/f'iphone-{args.variant}-mlx-{mode}-trial{trial}-{tags[config]}'
                for ext in ('.summary.json','.results.jsonl'):
                    src,dst = Path(str(source)+ext),Path(str(dest)+ext)
                    if dst.exists(): raise SystemExit(f'Refusing to overwrite {dst}')
                    shutil.copy2(src,dst)
    comparisons = {}
    for config in ('original','optimized'):
        output = OUT/f'iphone-{args.variant}-candle-vs-mlx-{tags[config]}.json'
        command = [sys.executable,'scripts/analyze_candle_iphone.py','--variant',args.variant,
                   '--parity-reference',str(parity_ref),'--benchmark-reference-dir',str(args.reference_dir),
                   '--tag',tags[config], '--expect-metal-loader', 'mmap' if config == 'original' else 'stream',
                   '--expect-original-attention', 'yes' if config == 'original' else 'no', '--expect-load-pool','off']
        subprocess.run(command,cwd=ROOT,check=True)
        data = json.loads(output.read_text())
        data['method']['order'] = ORDER
        data['method']['mlx_shared_between_comparisons'] = True
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
        comparisons[config] = {'path': str(output), 'sha256': sha(output)}
    shared = []
    for trial in (1,2,3):
        path = OUT/f'iphone-{args.variant}-mlx-benchmark-trial{trial}-{tags["mlx"]}.summary.json'
        shared.append(json.loads(path.read_text())['runID'])
    manifest = {'variant':args.variant,'tag':args.tag,'trial_order':ORDER,'warmup_per_case':5,'measured_per_case':30,
                'cases_per_process':3,'fresh_process_per_configuration':True,'mlx_shared_between_comparisons':True,
                'shared_mlx_trial_run_ids':shared,'parity_reference_sha256':sha(parity_ref),
                'benchmark_reference_sha256':{r['name']:sha(p) for r,p in zip(requests,references[1:])},
                'comparisons':comparisons}
    manifest_path = OUT/f'iphone-{args.variant}-candle-rotating-{args.tag}.json'
    if manifest_path.exists(): raise SystemExit(f'Refusing to overwrite {manifest_path}')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Balanced comparison saved: {manifest_path}')

if __name__ == '__main__': main()
