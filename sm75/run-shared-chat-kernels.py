#!/usr/bin/env python3
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

root = Path(__file__).resolve().parent.parent
out = root / ('kernel-results-' + time.strftime('%Y%m%d-%H%M%S'))
out.mkdir()
binary = root / 'build/bin/test-backend-ops'
env = dict(os.environ, GGML_TEST_SWIGLU_MMQ='1')
active = subprocess.run(['systemctl', 'is-active', '--quiet', 'cmp-idle-governor.service']).returncode == 0
records = []

def run(mode, enabled, label):
    current_env = dict(env, GGML_CUDA_SM75_SWIGLU_Q8_1=str(enabled))
    if mode == 'test' and enabled:
        current_env['GGML_CUDA_SM75_SWIGLU_Q8_1_TRACE'] = '1'
    command = [str(binary), mode, '-b', 'CUDA0', '-o', 'SWIGLU_MMQ']
    before = subprocess.check_output(['nvidia-smi', '--query-gpu=pstate,clocks.sm,clocks.mem,temperature.gpu,power.limit,memory.used', '--format=csv,noheader'], text=True)
    result = subprocess.run(command, env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out / (label + '.log')).write_text(result.stdout)
    (out / (label + '-state.txt')).write_text(before)
    if result.returncode != 0 or 'FAIL' in result.stdout:
        raise RuntimeError(f'{label} failed: see {out}')
    if mode == 'test':
        if '50/50 tests passed' not in result.stdout:
            raise RuntimeError(f'{label}: expected 50/50 test summary')
        if enabled and 'SM75_SWIGLU_Q8_1 type=' not in result.stdout:
            raise RuntimeError(f'{label}: fusion did not run')
        print(label, '50/50 passed', flush=True)
        return
    rows = re.findall(r'SWIGLU_MMQ\(([^\n]*?)\):.*?([0-9.]+) us/run', result.stdout, re.DOTALL)
    if len(rows) != 18:
        raise RuntimeError(f'{label}: expected 18 timings, got {len(rows)}')
    for shape, us in rows:
        records.append(dict(label=label, enabled=enabled, shape=shape, us=float(us)))
    (out / 'records.json').write_text(json.dumps(records, indent=2))
    print(label, '18 timings', flush=True)

try:
    for enabled in [0, 1]:
        run('test', enabled, f'correctness-{enabled}')
    if active:
        subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'cmp-idle-governor.service'], check=True)
    for enabled in [0, 1]:
        run('perf', enabled, f'warmup-{enabled}')
    rng = random.Random(20260909)
    for repeat in range(5):
        order = [0, 1]
        rng.shuffle(order)
        for enabled in order:
            run('perf', enabled, f'run-{repeat}-{enabled}')
    summary = []
    for shape in sorted({r['shape'] for r in records}):
        row = dict(shape=shape)
        for enabled in [0, 1]:
            values = [r['us'] for r in records if r['shape'] == shape and r['enabled'] == enabled and r['label'].startswith('run-')]
            row[str(enabled)] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
        row['throughput_change_pct'] = 100 * (row['0']['median'] / row['1']['median'] - 1)
        summary.append(row)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', out, flush=True)
finally:
    if active:
        subprocess.run(['sudo', '-n', 'systemctl', 'start', 'cmp-idle-governor.service'], check=True)
