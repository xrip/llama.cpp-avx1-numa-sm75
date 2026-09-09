#!/usr/bin/env python3
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parent.parent
models = Path(os.environ.get('SM75_MODELS', '/home/xrip/tmp/prefill-single-20260907/models'))
out = root / ('model-results-' + time.strftime('%Y%m%d-%H%M%S'))
out.mkdir()
url = 'http://127.0.0.1:18089'
server = None
telemetry = None
rows = []
requests = {}
active = subprocess.run(['systemctl', 'is-active', '--quiet', 'cmp-idle-governor.service']).returncode == 0

def api(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url + path, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)

try:
    if active:
        subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'cmp-idle-governor.service'], check=True)
    for model in ['Ornith-1.0-9B', 'Qwen3.5-9B']:
        for phase, enabled in enumerate([0, 1, 0]):
            label = f'{model}-{phase}-{enabled}'
            command = [str(root / 'build/bin/llama-server'), '-m', str(models / (model + '-UD-Q4_K_XL.gguf')),
                       '--device', 'CUDA0', '-ngl', 'all', '-fa', 'on', '-c', '32768', '-np', '1',
                       '-ctk', 'f16', '-ctv', 'f16', '-b', '2048', '-ub', '2048', '--fit', 'off',
                       '--no-mmproj', '--jinja', '--host', '127.0.0.1', '--port', '18089']
            env = dict(os.environ, GGML_CUDA_SM75_SWIGLU_Q8_1=str(enabled), CUDA_SCALE_LAUNCH_QUEUES='4x', GGML_CUDA_GRAPH_OPT='1')
            (out / (label + '-command.json')).write_text(json.dumps(dict(command=command, env={k: env[k] for k in ['GGML_CUDA_SM75_SWIGLU_Q8_1', 'CUDA_SCALE_LAUNCH_QUEUES', 'GGML_CUDA_GRAPH_OPT']}), indent=2))
            with (out / (label + '-server.log')).open('w') as log, (out / (label + '-gpu.csv')).open('w') as gpu_log:
                telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,pstate,clocks.sm,clocks.mem,temperature.gpu', '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu_log)
                server = subprocess.Popen(command, env=env, stdout=log, stderr=log)
                for attempt in range(300):
                    if server.poll() is not None:
                        raise RuntimeError(f'{label}: server exit {server.returncode}')
                    try:
                        api('/health')
                        break
                    except Exception:
                        time.sleep(1)
                else:
                    raise RuntimeError('server readiness timeout')
                if model not in requests:
                    text = (root / 'src/llama-context.cpp').read_text()
                    tokens = api('/tokenize', {'content': text * 4, 'add_special': True})['tokens']
                    tail = api('/tokenize', {'content': '\nExplain the main responsibilities of this code and list five correctness risks.\n', 'add_special': False})['tokens']
                    prompt = tokens[:16384-len(tail)] + tail
                    assert len(prompt) == 16384
                    requests[model] = dict(prompt=prompt, n_predict=128, cache_prompt=False, temperature=0.0,
                                           seed=1234, ignore_eos=True)
                    (out / (model + '-request.json')).write_text(json.dumps(requests[model]))
                for repeat in range(4):
                    response = api('/completion', requests[model])
                    (out / f'{label}-{repeat}.json').write_text(json.dumps(response, indent=2))
                    timings = response['timings']
                    if timings['prompt_n'] != 16384 or timings['predicted_n'] != 128:
                        raise RuntimeError(f'{label}: token count mismatch {timings}')
                    rows.append(dict(model=model, phase=phase, enabled=enabled, repeat=repeat, timings=timings, content=response['content']))
                    (out / 'records.json').write_text(json.dumps(rows, indent=2))
                    print(label, repeat, timings['prompt_per_second'], timings['predicted_per_second'], flush=True)
                server.terminate()
                server.wait(timeout=60)
                server = None
                telemetry.terminate()
                telemetry.wait(timeout=10)
                telemetry = None
    summary = []
    for model in requests:
        measured = [r for r in rows if r['model'] == model and r['repeat'] > 0]
        result = dict(model=model, same_output=len({r['content'] for r in measured}) == 1)
        for phase in [0, 1, 2]:
            group = [r for r in measured if r['phase'] == phase]
            result[str(phase)] = {metric: dict(median=statistics.median(r['timings'][metric] for r in group), minimum=min(r['timings'][metric] for r in group), maximum=max(r['timings'][metric] for r in group)) for metric in ['prompt_per_second', 'predicted_per_second']}
        summary.append(result)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', out, flush=True)
finally:
    for process in [server, telemetry]:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
    if active:
        subprocess.run(['sudo', '-n', 'systemctl', 'start', 'cmp-idle-governor.service'], check=True)
