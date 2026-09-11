#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parent.parent
models = Path(os.environ.get('SM75_MODELS', '/home/xrip/tmp/prefill-single-20260907/models'))
out = root / ('mtp-baseline-' + time.strftime('%Y%m%d-%H%M%S'))
out.mkdir()
url = 'http://127.0.0.1:18089'
command = [str(root / 'build/bin/llama-server'), '-m', str(models / 'Qwen3.8-27B-UD-IQ2_XXS.gguf'),
           '-md', str(models / 'mtp-Qwen3.8-27B-Q4_0.gguf'), '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--device', 'CUDA0', '-ngl', 'all', '-fa', 'on', '-c', '4096', '-np', '1',
           '-ctk', 'f16', '-ctv', 'f16', '-b', '512', '-ub', '512', '--fit', 'off',
           '--no-mmproj', '-lv', '5', '--host', '127.0.0.1', '--port', '18089']
env = dict(os.environ, GGML_CUDA_SM75_SWIGLU_Q8_1='0', CUDA_SCALE_LAUNCH_QUEUES='4x', GGML_CUDA_GRAPH_OPT='1', LLAMA_SM75_GDN_TXN=os.environ.get('LLAMA_SM75_GDN_TXN', '0'), LLAMA_SM75_MTP_SHARED_COMPUTE=os.environ.get('LLAMA_SM75_MTP_SHARED_COMPUTE', '0'))
(out / 'command.json').write_text(json.dumps(dict(command=command, env={k: env[k] for k in ['GGML_CUDA_SM75_SWIGLU_Q8_1', 'CUDA_SCALE_LAUNCH_QUEUES', 'GGML_CUDA_GRAPH_OPT', 'LLAMA_SM75_GDN_TXN', 'LLAMA_SM75_MTP_SHARED_COMPUTE']}), indent=2))

def api(path, data=None):
    req = urllib.request.Request(url + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)

active = subprocess.run(['systemctl', 'is-active', '--quiet', 'cmp-idle-governor.service']).returncode == 0
server = None
telemetry = None
print(out, flush=True)
try:
    if active:
        subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'cmp-idle-governor.service'], check=True)
    with (out / 'server.log').open('w') as log, (out / 'gpu.csv').open('w') as gpu:
        telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,pstate,clocks.sm,clocks.mem,temperature.gpu', '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu)
        server = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        for attempt in range(300):
            if server.poll() is not None:
                raise RuntimeError(f'server exit {server.returncode}')
            try:
                api('/health')
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError('readiness timeout')
        request = json.loads((Path(__file__).with_name('shared-chat-mtp-request.json')).read_text())
        (out / 'request.json').write_text(json.dumps(request))
        for repeat in range(4):
            result = api('/completion', request)
            assert result['timings']['prompt_n'] == 1024 and result['timings']['predicted_n'] == 128
            (out / f'response-{repeat}.json').write_text(json.dumps(result, indent=2))
            print(repeat, result['timings'], flush=True)
finally:
    for process in [server, telemetry]:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
    if active:
        subprocess.run(['sudo', '-n', 'systemctl', 'start', 'cmp-idle-governor.service'], check=True)
