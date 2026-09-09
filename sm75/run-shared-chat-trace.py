#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parent.parent
models = Path(os.environ.get('SM75_MODELS', '/home/xrip/tmp/prefill-single-20260907/models'))
url = 'http://127.0.0.1:18089'
command = [str(root / 'build/bin/llama-server'), '-m', str(models / 'Qwen3.5-9B-UD-Q4_K_XL.gguf'),
           '--device', 'CUDA0', '-ngl', 'all', '-fa', 'on', '-c', '8192', '-np', '1',
           '-ctk', 'f16', '-ctv', 'f16', '-b', '2048', '-ub', '2048', '--fit', 'off',
           '--no-mmproj', '-lv', '5', '--host', '127.0.0.1', '--port', '18089']

def api(path, data=None):
    req = urllib.request.Request(url + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)

with (root / 'model-dispatch-trace.log').open('w') as log:
    server = subprocess.Popen(command, env=dict(os.environ, GGML_CUDA_SM75_SWIGLU_Q8_1='1', GGML_CUDA_SM75_SWIGLU_Q8_1_TRACE='1'), stdout=log, stderr=log)
    try:
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
        tokens = api('/tokenize', {'content': (root / 'src/llama-context.cpp').read_text(), 'add_special': True})['tokens'][:2048]
        result = api('/completion', dict(prompt=tokens, n_predict=1, temperature=0, cache_prompt=False))
        print(result['timings'])
    finally:
        server.terminate()
        server.wait(timeout=60)
