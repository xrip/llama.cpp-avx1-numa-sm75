# SM75 test build

These are local experiment tools, not production launch scripts. Run from a Linux CUDA build with one SM75 device. The measured setup uses CUDA 13.3 and GCC 14. All three code options are off by default.

## Scope

- `GGML_CUDA_SM75_SWIGLU_Q8_1=1`: fuse split SwiGLU and Q8_1 input packing for eligible prefill MMQ graphs. Keep the existing cuBLAS path.
- `LLAMA_SM75_GDN_TXN=1`: use a prior state and replay log for Qwen35 recurrent rollback, one sequence, one SM75 device, F32 state, state width 128.
- `LLAMA_SM75_MTP_SHARED_COMPUTE=1`: share target/draft device compute buffers. Both contexts must use the same single SM75 device and run in turn.

The shared-compute option does not require the transaction option to be on. Their commits are ordered because they use the same device capability check. No VMM, host KV cache, lazy replay, multi-GPU support, or new allocation pool is added.

## Checks

Build `llama-server` and `test-backend-ops` in `build/`. From the repository root:

```sh
GGML_TEST_SWIGLU_MMQ=1 GGML_CUDA_SM75_SWIGLU_Q8_1=0 ./build/bin/test-backend-ops test -b CUDA0 -o SWIGLU_MMQ
GGML_TEST_SWIGLU_MMQ=1 GGML_CUDA_SM75_SWIGLU_Q8_1=1 ./build/bin/test-backend-ops test -b CUDA0 -o SWIGLU_MMQ
./build/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET

g++-14 -std=c++17 -O2 sm75/test-gdn-transaction.cpp -Iggml/include -Lbuild/bin -Wl,-rpath,'$ORIGIN/build/bin' -lggml -lggml-base -o test-gdn-transaction
g++-14 -std=c++17 -O2 sm75/test-shared-compute.cpp -Iggml/include -Lbuild/bin -Wl,-rpath,'$ORIGIN/build/bin' -lggml -lggml-base -o test-shared-compute
g++-14 -std=c++17 -O2 sm75/test-mtp-state.cpp -Iinclude -Iggml/include -Lbuild/bin -Wl,-rpath,'$ORIGIN/build/bin' -lllama -lggml -lggml-base -o test-mtp-state
./test-gdn-transaction
./test-shared-compute
LLAMA_SM75_GDN_TXN=0 ./test-mtp-state "$SM75_MODELS/Qwen3.8-27B-UD-IQ2_XXS.gguf" txn-state-off
LLAMA_SM75_GDN_TXN=1 ./test-mtp-state "$SM75_MODELS/Qwen3.8-27B-UD-IQ2_XXS.gguf" txn-state-on
for f in txn-state-off/*.bin; do cmp "$f" "txn-state-on/${f##*/}" || exit 1; done
```

## Model and speed runs

Set `SM75_MODELS` to the installed GGUF folder. The model scripts default to the measured host's folder. They require the exact filenames in each script. The MTP runner uses the fixed token request in `shared-chat-mtp-request.json`; it does not make a prompt from the current source tree.

```sh
python3 sm75/run-shared-chat-kernels.py
python3 sm75/run-shared-chat-model.py
LLAMA_SM75_GDN_TXN=0 LLAMA_SM75_MTP_SHARED_COMPUTE=0 python3 sm75/run-shared-chat-mtp.py
LLAMA_SM75_GDN_TXN=1 LLAMA_SM75_MTP_SHARED_COMPUTE=1 python3 sm75/run-shared-chat-mtp.py
LLAMA_SM75_GDN_TXN=0 LLAMA_SM75_MTP_SHARED_COMPUTE=0 python3 sm75/run-shared-chat-mtp.py
```

Run only one model script at a time; they use port 18089. Timing scripts stop `cmp-idle-governor.service` if active and restore it on normal exit or error. They need passwordless sudo for that service. Check the service after an interrupted run. `profile-shared-chat.sh` uses the host's private Nsight Compute installation; inspect its two paths before use.

Raw output goes to dated folders at the repository root. Keep large logs, state files, GGUF files, and profiler reports out of Git. See `SHARED-CHAT-PERFORMANCE.md` for the measured results and limits.
