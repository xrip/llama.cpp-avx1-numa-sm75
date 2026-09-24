# SM75 prefill experiments

## Base and scope

Base: `xrip/llama.cpp-avx1-numa-sm75`, public `master`, commit `af084752abb72652d51785602ef9536f631fec9d`.

Idea source: `SirDebugALot/ninfer-3090`, commit `0d0ad44d8dac9c08c2e28b7c8c86527e10ba30a4`.

This is a small experiment patch, not a port of the SM86 kernels and not a measured speed improvement. It uses the fork's existing cuBLAS path and MMQ specializations. It adds no BF16 MMA, asynchronous copy, new weight format, or persistent unpacked model copy.

Important finding: the base already has a cuBLAS override for four FFN shapes in `ggml-cuda.cu`, before `ggml_cuda_should_use_mmq()`. The default mode below preserves that override. Do not describe the base as always using MMQ on SM75.

## Controls

Environment variables are read once per process. Set them before starting the server or benchmark. Restart the process for each case. CUDA Graphs can retain the selected kernels.

| Control | Default | Meaning |
| --- | --- | --- |
| `GGML_CUDA_SM75_PREFILL_MODE` | `auto` | `auto`, `mmq`, or `cublas`; only the dense SM75 prefill override |
| `GGML_CUDA_SM75_PREFILL_MIN_BATCH` | `1024` | Minimum N in experimental `cublas` mode; integer from 512 to 1048576 |
| `GGML_CUDA_SM75_PREFILL_MAX_SCRATCH_MIB` | `512` | Conversion-scratch estimate cap in experimental `cublas` mode; integer from 1 to 65536 |
| `GGML_CUDA_SM75_MMQ_MAX_J` | `0` | Limit the search over existing MMQ token tiles: 0, 32, 64, 96, or 128; 0 means unchanged |
| `GGML_CUDA_SM75_PREFILL_TRACE` | `0` | 1 logs dispatch decisions and selected MMQ tiles; turn it off for timing |
| CMake `GGML_CUDA_SM75_Q8_0_FAST_HALF` | `OFF` | Compile the experimental Q8 V to FP16 conversion for the SM75 device target |

Invalid values produce a warning and use that setting's default. An empty value is invalid; use `unset` to remove a variable.

### Dense cuBLAS policy

All modes retain the existing checks for SM75, F32 activations/output, contiguous inputs, device-resident weights, and single-channel/single-sample matrices. The earlier MMVF/MMF/MMVQ decisions are unchanged. `MUL_MAT_ID` is not modified.

`auto` is the exact old override: Q4_K or IQ4_XS; K/M equal to 4096/12288, 12288/4096, 5120/17408, or 17408/5120; N between 1024 and 2048 inclusive. The new scratch cap and minimum-batch setting do not alter this baseline mode.

`mmq` disables this one FFN cuBLAS override and falls through to the existing backend selector. It is not a global force flag and cannot override `GGML_CUDA_FORCE_CUBLAS` or an earlier fallback.

`cublas` allows Q4_K, Q5_K, Q6_K, and IQ4_XS outside the old shape list. It requires K >= 256 and divisible by 256, M >= 128 and divisible by 128, N >= the selected minimum and divisible by 128, and a fitting scratch estimate. Other shapes fall through to the existing selector. Small batches below 512 do not enter this experiment.

The estimate is `4 * (M*K + K*N + M*N)` bytes, using the dimensions passed to this CUDA operation. This conservatively includes FP32 conversion overrides. Products are checked before multiplication. For example, K=5120, M=17408, N=2048 gives an estimate of 516 MiB, so the default 512 MiB cap rejects it. Raising the cap to 768 MiB admits that shape, but requires separate VRAM validation.

This is not a total-VRAM guarantee. It excludes cuBLAS internal workspace, allocator cache, model/KV storage, other live tensors, and concurrent operations. No memory availability probe or OOM recovery is added.

Compile-time force flags keep their existing priority. Existing fusions which call MMQ directly are not rerouted by this policy. Use trace output or a profiler to confirm which operations actually changed. Do not interpret the absence of a speed change as a measurement of a path that did not run.

### MMQ tile experiment

The cap applies only to SM75 Q4_K/IQ4_XS dense operations with no expert IDs, one channel/sample, complete 128-row tiles, and N >= 512 divisible by 128. All other operations retain the normal search up to J=128.

The patch changes only the host search range. It keeps the original configuration-validity and shared-memory checks. If no valid configuration fits the cap, it repeats the original search. It does not change I, thread counts, launch bounds, weight layout, Stream-K, fixup logic, or buffer padding. `ggml_cuda_mmq_get_J_max()` is deliberately unchanged, so a smaller launch does not shrink a buffer needed by another path.

The cap is not a promise to use exactly that width. It selects the best candidate from the existing configurations up to that width. This is not a new SM75-tuned configuration table or an autotuner.

### Q8 V conversion experiment

The helper constructs two FP16 bit patterns for `1152 + q`, then subtracts 1152 with `__hsub2_rn`. It covers every signed INT8 code, including -128. The existing half2 scale multiplication stays separate, with the same operands and rounding stage as before.

The existing Q8 loads, block offsets, lengths, and scale are unchanged. No wider memory access is introduced. Only `dequantize_V_q8_0<half, ...>` compiled for `__CUDA_ARCH__ == 750` changes. F32 conversion and other device targets keep the old code. There is no runtime flag for this helper; compare two builds.

F16 V/KV caches do not use this conversion. Q8 storage alone is not proof that this helper runs: another attention route may perform its own conversion. This helper can affect both prefill and decode when that route is used.

## Build and apply

This is a unified diff for `git apply`, not a commit for `git am`. It does not change the branch or publish anything. A local branch or uncommitted edits may differ from the public base; check before applying. Do not use `--reject` or force a partially applicable patch.

```sh
git rev-parse HEAD
git apply --check /path/to/sm75-ninfer-prefill-af084752.patch && \
    git apply /path/to/sm75-ninfer-prefill-af084752.patch
```

Reconfigure and rebuild the CUDA backend. Preserve the existing AVX1, NUMA, compiler, and build settings. For an existing build directory, these are the relevant additions:

```sh
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DGGML_CUDA_SM75_Q8_0_FAST_HALF=OFF
cmake --build build --parallel 4
```

Build a second directory with the same options and `-DGGML_CUDA_SM75_Q8_0_FAST_HALF=ON` for the Q8 experiment. Do not compare two builds with different compiler versions or unrelated CPU/CUDA flags.

## Measurement plan

Use the same model file, context, prompt, microbatch, GPU split, NUMA policy, clocks, power limit, MTP, KV types, and CUDA Graph settings in every A/B pair. Do not enable all experiments at once.

1. Compare the unpatched build with patched defaults: `auto`, J=0, Q8 helper OFF. This is the baseline-regression check.
2. Compare `auto`, `mmq`, and `cublas` with all other controls unchanged. Use N=512/1024/2048; N=512 requires setting the cublas minimum to 512.
3. Set mode `mmq` and compare J=0/32/64/96/128 on Q4_K/IQ4_XS. This avoids the old FFN override hiding the MMQ experiment.
4. Separately compare Q8-helper OFF/ON with Q8 V and the same dispatcher. Also test F16 KV as an unaffected control.

Example environment settings (then run the existing full benchmark/server command):

```sh
# Baseline
unset GGML_CUDA_SM75_PREFILL_MODE GGML_CUDA_SM75_PREFILL_MIN_BATCH
unset GGML_CUDA_SM75_PREFILL_MAX_SCRATCH_MIB GGML_CUDA_SM75_MMQ_MAX_J
unset GGML_CUDA_SM75_PREFILL_TRACE

# cuBLAS experiment, default 512 MiB estimate cap
export GGML_CUDA_SM75_PREFILL_MODE=cublas
export GGML_CUDA_SM75_PREFILL_MIN_BATCH=1024

# Separate MMQ experiment
export GGML_CUDA_SM75_PREFILL_MODE=mmq
export GGML_CUDA_SM75_MMQ_MAX_J=64
```

Trace is only for a diagnostic run. It logs every relevant dispatch and can distort timing. A reused CUDA Graph may not repeat host-side dispatch logging.

Check real backend numerical results, logits/perplexity, short decode, MTP acceptance, non-aligned tails, CUDA Graph replay, and each GPU separately. Then test the two-GPU configuration. A different MMQ reduction order or the existing cuBLAS compute policy can change floating-point results; require an agreed accuracy tolerance rather than identical generated text.

Record individual operation timings, full prefill timings, decode timings, and peak VRAM per GPU. Warm up each case, then repeat A/B/B/A. Keep raw commands and logs. There are no GPU measurements in this patch.

## Validation status at delivery

Performed: C++ host tests with GCC and Clang, GCC AddressSanitizer/UndefinedBehaviorSanitizer, strict environment parsing checks, exact old-policy comparisons over a shape grid, all 65,536 INT8 pairs, and a CPU IEEE binary16 reference check for all 256 codes multiplied by all 63,488 finite FP16 scales.

Not performed: a full repository build, nvcc compilation, actual CUDA execution, backend accuracy tests, long-context or two-GPU tests, performance or VRAM measurements. No speedup is claimed.

The separate delivery archive contains standalone host/CUDA tests and the exact validation report. These tests are not added to the repository's main test suite or CI. The CUDA test must still be compiled and run on SM75; a skipped GPU test is not a pass.
