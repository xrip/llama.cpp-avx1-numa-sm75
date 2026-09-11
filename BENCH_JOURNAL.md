# Benchmark journal — SM75 / Ivy Bridge box

Running record of every measurement, the exact command behind it, and what it
showed. Purpose: be able to attribute each gain to a specific cause.

## Hardware / software baseline

| | |
|---|---|
| GPU | NVIDIA CMP 50HX, cc **7.5** (Turing), 10240 MiB |
| CPU | 2× Xeon E5-2670 v2 (Ivy Bridge), 2 NUMA nodes × 10 cores, 40 threads |
| **CPU ISA** | **AVX yes, AVX2 no** — so the fork's AVX1 paths are the live path |
| RAM | 62 GiB (32 GiB per node) |
| CUDA | 13.3, CCCL 3.3.4 (`CUB_VERSION 300304`) |
| Driver | 610.43.03 |

Clock note: `nvidia-smi -lgc` is **accepted but ignored** on this card — it reports
success and still boosts to ~2100 MHz. No measurement here is clock-pinned.
Long-running cases are reliable only because they reach steady boost.

## Builds under test

| tag | tree | commit | cuBLAS |
|---|---|---|---|
| UP | `/home/xrip/upstream-pristine` | `f114f91f9` (pristine ggml-org) | off |
| UPc | same | same | **on** |
| US | `/home/xrip/llama.cpp` | `d5919ca1f` (our master) | off |
| USc | same | same | **on** |

Both at the same upstream base, so differences are the fork's work only.

```bash
cmake -B build-cublas -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DLLAMA_CURL=OFF -DGGML_CCACHE=ON -DGGML_CUDA_FORCE_CUBLAS=ON
```

---

## 1. Dense 9B, fully on GPU — Qwen3.5-9B-UD-Q4_K_XL

```bash
llama-server -m Qwen3.5-9B-UD-Q4_K_XL.gguf -ngl 99 \
  -b 2048 -ub 2048 -c 17408 --host 127.0.0.1 --port 8099 -t 8 --no-warmup
# 16384-token prompt sent as an exact token-id array, cache_prompt=false,
# greedy (temp 0, top_k 1, seed 1234), 3 reps, median.
```

| build | prefill t/s | tg t/s |
|---|---|---|
| UPc | 2234.12 | 66.24 |
| USc | 2236.98 | 67.00 |
| UP | 1789.13 | 66.13 |
| US | 1955.07 | 66.92 |

**Attribution**
- `-DGGML_CUDA_FORCE_CUBLAS=ON`: **+24.9% prefill** (UP→UPc, 1789→2234).
  This is a *stock upstream CMake option*, not fork code. Upstream gets it too.
- Fork vs upstream, **cuBLAS config**: prefill **+0.13%**, tg **+1.14%**.
- Fork vs upstream, **default config**: prefill +9.27% (the `sm75_ffn` dispatch
  block), tg +1.19%. Subsumed by FORCE_CUBLAS, not additive with it.
- `llama-bench` pp8 (default cfg only): **+47.7%** — the Turing mmvq threshold
  work. Guarded off under FORCE_CUBLAS, so it does not reach production builds.

**Caveat:** this model is fully GPU-resident, so *no CPU kernel ran*. This test
says nothing about the AVX1/NUMA work.

---

## 2. MoE 35B — Qwen3.6-35B-A3B-UD-Q4_K_XL (22.85 GB, experts on CPU)

```bash
GGML_CUDA_GRAPH_OPT=1 CUDA_SCALE_LAUNCH_QUEUES=4x \
numactl --interleave=all llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --temp 0.0 --jinja --no-mmproj-offload \
  -c 64000 -np 1 -fa on -ctk q4_0 -ctv q8_0 \
  --threads 20 --threads-batch 20 -b 2048 -ub 2048 \
  --fit on --fit-target 256 --load-mode none \
  --host 127.0.0.1 --port 8099
# 2048-token prompt, n_predict=128, ignore_eos=true, cache_prompt=false
```

### 2a. Methodology correction (important)

`moe3` and `moe4` gave **opposite signs** for the same binaries:

| run | upstream tg | ours tg | apparent verdict |
|---|---|---|---|
| moe3 | 22.81 | 21.35 | ours −6.4% |
| moe4 | 20.90 | 23.21 | ours +11.0% |

Same upstream binary swung **8.4% between runs** while reps *within* a run agreed
to <1%. Reps inside one load share NUMA page placement, page cache and thermal
state, so they are correlated and prove nothing. **Alternate arms across separate
model loads.** Two earlier conclusions drawn from within-run reps were withdrawn.

Also: `ignore_eos: true` is mandatory. Without it a synthetic random-token prompt
at temp 1.0 hits EOS after 2–63 tokens and "tg" is startup overhead.

### 2b. Fork vs upstream — 4 alternating loads per arm

| arm | pp median | pp spread | tg median | tg spread |
|---|---|---|---|---|
| upstream | 166.52 | 0.4% | 21.37 | 1.8% |
| **ours** | 167.02 | 0.7% | **22.90** | 0.4% |
| ours + `--threads 40 -Cr 0-39 --cpu-strict 1` | **173.26** | 1.1% | 20.39 | 1.3% |

- **Fork tg +7.1% vs upstream** — reproducible, spreads far below the effect.
  This is the AVX1 Q4_K path in `repack.cpp` paying off. (48 `q4_K` mentions in
  the CPU diff; an earlier grep for the *enum* `GGML_TYPE_Q4_K` missed them.)
- Prefill: ~0%.
- 40 threads: prefill +3.7%, **tg −11.0%** — 40 threads means hyperthreads on a
  2×10-core box. Prefill is compute-bound and gains; generation is
  memory-latency-bound and loses.

### 2c. CPU profile during generation (`perf record -F 499 -g`)

| % | symbol | class |
|---|---|---|
| 18.3 | `rep_movs_alternative` (kernel) | **fread of model file** |
| 13.4 | `ggml_vec_dot_q4_K_q8_K` | matmul |
| ~36.5 | `libgomp` (barrier spin) | **thread sync** |
| 7.7 | `ggml_vec_dot_q5_K_q8_K` | matmul |
| 4.2 | `clear_page_erms` (kernel) | page zeroing |
| 1.6 | tinyBLAS Q8_0 gemm | matmul |
| 1.0 | `ggml_vec_dot_q6_K_q8_K` | matmul |

Only **~24%** of generation CPU time is actual matmul. Call graph attributes the
18.3%:

```
llama_file::impl::read_raw_unsafe() -> _IO_fread -> read()
  -> ext4_file_read_iter -> filemap_read -> copy_page_to_iter
    -> rep_movs_alternative
```

i.e. page-cache→userspace copies of model data, consistent with
`--load-mode none` (mmap disabled). Note **no tensor in this model exceeds
1 GiB**, so `--lazy-mode auto` (threshold 4 GiB) never triggers — lazy mode is
not involved. `--lazy-mode on-direct` (PR #28136) targets the much larger
Qwen3.8-Flash and is not applicable here; it is also **not in our master**
(it lives on `perf/numa-dense-20260902`).

**Implication:** tuning `ggml_vec_dot_q4_K_q8_K` by 20% would move the total by
~2.7%. The sync and I/O costs are far larger targets.

### 2d. Thread / OpenMP sweep (3 alternating loads per arm)

Base = `--threads 20 --threads-batch 20`.

| arm | pp median | tg median | Δpp | Δtg |
|---|---|---|---|---|
| t20-b20 (base) | 166.95 | 22.93 | — | — |
| t20-b40 | 173.44 | 22.38 | +3.9% | −2.4% |
| t10-b40 | 174.31 | 19.09 | +4.4% | −16.7% |
| t20 + `OMP_WAIT_POLICY=passive` | 172.94 | **12.13** | +3.6% | **−47.1%** |
| **t20-b40 + `OMP_PROC_BIND=close OMP_PLACES=cores`** | **174.64** | 22.95 | **+4.6%** | +0.1% |

- **`OMP_WAIT_POLICY=passive` is catastrophic (−47% tg).** The libgomp spinning
  is *load-bearing*: MoE graphs have many small nodes, so barrier wake-up latency
  costs far more than spinning. The 36.5% is largely inherent, **not** recoverable
  by configuration. (This refuted the hypothesis that motivated the sweep.)
- **`OMP_PROC_BIND=close` + `OMP_PLACES=cores` is a free +4.6% prefill** at no tg
  cost — stops thread migration across the two NUMA nodes.

### 2e. Best known MoE launch so far

```bash
GGML_CUDA_GRAPH_OPT=1 CUDA_SCALE_LAUNCH_QUEUES=4x \
OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --interleave=all llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --temp 0.0 --jinja --no-mmproj-offload \
  -c 64000 -np 1 -fa on -ctk q4_0 -ctv q8_0 \
  --threads 20 --threads-batch 40 -b 2048 -ub 2048 \
  --fit on --fit-target 256 \
  --host 0.0.0.0 --port 8080
```
vs the original `--threads 20 --threads-batch 20 --load-mode none`:
**prefill +4.6%, tg unchanged.**

---

## 3. Open / in flight

- **`--load-mode` sweep** (`none` vs `auto` vs `mmap` vs `mmap+mlock`) — testing
  whether mmap removes the 18.3% fread cost. *Running.*
- **SM75 IQ2_XXS decode reuse** — implemented (`f39c81415`), not yet built or
  measured. Load-widening is impossible for IQ2_XXS (66-byte block, `qs` at
  offset 2, only 2-byte alignment), so only decode reuse applies.
- **IQ2_XXS** `Qwen3.8-27B-UD-IQ2_XXS` downloaded (6.77 GB). Prediction on
  record: ~0% from the fork, since there is no IQ2 code in it and at 7.27 GB the
  model is fully GPU-resident.

## 4. Method rules learned here

1. Alternate arms across **model loads**, not reps within a load (§2a).
2. `ignore_eos: true` for any tg measurement with synthetic prompts.
3. Check the GPU is idle with `pgrep -x llama-server` (name match — `pgrep -f`
   matches other command lines that merely mention the binary).
4. A stray server holding VRAM silently invalidates results — verify 0 MiB.
5. Prefer `test-backend-ops perf` (thousands of iterations) over `llama-bench`
   at tiny sizes, where clock ramp gives ±7–17% error bars.

---

## 5. `--load-mode` sweep — mmap is worse, `none` is correct

3 alternating loads per arm, best OpenMP settings applied to all arms
(`OMP_PROC_BIND=close OMP_PLACES=cores`, `--threads 20 --threads-batch 40`).

| load-mode | load s | pp median | tg median | Δpp | Δtg |
|---|---|---|---|---|---|
| **none** (what you run) | 29 | **174.11** | **23.10** | — | — |
| auto | 19 | 165.11 | 21.44 | −5.2% | −7.2% |
| mmap | 19 | 165.38 | 20.78 | −5.0% | −10.1% |
| mmap+mlock | 19 | 163.62 | 21.30 | −6.0% | −7.8% |

**`--load-mode none` wins; mmap costs 5–10%.** Cause is NUMA:
`numactl --interleave=all` only stripes **anonymous** memory. With `none` the
weights are malloc'd buffers and get interleaved across both nodes; with mmap the
pages are page-cache-backed and land wherever the kernel put them, so a large
share of expert reads become remote-node accesses on this 2-socket box. mmap
loads ~10 s faster and then runs slower.

**This also refutes the §2c reading of the 18.3% `fread`.** If that cost were in
the generation steady state, mmap (which has no read syscalls) would have helped.
It hurt. The v1 profile window (20 s after issuing the request) most likely caught
first-touch page population rather than steady-state generation. Re-profiling with
two warmups and a 45 s offset before recording; §2c should be treated as
unreliable until that lands.

## 6. Hypotheses tested and **rejected** (kept so they are not retried)

| hypothesis | expected | measured | verdict |
|---|---|---|---|
| `OMP_WAIT_POLICY=passive` recovers the 36% libgomp spin | large tg gain | **−47.1% tg** | rejected — spinning is load-bearing |
| mmap removes the 18% `fread` cost | tg gain | **−10.1% tg** | rejected — NUMA interleave beats mmap |
| fewer threads help the memory-bound tg | tg gain | −16.7% (t10) | rejected |
| `--lazy-mode` is causing on-demand reads | — | no tensor >1 GiB, threshold is 4 GiB | not applicable |
| PR 28366 (radix top-k) affects our prefill | prefill gain | compiled out (CCCL 3.3.4); our models never call TOP_K | no-op here |
| fork's Q4_K CPU work absent | — | 48 `q4_K` hits in CPU diff | wrong — grep was for the enum name |

## 7. Confirmed gains, ranked by size

| # | change | workload | gain | whose |
|---|---|---|---|---|
| 1 | `-DGGML_CUDA_FORCE_CUBLAS=ON` | dense 9B prefill | **+24.9%** | **upstream option** |
| 2 | fork AVX1 Q4_K CPU path | MoE tg | **+7.1%** | fork |
| 3 | `OMP_PROC_BIND=close OMP_PLACES=cores` + `--threads-batch 40` | MoE prefill | **+4.6%** | launch flags |
| 4 | Turing mmvq threshold tuning | dense pp8, non-cuBLAS only | +47.7% | this session |
| 5 | upstream branchless Q4_K/Q5_K unpack | mmvq kernel n≥5 | +4–10% | upstream |
| 6 | fork `sm75_ffn` cuBLAS dispatch | dense 9B prefill, non-cuBLAS only | +9.3% | fork (subsumed by #1) |

Note #4 and #6 apply only to builds **without** `FORCE_CUBLAS`, so they do not
reach the production configuration.

---

## 8. SM75 IQ2_XXS decode reuse — implemented, measured, **rejected**

Applied the technique that works for IQ4_XS (`sm75_iq4_decode`): hoist the
weight-side decode out of the `ncols_dst` loop so the four `iq2xxs_grid` lookups
and sign expansion happen once per block instead of once per output column.
Gated to Turing, `ncols_dst` 2..4.

Load widening — the *other* IQ4_XS trick — is **impossible** here:

| type | block size | `qs` offset | widening |
|---|---|---|---|
| IQ4_XS | 136 B | 8 | `int2` loads OK (existing fork path) |
| IQ2_XXS | **66 B** | **2** | only 2-byte aligned; `get_int_b2` is required |

66 is 2 mod 4, so block bases are not even 4-byte aligned.

Result, `test-backend-ops perf -p type_a=iq2_xxs`, medians of 5 reps after a
discarded warmup, interleaved, FORCE_CUBLAS builds:

| n | base us | exp us | delta | |
|---|---|---|---|---|
| 1 | 94.19 | 94.19 | +0.00% | control, identical code |
| 2 | 97.09 | 96.90 | −0.20% | reuse active |
| 3 | 118.96 | 119.73 | +0.65% | reuse active |
| 4 | 145.49 | 146.02 | +0.36% | reuse active |
| 5 | 157.10 | 157.33 | +0.15% | |
| 8 | 221.22 | 221.06 | −0.07% | |
| 512 | 1589.06 | 1583.31 | −0.36% | |

**n=2..4 mean +0.27%; noise floor −0.07% (identical-code rows).** No gain.
Correctness passed (2/2 backends).

**Why it fails where IQ4_XS succeeds:** the decoded state is 8 ints of grid plus
scale and `d` — about double `sm75_iq4_decoded` — so occupancy drops on Turing,
while the work saved is nearly free to begin with: `iq2xxs_grid` is 256 x 8 B =
**2 KB and L1-resident**. IQ4_XS reuse eliminates genuinely expensive work;
IQ2_XXS reuse eliminates L1 hits.

Reverted in `a261f5e95`. Implementation was correct; the premise was wrong.

**Generalisation for future SM75 work:** decode reuse only pays when the decoded
state is small *and* the eliminated work is expensive (cache-missing loads or
long dependency chains). Check the grid/table size first — if it fits in L1,
reuse will not pay for the register pressure.

---

## 9. SM75 shared-memory codebooks — **+6% tg on a real model**

Found by profiling, after two kernel ideas written from plausible reasoning both
failed. Nsight works on this card via the CMP unlock (Ubuntu `nsight-compute`
2022.4.1 + the hash-gated patch); `ERR_NVCMPGPU` is not a hard limit.

### Diagnosis

`mul_mat_vec_q<IQ2_XXS, ncols=1>`, grid 4096 x block (32,4):

| metric | value | reading |
|---|---|---|
| **`lg_throttle`** | **19.81/issue** | **43x the next stall — the bottleneck** |
| `wait` | 1.43 | |
| `long_scoreboard` | 0.46 | not memory latency |
| `mio_throttle` | 0.06 | shared/MIO pipe idle |
| L1 hit | 98.04% | not a cache problem |
| DRAM / SM | 22.99% / 28.54% | neither saturated |
| occupancy | 92.82% | not warp-starved |

The load/store **instruction queue** is full. A `vec_dot` issues ~16 loads
(4x16-bit from `get_int_b2`, 4x64-bit codebook, 8x32-bit activations), and a load
costs an LG issue slot **even when it hits L1**.

### Fix

Stage the codebook in shared memory once per block: `LDG` -> `LDS`, moving that
traffic to the idle MIO pipe. Grids are 1-8 KiB and fit easily.

Counters after: `lg_throttle` **19.81 -> 11.59**, `mio_throttle` 0.06 -> 0.14,
DRAM 23% -> 27.9%, SM 28.5% -> 36.1%.

### Kernel results (n=1, m=4096, k=14336)

| type | delta |
|---|---|
| iq3_s | **-15.7%** |
| iq2_xxs | **-13.6%** |
| iq3_xxs | -11.0% |
| iq2_xs | -10.2% |
| iq2_s | -9.2% |
| iq1_s | -5.2% |

IQ1_S/IQ1_M are gated to `ncols_dst <= 2`: they win at n=1/n=2 (-5.2%/-11.1%)
and lose at n>=4 (+7.7%/+2.2%) because the 8 KiB `iq1s_grid_gpu` costs a block
of occupancy per SM. IQ2_S is also 8 KiB and wins at every n, so size alone is
not the deciding factor.

### End to end — `llama-server`, Qwen3.8-27B-UD-IQ2_XXS, 16384-token prompt

| arm | prefill t/s | tg t/s |
|---|---|---|
| base | 699.17 | 19.213 |
| shared codebooks | 697.15 | **20.361** |
| | **-0.29%** (control) | **+5.98%** |

Raw tg, 3 alternating model loads: base `19.448 / 19.213 / 19.144`,
sgrid `20.455 / 20.361 / 20.290` — distributions do not overlap.

**The prefill column is the control**: at `-ub 2048` prefill goes through cuBLAS,
which the change does not touch, so it must stay flat. It does. Only the mmvq
path moved. Correctness: all 7 grid types and `MUL_MAT_ID` pass 2/2 backends.

### Two measurement traps hit while doing this

1. **Ordering bias.** Running all-base-then-all-exp let the first base invocation
   of each pass catch the 645 MHz idle clock, producing a bogus **-78%** for
   iq2_xxs. Alternate base/exp *per type* and discard a warm-up invocation.
2. **Gating on the wrong batch size.** IQ1 was first excluded outright on the
   strength of its n=4/n=8 regression — but token generation is `ncols_dst == 1`,
   where IQ1_S is 5.2% *faster*. Excluding it would have removed a win from the
   only path that mattered. Check which `n` the workload actually uses.
