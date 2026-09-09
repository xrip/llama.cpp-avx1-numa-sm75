# SM75 shared-chat performance log

Started: 2026-09-09.

## Scope and base

- Approved order: prefill SwiGLU/Q8_1 fusion, recurrent-state memory savings, shared target/draft compute memory.
- Source under review: BrunoPPassini/llama.cpp at `daa7e974f5b7`.
- Initial local control source was `03acdbdc5e6470470a1eeca11a2c87834dd36cf6`. A live fetch advanced origin/master to `e653dac15`; the isolated experiment was moved to this fresh base before source changes.
- Isolated worktree: `G:/llama/.codex-worktrees/sm75-shared-chat-20260909`.
- Ancestry check passed: this base includes kernel integration `888371054`.
- Main checkout remains at `912eaf4c0`; its existing staged and untracked work is outside this experiment.

## Hardware and access

The older two-card reports are not a current hardware check. `PREFILL-EXECUTION.md` records a later user-approved target of one CMP50HX with 10 GiB. Confirm the live test host before choosing model placement. Kernel results on one card and CPU-offloaded model results must be separate from two-card full-GPU results.

Initial access to `xrip@192.168.1.224` failed with `Permission denied (publickey,password)` in batch mode. The SSH executable is `C:/Program Files/Git/usr/bin/ssh.exe`; `ssh` is not on this session's PATH. No remote state was changed.

Access resolved with the existing `C:/Users/xr1p/.ssh/servers` key and `IdentitiesOnly=yes`. Live check: one CMP50HX, 10240 MiB total, 0 MiB used. No test/server process found. Production source is clean at `d5919ca1f`; it was not switched or rebuilt. Remote isolated worktree: `/home/xrip/tmp/sm75-shared-chat-20260909`. CUDA 13.3, GCC14, native SM75 Release build started. The idle governor was active at setup.

## Test rules

- Keep source SHA, build flags, model identity, actual tensor types, command, device placement, and raw log paths with each result.
- First pass: existing backend numerical tests for the changed operation and boundary shapes.
- Check CUDA Graph replay and the unfused control. A sanitizer failure to start is not a sanitizer pass.
- Use a warmup and repeated paired control/candidate runs; record median and min/max, not just the best run.
- Model tests: actual PP16384, fixed generation budget, no prompt cache reuse, same sampling, and peak VRAM.
- MTP tests: acceptance, partial acceptance, rejection/rollback, MTP2/MTP3, and long-context generation.
- Keep hardware settings fixed. Do not combine speed percentages from separate operations.
- No production promotion without measured benefit and correctness evidence.

## Experiment journal

| ID | Change | Build / correctness | Performance | Decision |
|---|---|---|---|---|
| 00 | Control setup | Clean isolated worktree; integration ancestry passed; live single-card check passed | No measurements | Use fresh fork base e653dac15 |
| 01 | SwiGLU to Q8_1 prefill fusion | Corrected build passed; both modes passed 50/50 cases; enabled dispatch trace confirmed; sanitizer unavailable | Fused graph throughput +4.40% to +6.38%; all active timing ranges separate | Whole-model check running |
| 02 | Recurrent-state transaction log | Not implemented | Not measured | After experiment 01 |
| 03 | Shared target/draft compute memory | Not implemented | Not measured | Device count must be confirmed |

## Design constraints found

- The source fusion uses fixed FFN width 17408, layer numbers, and Q6_K down projections. Use actual graph consumer type and shape instead.
- The local CUDA backend already has graph fusion checks. Use these checks for a short producer/consumer lifetime; avoid a persistent activation cache.
- MMQ consumes type-dependent Q8_1 layouts. Fusion must match the actual down-projection type and retain the ordinary path when it is not eligible.
- Existing gate/up/GLU decode fusion must retain its current dispatch.
- Fresh fork code already sends large Q4_K/IQ4_XS FFN batches to cuBLAS. The first experiment excludes Q4_K/IQ4_XS N1024-2048 so fusion cannot undo that measured path.
- Experiment 01 is opt-in with `GGML_CUDA_SM75_SWIGLU_Q8_1=1`. It fuses adjacent split-F32 SwiGLU and MMQ using existing graph dependency and memory-range checks. The quantizer uses the actual weight type. No activation cache is added.
- Source shared compute memory requires exactly one CUDA device. Two-device support is a separate change, not an environment-only setting.
- Its context link also requires target and MTP to use the same model object and one sequence each. A separately loaded draft GGUF may not enter this path; confirm the actual context/model link in the chosen MTP run.
- Source transaction logging is restricted to QWEN35, GPU-offloaded F32 recurrent state, and one sequence. Check the actual model architecture before use.

## Experiment 01 evidence

- `first-numerical.log` under the remote worktree: 30/30 CUDA numerical cases passed with the feature enabled. Cases cover Q4_K, IQ4_XS, Q6_K, N1/9/32/256/1024, K256/768, and 128/129 output rows.
- Test fixtures reuse `test_mul_mat` in the existing `tests/test-backend-ops.cpp`. They are selected only by `GGML_TEST_SWIGLU_MMQ`.
- The perf fixture times the complete SwiGLU/down graph. It keeps one graph instance per timing iteration, because repeating only the final matmul would add extra consumers and prevent fusion.
- Runner: `sm75/run-shared-chat-kernels.py`. Both modes use the same build. It keeps raw output and telemetry, does one warmup and five paired measured passes, and restores the idle governor's starting state.
- First paired runner attempt stopped after its warmup because CUDA Graph messages split timing lines. This was a parser error, not a kernel error. Raw run: `kernel-results-20260909-070356`. The reader now accepts those line breaks; a new run keeps the failed attempt intact.
- The second attempt (`kernel-results-20260909-070526`) showed near-zero changes. Source inspection found that `ggml_can_fuse` requires equal node shapes, so the SwiGLU/down pair was not fused. That run was stopped; its numerical passes and timings are not evidence for the new kernel. The corrected code uses `ggml_can_fuse_subgraph`, and the numerical runner now requires a dispatch trace before timing.
- The traced run `kernel-results-20260909-071237` passed the backend checks, but its runner counted only unsplit result lines. The reader now uses the backend's exact `30/30 tests passed` summary and separately requires the fusion marker. This second reader fix did not change the CUDA code.

### Minimal memory-port design

1. Transaction log: keep the existing persistent recurrent state, save the GDN update operands for the speculative suffix, and replay only its accepted prefix. Keep full snapshots as the default and as the fallback for unsupported model/state/sequence layouts. Do not add phase-arena reuse or lazy replay in this first change.
2. Shared compute: share only compatible target/draft scheduler buffer layouts. Keep a shared owner for the allocation, synchronize the previous context before allocation or input writes, and detach to private storage on growth. Keep the single-device restriction until a two-device test host is available.
3. Test both changes separately with the chosen MTP target and the same quant, placement, KV types, context, and draft length as its control. Record actual physical VRAM savings, not a sum of two logical scheduler size reports.
4. Only combine candidates after separate checks pass. CUDA Graph replay must survive alternating target/draft work and a graph-size change.

## Experiment 01 kernel results

Five paired measured passes after warmup. Same build, feature off/on. Units are microseconds per complete SwiGLU/down graph, including activation preparation. Throughput change is `control_time / candidate_time - 1`.

| Weight | Width | Output rows | Tokens | Control median us | Fused median us | Throughput change |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K | 12288 | 4096 | 256 | 825.53 | 780.60 | +5.76% |
| Q4_K | 17408 | 5120 | 256 | 1358.93 | 1299.24 | +4.59% |
| IQ4_XS | 12288 | 4096 | 256 | 732.94 | 688.98 | +6.38% |
| IQ4_XS | 17408 | 5120 | 256 | 1196.69 | 1135.66 | +5.37% |
| Q6_K | 12288 | 4096 | 256 | 935.83 | 885.70 | +5.66% |
| Q6_K | 17408 | 5120 | 256 | 1541.45 | 1476.55 | +4.40% |
| Q6_K | 12288 | 4096 | 1024 | 3264.06 | 3089.57 | +5.65% |
| Q6_K | 17408 | 5120 | 1024 | 5511.86 | 5270.61 | +4.58% |
| Q6_K | 12288 | 4096 | 2048 | 6532.27 | 6196.89 | +5.41% |
| Q6_K | 17408 | 5120 | 2048 | 11011.22 | 10437.29 | +5.50% |

All ten active-case min/max ranges are separate. The eight unchanged Q4_K/IQ4_XS cuBLAS controls at N1024/2048 have overlapping ranges; median changes are -0.40% to +0.47%. Do not add these graph gains to older kernel gains or call them whole-model gains.

Raw results and full min/max ranges: `sm75/shared-chat-results/kernel-results-20260909-071303/`. The same directory remains under the remote worktree. GPU settings were not changed; the idle governor was restored after the run.

Both installed 9B model files have 15 Q6_K, 5 Q5_K, and 12 Q4_K `ffn_down.weight` tensors, all with width 12288 and 4096 output rows. Thus 20 of 32 down projections can use this path at ubatch 2048; the Q4_K layers retain cuBLAS. Q5_K uses the same Q8_1 scale/sum layout as Q4_K. Extra numerical cases cover Q5_K and Q2_K, which exercises the third layout.

The expanded suite passed 50/50 with the feature off and 50/50 with it on. CUDA implementation files are unchanged from the timed build; only the test fixture grew. Evidence: remote `numerical-50-off.log` and `numerical-50-on.log`.

Compute Sanitizer was attempted on one Q6_K N32/K256 case. The numerical case passed, but the tool reported `Error: GPU debugging features are disabled` and exited nonzero. This is unavailable memory-check coverage, not a reported kernel memory violation and not a sanitizer pass. Evidence: `sanitizer.log`. No driver or GPU setting was changed to bypass this limit.

## Whole-model results and output gate

Completed control/candidate/control runs are saved in `shared-chat-results/model-results-20260909-072523/`. Each phase has one warmup and three measured requests with the same 16384 input tokens and 128 output tokens.

| Model | Control before PP tok/s | Fusion PP tok/s | Control after PP tok/s | Result |
| --- | ---: | ---: | ---: | --- |
| Ornith 9B UD-Q4_K_XL | 1960.23 | 1956.10 | 1939.32 | Within control drift |
| Qwen3.5 9B UD-Q4_K_XL | 1934.09 | 1946.02 | 1932.94 | +0.62% to +0.68%, subject to output caveat |

Peak GPU memory was unchanged: 7045 MiB for Ornith and 7031 MiB for Qwen. No whole-model memory saving is claimed. Decode medians stayed in the 65.87 to 66.37 tok/s range across both models and all phases; no decode gain is claimed.

The equal-output gate did not pass. All four responses within each phase differ, including both control phases, despite temperature 0 and seed 1234 in the returned settings. This establishes control-run variation, not its cause, and does not prove the candidate preserves model output. The timing data remain useful observations but are not an accepted equal-output model speedup. Numerical operator checks passed separately. Keep the feature opt-in pending a stable model correctness check.

The server hides GGML info logs at its default log level. A separate `-lv 5` trace confirms 20 of 32 FFN layers use fusion, with contiguous inputs and a passing fusion memory-range check. The other 12 Q4_K layers retain cuBLAS. Earlier absence of trace lines at the default level was not evidence of absent fusion. Evidence: `shared-chat-results/model-dispatch-trace.log`.

## Nsight check

Used the saved private Nsight Compute 2022.4.1.0 build 32308335 recipe on the current host. The executable SHA256 matches `ce0edb2afde930bc517a6a2da2cbcd5d53f4c8ec0947623e98ea478a8b444bd0`. The old matching section folder, `sudo -n`, and `--clock-control none` are required. Runner: `profile-shared-chat.sh`.

Explicit LaunchStats, Occupancy, SchedulerStats, WarpStateStats, SpeedOfLight, and MemoryWorkloadAnalysis sections collected real counters. `--set basic` found no metrics; a first explicit-section capture with the idle governor active ran at low clocks and is not timing evidence. The final capture paused the governor and restored it after the run.

The final report records the fused `quantize_mmq_q8_1` template with the SwiGLU boolean true: grid (2048,24,1), block (128,1,1), 28 registers/thread, SM frequency 2.07 GHz, DRAM throughput 523.75 GB/s (88.14%), and profiled duration 436.90 us. This is a profiled kernel observation, not an off/on speed comparison. Report and log: `shared-chat-results/fused-quant-metrics-p0.ncu-rep` and `.log`. The separate Compute Sanitizer limitation remains unresolved.

Final host check: idle governor active; no experiment server/test process left running. Production source and installed profiler were not changed. Experimental code remains in the isolated worktree, with opt-in trace diagnostics; it has not been promoted or committed.

## Memory-stage preparation

### Selected 27B IQ2 + MTP baseline

The user selected the installed 27B IQ2 + MTP setup on 2026-09-09. Baseline source is the isolated e653dac15 worktree with prefill fusion off. Files: `Qwen3.8-27B-UD-IQ2_XXS.gguf` and `mtp-Qwen3.8-27B-Q4_0.gguf`. Runner: `run-shared-chat-mtp.py`; raw evidence: `shared-chat-results/mtp-baseline-20260909-074348/`.

Settings: CUDA0, all model layers offloaded, context 4096, one slot, F16 KV, batch/ubatch 512, `--fit off`, `--spec-type draft-mtp --spec-draft-n-max 3`, PP1024, 128 output tokens, greedy sampling, seed 1234, no prompt cache. These are a small initial baseline, not a long-context capacity test. Log level 5 is held fixed; its overhead means absolute speed is not a deployment best-case result.

- One warmup and three measured requests completed. All four output texts are equal.
- Measured median PP: 457.27 tok/s (455.97 to 458.34).
- Measured median TG: 38.399 tok/s (38.349 to 38.407).
- Each request accepted 85 of 124 draft tokens, 68.55%.
- Sampled peak GPU memory: 8813 MiB out of 10240 MiB.
- Target recurrent memory: 598.50 MiB, comprising R 22.50 MiB and S 576.00 MiB; three rollback slots plus active state.
- Target/draft CUDA compute buffers: about 126.02 MiB each. These are distinct model objects, which must be considered before any shared-compute port.
- Idle governor was stopped for the run and restored in the runner's finally block. The experiment server exited normally.

Two initial launches failed before model load due to CLI names: `mtp` must be `draft-mtp`, and removed `--draft-max` must be `--spec-draft-n-max`. Corrected in the runner; no performance data from those launches are used.

The user approved the cross-layer design and further in-scope work. First scope: one SM75 CUDA device, one sequence, Qwen35 architecture with state dimension 128, two F32 S state planes plus per-token update records; replay the accepted prefix on partial rollback. The current snapshot path remains the default and fallback. No F16 snapshots, phase-arena reuse, or lazy replay are included. Shared compute is a separate opt-in experiment.

### Transaction implementation and checks

Opt-in: `LLAMA_SM75_GDN_TXN=1`. The kernel records the last `n_rs_seq + 1` updates and their prior state, including when prefill is larger than that window. S row zero is always current; R retains existing snapshot indexing. Replay waits for outstanding GPU writes before it runs. Sequence serialization reads current S separately from the pending R snapshot. State restore and clear invalidate old transaction metadata. Unsupported configurations use snapshots.

The new standalone `sm75/test-gdn-transaction.cpp` checks 20 forward cases and 80 replay prefixes (state dimensions 16/32/64/128, token counts 1/2/4/9/32, four log slots). All pass bit for bit on CUDA 13.3/SM75. Initial replay checks failed because the generated forward multiply-add order differs for small state dimensions: 16/32 round decay times state first; 64/128 round key times delta first. Explicit replay intrinsics matching that order passed. This is tested-build evidence, not a claim of parity on other CUDA toolchains or architectures. The production opt-in is gated to SM75 and dimension 128.

`sm75/test-mtp-state.cpp` checks the target model's accepted-prefix lengths 1 through 4 after a four-token verification batch, rejects a second pending rollback as the snapshot path does, checks save/load logits bitwise, and clears/restarts the sequence. Both modes pass. All nine resulting sequence-state files compare byte for byte between off and on. The files remain remote in `txn-state-off/` and `txn-state-on/`; logs are `txn-state-off.log` and `txn-state-on.log`. Prefix 1 corresponds to rejecting all three draft tokens while retaining the target token. Operator replay tests also cover retaining zero logged updates.

First model candidate: `mtp-baseline-20260909-075527`. All four texts equal the original baseline; acceptance stays 85/124. Peak GPU memory falls from 8813 to 8533 MiB (280 MiB). Target S is 288 instead of 576 MiB; the update log costs about 9.04 MiB. Total recurrent allocation is 319.54 instead of 598.50 MiB. This establishes a memory saving, not a speed gain.

Fresh off/on/off runs: `mtp-baseline-20260909-080032`, `080104`, `080135`. Median TG is 38.422 / 38.356 / 38.015 tok/s and PP is 453.420 / 449.417 / 448.536 tok/s. Control drift spans the candidate result, so no speed improvement is accepted. One SM75 guard build failed due to a misspelled constant (`CC_TURING`); corrected to `GGML_CUDA_CC_TURING`, and the next build completed before these runs.

The benchmark runner now reuses the original saved request token IDs. It must not tokenize changing source files for later comparisons. Fusion stays off during memory tests.

### Shared compute experiment

Separate opt-in: `LLAMA_SM75_MTP_SHARED_COMPUTE=1`. Scope is a Qwen35 MTP context and its target, one SM75 device, one sequence, matching scheduler devices/buffer types, no pipeline copies. Model weights and memory are not shared. Host compute buffers stay separate. Device buffers have shared ownership; growth detaches an allocator rather than invalidating the other allocator's tensors.

The context handoff covers encode/decode, including allocation, input writes, compute, and queued output copies. A new owner synchronizes the old owner before use. Shared buffer ownership is independent of context destruction order. Shared scheduler use must remain serialized; this is not support for concurrent graph execution on one arena. No VMM or sparse-memory feature is included.

`sm75/test-shared-compute.cpp` passes alternating graph execution, buffer growth from either allocator, use after the other allocator is freed, and both destruction orders. Evidence: `shared-chat-results/shared-allocator.log`. The test was then extended to select the larger of two reserved buffers after discarding old graphs.

The first shared run (`mtp-baseline-20260909-081204`) kept identical text and acceptance but still used 8533 MiB, so its enabled log is not proof of a saving. Setup initially shared before final capacity was known. The draft buffer is slightly larger than the target buffer, so the allocator now selects the larger fully reserved buffer and invalidates both old graphs before replacement. A second run (`081613`) still had no extra saving: enabling MTP hidden-state output triggers a later scheduler rebuild. The context pair now repeats sharing after scheduler reservation, while removing dead context pointers on destruction.

The completed final-setup run (`081835`) shows sharing after both reservations. All four texts and all 85/124 acceptance counts equal the baseline. Steady GPU memory is 8405 MiB, versus 8533 MiB with transactions alone and 8813 MiB with both options off: 408 MiB saved in total. The 200 ms telemetry also sampled a peak of 8405 MiB in this run, but it can miss short allocation transients while two buffers are being reserved; this is not a guarantee of the instantaneous startup peak.

Final standalone logs: `txn-operator-final.log` (80 bitwise replay and 20 forward checks), `shared-allocator-final.log` (including larger-buffer selection), and `txn-gdn-regression-final.log` (36/36 existing backend tests). These are copied under `shared-chat-results/`. Builds and `git diff --check` pass; Git emits normal LF/CRLF warnings on this Windows checkout. A check with `core.autocrlf` temporarily overridden falsely treated unchanged CRLF files as whole-file edits; the normal project configuration was used for the valid check. No line-ending normalization was applied.

### Final control / both options / control

Same final build, fixed original request IDs, one warmup and three measured requests per phase:

| Run suffix | Transactions | Shared compute | PP median tok/s | TG median tok/s | GPU memory MiB |
| --- | --- | --- | ---: | ---: | ---: |
| 081951 | off | off | 449.380 | 38.220 | 8813 |
| 082023 | on | on | 446.190 | 38.005 | 8405 |
| 082054 | off | off | 445.147 | 37.841 | 8813 |

All twelve texts equal the original baseline. Every response accepts 85 of 124 draft tokens. The candidate speed lies between the two controls; no speed gain is claimed. The accepted result is 408 MiB less working GPU memory on this 27B IQ2 + MTP / single CMP50HX setup. Raw files are under `shared-chat-results/mtp-baseline-20260909-<suffix>/`.

Use both environment variables with the isolated experiment build:

```sh
LLAMA_SM75_GDN_TXN=1 LLAMA_SM75_MTP_SHARED_COMPUTE=1 ./build/bin/llama-server ...
```

Both options are off by default. Prefill fusion remains a separate opt-in and was off throughout these memory tests. No production checkout, installed service, GPU clocks, or driver was changed. The idle governor is active again and GPU memory is back to 0 MiB after the test processes exit. No commit or push was made. Scope limits: only this SM75/CUDA toolchain and model setup were measured; no multi-GPU, multi-sequence, long-context capacity, or sanitizer coverage is claimed.

- The donor published these features together in `1a63cbf12`; transaction logging is not an isolated commit. Its recurrent source/header and CUDA GDN changes total 1239 added and 127 removed lines relative to the common base. This count includes related features and is not the size of a minimal transaction-log port.
- Keep only the transaction log first. Phase-arena reuse and lazy replay need separate results. Required tests include accept-all, reject-all, partial prefix, repeated rollback, and state save/load.
- Shared compute requires allocator buffer ownership, scheduler compatibility checks, buffer-growth detachment, and target/draft synchronization. Sparse VMM and host-KV are separate features and are outside this port.
- The donor hands off shared compute at `graph_compute`. Our `process_ubatch` sets inputs before that call. A port must acquire buffer ownership before allocation/input writes, not only before kernel launch. This is a source-level ordering concern, not a reproduced failure.
- The older intended MTP model was 27B Q4 on two 20 GiB cards. The user has now selected the current single 10 GiB card and installed 27B IQ2 target plus draft. Do not compare its absolute speed or memory use directly with the older two-card Q4 setup.
- Independent prefill model checks can use the two installed 9B UD-Q4_K_XL models from the existing single-card test setup. Runner: `run-shared-chat-model.py`; control/candidate/control, PP16384, 128 output tokens, greedy sampling, F16 KV, full GPU placement, one warmup and three measured requests per phase.

## Commit review, 2026-09-09

The user requested logical commits and a push. Work is grouped on `perf/sm75-shared-chat-20260909`, based on fork commit `e653dac15`. The main worktree's staged work is not included.

- `bd4df3e34`: opt-in prefill fusion, existing backend test cases, paired kernel runner.
- `59cbd8f6f`: opt-in recurrent replay, operator checks, full-model state checks.
- `73be57915`: opt-in shared compute buffers and allocator checks.
- A separate documentation commit holds this journal, model runners, profiler recipe, fixed MTP request, and test instructions.

KISS/DRY review: fusion reuses the existing Q8_1 kernel and MMQ path; replay uses the existing recurrent state flow; sharing uses reference counts on the existing buffers. Two temporary dispatch-check trace blocks were removed. The success trace remains useful for proving that fusion ran. Empty-buffer selection now prefers an existing buffer over an absent zero-size peer. The allocator contract states that allocated graphs must be discarded before sharing. No new pool, VMM, lazy replay, or general benchmark framework is added. Device and shape gates remain narrow and all options stay off by default.

Fresh checks after review: `llama-server` and `test-backend-ops` build passed; fusion-on 50/50 passed; replay 80 bitwise checks and 20 forward checks passed; shared-buffer alternation, growth, free-order, and larger-buffer checks passed. All committed Python runners pass syntax checks.

The final candidate smoke run `mtp-baseline-20260909-083610` uses the committed fixed request and both memory options. All four texts match the original baseline and all accept 85 of 124 draft tokens. Sampled GPU memory reaches 8405 MiB. Measured medians are PP 451.462 tok/s and TG 38.604 tok/s. This is a smoke check, not a new paired speed claim. The accepted paired result above remains 408 MiB less working GPU memory with no proven speed gain.

See `SHARED-CHAT-TESTS.md` for reproduction commands. Large raw logs, model files, saved states, and profiler reports stay outside Git. Earlier statements that no commit or push was made describe the state at the end of the experiment, before this review.
