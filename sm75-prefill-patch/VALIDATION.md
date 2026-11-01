# Validation report

Base: `af084752abb72652d51785602ef9536f631fec9d` in `xrip/llama.cpp-avx1-numa-sm75`.

## Completed

| Check | Result |
| --- | --- |
| C++17 host policy and bit-packing test, GCC, strict warnings | PASS: 134,968 checks |
| Same host test, Clang, strict warnings | PASS: 134,968 checks |
| Same host test, GCC AddressSanitizer + UndefinedBehaviorSanitizer | PASS; no reported errors |
| Fresh-process environment parsing cases | PASS: 18 cases |
| Ordered INT8 pairs through the actual shared bit-packing function | PASS: all 65,536 pairs |
| Host IEEE binary16 reference: each INT8 code x each finite FP16 scale | PASS: 16,252,928 products |
| Old policy comparison against the new default | PASS over the tested shape/type/batch grid |
| Exact old lines in every patch hunk and its context | PASS: 104 lines checked against retrieved source ranges |
| `git apply --check`, apply, reverse-check, reverse | PASS on source-context fixtures; resulting bytes compared |
| Added-line whitespace checks | PASS |

The host tests compile the actual new headers. The binary16 reference is a CPU test, not an execution of CUDA intrinsics. The scale check includes signed zero, subnormals, and the full finite FP16 range. The unexecuted CUDA test additionally includes infinities and NaN and is built with fast math.

## Limitations

The complete repository was not cloned into the execution environment: network clone/download was unavailable. Source ranges and original blob hashes were read using the GitHub connector. The application round trip used fixtures containing those exact ranges, not a full source checkout. No unread source line or placeholder is included in the patch. Run `verify-patch.py` or `git apply --check` on the real local checkout before applying.

Not run: nvcc compilation, the full CUDA backend build, the standalone CUDA test, actual SM75 execution, model/backend numerical or perplexity tests, CUDA Graph replay, two-GPU or tensor-split tests, long-context memory tests, or performance/VRAM measurements.

The patch is an opt-in experiment. It does not establish a speedup, a memory-availability guarantee, or production readiness. Defaults preserve the old dispatcher choices and leave the new Q8 device helper off. The MMQ cap tests existing configurations, not a new tuned kernel table.

The detailed machine-readable results are in `validation-results.json`. The separate `validation` directory contains test sources; it is not added to the repository's test suite or CI.
