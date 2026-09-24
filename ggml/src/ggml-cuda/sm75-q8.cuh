#pragma once

#include <cstdint>

#if defined(__CUDACC__) && !defined(GGML_USE_HIP)
#include <cuda_fp16.h>
#define GGML_SM75_Q8_HD __host__ __device__
#else
#define GGML_SM75_Q8_HD
#endif

// An unsigned byte u in the mantissa of 0x6400 encodes 1024 + u exactly.
// Flipping the INT8 sign bit gives u = q + 128. Subtracting 1152 then recovers
// every q in [-128, 127], including -128. No float conversion or wide load.
GGML_SM75_Q8_HD static constexpr uint32_t ggml_cuda_sm75_q8_pair_bits(uint8_t a, uint8_t b) {
    return 0x64006400u | uint32_t(a ^ 0x80u) | (uint32_t(b ^ 0x80u) << 16);
}

#undef GGML_SM75_Q8_HD

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750 && !defined(GGML_USE_HIP)
static __device__ __forceinline__ half2 ggml_cuda_sm75_q8_pair_to_half2(int8_t a, int8_t b) {
    const uint32_t bits = ggml_cuda_sm75_q8_pair_bits(uint8_t(a), uint8_t(b));
    const half2 encoded = __halves2half2(__ushort_as_half(uint16_t(bits)),
                                        __ushort_as_half(uint16_t(bits >> 16)));
    // The explicit rounding intrinsic prevents a following scale multiply from
    // contracting with the subtraction. Keep the same multiply as the old path.
    return __hsub2_rn(encoded, __float2half2_rn(1152.0f));
}
#endif
