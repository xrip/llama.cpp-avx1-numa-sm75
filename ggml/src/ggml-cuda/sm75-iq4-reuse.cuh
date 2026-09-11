#pragma once

#include <cstddef>

#ifndef GGML_SM75_IQ4_REUSE
#define GGML_SM75_IQ4_REUSE 1
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_TURING && GGML_SM75_IQ4_REUSE

static_assert(QI4_XS == 32 && VDR_IQ4_XS_Q8_1_MMVQ == 4, "IQ4_XS reuse expects four words per lane");
static_assert(sizeof(block_iq4_xs) % 8 == 0, "IQ4_XS block stride must preserve 8-byte alignment");
static_assert(offsetof(block_iq4_xs, qs) % 8 == 0, "IQ4_XS quants must have 8-byte alignment");

struct sm75_iq4_decoded {
    int2 values[4];
    int scale;
    float d;
};

static __device__ __forceinline__ sm75_iq4_decoded sm75_iq4_decode(const void * vx, int kbx, int iqs) {
    const block_iq4_xs * b = (const block_iq4_xs *) vx + kbx;
    const int2 * packed = (const int2 *) (b->qs + sizeof(int) * iqs);
    const int2 p0 = packed[0];
    const int2 p1 = packed[1];
    const int words[4] = {p0.x, p0.y, p1.x, p1.y};
    const uint2 header = *(const uint2 *) b;
    sm75_iq4_decoded result;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        result.values[j] = get_int_from_table_16(words[j], kvalues_iq4nl);
    }
    const int ls = int(((header.y >> iqs) & 15) | (((header.x >> (16 + iqs / 2)) & 3) << 4));
    result.scale = ls - 32;
    result.d = __half2float(__ushort_as_half((uint16_t) header.x));
    return result;
}

static __device__ __forceinline__ float sm75_iq4_dot(const sm75_iq4_decoded & w, const block_q8_1 * y) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sumi = ggml_cuda_dp4a(w.values[j].x, get_int_b4(y->qs, j), sumi);
        sumi = ggml_cuda_dp4a(w.values[j].y, get_int_b4(y->qs, j + 4), sumi);
    }
    sumi *= w.scale;
    const float d = w.d * __low2float(y->ds);
    return d * sumi;
}

#endif
