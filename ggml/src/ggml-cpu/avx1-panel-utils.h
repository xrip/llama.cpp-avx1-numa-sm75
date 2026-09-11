#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__AVX__)
#include <immintrin.h>

static inline void ggml_avx1_store_transpose4(
        int8_t * dst, size_t stride, __m128i r0, __m128i r1, __m128i r2, __m128i r3) {
    const __m128i a0 = _mm_unpacklo_epi32(r0, r1);
    const __m128i a1 = _mm_unpackhi_epi32(r0, r1);
    const __m128i a2 = _mm_unpacklo_epi32(r2, r3);
    const __m128i a3 = _mm_unpackhi_epi32(r2, r3);
    _mm_storeu_si128((__m128i *) (dst + 0 * stride), _mm_unpacklo_epi64(a0, a2));
    _mm_storeu_si128((__m128i *) (dst + 1 * stride), _mm_unpackhi_epi64(a0, a2));
    _mm_storeu_si128((__m128i *) (dst + 2 * stride), _mm_unpacklo_epi64(a1, a3));
    _mm_storeu_si128((__m128i *) (dst + 3 * stride), _mm_unpackhi_epi64(a1, a3));
}

static inline void ggml_avx1_iqp_signed_x4(
        int8_t * dst,
        uint32_t g0a, uint32_t g0b, uint32_t g1a, uint32_t g1b,
        uint32_t g2a, uint32_t g2b, uint32_t g3a, uint32_t g3b,
        uint32_t signs) {
    const __m128i sel = _mm_set1_epi64x((int64_t) UINT64_C(0x8040201008040201));
    const __m128i sv = _mm_set1_epi32((int32_t) signs);
    const __m128i sl = _mm_shuffle_epi8(sv, _mm_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i sh = _mm_shuffle_epi8(sv, _mm_setr_epi8(
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m128i ml = _mm_cmpeq_epi8(_mm_and_si128(sl, sel), sel);
    const __m128i mh = _mm_cmpeq_epi8(_mm_and_si128(sh, sel), sel);
    const __m128i gl = _mm_setr_epi32((int32_t) g0a, (int32_t) g0b, (int32_t) g1a, (int32_t) g1b);
    const __m128i gh = _mm_setr_epi32((int32_t) g2a, (int32_t) g2b, (int32_t) g3a, (int32_t) g3b);
    _mm_storeu_si128((__m128i *) (dst +  0), _mm_sub_epi8(_mm_xor_si128(gl, ml), ml));
    _mm_storeu_si128((__m128i *) (dst + 16), _mm_sub_epi8(_mm_xor_si128(gh, mh), mh));
}

static inline void ggml_avx1_iqp_interleave8x32(
        int8_t * dst, const int8_t (*vals)[256], int off) {
    for (int h = 0; h < 2; ++h) {
        for (int r = 0; r < 8; r += 4) {
            ggml_avx1_store_transpose4(dst + h * 128 + r * 4, 32,
                    _mm_loadu_si128((const __m128i *) (vals[r + 0] + off + h * 16)),
                    _mm_loadu_si128((const __m128i *) (vals[r + 1] + off + h * 16)),
                    _mm_loadu_si128((const __m128i *) (vals[r + 2] + off + h * 16)),
                    _mm_loadu_si128((const __m128i *) (vals[r + 3] + off + h * 16)));
        }
    }
}

// Input quantization is unchanged. Only the Q8_Kx4 byte layout and sums are built here.
static inline void ggml_avx1_pack_q8kx4(
        int8_t * qs, int16_t * bsums, const int8_t (*in)[256]) {
    const __m128i ones8 = _mm_set1_epi8(1);
    const __m128i ones16 = _mm_set1_epi16(1);
    for (int off = 0; off < 256; off += 16) {
        const __m128i r0 = _mm_loadu_si128((const __m128i *) (in[0] + off));
        const __m128i r1 = _mm_loadu_si128((const __m128i *) (in[1] + off));
        const __m128i r2 = _mm_loadu_si128((const __m128i *) (in[2] + off));
        const __m128i r3 = _mm_loadu_si128((const __m128i *) (in[3] + off));
        ggml_avx1_store_transpose4(qs + off * 4, 16, r0, r1, r2, r3);
        const __m128i s0 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, r0), ones16);
        const __m128i s1 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, r1), ones16);
        const __m128i s2 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, r2), ones16);
        const __m128i s3 = _mm_madd_epi16(_mm_maddubs_epi16(ones8, r3), ones16);
        const __m128i sum = _mm_hadd_epi32(_mm_hadd_epi32(s0, s1), _mm_hadd_epi32(s2, s3));
        const __m128i s16 = _mm_packs_epi32(sum, _mm_setzero_si128());
        const int base = (off / 64) * 16 + (off / 16) % 4;
        bsums[base +  0] = (int16_t) _mm_extract_epi16(s16, 0);
        bsums[base +  4] = (int16_t) _mm_extract_epi16(s16, 1);
        bsums[base +  8] = (int16_t) _mm_extract_epi16(s16, 2);
        bsums[base + 12] = (int16_t) _mm_extract_epi16(s16, 3);
    }
}

#endif
