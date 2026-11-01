#include "sm75-tuning.cuh"
#include "sm75-q8.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>

static unsigned checks = 0;
#define CHECK(expr) do { ++checks; if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; \
} } while (0)

static float decode_positive_half(uint16_t bits) {
    return std::ldexp(float(1024 + (bits & 1023)), int((bits >> 10) & 31) - 25);
}

int main(int argc, char ** argv) {
    using namespace ggml_cuda_sm75;
    if (argc == 2 && std::strcmp(argv[1], "--env") == 0) {
        const auto & o = get_options();
        std::printf("%s %lld %llu %d %d\n", mode_name(o.mode), (long long) o.min_batch,
                    (unsigned long long) o.max_scratch_mib, o.mmq_max_j, int(o.trace));
        return 0;
    }

    uint64_t value = 999;
    CHECK(parse_uint("512", 512, 1048576, value) && value == 512);
    CHECK(parse_uint("18446744073709551615", 0, UINT64_MAX, value) && value == UINT64_MAX);
    CHECK(!parse_uint("18446744073709551616", 0, UINT64_MAX, value));
    CHECK(!parse_uint("-1", 0, 128, value));
    CHECK(!parse_uint("+32", 0, 128, value));
    CHECK(!parse_uint(" 32", 0, 128, value));
    CHECK(!parse_uint("32 ", 0, 128, value));
    CHECK(!parse_uint("32x", 0, 128, value));
    CHECK(!parse_uint("", 0, 128, value));
    CHECK(!parse_uint(nullptr, 0, 128, value));
    CHECK(!parse_uint("511", 512, 1048576, value));
    CHECK(!parse_uint("1048577", 512, 1048576, value));
    CHECK(parse_uint("0", 0, 0, value) && value == 0);
    CHECK(!parse_uint("1", 0, 0, value));

    // Exhaustively compare the default with the old predicate over a shape grid.
    options o;
    const int64_t dims[] = {1, 128, 256, 2048, 4096, 5120, 12288, 17408};
    const int64_t batches[] = {1, 2, 3, 8, 127, 128, 511, 512, 1023, 1024, 1025, 1536, 2048, 2049, 4096};
    for (int64_t k : dims) {
        for (int64_t m : dims) {
            for (int64_t n : batches) {
                for (int t = 0; t != 4; ++t) {
                    const bool legacy_type = t < 2;
                    const bool old = legacy_type &&
                        ((k == 4096 && m == 12288) || (k == 12288 && m == 4096) ||
                         (k == 5120 && m == 17408) || (k == 17408 && m == 5120)) &&
                        n >= 1024 && n <= 2048;
                    CHECK(use_cublas(o, legacy_type, true, k, m, n) == old);
                }
            }
        }
    }

    o.mode = prefill_mode::mmq;
    CHECK(!use_cublas(o, true, true, 4096, 12288, 1024));
    o.mode = prefill_mode::cublas;
    CHECK(use_cublas(o, true, true, 4096, 12288, 1024));
    CHECK(use_cublas(o, false, true, 4096, 4096, 1024)); // Q5_K / Q6_K
    CHECK(!use_cublas(o, false, false, 4096, 4096, 1024));
    CHECK(!use_cublas(o, true, true, 4096, 12288, 512));
    o.min_batch = 512;
    CHECK(use_cublas(o, true, true, 4096, 12288, 512));
    CHECK(!use_cublas(o, true, true, 4096, 12288, 511));
    CHECK(!use_cublas(o, true, true, 4096, 12288, 1025));
    CHECK(!use_cublas(o, true, true, 4097, 12288, 1024));
    CHECK(!use_cublas(o, true, true, 4096, 12289, 1024));
    CHECK(!use_cublas(o, true, true, 4096, 12288, 1));
    o.max_scratch_mib = 1;
    CHECK(!use_cublas(o, true, true, 4096, 12288, 1024));

    CHECK(scratch_fits(256, 128, 512, 1)); // 917504 bytes
    CHECK(!scratch_fits(256, 256, 512, 1)); // 1310720 bytes
    CHECK(!scratch_fits(0, 128, 512, 512));
    CHECK(!scratch_fits(-1, 128, 512, 512));
    CHECK(!scratch_fits(256, 128, 512, 0));
    CHECK(!scratch_fits(256, 128, 512, UINT64_MAX));
    CHECK(!scratch_fits(INT64_MAX, INT64_MAX, INT64_MAX, 65536));
    // Exact budget edge: 4*(512*512 + 512*256 + 512*256) == 2 MiB.
    CHECK(scratch_fits(512, 512, 256, 2));
    CHECK(!scratch_fits(512, 512, 257, 2));

    CHECK(mmq_j_limit(o, true, 1024) == 128);
    for (int cap : {32, 64, 96, 128}) {
        o.mmq_max_j = cap;
        CHECK(mmq_j_limit(o, true, 1024) == cap);
        CHECK(mmq_j_limit(o, false, 1024) == 128);
        CHECK(mmq_j_limit(o, true, 1) == 128);
        CHECK(mmq_j_limit(o, true, 511) == 128);
        CHECK(mmq_j_limit(o, true, 1025) == 128);
    }

    // All 65,536 ordered INT8 pairs, using the exact bit-packing routine called
    // by the CUDA helper. This is a host arithmetic test, not a GPU kernel test.
    for (unsigned a = 0; a < 256; ++a) {
        for (unsigned b = 0; b < 256; ++b) {
            const uint32_t bits = ggml_cuda_sm75_q8_pair_bits(uint8_t(a), uint8_t(b));
            CHECK(decode_positive_half(uint16_t(bits)) - 1152.0f == float(a < 128 ? int(a) : int(a) - 256));
            CHECK(decode_positive_half(uint16_t(bits >> 16)) - 1152.0f == float(b < 128 ? int(b) : int(b) - 256));
        }
    }
    std::printf("PASS: %u checks; includes all 65536 ordered INT8 pairs\n", checks);
    return 0;
}
