#pragma once

// Opt-in host policy. Keep environment variables fixed while the process runs.
// CUDA Graphs can retain the selected kernels.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ggml_cuda_sm75 {

enum class prefill_mode { automatic, mmq, cublas };

struct options {
    prefill_mode mode = prefill_mode::automatic;
    int64_t min_batch = 1024;
    uint64_t max_scratch_mib = 512;
    int mmq_max_j = 0;
    bool trace = false;
};

// Reject overflow, signs, whitespace, and trailing text.
inline bool parse_uint(const char * text, uint64_t lo, uint64_t hi, uint64_t & value) {
    if (!text || !*text || lo > hi) {
        return false;
    }
    uint64_t n = 0;
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        const uint64_t digit = uint64_t(*p - '0');
        if (digit > hi || n > (hi - digit) / 10) {
            return false;
        }
        n = 10*n + digit;
    }
    if (n < lo) {
        return false;
    }
    value = n;
    return true;
}

inline void warn_env(const char * name, const char * value) {
    std::fprintf(stderr, "ggml_cuda_sm75: invalid %s='%s'; using its default\n", name, value ? value : "(unset)");
}

inline uint64_t env_uint(const char * name, uint64_t fallback, uint64_t lo, uint64_t hi) {
    const char * text = std::getenv(name);
    if (!text) {
        return fallback;
    }
    uint64_t value;
    if (!parse_uint(text, lo, hi, value)) {
        warn_env(name, text);
        return fallback;
    }
    return value;
}

inline options read_options() {
    options o;
    const char * mode = std::getenv("GGML_CUDA_SM75_PREFILL_MODE");
    if (mode) {
        if (std::strcmp(mode, "mmq") == 0) {
            o.mode = prefill_mode::mmq;
        } else if (std::strcmp(mode, "cublas") == 0) {
            o.mode = prefill_mode::cublas;
        } else if (std::strcmp(mode, "auto") != 0) {
            warn_env("GGML_CUDA_SM75_PREFILL_MODE", mode);
        }
    }
    o.min_batch = int64_t(env_uint("GGML_CUDA_SM75_PREFILL_MIN_BATCH", 1024, 512, 1048576));
    o.max_scratch_mib = env_uint("GGML_CUDA_SM75_PREFILL_MAX_SCRATCH_MIB", 512, 1, 65536);
    o.mmq_max_j = int(env_uint("GGML_CUDA_SM75_MMQ_MAX_J", 0, 0, 128));
    if (o.mmq_max_j != 0 && o.mmq_max_j != 32 && o.mmq_max_j != 64 &&
        o.mmq_max_j != 96 && o.mmq_max_j != 128) {
        warn_env("GGML_CUDA_SM75_MMQ_MAX_J", std::getenv("GGML_CUDA_SM75_MMQ_MAX_J"));
        o.mmq_max_j = 0;
    }
    o.trace = env_uint("GGML_CUDA_SM75_PREFILL_TRACE", 0, 0, 1) != 0;
    return o;
}

inline const options & get_options() {
    static const options o = read_options();
    return o;
}

inline const char * mode_name(prefill_mode mode) {
    switch (mode) {
        case prefill_mode::automatic: return "auto";
        case prefill_mode::mmq:       return "mmq";
        case prefill_mode::cublas:    return "cublas";
    }
    return "auto";
}

inline bool legacy_ffn_shape(int64_t k, int64_t m) {
    return (k == 4096 && m == 12288) || (k == 12288 && m == 4096) ||
           (k == 5120 && m == 17408) || (k == 17408 && m == 5120);
}

// Bound conversion scratch by 4*(M*K + K*N + M*N), also covering FP32 overrides.
// Excludes cuBLAS workspace and cached pool memory. Check products before multiplication.
inline bool scratch_fits(int64_t k, int64_t m, int64_t n, uint64_t limit_mib) {
    if (k <= 0 || m <= 0 || n <= 0 || limit_mib == 0 || limit_mib > 65536) {
        return false;
    }
    uint64_t budget = limit_mib * (1024u * 1024u / 4u);
    const uint64_t a[] = { uint64_t(m), uint64_t(k), uint64_t(m) };
    const uint64_t b[] = { uint64_t(k), uint64_t(n), uint64_t(n) };
    for (int i = 0; i < 3; ++i) {
        if (a[i] > budget / b[i]) {
            return false;
        }
        budget -= a[i]*b[i];
    }
    return true;
}

// Caller checks SM75, dense shapes, F32 I/O, contiguous inputs, and device weights.
// legacy_type: Q4_K/IQ4_XS; experiment_type: also Q5_K/Q6_K.
inline bool use_cublas(const options & o, bool legacy_type, bool experiment_type,
                       int64_t k, int64_t m, int64_t n) {
    if (o.mode == prefill_mode::automatic) {
        // Exactly the pre-existing af084752 policy, including its tail behavior.
        return legacy_type && legacy_ffn_shape(k, m) && n >= 1024 && n <= 2048;
    }
    if (o.mode == prefill_mode::mmq) {
        return false;
    }
    return experiment_type && n >= o.min_batch && n >= 512 && n % 128 == 0 &&
           k >= 256 && m >= 128 && k % 256 == 0 && m % 128 == 0 &&
           scratch_fits(k, m, n, o.max_scratch_mib);
}

// Select existing specializations only; do not change layouts or buffer sizes.
inline int mmq_j_limit(const options & o, bool eligible, int64_t n) {
    return eligible && n >= 512 && n % 128 == 0 && o.mmq_max_j != 0 ? o.mmq_max_j : 128;
}

} // namespace ggml_cuda_sm75
