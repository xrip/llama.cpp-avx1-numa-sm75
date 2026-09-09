#pragma once

#include <cstdint>

struct ggml_cuda_gdn_replay_args {
    const float * prior;
    float * state;
    const float * log;
    int64_t head_count;
    int32_t state_dim;
    int32_t n_keep;
};

using ggml_cuda_gdn_replay_t = bool (*)(const ggml_cuda_gdn_replay_args *, int32_t);
