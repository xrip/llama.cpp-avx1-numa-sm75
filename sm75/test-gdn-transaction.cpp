#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda-gdn-transaction.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) return 2;
    auto backend = ggml_backend_dev_init(dev, nullptr);
    auto replay = (ggml_cuda_gdn_replay_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_cuda_gdn_replay");
    if (!backend || !replay) return 2;
    int checks = 0;
    for (int dim : {16, 32, 64, 128}) {
        for (int tokens : {1, 2, 4, 9, 32}) {
            const int heads = 4;
            const int slots = 4;
            const int state_size = heads * dim * dim;
            auto ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
            auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, 2, tokens, 1);
            auto k = ggml_dup_tensor(ctx, q);
            auto v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, tokens, 1);
            auto g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, tokens, 1);
            auto b = ggml_dup_tensor(ctx, g);
            auto s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, dim, heads, 1);
            auto log = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, heads * (1 + 2 * dim), slots);
            auto prior = ggml_dup_tensor(ctx, s);
            auto state = ggml_dup_tensor(ctx, s);
            auto ref = ggml_gated_delta_net(ctx, q, k, v, g, b, s, tokens + 1);
            auto txn = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
            txn->src[6] = log;
            txn->src[7] = prior;
            auto gf_ref = ggml_new_graph(ctx);
            auto gf_txn = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf_ref, ref);
            ggml_build_forward_expand(gf_txn, txn);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) return 2;
            ggml_backend_buffer_clear(buffer, 0);
            std::mt19937 rng(1234 + tokens);
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
            std::vector<float> initial;
            for (auto tensor : {q, k, v, g, b, s}) {
                std::vector<float> data(ggml_nelements(tensor));
                for (auto & x : data) x = dist(rng);
                if (tensor == g) for (auto & x : data) x = -0.1f + x;
                if (tensor == b) for (auto & x : data) x += 0.5f;
                if (tensor == s) initial = data;
                ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
            }
            if (ggml_backend_graph_compute(backend, gf_ref) != GGML_STATUS_SUCCESS ||
                ggml_backend_graph_compute(backend, gf_txn) != GGML_STATUS_SUCCESS) return 2;
            const int attn = dim * heads * tokens;
            std::vector<float> expected(ggml_nelements(ref));
            std::vector<float> actual(ggml_nelements(txn));
            ggml_backend_tensor_get(ref, expected.data(), 0, ggml_nbytes(ref));
            ggml_backend_tensor_get(txn, actual.data(), 0, ggml_nbytes(txn));
            if (memcmp(expected.data(), actual.data(), actual.size() * sizeof(float))) {
                std::printf("FAIL forward dim=%d tokens=%d\n", dim, tokens);
                return 1;
            }
            const int logged = std::min(tokens, slots);
            for (int keep = logged; keep >= 0; --keep) {
                ggml_cuda_gdn_replay_args args = {(const float *) prior->data, (float *) state->data,
                    (const float *) log->data, heads, dim, keep};
                if (!replay(&args, 1)) return 2;
                std::vector<float> got(state_size);
                ggml_backend_tensor_get(state, got.data(), 0, ggml_nbytes(state));
                const float * want = keep == 0 && logged == tokens ? initial.data() :
                    expected.data() + attn + (logged - keep) * state_size;
                if (memcmp(want, got.data(), state_size * sizeof(float))) {
                    std::printf("FAIL replay dim=%d tokens=%d keep=%d\n", dim, tokens, keep);
                    return 1;
                }
                ++checks;
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    std::printf("PASS: %d bitwise replay checks and 20 forward checks\n", checks);
}
