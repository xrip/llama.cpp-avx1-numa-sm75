#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok) {
    if (!ok) std::exit(1);
}

struct graph {
    ggml_context * ctx;
    ggml_tensor * input;
    ggml_tensor * output;
    ggml_cgraph * gf;
    graph(int size) {
        ctx = ggml_init({1024 * 1024, nullptr, true});
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
        ggml_set_input(input);
        output = ggml_scale(ctx, input, 2.0f);
        ggml_set_output(output);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, output);
    }
    ~graph() { ggml_free(ctx); }
    void check(ggml_backend_t backend, float value) {
        std::vector<float> data(ggml_nelements(input), value);
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(output, data.data(), 0, ggml_nbytes(output));
        for (float x : data) require(x == 2 * value);
    }
};

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("CUDA0", nullptr);
    require(backend != nullptr);
    auto buft = ggml_backend_get_default_buffer_type(backend);
    for (int grow_dst : {0, 1}) {
        auto src = ggml_gallocr_new(buft);
        auto dst = ggml_gallocr_new(buft);
        graph a(1024), b(1024), big(1024 * 1024);
        require(ggml_gallocr_alloc_graph(src, a.gf));
        require(ggml_gallocr_share_buffers(dst, src));
        require(ggml_gallocr_alloc_graph(dst, b.gf));
        require(a.input->buffer == b.input->buffer);
        for (int i = 1; i <= 10; ++i) {
            a.check(backend, i);
            b.check(backend, -i);
        }
        require(ggml_gallocr_alloc_graph(grow_dst ? dst : src, big.gf));
        require(big.input->buffer != (grow_dst ? a.input : b.input)->buffer);
        big.check(backend, 0.5f);
        (grow_dst ? a : b).check(backend, 4);
        ggml_gallocr_free(grow_dst ? dst : src);
        (grow_dst ? a : b).check(backend, 5);
        ggml_gallocr_free(grow_dst ? src : dst);
    }
    for (int free_dst : {0, 1}) {
        auto src = ggml_gallocr_new(buft);
        auto dst = ggml_gallocr_new(buft);
        graph a(1024), b(1024);
        require(ggml_gallocr_alloc_graph(src, a.gf));
        require(ggml_gallocr_share_buffers(dst, src));
        require(ggml_gallocr_alloc_graph(dst, b.gf));
        ggml_gallocr_free(free_dst ? dst : src);
        (free_dst ? a : b).check(backend, 7);
        ggml_gallocr_free(free_dst ? src : dst);
    }
    for (int larger_dst : {0, 1}) {
        auto src = ggml_gallocr_new(buft);
        auto dst = ggml_gallocr_new(buft);
        {
            graph a(larger_dst ? 1024 : 2048), b(larger_dst ? 2048 : 1024);
            require(ggml_gallocr_alloc_graph(src, a.gf));
            require(ggml_gallocr_alloc_graph(dst, b.gf));
        }
        require(ggml_gallocr_share_buffers(dst, src));
        graph a(1024), b(1024);
        require(ggml_gallocr_alloc_graph(src, a.gf));
        require(ggml_gallocr_alloc_graph(dst, b.gf));
        require(a.input->buffer == b.input->buffer);
        a.check(backend, 9);
        b.check(backend, 10);
        ggml_gallocr_free(src);
        b.check(backend, 11);
        ggml_gallocr_free(dst);
    }
    ggml_backend_free(backend);
    std::puts("PASS: shared buffers, alternating graphs, growth detachment, both free orders, larger-buffer selection");
}
