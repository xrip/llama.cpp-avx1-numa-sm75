#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void require(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

static std::vector<uint8_t> save(llama_context * ctx) {
    std::vector<uint8_t> data(llama_state_seq_get_size(ctx, 0));
    require(llama_state_seq_get_data(ctx, data.data(), data.size(), 0) == data.size(), "save");
    return data;
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    auto model = llama_model_load_from_file(argv[1], mp);
    require(model != nullptr, "model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 3;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "context");
    auto mem = llama_get_memory(ctx);
    auto batch = llama_batch_init(512, 0, 1);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto decode = [&](int start, int count) {
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = 100 + (start + i) % 1000;
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == count - 1;
        }
        require(llama_decode(ctx, batch) == 0, "decode");
    };
    auto record = [&](const std::string & name, const std::vector<uint8_t> & data) {
        std::ofstream file(std::string(argv[2]) + "/" + name, std::ios::binary);
        file.write((const char *) data.data(), data.size());
        require(file.good(), "write");
    };
    decode(0, 32);
    auto base = save(ctx);
    record("base.bin", base);
    for (int keep = 1; keep <= 4; ++keep) {
        require(llama_state_seq_set_data(ctx, base.data(), base.size(), 0) == base.size(), "restore base");
        decode(32, 4);
        if (keep < 4) require(llama_memory_seq_rm(mem, 0, 32 + keep, -1), "rollback");
        auto cut = save(ctx);
        record("cut-" + std::to_string(keep) + ".bin", cut);
        if (keep < 4) require(!llama_memory_seq_rm(mem, 0, 32 + keep - 1, -1), "reject second pending rollback");
        decode(32 + keep, 1);
        std::vector<float> logits(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + vocab);
        require(llama_state_seq_set_data(ctx, cut.data(), cut.size(), 0) == cut.size(), "restore cut");
        decode(32 + keep, 1);
        require(memcmp(logits.data(), llama_get_logits_ith(ctx, -1), vocab * sizeof(float)) == 0, "save/load logits parity");
        record("after-" + std::to_string(keep) + ".bin", save(ctx));
    }
    require(llama_memory_seq_rm(mem, 0, 0, -1), "clear sequence");
    decode(0, 32);
    require(save(ctx) == base, "clear and restart");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::puts("PASS: accepted-prefix rollback, pending rollback rejection, state save/load, clear and restart");
}
