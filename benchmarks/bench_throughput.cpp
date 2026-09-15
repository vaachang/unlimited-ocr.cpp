#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "uocr/engine.h"
#include "uocr/log.h"

using namespace uocr;

namespace {
double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

ModelConfig synthetic_config() {
    ModelConfig c;
    c.vocab_size = 4096;
    c.hidden_size = 256;
    c.intermediate_size = 512;
    c.moe_intermediate_size = 128;
    c.num_hidden_layers = 4;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 4;
    c.first_k_dense_replace = 1;
    c.n_routed_experts = 8;
    c.n_shared_experts = 2;
    c.num_experts_per_tok = 2;
    c.norm_topk_prob = true;
    c.sliding_window = 128;
    c.max_position_embeddings = 32768;
    return c;
}
}  // namespace

int main(int argc, char** argv) {
    int batch = 8;
    int new_tokens = 64;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) new_tokens = std::atoi(argv[++i]);
    }

    ModelConfig cfg = synthetic_config();
    EngineConfig ecfg;
    ecfg.max_batch_size = batch;
    DecoderWeights weights = DecoderWeights::random(cfg, 2024);
    Engine engine(cfg, ecfg, std::move(weights), Backend::CPU);

    std::vector<int> prompt(64);
    for (int i = 0; i < 64; ++i) prompt[i] = (i * 17) % cfg.vocab_size;

    double t0 = now_ms();
    long total_tokens = 0;
    for (int i = 0; i < batch; ++i) {
        GenerationResult r = engine.generate(prompt, new_tokens, "doc0");
        total_tokens += static_cast<long>(r.tokens.size());
    }
    double t1 = now_ms();
    std::printf("batch=%d, new_tokens<=%d, generated %ld tokens in %.1f ms -> %.1f tok/s\n",
                batch, new_tokens, total_tokens, t1 - t0,
                total_tokens / ((t1 - t0) / 1000.0));

    auto st = engine.block_manager().stats();
    std::printf("prefix shared bytes: %zu, ring peak bytes: %zu\n", st.prefix_shared_bytes,
                st.ring_peak_bytes);
    return 0;
}
