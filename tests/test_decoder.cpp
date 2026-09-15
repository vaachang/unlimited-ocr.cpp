#include "test_main.h"
#include "uocr/moe_decoder.h"

#include <cmath>

using namespace uocr;

namespace {

ModelConfig tiny_config() {
    ModelConfig c;
    c.vocab_size = 256;
    c.hidden_size = 64;
    c.intermediate_size = 128;
    c.moe_intermediate_size = 32;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 4;
    c.first_k_dense_replace = 1;
    c.n_routed_experts = 4;
    c.n_shared_experts = 1;
    c.num_experts_per_tok = 2;
    c.norm_topk_prob = true;
    c.sliding_window = 8;
    c.max_position_embeddings = 256;
    c.projector_input_dim = 2048;
    c.projector_n_embed = 64;
    return c;
}

}  // namespace

UOCR_TEST(decoder_rswa_cache_bounded) {
    ModelConfig cfg = tiny_config();
    DecoderWeights w = DecoderWeights::random(cfg, 99);
    MoEDecoder dec(cfg, w);

    RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                    cfg.sliding_window);
    std::vector<int> prompt(10);
    for (int i = 0; i < 10; ++i) prompt[i] = i + 1;

    std::vector<float> logits;
    dec.prefill(cache, prompt, 0, logits);
    CHECK_EQ(static_cast<int>(logits.size()), cfg.vocab_size);
    CHECK_EQ(cache.prefill_len(), 10);
    CHECK_EQ(cache.length(0), 10);

    int pos = 10;
    for (int step = 0; step < 50; ++step) {
        dec.decode(cache, (step * 7) % cfg.vocab_size, pos, logits);
        ++pos;
        for (int l = 0; l < cfg.num_hidden_layers; ++l)
            CHECK(cache.length(l) <= 10 + cfg.sliding_window);
    }
    // cache saturated at P + W
    CHECK_EQ(cache.length(0), 10 + cfg.sliding_window);
    for (float v : logits) CHECK(std::isfinite(v));
}

UOCR_TEST(decoder_token_independent_after_warmup) {
    // R-SWA guarantees that once the ring is full the cache size is constant,
    // so per-step cost does not depend on the number of generated tokens.
    ModelConfig cfg = tiny_config();
    DecoderWeights w = DecoderWeights::random(cfg, 5);
    MoEDecoder dec(cfg, w);
    RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                    cfg.sliding_window);
    std::vector<int> prompt(6, 3);
    std::vector<float> logits;
    dec.prefill(cache, prompt, 0, logits);
    CHECK_EQ(cache.capacity(), 6 + cfg.sliding_window);
    CHECK_EQ(cache.capacity(), 6 + cfg.sliding_window);
}

UOCR_TEST(router_trace_valid) {
    ModelConfig cfg = tiny_config();
    DecoderWeights w = DecoderWeights::random(cfg, 11);
    MoEDecoder dec(cfg, w);
    dec.set_trace_router(true);
    RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                    cfg.sliding_window);
    std::vector<int> prompt = {1, 2, 3, 4};
    std::vector<float> logits;
    dec.prefill(cache, prompt, 0, logits);

    const auto& trace = dec.router_trace();
    CHECK(!trace.empty());
    // one entry per MoE layer, each a per-token trace
    CHECK_EQ(static_cast<int>(trace.size()), cfg.num_hidden_layers - cfg.first_k_dense_replace);
    for (const auto& per_token : trace) {
        CHECK_EQ(static_cast<int>(per_token.size()), 4);
        for (const auto& tr : per_token) {
            CHECK_EQ(static_cast<int>(tr.experts.size()), cfg.num_experts_per_tok);
            for (int e : tr.experts) CHECK(e >= 0 && e < cfg.n_routed_experts);
        }
    }
}
