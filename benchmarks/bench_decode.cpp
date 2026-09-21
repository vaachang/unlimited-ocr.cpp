#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
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
    bool real = false;
    std::string model_dir = "models";
    int prefill_len = 512;
    int decode_steps = 256;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) real = true;
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--prefill") && i + 1 < argc) prefill_len = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) decode_steps = std::atoi(argv[++i]);
    }

    ModelConfig cfg;
    DecoderWeights weights;
    if (real) {
        EngineConfig ecfg;
        ecfg.model_dir = model_dir;
        // This CPU decode benchmark targets the INT4 expert path (the engine
        // default is now BF16); flip it back on explicitly.
        ecfg.use_int4_experts = true;
        EngineConfig cfg_copy = ecfg;
        cfg = ModelConfig::from_json_file(model_dir + "/config.json");
        weights = DecoderWeights::load(model_dir + "/model-00001-of-000001.safetensors", cfg,
                                       cfg_copy.use_int4_experts, cfg_copy.int4_group_size);
    } else {
        cfg = synthetic_config();
        weights = DecoderWeights::random(cfg, 1234);
    }

    MoEDecoder dec(cfg, weights);
    RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                    cfg.sliding_window);

    std::vector<int> prompt(prefill_len);
    for (int i = 0; i < prefill_len; ++i) prompt[i] = (i * 131) % cfg.vocab_size;

    std::vector<float> logits;
    double t0 = now_ms();
    dec.prefill(cache, prompt, 0, logits);
    double t1 = now_ms();
    std::printf("prefill: %d tokens in %.2f ms (%.2f tok/s)\n", prefill_len, t1 - t0,
                prefill_len / ((t1 - t0) / 1000.0));

    std::printf("decode: %d steps\n", decode_steps);
    double tpots[3] = {0, 0, 0};
    int pos = prefill_len;
    for (int step = 0; step < decode_steps; ++step) {
        const int tok = (step * 7) % cfg.vocab_size;
        double s0 = now_ms();
        dec.decode(cache, tok, pos, logits);
        double s1 = now_ms();
        if (step == 0) tpots[0] = s1 - s0;
        if (step == decode_steps / 2) tpots[1] = s1 - s0;
        if (step == decode_steps - 1) tpots[2] = s1 - s0;
        ++pos;
    }
    std::printf("TPOT (first): %.3f ms | (middle): %.3f ms | (last): %.3f ms\n", tpots[0],
                tpots[1], tpots[2]);
    std::printf("cache capacity per layer: %d slots, ring window W=%d\n", cache.capacity(),
                cfg.sliding_window);
    double cache_mb = static_cast<double>(cache.bytes()) / (1024.0 * 1024.0);
    std::printf("KV cache (all layers): %.2f MB\n", cache_mb);
    return 0;
}
