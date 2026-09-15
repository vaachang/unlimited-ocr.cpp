#include "uocr/engine.h"

#include <chrono>
#include <filesystem>
#include <limits>

#include "uocr/log.h"

namespace uocr {

namespace {
double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}
}  // namespace

Engine::Engine(ModelConfig mcfg, EngineConfig ecfg, DecoderWeights weights, Backend backend)
    : mcfg_(std::move(mcfg)), ecfg_(std::move(ecfg)), weights_(std::move(weights)),
      backend_(backend) {
    decoder_ = std::make_unique<MoEDecoder>(mcfg_, weights_);

    // Derive pool sizes from the configured budget if not provided.
    const std::size_t per_req = kv_bytes_per_request(ecfg_.max_seq_len);
    const std::size_t prefix_budget = ecfg_.memory_pool_bytes > 0
                                          ? ecfg_.memory_pool_bytes / 2
                                          : per_req * static_cast<std::size_t>(ecfg_.max_batch_size);
    const std::size_t ring_budget = ecfg_.memory_pool_bytes > 0
                                        ? ecfg_.memory_pool_bytes / 2
                                        : per_req * static_cast<std::size_t>(ecfg_.max_batch_size);
    block_mgr_ = std::make_unique<BlockManager>(prefix_budget, ring_budget);
    scheduler_ = std::make_unique<ContinuousBatchScheduler>(ecfg_.max_batch_size, ecfg_.min_batch_size);
    sampler_ = std::make_unique<Sampler>(SamplingParams{ecfg_.temperature, ecfg_.top_p, ecfg_.top_k});
    UOCR_INFO("engine created (%s backend)", backend_ == Backend::CUDA ? "CUDA" : "CPU");
}

std::unique_ptr<Engine> Engine::load(const EngineConfig& ecfg, Backend backend) {
    namespace fs = std::filesystem;
    const fs::path dir(ecfg.model_dir);
    ModelConfig mcfg = ModelConfig::from_json_file((dir / "config.json").string());
    if (mcfg.image_token_id == 0) mcfg.image_token_id = 128815;

    std::string weights_path = ecfg.weights_file;
    if (weights_path.empty()) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".safetensors") {
                weights_path = entry.path().string();
                break;
            }
        }
    }
    UOCR_CHECK(!weights_path.empty(), "no .safetensors checkpoint found in " + ecfg.model_dir);

    DecoderWeights w =
        DecoderWeights::load(weights_path, mcfg, ecfg.use_int4_experts, ecfg.int4_group_size);
    return std::make_unique<Engine>(mcfg, ecfg, std::move(w), backend);
}

std::size_t Engine::kv_bytes_per_request(int prefill_len) const {
    const int cap = prefill_len + mcfg_.sliding_window;
    const std::size_t per_layer =
        static_cast<std::size_t>(cap) * mcfg_.num_key_value_heads * mcfg_.head_dim();
    return per_layer * mcfg_.num_hidden_layers * 2 * sizeof(float);  // K and V
}

std::vector<int> Engine::find_image_token_positions(const std::vector<int>& tokens,
                                                    int image_token_id) {
    std::vector<int> pos;
    for (std::size_t i = 0; i < tokens.size(); ++i)
        if (tokens[i] == image_token_id) pos.push_back(static_cast<int>(i));
    return pos;
}

GenerationResult Engine::generate(const std::vector<int>& prompt, int max_new_tokens,
                                  const std::string& doc_key) {
    GenerationResult res;
    res.prefill_tokens = static_cast<int>(prompt.size());

    auto cache = std::make_shared<RSWACache>(mcfg_.num_hidden_layers, mcfg_.num_key_value_heads,
                                             mcfg_.head_dim(), mcfg_.sliding_window);

    std::vector<float> logits;
    const double t0 = now_ms();
    decoder_->prefill(*cache, prompt, 0, logits);
    const double t1 = now_ms();
    res.ttft_ms = t1 - t0;

    if (!doc_key.empty()) {
        block_mgr_->acquire_prefix(doc_key, static_cast<int>(prompt.size()),
                                   kv_bytes_per_request(static_cast<int>(prompt.size())));
    }

    const int limit = max_new_tokens > 0 ? max_new_tokens : ecfg_.max_new_tokens;
    std::vector<int> history = prompt;
    const double td0 = now_ms();
    int pos = static_cast<int>(prompt.size());
    for (int step = 0; step < limit; ++step) {
        if (ecfg_.no_repeat_ngram_size > 0 && static_cast<int>(history.size()) >= ecfg_.no_repeat_ngram_size)
            Sampler::apply_no_repeat_ngram(logits.data(), mcfg_.vocab_size, history,
                                           ecfg_.no_repeat_ngram_size, ecfg_.ngram_window);
        const int tok = sampler_->sample(logits.data(), mcfg_.vocab_size);
        if (tok == mcfg_.eos_token_id) break;
        res.tokens.push_back(tok);
        history.push_back(tok);
        if (step + 1 >= limit) break;
        decoder_->decode(*cache, tok, pos, logits);
        ++pos;
    }
    const double td1 = now_ms();
    res.decode_ms = td1 - td0;
    res.decode_tokens = static_cast<int>(res.tokens.size());
    res.tpot_ms = res.decode_tokens > 0 ? res.decode_ms / res.decode_tokens : 0.0;

    if (!doc_key.empty()) block_mgr_->release_prefix(doc_key);
    return res;
}

}  // namespace uocr
