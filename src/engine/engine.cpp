#include "uocr/engine.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>

#include "uocr/log.h"
#if defined(UOCR_CUDA_ENABLED)
#include "uocr/gpu_decoder.h"
#endif

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
#if defined(UOCR_CUDA_ENABLED)
    if (backend_ == Backend::CUDA) {
        gpu_decoder_ = std::make_unique<cuda::GpuDecoder>(mcfg_, weights_);
        gpu_decoder_->set_use_graph(ecfg_.use_cuda_graph);
    }
#endif
    UOCR_INFO("engine created (%s backend)", backend_ == Backend::CUDA ? "CUDA" : "CPU");
}

Engine::~Engine() = default;

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

#if defined(UOCR_CUDA_ENABLED)
    if (gpu_decoder_) {
        std::vector<float> logits;
        const double t0 = now_ms();
        gpu_decoder_->prefill_tokens(prompt, logits);
        const double t1 = now_ms();
        res.ttft_ms = t1 - t0;
        const int limit = max_new_tokens > 0 ? max_new_tokens : ecfg_.max_new_tokens;
        std::vector<int> history = prompt;
        const double td0 = now_ms();
        int pos = static_cast<int>(prompt.size());
        for (int step = 0; step < limit; ++step) {
            if (ecfg_.no_repeat_ngram_size > 0 &&
                static_cast<int>(history.size()) >= ecfg_.no_repeat_ngram_size)
                Sampler::apply_no_repeat_ngram(logits.data(), mcfg_.vocab_size, history,
                                               ecfg_.no_repeat_ngram_size, ecfg_.ngram_window);
            const int tok = sampler_->sample(logits.data(), mcfg_.vocab_size);
            if (tok == mcfg_.eos_token_id) break;
            res.tokens.push_back(tok);
            history.push_back(tok);
            if (step + 1 >= limit) break;
            gpu_decoder_->decode_token(tok, pos, logits);
            ++pos;
        }
        const double td1 = now_ms();
        res.decode_ms = td1 - td0;
        res.decode_tokens = static_cast<int>(res.tokens.size());
        res.tpot_ms = res.decode_tokens > 0 ? res.decode_ms / res.decode_tokens : 0.0;
        return res;
    }
#endif

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

GenerationResult Engine::generate_from_image(const std::vector<int>& prompt,
                                             const std::vector<std::uint8_t>& images_seq_mask,
                                             const std::vector<float>& visual_embeddings,
                                             int hidden, int max_new_tokens,
                                             const std::string& doc_key) {
    UOCR_CHECK(images_seq_mask.size() == prompt.size(),
               "generate_from_image: mask size must match prompt size");
    const int seq = static_cast<int>(prompt.size());
    UOCR_CHECK(hidden == mcfg_.hidden_size, "generate_from_image: hidden size mismatch");

    Tensor inputs = decoder_->embed(prompt);
    int v = 0;
    const int total_v = hidden > 0 ? static_cast<int>(visual_embeddings.size() / hidden) : 0;
    for (int i = 0; i < seq; ++i) {
        if (!images_seq_mask[static_cast<std::size_t>(i)]) continue;
        UOCR_CHECK(v < total_v, "generate_from_image: not enough visual embeddings");
        std::memcpy(inputs.data() + static_cast<std::size_t>(i) * hidden,
                    visual_embeddings.data() + static_cast<std::size_t>(v) * hidden,
                    static_cast<std::size_t>(hidden) * sizeof(float));
        ++v;
    }
    UOCR_CHECK(v == total_v, "generate_from_image: visual embedding count mismatch");

    GenerationResult res;
    res.prefill_tokens = seq;

#if defined(UOCR_CUDA_ENABLED)
    if (gpu_decoder_) {
        std::vector<float> logits;
        const double t0 = now_ms();
        gpu_decoder_->prefill_embeds(inputs.data(), seq, logits);
        const double t1 = now_ms();
        res.ttft_ms = t1 - t0;
        const int limit = max_new_tokens > 0 ? max_new_tokens : ecfg_.max_new_tokens;
        std::vector<int> history = prompt;
        const double td0 = now_ms();
        int pos = seq;
        for (int step = 0; step < limit; ++step) {
            if (ecfg_.no_repeat_ngram_size > 0 &&
                static_cast<int>(history.size()) >= ecfg_.no_repeat_ngram_size)
                Sampler::apply_no_repeat_ngram(logits.data(), mcfg_.vocab_size, history,
                                               ecfg_.no_repeat_ngram_size, ecfg_.ngram_window);
            const int tok = sampler_->sample(logits.data(), mcfg_.vocab_size);
            if (tok == mcfg_.eos_token_id) break;
            res.tokens.push_back(tok);
            history.push_back(tok);
            if (step + 1 >= limit) break;
            gpu_decoder_->decode_token(tok, pos, logits);
            ++pos;
        }
        const double td1 = now_ms();
        res.decode_ms = td1 - td0;
        res.decode_tokens = static_cast<int>(res.tokens.size());
        res.tpot_ms = res.decode_tokens > 0 ? res.decode_ms / res.decode_tokens : 0.0;
        return res;
    }
#endif
    auto cache = std::make_shared<RSWACache>(mcfg_.num_hidden_layers, mcfg_.num_key_value_heads,
                                             mcfg_.head_dim(), mcfg_.sliding_window);

    std::vector<float> logits;
    const double t0 = now_ms();
    decoder_->prefill_embeds(*cache, inputs, logits);
    const double t1 = now_ms();
    res.ttft_ms = t1 - t0;

    if (!doc_key.empty())
        block_mgr_->acquire_prefix(doc_key, seq, kv_bytes_per_request(seq));

    const int limit = max_new_tokens > 0 ? max_new_tokens : ecfg_.max_new_tokens;
    std::vector<int> history = prompt;
    const double td0 = now_ms();
    int pos = seq;
    for (int step = 0; step < limit; ++step) {
        if (ecfg_.no_repeat_ngram_size > 0 &&
            static_cast<int>(history.size()) >= ecfg_.no_repeat_ngram_size)
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

std::vector<float> Engine::image_embeddings(const ImageRGB& image, bool crop_mode, int base_size,
                                            int image_size) {
    UOCR_CHECK(vision_ != nullptr, "image_embeddings: vision encoder not set");
    if (base_size <= 0) base_size = mcfg_.base_size;
    if (image_size <= 0) image_size = mcfg_.candidate_image_size;
    const int hidden = mcfg_.hidden_size;
    const int patch = mcfg_.patch_size;
    const int ds = mcfg_.downsample_ratio;

    auto encode_view = [&](const ImageRGB& view, int size) {
        std::vector<float> chw = to_tensor_normalized(view);
        Tensor out;
        vision_->encode(chw.data(), size, size, out);
        return out;
    };

    std::vector<float> result;

    if (!crop_mode) {
        ImageRGB view = resize_bicubic(image, image_size, image_size);
        view = pad_square(view, image_size, 127);
        Tensor g = encode_view(view, image_size);
        result.assign(g.data(), g.data() + g.numel());
        return result;
    }

    // global view at base_size (crop_mode=True)
    ImageRGB gview = pad_square(image, base_size, 127);
    Tensor global = encode_view(gview, base_size);

    const int nq = static_cast<int>(
        std::ceil(static_cast<double>(image_size / patch) / ds));
    // Reference `infer`: images that already fit within image_size use a single
    // global view (crop_ratio [1,1]); otherwise run dynamic_preprocess.
    const bool fits = image.width <= image_size && image.height <= image_size;
    DynamicPreprocess dp;
    if (!fits) dp = dynamic_preprocess(image, image_size);
    const bool use_local = !fits && (dp.width_crop_num > 1 || dp.height_crop_num > 1);
    if (!use_local) {
        result.assign(global.data(), global.data() + global.numel());
        return result;
    }

    // local crops: each encode returns rows*(cols)+1; drop its view_seperator
    const int global_total = static_cast<int>(global.dim(0));
    const int global_no_sep = global_total - 1;
    for (const ImageRGB& crop : dp.crops) {
        Tensor lf = encode_view(crop, image_size);
        const int n = static_cast<int>(lf.dim(0)) - 1;  // drop view_seperator
        result.insert(result.end(), lf.data(), lf.data() + static_cast<std::size_t>(n) * hidden);
    }
    result.insert(result.end(), global.data(),
                  global.data() + static_cast<std::size_t>(global_no_sep) * hidden);
    result.insert(result.end(), weights_.view_seperator.begin(), weights_.view_seperator.end());
    (void)nq;
    return result;
}

}  // namespace uocr
