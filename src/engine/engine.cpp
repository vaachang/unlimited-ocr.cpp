#include "uocr/engine.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "uocr/log.h"
#if defined(UOCR_CUDA_ENABLED)
#include "uocr/gpu_decoder.h"
#include "uocr/gpu_encoder.h"
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

    // Derive pool sizes from the configured budget if not provided.  The CUDA
    // path manages its own device KV cache, so the host block manager only
    // needs a small bookkeeping arena there; sizing it from max_seq_len *
    // max_batch_size would try to host-allocate tens of GB.
    std::size_t pool_bytes = ecfg_.memory_pool_bytes;
    if (pool_bytes == 0) {
        if (backend_ == Backend::CUDA) {
            pool_bytes = 64u << 20;
        } else {
            const std::size_t per_req = kv_bytes_per_request(ecfg_.max_seq_len);
            pool_bytes = per_req * static_cast<std::size_t>(ecfg_.max_batch_size) * 2;
        }
    }
    const std::size_t prefix_budget = pool_bytes / 2;
    const std::size_t ring_budget = pool_bytes / 2;
    block_mgr_ = std::make_unique<BlockManager>(prefix_budget, ring_budget);
    scheduler_ = std::make_unique<ContinuousBatchScheduler>(ecfg_.max_batch_size, ecfg_.min_batch_size);
    sampler_ = std::make_unique<Sampler>(SamplingParams{ecfg_.temperature, ecfg_.top_p, ecfg_.top_k});
#if defined(UOCR_CUDA_ENABLED)
    if (backend_ == Backend::CUDA) {
        gpu_decoder_ = std::make_unique<cuda::GpuDecoder>(mcfg_, weights_);
        gpu_decoder_->set_use_graph(ecfg_.use_cuda_graph);
        gpu_decoder_->set_prefill_device_moe(ecfg_.use_device_moe_prefill);
        gpu_decoder_->set_grouped_moe_bn(ecfg_.grouped_moe_bn);
        gpu_decoder_->set_grouped_moe_bm(ecfg_.grouped_moe_bm);
        if (ecfg_.graph_scope == "attn_dense")
            gpu_decoder_->set_graph_scope(cuda::GpuDecoder::GraphScope::kAttnDense);
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

std::vector<GenerationResult> Engine::generate_batch(const std::vector<std::vector<int>>& prompts,
                                                     int max_new_tokens) {
    const int limit = max_new_tokens > 0 ? max_new_tokens : ecfg_.max_new_tokens;
    std::vector<GenerationResult> results(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i)
        results[i].prefill_tokens = static_cast<int>(prompts[i].size());
    if (prompts.empty()) return results;

#if defined(UOCR_CUDA_ENABLED)
    if (gpu_decoder_) {
        const int slots = std::min<int>(ecfg_.max_batch_size,
                                        static_cast<int>(prompts.size()));
        int max_prompt = 1;
        for (const auto& p : prompts) max_prompt = std::max(max_prompt, static_cast<int>(p.size()));
        gpu_decoder_->batch_configure(slots, max_prompt + mcfg_.sliding_window);

        scheduler_ = std::make_unique<ContinuousBatchScheduler>(slots, ecfg_.min_batch_size);
        std::vector<int> ids;
        ids.reserve(prompts.size());
        for (const auto& p : prompts) {
            Request r;
            r.prompt_tokens = p;
            r.max_new_tokens = limit;
            ids.push_back(scheduler_->add_request(std::move(r)));
        }

        std::unordered_map<int, int> id_slot;
        std::vector<int> free_slots(slots);
        std::iota(free_slots.begin(), free_slots.end(), 0);
        std::unordered_map<int, std::vector<int>> history;

        auto release = [&](int id) {
            auto it = id_slot.find(id);
            if (it != id_slot.end()) {
                free_slots.push_back(it->second);
                id_slot.erase(it);
            }
        };
        auto sample = [&](std::vector<float>& lg, std::vector<int>& hist) {
            if (ecfg_.no_repeat_ngram_size > 0 &&
                static_cast<int>(hist.size()) >= ecfg_.no_repeat_ngram_size)
                Sampler::apply_no_repeat_ngram(lg.data(), mcfg_.vocab_size, hist,
                                               ecfg_.no_repeat_ngram_size, ecfg_.ngram_window);
            return sampler_->sample(lg.data(), mcfg_.vocab_size);
        };

        const double t0 = now_ms();
        while (scheduler_->has_work()) {
            Batch b = scheduler_->build_batch();
            // Admit / prefill new requests.  All requests admitted in this
            // step are prefilled together as one ragged forward pass: the MoE
            // expert GEMMs then see O(sum P_i) rows instead of one prompt at a
            // time, and the per-request serial prefill wall-clock disappears.
            if (!b.prefill.empty()) {
                std::vector<int> pf_ids, starts, lengths, pf_slots;
                std::vector<float> embeds;
                const int h = mcfg_.hidden_size;
                for (int id : b.prefill) {
                    Request* r = scheduler_->get(id);
                    if (!r) continue;
                    UOCR_CHECK(!free_slots.empty(), "generate_batch: no free slot");
                    const int slot = free_slots.back();
                    free_slots.pop_back();
                    id_slot[id] = slot;
                    const int P = static_cast<int>(r->prompt_tokens.size());
                    starts.push_back(static_cast<int>(embeds.size()) / h);
                    lengths.push_back(P);
                    pf_slots.push_back(slot);
                    pf_ids.push_back(id);
                    embeds.resize(embeds.size() + static_cast<std::size_t>(P) * h);
                    float* dst = embeds.data() + static_cast<std::size_t>(starts.back()) * h;
                    for (int t = 0; t < P; ++t)
                        weights_.embed_tokens.row(r->prompt_tokens[t],
                                                  dst + static_cast<std::size_t>(t) * h);
                }
                if (!pf_ids.empty()) {
                    std::vector<std::vector<float>> lgs;
                    const double p0 = now_ms();
                    gpu_decoder_->batch_prefill_embeds(embeds.data(), starts, lengths, pf_slots,
                                                       lgs);
                    const double pf_ms = now_ms() - p0;
                    for (std::size_t r = 0; r < pf_ids.size(); ++r) {
                        const int id = pf_ids[r];
                        Request* req = scheduler_->get(id);
                        if (!req || r >= lgs.size()) continue;
                        results[id].ttft_ms = pf_ms;
                        history[id] = req->prompt_tokens;
                        scheduler_->mark_prefilled(id);
                        const int tok = sample(lgs[r], history[id]);
                        if (tok == mcfg_.eos_token_id) {
                            scheduler_->finish(id);
                            release(id);
                            continue;
                        }
                        history[id].push_back(tok);
                        if (scheduler_->add_token(id, tok)) release(id);
                    }
                }
            }
            // Decode every running request together.
            if (!b.decode.empty()) {
                std::vector<int> dec_ids, toks, poss, slots;
                dec_ids.reserve(b.decode.size());
                toks.reserve(b.decode.size());
                poss.reserve(b.decode.size());
                slots.reserve(b.decode.size());
                for (int id : b.decode) {
                    Request* r = scheduler_->get(id);
                    if (!r || r->output_tokens.empty()) continue;
                    auto sit = id_slot.find(id);
                    if (sit == id_slot.end()) continue;
                    dec_ids.push_back(id);
                    toks.push_back(r->output_tokens.back());
                    poss.push_back(static_cast<int>(r->prompt_tokens.size() +
                                                    r->output_tokens.size() - 1));
                    slots.push_back(sit->second);
                }
                if (!toks.empty()) {
                    std::vector<std::vector<float>> lgs;
                    gpu_decoder_->batch_decode(toks, poss, slots, lgs);
                    const std::size_t n = std::min(dec_ids.size(), lgs.size());
                    for (std::size_t k = 0; k < n; ++k) {
                        const int id = dec_ids[k];
                        Request* r = scheduler_->get(id);
                        if (!r) continue;
                        const int tok = sample(lgs[k], history[id]);
                        if (tok == mcfg_.eos_token_id) {
                            scheduler_->finish(id);
                            release(id);
                            continue;
                        }
                        history[id].push_back(tok);
                        if (scheduler_->add_token(id, tok)) release(id);
                    }
                }
            }
        }
        const double t1 = now_ms();
        auto stats = scheduler_->stats();
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const Request* r = scheduler_->get(ids[i]);
            if (!r) continue;
            results[i].tokens = r->output_tokens;
            results[i].decode_tokens = static_cast<int>(r->output_tokens.size());
            results[i].prefill_tokens = static_cast<int>(r->prompt_tokens.size());
            results[i].decode_ms = t1 - t0;
            results[i].tpot_ms = r->output_tokens.empty()
                                     ? 0.0
                                     : (t1 - t0) / static_cast<double>(r->output_tokens.size());
        }
        UOCR_INFO("batch done: %llu requests, %llu decode tokens, %.1f ms",
                  static_cast<unsigned long long>(stats.total_requests),
                  static_cast<unsigned long long>(stats.decode_tokens), t1 - t0);
        return results;
    }
#endif

    // CPU fallback: sequential.
    for (std::size_t i = 0; i < prompts.size(); ++i)
        results[i] = generate(prompts[i], limit);
    return results;
}

std::vector<float> Engine::image_embeddings(const ImageRGB& image, bool crop_mode, int base_size,
                                            int image_size) {
#if defined(UOCR_CUDA_ENABLED)
    const bool have_gpu_vision = static_cast<bool>(gpu_vision_);
#else
    const bool have_gpu_vision = false;
#endif
    UOCR_CHECK(vision_ != nullptr || have_gpu_vision,
               "image_embeddings: vision encoder not set");
    if (base_size <= 0) base_size = mcfg_.base_size;
    if (image_size <= 0) image_size = mcfg_.candidate_image_size;
    const int hidden = mcfg_.hidden_size;
    const int patch = mcfg_.patch_size;
    const int ds = mcfg_.downsample_ratio;
    const int nq = static_cast<int>(std::ceil(static_cast<double>(image_size / patch) / ds));
    const int nqb = static_cast<int>(std::ceil(static_cast<double>(base_size / patch) / ds));

    auto encode_view = [&](const ImageRGB& view, int size) {
        std::vector<float> chw = to_tensor_normalized(view);
        Tensor out;
#if defined(UOCR_CUDA_ENABLED)
        if (gpu_vision_) {
            gpu_vision_->encode(chw.data(), size, size, out);
            return out;
        }
#endif
        vision_->encode(chw.data(), size, size, out);
        return out;
    };

    std::vector<float> result;

    if (!crop_mode) {
        ImageRGB view = resize_bicubic(image, image_size, image_size);
        view = pad_square(view, image_size, 127);
        Tensor g = encode_view(view, image_size);
        result.assign(g.data(), g.data() + g.numel());
        UOCR_CHECK(result.size() / static_cast<std::size_t>(hidden) ==
                       static_cast<std::size_t>(nq * (nq + 1) + 1),
                   "image_embeddings: single-view token count does not match layout formula");
        return result;
    }

    // global view at base_size (crop_mode=True)
    ImageRGB gview = pad_square(image, base_size, 127);
    Tensor global = encode_view(gview, base_size);

    // Reference `infer`: images that already fit within image_size use a single
    // global view (crop_ratio [1,1]); otherwise run dynamic_preprocess.
    const bool fits = image.width <= image_size && image.height <= image_size;
    DynamicPreprocess dp;
    if (!fits) dp = dynamic_preprocess(image, image_size);
    const bool use_local = !fits && (dp.width_crop_num > 1 || dp.height_crop_num > 1);
    if (!use_local) {
        result.assign(global.data(), global.data() + global.numel());
        UOCR_CHECK(result.size() / static_cast<std::size_t>(hidden) ==
                       static_cast<std::size_t>(nqb * (nqb + 1) + 1),
                   "image_embeddings: global token count does not match layout formula");
        return result;
    }

    // Multi-view (Gundam) assembly, matching reference `UnlimitedOCRModel`:
    //   * the local crops are laid out on the (height_crop_num x width_crop_num)
    //     grid and flattened to (hcn*nq) x (wcn*nq) BEFORE a single
    //     `image_newline` is appended to every merged row;
    //   * the global view (which already carries its own row newlines) follows;
    //   * a final `view_seperator` closes the sequence.
    // Simply concatenating the per-crop encodings would emit one newline per
    // individual crop instead, yielding a different token sequence and
    // desynchronising the prompt layout, so re-stitch the raw patch tokens.
    const int gl = nq;  // local grid edge
    const int wcn = dp.width_crop_num;
    const int hcn = dp.height_crop_num;
    UOCR_CHECK(static_cast<int>(dp.crops.size()) == wcn * hcn,
               "image_embeddings: crop count does not match crop grid");
    const int merged_rows = hcn * gl;
    const int merged_cols = wcn * gl;

    std::vector<float> merged(static_cast<std::size_t>(merged_rows) * merged_cols * hidden);
    const int lf_cols = gl + 1;  // raw tokens + one trailing image_newline per row
    for (int r = 0; r < hcn; ++r) {
        for (int c = 0; c < wcn; ++c) {
            Tensor lf = encode_view(dp.crops[static_cast<std::size_t>(r) * wcn + c], image_size);
            UOCR_CHECK(static_cast<int>(lf.dim(0)) == gl * lf_cols + 1,
                       "image_embeddings: unexpected local view token count");
            for (int ry = 0; ry < gl; ++ry) {
                const float* src = lf.data() + static_cast<std::size_t>(ry * lf_cols) * hidden;
                float* dst = merged.data() + (static_cast<std::size_t>(r * gl + ry) * merged_cols +
                                              static_cast<std::size_t>(c) * gl) *
                                                 hidden;
                std::memcpy(dst, src, static_cast<std::size_t>(gl) * hidden * sizeof(float));
            }
        }
    }

    const int global_total = static_cast<int>(global.dim(0));
    UOCR_CHECK(global_total == nqb * (nqb + 1) + 1,
               "image_embeddings: unexpected global view token count");
    const int global_raw = global_total - 1;  // drop only the view_seperator
    const int rows_total = merged_rows;
    const int cols_total = merged_cols + 1;

    result.assign(
        static_cast<std::size_t>(rows_total * cols_total + global_raw + 1) * hidden,
        0.0f);
    std::size_t pos = 0;
    for (int r = 0; r < rows_total; ++r) {
        std::memcpy(result.data() + pos * hidden,
                    merged.data() + static_cast<std::size_t>(r) * merged_cols * hidden,
                    static_cast<std::size_t>(merged_cols) * hidden * sizeof(float));
        pos += static_cast<std::size_t>(merged_cols);
        std::memcpy(result.data() + pos * hidden, weights_.image_newline.data(),
                    static_cast<std::size_t>(hidden) * sizeof(float));
        pos += 1;
    }
    std::memcpy(result.data() + pos * hidden, global.data(),
                static_cast<std::size_t>(global_raw) * hidden * sizeof(float));
    pos += static_cast<std::size_t>(global_raw);
    std::memcpy(result.data() + pos * hidden, weights_.view_seperator.data(),
                static_cast<std::size_t>(hidden) * sizeof(float));
    UOCR_CHECK(result.size() / static_cast<std::size_t>(hidden) ==
                   static_cast<std::size_t>(nqb * (nqb + 1) + 1 +
                                           (nq * wcn + 1) * (nq * hcn)),
               "image_embeddings: multi-view token count does not match layout formula");
    return result;
}

std::vector<ImageSpatialCrop> Engine::image_crops(const ImageRGB& image, bool crop_mode,
                                                  int image_size) const {
    std::vector<ImageSpatialCrop> crops;
    if (!crop_mode) {
        crops.push_back(ImageSpatialCrop{1, 1});
        return crops;
    }
    if (image_size <= 0) image_size = mcfg_.candidate_image_size;
    if (image.width <= image_size && image.height <= image_size) {
        crops.push_back(ImageSpatialCrop{1, 1});
        return crops;
    }
    DynamicPreprocess dp = dynamic_preprocess(image, image_size);
    crops.push_back(ImageSpatialCrop{dp.width_crop_num, dp.height_crop_num});
    return crops;
}

}  // namespace uocr
