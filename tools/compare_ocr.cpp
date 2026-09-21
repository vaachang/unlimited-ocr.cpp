// compare_ocr -- end-to-end image OCR alignment (E3/E4).
//
// Consumes the reference manifest produced by
//   tools/reference/export_reference.py --mode ocr
// and checks, in order:
//   1. prompt / <image> layout (input_ids + images_seq_mask)
//   2. image preprocessing (normalized global view)
//   3. assembled visual embeddings (Engine::image_embeddings)
//   4. prefill logits after scattering the visual embeddings
//   5. greedy decode tokens / logits

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "uocr/config.h"
#if defined(UOCR_CUDA_ENABLED)
#include "uocr/gpu_decoder.h"
#include "uocr/gpu_encoder.h"
#endif
#include "uocr/deep_encoder.h"
#include "uocr/engine.h"
#include "uocr/image.h"
#include "uocr/kv_cache.h"
#include "uocr/log.h"
#include "uocr/prompt.h"
#include "uocr/safetensors.h"
#include "uocr/sampler.h"
#include "uocr/tokenizer.h"
#include "uocr/weights.h"

using namespace uocr;
using nlohmann::json;

namespace {

std::vector<float> load_f32(const std::string& dir, const json& e) {
    i64 n = 1;
    for (const auto& d : e.at("shape")) n *= d.get<i64>();
    std::vector<float> v(static_cast<std::size_t>(n));
    std::ifstream f(dir + "/" + e.at("file").get<std::string>(), std::ios::binary);
    UOCR_CHECK(f.good(), "cannot open tensor bin");
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return v;
}

std::vector<int> load_i32(const std::string& dir, const json& e) {
    i64 n = 1;
    for (const auto& d : e.at("shape")) n *= d.get<i64>();
    std::vector<float> v = load_f32(dir, e);
    std::vector<int> out(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = static_cast<int>(std::lround(v[i]));
    return out;
}

// Prints the comparison and returns the relative L2 error (or -1 on size mismatch).
double report(const std::string& name, const std::vector<float>& a, const std::vector<float>& b,
              bool print = true) {
    if (a.size() != b.size()) {
        std::printf("%-22s SIZE MISMATCH ours=%zu ref=%zu\n", name.c_str(), a.size(), b.size());
        return -1.0;
    }
    double max_abs = 0, num = 0, den = 0;
    std::size_t top1_ours = 0, top1_ref = 0;
    double best_o = -1e30, best_r = -1e30;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
        max_abs = std::max(max_abs, e);
        num += e * e;
        den += static_cast<double>(b[i]) * b[i];
        if (a[i] > best_o) { best_o = a[i]; top1_ours = i; }
        if (b[i] > best_r) { best_r = b[i]; top1_ref = i; }
    }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    if (print)
        std::printf("%-22s max_abs=%.6g rel_l2=%.6g top1(ours=%zu ref=%zu)%s\n", name.c_str(), max_abs,
                    rel, top1_ours, top1_ref,
                    top1_ours == top1_ref ? "" : "  <-- TOP1 DIFF");
    return rel;
}

std::vector<std::string> split_on(const std::string& s, const std::string& sep) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (true) {
        const std::size_t p = s.find(sep, pos);
        if (p == std::string::npos) { out.push_back(s.substr(pos)); break; }
        out.push_back(s.substr(pos, p - pos));
        pos = p + sep.size();
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models";
    std::string ref_dir = "/tmp/opencode/ref_ocr";
    bool gpu_vision = false;
    bool gpu = false;
    bool int4 = false;
    bool strict = false;
    double visual_tol = 0.15;   // bf16/f32 drift is ~0.06
    double logits_tol = 0.15;
    int int4_group = 128;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--gpu-vision")) gpu_vision = true;
        else if (!std::strcmp(argv[i], "--gpu")) gpu = true;
        else if (!std::strcmp(argv[i], "--int4")) int4 = true;
        else if (!std::strcmp(argv[i], "--strict")) strict = true;
        else if (!std::strcmp(argv[i], "--visual-tol") && i + 1 < argc)
            visual_tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--logits-tol") && i + 1 < argc)
            logits_tol = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--int4-group") && i + 1 < argc)
            int4_group = std::atoi(argv[++i]);
    }

    std::ifstream mf(ref_dir + "/manifest.json");
    UOCR_CHECK(mf.good(), "cannot open " + ref_dir + "/manifest.json");
    json manifest = json::parse(mf);
    const json& T = manifest.at("tensors");

    ModelConfig cfg = ModelConfig::from_json_file(model_dir + "/config.json");
    Tokenizer tok = Tokenizer::from_file(model_dir + "/tokenizer.json");
    const std::string prompt = manifest.at("prompt").get<std::string>();
    const bool crop_mode = manifest.at("crop_mode").get<bool>();

    // ---- 2. image preprocessing ----
    const int h = manifest.at("image_hw")[0].get<int>();
    const int w = manifest.at("image_hw")[1].get<int>();
    ImageRGB image;
    image.width = w;
    image.height = h;
    image.pixels.resize(static_cast<std::size_t>(w) * h * 3);
    {
        std::ifstream f(ref_dir + "/ocr_image.bin", std::ios::binary);
        UOCR_CHECK(f.good(), "cannot open ocr_image.bin");
        f.read(reinterpret_cast<char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
    }
    std::vector<float> ref_global = load_f32(ref_dir, T.at("image_global"));
    std::vector<float> our_global;
    if (crop_mode) {
        our_global = to_tensor_normalized(pad_square(image, cfg.base_size, 127));
    } else {
        ImageRGB sq = resize_bicubic(image, cfg.candidate_image_size, cfg.candidate_image_size);
        our_global = to_tensor_normalized(pad_square(sq, cfg.candidate_image_size, 127));
    }
    report("image_global", our_global, ref_global);

    // ---- 3. visual embeddings (Engine) ----
    EngineConfig ecfg;
    ecfg.model_dir = model_dir;
    ecfg.use_int4_experts = int4;
    ecfg.int4_group_size = int4_group;
    ecfg.max_seq_len = 1024;
    ecfg.memory_pool_bytes = 1 << 20;
#if defined(UOCR_CUDA_ENABLED)
    auto engine = Engine::load(ecfg, gpu ? Backend::CUDA : Backend::CPU);
    cuda::GpuDecoder* gdec = gpu ? engine->gpu_decoder() : nullptr;
    UOCR_CHECK(!gpu || gdec != nullptr, "--gpu: engine did not create a CUDA decoder");
#else
    UOCR_CHECK(!gpu, "--gpu requires a CUDA build (ENGINE_BACKEND=CUDA)");
    auto engine = Engine::load(ecfg, Backend::CPU);
#endif
#if defined(UOCR_CUDA_ENABLED)
    std::printf("decoder backend        : %s\n",
                gdec ? (int4 ? "CUDA/INT4" : "CUDA/BF16") : "CPU/f32");
#else
    std::printf("decoder backend        : CPU/f32\n");
#endif

    // ---- 1. layout (same crop grid the vision path will use) ----
    std::vector<ImageSpatialCrop> crops = engine->image_crops(image, crop_mode);
    PromptLayout layout = build_ocr_prompt(tok, split_on(prompt, "<image>"), crops, cfg, crop_mode);
    std::vector<int> ref_ids = load_i32(ref_dir, T.at("input_ids"));
    std::vector<int> ref_mask = load_i32(ref_dir, T.at("images_seq_mask"));
    bool layout_ok = layout.input_ids == ref_ids &&
                     layout.images_seq_mask.size() == ref_mask.size();
    for (std::size_t i = 0; layout_ok && i < ref_mask.size(); ++i)
        if (layout.images_seq_mask[i] != static_cast<std::uint8_t>(ref_mask[i])) layout_ok = false;
    std::printf("layout                 : ids=%zu mask_true=%zu  %s\n", layout.input_ids.size(),
                static_cast<std::size_t>(std::count(layout.images_seq_mask.begin(),
                                                    layout.images_seq_mask.end(), 1)),
                layout_ok ? "OK" : "MISMATCH");

    const std::string weights_path = model_dir + "/model-00001-of-000001.safetensors";
    std::printf("loading vision weights (this takes a while) ...\n");
    SafetensorsFile st(weights_path);
    VisionWeights vw = VisionWeights::load(st, cfg);
    DecoderWeights dw = DecoderWeights::load(weights_path, cfg, false, 128);
#if defined(UOCR_CUDA_ENABLED)
    if (gpu_vision) {
        engine->set_vision_gpu(std::make_shared<cuda::GpuEncoder>(cfg, vw, dw));
        std::printf("using GPU DeepEncoder\n");
    } else
#endif
    {
        engine->set_vision(std::make_shared<DeepEncoder>(cfg, std::move(vw), std::move(dw)));
    }

    std::vector<float> visual = engine->image_embeddings(image, crop_mode);
    std::vector<float> ref_visual = load_f32(ref_dir, T.at("visual_scattered"));
    const double visual_rel = report("visual_embeddings", visual, ref_visual);
    std::printf("num_visual_tokens      : ours=%d ref=%d\n", static_cast<int>(visual.size() / cfg.hidden_size),
                manifest.at("num_visual_tokens").get<int>());

    // ---- 4. prefill logits ----
    const int seq = static_cast<int>(layout.input_ids.size());
    Tensor inputs = engine->embed_tokens(layout.input_ids);
    const int hidden = cfg.hidden_size;
    int v = 0;
    for (int i = 0; i < seq; ++i) {
        if (!layout.images_seq_mask[static_cast<std::size_t>(i)]) continue;
        std::memcpy(inputs.data() + static_cast<std::size_t>(i) * hidden,
                    visual.data() + static_cast<std::size_t>(v) * hidden,
                    static_cast<std::size_t>(hidden) * sizeof(float));
        ++v;
    }
    RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                    cfg.sliding_window);
    std::vector<float> our_logits;
#if defined(UOCR_CUDA_ENABLED)
    if (gdec) {
        gdec->prefill_embeds(inputs.data(), seq, our_logits);
    } else
#endif
    {
        engine->decoder().prefill_embeds(cache, inputs, our_logits);
    }
    std::vector<float> ref_logits = load_f32(ref_dir, T.at("prefill_logits"));
    const double logits_rel = report("prefill_logits", our_logits, ref_logits);

    // ---- 5. greedy decode (with optional no-repeat-ngram processor) ----
    const json& steps = manifest.at("decode_steps");
    const int ngram = manifest.value("ngram_size", 0);
    const int ngram_window = manifest.value("ngram_window", 0);
    int pos = seq;
    int token_matches = 0, token_total = 0;
    std::vector<float> cur_logits = our_logits;
    std::vector<int> history = layout.input_ids;
    for (std::size_t s = 0; s < steps.size(); ++s) {
        int want = steps[s].at("token").get<int>();
        if (ngram > 0 && ngram_window > 0)
            Sampler::apply_no_repeat_ngram(cur_logits.data(), cfg.vocab_size, history, ngram,
                                           ngram_window);
        int got = 0;
        float best = -1e30f;
        for (std::size_t i = 0; i < cur_logits.size(); ++i)
            if (cur_logits[i] > best) { best = cur_logits[i]; got = static_cast<int>(i); }
        history.push_back(got);
        ++token_total;
        if (got == want) ++token_matches;
        std::printf("decode step %zu           : token ours=%d ref=%d %s\n", s, got, want,
                    got == want ? "OK" : "DIFF");
        if (s + 1 >= steps.size()) break;
        std::vector<float> next_logits;
#if defined(UOCR_CUDA_ENABLED)
        if (gdec) gdec->decode_token(got, pos, next_logits);
        else
#endif
            engine->decoder().decode(cache, got, pos, next_logits);
        cur_logits = std::move(next_logits);
        ++pos;
    }
    const bool greedy_ok = token_total > 0 && token_matches == token_total;
    const bool visual_ok = visual_rel >= 0.0 && visual_rel <= visual_tol;
    const bool logits_ok = logits_rel >= 0.0 && logits_rel <= logits_tol;
    const bool ok = layout_ok && greedy_ok && visual_ok && logits_ok;
    std::printf("\nsummary: layout=%s visual_rel_l2=%.4g (tol %.3g) prefill_rel_l2=%.4g (tol %.3g) "
                "greedy=%d/%d -> %s\n",
                layout_ok ? "OK" : "BAD", visual_rel, visual_tol, logits_rel, logits_tol,
                token_matches, token_total, ok ? "PASS" : "FAIL");
    return (strict && !ok) ? 1 : 0;
}
