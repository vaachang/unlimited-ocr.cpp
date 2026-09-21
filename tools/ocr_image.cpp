// ocr_image -- run Unlimited-OCR on an image file and print the recognised text.
//
//   ocr_image --model models --image page.png [--prompt "<image>\nFree OCR."]
//
// The backend defaults to CUDA (with the GPU DeepEncoder) when the binary was
// built with ENGINE_BACKEND=CUDA; pass --cpu to force the reference CPU path.
// Expert weights are BF16 by default; --int4 enables the (faster, smaller, but
// much less accurate) group-128 RTN experts.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "uocr/config.h"
#if defined(UOCR_CUDA_ENABLED)
#include "uocr/cuda_ops.h"
#include "uocr/gpu_encoder.h"
#endif
#include "uocr/deep_encoder.h"
#include "uocr/engine.h"
#include "uocr/image.h"
#include "uocr/log.h"
#include "uocr/prompt.h"
#include "uocr/safetensors.h"
#include "uocr/tokenizer.h"
#include "uocr/weights.h"

using namespace uocr;

namespace {

std::vector<std::string> split_on(const std::string& s, const std::string& sep) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (true) {
        const std::size_t p = s.find(sep, pos);
        if (p == std::string::npos) {
            out.push_back(s.substr(pos));
            break;
        }
        out.push_back(s.substr(pos, p - pos));
        pos = p + sep.size();
    }
    return out;
}

void usage(const char* argv0) {
    std::printf(
        "usage: %s --image FILE [options]\n"
        "  --model DIR            model directory (default: models)\n"
        "  --image FILE           input image: PNG (if libpng) or binary P6 PPM\n"
        "  --prompt STR           OCR prompt containing one <image> (default: \"<image>\\nFree OCR.\")\n"
        "  --max-new-tokens N     generation cap (default: 2048)\n"
        "  --no-crop-mode         disable Gundam dynamic crops (single 640px view)\n"
        "  --cpu                  force the CPU backend + CPU DeepEncoder\n"
        "  --cpu-vision           CUDA decoder but CPU DeepEncoder\n"
        "  --int4                 use INT4 expert weights (default: BF16)\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models";
    std::string image_path;
    std::string prompt = "<image>\nFree OCR.";
    int max_new = 2048;
    bool crop_mode = true;
    bool force_cpu = false;
    bool cpu_vision = false;
    bool int4 = false;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--image") && i + 1 < argc) image_path = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--max-new-tokens") && i + 1 < argc)
            max_new = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-crop-mode")) crop_mode = false;
        else if (!std::strcmp(argv[i], "--cpu")) force_cpu = true;
        else if (!std::strcmp(argv[i], "--cpu-vision")) cpu_vision = true;
        else if (!std::strcmp(argv[i], "--int4")) int4 = true;
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }
    if (image_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    ImageRGB image = load_image(image_path);
    if (image.empty()) {
        std::fprintf(stderr,
                     "failed to load image '%s' (supported: PNG if built with libpng, or P6 PPM)\n",
                     image_path.c_str());
        return 1;
    }
    std::printf("image: %dx%d (%s), prompt: %.40s%s\n", image.width, image.height,
                image_path.c_str(), prompt.c_str(), prompt.size() > 40 ? "..." : "");

    ModelConfig cfg = ModelConfig::from_json_file(model_dir + "/config.json");
    Tokenizer tok = Tokenizer::from_file(model_dir + "/tokenizer.json");

    const std::string weights_path = model_dir + "/model-00001-of-000001.safetensors";
    SafetensorsFile st(weights_path);
    VisionWeights vw = VisionWeights::load(st, cfg);
    DecoderWeights dw = DecoderWeights::load(weights_path, cfg, false, 128);

    EngineConfig ecfg;
    ecfg.model_dir = model_dir;
    ecfg.use_int4_experts = int4;
    ecfg.int4_group_size = 128;
    ecfg.max_seq_len = 4096;
    ecfg.max_new_tokens = max_new;
    // The host block manager only needs a small bookkeeping arena: on the CPU
    // backend a zero budget would size the pool from max_seq_len * max_batch.
    ecfg.memory_pool_bytes = 1u << 20;

    Backend backend = Backend::CPU;
#if defined(UOCR_CUDA_ENABLED)
    if (!force_cpu && cuda::available()) backend = Backend::CUDA;
#else
    if (!force_cpu) std::fprintf(stderr, "note: built without CUDA, using the CPU backend\n");
#endif

    auto engine = Engine::load(ecfg, backend);
    std::printf("backend: %s, experts: %s, vision: %s\n",
                backend == Backend::CUDA ? "CUDA" : "CPU", int4 ? "INT4" : "BF16",
                (backend == Backend::CUDA && !cpu_vision) ? "GPU" : "CPU");
    if (int4)
        std::fprintf(stderr,
                     "warning: INT4 experts use group-128 round-to-nearest quantization "
                     "(~10%% weight error); prefer BF16 unless memory-bound\n");

#if defined(UOCR_CUDA_ENABLED)
    if (backend == Backend::CUDA && !cpu_vision) {
        engine->set_vision_gpu(std::make_shared<cuda::GpuEncoder>(cfg, vw, dw));
    } else
#endif
    {
        engine->set_vision(std::make_shared<DeepEncoder>(cfg, std::move(vw), std::move(dw)));
    }

    // Build the token layout with the same crop grid the vision path will use,
    // so the number of <image> tokens matches the visual embedding count.
    std::vector<ImageSpatialCrop> crops = engine->image_crops(image, crop_mode);
    PromptLayout layout =
        build_ocr_prompt(tok, split_on(prompt, "<image>"), crops, cfg, crop_mode);

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> visual = engine->image_embeddings(image, crop_mode);
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("visual tokens: %zu (%.0f ms)\n", visual.size() / cfg.hidden_size,
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    GenerationResult res = engine->generate_from_image(layout.input_ids, layout.images_seq_mask,
                                                       visual, cfg.hidden_size, max_new);
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("prefill %d tok (%.0f ms TTFT), generated %d tok (%.1f ms/tok)\n",
                res.prefill_tokens, res.ttft_ms, res.decode_tokens, res.tpot_ms);

    std::printf("\n===== OCR =====\n%s\n===============\n", tok.decode(res.tokens, false).c_str());
    (void)t2;
    return 0;
}
