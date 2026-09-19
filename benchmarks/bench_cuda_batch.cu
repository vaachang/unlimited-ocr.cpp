// Continuous-batching throughput benchmark (device decoder + scheduler).
//
// Runs `Engine::generate_batch` on B synthetic prompts for B = 1..max_batch and
// reports prefill latency, decode throughput (tok/s) and peak device memory.
//
// Usage: bench_cuda_batch [--real] [--int4] [--model DIR]
//                         [--prompt N] [--steps N] [--max-batch N]

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "uocr/config.h"
#include "uocr/cuda_ops.h"
#include "uocr/engine.h"

using uocr::Backend;
using uocr::DecoderWeights;
using uocr::Engine;
using uocr::EngineConfig;
using uocr::ModelConfig;

namespace {

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

std::size_t used_vram() {
    std::size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    return total_b - free_b;
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
    bool real = false, int4 = false, no_graph = false;
    std::string model_dir = "models";
    int prompt_len = 64, steps = 32, max_batch = 16;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) real = true;
        else if (!std::strcmp(argv[i], "--int4")) int4 = true;
        else if (!std::strcmp(argv[i], "--no-graph")) no_graph = true;
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_len = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-batch") && i + 1 < argc) max_batch = std::atoi(argv[++i]);
    }
    if (!uocr::cuda::available()) {
        std::printf("no CUDA device\n");
        return 0;
    }
    auto info = uocr::cuda::device_info();
    std::printf("device: %s (sm_%d%d, %d SMs, %.1f GB)\n", info.name, info.compute_major,
                info.compute_minor, info.multi_processor_count,
                static_cast<double>(info.total_memory) / (1024.0 * 1024.0 * 1024.0));

    ModelConfig cfg;
    DecoderWeights weights;
    if (real) {
        cfg = ModelConfig::from_json_file(model_dir + "/config.json");
        weights = DecoderWeights::load(model_dir + "/model-00001-of-000001.safetensors", cfg, int4,
                                       128);
    } else {
        cfg = synthetic_config();
        weights = DecoderWeights::random(cfg, 1234);
    }
    EngineConfig ecfg;
    ecfg.max_batch_size = max_batch;
    ecfg.min_batch_size = 1;
    ecfg.use_int4_experts = int4;
    ecfg.no_repeat_ngram_size = 0;
    ecfg.max_new_tokens = steps;
    ecfg.use_cuda_graph = !no_graph;

    const std::size_t before = used_vram();
    Engine engine(cfg, ecfg, std::move(weights), Backend::CUDA);

    std::printf("prompt=%d steps=%d max_batch=%d experts=%s graph=%s\n", prompt_len, steps,
                max_batch, int4 ? "INT4" : "BF16", no_graph ? "off" : "on");
    std::printf("%6s %12s %12s %14s %12s\n", "batch", "prefill_ms", "decode_ms", "tok/s", "peakMB");

    std::vector<std::vector<int>> all_prompts(max_batch);
    for (int b = 0; b < max_batch; ++b) {
        all_prompts[b].resize(prompt_len);
        for (int t = 0; t < prompt_len; ++t)
            all_prompts[b][t] = (b * 977 + t * 131 + 1) % cfg.vocab_size;
    }

    for (int B : {1, 2, 4, 8, 16}) {
        if (B > max_batch) break;
        std::vector<std::vector<int>> prompts(all_prompts.begin(), all_prompts.begin() + B);
        // Warmup: captures the batched CUDA graph (first decode) so the timed
        // run measures steady state, not the one-off capture cost.
        (void)engine.generate_batch(prompts, steps);
        double t_prefill = 0.0;
        const double t0 = now_ms();
        auto res = engine.generate_batch(prompts, steps);
        const double t1 = now_ms();
        const std::size_t v1 = used_vram();
        long total_tokens = 0;
        for (const auto& r : res) {
            total_tokens += static_cast<long>(r.tokens.size());
            t_prefill += r.ttft_ms;
        }
        const double tps = total_tokens / ((t1 - t0) / 1000.0);
        std::printf("%6d %12.1f %12.1f %14.1f %12.1f\n", B, t_prefill / B, t1 - t0, tps,
                    static_cast<double>(v1 - before) / (1024.0 * 1024.0));
    }
    return 0;
}
