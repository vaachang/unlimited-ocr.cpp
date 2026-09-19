// CUDA decode benchmark and CUDA-Graph ablation.
//
// Measures prefill latency, steady decode TPOT and device memory for the
// device decoder under three configurations:
//   plain      - device decoder, kernel launches every step (no graph)
//   full       - one captured graph covering the whole decode step
//   attn_dense - attention subgraphs captured, MoE issued outside the graph
//
// Usage:
//   bench_cuda_decode [--real] [--int4] [--model DIR]
//                     [--prefill N] [--steps N] [--group G]

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
#include "uocr/gpu_decoder.h"
#include "uocr/log.h"
#include "uocr/weights.h"

namespace {

using uocr::ModelConfig;
using uocr::DecoderWeights;
namespace cuda = uocr::cuda;

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

struct RunResult {
    double ttft_ms = 0.0;
    double tpot_first = 0.0;
    double tpot_steady = 0.0;
    double vram_mb = 0.0;
};

RunResult run_mode(const char* name, const ModelConfig& cfg, const DecoderWeights& w, int prefill,
                   int steps, cuda::GpuDecoder::GraphScope scope, bool use_graph) {
    const std::size_t before = used_vram();
    auto dec = std::make_unique<cuda::GpuDecoder>(cfg, w);
    dec->set_use_graph(use_graph);
    dec->set_graph_scope(scope);
    const std::size_t after = used_vram();

    std::vector<int> prompt(prefill);
    for (int i = 0; i < prefill; ++i) prompt[i] = (i * 131 + 1) % cfg.vocab_size;

    std::vector<float> logits;
    const double t0 = now_ms();
    dec->prefill_tokens(prompt, logits);
    const double t1 = now_ms();

    RunResult r;
    r.ttft_ms = t1 - t0;
    r.vram_mb = static_cast<double>(after - before) / (1024.0 * 1024.0);

    int pos = prefill;
    double sum = 0.0;
    int steady_steps = 0;
    for (int step = 0; step < steps; ++step) {
        const int tok = (step * 7 + 3) % cfg.vocab_size;
        const double s0 = now_ms();
        dec->decode_token(tok, pos, logits);
        const double s1 = now_ms();
        if (step == 0) r.tpot_first = s1 - s0;
        if (step >= 2) {  // skip first capture/launch warmup
            sum += s1 - s0;
            ++steady_steps;
        }
        ++pos;
    }
    r.tpot_steady = steady_steps > 0 ? sum / steady_steps : 0.0;
    std::printf("%-10s prefill=%.2f ms  TPOT(first)=%.3f ms  TPOT(steady)=%.3f ms  "
                "VRAM=%.1f MB  [fwd=%.3f logits=%.3f]\n",
                name, r.ttft_ms, r.tpot_first, r.tpot_steady, r.vram_mb, dec->last_forward_ms(),
                dec->last_logits_ms());
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    bool real = false, int4 = false;
    std::string model_dir = "models";
    int prefill = 128, steps = 64, group = 128, layers = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--real")) real = true;
        else if (!std::strcmp(argv[i], "--int4")) int4 = true;
        else if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--prefill") && i + 1 < argc) prefill = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--group") && i + 1 < argc) group = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--layers") && i + 1 < argc) layers = std::atoi(argv[++i]);
    }
    if (!cuda::available()) {
        std::printf("no CUDA device available\n");
        return 0;
    }
    auto info = cuda::device_info();
    std::printf("device: %s (sm_%d%d, %d SMs, %.1f GB)\n", info.name, info.compute_major,
                info.compute_minor, info.multi_processor_count,
                static_cast<double>(info.total_memory) / (1024.0 * 1024.0 * 1024.0));

    ModelConfig cfg;
    DecoderWeights weights;
    if (real) {
        cfg = ModelConfig::from_json_file(model_dir + "/config.json");
        weights = DecoderWeights::load(model_dir + "/model-00001-of-000001.safetensors", cfg, int4,
                                       group);
        std::printf("weights: real %s (experts %s, group=%d)\n", model_dir.c_str(),
                    int4 ? "INT4" : "BF16", group);
    } else {
        cfg = synthetic_config();
        weights = DecoderWeights::random(cfg, 1234);
        std::printf("weights: synthetic\n");
    }
    if (layers > 0 && layers < cfg.num_hidden_layers) cfg.num_hidden_layers = layers;
    std::printf("prefill=%d steps=%d layers=%d\n", prefill, steps, cfg.num_hidden_layers);

    // Sequential: only one decoder is resident at a time.
    run_mode("plain", cfg, weights, prefill, steps, cuda::GpuDecoder::GraphScope::kFull, false);
    run_mode("full", cfg, weights, prefill, steps, cuda::GpuDecoder::GraphScope::kFull, true);
    run_mode("attn_dense", cfg, weights, prefill, steps, cuda::GpuDecoder::GraphScope::kAttnDense,
             true);
    return 0;
}
