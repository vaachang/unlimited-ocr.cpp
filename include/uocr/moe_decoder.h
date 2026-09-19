#pragma once

// MoE decoder with R-SWA attention (DeepSeek-V2 style block, plain MHA because
// `use_mla == false` for Unlimited-OCR).
//
// The decoder is backend agnostic: it only depends on the WeightMatrix /
// RSWACache abstractions, so the same code path drives the CPU reference and
// (via the ops dispatch) CUDA kernels.

#include <memory>
#include <vector>

#include "uocr/config.h"
#include "uocr/kv_cache.h"
#include "uocr/tensor.h"
#include "uocr/weights.h"

namespace uocr {

// Result of a single decode step.
struct DecodeOutput {
    std::vector<float> logits;  // [vocab]
};

class MoEDecoder {
public:
    MoEDecoder(ModelConfig cfg, DecoderWeights weights);

    const ModelConfig& config() const { return cfg_; }
    const DecoderWeights& weights() const { return w_; }

    // Embed tokens into hidden states [seq, hidden].  Images are already
    // inserted by the caller (vision embeddings replacing image tokens).
    Tensor embed(const std::vector<int>& tokens) const;

    // Run the decoder over already-embedded inputs.
    //   `cache` is configured for the request (prefill_len = start_pos + seq
    //   for the first prefill).
    //   `positions` has `seq` entries.
    //   `prefill` selects causal attention + writes the reference region.
    //   `q_start` is the global position of the first query token.
    //   `final_norm` applies the model's final RMSNorm (prefill may skip it).
    //   `logits` outputs the lm_head projection of the last token.
    void forward(const Tensor& inputs, const std::vector<int>& positions, RSWACache& cache,
                 bool prefill, int q_start, bool final_norm, std::vector<float>& logits,
                 std::vector<Tensor>* layer_outputs = nullptr);

    // Convenience wrappers.
    void prefill(RSWACache& cache, const std::vector<int>& tokens, int start_pos,
                 std::vector<float>& logits);
    // Prefill from already-built embeddings (text + scattered visual tokens).
    void prefill_embeds(RSWACache& cache, const Tensor& inputs, std::vector<float>& logits,
                        std::vector<Tensor>* layer_outputs = nullptr);
    void decode(RSWACache& cache, int token, int pos, std::vector<float>& logits);

    // Router results for the last token (used by tests / introspection).
    struct RouterTrace {
        std::vector<int> experts;
        std::vector<float> weights;
    };
    const std::vector<std::vector<RouterTrace>>& router_trace() const { return router_trace_; }
    void clear_router_trace() { router_trace_.clear(); }
    void set_trace_router(bool v) { trace_router_ = v; }

    // Round the MLP/MoE block input (post_attention_layernorm output) to bf16,
    // emulating the reference gate's `hidden_states.type(torch.float32)` on a
    // bf16 autocast activation.  Off by default; experiments show it does not
    // measurably reduce router flips (see ALIGNMENT.md 5.2).
    void set_bf16_rounding(bool v) { bf16_rounding_ = v; }

private:
    void layer_forward(int layer_idx, const Tensor& x, const std::vector<int>& positions,
                       RSWACache& cache, bool prefill, int q_start, Tensor& out);
    void mlp_forward(const MLPWeights& mlp, const Tensor& x, Tensor& out);
    void moe_forward(const LayerWeights& L, const Tensor& x, Tensor& out);
    void routed_expert(const LayerWeights& L, const float* x, float* out, int expert_idx);

    ModelConfig cfg_;
    DecoderWeights w_;

    // scratch buffers (reused between calls to avoid allocations)
    mutable std::vector<float> scratch_;
    bool trace_router_ = false;
    bool bf16_rounding_ = false;
    std::vector<std::vector<RouterTrace>> router_trace_;
};

// Standalone fused ops reused by tests.  These implement the reference
// semantics used by MoEDecoder.
namespace moe_detail {
void rmsnorm(const float* x, const float* w, float* out, int rows, int cols, float eps);
void moe_gate(const float* x, const WeightMatrix& router, const ModelConfig& cfg, int rows,
              std::vector<int>& expert_ids, std::vector<float>& expert_weights);
}  // namespace moe_detail

}  // namespace uocr
