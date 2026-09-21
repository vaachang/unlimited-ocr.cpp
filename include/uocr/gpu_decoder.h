#pragma once

// Device-resident MoE decoder (CUDA builds only).
//
// Weights are uploaded once (bf16) and every decoder op runs on the GPU:
// RMSNorm, QKV/O projections, RoPE, R-SWA attention and the MoE/dense blocks.
// Control flow (MoE top-k selection, expert grouping) stays on the host, which
// keeps the first version simple while removing all per-step weight transfers.
//
// Embedding lookup and the lm_head projection are done on the host (they touch
// large tables but only one row per decode step).

#if defined(UOCR_CUDA_ENABLED)

#include <cuda_runtime.h>

#include <vector>

#include "uocr/common.h"
#include "uocr/config.h"
#include "uocr/gpu_cache.h"
#include "uocr/weights.h"

namespace uocr {
namespace cuda {

class GpuDecoder {
public:
    GpuDecoder(ModelConfig cfg, const DecoderWeights& weights);
    ~GpuDecoder();

    GpuDecoder(const GpuDecoder&) = delete;
    GpuDecoder& operator=(const GpuDecoder&) = delete;

    // Reset the KV cache for a prefill of `prefill_len` tokens.
    void reset(int prefill_len);

    void prefill_tokens(const std::vector<int>& tokens, std::vector<float>& logits);
    void prefill_embeds(const float* host_embeds, int seq, std::vector<float>& logits);
    void decode_token(int token, int pos, std::vector<float>& logits);

    // CUDA-Graph capture scope for the decode step.
    enum class GraphScope {
        kFull,        // attention + dense + MoE all inside one captured graph
        kAttnDense,   // only attention/dense subgraphs captured; MoE issued outside
    };

    // Enable the CUDA-Graph decode path.  Routing moves to the device and the
    // steady decode step is captured once, then replayed.  Requires a CUDA
    // device and is a no-op elsewhere.
    void set_use_graph(bool v) { use_graph_ = v; }
    bool use_graph() const { return use_graph_; }
    bool graph_ready() const { return graph_ready_; }
    void set_graph_scope(GraphScope s) { graph_scope_ = s; }
    GraphScope graph_scope() const { return graph_scope_; }

    // Run device-resident MoE during prefill (device router + fused masked
    // expert kernel) instead of per-expert tensor-core GEMMs.  The masked
    // kernel wins when the MoE batch is small (few tokens per expert), which is
    // the common case for one-request prefill.
    void set_prefill_device_moe(bool v) { prefill_dev_moe_ = v; }
    bool prefill_device_moe() const { return prefill_dev_moe_; }

    // Timing of the last decode step (CUDA events): decoder forward and the
    // final lm_head projection + logits D2H.
    double last_forward_ms() const { return last_forward_ms_; }
    double last_logits_ms() const { return last_logits_ms_; }

    // ---- continuous batching ----
    // Each slot owns an independent R-SWA KV cache.  Prefill slots in one
    // ragged batch with `batch_prefill_embeds`, then advance all active slots
    // together with `batch_decode`.
    void batch_configure(int slots, int capacity);
    // `slots` maps decode order (row) to cache slot; it may be any permutation
    // of the configured slots, so requests are free to occupy any slot.
    void batch_decode(const std::vector<int>& tokens, const std::vector<int>& positions,
                      const std::vector<int>& slots, std::vector<std::vector<float>>& logits);
    // Ragged multi-request prefill: `host_embeds` packs every request's prompt
    // rows back-to-back; request r owns rows [starts[r], starts[r]+lengths[r])
    // and writes into cache slot slots[r].  Returns the last-token logits per
    // request (same order).
    void batch_prefill_embeds(const float* host_embeds, const std::vector<int>& starts,
                              const std::vector<int>& lengths, const std::vector<int>& slots,
                              std::vector<std::vector<float>>& logits);
    int batch_slots() const { return batch_slots_; }
    // Number of captured batched-decode graphs (one per observed batch size).
    int batch_graph_count() const;
    // Number of layers that used the grouped INT4 expert GEMM (test hook).
    long grouped_moe_calls() const { return grouped_moe_calls_; }
    // Tile size (columns / rows per block) for the grouped INT4 expert GEMM.
    void set_grouped_moe_bn(int v) { grouped_bn_ = v; }
    void set_grouped_moe_bm(int v) { grouped_bm_ = v; }

    const ModelConfig& config() const { return cfg_; }

private:
    struct DevLinear {
        std::uint16_t* w = nullptr;  // bf16 [rows, cols]
        float* bias = nullptr;
        int rows = 0;
        int cols = 0;
    };
    // One concatenated INT4 expert table [n_experts, rows, cols] (AWQ layout).
    struct DevExpertTable {
        std::uint8_t* packed = nullptr;
        float* scales = nullptr;
        float* zeros = nullptr;
        int rows = 0, cols = 0, group = 128;
        int pstride = 0;  // bytes per expert
        int sstride = 0;  // scales/zeros per expert
        int ng = 0;       // groups per row
    };

    struct DevLayer {
        float* in_ln = nullptr;
        float* post_ln = nullptr;
        DevLinear q, k, v, o;
        bool is_moe = false;
        DevLinear dense_gate, dense_up, dense_down;
        std::uint16_t* router = nullptr;  // [n_experts, hidden]
        std::vector<DevLinear> experts;   // each .w is one expert matrix
        DevLinear shared_gate, shared_up, shared_down;
        // Device arrays of per-expert weight pointers for the fused kernel.
        const std::uint16_t** gate_ptrs = nullptr;
        const std::uint16_t** up_ptrs = nullptr;
        const std::uint16_t** down_ptrs = nullptr;
        // INT4 expert tables (populated instead of `experts` when the host
        // weights are quantized).
        bool int4_experts = false;
        DevExpertTable gate_i4, up_i4, down_i4;
    };

    void upload_linear(const Linear& src, DevLinear& dst);
    void upload_expert_table(const std::vector<ExpertWeights>& experts, int which,
                             DevExpertTable& dst);
    // Device-resident token embedding table + per-step id staging (task 2.5).
    void upload_embedding(const WeightMatrix& emb);
    void ensure_token_ids(int n);
    void forward(const float* x_dev, int seq, const int* positions_dev, bool prefill, int q_start,
                 float* out_dev, cudaStream_t stream);
    void layer_forward(int li, const float* x, int seq, const int* positions, bool prefill,
                       int q_start, float* out, cudaStream_t stream);
    // Split layer pieces, used by both the full forward and the graph-scope
    // variants.  attention_block writes h1 / normed2; mlp_block turns those
    // into the layer output.
    void attention_block(int li, const float* x, int seq, const int* positions, bool prefill,
                         int q_start, float* h1, float* normed2, cudaStream_t stream);
    // `dev_moe` selects device routing; `grouped` additionally fuses all experts
    // into one grouped GEMM (INT4 only, large-M prefill).
    void mlp_block(int li, const float* h1, const float* normed2, int seq, bool dev_moe,
                   bool grouped, float* out, cudaStream_t stream);
    void ensure_scratch(int seq);
    void ensure_router_scratch(int seq);
    void final_logits(const float* hidden_dev, int seq, std::vector<float>& logits);
    void final_logits_from_normed(std::vector<float>& logits, cudaStream_t stream = 0);

    // CUDA-Graph decode path.
    void invalidate_graph();
    void capture_decode_graph();
    void capture_attn_dense_graphs();
    void run_graph_decode(int token, int pos, std::vector<float>& logits);
    void run_graph_decode_attn_dense(int token, int pos, std::vector<float>& logits);

    // Batched-decode graphs.  The batch size fixes every grid dimension in
    // `forward_batch`, while the row -> slot mapping, positions and token
    // embeddings are read from persistent device buffers at replay time.  That
    // means one graph per observed batch size is enough even as requests come
    // and go (the active slot set may change between replays).
    void invalidate_batch_graphs();
    void capture_batch_graph(int batch);

    // Batched layer pieces (device MoE, per-slot R-SWA attention).
    void attention_block_batch(int li, int batch, const float* x, const int* positions,
                               const int* slots, float* h1, float* normed2, cudaStream_t stream);
    void forward_batch(const float* x, int batch, const int* positions, const int* slots,
                       float* out, cudaStream_t stream);

    // Ragged prefill layer pieces (device MoE; per-row slot/position).
    void attention_block_ragged(int li, int total, const float* x, const int* positions,
                                const int* slots, float* h1, float* normed2, cudaStream_t stream);
    void forward_ragged(const float* x, int total, const int* positions, const int* slots,
                        float* out, cudaStream_t stream);

    ModelConfig cfg_;
    const DecoderWeights* host_weights_ = nullptr;
    GpuRSWACache cache_;
    std::vector<DevLayer> layers_;
    float* final_norm_ = nullptr;
    DevLinear lm_head_;  // device bf16 copy for the final projection

    // scratch (device)
    void* scratch_ = nullptr;
    std::size_t scratch_bytes_ = 0;
    int scratch_seq_ = 0;
    float* d_xin_ = nullptr;
    int* d_pos_ = nullptr;
    int* d_row_idx_ = nullptr;
    float* d_row_w_ = nullptr;
    float* d_ping_ = nullptr;
    float* d_pong_ = nullptr;
    float* d_normed_ = nullptr;
    float* d_hidden_ = nullptr;
    float* d_logits_ = nullptr;  // [vocab_size]

    // device router scratch
    int* d_router_ids_ = nullptr;
    float* d_router_w_ = nullptr;
    int* d_assign_token_ = nullptr;
    float* d_assign_w_ = nullptr;
    int* d_count_ = nullptr;
    float* d_act_ = nullptr;  // [n_experts, cap, moe_intermediate_size]
    int router_seq_ = 0;
    int router_cap_ = 0;

    // CUDA-Graph state
    bool use_graph_ = false;
    bool prefill_dev_moe_ = false;
    // True when the resident expert weights are INT4 (enables the grouped GEMM).
    bool int4_experts_ = false;
    long grouped_moe_calls_ = 0;
    int grouped_bn_ = 32;
    int grouped_bm_ = 128;
    bool graph_ready_ = false;
    GraphScope graph_scope_ = GraphScope::kFull;
    int graph_prefill_len_ = -1;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::vector<cudaGraph_t> attn_graphs_;
    std::vector<cudaGraphExec_t> attn_graph_execs_;
    // Batched decode graphs, indexed by (batch - 1); null entries are not yet
    // captured.  Separate from the single-request graph above.
    std::vector<cudaGraph_t> batch_graphs_;
    std::vector<cudaGraphExec_t> batch_graph_execs_;
    // Device-resident embedding table (`d_embed_` is bf16 when embed_bf16_).
    void* d_embed_ = nullptr;
    bool embed_bf16_ = false;
    int* d_token_ids_ = nullptr;  // [token_ids_cap_] staging for embed_gather
    int token_ids_cap_ = 0;
    int* h_tok_pinned_ = nullptr;
    int* h_pos_pinned_ = nullptr;
    cudaEvent_t ev_a_ = nullptr, ev_b_ = nullptr, ev_c_ = nullptr;
    float last_forward_ms_ = 0.0f, last_logits_ms_ = 0.0f;

    // batched (continuous) decode state
    int batch_slots_ = 0, batch_cap_ = 0, batch_stride_ = 0;
    std::vector<float*> batch_k_, batch_v_;  // [num_layers] -> [slots, cap, kvh*hd]
    int* d_batch_len_ = nullptr;             // [num_layers * slots]
    int* d_batch_ring_ = nullptr;            // [num_layers * slots]
    int* d_batch_prefill_ = nullptr;         // [slots] per-slot prefill length
    int* d_batch_slots_ = nullptr;           // [batch] row -> slot mapping
    int batch_slots_cap_ = 0;                // allocated length of d_batch_slots_
    int* d_batch_row_slot_ = nullptr;        // [ragged slots] row -> slot for prefill
    int ragged_slots_cap_ = 0;               // allocated length of d_batch_row_slot_
    int* d_batch_last_idx_ = nullptr;        // [slots] last-token row per request
    std::vector<int> slot_prefill_len_;      // [slots]
    float* d_logits_batch_ = nullptr;
    int batch_logits_cap_ = 0;
};

}  // namespace cuda
}  // namespace uocr

#endif  // UOCR_CUDA_ENABLED
