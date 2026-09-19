#pragma once

// CUDA kernel entry points.  These mirror the CPU reference kernels in
// uocr/ops.h and are only compiled when ENGINE_BACKEND=CUDA.
//
// All wrappers are synchronous unless a stream is passed.

#include <cuda_runtime.h>

#include "uocr/common.h"

namespace uocr {
namespace cuda {

// y[m,n] = x[m,k] * W[n,k]^T + bias[n]; W may be f32 or bf16.
void matmul_t(const float* x, const float* w, const float* bias, float* y, int m, int n, int k,
              cudaStream_t stream = 0);
void matmul_t_bf16(const float* x, const std::uint16_t* w, const float* bias, float* y, int m,
                   int n, int k, cudaStream_t stream = 0);

// y[n] = W[n,k] * x[k] + bias (single activation vector).  Unlike the tiled
// GEMM this reads each weight exactly once, which matters for m == 1 decode.
void matvec_bf16(const float* x, const std::uint16_t* w, const float* bias, float* y, int n,
                 int k, cudaStream_t stream = 0);

// out[r,c] = x[r,c] / rms(x[r,:]) * weight[c]
void rmsnorm(const float* x, const float* weight, float* out, int rows, int cols, float eps,
             cudaStream_t stream = 0);

// Fused per-head RMSNorm + RoPE, applied in place to q/k.  (Unlimited-OCR's
// attention does not use qk-norm, but the fused kernel is provided for
// completeness and for Qwen-style heads.)
void rmsnorm_rope(float* q, float* k, const float* weight, const int* positions, int seq,
                  int n_heads, int n_kv_heads, int head_dim, float theta, float eps,
                  cudaStream_t stream = 0);

// RoPE in place on already-normalized q/k.
void rope(float* q, float* k, const int* positions, int seq, int n_heads, int n_kv_heads,
          int head_dim, float theta, cudaStream_t stream = 0);

// R-SWA decode attention over a [capacity, kv_heads, head_dim] cache.
// Attends the first `kv_len` slots.  q/out: [heads, head_dim].
void rswa_attention_decode(const float* q, const float* kcache, const float* vcache, int kv_len,
                           int heads, int kv_heads, int head_dim, float* out,
                           cudaStream_t stream = 0);

// General R-SWA attention for a whole query block (prefill or decode).
// q/out: [seq, heads, head_dim]; cache: [capacity, kv_heads, head_dim].
// `q_start` is the absolute position of the first query; when `causal` is set,
// query s attends slots [0, min(kv_len, q_start+s+1)).
void rswa_attention(const float* q, const float* kcache, const float* vcache, int kv_len, int seq,
                    int q_start, int heads, int kv_heads, int head_dim, bool causal, float* out,
                    cudaStream_t stream = 0);

// Device-length aware decode attention: the effective length is read from
// `d_len` at kernel execution time and slots >= *d_len are masked out.  This
// keeps a single captured graph valid across the warmup -> ring transition.
// q/out: [seq, heads, head_dim].
void rswa_attention_devlen(const float* q, const float* kcache, const float* vcache,
                           const int* d_len, int seq, int q_start, int heads, int kv_heads,
                           int head_dim, bool causal, float* out, cudaStream_t stream = 0);

// Device-side R-SWA append.  Advances a device-resident `len` / `ring_pos`
// pair so the recorded kernel sequence is replayable inside a CUDA Graph
// (the write slot is resolved on device, not baked at capture time).
void rswa_append_decode(const float* k, const float* v, float* kcache, float* vcache, int* d_len,
                        int* d_ring_pos, int prefill_len, int window, int kv_heads, int head_dim,
                        cudaStream_t stream = 0);

// Batched variant of `rswa_append_decode`: one block per batch row, so a whole
// decode step costs a single launch per layer instead of one per slot.
// `k`/`v` are [batch, kv_heads, head_dim] (row `b`).  The destination cache is
// selected by `d_slots[b]` and starts at
// `kcache_base + d_slots[b] * batch_cap * kv_heads * head_dim`; the per-slot
// ring state is indexed by the same slot id.  `d_len`, `d_ring_pos`,
// `d_prefill_len` and `d_slots` are device arrays.
void rswa_append_decode_batch(const float* k, const float* v, float* kcache_base,
                              float* vcache_base, int* d_len, int* d_ring_pos,
                              const int* d_prefill_len, const int* d_slots, int batch,
                              int batch_cap, int window, int kv_heads, int head_dim,
                              cudaStream_t stream = 0);

// Batched R-SWA decode attention: grid = (batch, heads).  Row `b` of `q`/`out`
// ([batch, heads, head_dim]) attends the cache selected by `d_slots[b]` with
// effective length `d_len[d_slots[b]]`.
void rswa_attention_batch(const float* q, const float* kcache_base, const float* vcache_base,
                          const int* d_len, const int* d_slots, int batch, int batch_cap,
                          int heads, int kv_heads, int head_dim, float* out,
                          cudaStream_t stream = 0);

// MoE router on device: softmax/sigmoid -> greedy top-k -> per-expert grouping.
// `d_ids` / `d_weights` are [seq, top_k]; `d_assign_token` / `d_assign_w` are
// [n_experts, cap] and `d_count` is [n_experts] (must be zeroed by the caller).
void moe_router_topk(const float* router_logits, int seq, int n_experts, int top_k,
                     bool norm_topk_prob, float routed_scaling_factor, bool scoring_sigmoid,
                     int* d_ids, float* d_weights, int* d_assign_token, float* d_assign_w,
                     int* d_count, int cap, cudaStream_t stream = 0);

// Fused "all experts, masked skip" MLP.  Grid = n_experts: each block loops
// over the tokens assigned to its expert (at most `cap`) and scatter-adds the
// weighted expert output into `out` [seq, hidden].  Gate/up/down weight tables
// are device pointer arrays of length n_experts.
// `act` is a caller-owned workspace of [n_experts, cap, inter] floats used to
// stage the gate/up activation between the two expert kernels.
void moe_experts_masked(const float* x, int seq, const std::uint16_t* const* gate_w,
                        const std::uint16_t* const* up_w, const std::uint16_t* const* down_w,
                        const int* assign_token, const float* assign_w, const int* count,
                        int n_experts, int cap, int hidden, int inter, float* act, float* out,
                        cudaStream_t stream = 0);

// INT4 (AWQ) variant of `moe_experts_masked`.  Weight tables are concatenated
// per expert (strides `*_pstride` in bytes and `*_sstride` in floats); weights
// are dequantized on the fly inside the kernel.
void moe_experts_masked_int4(
    const float* x, int seq, const std::uint8_t* gate_packed, const float* gate_scales,
    const float* gate_zeros, int gate_pstride, int gate_sstride, int gate_ng,
    const std::uint8_t* up_packed, const float* up_scales, const float* up_zeros, int up_pstride,
    int up_sstride, int up_ng, const std::uint8_t* down_packed, const float* down_scales,
    const float* down_zeros, int down_pstride, int down_sstride, int down_ng,
    const int* assign_token, const float* assign_w, const int* count, int n_experts, int cap,
    int hidden, int inter, int group, float* act, float* out, cudaStream_t stream = 0);

// MoE INT4 GEMM: y[m,n] = x[m,k] * dequant(W_int4[n,k]); W is packed 2-per-byte
// with per-group affine scale/zero.  Scalar reference (correctness baseline).
void moe_gemm_int4(const float* x, const std::uint8_t* packed, const float* scales,
                   const float* zeros, int m, int n, int k, int group_size, float* y,
                   cudaStream_t stream = 0);

// Tensor-core W4A16 variant: INT4 weights are dequantized to bf16 in registers
// and multiplied with bf16 activations via `mma.sync.aligned.m16n8k16` with an
// f32 accumulator.  Same signature and numerics (up to bf16 rounding) as
// `moe_gemm_int4`.
void moe_gemm_int4_tc(const float* x, const std::uint8_t* packed, const float* scales,
                      const float* zeros, int m, int n, int k, int group_size, float* y,
                      cudaStream_t stream = 0);

// out[i] = silu(gate[i]) * up[i]
void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t stream = 0);

// dst[i] += scale * src[i]
void add_scaled(float* dst, const float* src, float scale, int n, cudaStream_t stream = 0);

// out[row_idx[r] * cols + c] += weight[r] * vals[r * cols + c]
void scatter_add_scaled(float* out, const float* vals, const int* row_idx, const float* weights,
                        int rows, int cols, cudaStream_t stream = 0);

// dst[r, :] = src[row_idx[r], :]
void gather_rows(float* dst, const float* src, const int* row_idx, int rows, int cols,
                 cudaStream_t stream = 0);

// Device information.
struct DeviceInfo {
    char name[256];
    int compute_major;
    int compute_minor;
    std::size_t total_memory;
    int multi_processor_count;
};
DeviceInfo device_info(int ordinal = 0);
bool available();

}  // namespace cuda
}  // namespace uocr
