#pragma once

// Device-resident R-SWA KV cache (CUDA builds only).
//
// Mirrors the host RSWACache state machine (fixed reference region + ring
// window) but keeps K/V in GPU memory.  Attention is dispatched to
// `uocr::cuda::rswa_attention`.  The cache base addresses and capacity are
// stable across steps, which is what the future CUDA Graph capture needs.

#if defined(UOCR_CUDA_ENABLED)

#include <cuda_runtime.h>

#include <vector>

#include "uocr/common.h"

namespace uocr {
namespace cuda {

class GpuRSWACache {
public:
    GpuRSWACache() = default;
    GpuRSWACache(int num_layers, int kv_heads, int head_dim, int window);
    ~GpuRSWACache();

    GpuRSWACache(const GpuRSWACache&) = delete;
    GpuRSWACache& operator=(const GpuRSWACache&) = delete;

    void configure(int num_layers, int kv_heads, int head_dim, int window);
    void reset(int prefill_len);

    // Copy prefill K/V (device pointers, [seq, kv_heads, head_dim]) into slots
    // [0, seq) and set len = seq.
    void write_prefill(int layer, const float* d_k, const float* d_v, int seq,
                       cudaStream_t stream = 0);

    // Append/overwrite one decode token (device K/V of [kv_heads, head_dim]).
    void append_decode(int layer, const float* d_k, const float* d_v, cudaStream_t stream = 0);

    // Device-resident variant used by the CUDA-Graph path.  `len` and the ring
    // cursor live on the device, so a captured append kernel resolves its write
    // slot at replay time.  `reset_device_state` initialises them for a prefill
    // of `prefill_len`; `len_dev` is fed to `rswa_attention_devlen`.
    void reset_device_state(int prefill_len, cudaStream_t stream = 0);
    void append_decode_device(int layer, const float* d_k, const float* d_v,
                              cudaStream_t stream = 0);
    const int* len_dev(int layer) const { return d_len_ + layer; }

    // q/out: device [seq, heads, head_dim].
    void attention(int layer, const float* d_q, int seq, int q_start, float* d_out, int heads,
                   bool causal, cudaStream_t stream = 0) const;

    int len(int layer) const { return len_[layer]; }
    int prefill_len() const { return prefill_len_; }
    int window() const { return window_; }
    int capacity() const { return capacity_; }
    int num_layers() const { return num_layers_; }

    const float* keys(int layer) const { return k_[layer]; }
    const float* values(int layer) const { return v_[layer]; }

private:
    int num_layers_ = 0, kv_heads_ = 0, head_dim_ = 0, window_ = 0;
    int prefill_len_ = 0, capacity_ = 0;
    std::vector<float*> k_, v_;
    std::vector<int> len_, ring_pos_;
    std::vector<char> ring_started_;
    int* d_len_ = nullptr;       // [num_layers]
    int* d_ring_pos_ = nullptr;  // [num_layers]
    bool configured_ = false;
};

}  // namespace cuda
}  // namespace uocr

#endif  // UOCR_CUDA_ENABLED
