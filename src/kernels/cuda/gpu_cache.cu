#include "uocr/gpu_cache.h"

#include <cstdio>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {

#ifndef UOCR_CUDA_CHECK
#define UOCR_CUDA_CHECK(expr)                                                        \
    do {                                                                             \
        cudaError_t _e = (expr);                                                     \
        if (_e != cudaSuccess)                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), \
                         __FILE__, __LINE__);                                        \
    } while (0)
#endif

GpuRSWACache::GpuRSWACache(int num_layers, int kv_heads, int head_dim, int window) {
    configure(num_layers, kv_heads, head_dim, window);
}

GpuRSWACache::~GpuRSWACache() {
    for (float* p : k_)
        if (p) cudaFree(p);
    for (float* p : v_)
        if (p) cudaFree(p);
    if (d_len_) cudaFree(d_len_);
    if (d_ring_pos_) cudaFree(d_ring_pos_);
}

void GpuRSWACache::configure(int num_layers, int kv_heads, int head_dim, int window) {
    for (float* p : k_)
        if (p) cudaFree(p);
    for (float* p : v_)
        if (p) cudaFree(p);
    if (d_len_) cudaFree(d_len_);
    if (d_ring_pos_) cudaFree(d_ring_pos_);
    d_len_ = nullptr;
    d_ring_pos_ = nullptr;
    num_layers_ = num_layers;
    kv_heads_ = kv_heads;
    head_dim_ = head_dim;
    window_ = window;
    k_.assign(num_layers, nullptr);
    v_.assign(num_layers, nullptr);
    len_.assign(num_layers, 0);
    ring_pos_.assign(num_layers, 0);
    ring_started_.assign(num_layers, 0);
    prefill_len_ = capacity_ = 0;
    configured_ = true;
}

void GpuRSWACache::reset(int prefill_len) {
    UOCR_CHECK(configured_, "GpuRSWACache not configured");
    const int new_capacity = prefill_len + window_;
    // Allocate when the shape changes so that a larger prefill on a reused
    // cache cannot overflow the previous allocation.
    if (new_capacity > capacity_ || k_.empty() || !k_[0]) {
        const std::size_t bytes =
            static_cast<std::size_t>(new_capacity) * kv_heads_ * head_dim_ * sizeof(float);
        for (int l = 0; l < num_layers_; ++l) {
            if (k_[l]) cudaFree(k_[l]);
            if (v_[l]) cudaFree(v_[l]);
            UOCR_CUDA_CHECK(cudaMalloc(&k_[l], bytes));
            UOCR_CUDA_CHECK(cudaMalloc(&v_[l], bytes));
        }
        capacity_ = new_capacity;
    }
    prefill_len_ = prefill_len;
    const std::size_t bytes =
        static_cast<std::size_t>(capacity_) * kv_heads_ * head_dim_ * sizeof(float);
    for (int l = 0; l < num_layers_; ++l) {
        UOCR_CUDA_CHECK(cudaMemset(k_[l], 0, bytes));
        UOCR_CUDA_CHECK(cudaMemset(v_[l], 0, bytes));
        len_[l] = prefill_len;
        ring_pos_[l] = 0;
        ring_started_[l] = 0;
    }
    if (!d_len_) {
        UOCR_CUDA_CHECK(cudaMalloc(&d_len_, num_layers_ * sizeof(int)));
        UOCR_CUDA_CHECK(cudaMalloc(&d_ring_pos_, num_layers_ * sizeof(int)));
    }
    reset_device_state(prefill_len);
}

void GpuRSWACache::reset_device_state(int prefill_len, cudaStream_t stream) {
    std::vector<int> zeros(num_layers_, 0);
    std::vector<int> lens(num_layers_, prefill_len);
    UOCR_CUDA_CHECK(cudaMemcpyAsync(d_len_, lens.data(), lens.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
    UOCR_CUDA_CHECK(cudaMemcpyAsync(d_ring_pos_, zeros.data(), zeros.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));
}

void GpuRSWACache::append_decode_device(int layer, const float* d_k, const float* d_v,
                                        cudaStream_t stream) {
    cuda::rswa_append_decode(d_k, d_v, k_[layer], v_[layer], d_len_ + layer,
                             d_ring_pos_ + layer, prefill_len_, window_, kv_heads_, head_dim_,
                             stream);
}

void GpuRSWACache::write_prefill(int layer, const float* d_k, const float* d_v, int seq,
                                 cudaStream_t stream) {
    const std::size_t bytes = static_cast<std::size_t>(seq) * kv_heads_ * head_dim_ * sizeof(float);
    UOCR_CUDA_CHECK(cudaMemcpyAsync(k_[layer], d_k, bytes, cudaMemcpyDeviceToDevice, stream));
    UOCR_CUDA_CHECK(cudaMemcpyAsync(v_[layer], d_v, bytes, cudaMemcpyDeviceToDevice, stream));
    len_[layer] = seq;
    prefill_len_ = seq;
}

void GpuRSWACache::append_decode(int layer, const float* d_k, const float* d_v,
                                 cudaStream_t stream) {
    const std::size_t stride = static_cast<std::size_t>(kv_heads_) * head_dim_;
    const int len = len_[layer];
    if (len < prefill_len_ + window_) {
        UOCR_CUDA_CHECK(cudaMemcpyAsync(k_[layer] + static_cast<std::size_t>(len) * stride, d_k,
                                        stride * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        UOCR_CUDA_CHECK(cudaMemcpyAsync(v_[layer] + static_cast<std::size_t>(len) * stride, d_v,
                                        stride * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        len_[layer] = len + 1;
        if (len + 1 >= prefill_len_ + window_) {
            ring_started_[layer] = 1;
            ring_pos_[layer] = 0;
        }
        return;
    }
    if (!ring_started_[layer]) {
        ring_started_[layer] = 1;
        ring_pos_[layer] = 0;
    }
    const int slot = prefill_len_ + ring_pos_[layer];
    UOCR_CUDA_CHECK(cudaMemcpyAsync(k_[layer] + static_cast<std::size_t>(slot) * stride, d_k,
                                    stride * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    UOCR_CUDA_CHECK(cudaMemcpyAsync(v_[layer] + static_cast<std::size_t>(slot) * stride, d_v,
                                    stride * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    ring_pos_[layer] = (ring_pos_[layer] + 1) % window_;
}

void GpuRSWACache::attention(int layer, const float* d_q, int seq, int q_start, float* d_out,
                             int heads, bool causal, cudaStream_t stream) const {
    cuda::rswa_attention(d_q, k_[layer], v_[layer], len_[layer], seq, q_start, heads, kv_heads_,
                         head_dim_, causal, d_out, stream);
}

}  // namespace cuda
}  // namespace uocr
