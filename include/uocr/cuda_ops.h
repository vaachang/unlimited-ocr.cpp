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

// MoE INT4 GEMM: y[m,n] = x[m,k] * dequant(W_int4[n,k]); W is packed 2-per-byte
// with per-group affine scale/zero.
void moe_gemm_int4(const float* x, const std::uint8_t* packed, const float* scales,
                   const float* zeros, int m, int n, int k, int group_size, float* y,
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
