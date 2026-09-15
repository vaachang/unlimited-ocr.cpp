#pragma once

// CPU reference kernels.  Every function in this header has a CUDA counterpart
// in src/kernels/cuda with the same semantics, so the engine can switch
// backends without touching model code.
//
// Conventions:
//   * activations are row-major float32
//   * weight matrices are stored [out_features, in_features] (i.e. the layout
//     used by nn.Linear), so a "linear" is y = x * W^T
//   * shapes are passed explicitly as ints for clarity at the call site

#include "uocr/common.h"

namespace uocr {
namespace ops {

// y[m,n] = x[m,k] * W[n,k]^T + bias[n]
void matmul_t(const float* x, const float* w, const float* bias, float* y,
              int m, int n, int k);

// Same as above but W is bf16 (as stored in the checkpoint).
void matmul_t_bf16(const float* x, const std::uint16_t* w, const float* bias, float* y,
                   int m, int n, int k);

// y[m,n] = x[m,k] * W[n,k]^T without transposing W (W stored [k,n], row-major).
void matmul_nn(const float* x, const float* w, const float* bias, float* y,
               int m, int n, int k);

void add_inplace(float* a, const float* b, i64 n);

// out[r,c] = x[r,c] / rms(x[r,:]) * weight[c]
void rmsnorm(const float* x, const float* weight, float* out, int rows, int cols, float eps);

// out[r,c] = ((x-mean)/sqrt(var+eps)) * weight[c] + bias[c]
void layernorm(const float* x, const float* weight, const float* bias, float* out,
               int rows, int cols, float eps);

// In-place softmax over each row.
void softmax(float* x, int rows, int cols);

// out = silu(gate) * up  elementwise for `n` elements
void silu_mul(const float* gate, const float* up, float* out, i64 n);
void gelu(float* x, i64 n);
void quick_gelu(float* x, i64 n);

// RoPE (rotary position embedding) applied in place to q and k.
// q: [seq, n_heads, head_dim], k: [seq, n_kv_heads, head_dim]
void rope(float* q, float* k, const int* positions, int seq, int n_heads, int n_kv_heads,
          int head_dim, float theta);

// build cos/sin tables cached by (max_seq, head_dim, theta)
void rope_tables(int head_dim, int max_seq, float theta, float* cos_out, float* sin_out);

}  // namespace ops
}  // namespace uocr
