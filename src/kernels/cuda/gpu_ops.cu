// Small elementwise / scatter helpers used by the device decoder.

#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

__global__ void silu_mul_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float s = g / (1.0f + __expf(-g));
    out[i] = s * up[i];
}

__global__ void add_scaled_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                  float scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += scale * src[i];
}

__global__ void scatter_add_kernel(float* __restrict__ out, const float* __restrict__ vals,
                                   const int* __restrict__ row_idx,
                                   const float* __restrict__ weights, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const int r = idx / cols;
    const int c = idx - r * cols;
    out[static_cast<std::size_t>(row_idx[r]) * cols + c] += weights[r] * vals[idx];
}

__global__ void gather_rows_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                   const int* __restrict__ row_idx, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    const int r = idx / cols;
    const int c = idx - r * cols;
    dst[idx] = src[static_cast<std::size_t>(row_idx[r]) * cols + c];
}

}  // namespace

void gather_rows(float* dst, const float* src, const int* row_idx, int rows, int cols,
                 cudaStream_t stream) {
    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;
    gather_rows_kernel<<<blocks, threads, 0, stream>>>(dst, src, row_idx, rows, cols);
}

void silu_mul(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

void add_scaled(float* dst, const float* src, float scale, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    add_scaled_kernel<<<blocks, threads, 0, stream>>>(dst, src, scale, n);
}

void scatter_add_scaled(float* out, const float* vals, const int* row_idx, const float* weights,
                        int rows, int cols, cudaStream_t stream) {
    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;
    scatter_add_kernel<<<blocks, threads, 0, stream>>>(out, vals, row_idx, weights, rows, cols);
}

}  // namespace cuda
}  // namespace uocr
