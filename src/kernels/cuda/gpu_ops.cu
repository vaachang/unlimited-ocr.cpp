// Small elementwise / scatter helpers used by the device decoder.

#include <cuda_bf16.h>
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

__global__ void embed_gather_bf16_kernel(const int* __restrict__ ids,
                                         const std::uint16_t* __restrict__ table,
                                         float* __restrict__ out, int rows, int hidden) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * hidden) return;
    const int r = idx / hidden;
    const int c = idx - r * hidden;
    const __nv_bfloat16 v = __ushort_as_bfloat16(
        __ldg(table + static_cast<std::size_t>(ids[r]) * hidden + c));
    out[idx] = __bfloat162float(v);
}

__global__ void embed_gather_f32_kernel(const int* __restrict__ ids,
                                        const float* __restrict__ table,
                                        float* __restrict__ out, int rows, int hidden) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * hidden) return;
    const int r = idx / hidden;
    const int c = idx - r * hidden;
    out[idx] = __ldg(table + static_cast<std::size_t>(ids[r]) * hidden + c);
}

}  // namespace

void embed_gather(const int* ids, const void* table, bool bf16_table, float* out, int rows,
                  int hidden, cudaStream_t stream) {
    if (rows <= 0 || hidden <= 0) return;
    const int threads = 256;
    const int total = rows * hidden;
    const int blocks = (total + threads - 1) / threads;
    if (bf16_table)
        embed_gather_bf16_kernel<<<blocks, threads, 0, stream>>>(
            ids, static_cast<const std::uint16_t*>(table), out, rows, hidden);
    else
        embed_gather_f32_kernel<<<blocks, threads, 0, stream>>>(
            ids, static_cast<const float*>(table), out, rows, hidden);
}

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
