// Backend utilities: dense matmul kernels and device introspection.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstring>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

__global__ void matmul_t_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                const float* __restrict__ bias, float* __restrict__ y, int m,
                                int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    const float* xrow = x + static_cast<std::size_t>(row) * k;
    const float* wrow = w + static_cast<std::size_t>(col) * k;
    float acc = bias ? bias[col] : 0.0f;
    for (int t = 0; t < k; ++t) acc += xrow[t] * wrow[t];
    y[static_cast<std::size_t>(row) * n + col] = acc;
}

__global__ void matmul_t_bf16_kernel(const float* __restrict__ x,
                                     const std::uint16_t* __restrict__ w,
                                     const float* __restrict__ bias, float* __restrict__ y, int m,
                                     int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    const float* xrow = x + static_cast<std::size_t>(row) * k;
    const std::uint16_t* wrow = w + static_cast<std::size_t>(col) * k;
    float acc = bias ? bias[col] : 0.0f;
    for (int t = 0; t < k; ++t) acc += xrow[t] * __bfloat162float(__ushort_as_bfloat16(wrow[t]));
    y[static_cast<std::size_t>(row) * n + col] = acc;
}

// One warp per output row.  Lanes read consecutive bf16 pairs so every load
// transaction is fully coalesced (the naive one-thread-per-row layout strides
// by `k` between lanes and wastes ~16x bandwidth).
__device__ __forceinline__ float warp_dot_bf16(const std::uint16_t* __restrict__ wrow,
                                               const float* __restrict__ x, int k) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    int c = lane * 2;
    for (; c + 1 < k; c += 64) {
        const std::uint32_t two = *reinterpret_cast<const std::uint32_t*>(wrow + c);
        const __nv_bfloat162 wb = *reinterpret_cast<const __nv_bfloat162*>(&two);
        const float2 wf = __bfloat1622float2(wb);
        acc += wf.x * x[c] + wf.y * x[c + 1];
    }
    if (c < k) acc += __bfloat162float(__ushort_as_bfloat16(wrow[c])) * x[c];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

__global__ void matvec_bf16_kernel(const float* __restrict__ x,
                                   const std::uint16_t* __restrict__ w,
                                   const float* __restrict__ bias, float* __restrict__ y, int n,
                                   int k) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= n) return;
    const float acc = warp_dot_bf16(w + static_cast<std::size_t>(warp) * k, x, k);
    if ((threadIdx.x & 31) == 0) y[warp] = acc + (bias ? bias[warp] : 0.0f);
}

}  // namespace

void matvec_bf16(const float* x, const std::uint16_t* w, const float* bias, float* y, int n, int k,
                 cudaStream_t stream) {
    const int threads = 256;  // 8 warps -> 8 output rows per block
    const dim3 grid((n + 7) / 8);
    matvec_bf16_kernel<<<grid, threads, 0, stream>>>(x, w, bias, y, n, k);
}

void matmul_t(const float* x, const float* w, const float* bias, float* y, int m, int n, int k,
              cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    matmul_t_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, m, n, k);
}

void matmul_t_bf16(const float* x, const std::uint16_t* w, const float* bias, float* y, int m,
                   int n, int k, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    matmul_t_bf16_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, m, n, k);
}

bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

DeviceInfo device_info(int ordinal) {
    DeviceInfo info{};
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, ordinal) == cudaSuccess) {
        std::strncpy(info.name, prop.name, sizeof(info.name) - 1);
        info.compute_major = prop.major;
        info.compute_minor = prop.minor;
        info.total_memory = prop.totalGlobalMem;
        info.multi_processor_count = prop.multiProcessorCount;
    }
    return info;
}

}  // namespace cuda
}  // namespace uocr
