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

// Shared-memory tiled BF16 GEMM.  Each block computes a 64x64 output tile and
// stages 64x16 A/B panels with coalesced global loads; a 16x16 thread block
// computes a 4x4 register tile per thread.  The panels are padded by one
// column to avoid shared-memory bank conflicts.
constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;

__global__ void matmul_t_bf16_kernel(const float* __restrict__ x,
                                     const std::uint16_t* __restrict__ w,
                                     const float* __restrict__ bias, float* __restrict__ y, int m,
                                     int n, int k) {
    __shared__ float As[kBM][kBK + 1];
    __shared__ float Bs[kBN][kBK + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int m0 = blockIdx.y * kBM;
    const int n0 = blockIdx.x * kBN;
    float acc[4][4] = {};

    for (int k0 = 0; k0 < k; k0 += kBK) {
#pragma unroll
        for (int l = 0; l < (kBM * kBK) / 256; ++l) {
            const int idx = tid + l * 256;
            const int r = idx / kBK, c = idx % kBK;
            const int gm = m0 + r, gk = k0 + c;
            As[r][c] = (gm < m && gk < k) ? x[static_cast<std::size_t>(gm) * k + gk] : 0.0f;
        }
#pragma unroll
        for (int l = 0; l < (kBN * kBK) / 256; ++l) {
            const int idx = tid + l * 256;
            const int r = idx / kBK, c = idx % kBK;
            const int gn = n0 + r, gk = k0 + c;
            Bs[r][c] = (gn < n && gk < k)
                           ? __bfloat162float(__ushort_as_bfloat16(
                                 w[static_cast<std::size_t>(gn) * k + gk]))
                           : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kBK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[ty * 4 + i][kk];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[tx * 4 + j][kk];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = m0 + ty * 4 + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = n0 + tx * 4 + j;
            if (col >= n) continue;
            y[static_cast<std::size_t>(row) * n + col] = acc[i][j] + (bias ? bias[col] : 0.0f);
        }
    }
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
    dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
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
