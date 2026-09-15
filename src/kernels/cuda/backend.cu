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

}  // namespace

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
