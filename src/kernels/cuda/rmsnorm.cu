// RMSNorm kernel (see uocr/ops.h for the reference semantics).

#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ y, int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + static_cast<std::size_t>(row) * cols;
    float* yr = y + static_cast<std::size_t>(row) * cols;

    float ss = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) ss += xr[c] * xr[c];

    // block reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sdata[0] / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = xr[c] * inv * w[c];
}

}  // namespace

void rmsnorm(const float* x, const float* weight, float* out, int rows, int cols, float eps,
             cudaStream_t stream) {
    const int threads = 256;
    rmsnorm_kernel<<<rows, threads, threads * sizeof(float), stream>>>(x, weight, out, rows, cols,
                                                                       eps);
}

}  // namespace cuda
}  // namespace uocr
