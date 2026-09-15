// RoPE and fused per-head RMSNorm+RoPE kernels.

#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

// One thread per (position, head, pair).
__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k,
                            const int* __restrict__ positions, int seq, int n_heads,
                            int n_kv_heads, int head_dim, float theta) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = head_dim / 2;
    if (pair >= half) return;

    const int row = blockIdx.y;          // sequence position
    const int head = blockIdx.z;         // head index
    const int pos = positions ? positions[row] : row;
    const float freq = powf(theta, -2.0f * pair / static_cast<float>(head_dim));
    const float angle = pos * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);

    if (head < n_heads && q) {
        float* v = q + (static_cast<std::size_t>(row) * n_heads + head) * head_dim;
        const float x0 = v[pair];
        const float x1 = v[pair + half];
        v[pair] = x0 * c - x1 * s;
        v[pair + half] = x0 * s + x1 * c;
    }
    if (head < n_kv_heads && k) {
        float* v = k + (static_cast<std::size_t>(row) * n_kv_heads + head) * head_dim;
        const float x0 = v[pair];
        const float x1 = v[pair + half];
        v[pair] = x0 * c - x1 * s;
        v[pair + half] = x0 * s + x1 * c;
    }
}

// Per-head RMSNorm (single block per row) then RoPE.
__global__ void rmsnorm_head_kernel(float* __restrict__ q, float* __restrict__ k,
                                    const float* __restrict__ w, const int* __restrict__ positions,
                                    int n_heads, int n_kv_heads, int head_dim, float theta,
                                    float eps) {
    // block = one row, one head-type handled by blockIdx.y (0=q,1=k)
    const int row = blockIdx.x;
    const int type = blockIdx.y;
    const int heads = type == 0 ? n_heads : n_kv_heads;
    float* base = type == 0 ? q : k;
    if (!base) return;

    for (int h = 0; h < heads; ++h) {
        float* v = base + (static_cast<std::size_t>(row) * heads + h) * head_dim;
        float ss = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) ss += v[d] * v[d];
        extern __shared__ float sdata[];
        sdata[threadIdx.x] = ss;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        const float inv = rsqrtf(sdata[0] / head_dim + eps);
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) v[d] = v[d] * inv * w[d];
        __syncthreads();

        const int pos = positions ? positions[row] : row;
        const int half = head_dim / 2;
        for (int p = threadIdx.x; p < half; p += blockDim.x) {
            const float freq = powf(theta, -2.0f * p / static_cast<float>(head_dim));
            const float angle = pos * freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float x0 = v[p];
            const float x1 = v[p + half];
            v[p] = x0 * c - x1 * s;
            v[p + half] = x0 * s + x1 * c;
        }
        __syncthreads();
    }
}

}  // namespace

void rope(float* q, float* k, const int* positions, int seq, int n_heads, int n_kv_heads,
          int head_dim, float theta, cudaStream_t stream) {
    const int half = head_dim / 2;
    const int threads = 64;
    const int blocks = (half + threads - 1) / threads;
    dim3 grid(blocks, seq, n_heads > n_kv_heads ? n_heads : n_kv_heads);
    rope_kernel<<<grid, threads, 0, stream>>>(q, k, positions, seq, n_heads, n_kv_heads, head_dim,
                                              theta);
}

void rmsnorm_rope(float* q, float* k, const float* weight, const int* positions, int seq,
                  int n_heads, int n_kv_heads, int head_dim, float theta, float eps,
                  cudaStream_t stream) {
    const int threads = 128;
    dim3 grid(seq, 2);
    rmsnorm_head_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        q, k, weight, positions, n_heads, n_kv_heads, head_dim, theta, eps);
}

}  // namespace cuda
}  // namespace uocr
