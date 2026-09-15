// R-SWA decode attention kernel.
//
// One thread block per query head.  The block walks the cache slots
// [0, kv_len) with an online (flash-style) softmax, so no score matrix is
// materialized.  Reference-region tokens and ring tokens are distinguished
// only by their slot index, which keeps the kernel CUDA-Graph friendly: the
// cache base address and kv_len are stable across replays.

#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

template <int MAX_HD>
__global__ void rswa_decode_kernel(const float* __restrict__ q, const float* __restrict__ kcache,
                                   const float* __restrict__ vcache, int kv_len, int heads,
                                   int kv_heads, int head_dim, float scale,
                                   float* __restrict__ out) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int group = heads / kv_heads;
    const int kvh = head / group;

    __shared__ float q_sh[MAX_HD];
    __shared__ float acc[MAX_HD];
    __shared__ float red[256];
    __shared__ float m_sh, l_sh;

    if (tid == 0) {
        m_sh = -1e30f;
        l_sh = 0.0f;
    }
    if (tid < head_dim) {
        q_sh[tid] = q[static_cast<std::size_t>(head) * head_dim + tid];
        acc[tid] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < kv_len; ++t) {
        const std::size_t base =
            (static_cast<std::size_t>(t) * kv_heads + kvh) * head_dim;
        float part = 0.0f;
        if (tid < head_dim) part = q_sh[tid] * kcache[base + tid];
        red[tid] = part;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(m_sh, dot);
            const float alpha = __expf(m_sh - m_new);
            const float beta = __expf(dot - m_new);
            red[0] = alpha;
            red[1] = beta;
            m_sh = m_new;
            l_sh = l_sh * alpha + beta;
        }
        __syncthreads();
        const float alpha = red[0];
        const float beta = red[1];
        if (tid < head_dim) acc[tid] = acc[tid] * alpha + beta * vcache[base + tid];
        __syncthreads();
    }

    if (tid < head_dim) {
        const float inv = l_sh > 0.0f ? 1.0f / l_sh : 0.0f;
        out[static_cast<std::size_t>(head) * head_dim + tid] = acc[tid] * inv;
    }
}

}  // namespace

void rswa_attention_decode(const float* q, const float* kcache, const float* vcache, int kv_len,
                           int heads, int kv_heads, int head_dim, float* out,
                           cudaStream_t stream) {
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int threads = 256;
    if (head_dim <= 64)
        rswa_decode_kernel<64><<<heads, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                              kv_heads, head_dim, scale, out);
    else if (head_dim <= 128)
        rswa_decode_kernel<128><<<heads, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                               kv_heads, head_dim, scale, out);
    else
        rswa_decode_kernel<256><<<heads, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                               kv_heads, head_dim, scale, out);
}

}  // namespace cuda
}  // namespace uocr
