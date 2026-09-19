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
                                   float* __restrict__ out, const int* __restrict__ d_len) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int group = heads / kv_heads;
    const int kvh = head / group;
    // The effective length is either baked at capture time (d_len == nullptr) or
    // read on device so a captured graph can replay across the warmup boundary.
    if (d_len != nullptr) kv_len = *d_len;

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

// Batched decode attention: one block per (batch slot, query head).  Identical
// online-softmax walk as `rswa_decode_kernel`, but the per-slot KV base and the
// effective length come from the batch arrays, so a whole step is one launch.
template <int MAX_HD>
__global__ void rswa_decode_batch_kernel(const float* __restrict__ q,
                                         const float* __restrict__ kcache_base,
                                         const float* __restrict__ vcache_base,
                                         const int* __restrict__ d_len,
                                         const int* __restrict__ d_slots, int heads, int kv_heads,
                                         int head_dim, int batch_cap, float scale,
                                         float* __restrict__ out) {
    const int b = blockIdx.x;
    const int head = blockIdx.y;
    const int slot = d_slots[b];
    const int tid = threadIdx.x;
    const int group = heads / kv_heads;
    const int kvh = head / group;
    const int stride = kv_heads * head_dim;
    const int kv_len = d_len[slot];
    const float* kcache = kcache_base + static_cast<std::size_t>(slot) * batch_cap * stride;
    const float* vcache = vcache_base + static_cast<std::size_t>(slot) * batch_cap * stride;
    const std::size_t qoff = (static_cast<std::size_t>(b) * heads + head) * head_dim;

    __shared__ float q_sh[MAX_HD];
    __shared__ float acc[MAX_HD];
    __shared__ float red[256];
    __shared__ float m_sh, l_sh;

    if (tid == 0) {
        m_sh = -1e30f;
        l_sh = 0.0f;
    }
    if (tid < head_dim) {
        q_sh[tid] = q[qoff + tid];
        acc[tid] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < kv_len; ++t) {
        const std::size_t base = (static_cast<std::size_t>(t) * kv_heads + kvh) * head_dim;
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
        out[qoff + tid] = acc[tid] * inv;
    }
}

// General kernel: one block per (query s, head).  Used for prefill (causal) and
// as a fallback for decode.
template <int MAX_HD>
__global__ void rswa_attn_kernel(const float* __restrict__ q, const float* __restrict__ kcache,
                                 const float* __restrict__ vcache, int kv_len, int heads,
                                 int kv_heads, int head_dim, float scale, int q_start, int causal,
                                 float* __restrict__ out) {
    const int qidx = blockIdx.x;
    const int head = qidx % heads;
    const int s = qidx / heads;
    const int tid = threadIdx.x;
    const int group = heads / kv_heads;
    const int kvh = head / group;
    const int qpos = q_start + s;
    const int limit = causal ? min(kv_len, qpos + 1) : kv_len;
    const float* qh = q + static_cast<std::size_t>(qidx) * head_dim;

    __shared__ float q_sh[MAX_HD];
    __shared__ float acc[MAX_HD];
    __shared__ float red[256];
    __shared__ float m_sh, l_sh;

    if (tid == 0) {
        m_sh = -1e30f;
        l_sh = 0.0f;
    }
    if (tid < head_dim) {
        q_sh[tid] = qh[tid];
        acc[tid] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < limit; ++t) {
        const std::size_t base =
            (static_cast<std::size_t>(t) * kv_heads + kvh) * head_dim;
        float part = 0.0f;
        if (tid < head_dim) part = q_sh[tid] * kcache[base + tid];
        red[tid] = part;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
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
        out[static_cast<std::size_t>(qidx) * head_dim + tid] = acc[tid] * inv;
    }
}

// Batched append: one block per batch slot.  Each slot owns an independent ring
// cursor and (possibly different) prefill length, read from the batch arrays.
__global__ void rswa_append_batch_kernel(const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         float* __restrict__ kcache_base,
                                         float* __restrict__ vcache_base,
                                         int* __restrict__ d_len, int* __restrict__ d_ring_pos,
                                         const int* __restrict__ d_prefill_len,
                                         const int* __restrict__ d_slots, int batch_cap,
                                         int window, int stride) {
    const int b = blockIdx.x;
    const int slot = d_slots[b];
    const int prefill_len = d_prefill_len[slot];
    int len = d_len[slot];
    int row;
    if (len < prefill_len + window) {
        row = len;
        d_len[slot] = len + 1;
        if (len + 1 >= prefill_len + window) d_ring_pos[slot] = 0;
    } else {
        row = prefill_len + d_ring_pos[slot];
        d_ring_pos[slot] = (d_ring_pos[slot] + 1) % window;
    }
    const float* kb = k + static_cast<std::size_t>(b) * stride;
    const float* vb = v + static_cast<std::size_t>(b) * stride;
    float* kc = kcache_base + (static_cast<std::size_t>(slot) * batch_cap + row) * stride;
    float* vc = vcache_base + (static_cast<std::size_t>(slot) * batch_cap + row) * stride;
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
        kc[i] = kb[i];
        vc[i] = vb[i];
    }
}

}  // namespace

void rswa_attention(const float* q, const float* kcache, const float* vcache, int kv_len, int seq,
                    int q_start, int heads, int kv_heads, int head_dim, bool causal, float* out,
                    cudaStream_t stream) {
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int threads = 256;
    const int blocks = seq * heads;
    if (head_dim <= 64)
        rswa_attn_kernel<64><<<blocks, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                             kv_heads, head_dim, scale, q_start,
                                                             causal ? 1 : 0, out);
    else if (head_dim <= 128)
        rswa_attn_kernel<128><<<blocks, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                              kv_heads, head_dim, scale, q_start,
                                                              causal ? 1 : 0, out);
    else
        rswa_attn_kernel<256><<<blocks, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                              kv_heads, head_dim, scale, q_start,
                                                              causal ? 1 : 0, out);
}

void rswa_attention_decode(const float* q, const float* kcache, const float* vcache, int kv_len,
                           int heads, int kv_heads, int head_dim, float* out,
                           cudaStream_t stream) {
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int threads = 256;
    if (head_dim <= 64)
        rswa_decode_kernel<64><<<heads, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                              kv_heads, head_dim, scale, out,
                                                              nullptr);
    else if (head_dim <= 128)
        rswa_decode_kernel<128><<<heads, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                               kv_heads, head_dim, scale, out,
                                                               nullptr);
    else
        rswa_decode_kernel<256><<<heads, threads, 0, stream>>>(q, kcache, vcache, kv_len, heads,
                                                               kv_heads, head_dim, scale, out,
                                                               nullptr);
}

void rswa_attention_devlen(const float* q, const float* kcache, const float* vcache,
                           const int* d_len, int seq, int q_start, int heads, int kv_heads,
                           int head_dim, bool causal, float* out, cudaStream_t stream) {
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int threads = 256;
    const int blocks = seq * heads;
    if (head_dim <= 64)
        rswa_decode_kernel<64><<<blocks, threads, 0, stream>>>(q, kcache, vcache, 0, heads,
                                                               kv_heads, head_dim, scale, out,
                                                               d_len);
    else if (head_dim <= 128)
        rswa_decode_kernel<128><<<blocks, threads, 0, stream>>>(q, kcache, vcache, 0, heads,
                                                                kv_heads, head_dim, scale, out,
                                                                d_len);
    else
        rswa_decode_kernel<256><<<blocks, threads, 0, stream>>>(q, kcache, vcache, 0, heads,
                                                                kv_heads, head_dim, scale, out,
                                                                d_len);
}

void rswa_attention_batch(const float* q, const float* kcache_base, const float* vcache_base,
                          const int* d_len, const int* d_slots, int batch, int batch_cap,
                          int heads, int kv_heads, int head_dim, float* out,
                          cudaStream_t stream) {
    if (batch <= 0) return;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int threads = 256;
    dim3 grid(batch, heads);
    if (head_dim <= 64)
        rswa_decode_batch_kernel<64><<<grid, threads, 0, stream>>>(
            q, kcache_base, vcache_base, d_len, d_slots, heads, kv_heads, head_dim, batch_cap,
            scale, out);
    else if (head_dim <= 128)
        rswa_decode_batch_kernel<128><<<grid, threads, 0, stream>>>(
            q, kcache_base, vcache_base, d_len, d_slots, heads, kv_heads, head_dim, batch_cap,
            scale, out);
    else
        rswa_decode_batch_kernel<256><<<grid, threads, 0, stream>>>(
            q, kcache_base, vcache_base, d_len, d_slots, heads, kv_heads, head_dim, batch_cap,
            scale, out);
}

namespace {

// Single block writes one row of K/V and advances the device ring pointer.
__global__ void rswa_append_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                   float* __restrict__ kcache, float* __restrict__ vcache,
                                   int* __restrict__ d_len, int* __restrict__ d_ring_pos,
                                   int prefill_len, int window, int stride) {
    int len = *d_len;
    int slot;
    if (len < prefill_len + window) {
        slot = len;
        *d_len = len + 1;
        if (len + 1 >= prefill_len + window) *d_ring_pos = 0;
    } else {
        slot = prefill_len + *d_ring_pos;
        *d_ring_pos = (*d_ring_pos + 1) % window;
    }
    for (int i = threadIdx.x; i < stride; i += blockDim.x) {
        kcache[static_cast<std::size_t>(slot) * stride + i] = k[i];
        vcache[static_cast<std::size_t>(slot) * stride + i] = v[i];
    }
}

}  // namespace

void rswa_append_decode(const float* k, const float* v, float* kcache, float* vcache, int* d_len,
                        int* d_ring_pos, int prefill_len, int window, int kv_heads, int head_dim,
                        cudaStream_t stream) {
    const int stride = kv_heads * head_dim;
    rswa_append_kernel<<<1, 256, 0, stream>>>(k, v, kcache, vcache, d_len, d_ring_pos, prefill_len,
                                              window, stride);
}

void rswa_append_decode_batch(const float* k, const float* v, float* kcache_base,
                              float* vcache_base, int* d_len, int* d_ring_pos,
                              const int* d_prefill_len, const int* d_slots, int batch,
                              int batch_cap, int window, int kv_heads, int head_dim,
                              cudaStream_t stream) {
    if (batch <= 0) return;
    const int stride = kv_heads * head_dim;
    rswa_append_batch_kernel<<<batch, 256, 0, stream>>>(k, v, kcache_base, vcache_base, d_len,
                                                        d_ring_pos, d_prefill_len, d_slots,
                                                        batch_cap, window, stride);
}

}  // namespace cuda
}  // namespace uocr
