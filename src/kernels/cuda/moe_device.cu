// Device-side MoE routing + masked expert MLP kernels.
//
// All launch configurations depend only on static model shape (never on the
// routing result), which is what makes the decode step capturable by a CUDA
// Graph.  The router groups tokens per expert using atomic counters; the
// expert kernels launch a fixed grid per expert and return immediately when
// the expert received no tokens.
//
// The expert MLP is split into gate_up + down so the output-feature axis is
// spread over many warps.  Each warp computes one output feature with
// coalesced vectorized weight reads (one warp per row), mirroring the dense
// matvec kernel.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

// One block per token.  `srow` holds the (unnormalised) expert probabilities.
__global__ void moe_router_topk_kernel(const float* __restrict__ logits, int n_experts, int top_k,
                                       int norm_topk_prob, float scaling, int sigmoid,
                                       int* __restrict__ ids, float* __restrict__ weights,
                                       int* __restrict__ assign_token,
                                       float* __restrict__ assign_w, int* __restrict__ count,
                                       int cap) {
    extern __shared__ float srow[];
    __shared__ float sred[256];
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* src = logits + static_cast<std::size_t>(t) * n_experts;

    if (sigmoid) {
        for (int e = tid; e < n_experts; e += blockDim.x)
            srow[e] = 1.0f / (1.0f + __expf(-src[e]));
    } else {
        float lmax = -1e30f;
        for (int e = tid; e < n_experts; e += blockDim.x) lmax = fmaxf(lmax, src[e]);
        sred[tid] = lmax;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sred[tid] = fmaxf(sred[tid], sred[tid + s]);
            __syncthreads();
        }
        const float mx = sred[0];
        __syncthreads();
        float lsum = 0.0f;
        for (int e = tid; e < n_experts; e += blockDim.x) {
            const float p = __expf(src[e] - mx);
            srow[e] = p;
            lsum += p;
        }
        sred[tid] = lsum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sred[tid] += sred[tid + s];
            __syncthreads();
        }
        const float sum = sred[0];
        __syncthreads();
        for (int e = tid; e < n_experts; e += blockDim.x) srow[e] /= sum;
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < top_k; ++j) {
            int best = -1;
            float bestv = -1e30f;
            for (int e = 0; e < n_experts; ++e)
                if (srow[e] > bestv) {
                    bestv = srow[e];
                    best = e;
                }
            ids[static_cast<std::size_t>(t) * top_k + j] = best;
            weights[static_cast<std::size_t>(t) * top_k + j] = bestv;
            srow[best] = -1e30f;
        }
        float* wrow = weights + static_cast<std::size_t>(t) * top_k;
        if (top_k > 1 && norm_topk_prob) {
            float sum = 0.0f;
            for (int j = 0; j < top_k; ++j) sum += wrow[j];
            for (int j = 0; j < top_k; ++j) wrow[j] = wrow[j] / (sum + 1e-20f) * scaling;
        } else {
            for (int j = 0; j < top_k; ++j) wrow[j] *= scaling;
        }
        for (int j = 0; j < top_k; ++j) {
            const int e = ids[static_cast<std::size_t>(t) * top_k + j];
            const int slot = atomicAdd(&count[e], 1);
            if (slot < cap) {
                assign_token[static_cast<std::size_t>(e) * cap + slot] = t;
                assign_w[static_cast<std::size_t>(e) * cap + slot] = wrow[j];
            }
        }
    }
}

// Warp-wide dot product of a bf16 row with `x`, coalesced and 2-wide
// vectorized.  Result is valid on lane 0.
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

// Same for a packed INT4 row with per-group (scale, zero).
__device__ __forceinline__ float warp_dot_i4(const std::uint8_t* __restrict__ prow,
                                             const float* __restrict__ sc,
                                             const float* __restrict__ z,
                                             const float* __restrict__ x, int k, int group) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    int c = lane * 2;
    for (; c + 1 < k; c += 64) {
        const std::uint8_t b = prow[c >> 1];
        const float q0 = static_cast<float>(b & 0x0f);
        const float q1 = static_cast<float>(b >> 4);
        const int gg = c / group;
        const float s = sc[gg], zz = z[gg];
        acc += (q0 - zz) * s * x[c] + (q1 - zz) * s * x[c + 1];
    }
    if (c < k) {
        const std::uint8_t b = prow[c >> 1];
        const float q = static_cast<float>(b & 0x0f);
        const int gg = c / group;
        acc += (q - z[gg]) * sc[gg] * x[c];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

// act[e, s, i] = silu(gate_i(x_t)) * up_i(x_t); one warp per output i.
__global__ void expert_gate_up_kernel(const float* __restrict__ x,
                                      const std::uint16_t* const* __restrict__ gate_w,
                                      const std::uint16_t* const* __restrict__ up_w,
                                      const int* __restrict__ assign_token,
                                      const int* __restrict__ count, int cap, int hidden, int inter,
                                      float* __restrict__ act) {
    const int e = blockIdx.x;
    const int n = count[e];
    if (n <= 0) return;
    const int warp = threadIdx.x >> 5;
    const int i = blockIdx.y * (blockDim.x >> 5) + warp;
    if (i >= inter) return;
    const std::uint16_t* grow = gate_w[e] + static_cast<std::size_t>(i) * hidden;
    const std::uint16_t* urow = up_w[e] + static_cast<std::size_t>(i) * hidden;
    const int lane = threadIdx.x & 31;
    for (int s = 0; s < n; ++s) {
        const int t = assign_token[static_cast<std::size_t>(e) * cap + s];
        const float* xr = x + static_cast<std::size_t>(t) * hidden;
        const float g = warp_dot_bf16(grow, xr, hidden);
        const float u = warp_dot_bf16(urow, xr, hidden);
        if (lane == 0)
            act[(static_cast<std::size_t>(e) * cap + s) * inter + i] = (g / (1.0f + __expf(-g))) * u;
    }
}

// out[t] += w * down(act[e, s]); one warp per output j.
__global__ void expert_down_kernel(const std::uint16_t* const* __restrict__ down_w,
                                   const int* __restrict__ assign_token,
                                   const float* __restrict__ assign_w, const int* __restrict__ count,
                                   int cap, int hidden, int inter, const float* __restrict__ act,
                                   float* __restrict__ out) {
    const int e = blockIdx.x;
    const int n = count[e];
    if (n <= 0) return;
    const int warp = threadIdx.x >> 5;
    const int j = blockIdx.y * (blockDim.x >> 5) + warp;
    if (j >= hidden) return;
    const std::uint16_t* drow = down_w[e] + static_cast<std::size_t>(j) * inter;
    const int lane = threadIdx.x & 31;
    for (int s = 0; s < n; ++s) {
        const float* a = act + (static_cast<std::size_t>(e) * cap + s) * inter;
        const float acc = warp_dot_bf16(drow, a, inter);
        if (lane == 0) {
            const int t = assign_token[static_cast<std::size_t>(e) * cap + s];
            const float w = assign_w[static_cast<std::size_t>(e) * cap + s];
            atomicAdd(&out[static_cast<std::size_t>(t) * hidden + j], w * acc);
        }
    }
}

__global__ void expert_gate_up_int4_kernel(
    const float* __restrict__ x, const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales, const float* __restrict__ gate_zeros, int gate_pstride,
    int gate_sstride, int gate_ng, const std::uint8_t* __restrict__ up_packed,
    const float* __restrict__ up_scales, const float* __restrict__ up_zeros, int up_pstride,
    int up_sstride, int up_ng, const int* __restrict__ assign_token,
    const int* __restrict__ count, int cap, int hidden, int inter, int group,
    float* __restrict__ act) {
    const int e = blockIdx.x;
    const int n = count[e];
    if (n <= 0) return;
    const int warp = threadIdx.x >> 5;
    const int i = blockIdx.y * (blockDim.x >> 5) + warp;
    if (i >= inter) return;
    const int gpr = (hidden + 1) / 2;
    const std::uint8_t* grow =
        gate_packed + static_cast<std::size_t>(e) * gate_pstride + static_cast<std::size_t>(i) * gpr;
    const std::uint8_t* urow =
        up_packed + static_cast<std::size_t>(e) * up_pstride + static_cast<std::size_t>(i) * gpr;
    const float* gsc =
        gate_scales + static_cast<std::size_t>(e) * gate_sstride + static_cast<std::size_t>(i) * gate_ng;
    const float* gzr =
        gate_zeros + static_cast<std::size_t>(e) * gate_sstride + static_cast<std::size_t>(i) * gate_ng;
    const float* usc =
        up_scales + static_cast<std::size_t>(e) * up_sstride + static_cast<std::size_t>(i) * up_ng;
    const float* uzr =
        up_zeros + static_cast<std::size_t>(e) * up_sstride + static_cast<std::size_t>(i) * up_ng;
    const int lane = threadIdx.x & 31;
    for (int s = 0; s < n; ++s) {
        const int t = assign_token[static_cast<std::size_t>(e) * cap + s];
        const float* xr = x + static_cast<std::size_t>(t) * hidden;
        const float g = warp_dot_i4(grow, gsc, gzr, xr, hidden, group);
        const float u = warp_dot_i4(urow, usc, uzr, xr, hidden, group);
        if (lane == 0)
            act[(static_cast<std::size_t>(e) * cap + s) * inter + i] = (g / (1.0f + __expf(-g))) * u;
    }
}

__global__ void expert_down_int4_kernel(
    const std::uint8_t* __restrict__ down_packed, const float* __restrict__ down_scales,
    const float* __restrict__ down_zeros, int down_pstride, int down_sstride, int down_ng,
    const int* __restrict__ assign_token, const float* __restrict__ assign_w,
    const int* __restrict__ count, int cap, int hidden, int inter, int group,
    const float* __restrict__ act, float* __restrict__ out) {
    const int e = blockIdx.x;
    const int n = count[e];
    if (n <= 0) return;
    const int warp = threadIdx.x >> 5;
    const int j = blockIdx.y * (blockDim.x >> 5) + warp;
    if (j >= hidden) return;
    const int dpr = (inter + 1) / 2;
    const std::uint8_t* drow =
        down_packed + static_cast<std::size_t>(e) * down_pstride + static_cast<std::size_t>(j) * dpr;
    const float* dsc =
        down_scales + static_cast<std::size_t>(e) * down_sstride + static_cast<std::size_t>(j) * down_ng;
    const float* dzr =
        down_zeros + static_cast<std::size_t>(e) * down_sstride + static_cast<std::size_t>(j) * down_ng;
    const int lane = threadIdx.x & 31;
    for (int s = 0; s < n; ++s) {
        const float* a = act + (static_cast<std::size_t>(e) * cap + s) * inter;
        const float acc = warp_dot_i4(drow, dsc, dzr, a, inter, group);
        if (lane == 0) {
            const int t = assign_token[static_cast<std::size_t>(e) * cap + s];
            const float w = assign_w[static_cast<std::size_t>(e) * cap + s];
            atomicAdd(&out[static_cast<std::size_t>(t) * hidden + j], w * acc);
        }
    }
}

}  // namespace

void moe_router_topk(const float* router_logits, int seq, int n_experts, int top_k,
                     bool norm_topk_prob, float routed_scaling_factor, bool scoring_sigmoid,
                     int* d_ids, float* d_weights, int* d_assign_token, float* d_assign_w,
                     int* d_count, int cap, cudaStream_t stream) {
    if (seq == 0 || n_experts == 0) return;
    const int threads = 128;
    const std::size_t shmem = static_cast<std::size_t>(n_experts) * sizeof(float);
    moe_router_topk_kernel<<<seq, threads, shmem, stream>>>(
        router_logits, n_experts, top_k, norm_topk_prob ? 1 : 0, routed_scaling_factor,
        scoring_sigmoid ? 1 : 0, d_ids, d_weights, d_assign_token, d_assign_w, d_count, cap);
}

void moe_experts_masked(const float* x, int seq, const std::uint16_t* const* gate_w,
                        const std::uint16_t* const* up_w, const std::uint16_t* const* down_w,
                        const int* assign_token, const float* assign_w, const int* count,
                        int n_experts, int cap, int hidden, int inter, float* act, float* out,
                        cudaStream_t stream) {
    if (n_experts == 0) return;
    const int threads = 256;  // 8 warps
    const int warps = threads / 32;
    const dim3 gu_grid(n_experts, (inter + warps - 1) / warps);
    expert_gate_up_kernel<<<gu_grid, threads, 0, stream>>>(x, gate_w, up_w, assign_token, count,
                                                           cap, hidden, inter, act);
    const dim3 d_grid(n_experts, (hidden + warps - 1) / warps);
    expert_down_kernel<<<d_grid, threads, 0, stream>>>(down_w, assign_token, assign_w, count, cap,
                                                       hidden, inter, act, out);
    (void)seq;
}

void moe_experts_masked_int4(
    const float* x, int seq, const std::uint8_t* gate_packed, const float* gate_scales,
    const float* gate_zeros, int gate_pstride, int gate_sstride, int gate_ng,
    const std::uint8_t* up_packed, const float* up_scales, const float* up_zeros, int up_pstride,
    int up_sstride, int up_ng, const std::uint8_t* down_packed, const float* down_scales,
    const float* down_zeros, int down_pstride, int down_sstride, int down_ng,
    const int* assign_token, const float* assign_w, const int* count, int n_experts, int cap,
    int hidden, int inter, int group, float* act, float* out, cudaStream_t stream) {
    if (n_experts == 0) return;
    const int threads = 256;
    const int warps = threads / 32;
    const dim3 gu_grid(n_experts, (inter + warps - 1) / warps);
    expert_gate_up_int4_kernel<<<gu_grid, threads, 0, stream>>>(
        x, gate_packed, gate_scales, gate_zeros, gate_pstride, gate_sstride, gate_ng, up_packed,
        up_scales, up_zeros, up_pstride, up_sstride, up_ng, assign_token, count, cap, hidden, inter,
        group, act);
    const dim3 d_grid(n_experts, (hidden + warps - 1) / warps);
    expert_down_int4_kernel<<<d_grid, threads, 0, stream>>>(down_packed, down_scales, down_zeros,
                                                           down_pstride, down_sstride, down_ng,
                                                           assign_token, assign_w, count, cap,
                                                           hidden, inter, group, act, out);
    (void)seq;
}

}  // namespace cuda
}  // namespace uocr
