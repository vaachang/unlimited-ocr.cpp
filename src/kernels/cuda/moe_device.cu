// Device-side MoE routing + fused "all experts, masked skip" MLP.
//
// Both kernels have a launch configuration that depends only on compile-time
// model shape (never on the routing result), which is what makes the decode
// step capturable by a CUDA Graph.  The router groups tokens per expert using
// atomic counters; the expert kernel launches one block per expert and skips
// the block immediately when the expert received no tokens.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>

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

    // Greedy top-k: one thread scans sequentially (n_experts is small).
    if (threadIdx.x == 0) {
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

__device__ __forceinline__ float bf16_to_f32(std::uint16_t h) {
    return __bfloat162float(__ushort_as_bfloat16(h));
}

// Grid = n_experts.  Each block processes the tokens assigned to its expert.
__global__ void moe_experts_masked_kernel(
    const float* __restrict__ x, const std::uint16_t* const* __restrict__ gate_w,
    const std::uint16_t* const* __restrict__ up_w,
    const std::uint16_t* const* __restrict__ down_w, const int* __restrict__ assign_token,
    const float* __restrict__ assign_w, const int* __restrict__ count, int cap, int hidden,
    int inter, float* __restrict__ out) {
    extern __shared__ float act[];
    const int e = blockIdx.x;
    const int n = count[e];
    if (n <= 0) return;
    const std::uint16_t* gw = gate_w[e];
    const std::uint16_t* uw = up_w[e];
    const std::uint16_t* dw = down_w[e];

    for (int s = 0; s < n; ++s) {
        const int t = assign_token[static_cast<std::size_t>(e) * cap + s];
        const float w = assign_w[static_cast<std::size_t>(e) * cap + s];
        const float* xr = x + static_cast<std::size_t>(t) * hidden;

        for (int i = threadIdx.x; i < inter; i += blockDim.x) {
            const std::uint16_t* grow = gw + static_cast<std::size_t>(i) * hidden;
            const std::uint16_t* urow = uw + static_cast<std::size_t>(i) * hidden;
            float g = 0.0f, u = 0.0f;
            for (int c = 0; c < hidden; ++c) {
                const float xv = xr[c];
                g += bf16_to_f32(grow[c]) * xv;
                u += bf16_to_f32(urow[c]) * xv;
            }
            act[i] = (g / (1.0f + __expf(-g))) * u;
        }
        __syncthreads();

        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            const std::uint16_t* drow = dw + static_cast<std::size_t>(j) * inter;
            float acc = 0.0f;
            for (int c = 0; c < inter; ++c) acc += bf16_to_f32(drow[c]) * act[c];
            atomicAdd(&out[static_cast<std::size_t>(t) * hidden + j], w * acc);
        }
        __syncthreads();
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
                        int n_experts, int cap, int hidden, int inter, float* out,
                        cudaStream_t stream) {
    if (n_experts == 0) return;
    const int threads = 256;
    const std::size_t shmem = static_cast<std::size_t>(inter) * sizeof(float);
    moe_experts_masked_kernel<<<n_experts, threads, shmem, stream>>>(
        x, gate_w, up_w, down_w, assign_token, assign_w, count, cap, hidden, inter, out);
    (void)seq;
}

}  // namespace cuda
}  // namespace uocr
