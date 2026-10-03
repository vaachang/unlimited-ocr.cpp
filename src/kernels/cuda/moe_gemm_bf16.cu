// Grouped BF16 MoE expert GEMM kernels (task 2.14).
//
// These mirror the grouped INT4 kernels in `moe_gemm_int4.cu`: one launch per
// projection computes every expert for a device-resident grouping table
// (`assign_token` / `assign_w` / `count`, produced by `moe_router_topk`), so a
// ragged multi-request prefill with BF16 experts no longer needs a host router
// D2H plus a per-expert GEMM loop.
//
// The difference is only the weight source: BF16 experts live in one device
// allocation per expert, addressed through a device array of row-major bf16
// pointers (the same `gate_ptrs` / `up_ptrs` / `down_ptrs` arrays the masked
// kernel uses).  There is no dequantization step, so the weight panel is copied
// straight into shared memory and fed to `ldmatrix` / `mma` exactly like the
// INT4 path after dequant.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// Stage a [BN, BK] bf16 weight panel of one expert into shared memory with row
// stride `stride`.  `base` is the expert's [n, k] row-major bf16 matrix; the
// panel is rows [n0, n0+BN) and columns [k0, k0+BK).  Out-of-range entries are
// zeroed so the padded k tail cannot affect the mma.  The fast path uses 16-byte
// `uint4` loads (8 bf16) when the whole panel is in range; model k and the BK
// tile are multiples of 8, so this is the common case.
template <int BN, int BK>
__device__ __forceinline__ void stage_weight_bf16(__nv_bfloat16* __restrict__ sw, int stride,
                                                  const std::uint16_t* __restrict__ base, int n0,
                                                  int n, int k0, int k, int tid) {
    if (n0 + BN <= n && k0 + BK <= k && (k & 7) == 0) {
        constexpr int V = 8;
        constexpr int NV = BK / V;
        for (int idx = tid; idx < BN * NV; idx += blockDim.x) {
            const int r = idx / NV, c = (idx % NV) * V;
            const uint4 pack = *reinterpret_cast<const uint4*>(
                base + static_cast<std::size_t>(n0 + r) * k + k0 + c);
            *reinterpret_cast<uint4*>(&sw[r * stride + c]) = pack;
        }
        return;
    }
    for (int idx = tid; idx < BN * BK; idx += blockDim.x) {
        const int r = idx / BK, cc = idx % BK;
        const int gn = n0 + r, kk = k0 + cc;
        float wv = 0.0f;
        if (gn < n && kk < k)
            wv = __bfloat162float(__ushort_as_bfloat16(base[static_cast<std::size_t>(gn) * k + kk]));
        sw[r * stride + cc] = __float2bfloat16(wv);
    }
}

// Stage a [BM, BK] activation tile by gathering rows through `atok` (the
// expert's assignment list).  Rows >= `rows` become zero.
template <int BM, int BK>
__device__ __forceinline__ void bf16_stage_act_gather(__nv_bfloat16 (*sA)[BK + 8],
                                                      const float* __restrict__ x,
                                                      const int* __restrict__ atok, int m0,
                                                      int rows, int hidden, int k0, int tid) {
    constexpr int F4R = BK / 4;
    constexpr int NF4 = BM * F4R;
    for (int idx = tid; idx < NF4; idx += blockDim.x) {
        const int r = idx / F4R, c4 = (idx % F4R) * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < rows) {
            const float* xr = x + static_cast<std::size_t>(atok[m0 + r]) * hidden;
            const int gk = k0 + c4;
            if (gk + 4 <= hidden) {
                v = *reinterpret_cast<const float4*>(xr + gk);
            } else {
                float* vp = reinterpret_cast<float*>(&v);
#pragma unroll
                for (int t = 0; t < 4; ++t) vp[t] = (gk + t < hidden) ? xr[gk + t] : 0.0f;
            }
        }
        const __nv_bfloat162 p0 = __float22bfloat162_rn(make_float2(v.x, v.y));
        const __nv_bfloat162 p1 = __float22bfloat162_rn(make_float2(v.z, v.w));
        const unsigned u0 = *reinterpret_cast<const unsigned*>(&p0);
        const unsigned u1 = *reinterpret_cast<const unsigned*>(&p1);
        *reinterpret_cast<uint2*>(&sA[r][c4]) = make_uint2(u0, u1);
    }
}

// Stage a contiguous [BM, BK] activation tile from `a` (row stride `kdim`).
template <int BM, int BK>
__device__ __forceinline__ void bf16_stage_act_rows(__nv_bfloat16 (*sA)[BK + 8],
                                                    const float* __restrict__ a, int m0, int rows,
                                                    int kdim, int k0, int tid) {
    constexpr int F4R = BK / 4;
    constexpr int NF4 = BM * F4R;
    for (int idx = tid; idx < NF4; idx += blockDim.x) {
        const int r = idx / F4R, c4 = (idx % F4R) * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < rows) {
            const float* xr = a + static_cast<std::size_t>(m0 + r) * kdim;
            const int gk = k0 + c4;
            if (gk + 4 <= kdim) {
                v = *reinterpret_cast<const float4*>(xr + gk);
            } else {
                float* vp = reinterpret_cast<float*>(&v);
#pragma unroll
                for (int t = 0; t < 4; ++t) vp[t] = (gk + t < kdim) ? xr[gk + t] : 0.0f;
            }
        }
        const __nv_bfloat162 p0 = __float22bfloat162_rn(make_float2(v.x, v.y));
        const __nv_bfloat162 p1 = __float22bfloat162_rn(make_float2(v.z, v.w));
        const unsigned u0 = *reinterpret_cast<const unsigned*>(&p0);
        const unsigned u1 = *reinterpret_cast<const unsigned*>(&p1);
        *reinterpret_cast<uint2*>(&sA[r][c4]) = make_uint2(u0, u1);
    }
}

// gate + up for every expert, with fused SiLU, writing act[e, slot, :].
template <int BN, int BK, int BM>
__global__ void moe_grouped_gate_up_bf16_kernel(
    const float* __restrict__ x, const std::uint16_t* const* __restrict__ gate_w,
    const std::uint16_t* const* __restrict__ up_w, const int* __restrict__ assign_token,
    const int* __restrict__ count, int cap, int hidden, int inter, float* __restrict__ act) {
    constexpr int NT = BN / 8;
    constexpr int WS = BK + 8;
    alignas(16) __shared__ __nv_bfloat16 sA[BM][WS];
    alignas(16) __shared__ __nv_bfloat16 sWg[BN][WS];
    alignas(16) __shared__ __nv_bfloat16 sWu[BN][WS];

    const int e = blockIdx.y;
    const int n = count[e];
    if (n <= 0) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * BN;
    const std::uint16_t* gbase = gate_w[e];
    const std::uint16_t* ubase = up_w[e];
    const int* atok = assign_token + static_cast<std::size_t>(e) * cap;
    const int g = lane >> 2;
    const int tig = lane & 3;

    for (int m0 = 0; m0 < n; m0 += BM) {
        const int rows = min(BM, n - m0);
        const bool active = (warp * 16) < rows;
        float cg[NT][4] = {};
        float cu[NT][4] = {};
        for (int k0 = 0; k0 < hidden; k0 += BK) {
            bf16_stage_act_gather<BM, BK>(sA, x, atok, m0, rows, hidden, k0, tid);
            stage_weight_bf16<BN, BK>(&sWg[0][0], WS, gbase, n0, inter, k0, hidden, tid);
            stage_weight_bf16<BN, BK>(&sWu[0][0], WS, ubase, n0, inter, k0, hidden, tid);
            __syncthreads();
            if (active) {
#pragma unroll
                for (int kk = 0; kk < BK; kk += 16) {
                    const __nv_bfloat16* a_ptr = &sA[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8];
                    unsigned a0, a1, a2, a3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                                 : "r"(smem_addr(a_ptr)));
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const __nv_bfloat16* bg =
                            &sWg[nt * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8];
                        const __nv_bfloat16* bu =
                            &sWu[nt * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8];
                        unsigned bg0, bg1, bu0, bu1;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                                     : "=r"(bg0), "=r"(bg1)
                                     : "r"(smem_addr(bg)));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                                     : "=r"(bu0), "=r"(bu1)
                                     : "r"(smem_addr(bu)));
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            : "+f"(cg[nt][0]), "+f"(cg[nt][1]), "+f"(cg[nt][2]), "+f"(cg[nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bg0), "r"(bg1));
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            : "+f"(cu[nt][0]), "+f"(cu[nt][1]), "+f"(cu[nt][2]), "+f"(cu[nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bu0), "r"(bu1));
                    }
                }
            }
            __syncthreads();
        }
        if (active) {
            const int r0 = warp * 16 + g, r1 = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int c0 = n0 + nt * 8 + tig * 2, c1 = c0 + 1;
                if (r0 < rows) {
                    float* dst = act + (static_cast<std::size_t>(e) * cap + m0 + r0) * inter;
                    if (c0 < inter)
                        dst[c0] = (cg[nt][0] / (1.0f + __expf(-cg[nt][0]))) * cu[nt][0];
                    if (c1 < inter)
                        dst[c1] = (cg[nt][1] / (1.0f + __expf(-cg[nt][1]))) * cu[nt][1];
                }
                if (r1 < rows) {
                    float* dst = act + (static_cast<std::size_t>(e) * cap + m0 + r1) * inter;
                    if (c0 < inter)
                        dst[c0] = (cg[nt][2] / (1.0f + __expf(-cg[nt][2]))) * cu[nt][2];
                    if (c1 < inter)
                        dst[c1] = (cg[nt][3] / (1.0f + __expf(-cg[nt][3]))) * cu[nt][3];
                }
            }
        }
    }
}

// down for every expert, scatter-added with the routing weight into out[t, :].
template <int BN, int BK, int BM>
__global__ void moe_grouped_down_bf16_kernel(
    const float* __restrict__ act, const std::uint16_t* const* __restrict__ down_w,
    const int* __restrict__ assign_token, const float* __restrict__ assign_w,
    const int* __restrict__ count, int cap, int hidden, int inter, float* __restrict__ out) {
    constexpr int NT = BN / 8;
    constexpr int WS = BK + 8;
    alignas(16) __shared__ __nv_bfloat16 sA[BM][WS];
    alignas(16) __shared__ __nv_bfloat16 sW[BN][WS];

    const int e = blockIdx.y;
    const int n = count[e];
    if (n <= 0) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * BN;  // output feature tile over `hidden`
    const std::uint16_t* wbase = down_w[e];
    const int* atok = assign_token + static_cast<std::size_t>(e) * cap;
    const float* aw = assign_w + static_cast<std::size_t>(e) * cap;
    const float* act_e = act + static_cast<std::size_t>(e) * cap * inter;
    const int g = lane >> 2;
    const int tig = lane & 3;

    for (int m0 = 0; m0 < n; m0 += BM) {
        const int rows = min(BM, n - m0);
        const bool active = (warp * 16) < rows;
        float c[NT][4] = {};
        for (int k0 = 0; k0 < inter; k0 += BK) {
            bf16_stage_act_rows<BM, BK>(sA, act_e, m0, rows, inter, k0, tid);
            stage_weight_bf16<BN, BK>(&sW[0][0], WS, wbase, n0, hidden, k0, inter, tid);
            __syncthreads();
            if (active) {
#pragma unroll
                for (int kk = 0; kk < BK; kk += 16) {
                    const __nv_bfloat16* a_ptr = &sA[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8];
                    unsigned a0, a1, a2, a3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                                 : "r"(smem_addr(a_ptr)));
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const __nv_bfloat16* b_ptr =
                            &sW[nt * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8];
                        unsigned b0, b1;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                                     : "=r"(b0), "=r"(b1)
                                     : "r"(smem_addr(b_ptr)));
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                            : "+f"(c[nt][0]), "+f"(c[nt][1]), "+f"(c[nt][2]), "+f"(c[nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    }
                }
            }
            __syncthreads();
        }
        if (active) {
            const int r0 = warp * 16 + g, r1 = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int c0 = n0 + nt * 8 + tig * 2, c1 = c0 + 1;
                if (r0 < rows) {
                    const int tok = atok[m0 + r0];
                    const float w = aw[m0 + r0];
                    if (c0 < hidden) atomicAdd(&out[static_cast<std::size_t>(tok) * hidden + c0],
                                               w * c[nt][0]);
                    if (c1 < hidden) atomicAdd(&out[static_cast<std::size_t>(tok) * hidden + c1],
                                               w * c[nt][1]);
                }
                if (r1 < rows) {
                    const int tok = atok[m0 + r1];
                    const float w = aw[m0 + r1];
                    if (c0 < hidden) atomicAdd(&out[static_cast<std::size_t>(tok) * hidden + c0],
                                               w * c[nt][2]);
                    if (c1 < hidden) atomicAdd(&out[static_cast<std::size_t>(tok) * hidden + c1],
                                               w * c[nt][3]);
                }
            }
        }
    }
}

}  // namespace

void moe_grouped_gate_up_bf16(const float* x, const std::uint16_t* const* gate_w,
                              const std::uint16_t* const* up_w, const int* assign_token,
                              const int* count, int n_experts, int cap, int hidden, int inter,
                              float* act, int bn, int bm, cudaStream_t stream) {
    if (n_experts <= 0 || inter <= 0) return;
    constexpr int BK = 64;
#define UOCR_GROUPED_GU_BF16(BN, BM)                                                       \
    moe_grouped_gate_up_bf16_kernel<BN, BK, BM>                                            \
        <<<dim3((inter + BN - 1) / BN, n_experts), dim3((BM / 16) * 32), 0, stream>>>(     \
            x, gate_w, up_w, assign_token, count, cap, hidden, inter, act)
#define UOCR_GROUPED_GU_BF16_BN(BN)     \
    do {                                \
        if (bm == 128) UOCR_GROUPED_GU_BF16(BN, 128); \
        else UOCR_GROUPED_GU_BF16(BN, 64); \
    } while (0)
    switch (bn) {
        case 8: UOCR_GROUPED_GU_BF16_BN(8); break;
        case 32: UOCR_GROUPED_GU_BF16_BN(32); break;
        case 64: UOCR_GROUPED_GU_BF16_BN(64); break;
        default: UOCR_GROUPED_GU_BF16_BN(16); break;
    }
#undef UOCR_GROUPED_GU_BF16_BN
#undef UOCR_GROUPED_GU_BF16
}

void moe_grouped_down_bf16(const float* act, const std::uint16_t* const* down_w,
                           const int* assign_token, const float* assign_w, const int* count,
                           int n_experts, int cap, int hidden, int inter, float* out, int bn,
                           int bm, cudaStream_t stream) {
    if (n_experts <= 0 || hidden <= 0) return;
    constexpr int BK = 64;
#define UOCR_GROUPED_D_BF16(BN, BM)                                                        \
    moe_grouped_down_bf16_kernel<BN, BK, BM>                                               \
        <<<dim3((hidden + BN - 1) / BN, n_experts), dim3((BM / 16) * 32), 0, stream>>>(    \
            act, down_w, assign_token, assign_w, count, cap, hidden, inter, out)
#define UOCR_GROUPED_D_BF16_BN(BN)      \
    do {                                \
        if (bm == 128) UOCR_GROUPED_D_BF16(BN, 128); \
        else UOCR_GROUPED_D_BF16(BN, 64); \
    } while (0)
    switch (bn) {
        case 8: UOCR_GROUPED_D_BF16_BN(8); break;
        case 32: UOCR_GROUPED_D_BF16_BN(32); break;
        case 64: UOCR_GROUPED_D_BF16_BN(64); break;
        default: UOCR_GROUPED_D_BF16_BN(16); break;
    }
#undef UOCR_GROUPED_D_BF16_BN
#undef UOCR_GROUPED_D_BF16
}

}  // namespace cuda
}  // namespace uocr
