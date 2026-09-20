// MoE INT4 GEMM kernel.
//
// y[m,n] = x[m,k] * dequant(W[n,k])^T
//
// W is stored [n,k] packed 2 int4 values per byte (low nibble = even column)
// with per-output-row, per-`group_size` affine (scale, zero).  This reference
// kernel dequantizes into registers and accumulates in FP32.  The tensor-core
// path (mma.sync.aligned.m16n8k32.s4.s4.s32) can be dropped in behind the same
// signature; the sm_120 build keeps this portable version as the correctness
// baseline.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

__global__ void moe_gemm_int4_kernel(const float* __restrict__ x,
                                     const std::uint8_t* __restrict__ packed,
                                     const float* __restrict__ scales,
                                     const float* __restrict__ zeros, int m, int n, int k,
                                     int group_size, float* __restrict__ y) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // output row (token)
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
    if (row >= m || col >= n) return;

    const int packed_row = (k + 1) / 2;
    const int ng = (k + group_size - 1) / group_size;
    const std::uint8_t* wrow = packed + static_cast<std::size_t>(col) * packed_row;
    const float* srow = scales + static_cast<std::size_t>(col) * ng;
    const float* zrow = zeros + static_cast<std::size_t>(col) * ng;
    const float* xrow = x + static_cast<std::size_t>(row) * k;

    float acc = 0.0f;
    for (int c = 0; c < k; ++c) {
        const std::uint8_t byte = __ldg(wrow + c / 2);
        const int q = (c & 1) ? (byte >> 4) : (byte & 0x0f);
        const int g = c / group_size;
        const float wv = (static_cast<float>(q) - zrow[g]) * srow[g];
        acc += xrow[c] * wv;
    }
    y[static_cast<std::size_t>(row) * n + col] = acc;
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// W4A16 tensor-core kernel with shared-memory staging and `ldmatrix`.
//
// Block = 4 warps and computes a 64(m) x BN(n) tile.  Each k-step dequantizes
// the INT4 weight panel sW[BN][16] once into shared memory; every warp then
// handles a 16-row slice of the m dimension and iterates the BN/8 n8
// sub-tiles.  A and B fragments are pulled with `ldmatrix.x4`/`x2` and the mma
// is `mma.m16n8k16.bf16.bf16.f32`.
//
// BN is a template parameter: wider tiles do more mma per staged panel but
// reduce the block count, which matters for the small-N expert GEMMs
// (N=896/1280, M~96) where too few blocks leaves SMs idle.  BN=8 is the
// default; see `moe_gemm_int4_tc_n` for the tuning sweep.
//
// Warps whose 16-row slice is fully beyond `m` skip the mma loop but still
// participate in `__syncthreads`, so small-M calls (decode) do not pay for the
// full 64-row tile.
//
// Fragment layouts follow PTX ISA: A is 4x[2xb16] (a0..a3), B is 2x[2xb16]
// (b0,b1).
template <int BN, int BK = 16>
__global__ void moe_gemm_int4_tc_kernel(const float* __restrict__ x,
                                        const std::uint8_t* __restrict__ packed,
                                        const float* __restrict__ scales,
                                        const float* __restrict__ zeros, int m, int n, int k,
                                        int group_size, float* __restrict__ y) {
    constexpr int BM = 64;      // rows per block (4 warps x 16)
    constexpr int NT = BN / 8;  // n8 sub-tiles per warp
    constexpr int PAD = 8;      // bank-conflict-free ldmatrix (see backend.cu)
    alignas(16) __shared__ __nv_bfloat16 sA[BM][BK + PAD];
    alignas(16) __shared__ __nv_bfloat16 sW[BN][BK + PAD];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int packed_row = (k + 1) / 2;
    const int ng = (k + group_size - 1) / group_size;
    const bool active = (m0 + warp * 16) < m;

    const int g = lane >> 2;
    const int tig = lane & 3;
    float c[NT][4] = {};

    for (int k0 = 0; k0 < k; k0 += BK) {
        // ---- stage activations [BM, BK] as bf16 (float4 loads, c fastest) ----
        {
            constexpr int F4R = BK / 4;
            constexpr int NF4 = BM * F4R;
#pragma unroll
            for (int l = 0; l < NF4 / 128; ++l) {
                const int idx = tid + l * 128;
                const int r = idx / F4R, c4 = (idx % F4R) * 4;
                const int gm = m0 + r, gk = k0 + c4;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (gm < m) {
                    if (gk + 4 <= k) {
                        v = *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(gm) * k +
                                                             gk);
                    } else {
                        float* vp = reinterpret_cast<float*>(&v);
#pragma unroll
                        for (int t = 0; t < 4; ++t)
                            vp[t] = (gk + t < k)
                                        ? x[static_cast<std::size_t>(gm) * k + gk + t]
                                        : 0.0f;
                    }
                }
                const __nv_bfloat162 p0 = __float22bfloat162_rn(make_float2(v.x, v.y));
                const __nv_bfloat162 p1 = __float22bfloat162_rn(make_float2(v.z, v.w));
                const unsigned u0 = *reinterpret_cast<const unsigned*>(&p0);
                const unsigned u1 = *reinterpret_cast<const unsigned*>(&p1);
                *reinterpret_cast<uint2*>(&sA[r][c4]) = make_uint2(u0, u1);
            }
        }
        // ---- dequantize the INT4 weight panel into sW[n][k] ----
#pragma unroll
        for (int l = 0; l < (BN * BK) / 128; ++l) {
            const int idx = tid + l * 128;
            const int r = idx / BK, cc = idx % BK;
            const int gn = n0 + r, kk = k0 + cc;
            float wv = 0.0f;
            if (gn < n && kk < k) {
                const std::uint8_t byte =
                    packed[static_cast<std::size_t>(gn) * packed_row + (kk >> 1)];
                const int q = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
                const int gg = kk / group_size;
                if (gg < ng) {
                    const float s = scales[static_cast<std::size_t>(gn) * ng + gg];
                    const float z = zeros[static_cast<std::size_t>(gn) * ng + gg];
                    wv = (static_cast<float>(q) - z) * s;
                }
            }
            sW[r][cc] = __float2bfloat16(wv);
        }
        __syncthreads();

        if (active) {
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                // ---- ldmatrix fragments ----
                const __nv_bfloat16* a_ptr =
                    &sA[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8];
                unsigned a0, a1, a2, a3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                             : "r"(smem_addr(a_ptr)));
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    // x2 only consumes addresses from lanes 0-15; mirror them for
                    // the rest so no lane forms an out-of-bounds pointer.
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

    if (!active) return;
    const int r0 = m0 + warp * 16 + g, r1 = r0 + 8;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int c0 = n0 + nt * 8 + tig * 2, c1 = c0 + 1;
        if (r0 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r0) * n + c0] = c[nt][0];
            if (c1 < n) y[static_cast<std::size_t>(r0) * n + c1] = c[nt][1];
        }
        if (r1 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r1) * n + c0] = c[nt][2];
            if (c1 < n) y[static_cast<std::size_t>(r1) * n + c1] = c[nt][3];
        }
    }
}

// ---------------------------------------------------------------------------
// cp.async-pipelined W4A16 kernel.
//
// The synchronous kernel above re-stages both the activation panel and the
// dequantized weight panel every k-step with plain global loads, so each
// `__syncthreads` exposes the full global latency.  This variant:
//   * prefetches the *packed* INT4 weight panel with `cp.async` into a
//     multi-stage ring and dequantizes it from shared memory;
//   * keeps the activation fragments in registers, software-prefetched one
//     k-step ahead, so activations never round-trip through shared memory.
// Only one `__syncthreads` per k-step is needed on the dequantized panel.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void i4_cp_async4(void* smem, const void* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(smem)), "l"(gmem));
}
__device__ __forceinline__ void i4_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void i4_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Async-copy the packed weight panel [BN, BK] (BK/2 bytes per row) into `raw`.
// Full rows use 4-byte cp.async chunks; the k tail / out-of-range rows are
// filled with regular loads/stores (visible after the following sync).
template <int BN, int BK>
__device__ __forceinline__ void int4_stage_raw(std::uint8_t* __restrict__ raw,
                                               const std::uint8_t* __restrict__ packed, int n0,
                                               int n, int k0, int k, int packed_row, bool cpok,
                                               int tid) {
    constexpr int kThreads = 128;
    constexpr int PROW = BK / 2;
    constexpr int CH = 4;
    constexpr int CPR = PROW / CH;
    constexpr int NCH = BN * CPR;
#pragma unroll
    for (int i = tid; i < NCH; i += kThreads) {
        const int r = i / CPR;
        const int c = (i % CPR) * CH;
        const int gn = n0 + r;
        std::uint8_t* dst = raw + r * PROW + c;
        const int base = k0 / 2 + c;
        if (gn < n && cpok && k0 + BK <= k) {
            i4_cp_async4(dst, packed + static_cast<std::size_t>(gn) * packed_row + base);
        } else if (gn < n) {
#pragma unroll
            for (int t = 0; t < CH; ++t)
                dst[t] = (2 * (base + t) < k)
                             ? __ldg(packed + static_cast<std::size_t>(gn) * packed_row + base + t)
                             : 0;
        } else {
#pragma unroll
            for (int t = 0; t < CH; ++t) dst[t] = 0;
        }
    }
}

// Dequantize the raw packed panel into a bf16 panel [BN, BK] (row stride
// `wstride`).  All 128 threads participate (including inactive warps).
template <int BN, int BK>
__device__ __forceinline__ void int4_dequant_panel(__nv_bfloat16* __restrict__ sw, int wstride,
                                                   const std::uint8_t* __restrict__ raw,
                                                   const float* __restrict__ scales,
                                                   const float* __restrict__ zeros, int n0, int n,
                                                   int k0, int k, int group_size, int ng, int tid) {
    constexpr int kThreads = 128;
    constexpr int PROW = BK / 2;
#pragma unroll
    for (int i = tid; i < BN * BK; i += kThreads) {
        const int r = i / BK;
        const int cc = i % BK;
        const int gn = n0 + r;
        float wv = 0.0f;
        if (gn < n && k0 + cc < k) {
            const std::uint8_t byte = raw[r * PROW + (cc >> 1)];
            const int q = (cc & 1) ? (byte >> 4) : (byte & 0x0f);
            const int gg = (k0 + cc) / group_size;
            if (gg < ng) {
                const float s = scales[static_cast<std::size_t>(gn) * ng + gg];
                const float z = zeros[static_cast<std::size_t>(gn) * ng + gg];
                wv = (static_cast<float>(q) - z) * s;
            }
        }
        sw[r * wstride + cc] = __float2bfloat16(wv);
    }
}

// Build the two m16n8k16 A fragments (16 rows x BK=32) for this warp directly
// from global activations.  `row_base` is the warp's first row.
template <int BK>
__device__ __forceinline__ void int4_load_a_frag(const float* __restrict__ x, int row_base,
                                                  int lane, int m, int k0, int k,
                                                  unsigned a[2][4]) {
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int r0 = row_base + gid;
    const int r1 = r0 + 8;
    const bool ok0 = r0 < m, ok1 = r1 < m;
    const float* x0 = ok0 ? x + static_cast<std::size_t>(r0) * k : nullptr;
    const float* x1 = ok1 ? x + static_cast<std::size_t>(r1) * k : nullptr;
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
        const int c0 = k0 + kh * 16 + tig * 2;
        const int c1 = c0 + 8;
        float v[8];
        const bool v0 = c0 + 1 < k, v1 = c1 + 1 < k;
        v[0] = (x0 && v0) ? x0[c0] : 0.0f;
        v[1] = (x0 && v0) ? x0[c0 + 1] : 0.0f;
        v[2] = (x1 && v0) ? x1[c0] : 0.0f;
        v[3] = (x1 && v0) ? x1[c0 + 1] : 0.0f;
        v[4] = (x0 && v1) ? x0[c1] : 0.0f;
        v[5] = (x0 && v1) ? x0[c1 + 1] : 0.0f;
        v[6] = (x1 && v1) ? x1[c1] : 0.0f;
        v[7] = (x1 && v1) ? x1[c1 + 1] : 0.0f;
        const __nv_bfloat162 b0 = __float22bfloat162_rn(make_float2(v[0], v[1]));
        const __nv_bfloat162 b1 = __float22bfloat162_rn(make_float2(v[2], v[3]));
        const __nv_bfloat162 b2 = __float22bfloat162_rn(make_float2(v[4], v[5]));
        const __nv_bfloat162 b3 = __float22bfloat162_rn(make_float2(v[6], v[7]));
        a[kh][0] = *reinterpret_cast<const unsigned*>(&b0);
        a[kh][1] = *reinterpret_cast<const unsigned*>(&b1);
        a[kh][2] = *reinterpret_cast<const unsigned*>(&b2);
        a[kh][3] = *reinterpret_cast<const unsigned*>(&b3);
    }
}

template <int BN, int STAGES>
__global__ void moe_gemm_int4_tc_pipe_kernel(const float* __restrict__ x,
                                             const std::uint8_t* __restrict__ packed,
                                             const float* __restrict__ scales,
                                             const float* __restrict__ zeros, int m, int n, int k,
                                             int group_size, float* __restrict__ y) {
    constexpr int BM = 64;  // 4 warps x 16 rows
    constexpr int BK = 32;
    constexpr int NT = BN / 8;
    constexpr int WSTRIDE = BK + 8;  // bank-conflict-free ldmatrix
    constexpr int PROW = BK / 2;
    constexpr int LEAD = STAGES - 1;

    alignas(16) __shared__ std::uint8_t sRaw[STAGES][BN][PROW];
    alignas(16) __shared__ __nv_bfloat16 sW[2][BN][WSTRIDE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int nk = (k + BK - 1) / BK;
    const int packed_row = (k + 1) / 2;
    const int ng = (k + group_size - 1) / group_size;
    const bool active = (m0 + warp * 16) < m;
    const bool cpok = (packed_row % 4 == 0);

    float c[NT][4] = {};
    unsigned a_cur[2][4] = {};
    if (active) int4_load_a_frag<BK>(x, m0 + warp * 16, lane, m, 0, k, a_cur);

    // Prologue: prefetch the first LEAD packed panels.
#pragma unroll
    for (int s = 0; s < LEAD; ++s) {
        if (s < nk) {
            int4_stage_raw<BN, BK>(&sRaw[s][0][0], packed, n0, n, s * BK, k, packed_row, cpok, tid);
            i4_cp_commit();
        }
    }

    for (int k0i = 0; k0i < nk; ++k0i) {
        if (k0i + LEAD < nk)
            i4_cp_wait<LEAD - 1>();
        else
            i4_cp_wait<0>();
        __syncthreads();

        int4_dequant_panel<BN, BK>(&sW[k0i & 1][0][0], WSTRIDE, &sRaw[k0i % STAGES][0][0],
                                   scales, zeros, n0, n, k0i * BK, k, group_size, ng, tid);

        unsigned a_next[2][4];
        if (active && k0i + 1 < nk)
            int4_load_a_frag<BK>(x, m0 + warp * 16, lane, m, (k0i + 1) * BK, k, a_next);

        if (k0i + LEAD < nk) {
            int4_stage_raw<BN, BK>(&sRaw[(k0i + LEAD) % STAGES][0][0], packed, n0, n,
                                   (k0i + LEAD) * BK, k, packed_row, cpok, tid);
            i4_cp_commit();
        }
        __syncthreads();

        if (active) {
            const __nv_bfloat16* bufW = &sW[k0i & 1][0][0];
#pragma unroll
            for (int kh = 0; kh < 2; ++kh) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const __nv_bfloat16* b_ptr = bufW + (nt * 8 + (lane & 7)) * WSTRIDE +
                                                 kh * 16 + ((lane >> 3) & 1) * 8;
                    unsigned b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                                 : "=r"(b0), "=r"(b1)
                                 : "r"(smem_addr(b_ptr)));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(c[nt][0]), "+f"(c[nt][1]), "+f"(c[nt][2]), "+f"(c[nt][3])
                        : "r"(a_cur[kh][0]), "r"(a_cur[kh][1]), "r"(a_cur[kh][2]),
                          "r"(a_cur[kh][3]), "r"(b0), "r"(b1));
                }
            }
        }
        if (active && k0i + 1 < nk) {
#pragma unroll
            for (int kh = 0; kh < 2; ++kh)
#pragma unroll
                for (int j = 0; j < 4; ++j) a_cur[kh][j] = a_next[kh][j];
        }
    }

    if (!active) return;
    const int r0 = m0 + warp * 16 + (lane >> 2);
    const int r1 = r0 + 8;
    const int tig = lane & 3;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int c0 = n0 + nt * 8 + tig * 2;
        const int c1 = c0 + 1;
        if (r0 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r0) * n + c0] = c[nt][0];
            if (c1 < n) y[static_cast<std::size_t>(r0) * n + c1] = c[nt][1];
        }
        if (r1 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r1) * n + c0] = c[nt][2];
            if (c1 < n) y[static_cast<std::size_t>(r1) * n + c1] = c[nt][3];
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped expert GEMM driven by the device grouping table.
//
// The per-expert loop in `GpuDecoder::mlp_block` issues 3 x n_experts tiny
// launches per layer plus a router D2H/H2D sync.  These two kernels replace it
// with one launch per layer for gate+up (+fused SiLU) and one for down:
//
//   grid = (ceil(inter/BN) or ceil(hidden/BN), n_experts)
//
// Each block owns one (expert, n-tile) pair and loops over that expert's
// assigned tokens (`count[e]`, `assign_token[e, :]`, both device-resident), so
// the grid depends only on static shape.  `act` is staged in the same
// [n_experts, cap, inter] layout the masked kernel uses, so down can read it
// back slot-by-slot.
// ---------------------------------------------------------------------------

// Dequantize one [BN, BK] INT4 weight panel of expert `e` (base already
// advanced) into bf16 shared memory with row stride `stride`.
template <int BN, int BK>
__device__ __forceinline__ void int4_dequant_panel_expert(
    __nv_bfloat16* __restrict__ sw, int stride, const std::uint8_t* __restrict__ base,
    const float* __restrict__ scales, const float* __restrict__ zeros, int n0, int n, int k0,
    int k, int group, int ng, int tid, int packed_row) {
    for (int idx = tid; idx < BN * BK; idx += blockDim.x) {
        const int r = idx / BK, cc = idx % BK;
        const int gn = n0 + r, kk = k0 + cc;
        float wv = 0.0f;
        if (gn < n && kk < k) {
            const std::uint8_t byte = base[static_cast<std::size_t>(gn) * packed_row + (kk >> 1)];
            const int q = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
            const int gg = kk / group;
            if (gg < ng) {
                const float s = scales[static_cast<std::size_t>(gn) * ng + gg];
                const float z = zeros[static_cast<std::size_t>(gn) * ng + gg];
                wv = (static_cast<float>(q) - z) * s;
            }
        }
        sw[r * stride + cc] = __float2bfloat16(wv);
    }
}

// Stage a [BM, BK] activation tile by gathering rows through `atok` (the
// expert's assignment list).  Rows >= `rows` become zero.
template <int BM, int BK>
__device__ __forceinline__ void stage_act_gather(__nv_bfloat16 (*sA)[BK + 8],
                                                 const float* __restrict__ x,
                                                 const int* __restrict__ atok, int m0, int rows,
                                                 int hidden, int k0, int tid) {
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
                for (int t = 0; t < 4; ++t)
                    vp[t] = (gk + t < hidden) ? xr[gk + t] : 0.0f;
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
__device__ __forceinline__ void stage_act_rows(__nv_bfloat16 (*sA)[BK + 8],
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
                for (int t = 0; t < 4; ++t)
                    vp[t] = (gk + t < kdim) ? xr[gk + t] : 0.0f;
            }
        }
        const __nv_bfloat162 p0 = __float22bfloat162_rn(make_float2(v.x, v.y));
        const __nv_bfloat162 p1 = __float22bfloat162_rn(make_float2(v.z, v.w));
        const unsigned u0 = *reinterpret_cast<const unsigned*>(&p0);
        const unsigned u1 = *reinterpret_cast<const unsigned*>(&p1);
        *reinterpret_cast<uint2*>(&sA[r][c4]) = make_uint2(u0, u1);
    }
}

// gate + up for every expert, with fused SiLU.
template <int BN, int BK, int BM>
__global__ void moe_grouped_gate_up_int4_kernel(
    const float* __restrict__ x, const std::uint8_t* __restrict__ gate_packed,
    const float* __restrict__ gate_scales, const float* __restrict__ gate_zeros,
    int gate_pstride, int gate_sstride, int gate_ng, const std::uint8_t* __restrict__ up_packed,
    const float* __restrict__ up_scales, const float* __restrict__ up_zeros, int up_pstride,
    int up_sstride, int up_ng, const int* __restrict__ assign_token,
    const int* __restrict__ count, int cap, int hidden, int inter, int group,
    float* __restrict__ act) {
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
    const int packed_row = (hidden + 1) / 2;
    const int ng = (hidden + group - 1) / group;
    const std::uint8_t* gb = gate_packed + static_cast<std::size_t>(e) * gate_pstride;
    const std::uint8_t* ub = up_packed + static_cast<std::size_t>(e) * up_pstride;
    const float* gsc = gate_scales + static_cast<std::size_t>(e) * gate_sstride;
    const float* gzr = gate_zeros + static_cast<std::size_t>(e) * gate_sstride;
    const float* usc = up_scales + static_cast<std::size_t>(e) * up_sstride;
    const float* uzr = up_zeros + static_cast<std::size_t>(e) * up_sstride;
    const int* atok = assign_token + static_cast<std::size_t>(e) * cap;
    const int g = lane >> 2;
    const int tig = lane & 3;

    for (int m0 = 0; m0 < n; m0 += BM) {
        const int rows = min(BM, n - m0);
        const bool active = (warp * 16) < rows;
        float cg[NT][4] = {};
        float cu[NT][4] = {};
        for (int k0 = 0; k0 < hidden; k0 += BK) {
            stage_act_gather<BM, BK>(sA, x, atok, m0, rows, hidden, k0, tid);
            int4_dequant_panel_expert<BN, BK>(&sWg[0][0], WS, gb, gsc, gzr, n0, inter, k0, hidden,
                                              group, ng, tid, packed_row);
            int4_dequant_panel_expert<BN, BK>(&sWu[0][0], WS, ub, usc, uzr, n0, inter, k0, hidden,
                                              group, ng, tid, packed_row);
            __syncthreads();
            if (active) {
#pragma unroll
                for (int kk = 0; kk < BK; kk += 16) {
                    const __nv_bfloat16* a_ptr =
                        &sA[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8];
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

// down for every expert, scatter-added with the routing weight.
template <int BN, int BK, int BM>
__global__ void moe_grouped_down_int4_kernel(
    const float* __restrict__ act, const std::uint8_t* __restrict__ packed,
    const float* __restrict__ scales, const float* __restrict__ zeros, int pstride, int sstride,
    int ng, const int* __restrict__ assign_token, const float* __restrict__ assign_w,
    const int* __restrict__ count, int cap, int hidden, int inter, int group,
    float* __restrict__ out) {
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
    const int packed_row = (inter + 1) / 2;
    const std::uint8_t* wb = packed + static_cast<std::size_t>(e) * pstride;
    const float* wsc = scales + static_cast<std::size_t>(e) * sstride;
    const float* wzr = zeros + static_cast<std::size_t>(e) * sstride;
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
            stage_act_rows<BM, BK>(sA, act_e, m0, rows, inter, k0, tid);
            int4_dequant_panel_expert<BN, BK>(&sW[0][0], WS, wb, wsc, wzr, n0, hidden, k0, inter,
                                              group, ng, tid, packed_row);
            __syncthreads();
            if (active) {
#pragma unroll
                for (int kk = 0; kk < BK; kk += 16) {
                    const __nv_bfloat16* a_ptr =
                        &sA[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8];
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

void moe_grouped_gate_up_int4(
    const float* x, const std::uint8_t* gate_packed, const float* gate_scales,
    const float* gate_zeros, int gate_pstride, int gate_sstride, int gate_ng,
    const std::uint8_t* up_packed, const float* up_scales, const float* up_zeros,
    int up_pstride, int up_sstride, int up_ng, const int* assign_token,
    const int* count, int n_experts, int cap, int hidden, int inter, int group, float* act,
    int bn, int bm, cudaStream_t stream) {
    if (n_experts <= 0 || inter <= 0) return;
    constexpr int BK = 64;
#define UOCR_GROUPED_GU(BN, BM)                                                        \
    moe_grouped_gate_up_int4_kernel<BN, BK, BM>                                        \
        <<<dim3((inter + BN - 1) / BN, n_experts), dim3((BM / 16) * 32), 0, stream>>>( \
            x, gate_packed, gate_scales, gate_zeros, gate_pstride, gate_sstride,        \
            gate_ng, up_packed, up_scales, up_zeros, up_pstride, up_sstride, up_ng,    \
            assign_token, count, cap, hidden, inter, group, act)
#define UOCR_GROUPED_GU_BN(BN)        \
    do {                              \
        if (bm == 128) UOCR_GROUPED_GU(BN, 128); \
        else UOCR_GROUPED_GU(BN, 64); \
    } while (0)
    switch (bn) {
        case 8: UOCR_GROUPED_GU_BN(8); break;
        case 32: UOCR_GROUPED_GU_BN(32); break;
        case 64: UOCR_GROUPED_GU_BN(64); break;
        default: UOCR_GROUPED_GU_BN(16); break;
    }
#undef UOCR_GROUPED_GU_BN
#undef UOCR_GROUPED_GU
}

void moe_grouped_down_int4(
    const float* act, const std::uint8_t* packed, const float* scales, const float* zeros,
    int pstride, int sstride, int ng, const int* assign_token, const float* assign_w,
    const int* count, int n_experts, int cap, int hidden, int inter, int group, float* out,
    int bn, int bm, cudaStream_t stream) {
    if (n_experts <= 0 || hidden <= 0) return;
    constexpr int BK = 64;
#define UOCR_GROUPED_D(BN, BM)                                                              \
    moe_grouped_down_int4_kernel<BN, BK, BM>                                                \
        <<<dim3((hidden + BN - 1) / BN, n_experts), dim3((BM / 16) * 32), 0, stream>>>(    \
            act, packed, scales, zeros, pstride, sstride, ng, assign_token, assign_w,      \
            count, cap, hidden, inter, group, out)
#define UOCR_GROUPED_D_BN(BN)         \
    do {                              \
        if (bm == 128) UOCR_GROUPED_D(BN, 128); \
        else UOCR_GROUPED_D(BN, 64);  \
    } while (0)
    switch (bn) {
        case 8: UOCR_GROUPED_D_BN(8); break;
        case 32: UOCR_GROUPED_D_BN(32); break;
        case 64: UOCR_GROUPED_D_BN(64); break;
        default: UOCR_GROUPED_D_BN(16); break;
    }
#undef UOCR_GROUPED_D_BN
#undef UOCR_GROUPED_D
}

void moe_gemm_int4(const float* x, const std::uint8_t* packed, const float* scales,
                   const float* zeros, int m, int n, int k, int group_size, float* y,
                   cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    moe_gemm_int4_kernel<<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size,
                                                     y);
}

void moe_gemm_int4_tc_nk(const float* x, const std::uint8_t* packed, const float* scales,
                         const float* zeros, int m, int n, int k, int group_size, float* y,
                         int bn, int bk, cudaStream_t stream) {
    const dim3 block(128);  // 4 warps -> 64 rows per block
    const dim3 grid((n + bn - 1) / bn, (m + 63) / 64);
    switch (bk) {
        case 32:
            switch (bn) {
                case 16: moe_gemm_int4_tc_kernel<16, 32><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                case 32: moe_gemm_int4_tc_kernel<32, 32><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                case 64: moe_gemm_int4_tc_kernel<64, 32><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                default: moe_gemm_int4_tc_kernel<8, 32><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
            }
            break;
        case 64:
            switch (bn) {
                case 16: moe_gemm_int4_tc_kernel<16, 64><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                case 32: moe_gemm_int4_tc_kernel<32, 64><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                case 64: moe_gemm_int4_tc_kernel<64, 64><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                default: moe_gemm_int4_tc_kernel<8, 64><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
            }
            break;
        default:
            switch (bn) {
                case 16: moe_gemm_int4_tc_kernel<16, 16><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                case 32: moe_gemm_int4_tc_kernel<32, 16><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                case 64: moe_gemm_int4_tc_kernel<64, 16><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
                default: moe_gemm_int4_tc_kernel<8, 16><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
            }
            break;
    }
}

void moe_gemm_int4_tc_n(const float* x, const std::uint8_t* packed, const float* scales,
                        const float* zeros, int m, int n, int k, int group_size, float* y,
                        int bn, cudaStream_t stream) {
    moe_gemm_int4_tc_nk(x, packed, scales, zeros, m, n, k, group_size, y, bn, 16, stream);
}

void moe_gemm_int4_tc_pipe(const float* x, const std::uint8_t* packed, const float* scales,
                           const float* zeros, int m, int n, int k, int group_size, float* y,
                           int bn, cudaStream_t stream) {
    constexpr int kStages = 3;
    const dim3 block(128);
    const dim3 grid((n + bn - 1) / bn, (m + 63) / 64);
    switch (bn) {
        case 8: moe_gemm_int4_tc_pipe_kernel<8, kStages><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
        case 16: moe_gemm_int4_tc_pipe_kernel<16, kStages><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
        case 32: moe_gemm_int4_tc_pipe_kernel<32, kStages><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
        default: moe_gemm_int4_tc_pipe_kernel<64, kStages><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size, y); break;
    }
}

void moe_gemm_int4_tc(const float* x, const std::uint8_t* packed, const float* scales,
                      const float* zeros, int m, int n, int k, int group_size, float* y,
                      cudaStream_t stream) {
    // The vectorised activation staging needs k % 4 == 0; otherwise use the
    // scalar reference (same numerics up to bf16 rounding).
    if ((k & 3) != 0) {
        moe_gemm_int4(x, packed, scales, zeros, m, n, k, group_size, y, stream);
        return;
    }
    // bn=8 keeps the block count high (latency hiding) and bk=64 amortises the
    // per-k-step staging/sync overhead; see `BENCHMARKS.md` 2.5.
    moe_gemm_int4_tc_nk(x, packed, scales, zeros, m, n, k, group_size, y, 8, 64, stream);
}

}  // namespace cuda
}  // namespace uocr
