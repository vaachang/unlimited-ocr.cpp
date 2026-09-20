// Backend utilities: dense matmul kernels and device introspection.
//
// `matmul_t_bf16` is the hot dense projection / lm_head path.  One cp.async
// pipelined tensor-core kernel (`matmul_t_bf16_tc_pipe_kernel`) backs it:
//
//   * `SMALL=true`  -- m <= 16 (decode, small batch): block = 16(m) x BN(n),
//     4 warps split the n dimension so all warps issue mma.
//   * `SMALL=false` -- m > 16 (prefill): block = 64(m) x BN(n), 4 warps each
//     own a 16-row slice.
//
// Both stream the weights with `cp.async` (multi-stage ring) while the current
// panel is consumed, and pad the shared rows by 8 bf16 so `ldmatrix` hits every
// bank.  `matmul_t_bf16_ref` keeps the original CUDA-core tiled kernel as the
// correctness A/B reference.

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
// column to avoid shared-memory bank conflicts.  Kept as
// `matmul_t_bf16_ref` for the tensor-core correctness A/B.
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

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(smem)), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Stage a [BN, BK] bf16 weight panel (`w` is [n, k] row-major) into a padded
// shared row `sw[BN][BK+PAD]`.  Every thread copies `BN*BK/8/128` 16-byte
// chunks with cp.async; out-of-range rows are zero-filled and a partial k tail
// is filled with regular loads so the mma never reads garbage.
template <int BN, int BK, int PAD>
__device__ __forceinline__ void stage_w_bf16(__nv_bfloat16* __restrict__ sw,
                                             const std::uint16_t* __restrict__ w, int n0, int n,
                                             int k0, int k, int tid) {
    constexpr int kThreads = 128;
    constexpr int RS = BK + PAD;
    constexpr int CPR = BK / 8;  // 16-byte chunks per row
    constexpr int NCH = BN * CPR;
#pragma unroll
    for (int i = tid; i < NCH; i += kThreads) {
        const int r = i / CPR;
        const int cg = (i % CPR) * 8;
        const int gn = n0 + r;
        const int gk = k0 + cg;
        __nv_bfloat16* dst = sw + r * RS + cg;
        if (gn >= n) {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        } else if (gk + 8 <= k) {
            cp_async16(dst, w + static_cast<std::size_t>(gn) * k + gk);
        } else {
            __nv_bfloat16 tmp[8];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int g = gk + t;
                tmp[t] = (g < k) ? __ushort_as_bfloat16(
                                       __ldg(w + static_cast<std::size_t>(gn) * k + g))
                                 : __float2bfloat16(0.0f);
            }
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(tmp);
        }
    }
}

// Stage a [BM, BK] float activation panel (`x` is [m, k] row-major) by
// converting to bf16 on the fly.  Loads are vectorised float4 and come from L2
// after the first block, so they run alongside the async weight copies.
template <int BM, int BK, int PAD>
__device__ __forceinline__ void stage_a_bf16(__nv_bfloat16* __restrict__ sa,
                                             const float* __restrict__ x, int m0, int m, int k0,
                                             int k, int tid) {
    constexpr int kThreads = 128;
    constexpr int RS = BK + PAD;
    constexpr int F4R = BK / 4;
    constexpr int NF4 = BM * F4R;
#pragma unroll
    for (int i = tid; i < NF4; i += kThreads) {
        const int r = i / F4R;
        const int c = (i % F4R) * 4;
        const int gm = m0 + r;
        const int gk = k0 + c;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (gm < m) {
            if (gk + 4 <= k) {
                v = *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(gm) * k + gk);
            } else {
                float* vp = reinterpret_cast<float*>(&v);
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    vp[t] = (gk + t < k) ? x[static_cast<std::size_t>(gm) * k + gk + t] : 0.0f;
            }
        }
        const __nv_bfloat162 p0 = __float22bfloat162_rn(make_float2(v.x, v.y));
        const __nv_bfloat162 p1 = __float22bfloat162_rn(make_float2(v.z, v.w));
        const uint32_t u0 = *reinterpret_cast<const uint32_t*>(&p0);
        const uint32_t u1 = *reinterpret_cast<const uint32_t*>(&p1);
        *reinterpret_cast<uint2*>(sa + r * RS + c) = make_uint2(u0, u1);
    }
}

// cp.async-pipelined BF16xBF16 tensor-core GEMM: y[m,n] = x[m,k] * W[n,k]^T.
//
//   BM    : rows per block (16 for `SMALL`, else 64)
//   BN    : columns per block (template parameter)
//   STAGES: number of shared-memory ring slots for the weight pipeline
//
// Shared rows are padded by 8 bf16 (`PAD`) so `ldmatrix` is bank-conflict free
// for BK=32.  The loop keeps `STAGES-1` weight groups in flight and stages the
// next activation panel with regular (L2-resident) loads before the current
// panel is consumed, so one `__syncthreads` per k-step suffices.
template <int BN, int STAGES, bool SMALL>
__global__ void matmul_t_bf16_tc_pipe_kernel(const float* __restrict__ x,
                                             const std::uint16_t* __restrict__ w,
                                             const float* __restrict__ bias, float* __restrict__ y,
                                             int m, int n, int k) {
    constexpr int BM = SMALL ? 16 : 64;
    constexpr int BK = 32;
    constexpr int PAD = 8;
    constexpr int RS = BK + PAD;
    // n8 sub-tiles each warp iterates.  SMALL splits BN across the 4 warps;
    // otherwise every warp covers the full BN from its own row slice.
    constexpr int NT = SMALL ? (BN / 32) : (BN / 8);

    alignas(16) __shared__ __nv_bfloat16 sW[STAGES][BN][RS];
    alignas(16) __shared__ __nv_bfloat16 sA[2][BM][RS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int nk = (k + BK - 1) / BK;
    const bool warp_active = SMALL || (m0 + warp * 16 < m);

    // Prologue: stage activation panel 0 (sync loads) and the first STAGES-1
    // weight panels (async, one commit group each).
    stage_a_bf16<BM, BK, PAD>(&sA[0][0][0], x, m0, m, 0, k, tid);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nk) {
            stage_w_bf16<BN, BK, PAD>(&sW[s][0][0], w, n0, n, s * BK, k, tid);
            cp_async_commit();
        }
    }

    float c[NT][4] = {};

    for (int k0i = 0; k0i < nk; ++k0i) {
        // Wait until at most STAGES-2 weight groups are still in flight, i.e.
        // weight panel `k0i` is resident.  Near the tail fewer groups were
        // issued, so drain everything.
        if (k0i + STAGES - 1 < nk)
            cp_async_wait<STAGES - 2>();
        else
            cp_async_wait<0>();
        __syncthreads();

        // Kick off the weight panel STAGES-1 steps ahead (overlaps compute).
        if (k0i + STAGES - 1 < nk) {
            stage_w_bf16<BN, BK, PAD>(&sW[(k0i + STAGES - 1) % STAGES][0][0], w, n0, n,
                                      (k0i + STAGES - 1) * BK, k, tid);
            cp_async_commit();
        }
        // Stage the next activation panel; the sync above freed its buffer.
        if (k0i + 1 < nk)
            stage_a_bf16<BM, BK, PAD>(&sA[(k0i + 1) & 1][0][0], x, m0, m, (k0i + 1) * BK, k, tid);

        if (warp_active) {
            const __nv_bfloat16* bufA = &sA[k0i & 1][0][0];
            const __nv_bfloat16* bufW = &sW[k0i % STAGES][0][0];
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                const int arow = (SMALL ? 0 : warp * 16) + (lane & 15);
                const __nv_bfloat16* a_ptr = bufA + arow * RS + kk + (lane >> 4) * 8;
                unsigned a0, a1, a2, a3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                             : "r"(smem_addr(a_ptr)));
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int nt = (SMALL ? warp * NT : 0) + t;
                    const __nv_bfloat16* b_ptr =
                        bufW + (nt * 8 + (lane & 7)) * RS + kk + ((lane >> 3) & 1) * 8;
                    unsigned b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                                 : "=r"(b0), "=r"(b1)
                                 : "r"(smem_addr(b_ptr)));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                        : "+f"(c[t][0]), "+f"(c[t][1]), "+f"(c[t][2]), "+f"(c[t][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
        }
    }

    if (!warp_active) return;
    const int r0 = m0 + (SMALL ? 0 : warp * 16) + (lane >> 2);
    const int r1 = r0 + 8;
    const int tig = lane & 3;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int nt = (SMALL ? warp * NT : 0) + t;
        const int c0 = n0 + nt * 8 + tig * 2;
        const int c1 = c0 + 1;
        if (r0 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r0) * n + c0] = c[t][0] + (bias ? bias[c0] : 0.0f);
            if (c1 < n) y[static_cast<std::size_t>(r0) * n + c1] = c[t][1] + (bias ? bias[c1] : 0.0f);
        }
        if (r1 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r1) * n + c0] = c[t][2] + (bias ? bias[c0] : 0.0f);
            if (c1 < n) y[static_cast<std::size_t>(r1) * n + c1] = c[t][3] + (bias ? bias[c1] : 0.0f);
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

void matmul_t_bf16_ref(const float* x, const std::uint16_t* w, const float* bias, float* y, int m,
                       int n, int k, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
    matmul_t_bf16_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, m, n, k);
}

void matmul_t_bf16(const float* x, const std::uint16_t* w, const float* bias, float* y, int m,
                   int n, int k, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    // The cp.async pipeline copies 16-byte chunks and float4-loads activations,
    // which needs `k` to be a multiple of 8.  Fall back for exotic shapes.
    if ((k & 7) != 0) {
        matmul_t_bf16_ref(x, w, bias, y, m, n, k, stream);
        return;
    }
    const dim3 block(128);  // 4 warps
    if (m <= 16) {
        // Small-m: 4 warps split n so every warp issues mma (lm_head decode,
        // ragged final logits).
        constexpr int kSmallBN = 64;
        constexpr int kSmallStages = 4;
        const dim3 grid((n + kSmallBN - 1) / kSmallBN, 1);
        matmul_t_bf16_tc_pipe_kernel<kSmallBN, kSmallStages, true>
            <<<grid, block, 0, stream>>>(x, w, bias, y, m, n, k);
        return;
    }
    // m > 16 (prefill): the cp.async pipeline hides the weight latency.
    constexpr int kBN = 64;
    constexpr int kStages = 3;
    const dim3 grid((n + kBN - 1) / kBN, (m + 63) / 64);
    matmul_t_bf16_tc_pipe_kernel<kBN, kStages, false>
        <<<grid, block, 0, stream>>>(x, w, bias, y, m, n, k);
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
