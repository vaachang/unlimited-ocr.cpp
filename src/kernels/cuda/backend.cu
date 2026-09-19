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

// Shared-memory tiled BF16 GEMM.  Each block computes a 64x64 output tile and
// stages 64x16 A/B panels with coalesced global loads; a 16x16 thread block
// computes a 4x4 register tile per thread.  The panels are padded by one
// column to avoid shared-memory bank conflicts.
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

// BF16xBF16 tensor-core GEMM (W4A16-style activations are not used here: both
// operands are bf16).  y[m,n] = x[m,k] * W[n,k]^T + bias[n].
//
// Block = 4 warps and computes a 64(m) x 64(n) tile.  x is converted to bf16
// while staged into shared memory; W is loaded directly (already bf16).  Each
// warp owns a 16-row slice of the m dimension and iterates the 8 n8 sub-tiles
// with one `mma.m16n8k16` per sub-tile; A is read with `ldmatrix.x4`, B with
// `ldmatrix.x2`.  A warp whose 16-row slice is entirely beyond `m` skips the
// mma loop, so small-m calls (decode, lm_head) do not pay for the full 64-row
// tile.
constexpr int kTCM = 64;  // rows per block
constexpr int kTCN = 64;  // columns per block
constexpr int kTCK = 16;  // k per step

__global__ void matmul_t_bf16_tc_kernel(const float* __restrict__ x,
                                        const std::uint16_t* __restrict__ w,
                                        const float* __restrict__ bias, float* __restrict__ y,
                                        int m, int n, int k) {
    __shared__ __nv_bfloat16 sA[kTCM][kTCK];
    __shared__ __nv_bfloat16 sW[kTCN][kTCK];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int m0 = blockIdx.y * kTCM;
    const int n0 = blockIdx.x * kTCN;
    const bool active = (m0 + warp * 16) < m;

    float c[8][4] = {};

    for (int k0 = 0; k0 < k; k0 += kTCK) {
        // ---- stage activations [64,16] as bf16 (coalesced) ----
#pragma unroll
        for (int l = 0; l < (kTCM * kTCK) / 128; ++l) {
            const int idx = tid + l * 128;
            const int r = idx / kTCK, cc = idx % kTCK;
            const int gm = m0 + r, gk = k0 + cc;
            sA[r][cc] = (gm < m && gk < k)
                            ? __float2bfloat16(x[static_cast<std::size_t>(gm) * k + gk])
                            : __float2bfloat16(0.0f);
        }
        // ---- stage weights [64,16] (already bf16) ----
#pragma unroll
        for (int l = 0; l < (kTCN * kTCK) / 128; ++l) {
            const int idx = tid + l * 128;
            const int r = idx / kTCK, cc = idx % kTCK;
            const int gn = n0 + r, gk = k0 + cc;
            sW[r][cc] = (gn < n && gk < k)
                            ? __ushort_as_bfloat16(w[static_cast<std::size_t>(gn) * k + gk])
                            : __float2bfloat16(0.0f);
        }
        __syncthreads();

        if (active) {
            const __nv_bfloat16* a_ptr = &sA[warp * 16 + (lane & 15)][(lane >> 4) * 8];
            unsigned a0, a1, a2, a3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                         : "r"(smem_addr(a_ptr)));
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const __nv_bfloat16* b_ptr = &sW[nt * 8 + (lane & 7)][((lane >> 3) & 1) * 8];
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
        __syncthreads();
    }

    if (!active) return;
    const int r0 = m0 + warp * 16 + (lane >> 2);
    const int r1 = r0 + 8;
    const int tig = lane & 3;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int c0 = n0 + nt * 8 + tig * 2;
        const int c1 = c0 + 1;
        if (r0 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r0) * n + c0] = c[nt][0] + (bias ? bias[c0] : 0.0f);
            if (c1 < n) y[static_cast<std::size_t>(r0) * n + c1] = c[nt][1] + (bias ? bias[c1] : 0.0f);
        }
        if (r1 < m) {
            if (c0 < n) y[static_cast<std::size_t>(r1) * n + c0] = c[nt][2] + (bias ? bias[c0] : 0.0f);
            if (c1 < n) y[static_cast<std::size_t>(r1) * n + c1] = c[nt][3] + (bias ? bias[c1] : 0.0f);
        }
    }
}

// Small-m tensor-core GEMM: BM=16 rows, BN=64 columns, with the 4 warps
// splitting the n dimension (16 columns each).  Used when m <= 16 so the mma
// work is spread over all warps instead of leaving three of them idle (the
// 64-row kernel above only activates one warp for m=16).  This is the hot path
// for the lm_head projection during decode.
constexpr int kTSM = 16;  // rows per block
constexpr int kTSN = 64;  // columns per block
constexpr int kTSK = 16;  // k per step

__global__ void matmul_t_bf16_tc_small_kernel(const float* __restrict__ x,
                                              const std::uint16_t* __restrict__ w,
                                              const float* __restrict__ bias, float* __restrict__ y,
                                              int m, int n, int k) {
    __shared__ __nv_bfloat16 sA[kTSM][kTSK];
    __shared__ __nv_bfloat16 sW[kTSN][kTSK];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int m0 = blockIdx.y * kTSM;
    const int n0 = blockIdx.x * kTSN;
    // Each warp owns two adjacent n8 sub-tiles: columns [warp*16, warp*16+16).
    const int nt_base = warp * 2;

    float c[2][4] = {};

    for (int k0 = 0; k0 < k; k0 += kTSK) {
#pragma unroll
        for (int l = 0; l < (kTSM * kTSK) / 128; ++l) {
            const int idx = tid + l * 128;
            const int r = idx / kTSK, cc = idx % kTSK;
            const int gm = m0 + r, gk = k0 + cc;
            sA[r][cc] = (gm < m && gk < k)
                            ? __float2bfloat16(x[static_cast<std::size_t>(gm) * k + gk])
                            : __float2bfloat16(0.0f);
        }
#pragma unroll
        for (int l = 0; l < (kTSN * kTSK) / 128; ++l) {
            const int idx = tid + l * 128;
            const int r = idx / kTSK, cc = idx % kTSK;
            const int gn = n0 + r, gk = k0 + cc;
            sW[r][cc] = (gn < n && gk < k)
                            ? __ushort_as_bfloat16(w[static_cast<std::size_t>(gn) * k + gk])
                            : __float2bfloat16(0.0f);
        }
        __syncthreads();

        const __nv_bfloat16* a_ptr = &sA[lane & 15][(lane >> 4) * 8];
        unsigned a0, a1, a2, a3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "r"(smem_addr(a_ptr)));
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const __nv_bfloat16* b_ptr = &sW[(nt_base + t) * 8 + (lane & 7)][((lane >> 3) & 1) * 8];
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
        __syncthreads();
    }

    const int r0 = m0 + (lane >> 2);
    const int r1 = r0 + 8;
    const int tig = lane & 3;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int c0 = n0 + (nt_base + t) * 8 + tig * 2;
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
    const dim3 block(128);  // 4 warps
    if (m <= kTSM) {
        // Spread the mma work over all four warps along n (lm_head decode).
        const dim3 grid((n + kTSN - 1) / kTSN, (m + kTSM - 1) / kTSM);
        matmul_t_bf16_tc_small_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, m, n, k);
        return;
    }
    const dim3 grid((n + kTCN - 1) / kTCN, (m + kTCM - 1) / kTCM);
    matmul_t_bf16_tc_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, m, n, k);
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
