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
template <int BN>
__global__ void moe_gemm_int4_tc_kernel(const float* __restrict__ x,
                                        const std::uint8_t* __restrict__ packed,
                                        const float* __restrict__ scales,
                                        const float* __restrict__ zeros, int m, int n, int k,
                                        int group_size, float* __restrict__ y) {
    constexpr int BM = 64;      // rows per block (4 warps x 16)
    constexpr int BK = 16;
    constexpr int NT = BN / 8;  // n8 sub-tiles per warp
    __shared__ __nv_bfloat16 sA[BM][BK];
    __shared__ __nv_bfloat16 sW[BN][BK];

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
        // ---- stage activations [BM, BK] as bf16 (coalesced; c fastest) ----
#pragma unroll
        for (int l = 0; l < (BM * BK) / 128; ++l) {
            const int idx = tid + l * 128;
            const int r = idx / BK, cc = idx % BK;
            const int gm = m0 + r, gk = k0 + cc;
            sA[r][cc] = (gm < m && gk < k)
                            ? __float2bfloat16(x[static_cast<std::size_t>(gm) * k + gk])
                            : __float2bfloat16(0.0f);
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
            // ---- ldmatrix fragments ----
            const __nv_bfloat16* a_ptr = &sA[warp * 16 + (lane & 15)][(lane >> 4) * 8];
            unsigned a0, a1, a2, a3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                         : "r"(smem_addr(a_ptr)));
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                // x2 only consumes addresses from lanes 0-15; mirror them for
                // the rest so no lane forms an out-of-bounds pointer.
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

}  // namespace

void moe_gemm_int4(const float* x, const std::uint8_t* packed, const float* scales,
                   const float* zeros, int m, int n, int k, int group_size, float* y,
                   cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    moe_gemm_int4_kernel<<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k, group_size,
                                                     y);
}

void moe_gemm_int4_tc_n(const float* x, const std::uint8_t* packed, const float* scales,
                        const float* zeros, int m, int n, int k, int group_size, float* y,
                        int bn, cudaStream_t stream) {
    const dim3 block(128);  // 4 warps -> 64 rows per block
    const dim3 grid((n + bn - 1) / bn, (m + 63) / 64);
    switch (bn) {
        case 16:
            moe_gemm_int4_tc_kernel<16><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n,
                                                                    k, group_size, y);
            break;
        case 32:
            moe_gemm_int4_tc_kernel<32><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n,
                                                                    k, group_size, y);
            break;
        case 64:
            moe_gemm_int4_tc_kernel<64><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n,
                                                                    k, group_size, y);
            break;
        default:
            moe_gemm_int4_tc_kernel<8><<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n,
                                                                   k, group_size, y);
            break;
    }
}

void moe_gemm_int4_tc(const float* x, const std::uint8_t* packed, const float* scales,
                      const float* zeros, int m, int n, int k, int group_size, float* y,
                      cudaStream_t stream) {
    moe_gemm_int4_tc_n(x, packed, scales, zeros, m, n, k, group_size, y, 8, stream);
}

}  // namespace cuda
}  // namespace uocr
