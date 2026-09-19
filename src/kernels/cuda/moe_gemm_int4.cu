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

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    const unsigned short l = __bfloat16_as_ushort(__float2bfloat16_rn(lo));
    const unsigned short h = __bfloat16_as_ushort(__float2bfloat16_rn(hi));
    return static_cast<unsigned>(l) | (static_cast<unsigned>(h) << 16);
}

// W4A16 tensor-core kernel.  Each warp computes a 16(m) x 8(n) output tile and
// walks K in steps of 16, dequantizing the INT4 weights to bf16 in registers
// before the `mma.m16n8k16.bf16` instruction (f32 accumulator).
//
// Fragment layouts follow PTX ISA 9.7.16.5.8:
//   A (16x16): a0,a1 = (g,tig*2),(g,tig*2+1); a2,a3 = (g+8,*);
//              a4,a5 = (g,tig*2+8),(g,tig*2+9); a6,a7 = (g+8,*)
//   B (16x8):  b0,b1 = (tig*2,*),(tig*2+1,*); b2,b3 = (tig*2+8,*),(tig*2+9,*)
//   C (16x8):  c0,c1 = (g,tig*2),(g,tig*2+1); c2,c3 = (g+8,*)
__global__ void moe_gemm_int4_tc_kernel(const float* __restrict__ x,
                                        const std::uint8_t* __restrict__ packed,
                                        const float* __restrict__ scales,
                                        const float* __restrict__ zeros, int m, int n, int k,
                                        int group_size, float* __restrict__ y) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int m0 = blockIdx.y * 16;
    const int n0 = blockIdx.x * 8;
    const int packed_row = (k + 1) / 2;
    const int ng = (k + group_size - 1) / group_size;

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k0 = 0; k0 < k; k0 += 16) {
        auto xload = [&](int r, int cc) -> float {
            const int mm = m0 + r;
            const int kk = k0 + cc;
            if (mm >= m || kk >= k) return 0.0f;
            return x[static_cast<std::size_t>(mm) * k + kk];
        };
        const int wn = n0 + g;
        auto wload = [&](int kk) -> float {
            if (wn >= n || kk >= k) return 0.0f;
            const std::uint8_t* prow = packed + static_cast<std::size_t>(wn) * packed_row;
            const std::uint8_t byte = prow[kk >> 1];
            const int q = (kk & 1) ? (byte >> 4) : (byte & 0x0f);
            const int gg = kk / group_size;
            return (static_cast<float>(q) - zeros[static_cast<std::size_t>(wn) * ng + gg]) *
                   scales[static_cast<std::size_t>(wn) * ng + gg];
        };

        const unsigned a0 = pack_bf16(xload(g, tig * 2), xload(g, tig * 2 + 1));
        const unsigned a1 = pack_bf16(xload(g + 8, tig * 2), xload(g + 8, tig * 2 + 1));
        const unsigned a2 = pack_bf16(xload(g, tig * 2 + 8), xload(g, tig * 2 + 9));
        const unsigned a3 = pack_bf16(xload(g + 8, tig * 2 + 8), xload(g + 8, tig * 2 + 9));
        const unsigned b0 = pack_bf16(wload(k0 + tig * 2), wload(k0 + tig * 2 + 1));
        const unsigned b1 = pack_bf16(wload(k0 + tig * 2 + 8), wload(k0 + tig * 2 + 9));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    const int r0 = m0 + g, r1 = m0 + g + 8;
    const int c0 = n0 + tig * 2, c1 = n0 + tig * 2 + 1;
    if (r0 < m) {
        if (c0 < n) y[static_cast<std::size_t>(r0) * n + c0] = c[0];
        if (c1 < n) y[static_cast<std::size_t>(r0) * n + c1] = c[1];
    }
    if (r1 < m) {
        if (c0 < n) y[static_cast<std::size_t>(r1) * n + c0] = c[2];
        if (c1 < n) y[static_cast<std::size_t>(r1) * n + c1] = c[3];
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

void moe_gemm_int4_tc(const float* x, const std::uint8_t* packed, const float* scales,
                      const float* zeros, int m, int n, int k, int group_size, float* y,
                      cudaStream_t stream) {
    const dim3 block(32);
    const dim3 grid((n + 7) / 8, (m + 15) / 16);
    moe_gemm_int4_tc_kernel<<<grid, block, 0, stream>>>(x, packed, scales, zeros, m, n, k,
                                                        group_size, y);
}

}  // namespace cuda
}  // namespace uocr
