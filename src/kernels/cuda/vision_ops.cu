// CUDA kernels for the DeepEncoder (SAM-ViT + CLIP-L + projector) port.
//
// These mirror the CPU reference ops in `src/engine/deep_encoder.cpp`:
// row-wise LayerNorm, GELU/QuickGELU, im2col for convs (the conv is then a
// bf16 tensor-core GEMM), window partition/unpartition and a flash-style
// attention kernel (online softmax) supporting SAM's decomposed relative
// position bias.
//
// All activations are f32; linear layers use bf16 weights via `matmul_t_bf16`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {
namespace {

// ---------------------------------------------------------------------------
// Elementwise / reduction kernels
// ---------------------------------------------------------------------------
__global__ void layernorm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                      const float* __restrict__ b, float* __restrict__ y, int rows,
                                      int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + static_cast<std::size_t>(row) * cols;
    float* yr = y + static_cast<std::size_t>(row) * cols;

    float mean = 0.0f, var = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) mean += xr[c];
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = mean;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    mean = sdata[0] / cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float d = xr[c] - mean;
        var += d * d;
    }
    __syncthreads();
    sdata[threadIdx.x] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sdata[0] / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = (xr[c] - mean) * inv;
        if (w) v *= w[c];
        if (b) v += b[c];
        yr[c] = v;
    }
}

__global__ void gelu_kernel(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    }
}

__global__ void quick_gelu_kernel(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i] = v / (1.0f + expf(-1.702f * v));
    }
}

__global__ void add_bias_rows_kernel(float* __restrict__ y, const float* __restrict__ b, int rows,
                                     int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    y[i] += b[i % cols];
}

__global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// ---------------------------------------------------------------------------
// im2col for NCHW-ish input (actually CHW, one image) -> [Ho*Wo, Cin*Kh*Kw]
// ---------------------------------------------------------------------------
__global__ void im2col_chw_kernel(const float* __restrict__ x, float* __restrict__ out, int Cin,
                                  int H, int W, int Kh, int Kw, int stride, int pad, int Ho,
                                  int Wo) {
    const std::size_t total = static_cast<std::size_t>(Ho) * Wo * Cin * Kh * Kw;
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    const int kw = idx % Kw;
    std::size_t t = idx / Kw;
    const int kh = t % Kh;
    t /= Kh;
    const int ci = t % Cin;
    const int o = t / Cin;
    const int oh = o / Wo, ow = o % Wo;
    const int ih = oh * stride - pad + kh;
    const int iw = ow * stride - pad + kw;
    out[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                   ? x[(static_cast<std::size_t>(ci) * H + ih) * W + iw]
                   : 0.0f;
}

// im2col for HWC activation [H,W,C] -> [Ho*Wo, C*Kh*Kw], index (ci,kh,kw).
__global__ void im2col_hwc_kernel(const float* __restrict__ x, float* __restrict__ out, int C,
                                  int H, int W, int Kh, int Kw, int stride, int pad, int Ho,
                                  int Wo) {
    const std::size_t total = static_cast<std::size_t>(Ho) * Wo * C * Kh * Kw;
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    const int kw = idx % Kw;
    std::size_t t = idx / Kw;
    const int kh = t % Kh;
    t /= Kh;
    const int ci = t % C;
    const int o = t / C;
    const int oh = o / Wo, ow = o % Wo;
    const int ih = oh * stride - pad + kh;
    const int iw = ow * stride - pad + kw;
    out[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                   ? x[(static_cast<std::size_t>(ih) * W + iw) * C + ci]
                   : 0.0f;
}

// ---------------------------------------------------------------------------
// Window partition/unpartition for HWC activation [H, W, C].
// ---------------------------------------------------------------------------
__global__ void window_partition_kernel(const float* __restrict__ x, float* __restrict__ out,
                                        int H, int W, int C, int ws, int nh, int nw) {
    const std::size_t total = static_cast<std::size_t>(nh) * nw * ws * ws * C;
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % C;
    std::size_t t = idx / C;
    const int j = t % ws;
    t /= ws;
    const int i = t % ws;
    const int win = t / ws;
    const int wi = win / nw, wj = win % nw;
    const int ih = wi * ws + i, iw = wj * ws + j;
    out[idx] = (ih < H && iw < W) ? x[(static_cast<std::size_t>(ih) * W + iw) * C + c] : 0.0f;
}

__global__ void window_unpartition_kernel(const float* __restrict__ win, float* __restrict__ x,
                                          int H, int W, int C, int ws, int nh, int nw) {
    const std::size_t total = static_cast<std::size_t>(nh) * nw * ws * ws * C;
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % C;
    std::size_t t = idx / C;
    const int j = t % ws;
    t /= ws;
    const int i = t % ws;
    const int wnn = t / ws;
    const int wi = wnn / nw, wj = wnn % nw;
    const int ih = wi * ws + i, iw = wj * ws + j;
    if (ih < H && iw < W) x[(static_cast<std::size_t>(ih) * W + iw) * C + c] = win[idx];
}

// ---------------------------------------------------------------------------
// Flash attention (online softmax).  One warp per query; K/V tiles staged in
// shared memory and shared by all warps in the block.
//
//   qkv: [B, S, 3*C] f32, C = heads*hd
//   out: [B, S, C]
//   relh/relw: SAM decomposed relative position tables, [(2H-1)*hd] /
//              [(2W-1)*hd], or nullptr for plain attention (CLIP).
//
// hd is fixed at 64 (two head dims per lane).
// ---------------------------------------------------------------------------
constexpr int kAttnHD = 64;
constexpr int kAttnKT = 64;   // keys per tile
constexpr int kAttnWarps = 8; // warps per block
// Queries per warp.  A block loads each K/V tile once and reuses it for
// `kAttnWarps * kAttnQP` queries; raising this from 1 cuts the (otherwise
// O(S/8)x redundant) K/V global traffic, which dominates the SAM global
// attention at 1024.
constexpr int kAttnQP = 4;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

template <bool RELPOS>
__global__ void attention_flash_kernel(const float* __restrict__ qkv,
                                       const float* __restrict__ relh,
                                       const float* __restrict__ relw, float* __restrict__ out,
                                       int S, int H, int W, int heads) {
    const int C = heads * kAttnHD;
    const int bh = blockIdx.x;
    const int b = bh / heads;
    const int h = bh % heads;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int qbase = (blockIdx.y * kAttnWarps + warp) * kAttnQP;
    constexpr int NQ = kAttnWarps * kAttnQP;

    extern __shared__ float smem[];
    // RELPOS: a [NQ][H+W] per-query relative-position score table, then K/V.
    // Splitting `q . (Rh[ih] + Rw[iw])` into `q.Rh[ih] + q.Rw[iw]` lets the
    // expensive inner loop use two shared lookups instead of four global loads
    // plus a warp reduction per key (which dominated the SAM global attention).
    float* sRel = smem;
    float* sK = smem + (RELPOS ? static_cast<std::size_t>(NQ) * (H + W) : 0);
    float* sV = sK + kAttnKT * kAttnHD;

    const float scale = 1.0f / sqrtf(static_cast<float>(kAttnHD));
    // Each lane owns head dims {2*lane, 2*lane+1}.
    const int d0 = 2 * lane;
    float q0[kAttnQP], q1[kAttnQP];
    bool qok[kAttnQP];
    float acc0[kAttnQP], acc1[kAttnQP], m[kAttnQP], l[kAttnQP];
#pragma unroll
    for (int t = 0; t < kAttnQP; ++t) {
        const int qi = qbase + t;
        qok[t] = qi < S;
        q0[t] = 0.0f;
        q1[t] = 0.0f;
        acc0[t] = acc1[t] = 0.0f;
        m[t] = -INFINITY;
        l[t] = 0.0f;
        if (qok[t]) {
            const float* qp = qkv + (static_cast<std::size_t>(b) * S + qi) * 3 * C + h * kAttnHD;
            q0[t] = qp[d0];
            q1[t] = qp[d0 + 1];
        }
    }

    if (RELPOS) {
        const int RH = H + W;
#pragma unroll
        for (int t = 0; t < kAttnQP; ++t) {
            if (!qok[t]) continue;
            const int qi = qbase + t;
            const int qh = W > 0 ? qi / W : 0;
            const int qw = W > 0 ? qi % W : 0;
            float* row = sRel + (warp * kAttnQP + t) * RH;
            for (int kh = 0; kh < H; ++kh) {
                const float* rp = relh + static_cast<std::size_t>(qh - kh + H - 1) * kAttnHD;
                float r = q0[t] * rp[d0] + q1[t] * rp[d0 + 1];
                r = warp_sum(r);
                if (lane == 0) row[kh] = r;
            }
            for (int kw = 0; kw < W; ++kw) {
                const float* rp = relw + static_cast<std::size_t>(qw - kw + W - 1) * kAttnHD;
                float r = q0[t] * rp[d0] + q1[t] * rp[d0 + 1];
                r = warp_sum(r);
                if (lane == 0) row[H + kw] = r;
            }
        }
        __syncthreads();
    }

    const int nkt = (S + kAttnKT - 1) / kAttnKT;
    for (int kt = 0; kt < nkt; ++kt) {
        // ---- stage K/V tiles (all threads participate) ----
        for (int i = threadIdx.x; i < kAttnKT * kAttnHD; i += blockDim.x) {
            const int j = i / kAttnHD;
            const int d = i % kAttnHD;
            const int key = kt * kAttnKT + j;
            float kv = 0.0f, vv = 0.0f;
            if (key < S) {
                const float* base =
                    qkv + (static_cast<std::size_t>(b) * S + key) * 3 * C + h * kAttnHD;
                kv = base[C + d];
                vv = base[2 * C + d];
            }
            sK[j * kAttnHD + d] = kv;
            sV[j * kAttnHD + d] = vv;
        }
        __syncthreads();

        const int jmax = min(kAttnKT, S - kt * kAttnKT);
        // Fully unroll so the per-query arrays stay in registers (a dynamic
        // index would spill them to local memory).
#pragma unroll
        for (int t = 0; t < kAttnQP; ++t) {
            if (!qok[t]) continue;
            const float* row = sRel + (warp * kAttnQP + t) * (H + W);
            const float* kt_row = row + H;
            float mt = m[t], lt = l[t], a0 = acc0[t], a1 = acc1[t];
            for (int j = 0; j < jmax; ++j) {
                const int key = kt * kAttnKT + j;
                const float2 kk = *reinterpret_cast<const float2*>(sK + j * kAttnHD + d0);
                float dot = q0[t] * kk.x + q1[t] * kk.y;
                dot = warp_sum(dot);
                float s = dot * scale;
                if (RELPOS) s += row[key / W] + kt_row[key % W];
                const float mnew = fmaxf(mt, s);
                // __expf (MUFU) instead of the ~2x-ULP-accurate libm expf: the
                // softmax probability error is ~1e-6, far below the 6e-4 budget,
                // and it removes a large instruction-count cost in the inner loop.
                const float alpha = __expf(mt - mnew);
                const float beta = __expf(s - mnew);
                const float2 vv = *reinterpret_cast<const float2*>(sV + j * kAttnHD + d0);
                lt = lt * alpha + beta;
                a0 = a0 * alpha + beta * vv.x;
                a1 = a1 * alpha + beta * vv.y;
                mt = mnew;
            }
            m[t] = mt;
            l[t] = lt;
            acc0[t] = a0;
            acc1[t] = a1;
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < kAttnQP; ++t) {
        if (!qok[t]) continue;
        const int qi = qbase + t;
        const float inv = l[t] > 0.0f ? 1.0f / l[t] : 0.0f;
        float* op = out + (static_cast<std::size_t>(b) * S + qi) * C + h * kAttnHD;
        op[d0] = acc0[t] * inv;
        op[d0 + 1] = acc1[t] * inv;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
void layernorm_rows(const float* x, const float* w, const float* b, float* y, int rows, int cols,
                    float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    const int threads = 256;
    layernorm_rows_kernel<<<rows, threads, threads * sizeof(float), stream>>>(x, w, b, y, rows, cols,
                                                                             eps);
}

void gelu_inplace(float* x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int threads = 256;
    gelu_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x, n);
}

void quick_gelu_inplace(float* x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int threads = 256;
    quick_gelu_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x, n);
}

void add_bias_rows(float* y, const float* bias, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || bias == nullptr) return;
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    const int threads = 256;
    add_bias_rows_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(y, bias, rows, cols);
}

void add_inplace(float* y, const float* x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int threads = 256;
    add_inplace_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(y, x, n);
}

void im2col_chw(const float* x, float* out, int Cin, int H, int W, int Kh, int Kw, int stride,
                int pad, int Ho, int Wo, cudaStream_t stream) {
    const std::size_t total = static_cast<std::size_t>(Ho) * Wo * Cin * Kh * Kw;
    if (total == 0) return;
    const int threads = 256;
    im2col_chw_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(x, out, Cin, H, W,
                                                                                Kh, Kw, stride, pad,
                                                                                Ho, Wo);
}

void im2col_hwc(const float* x, float* out, int Cin, int H, int W, int Kh, int Kw, int stride,
                int pad, int Ho, int Wo, cudaStream_t stream) {
    const std::size_t total = static_cast<std::size_t>(Ho) * Wo * Cin * Kh * Kw;
    if (total == 0) return;
    const int threads = 256;
    im2col_hwc_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(x, out, Cin, H, W,
                                                                                Kh, Kw, stride, pad,
                                                                                Ho, Wo);
}

void window_partition(const float* x, float* out, int H, int W, int C, int ws, int nh, int nw,
                      cudaStream_t stream) {
    const std::size_t total = static_cast<std::size_t>(nh) * nw * ws * ws * C;
    if (total == 0) return;
    const int threads = 256;
    window_partition_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(x, out, H, W,
                                                                                     C, ws, nh, nw);
}

void window_unpartition(const float* win, float* x, int H, int W, int C, int ws, int nh, int nw,
                        cudaStream_t stream) {
    const std::size_t total = static_cast<std::size_t>(nh) * nw * ws * ws * C;
    if (total == 0) return;
    const int threads = 256;
    window_unpartition_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(win, x, H,
                                                                                       W, C, ws, nh,
                                                                                       nw);
}

void attention_flash(const float* qkv, const float* relh, const float* relw, float* out, int B,
                     int S, int H, int W, int heads, bool relpos, cudaStream_t stream) {
    if (B <= 0 || S <= 0 || heads <= 0) return;
    const dim3 block(32 * kAttnWarps);
    const dim3 grid(B * heads, (S + kAttnWarps * kAttnQP - 1) / (kAttnWarps * kAttnQP));
    const std::size_t kv = static_cast<std::size_t>(2) * kAttnKT * kAttnHD;
    const std::size_t rel = relpos
                                ? static_cast<std::size_t>(kAttnWarps * kAttnQP) *
                                      static_cast<std::size_t>(H + W)
                                : 0;
    const std::size_t shmem = (kv + rel) * sizeof(float);
    if (relpos) {
        // Large images push the relpos table past the 48KB default.
        cudaFuncSetAttribute(attention_flash_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmem));
        attention_flash_kernel<true>
            <<<grid, block, shmem, stream>>>(qkv, relh, relw, out, S, H, W, heads);
    } else {
        attention_flash_kernel<false>
            <<<grid, block, shmem, stream>>>(qkv, relh, relw, out, S, H, W, heads);
    }
}

}  // namespace cuda
}  // namespace uocr
