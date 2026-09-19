// bf16 GEMM micro-benchmark: CUDA-core tiled vs tensor-core (`ldmatrix`/mma).
//
// Shapes follow the decoder's dense projections (n,k) with the m values seen
// during prefill/decode, plus the lm_head (n = vocab).
//
// Usage: bench_bf16_gemm [--n 1280] [--k 1280] [--iters 200]

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "uocr/common.h"
#include "uocr/cuda_ops.h"

namespace {

double time_us(const std::function<void()>& fn, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    fn();  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return static_cast<double>(ms) * 1000.0 / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int n = 1280, k = 1280, iters = 200, mmax = 273;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    }
    if (!uocr::cuda::available()) {
        std::printf("no CUDA device\n");
        return 0;
    }

    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> hx(static_cast<std::size_t>(mmax) * k);
    for (auto& v : hx) v = dist(rng);
    std::vector<std::uint16_t> hw(static_cast<std::size_t>(n) * k);
    std::vector<float> hwf(hw.size());
    for (auto& v : hwf) v = dist(rng);
    for (std::size_t i = 0; i < hw.size(); ++i) hw[i] = uocr::f32_to_bf16(hwf[i]);

    float* d_x = nullptr;
    float* d_y = nullptr;
    std::uint16_t* d_w = nullptr;
    cudaMalloc(&d_x, hx.size() * sizeof(float));
    cudaMalloc(&d_y, static_cast<std::size_t>(mmax) * n * sizeof(float));
    cudaMalloc(&d_w, hw.size() * sizeof(std::uint16_t));
    cudaMemcpy(d_x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), hw.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);

    std::printf("bf16 GEMM n=%d k=%d iters=%d\n", n, k, iters);
    std::printf("%6s %14s %14s %10s %8s\n", "m", "tiled(us)", "tc(us)", "TC TFLOPS", "speedup");
    for (int m : {1, 8, 16, 64, 128, 273}) {
        double t_ref = 0.0, t_tc = 0.0;
        if (m == 1) {
            // matvec is the single-row fast path; still report it as "tiled".
            t_ref = time_us([&] { uocr::cuda::matvec_bf16(d_x, d_w, nullptr, d_y, n, k); }, iters);
            t_tc = t_ref;
        } else {
            t_ref = time_us(
                [&] { uocr::cuda::matmul_t_bf16_ref(d_x, d_w, nullptr, d_y, m, n, k); }, iters);
            t_tc = time_us([&] { uocr::cuda::matmul_t_bf16(d_x, d_w, nullptr, d_y, m, n, k); },
                           iters);
        }
        const double flop = 2.0 * m * n * k;
        const double tflops = flop / (t_tc * 1e-6) / 1e12;
        std::printf("%6d %14.1f %14.1f %10.3f %8.2f\n", m, t_ref, t_tc, tflops,
                    t_tc > 0 ? t_ref / t_tc : 0.0);
    }
    return 0;
}
