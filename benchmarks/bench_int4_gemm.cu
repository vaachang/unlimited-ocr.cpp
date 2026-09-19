// INT4 (W4A16) GEMM micro-benchmark: scalar vs tensor-core (`ldmatrix`) paths.
//
// Usage: bench_int4_gemm [--n 896] [--k 1280] [--group 128] [--iters 200]

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "uocr/cuda_ops.h"
#include "uocr/quant.h"

namespace {

float* upload_f32(const std::vector<float>& v) {
    float* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(float));
    cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

double time_us(std::function<void()> fn, int iters) {
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
    int n = 896, k = 1280, group = 128, iters = 200;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--group") && i + 1 < argc) group = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    }
    if (!uocr::cuda::available()) {
        std::printf("no CUDA device\n");
        return 0;
    }

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> hw(static_cast<std::size_t>(n) * k);
    for (auto& v : hw) v = dist(rng);
    uocr::QuantizedMatrix qm = uocr::quantize_int4_awq(hw.data(), n, k, group);
    std::vector<float> hx(static_cast<std::size_t>(273) * k);
    for (auto& v : hx) v = dist(rng);

    float* d_x = upload_f32(hx);
    float* d_y = nullptr;
    cudaMalloc(&d_y, static_cast<std::size_t>(273) * n * sizeof(float));
    std::uint8_t* d_p = nullptr;
    float *d_s = nullptr, *d_z = nullptr;
    cudaMalloc(&d_p, qm.packed.size());
    cudaMalloc(&d_s, qm.scales.size() * sizeof(float));
    cudaMalloc(&d_z, qm.zeros.size() * sizeof(float));
    cudaMemcpy(d_p, qm.packed.data(), qm.packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, qm.scales.data(), qm.scales.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, qm.zeros.data(), qm.zeros.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::printf("INT4 GEMM n=%d k=%d group=%d iters=%d\n", n, k, group, iters);
    std::printf("%6s %14s %14s %10s\n", "m", "scalar(us)", "tensorcore(us)", "TC TFLOPS");
    for (int m : {1, 8, 32, 64, 128, 273}) {
        if (m > 273) continue;
        const double t_scalar = time_us([&] {
            uocr::cuda::moe_gemm_int4(d_x, d_p, d_s, d_z, m, n, k, group, d_y);
        }, iters);
        const double t_tc = time_us([&] {
            uocr::cuda::moe_gemm_int4_tc(d_x, d_p, d_s, d_z, m, n, k, group, d_y);
        }, iters);
        const double flop = 2.0 * m * n * k;
        const double tflops = flop / (t_tc * 1e-6) / 1e12;
        std::printf("%6d %14.1f %14.1f %10.3f\n", m, t_scalar, t_tc, tflops);
    }
    return 0;
}
