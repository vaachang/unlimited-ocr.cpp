// CUDA backend tests: R-SWA decode attention and INT4 MoE GEMM vs CPU reference.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "uocr/cuda_ops.h"
#include "uocr/kv_cache.h"
#include "uocr/quant.h"

using namespace uocr;

#define CUDA_CHECK(x)                                                      \
    do {                                                                   \
        cudaError_t err = (x);                                             \
        if (err != cudaSuccess) {                                          \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            return 1;                                                      \
        }                                                                  \
    } while (0)

static std::vector<float> rand_vec(std::mt19937& rng, std::size_t n) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

int main() {
    if (!cuda::available()) {
        std::printf("no CUDA device available; skipping CUDA tests\n");
        return 0;
    }
    auto info = cuda::device_info();
    std::printf("device: %s (sm_%d%d, %d SMs)\n", info.name, info.compute_major,
                info.compute_minor, info.multi_processor_count);

    std::mt19937 rng(1234);
    int failures = 0;

    // ---- R-SWA decode attention ----
    {
        const int layers = 1, kv_heads = 10, heads = 10, hd = 128, W = 128, P = 257;
        RSWACache cache(layers, kv_heads, hd, W);
        cache.reset(P);
        auto pre_k = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
        auto pre_v = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
        cache.write_prefill(0, pre_k.data(), pre_v.data(), P);
        for (int i = 0; i < 50; ++i) {
            auto k = rand_vec(rng, kv_heads * hd);
            auto v = rand_vec(rng, kv_heads * hd);
            cache.append_decode(0, k.data(), v.data());
        }
        const int kv_len = cache.length(0);

        auto q = rand_vec(rng, static_cast<std::size_t>(heads) * hd);
        std::vector<float> cpu_out(static_cast<std::size_t>(heads) * hd);
        cache.attention(0, q.data(), 1, kv_len - 1, cpu_out.data(), heads, false);

        float* d_q = nullptr;
        float* d_k = nullptr;
        float* d_v = nullptr;
        float* d_out = nullptr;
        const std::size_t kv_bytes = static_cast<std::size_t>(cache.capacity()) * kv_heads * hd * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_q, q.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k, kv_bytes));
        CUDA_CHECK(cudaMalloc(&d_v, kv_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, cpu_out.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k, cache.keys(0), kv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, cache.values(0), kv_bytes, cudaMemcpyHostToDevice));

        cuda::rswa_attention_decode(d_q, d_k, d_v, kv_len, heads, kv_heads, hd, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> gpu_out(cpu_out.size());
        CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_out);

        float max_err = 0.0f;
        for (std::size_t i = 0; i < cpu_out.size(); ++i)
            max_err = std::max(max_err, std::fabs(cpu_out[i] - gpu_out[i]));
        std::printf("R-SWA attention: kv_len=%d max_err=%.6f\n", kv_len, max_err);
        if (max_err > 1e-3f) {
            std::printf("  FAIL\n");
            ++failures;
        } else {
            std::printf("  OK\n");
        }
    }

    // ---- INT4 MoE GEMM ----
    {
        const int m = 8, n = 64, k = 256, group = 128;
        auto w = rand_vec(rng, static_cast<std::size_t>(n) * k);
        auto x = rand_vec(rng, static_cast<std::size_t>(m) * k);
        QuantizedMatrix qm = quantize_int4_awq(w.data(), n, k, group);

        // CPU reference via dequantize
        std::vector<float> wf;
        qm.dequantize(wf);
        std::vector<float> cpu_y(static_cast<std::size_t>(m) * n, 0.0f);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int c = 0; c < k; ++c)
                    acc += x[static_cast<std::size_t>(i) * k + c] * wf[static_cast<std::size_t>(j) * k + c];
                cpu_y[static_cast<std::size_t>(i) * n + j] = acc;
            }

        float *d_x, *d_y, *d_s, *d_z;
        std::uint8_t* d_p;
        CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, cpu_y.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_p, qm.packed.size()));
        CUDA_CHECK(cudaMalloc(&d_s, qm.scales.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_z, qm.zeros.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p, qm.packed.data(), qm.packed.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_s, qm.scales.data(), qm.scales.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z, qm.zeros.data(), qm.zeros.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        cuda::moe_gemm_int4(d_x, d_p, d_s, d_z, m, n, k, group, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gpu_y(cpu_y.size());
        CUDA_CHECK(cudaMemcpy(gpu_y.data(), d_y, gpu_y.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_p);
        cudaFree(d_s);
        cudaFree(d_z);

        float max_err = 0.0f;
        for (std::size_t i = 0; i < cpu_y.size(); ++i)
            max_err = std::max(max_err, std::fabs(cpu_y[i] - gpu_y[i]));
        std::printf("INT4 MoE GEMM: [%d,%d]x[%d,%d] max_err=%.6f\n", m, n, k, k, max_err);
        if (max_err > 1e-3f) {
            std::printf("  FAIL\n");
            ++failures;
        } else {
            std::printf("  OK\n");
        }
    }

    std::printf("%s\n", failures == 0 ? "all CUDA tests passed" : "CUDA tests FAILED");
    return failures == 0 ? 0 : 1;
}
