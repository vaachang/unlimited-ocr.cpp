// CUDA backend tests: R-SWA decode attention and INT4 MoE GEMM vs CPU reference.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "uocr/cuda_ops.h"
#include "uocr/gpu_cache.h"
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

    // ---- device R-SWA cache: ring overwrite + prefill causal attention ----
    {
        const int layers = 1, kv_heads = 10, heads = 10, hd = 128, W = 8, P = 4;
        auto upload = [&](const std::vector<float>& h) -> float* {
            float* d = nullptr;
            if (cudaMalloc(&d, h.size() * sizeof(float)) != cudaSuccess) {
                std::printf("cudaMalloc failed\n");
                return nullptr;
            }
            if (cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
                std::printf("cudaMemcpy H2D failed\n");
                return nullptr;
            }
            return d;
        };

        // decode path: drive host + device caches with the same K/V stream
        RSWACache hc(layers, kv_heads, hd, W);
        hc.reset(P);
        cuda::GpuRSWACache gc(layers, kv_heads, hd, W);
        gc.reset(P);
        std::vector<float> pk = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
        std::vector<float> pv = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
        hc.write_prefill(0, pk.data(), pv.data(), P);
        {
            float* dk = upload(pk);
            float* dv = upload(pv);
            gc.write_prefill(0, dk, dv, P);
            cudaFree(dk);
            cudaFree(dv);
        }
        for (int t = 0; t < 24; ++t) {
            std::vector<float> k = rand_vec(rng, kv_heads * hd);
            std::vector<float> v = rand_vec(rng, kv_heads * hd);
            hc.append_decode(0, k.data(), v.data());
            float* dk = upload(k);
            float* dv = upload(v);
            gc.append_decode(0, dk, dv);
            cudaFree(dk);
            cudaFree(dv);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> q = rand_vec(rng, static_cast<std::size_t>(heads) * hd);
        std::vector<float> cpu_out(static_cast<std::size_t>(heads) * hd);
        hc.attention(0, q.data(), 1, hc.length(0) - 1, cpu_out.data(), heads, false);
        float* dq = upload(q);
        float* dout = nullptr;
        CUDA_CHECK(cudaMalloc(&dout, cpu_out.size() * sizeof(float)));
        gc.attention(0, dq, 1, hc.length(0) - 1, dout, heads, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gpu_out(cpu_out.size());
        CUDA_CHECK(cudaMemcpy(gpu_out.data(), dout, gpu_out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float dec_err = 0.0f;
        for (std::size_t i = 0; i < cpu_out.size(); ++i)
            dec_err = std::max(dec_err, std::fabs(cpu_out[i] - gpu_out[i]));
        cudaFree(dq);
        cudaFree(dout);

        const std::size_t cap = static_cast<std::size_t>(hc.capacity()) * kv_heads * hd;
        std::vector<float> gk(cap), gv(cap);
        CUDA_CHECK(cudaMemcpy(gk.data(), gc.keys(0), cap * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gv.data(), gc.values(0), cap * sizeof(float), cudaMemcpyDeviceToHost));
        float kerr = 0.0f, verr = 0.0f;
        for (std::size_t i = 0; i < cap; ++i) {
            kerr = std::max(kerr, std::fabs(hc.keys(0)[i] - gk[i]));
            verr = std::max(verr, std::fabs(hc.values(0)[i] - gv[i]));
        }
        std::printf("GpuRSWACache: len=%d decode_max_err=%.6f cache_k_err=%.6f cache_v_err=%.6f\n",
                    gc.len(0), dec_err, kerr, verr);
        if (dec_err > 1e-3f || kerr > 1e-5f || verr > 1e-5f) {
            std::printf("  FAIL\n");
            ++failures;
        } else {
            std::printf("  OK\n");
        }

        // prefill causal attention
        RSWACache hp(layers, kv_heads, hd, W);
        hp.reset(P);
        hp.write_prefill(0, pk.data(), pv.data(), P);
        cuda::GpuRSWACache gp(layers, kv_heads, hd, W);
        gp.reset(P);
        {
            float* dk = upload(pk);
            float* dv = upload(pv);
            gp.write_prefill(0, dk, dv, P);
            cudaFree(dk);
            cudaFree(dv);
        }
        std::vector<float> qa = rand_vec(rng, static_cast<std::size_t>(P) * heads * hd);
        std::vector<float> cpu_a(qa.size());
        hp.attention(0, qa.data(), P, 0, cpu_a.data(), heads, true);
        float* dqa = upload(qa);
        float* douta = nullptr;
        CUDA_CHECK(cudaMalloc(&douta, cpu_a.size() * sizeof(float)));
        gp.attention(0, dqa, P, 0, douta, heads, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gpu_a(cpu_a.size());
        CUDA_CHECK(cudaMemcpy(gpu_a.data(), douta, gpu_a.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float pre_err = 0.0f;
        for (std::size_t i = 0; i < cpu_a.size(); ++i)
            pre_err = std::max(pre_err, std::fabs(cpu_a[i] - gpu_a[i]));
        cudaFree(dqa);
        cudaFree(douta);
        std::printf("GpuRSWACache prefill causal: max_err=%.6f\n", pre_err);
        if (pre_err > 1e-3f) {
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

        // tensor-core W4A16 variant (same buffers)
        CUDA_CHECK(cudaMemset(d_y, 0, cpu_y.size() * sizeof(float)));
        cuda::moe_gemm_int4_tc(d_x, d_p, d_s, d_z, m, n, k, group, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> tc_y(cpu_y.size());
        CUDA_CHECK(cudaMemcpy(tc_y.data(), d_y, tc_y.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float tc_err = 0.0f, rel = 0.0f, den = 0.0f;
        for (std::size_t i = 0; i < cpu_y.size(); ++i) {
            tc_err = std::max(tc_err, std::fabs(cpu_y[i] - tc_y[i]));
            rel += (cpu_y[i] - tc_y[i]) * (cpu_y[i] - tc_y[i]);
            den += cpu_y[i] * cpu_y[i];
        }
        const float tc_rel = std::sqrt(rel / (den + 1e-30));
        std::printf("INT4 MoE GEMM (tensor-core bf16): [%d,%d]x[%d,%d] max_err=%.6f rel_l2=%.5f\n",
                    m, n, k, k, tc_err, tc_rel);
        // bf16 rounding of both operands bounds the error to ~1e-2 relative.
        if (tc_rel > 0.02f) {
            std::printf("  FAIL\n");
            ++failures;
        } else {
            std::printf("  OK\n");
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_p);
        cudaFree(d_s);
        cudaFree(d_z);
    }

    // ---- tensor-core W4A16 with non-multiple-of-16 K and M ----
    {
        const int m = 5, n = 24, k = 48, group = 16;
        auto w = rand_vec(rng, static_cast<std::size_t>(n) * k);
        auto x = rand_vec(rng, static_cast<std::size_t>(m) * k);
        QuantizedMatrix qm = quantize_int4_awq(w.data(), n, k, group);
        std::vector<float> wf;
        qm.dequantize(wf);
        std::vector<float> cpu_y(static_cast<std::size_t>(m) * n, 0.0f);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int c = 0; c < k; ++c)
                    acc += x[static_cast<std::size_t>(i) * k + c] *
                           wf[static_cast<std::size_t>(j) * k + c];
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
        cuda::moe_gemm_int4_tc(d_x, d_p, d_s, d_z, m, n, k, group, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> tc_y(cpu_y.size());
        CUDA_CHECK(cudaMemcpy(tc_y.data(), d_y, tc_y.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float err = 0.0f, rn = 0.0f, rd = 0.0f;
        for (std::size_t i = 0; i < cpu_y.size(); ++i) {
            err = std::max(err, std::fabs(cpu_y[i] - tc_y[i]));
            rn += (cpu_y[i] - tc_y[i]) * (cpu_y[i] - tc_y[i]);
            rd += cpu_y[i] * cpu_y[i];
        }
        const float rel = std::sqrt(rn / (rd + 1e-30));
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_p);
        cudaFree(d_s);
        cudaFree(d_z);
        std::printf("INT4 MoE GEMM (tensor-core, ragged M/K): [%d,%d]x[%d,%d] max_err=%.6f "
                    "rel_l2=%.5f\n",
                    m, n, k, k, err, rel);
        if (rel > 0.02f) {
            std::printf("  FAIL\n");
            ++failures;
        } else {
            std::printf("  OK\n");
        }
    }

    std::printf("%s\n", failures == 0 ? "all CUDA tests passed" : "CUDA tests FAILED");
    return failures == 0 ? 0 : 1;
}
