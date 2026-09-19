// CUDA backend tests: R-SWA decode attention and INT4 MoE GEMM vs CPU reference.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "uocr/config.h"
#include "uocr/cuda_ops.h"
#include "uocr/engine.h"
#include "uocr/gpu_cache.h"
#include "uocr/gpu_decoder.h"
#include "uocr/kv_cache.h"
#include "uocr/moe_decoder.h"
#include "uocr/quant.h"
#include "uocr/weights.h"

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

    // ---- batched R-SWA append/attention vs per-slot device kernels ----
    {
        const int kv_heads = 4, heads = 4, hd = 64, W = 8, batch = 3;
        const int prefill[batch] = {4, 6, 3};
        const int cap = 6 + W;
        const std::size_t stride = static_cast<std::size_t>(kv_heads) * hd;
        const std::size_t slot_elems = static_cast<std::size_t>(cap) * stride;

        std::vector<float> init_k(static_cast<std::size_t>(batch) * slot_elems, 0.0f);
        std::vector<float> init_v(init_k.size(), 0.0f);
        for (int b = 0; b < batch; ++b) {
            auto k = rand_vec(rng, static_cast<std::size_t>(prefill[b]) * stride);
            auto v = rand_vec(rng, static_cast<std::size_t>(prefill[b]) * stride);
            std::copy(k.begin(), k.end(), init_k.begin() + static_cast<std::size_t>(b) * slot_elems);
            std::copy(v.begin(), v.end(), init_v.begin() + static_cast<std::size_t>(b) * slot_elems);
        }

        auto upload = [&](const std::vector<float>& h) -> float* {
            float* d = nullptr;
            if (cudaMalloc(&d, h.size() * sizeof(float)) != cudaSuccess) return nullptr;
            cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
            return d;
        };
        float* dk_batch = upload(init_k);
        float* dv_batch = upload(init_v);
        float* dk_ref = upload(init_k);
        float* dv_ref = upload(init_v);
        int *d_len_b = nullptr, *d_ring_b = nullptr, *d_pref = nullptr, *d_slots = nullptr;
        int *d_len_r = nullptr, *d_ring_r = nullptr;
        // Decode row -> cache slot permutation (non-identity on purpose).
        const int slots[batch] = {2, 1, 0};
        CUDA_CHECK(cudaMalloc(&d_len_b, batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ring_b, batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pref, batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_slots, batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_len_r, batch * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ring_r, batch * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_len_b, prefill, batch * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_len_r, prefill, batch * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pref, prefill, batch * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_slots, slots, batch * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<int> zero(batch, 0);
        CUDA_CHECK(cudaMemcpy(d_ring_b, zero.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ring_r, zero.data(), batch * sizeof(int), cudaMemcpyHostToDevice));

        const int steps = 3 * W;  // warmup, fill, and several ring wraps
        float worst = 0.0f;
        for (int step = 0; step < steps; ++step) {
            std::vector<float> kstep(static_cast<std::size_t>(batch) * stride);
            std::vector<float> vstep(kstep.size());
            for (int b = 0; b < batch; ++b) {
                auto k = rand_vec(rng, stride);
                auto v = rand_vec(rng, stride);
                std::copy(k.begin(), k.end(), kstep.begin() + static_cast<std::size_t>(b) * stride);
                std::copy(v.begin(), v.end(), vstep.begin() + static_cast<std::size_t>(b) * stride);
            }
            std::vector<float> q = rand_vec(rng, static_cast<std::size_t>(batch) * heads * hd);
            float* d_kstep = upload(kstep);
            float* d_vstep = upload(vstep);
            float* d_q = upload(q);
            float* d_out_b = nullptr;
            float* d_out_r = nullptr;
            CUDA_CHECK(cudaMalloc(&d_out_b, q.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out_r, q.size() * sizeof(float)));

            cuda::rswa_append_decode_batch(d_kstep, d_vstep, dk_batch, dv_batch, d_len_b, d_ring_b,
                                           d_pref, d_slots, batch, cap, W, kv_heads, hd);
            cuda::rswa_attention_batch(d_q, dk_batch, dv_batch, d_len_b, d_slots, batch, cap,
                                       heads, kv_heads, hd, d_out_b);

            for (int b = 0; b < batch; ++b) {
                const int sl = slots[b];
                cuda::rswa_append_decode(d_kstep + static_cast<std::size_t>(b) * stride,
                                         d_vstep + static_cast<std::size_t>(b) * stride,
                                         dk_ref + static_cast<std::size_t>(sl) * slot_elems,
                                         dv_ref + static_cast<std::size_t>(sl) * slot_elems,
                                         d_len_r + sl, d_ring_r + sl, prefill[sl], W, kv_heads, hd);
                cuda::rswa_attention_devlen(
                    d_q + static_cast<std::size_t>(b) * heads * hd,
                    dk_ref + static_cast<std::size_t>(sl) * slot_elems,
                    dv_ref + static_cast<std::size_t>(sl) * slot_elems, d_len_r + sl, 1, 0, heads,
                    kv_heads, hd, false,
                    d_out_r + static_cast<std::size_t>(b) * heads * hd);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> out_b(q.size()), out_r(q.size());
            CUDA_CHECK(cudaMemcpy(out_b.data(), d_out_b, q.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(out_r.data(), d_out_r, q.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < q.size(); ++i)
                worst = std::max(worst, std::fabs(out_b[i] - out_r[i]));

            cudaFree(d_kstep);
            cudaFree(d_vstep);
            cudaFree(d_q);
            cudaFree(d_out_b);
            cudaFree(d_out_r);
        }

        // The batched ring cursors must have advanced identically.
        std::vector<int> len_b(batch), len_r(batch);
        CUDA_CHECK(cudaMemcpy(len_b.data(), d_len_b, batch * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(len_r.data(), d_len_r, batch * sizeof(int), cudaMemcpyDeviceToHost));
        const bool len_ok = len_b == len_r;

        // Final cache contents must match the per-slot reference exactly.
        const std::size_t total = static_cast<std::size_t>(batch) * slot_elems;
        std::vector<float> kb(total), kr(total), vb(total), vr(total);
        CUDA_CHECK(cudaMemcpy(kb.data(), dk_batch, total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(kr.data(), dk_ref, total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vb.data(), dv_batch, total * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vr.data(), dv_ref, total * sizeof(float), cudaMemcpyDeviceToHost));
        float cache_err = 0.0f;
        for (std::size_t i = 0; i < total; ++i) {
            cache_err = std::max(cache_err, std::fabs(kb[i] - kr[i]));
            cache_err = std::max(cache_err, std::fabs(vb[i] - vr[i]));
        }

        cudaFree(dk_batch);
        cudaFree(dv_batch);
        cudaFree(dk_ref);
        cudaFree(dv_ref);
        cudaFree(d_len_b);
        cudaFree(d_ring_b);
        cudaFree(d_pref);
        cudaFree(d_slots);
        cudaFree(d_len_r);
        cudaFree(d_ring_r);

        std::printf("Batched R-SWA (B=%d, W=%d, %d steps): attn_max_err=%.6f cache_err=%.6f %s\n",
                    batch, W, steps, worst, cache_err, len_ok ? "" : "len_mismatch");
        if (worst > 1e-4f || cache_err > 1e-5f || !len_ok) {
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

    // ---- bf16 tensor-core GEMM vs CUDA-core reference (various m/n/k) ----
    {
        struct Shape { int m, n, k; };
        const Shape shapes[] = {{16, 64, 128}, {64, 128, 256}, {3, 40, 48}, {273, 96, 160}};
        float worst_rel = 0.0f;
        for (const Shape& s : shapes) {
            const int m = s.m, n = s.n, k = s.k;
            auto x = rand_vec(rng, static_cast<std::size_t>(m) * k);
            auto wf = rand_vec(rng, static_cast<std::size_t>(n) * k);
            auto bias = rand_vec(rng, n);
            std::vector<std::uint16_t> wb(wf.size());
            for (std::size_t i = 0; i < wf.size(); ++i) wb[i] = f32_to_bf16(wf[i]);

            float *d_x, *d_b, *d_y, *d_yref;
            std::uint16_t* d_w;
            CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_w, wb.size() * sizeof(std::uint16_t)));
            CUDA_CHECK(cudaMalloc(&d_b, bias.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_y, static_cast<std::size_t>(m) * n * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_yref, static_cast<std::size_t>(m) * n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_w, wb.data(), wb.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b, bias.data(), bias.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
            cuda::matmul_t_bf16(d_x, d_w, d_b, d_y, m, n, k);
            cuda::matmul_t_bf16_ref(d_x, d_w, d_b, d_yref, m, n, k);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<float> y(static_cast<std::size_t>(m) * n), yref(y.size());
            CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(yref.data(), d_yref, yref.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double num = 0, den = 0;
            float max_err = 0.0f;
            for (std::size_t i = 0; i < y.size(); ++i) {
                const double e = static_cast<double>(y[i]) - yref[i];
                num += e * e;
                den += static_cast<double>(yref[i]) * yref[i];
                max_err = std::max(max_err, static_cast<float>(std::fabs(e)));
            }
            const float rel = static_cast<float>(std::sqrt(num / (den + 1e-30)));
            worst_rel = std::max(worst_rel, rel);
            std::printf("bf16 TC GEMM: [%d,%d]x[%d,%d] max_err=%.5f rel_l2=%.5f\n", m, n, k, k,
                        max_err, rel);
            cudaFree(d_x);
            cudaFree(d_w);
            cudaFree(d_b);
            cudaFree(d_y);
            cudaFree(d_yref);
        }
        std::printf("bf16 TC GEMM worst rel_l2 vs ref=%.5f %s\n", worst_rel,
                    worst_rel <= 0.02f ? "OK" : "FAIL");
        if (worst_rel > 0.02f) ++failures;
    }

    // ---- full device decoder vs CPU MoEDecoder (tiny synthetic config) ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 4;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 8;
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        DecoderWeights w = DecoderWeights::random(cfg, 99);
        MoEDecoder cpu(cfg, w);
        cuda::GpuDecoder gpu(cfg, w);

        std::vector<int> prompt(6);
        for (int i = 0; i < 6; ++i) prompt[i] = i + 1;

        RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                        cfg.sliding_window);
        std::vector<float> cpu_logits, gpu_logits;
        cpu.prefill(cache, prompt, 0, cpu_logits);
        gpu.prefill_tokens(prompt, gpu_logits);

        auto diff = [](const std::vector<float>& a, const std::vector<float>& b) {
            double num = 0, den = 0, mx = 0;
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
                mx = std::max(mx, e);
                num += e * e;
                den += static_cast<double>(b[i]) * b[i];
            }
            return std::pair<double, double>(mx, std::sqrt(num / (den + 1e-30)));
        };
        auto [p_mx, p_rel] = diff(gpu_logits, cpu_logits);
        std::printf("GpuDecoder prefill: vocab=%zu max_abs=%.5f rel_l2=%.5f\n", cpu_logits.size(),
                    p_mx, p_rel);
        float worst = static_cast<float>(p_rel);
        if (p_rel > 0.05) {
            std::printf("  FAIL\n");
            ++failures;
        } else {
            std::printf("  OK\n");
        }

        int pos = static_cast<int>(prompt.size());
        for (int step = 0; step < 6; ++step) {
            int tok = 0;
            float best = -1e30f;
            for (std::size_t i = 0; i < cpu_logits.size(); ++i)
                if (cpu_logits[i] > best) { best = cpu_logits[i]; tok = static_cast<int>(i); }
            cpu.decode(cache, tok, pos, cpu_logits);
            gpu.decode_token(tok, pos, gpu_logits);
            ++pos;
            auto [d_mx, d_rel] = diff(gpu_logits, cpu_logits);
            worst = std::max(worst, static_cast<float>(d_rel));
            if (d_rel > 0.05) ++failures;
        }
        std::printf("GpuDecoder decode (6 steps): worst rel_l2=%.5f %s\n", worst,
                    worst <= 0.05 ? "OK" : "FAIL");
    }

    // ---- ragged multi-request prefill vs per-request prefill ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 8;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 8;
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        DecoderWeights w = DecoderWeights::random(cfg, 808);
        cuda::GpuDecoder gpu(cfg, w);
        std::vector<std::vector<int>> prompts = {{1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10, 11}};
        std::vector<int> slots = {2, 0, 1};
        gpu.batch_configure(3, 16);  // capacity >= max prefill + window

        const int h = cfg.hidden_size;
        std::vector<float> embeds;
        std::vector<int> starts, lengths;
        for (std::size_t r = 0; r < prompts.size(); ++r) {
            starts.push_back(static_cast<int>(embeds.size()) / h);
            lengths.push_back(static_cast<int>(prompts[r].size()));
            embeds.resize(embeds.size() + prompts[r].size() * h);
            float* dst = embeds.data() + static_cast<std::size_t>(starts.back()) * h;
            for (std::size_t t = 0; t < prompts[r].size(); ++t)
                w.embed_tokens.row(prompts[r][t], dst + t * h);
        }
        std::vector<std::vector<float>> ragged;
        gpu.batch_prefill_embeds(embeds.data(), starts, lengths, slots, ragged);

        auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                const double e = static_cast<double>(a[i]) - b[i];
                num += e * e;
                den += static_cast<double>(b[i]) * b[i];
            }
            return std::sqrt(num / (den + 1e-30));
        };
        float worst = 0.0f;
        for (std::size_t r = 0; r < prompts.size(); ++r) {
            std::vector<float> ref;
            gpu.prefill_tokens(prompts[r], ref);
            worst = std::max(worst, static_cast<float>(rel(ragged[r], ref)));
        }
        std::printf("Ragged prefill (%zu requests): worst rel_l2 vs per-request=%.5f %s\n",
                    prompts.size(), worst, worst <= 0.05f ? "OK" : "FAIL");
        if (worst > 0.05f) ++failures;
    }

    // ---- batched decode CUDA Graph vs plain (slot permutation + ring wrap) ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 8;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 8;  // small: ring overwrites during the run
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        DecoderWeights w = DecoderWeights::random(cfg, 4242);
        const int B = 3;
        const int h = cfg.hidden_size;
        std::vector<std::vector<int>> prompts = {{1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10, 11, 12, 13, 14}};
        const std::vector<int> slot_map = {3, 1, 2};  // non-identity permutation
        const int steps = 14;

        // Runs `steps` batched decode steps with a fixed token/position script
        // (independent of sampling) so the graph and plain runs are comparable.
        auto run = [&](bool use_graph, std::vector<std::vector<float>>& last,
                       int& captured) {
            cuda::GpuDecoder gpu(cfg, w);
            gpu.set_use_graph(use_graph);
            gpu.batch_configure(4, 32);
            std::vector<float> embeds;
            std::vector<int> starts, lengths;
            for (int r = 0; r < B; ++r) {
                starts.push_back(static_cast<int>(embeds.size()) / h);
                lengths.push_back(static_cast<int>(prompts[r].size()));
                embeds.resize(embeds.size() + static_cast<std::size_t>(prompts[r].size()) * h);
                float* dst = embeds.data() + static_cast<std::size_t>(starts.back()) * h;
                for (std::size_t t = 0; t < prompts[r].size(); ++t)
                    w.embed_tokens.row(prompts[r][t], dst + static_cast<std::size_t>(t) * h);
            }
            std::vector<std::vector<float>> pf;
            gpu.batch_prefill_embeds(embeds.data(), starts, lengths, slot_map, pf);

            for (int s = 0; s < steps; ++s) {
                std::vector<int> toks(B), poss(B);
                for (int b = 0; b < B; ++b) {
                    toks[b] = (s * 31 + b * 7 + 1) % cfg.vocab_size;
                    poss[b] = lengths[b] + s;  // next position for this slot
                }
                gpu.batch_decode(toks, poss, slot_map, last);
            }
            captured = gpu.batch_graph_count();
        };

        std::vector<std::vector<float>> plain, graph;
        int plain_captured = -1, graph_captured = -1;
        run(false, plain, plain_captured);
        run(true, graph, graph_captured);

        auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                const double e = static_cast<double>(a[i]) - b[i];
                num += e * e;
                den += static_cast<double>(b[i]) * b[i];
            }
            return std::sqrt(num / (den + 1e-30));
        };
        float worst = 0.0f;
        for (int b = 0; b < B; ++b)
            worst = std::max(worst, static_cast<float>(rel(graph[b], plain[b])));
        const bool ok = worst <= 1e-4f && graph_captured >= 1 && plain_captured == 0;
        std::printf("Batched decode graph vs plain: rel_l2=%.7f captured(plain=%d graph=%d) %s\n",
                    worst, plain_captured, graph_captured, ok ? "OK" : "FAIL");
        if (!ok) ++failures;
    }

    // ---- Engine CUDA branch vs CPU branch (same tiny model) ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 4;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 8;
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        EngineConfig ecfg;
        ecfg.memory_pool_bytes = 1 << 20;
        ecfg.max_seq_len = 64;
        ecfg.use_int4_experts = false;
        ecfg.no_repeat_ngram_size = 0;

        DecoderWeights w = DecoderWeights::random(cfg, 7);
        Engine cpu(cfg, ecfg, w, Backend::CPU);
        Engine gpu(cfg, ecfg, w, Backend::CUDA);
        std::vector<int> prompt = {1, 2, 3, 4, 5};
        auto rc = cpu.generate(prompt, 4);
        auto rg = gpu.generate(prompt, 4);
        bool same = rc.tokens == rg.tokens;
        std::printf("Engine CUDA greedy: cpu=[");
        for (int t : rc.tokens) std::printf("%d ", t);
        std::printf("] gpu=[");
        for (int t : rg.tokens) std::printf("%d ", t);
        std::printf("] %s\n", same ? "OK" : "DIFF");
        if (!same) ++failures;
    }

    // ---- CUDA Graph decode vs non-graph decode across the ring overwrite ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 4;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 4;  // tiny window: ring wraps quickly
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        DecoderWeights w = DecoderWeights::random(cfg, 2024);
        MoEDecoder cpu(cfg, w);
        cuda::GpuDecoder g_plain(cfg, w);
        cuda::GpuDecoder g_graph(cfg, w);
        cuda::GpuDecoder g_attn(cfg, w);
        g_plain.set_use_graph(false);
        g_graph.set_use_graph(true);
        g_attn.set_use_graph(true);
        g_attn.set_graph_scope(cuda::GpuDecoder::GraphScope::kAttnDense);

        std::vector<int> prompt(6);
        for (int i = 0; i < 6; ++i) prompt[i] = i + 1;

        RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                        cfg.sliding_window);
        std::vector<float> cpu_logits, p_logits, q_logits, a_logits;
        cpu.prefill(cache, prompt, 0, cpu_logits);
        g_plain.prefill_tokens(prompt, p_logits);
        g_graph.prefill_tokens(prompt, q_logits);
        g_attn.prefill_tokens(prompt, a_logits);

        auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                const double e = static_cast<double>(a[i]) - b[i];
                num += e * e;
                den += static_cast<double>(b[i]) * b[i];
            }
            return std::sqrt(num / (den + 1e-30));
        };

        int pos = static_cast<int>(prompt.size());
        float worst = 0.0f, worst_cpu = 0.0f, worst_attn = 0.0f;
        const int steps = 20;  // P+W = 10: warmup, ring fill and wrap
        for (int step = 0; step < steps; ++step) {
            int tok = 0;
            float best = -1e30f;
            for (std::size_t i = 0; i < cpu_logits.size(); ++i)
                if (cpu_logits[i] > best) { best = cpu_logits[i]; tok = static_cast<int>(i); }
            cpu.decode(cache, tok, pos, cpu_logits);
            g_plain.decode_token(tok, pos, p_logits);
            g_graph.decode_token(tok, pos, q_logits);
            g_attn.decode_token(tok, pos, a_logits);
            ++pos;
            const float gd = static_cast<float>(rel(q_logits, p_logits));
            const float cd = static_cast<float>(rel(q_logits, cpu_logits));
            const float ad = static_cast<float>(rel(a_logits, p_logits));
            worst = std::max(worst, gd);
            worst_cpu = std::max(worst_cpu, cd);
            worst_attn = std::max(worst_attn, ad);
            if (gd > 0.05f || cd > 0.05f || ad > 0.05f) ++failures;
        }
        std::printf("GpuDecoder graph decode (%d steps, W=%d): graph_vs_plain rel_l2=%.5f "
                    "graph_vs_cpu rel_l2=%.5f attn_dense_vs_plain rel_l2=%.5f %s\n",
                    steps, cfg.sliding_window, worst, worst_cpu, worst_attn,
                    (worst <= 0.05f && worst_cpu <= 0.05f && worst_attn <= 0.05f) ? "OK" : "FAIL");
        if (!g_graph.graph_ready() || !g_attn.graph_ready()) {
            std::printf("  FAIL: graph was not captured\n");
            ++failures;
        }
    }

    // ---- device INT4 expert weights vs CPU (both plain and graph paths) ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 8;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 4;
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        DecoderWeights w = DecoderWeights::random(cfg, 4242);
        auto quantize_linear = [](Linear& lin, int group) {
            const int rows = lin.weight.rows, cols = lin.weight.cols;
            if (lin.weight.fmt != WeightFormat::F32_OWNED) return;
            lin.weight.q = quantize_int4_awq(lin.weight.f32.data(), rows, cols, group);
            lin.weight.fmt = WeightFormat::INT4;
            lin.weight.f32.clear();
        };
        for (auto& L : w.layers)
            if (L.is_moe)
                for (auto& ex : L.experts) {
                    quantize_linear(ex.gate, 32);
                    quantize_linear(ex.up, 32);
                    quantize_linear(ex.down, 32);
                }

        MoEDecoder cpu(cfg, w);
        cuda::GpuDecoder g_plain(cfg, w);
        cuda::GpuDecoder g_graph(cfg, w);
        g_plain.set_use_graph(false);
        g_graph.set_use_graph(true);

        std::vector<int> prompt(6);
        for (int i = 0; i < 6; ++i) prompt[i] = i + 1;

        RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                        cfg.sliding_window);
        std::vector<float> cpu_logits, p_logits, q_logits;
        cpu.prefill(cache, prompt, 0, cpu_logits);
        g_plain.prefill_tokens(prompt, p_logits);
        g_graph.prefill_tokens(prompt, q_logits);

        auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                const double e = static_cast<double>(a[i]) - b[i];
                num += e * e;
                den += static_cast<double>(b[i]) * b[i];
            }
            return std::sqrt(num / (den + 1e-30));
        };

        int pos = static_cast<int>(prompt.size());
        float worst_plain = 0.0f, worst_graph = 0.0f;
        for (int step = 0; step < 16; ++step) {
            int tok = 0;
            float best = -1e30f;
            for (std::size_t i = 0; i < cpu_logits.size(); ++i)
                if (cpu_logits[i] > best) { best = cpu_logits[i]; tok = static_cast<int>(i); }
            cpu.decode(cache, tok, pos, cpu_logits);
            g_plain.decode_token(tok, pos, p_logits);
            g_graph.decode_token(tok, pos, q_logits);
            ++pos;
            const float cp = static_cast<float>(rel(p_logits, cpu_logits));
            const float cq = static_cast<float>(rel(q_logits, cpu_logits));
            worst_plain = std::max(worst_plain, cp);
            worst_graph = std::max(worst_graph, cq);
            if (cp > 0.1f || cq > 0.1f) ++failures;
        }
        std::printf("GpuDecoder INT4 experts: plain_vs_cpu rel_l2=%.5f graph_vs_cpu rel_l2=%.5f %s\n",
                    worst_plain, worst_graph,
                    (worst_plain <= 0.1f && worst_graph <= 0.1f) ? "OK" : "FAIL");
    }

    // ---- continuous batching vs sequential device decode ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 8;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 8;
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        EngineConfig ecfg;
        ecfg.memory_pool_bytes = 1 << 20;
        ecfg.max_seq_len = 64;
        ecfg.max_batch_size = 4;
        ecfg.min_batch_size = 1;
        ecfg.use_int4_experts = false;
        ecfg.no_repeat_ngram_size = 0;
        ecfg.use_cuda_graph = true;

        DecoderWeights w = DecoderWeights::random(cfg, 31337);
        std::vector<std::vector<int>> prompts = {
            {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}, {1, 2, 3}, {4, 5, 6, 7, 8, 9, 10}};

        Engine gpu_seq(cfg, ecfg, w, Backend::CUDA);
        Engine gpu_batch(cfg, ecfg, w, Backend::CUDA);
        Engine cpu(cfg, ecfg, w, Backend::CPU);

        std::vector<int> seq_flatten, batch_flatten, cpu_flatten;
        for (const auto& p : prompts) {
            auto rs = gpu_seq.generate(p, 6);
            for (int t : rs.tokens) seq_flatten.push_back(t);
            seq_flatten.push_back(-1);
        }
        auto rb = gpu_batch.generate_batch(prompts, 6);
        // Second call with the same shapes must reuse the per-slot KV
        // allocation and the captured batched graphs, producing identical
        // tokens (guards the `batch_configure` reuse path).
        auto rb2 = gpu_batch.generate_batch(prompts, 6);
        bool reuse_ok = rb2.size() == rb.size();
        for (std::size_t i = 0; i < rb.size() && reuse_ok; ++i)
            if (rb2[i].tokens != rb[i].tokens) reuse_ok = false;
        bool order_ok = true;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            if (rb[i].prefill_tokens != static_cast<int>(prompts[i].size())) order_ok = false;
            for (int t : rb[i].tokens) batch_flatten.push_back(t);
            batch_flatten.push_back(-1);
        }
        for (const auto& p : prompts) {
            auto rc = cpu.generate(p, 6);
            for (int t : rc.tokens) cpu_flatten.push_back(t);
            cpu_flatten.push_back(-1);
        }
        const bool same_seq = seq_flatten == batch_flatten;
        const bool same_cpu = cpu_flatten == batch_flatten;
        std::printf("Engine batch (%zu prompts):", prompts.size());
        for (std::size_t i = 0; i < rb.size(); ++i) {
            std::printf(" [");
            for (int t : rb[i].tokens) std::printf("%d ", t);
            std::printf("]");
        }
        std::printf("\n");
        std::printf("  batch_vs_sequential=%s batch_vs_cpu=%s reuse=%s order=%s %s\n",
                    same_seq ? "OK" : "DIFF", same_cpu ? "OK" : "DIFF", reuse_ok ? "OK" : "DIFF",
                    order_ok ? "OK" : "BAD",
                    (same_seq && same_cpu && order_ok && reuse_ok) ? "OK" : "FAIL");
        if (!same_seq || !order_ok || !reuse_ok) ++failures;
    }

    // ---- continuous batching with slot recycling (more requests than slots) ----
    {
        ModelConfig cfg;
        cfg.vocab_size = 256;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 128;
        cfg.moe_intermediate_size = 32;
        cfg.num_hidden_layers = 2;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.first_k_dense_replace = 1;
        cfg.n_routed_experts = 8;
        cfg.n_shared_experts = 1;
        cfg.num_experts_per_tok = 2;
        cfg.norm_topk_prob = true;
        cfg.sliding_window = 8;
        cfg.max_position_embeddings = 256;
        cfg.projector_input_dim = 2048;
        cfg.projector_n_embed = 64;

        EngineConfig ecfg;
        ecfg.memory_pool_bytes = 1 << 20;
        ecfg.max_seq_len = 64;
        ecfg.max_batch_size = 3;  // 6 requests -> slots are reused
        ecfg.min_batch_size = 1;
        ecfg.use_int4_experts = false;
        ecfg.no_repeat_ngram_size = 0;

        DecoderWeights w = DecoderWeights::random(cfg, 555);
        std::vector<std::vector<int>> prompts = {
            {1, 2, 3},       {4, 5, 6, 7, 8}, {9, 10},         {2, 4, 6, 8},
            {11, 12, 13, 14}, {1, 5, 9, 3, 7, 2}};

        Engine gpu_seq(cfg, ecfg, w, Backend::CUDA);
        Engine gpu_batch(cfg, ecfg, w, Backend::CUDA);
        Engine cpu(cfg, ecfg, w, Backend::CPU);

        auto rb = gpu_batch.generate_batch(prompts, 5);
        bool same_seq = true, same_cpu = true, order_ok = rb.size() == prompts.size();
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            auto rs = gpu_seq.generate(prompts[i], 5);
            auto rc = cpu.generate(prompts[i], 5);
            if (rb[i].tokens != rs.tokens) same_seq = false;
            if (rb[i].tokens != rc.tokens) same_cpu = false;
            if (rb[i].prefill_tokens != static_cast<int>(prompts[i].size())) order_ok = false;
        }
        std::printf("Engine batch recycling (%zu prompts, %d slots): seq=%s cpu=%s order=%s\n",
                    prompts.size(), ecfg.max_batch_size, same_seq ? "OK" : "DIFF",
                    same_cpu ? "OK" : "DIFF", order_ok ? "OK" : "BAD");
        if (!same_seq || !same_cpu || !order_ok) {
            std::printf("  FAIL\n");
            ++failures;
        } else {
            std::printf("  OK\n");
        }
    }

    std::printf("%s\n", failures == 0 ? "all CUDA tests passed" : "CUDA tests FAILED");
    return failures == 0 ? 0 : 1;
}
