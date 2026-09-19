// CUDA backend tests: R-SWA decode attention and INT4 MoE GEMM vs CPU reference.

#include <cuda_runtime.h>

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
        std::printf("  batch_vs_sequential=%s batch_vs_cpu=%s order=%s %s\n",
                    same_seq ? "OK" : "DIFF", same_cpu ? "OK" : "DIFF", order_ok ? "OK" : "BAD",
                    (same_seq && same_cpu && order_ok) ? "OK" : "FAIL");
        if (!same_seq || !order_ok) ++failures;
    }

    std::printf("%s\n", failures == 0 ? "all CUDA tests passed" : "CUDA tests FAILED");
    return failures == 0 ? 0 : 1;
}
