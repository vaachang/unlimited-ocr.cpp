#include "uocr/gpu_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "uocr/cuda_ops.h"

namespace uocr {
namespace cuda {

namespace {

void cu_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
}

std::vector<std::uint16_t> to_bf16(const WeightMatrix& m) {
    std::vector<float> f;
    if (m.fmt == WeightFormat::INT4) {
        m.to_f32(f);
    } else if (m.fmt == WeightFormat::F32_OWNED) {
        f = m.f32;
    } else if (m.fmt == WeightFormat::F32_EXT) {
        const float* p = static_cast<const float*>(m.ext);
        f.assign(p, p + static_cast<std::size_t>(m.rows) * m.cols);
    } else if (m.fmt == WeightFormat::BF16_EXT) {
        const std::uint16_t* p = static_cast<const std::uint16_t*>(m.ext);
        return std::vector<std::uint16_t>(p, p + static_cast<std::size_t>(m.rows) * m.cols);
    }
    std::vector<std::uint16_t> out(f.size());
    for (std::size_t i = 0; i < f.size(); ++i) out[i] = f32_to_bf16(f[i]);
    return out;
}

// Host softmax + greedy top-k, mirroring moe_detail::moe_gate.
void host_topk(const float* logits, int rows, int n_experts, const ModelConfig& cfg,
               std::vector<int>& ids, std::vector<float>& weights) {
    const int k = cfg.num_experts_per_tok;
    ids.assign(static_cast<std::size_t>(rows) * k, 0);
    weights.assign(static_cast<std::size_t>(rows) * k, 0.0f);
    std::vector<float> row(n_experts);
    for (int r = 0; r < rows; ++r) {
        const float* src = logits + static_cast<std::size_t>(r) * n_experts;
        if (cfg.scoring_func == "sigmoid") {
            for (int e = 0; e < n_experts; ++e) row[e] = 1.0f / (1.0f + std::exp(-src[e]));
        } else {
            float mx = -std::numeric_limits<float>::infinity();
            for (int e = 0; e < n_experts; ++e) mx = std::max(mx, src[e]);
            float sum = 0.0f;
            for (int e = 0; e < n_experts; ++e) {
                row[e] = std::exp(src[e] - mx);
                sum += row[e];
            }
            for (int e = 0; e < n_experts; ++e) row[e] /= sum;
        }
        for (int j = 0; j < k; ++j) {
            int best = -1;
            float bestv = -1e30f;
            for (int e = 0; e < n_experts; ++e)
                if (row[e] > bestv) { bestv = row[e]; best = e; }
            ids[static_cast<std::size_t>(r) * k + j] = best;
            weights[static_cast<std::size_t>(r) * k + j] = bestv;
            row[best] = -1e30f;
        }
        float* wrow = weights.data() + static_cast<std::size_t>(r) * k;
        if (k > 1 && cfg.norm_topk_prob) {
            float sum = 0.0f;
            for (int j = 0; j < k; ++j) sum += wrow[j];
            for (int j = 0; j < k; ++j) wrow[j] = wrow[j] / (sum + 1e-20f) * cfg.routed_scaling_factor;
        } else {
            for (int j = 0; j < k; ++j) wrow[j] *= cfg.routed_scaling_factor;
        }
    }
}

}  // namespace

GpuDecoder::GpuDecoder(ModelConfig cfg, const DecoderWeights& weights)
    : cfg_(std::move(cfg)), host_weights_(&weights) {
    const int h = cfg_.hidden_size;
    cache_.configure(cfg_.num_hidden_layers, cfg_.num_key_value_heads, cfg_.head_dim(),
                     cfg_.sliding_window);
    layers_.resize(cfg_.num_hidden_layers);
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        const LayerWeights& lw = weights.layers[li];
        DevLayer& dl = layers_[li];
        cu_check(cudaMalloc(&dl.in_ln, h * sizeof(float)), "in_ln");
        cu_check(cudaMemcpy(dl.in_ln, lw.input_layernorm.data(), h * sizeof(float),
                            cudaMemcpyHostToDevice), "in_ln copy");
        cu_check(cudaMalloc(&dl.post_ln, h * sizeof(float)), "post_ln");
        cu_check(cudaMemcpy(dl.post_ln, lw.post_attention_layernorm.data(), h * sizeof(float),
                            cudaMemcpyHostToDevice), "post_ln copy");
        upload_linear(lw.q_proj, dl.q);
        upload_linear(lw.k_proj, dl.k);
        upload_linear(lw.v_proj, dl.v);
        upload_linear(lw.o_proj, dl.o);
        dl.is_moe = lw.is_moe;
        if (!lw.is_moe) {
            upload_linear(lw.dense.gate, dl.dense_gate);
            upload_linear(lw.dense.up, dl.dense_up);
            upload_linear(lw.dense.down, dl.dense_down);
        } else {
            std::vector<std::uint16_t> router = to_bf16(lw.router);
            cu_check(cudaMalloc(&dl.router, router.size() * sizeof(std::uint16_t)), "router");
            cu_check(cudaMemcpy(dl.router, router.data(), router.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice), "router copy");
            dl.experts.resize(static_cast<std::size_t>(cfg_.n_routed_experts) * 3);
            for (int e = 0; e < cfg_.n_routed_experts; ++e) {
                upload_linear(lw.experts[e].gate, dl.experts[static_cast<std::size_t>(e) * 3 + 0]);
                upload_linear(lw.experts[e].up, dl.experts[static_cast<std::size_t>(e) * 3 + 1]);
                upload_linear(lw.experts[e].down, dl.experts[static_cast<std::size_t>(e) * 3 + 2]);
            }
            upload_linear(lw.shared.gate, dl.shared_gate);
            upload_linear(lw.shared.up, dl.shared_up);
            upload_linear(lw.shared.down, dl.shared_down);
        }
    }
    cu_check(cudaMalloc(&final_norm_, h * sizeof(float)), "final_norm");
    cu_check(cudaMemcpy(final_norm_, weights.final_norm.data(), h * sizeof(float),
                        cudaMemcpyHostToDevice), "final_norm copy");
}

GpuDecoder::~GpuDecoder() {
    auto f = [](void* p) { if (p) cudaFree(p); };
    for (DevLayer& L : layers_) {
        f(L.in_ln);
        f(L.post_ln);
        f(L.q.w);
        f(L.q.bias);
        f(L.k.w);
        f(L.k.bias);
        f(L.v.w);
        f(L.v.bias);
        f(L.o.w);
        f(L.o.bias);
        f(L.dense_gate.w);
        f(L.dense_gate.bias);
        f(L.dense_up.w);
        f(L.dense_up.bias);
        f(L.dense_down.w);
        f(L.dense_down.bias);
        f(L.router);
        for (DevLinear& d : L.experts) {
            f(d.w);
            f(d.bias);
        }
        f(L.shared_gate.w);
        f(L.shared_gate.bias);
        f(L.shared_up.w);
        f(L.shared_up.bias);
        f(L.shared_down.w);
        f(L.shared_down.bias);
    }
    f(final_norm_);
    f(scratch_);
    f(d_xin_);
    f(d_pos_);
    f(d_row_idx_);
    f(d_row_w_);
    f(d_ping_);
    f(d_pong_);
    f(d_normed_);
    f(d_hidden_);
}

void GpuDecoder::upload_linear(const Linear& src, DevLinear& dst) {
    if (src.weight.empty()) return;
    std::vector<std::uint16_t> w = to_bf16(src.weight);
    dst.rows = src.weight.rows;
    dst.cols = src.weight.cols;
    cu_check(cudaMalloc(&dst.w, w.size() * sizeof(std::uint16_t)), "linear w");
    cu_check(cudaMemcpy(dst.w, w.data(), w.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice), "linear w copy");
    if (src.has_bias && !src.bias.empty()) {
        cu_check(cudaMalloc(&dst.bias, src.bias.size() * sizeof(float)), "linear bias");
        cu_check(cudaMemcpy(dst.bias, src.bias.data(), src.bias.size() * sizeof(float),
                            cudaMemcpyHostToDevice), "linear bias copy");
    }
}

void GpuDecoder::ensure_scratch(int seq) {
    if (seq <= scratch_seq_ && scratch_ != nullptr) return;
    auto f = [](void* p) { if (p) cudaFree(p); };
    f(scratch_);
    f(d_xin_);
    f(d_pos_);
    f(d_row_idx_);
    f(d_row_w_);
    f(d_ping_);
    f(d_pong_);
    f(d_normed_);
    f(d_hidden_);
    scratch_ = nullptr;
    d_xin_ = nullptr;
    d_pos_ = nullptr;
    d_row_idx_ = nullptr;
    d_row_w_ = nullptr;
    d_ping_ = nullptr;
    d_pong_ = nullptr;
    d_normed_ = nullptr;
    d_hidden_ = nullptr;

    scratch_seq_ = seq;
    const int h = cfg_.hidden_size;
    const int ne = cfg_.n_routed_experts;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    const std::size_t per_seq = static_cast<std::size_t>(seq);
    const std::size_t total = per_seq * (13 * h + 4 * mi + ne) + 2 * per_seq * h;
    scratch_bytes_ = total * sizeof(float);
    cu_check(cudaMalloc(&scratch_, scratch_bytes_), "scratch");
    cu_check(cudaMalloc(&d_xin_, static_cast<std::size_t>(seq) * h * sizeof(float)), "d_xin");
    cu_check(cudaMalloc(&d_pos_, static_cast<std::size_t>(seq) * sizeof(int)), "d_pos");
    const int k = cfg_.num_experts_per_tok;
    cu_check(cudaMalloc(&d_row_idx_, static_cast<std::size_t>(seq) * k * sizeof(int)), "row_idx");
    cu_check(cudaMalloc(&d_row_w_, static_cast<std::size_t>(seq) * k * sizeof(float)), "row_w");
    cu_check(cudaMalloc(&d_ping_, static_cast<std::size_t>(seq) * h * sizeof(float)), "ping");
    cu_check(cudaMalloc(&d_pong_, static_cast<std::size_t>(seq) * h * sizeof(float)), "pong");
    cu_check(cudaMalloc(&d_normed_, static_cast<std::size_t>(h) * sizeof(float)), "normed");
    cu_check(cudaMalloc(&d_hidden_, static_cast<std::size_t>(seq) * h * sizeof(float)), "hidden");
}

void GpuDecoder::reset(int prefill_len) { cache_.reset(prefill_len); }

void GpuDecoder::layer_forward(int li, const float* x, int seq, const int* positions, bool prefill,
                               int q_start, float* out) {
    const int h = cfg_.hidden_size;
    const int heads = cfg_.num_attention_heads;
    const int kv_heads = cfg_.num_key_value_heads;
    const int hd = cfg_.head_dim();
    DevLayer& L = layers_[li];

    float* base = static_cast<float*>(scratch_);
    float* normed = base + 0;
    float* q = base + static_cast<std::size_t>(seq) * h * 1;
    float* k = base + static_cast<std::size_t>(seq) * h * 2;
    float* v = base + static_cast<std::size_t>(seq) * h * 3;
    float* ctx = base + static_cast<std::size_t>(seq) * h * 4;
    float* attn = base + static_cast<std::size_t>(seq) * h * 5;
    float* h1 = base + static_cast<std::size_t>(seq) * h * 6;
    float* normed2 = base + static_cast<std::size_t>(seq) * h * 7;
    float* mlp = base + static_cast<std::size_t>(seq) * h * 8;
    float* moe_out = base + static_cast<std::size_t>(seq) * h * 9;
    float* shared_out = base + static_cast<std::size_t>(seq) * h * 10;
    float* xg = base + static_cast<std::size_t>(seq) * h * 11;
    float* yg = base + static_cast<std::size_t>(seq) * h * 12;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    float* gate = base + static_cast<std::size_t>(seq) * (13 * h);
    float* up = gate + static_cast<std::size_t>(seq) * mi;
    float* act = up + static_cast<std::size_t>(seq) * mi;
    float* router = act + static_cast<std::size_t>(seq) * mi;

    cuda::rmsnorm(x, L.in_ln, normed, seq, h, cfg_.rms_norm_eps);
    cuda::matmul_t_bf16(normed, L.q.w, L.q.bias, q, seq, h, h);
    cuda::matmul_t_bf16(normed, L.k.w, L.k.bias, k, seq, h, h);
    cuda::matmul_t_bf16(normed, L.v.w, L.v.bias, v, seq, h, h);
    cuda::rope(q, k, positions, seq, heads, kv_heads, hd, cfg_.rope_theta);
    if (prefill)
        cache_.write_prefill(li, k, v, seq);
    else
        cache_.append_decode(li, k, v);
    cache_.attention(li, q, seq, q_start, ctx, heads, prefill);
    cuda::matmul_t_bf16(ctx, L.o.w, L.o.bias, attn, seq, h, h);

    cu_check(cudaMemcpyAsync(h1, x, static_cast<std::size_t>(seq) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice), "h1 copy");
    cuda::add_scaled(h1, attn, 1.0f, seq * h);
    cuda::rmsnorm(h1, L.post_ln, normed2, seq, h, cfg_.rms_norm_eps);

    if (!L.is_moe) {
        const int inter = cfg_.intermediate_size;
        cuda::matmul_t_bf16(normed2, L.dense_gate.w, L.dense_gate.bias, gate, seq, inter, h);
        cuda::matmul_t_bf16(normed2, L.dense_up.w, L.dense_up.bias, up, seq, inter, h);
        cuda::silu_mul(gate, up, act, seq * inter);
        cuda::matmul_t_bf16(act, L.dense_down.w, L.dense_down.bias, mlp, seq, h, inter);
        cu_check(cudaMemcpyAsync(out, h1, static_cast<std::size_t>(seq) * h * sizeof(float),
                                 cudaMemcpyDeviceToDevice), "out copy");
        cuda::add_scaled(out, mlp, 1.0f, seq * h);
        return;
    }

    const int ne = cfg_.n_routed_experts;
    const int kt = cfg_.num_experts_per_tok;
    const int inter = cfg_.moe_intermediate_size;
    cuda::matmul_t_bf16(normed2, L.router, nullptr, router, seq, ne, h);
    std::vector<float> rlog(static_cast<std::size_t>(seq) * ne);
    cu_check(cudaMemcpy(rlog.data(), router, rlog.size() * sizeof(float),
                        cudaMemcpyDeviceToHost), "router D2H");
    std::vector<int> ids;
    std::vector<float> ws;
    host_topk(rlog.data(), seq, ne, cfg_, ids, ws);

    cu_check(cudaMemsetAsync(moe_out, 0, static_cast<std::size_t>(seq) * h * sizeof(float)),
             "moe_out zero");

    std::vector<std::vector<int>> e_tokens(ne);
    std::vector<std::vector<float>> e_weights(ne);
    for (int t = 0; t < seq; ++t)
        for (int j = 0; j < kt; ++j) {
            const int e = ids[static_cast<std::size_t>(t) * kt + j];
            e_tokens[e].push_back(t);
            e_weights[e].push_back(ws[static_cast<std::size_t>(t) * kt + j]);
        }

    for (int e = 0; e < ne; ++e) {
        const int rows = static_cast<int>(e_tokens[e].size());
        if (rows == 0) continue;
        cu_check(cudaMemcpy(d_row_idx_, e_tokens[e].data(), rows * sizeof(int),
                            cudaMemcpyHostToDevice), "row_idx");
        cu_check(cudaMemcpy(d_row_w_, e_weights[e].data(), rows * sizeof(float),
                            cudaMemcpyHostToDevice), "row_w");
        cuda::gather_rows(xg, normed2, d_row_idx_, rows, h);
        DevLinear& g = L.experts[static_cast<std::size_t>(e) * 3 + 0];
        DevLinear& u = L.experts[static_cast<std::size_t>(e) * 3 + 1];
        DevLinear& d = L.experts[static_cast<std::size_t>(e) * 3 + 2];
        cuda::matmul_t_bf16(xg, g.w, g.bias, gate, rows, inter, h);
        cuda::matmul_t_bf16(xg, u.w, u.bias, up, rows, inter, h);
        cuda::silu_mul(gate, up, act, rows * inter);
        cuda::matmul_t_bf16(act, d.w, d.bias, yg, rows, h, inter);
        cuda::scatter_add_scaled(moe_out, yg, d_row_idx_, d_row_w_, rows, h);
    }

    const int sinter = cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1);
    cuda::matmul_t_bf16(normed2, L.shared_gate.w, L.shared_gate.bias, gate, seq, sinter, h);
    cuda::matmul_t_bf16(normed2, L.shared_up.w, L.shared_up.bias, up, seq, sinter, h);
    cuda::silu_mul(gate, up, act, seq * sinter);
    cuda::matmul_t_bf16(act, L.shared_down.w, L.shared_down.bias, shared_out, seq, h, sinter);

    cu_check(cudaMemcpyAsync(out, h1, static_cast<std::size_t>(seq) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice), "out copy");
    cuda::add_scaled(out, moe_out, 1.0f, seq * h);
    cuda::add_scaled(out, shared_out, 1.0f, seq * h);
}

void GpuDecoder::forward(const float* x_dev, int seq, const int* positions_dev, bool prefill,
                         int q_start, float* out_dev) {
    ensure_scratch(seq);
    const int h = cfg_.hidden_size;
    const float* cur = x_dev;
    float* next = d_ping_;
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        layer_forward(li, cur, seq, positions_dev, prefill, q_start, next);
        cur = next;
        next = (next == d_ping_) ? d_pong_ : d_ping_;
    }
    cu_check(cudaMemcpyAsync(out_dev, cur, static_cast<std::size_t>(seq) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice), "forward out");
}

void GpuDecoder::final_logits(const float* hidden_dev, int seq, std::vector<float>& logits) {
    const int h = cfg_.hidden_size;
    cuda::rmsnorm(hidden_dev + static_cast<std::size_t>(seq - 1) * h, final_norm_, d_normed_, 1, h,
                  cfg_.rms_norm_eps);
    std::vector<float> normed(h);
    cu_check(cudaMemcpy(normed.data(), d_normed_, h * sizeof(float), cudaMemcpyDeviceToHost),
             "normed D2H");
    logits.assign(cfg_.vocab_size, 0.0f);
    host_weights_->lm_head.matvec(normed.data(), logits.data());
}

void GpuDecoder::prefill_embeds(const float* host_embeds, int seq, std::vector<float>& logits) {
    cache_.reset(seq);
    ensure_scratch(seq);
    const int h = cfg_.hidden_size;
    cu_check(cudaMemcpy(d_xin_, host_embeds, static_cast<std::size_t>(seq) * h * sizeof(float),
                        cudaMemcpyHostToDevice), "embeds H2D");
    std::vector<int> pos(seq);
    for (int i = 0; i < seq; ++i) pos[i] = i;
    cu_check(cudaMemcpy(d_pos_, pos.data(), seq * sizeof(int), cudaMemcpyHostToDevice), "pos H2D");
    forward(d_xin_, seq, d_pos_, /*prefill=*/true, 0, d_hidden_);
    final_logits(d_hidden_, seq, logits);
}

void GpuDecoder::prefill_tokens(const std::vector<int>& tokens, std::vector<float>& logits) {
    const int seq = static_cast<int>(tokens.size());
    const int h = cfg_.hidden_size;
    std::vector<float> embeds(static_cast<std::size_t>(seq) * h);
    for (int t = 0; t < seq; ++t)
        host_weights_->embed_tokens.row(tokens[t], embeds.data() + static_cast<std::size_t>(t) * h);
    prefill_embeds(embeds.data(), seq, logits);
}

void GpuDecoder::decode_token(int token, int pos, std::vector<float>& logits) {
    const int h = cfg_.hidden_size;
    std::vector<float> embed(h);
    host_weights_->embed_tokens.row(token, embed.data());
    ensure_scratch(1);
    cu_check(cudaMemcpy(d_xin_, embed.data(), h * sizeof(float), cudaMemcpyHostToDevice), "embed H2D");
    cu_check(cudaMemcpy(d_pos_, &pos, sizeof(int), cudaMemcpyHostToDevice), "pos H2D");
    forward(d_xin_, 1, d_pos_, /*prefill=*/false, pos, d_hidden_);
    final_logits(d_hidden_, 1, logits);
}

}  // namespace cuda
}  // namespace uocr
