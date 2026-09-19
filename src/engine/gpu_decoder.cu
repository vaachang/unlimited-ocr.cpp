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

// Pointers into the shared layer scratch buffer.  The layout is identical for
// every layer so a per-layer captured graph can bake the same addresses.
struct LayerScratch {
    float *normed, *q, *k, *v, *ctx, *attn, *h1, *normed2;
    float *mlp, *moe_out, *shared_out, *xg, *yg, *gate, *up, *act, *router;
};

LayerScratch layer_scratch(void* base_v, int seq, int h, int mi) {
    float* base = static_cast<float*>(base_v);
    LayerScratch s;
    s.normed = base + 0;
    s.q = base + static_cast<std::size_t>(seq) * h * 1;
    s.k = base + static_cast<std::size_t>(seq) * h * 2;
    s.v = base + static_cast<std::size_t>(seq) * h * 3;
    s.ctx = base + static_cast<std::size_t>(seq) * h * 4;
    s.attn = base + static_cast<std::size_t>(seq) * h * 5;
    s.h1 = base + static_cast<std::size_t>(seq) * h * 6;
    s.normed2 = base + static_cast<std::size_t>(seq) * h * 7;
    s.mlp = base + static_cast<std::size_t>(seq) * h * 8;
    s.moe_out = base + static_cast<std::size_t>(seq) * h * 9;
    s.shared_out = base + static_cast<std::size_t>(seq) * h * 10;
    s.xg = base + static_cast<std::size_t>(seq) * h * 11;
    s.yg = base + static_cast<std::size_t>(seq) * h * 12;
    s.gate = base + static_cast<std::size_t>(seq) * (13 * h);
    s.up = s.gate + static_cast<std::size_t>(seq) * mi;
    s.act = s.up + static_cast<std::size_t>(seq) * mi;
    s.router = s.act + static_cast<std::size_t>(seq) * mi;
    return s;
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
            dl.int4_experts =
                !lw.experts.empty() && lw.experts[0].gate.weight.fmt == WeightFormat::INT4;
            if (dl.int4_experts) {
                upload_expert_table(lw.experts, 0, dl.gate_i4);
                upload_expert_table(lw.experts, 1, dl.up_i4);
                upload_expert_table(lw.experts, 2, dl.down_i4);
            } else {
                dl.experts.resize(static_cast<std::size_t>(cfg_.n_routed_experts) * 3);
                std::vector<const std::uint16_t*> gp(cfg_.n_routed_experts);
                std::vector<const std::uint16_t*> up(cfg_.n_routed_experts);
                std::vector<const std::uint16_t*> dp(cfg_.n_routed_experts);
                for (int e = 0; e < cfg_.n_routed_experts; ++e) {
                    upload_linear(lw.experts[e].gate,
                                  dl.experts[static_cast<std::size_t>(e) * 3 + 0]);
                    upload_linear(lw.experts[e].up,
                                  dl.experts[static_cast<std::size_t>(e) * 3 + 1]);
                    upload_linear(lw.experts[e].down,
                                  dl.experts[static_cast<std::size_t>(e) * 3 + 2]);
                    gp[e] = dl.experts[static_cast<std::size_t>(e) * 3 + 0].w;
                    up[e] = dl.experts[static_cast<std::size_t>(e) * 3 + 1].w;
                    dp[e] = dl.experts[static_cast<std::size_t>(e) * 3 + 2].w;
                }
                const std::size_t pbytes =
                    static_cast<std::size_t>(cfg_.n_routed_experts) * sizeof(std::uint16_t*);
                cu_check(cudaMalloc(reinterpret_cast<void**>(&dl.gate_ptrs), pbytes), "gate_ptrs");
                cu_check(cudaMalloc(reinterpret_cast<void**>(&dl.up_ptrs), pbytes), "up_ptrs");
                cu_check(cudaMalloc(reinterpret_cast<void**>(&dl.down_ptrs), pbytes), "down_ptrs");
                cu_check(cudaMemcpy(dl.gate_ptrs, gp.data(), pbytes, cudaMemcpyHostToDevice),
                         "gate_ptrs copy");
                cu_check(cudaMemcpy(dl.up_ptrs, up.data(), pbytes, cudaMemcpyHostToDevice),
                         "up_ptrs copy");
                cu_check(cudaMemcpy(dl.down_ptrs, dp.data(), pbytes, cudaMemcpyHostToDevice),
                         "down_ptrs copy");
            }
            upload_linear(lw.shared.gate, dl.shared_gate);
            upload_linear(lw.shared.up, dl.shared_up);
            upload_linear(lw.shared.down, dl.shared_down);
        }
    }
    cu_check(cudaMalloc(&final_norm_, h * sizeof(float)), "final_norm");
    cu_check(cudaMemcpy(final_norm_, weights.final_norm.data(), h * sizeof(float),
                        cudaMemcpyHostToDevice), "final_norm copy");
    cu_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    cu_check(cudaMallocHost(&h_embed_pinned_, static_cast<std::size_t>(h) * sizeof(float)),
             "pinned embed");
    cu_check(cudaMallocHost(&h_pos_pinned_, sizeof(int)), "pinned pos");
    *h_pos_pinned_ = 0;
    for (int i = 0; i < h; ++i) h_embed_pinned_[i] = 0.0f;
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
        f(const_cast<std::uint16_t**>(L.gate_ptrs));
        f(const_cast<std::uint16_t**>(L.up_ptrs));
        f(const_cast<std::uint16_t**>(L.down_ptrs));
        f(L.gate_i4.packed);
        f(L.gate_i4.scales);
        f(L.gate_i4.zeros);
        f(L.up_i4.packed);
        f(L.up_i4.scales);
        f(L.up_i4.zeros);
        f(L.down_i4.packed);
        f(L.down_i4.scales);
        f(L.down_i4.zeros);
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
    f(d_router_ids_);
    f(d_router_w_);
    f(d_assign_token_);
    f(d_assign_w_);
    f(d_count_);
    invalidate_graph();
    if (stream_) cudaStreamDestroy(stream_);
    if (h_embed_pinned_) cudaFreeHost(h_embed_pinned_);
    if (h_pos_pinned_) cudaFreeHost(h_pos_pinned_);
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

void GpuDecoder::upload_expert_table(const std::vector<ExpertWeights>& experts, int which,
                                     DevExpertTable& dst) {
    if (experts.empty()) return;
    auto select = [&](const ExpertWeights& x) -> const Linear& {
        return which == 0 ? x.gate : (which == 1 ? x.up : x.down);
    };
    const Linear& first = select(experts[0]);
    if (first.weight.fmt != WeightFormat::INT4) return;
    const int rows = first.weight.rows;
    const int cols = first.weight.cols;
    const int group = first.weight.q.group_size;
    const int gpr = (cols + 1) / 2;
    const int ng = (cols + group - 1) / group;
    const int ne = static_cast<int>(experts.size());

    dst.rows = rows;
    dst.cols = cols;
    dst.group = group;
    dst.ng = ng;
    dst.pstride = rows * gpr;
    dst.sstride = rows * ng;
    cu_check(cudaMalloc(&dst.packed, static_cast<std::size_t>(ne) * dst.pstride), "i4 packed");
    cu_check(cudaMalloc(&dst.scales, static_cast<std::size_t>(ne) * dst.sstride * sizeof(float)),
             "i4 scales");
    cu_check(cudaMalloc(&dst.zeros, static_cast<std::size_t>(ne) * dst.sstride * sizeof(float)),
             "i4 zeros");
    for (int e = 0; e < ne; ++e) {
        const QuantizedMatrix& q = select(experts[e]).weight.q;
        UOCR_CHECK(static_cast<int>(q.packed.size()) == dst.pstride,
                   "INT4 expert packed size mismatch");
        UOCR_CHECK(static_cast<int>(q.scales.size()) == dst.sstride,
                   "INT4 expert scale size mismatch");
        cu_check(cudaMemcpy(dst.packed + static_cast<std::size_t>(e) * dst.pstride, q.packed.data(),
                            q.packed.size(), cudaMemcpyHostToDevice),
                 "i4 packed copy");
        cu_check(cudaMemcpy(dst.scales + static_cast<std::size_t>(e) * dst.sstride, q.scales.data(),
                            q.scales.size() * sizeof(float), cudaMemcpyHostToDevice),
                 "i4 scales copy");
        cu_check(cudaMemcpy(dst.zeros + static_cast<std::size_t>(e) * dst.sstride, q.zeros.data(),
                            q.zeros.size() * sizeof(float), cudaMemcpyHostToDevice),
                 "i4 zeros copy");
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

void GpuDecoder::ensure_router_scratch(int seq) {
    if (seq <= router_seq_ && d_count_ != nullptr) return;
    auto f = [](void* p) { if (p) cudaFree(p); };
    f(d_router_ids_);
    f(d_router_w_);
    f(d_assign_token_);
    f(d_assign_w_);
    f(d_count_);
    d_router_ids_ = nullptr;
    d_router_w_ = nullptr;
    d_assign_token_ = nullptr;
    d_assign_w_ = nullptr;
    d_count_ = nullptr;

    router_seq_ = seq;
    router_cap_ = seq < 1 ? 1 : seq;
    const int ne = cfg_.n_routed_experts;
    const int kt = cfg_.num_experts_per_tok;
    cu_check(cudaMalloc(&d_router_ids_, static_cast<std::size_t>(seq) * kt * sizeof(int)),
             "router ids");
    cu_check(cudaMalloc(&d_router_w_, static_cast<std::size_t>(seq) * kt * sizeof(float)),
             "router weights");
    cu_check(cudaMalloc(&d_assign_token_, static_cast<std::size_t>(ne) * router_cap_ * sizeof(int)),
             "assign token");
    cu_check(cudaMalloc(&d_assign_w_, static_cast<std::size_t>(ne) * router_cap_ * sizeof(float)),
             "assign weight");
    cu_check(cudaMalloc(&d_count_, static_cast<std::size_t>(ne) * sizeof(int)), "expert count");
}

void GpuDecoder::reset(int prefill_len) {
    if (graph_ready_ && graph_prefill_len_ != prefill_len) invalidate_graph();
    cache_.reset(prefill_len);
}

void GpuDecoder::attention_block(int li, const float* x, int seq, const int* positions, bool prefill,
                                 int q_start, float* h1, float* normed2, cudaStream_t stream) {
    const int h = cfg_.hidden_size;
    const int heads = cfg_.num_attention_heads;
    const int kv_heads = cfg_.num_key_value_heads;
    const int hd = cfg_.head_dim();
    const bool dev_moe = use_graph_ && !prefill;
    DevLayer& L = layers_[li];
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    LayerScratch s = layer_scratch(scratch_, seq, h, mi);

    cuda::rmsnorm(x, L.in_ln, s.normed, seq, h, cfg_.rms_norm_eps, stream);
    cuda::matmul_t_bf16(s.normed, L.q.w, L.q.bias, s.q, seq, h, h, stream);
    cuda::matmul_t_bf16(s.normed, L.k.w, L.k.bias, s.k, seq, h, h, stream);
    cuda::matmul_t_bf16(s.normed, L.v.w, L.v.bias, s.v, seq, h, h, stream);
    cuda::rope(s.q, s.k, positions, seq, heads, kv_heads, hd, cfg_.rope_theta, stream);
    if (prefill) {
        cache_.write_prefill(li, s.k, s.v, seq, stream);
        cache_.attention(li, s.q, seq, q_start, s.ctx, heads, /*causal=*/true, stream);
    } else if (dev_moe) {
        cache_.append_decode_device(li, s.k, s.v, stream);
        cuda::rswa_attention_devlen(s.q, cache_.keys(li), cache_.values(li), cache_.len_dev(li),
                                    seq, q_start, heads, kv_heads, hd, /*causal=*/false, s.ctx,
                                    stream);
    } else {
        cache_.append_decode(li, s.k, s.v, stream);
        cache_.attention(li, s.q, seq, q_start, s.ctx, heads, /*causal=*/false, stream);
    }
    cuda::matmul_t_bf16(s.ctx, L.o.w, L.o.bias, s.attn, seq, h, h, stream);

    cu_check(cudaMemcpyAsync(h1, x, static_cast<std::size_t>(seq) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "h1 copy");
    cuda::add_scaled(h1, s.attn, 1.0f, seq * h, stream);
    cuda::rmsnorm(h1, L.post_ln, normed2, seq, h, cfg_.rms_norm_eps, stream);
}

void GpuDecoder::mlp_block(int li, const float* h1, const float* normed2, int seq, bool dev_moe,
                           float* out, cudaStream_t stream) {
    const int h = cfg_.hidden_size;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    LayerScratch s = layer_scratch(scratch_, seq, h, mi);
    DevLayer& L = layers_[li];

    if (!L.is_moe) {
        const int inter = cfg_.intermediate_size;
        cuda::matmul_t_bf16(normed2, L.dense_gate.w, L.dense_gate.bias, s.gate, seq, inter, h,
                            stream);
        cuda::matmul_t_bf16(normed2, L.dense_up.w, L.dense_up.bias, s.up, seq, inter, h, stream);
        cuda::silu_mul(s.gate, s.up, s.act, seq * inter, stream);
        cuda::matmul_t_bf16(s.act, L.dense_down.w, L.dense_down.bias, s.mlp, seq, h, inter, stream);
        cu_check(cudaMemcpyAsync(out, h1, static_cast<std::size_t>(seq) * h * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream), "out copy");
        cuda::add_scaled(out, s.mlp, 1.0f, seq * h, stream);
        return;
    }

    const int ne = cfg_.n_routed_experts;
    const int kt = cfg_.num_experts_per_tok;
    const int inter = cfg_.moe_intermediate_size;

    if (dev_moe) {
        cuda::matmul_t_bf16(normed2, L.router, nullptr, s.router, seq, ne, h, stream);
        cu_check(cudaMemsetAsync(d_count_, 0, static_cast<std::size_t>(ne) * sizeof(int), stream),
                 "count zero");
        cuda::moe_router_topk(s.router, seq, ne, kt, cfg_.norm_topk_prob,
                              cfg_.routed_scaling_factor, cfg_.scoring_func == "sigmoid",
                              d_router_ids_, d_router_w_, d_assign_token_, d_assign_w_, d_count_,
                              router_cap_, stream);
        cu_check(cudaMemsetAsync(s.moe_out, 0, static_cast<std::size_t>(seq) * h * sizeof(float),
                                 stream), "moe_out zero");
        if (L.int4_experts) {
            cuda::moe_experts_masked_int4(
                normed2, seq, L.gate_i4.packed, L.gate_i4.scales, L.gate_i4.zeros,
                L.gate_i4.pstride, L.gate_i4.sstride, L.gate_i4.ng, L.up_i4.packed,
                L.up_i4.scales, L.up_i4.zeros, L.up_i4.pstride, L.up_i4.sstride, L.up_i4.ng,
                L.down_i4.packed, L.down_i4.scales, L.down_i4.zeros, L.down_i4.pstride,
                L.down_i4.sstride, L.down_i4.ng, d_assign_token_, d_assign_w_, d_count_, ne,
                router_cap_, h, inter, L.gate_i4.group, s.moe_out, stream);
        } else {
            cuda::moe_experts_masked(normed2, seq, L.gate_ptrs, L.up_ptrs, L.down_ptrs,
                                     d_assign_token_, d_assign_w_, d_count_, ne, router_cap_, h,
                                     inter, s.moe_out, stream);
        }
    } else {
        cuda::matmul_t_bf16(normed2, L.router, nullptr, s.router, seq, ne, h, stream);
        std::vector<float> rlog(static_cast<std::size_t>(seq) * ne);
        cu_check(cudaMemcpy(rlog.data(), s.router, rlog.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "router D2H");
        std::vector<int> ids;
        std::vector<float> ws;
        host_topk(rlog.data(), seq, ne, cfg_, ids, ws);

        cu_check(cudaMemsetAsync(s.moe_out, 0, static_cast<std::size_t>(seq) * h * sizeof(float),
                                 stream), "moe_out zero");

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
            cuda::gather_rows(s.xg, normed2, d_row_idx_, rows, h, stream);
            if (L.int4_experts) {
                const std::size_t ep = static_cast<std::size_t>(e);
                cuda::moe_gemm_int4_tc(s.xg, L.gate_i4.packed + ep * L.gate_i4.pstride,
                                       L.gate_i4.scales + ep * L.gate_i4.sstride,
                                       L.gate_i4.zeros + ep * L.gate_i4.sstride, rows, inter, h,
                                       L.gate_i4.group, s.gate, stream);
                cuda::moe_gemm_int4_tc(s.xg, L.up_i4.packed + ep * L.up_i4.pstride,
                                       L.up_i4.scales + ep * L.up_i4.sstride,
                                       L.up_i4.zeros + ep * L.up_i4.sstride, rows, inter, h,
                                       L.up_i4.group, s.up, stream);
                cuda::silu_mul(s.gate, s.up, s.act, rows * inter, stream);
                cuda::moe_gemm_int4_tc(s.act, L.down_i4.packed + ep * L.down_i4.pstride,
                                       L.down_i4.scales + ep * L.down_i4.sstride,
                                       L.down_i4.zeros + ep * L.down_i4.sstride, rows, h, inter,
                                       L.down_i4.group, s.yg, stream);
            } else {
                DevLinear& g = L.experts[static_cast<std::size_t>(e) * 3 + 0];
                DevLinear& u = L.experts[static_cast<std::size_t>(e) * 3 + 1];
                DevLinear& d = L.experts[static_cast<std::size_t>(e) * 3 + 2];
                cuda::matmul_t_bf16(s.xg, g.w, g.bias, s.gate, rows, inter, h, stream);
                cuda::matmul_t_bf16(s.xg, u.w, u.bias, s.up, rows, inter, h, stream);
                cuda::silu_mul(s.gate, s.up, s.act, rows * inter, stream);
                cuda::matmul_t_bf16(s.act, d.w, d.bias, s.yg, rows, h, inter, stream);
            }
            cuda::scatter_add_scaled(s.moe_out, s.yg, d_row_idx_, d_row_w_, rows, h, stream);
        }
    }

    const int sinter = cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1);
    cuda::matmul_t_bf16(normed2, L.shared_gate.w, L.shared_gate.bias, s.gate, seq, sinter, h,
                        stream);
    cuda::matmul_t_bf16(normed2, L.shared_up.w, L.shared_up.bias, s.up, seq, sinter, h, stream);
    cuda::silu_mul(s.gate, s.up, s.act, seq * sinter, stream);
    cuda::matmul_t_bf16(s.act, L.shared_down.w, L.shared_down.bias, s.shared_out, seq, h, sinter,
                        stream);

    cu_check(cudaMemcpyAsync(out, h1, static_cast<std::size_t>(seq) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "out copy");
    cuda::add_scaled(out, s.moe_out, 1.0f, seq * h, stream);
    cuda::add_scaled(out, s.shared_out, 1.0f, seq * h, stream);
}

void GpuDecoder::layer_forward(int li, const float* x, int seq, const int* positions, bool prefill,
                               int q_start, float* out, cudaStream_t stream) {
    const int h = cfg_.hidden_size;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    LayerScratch s = layer_scratch(scratch_, seq, h, mi);
    // Decode steps run the device-resident MoE path (device router + fused
    // expert kernel) so that the whole step is CUDA-Graph capturable.
    const bool dev_moe = use_graph_ && !prefill;
    attention_block(li, x, seq, positions, prefill, q_start, s.h1, s.normed2, stream);
    mlp_block(li, s.h1, s.normed2, seq, dev_moe, out, stream);
}

void GpuDecoder::forward(const float* x_dev, int seq, const int* positions_dev, bool prefill,
                         int q_start, float* out_dev, cudaStream_t stream) {
    ensure_scratch(seq);
    const int h = cfg_.hidden_size;
    const float* cur = x_dev;
    float* next = d_ping_;
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        layer_forward(li, cur, seq, positions_dev, prefill, q_start, next, stream);
        cur = next;
        next = (next == d_ping_) ? d_pong_ : d_ping_;
    }
    cu_check(cudaMemcpyAsync(out_dev, cur, static_cast<std::size_t>(seq) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "forward out");
}

void GpuDecoder::final_logits_from_normed(std::vector<float>& logits) {
    const int h = cfg_.hidden_size;
    std::vector<float> normed(h);
    cu_check(cudaMemcpy(normed.data(), d_normed_, h * sizeof(float), cudaMemcpyDeviceToHost),
             "normed D2H");
    logits.assign(cfg_.vocab_size, 0.0f);
    host_weights_->lm_head.matvec(normed.data(), logits.data());
}

void GpuDecoder::final_logits(const float* hidden_dev, int seq, std::vector<float>& logits) {
    const int h = cfg_.hidden_size;
    cuda::rmsnorm(hidden_dev + static_cast<std::size_t>(seq - 1) * h, final_norm_, d_normed_, 1, h,
                  cfg_.rms_norm_eps);
    final_logits_from_normed(logits);
}

void GpuDecoder::invalidate_graph() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    for (cudaGraphExec_t e : attn_graph_execs_)
        if (e) cudaGraphExecDestroy(e);
    for (cudaGraph_t g : attn_graphs_)
        if (g) cudaGraphDestroy(g);
    attn_graph_execs_.clear();
    attn_graphs_.clear();
    graph_ready_ = false;
    graph_prefill_len_ = -1;
}

void GpuDecoder::capture_decode_graph() {
    const int h = cfg_.hidden_size;
    ensure_scratch(1);
    ensure_router_scratch(1);
    cu_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture");
    cu_check(cudaMemcpyAsync(d_xin_, h_embed_pinned_, static_cast<std::size_t>(h) * sizeof(float),
                             cudaMemcpyHostToDevice, stream_), "capture embed");
    cu_check(cudaMemcpyAsync(d_pos_, h_pos_pinned_, sizeof(int), cudaMemcpyHostToDevice, stream_),
             "capture pos");
    forward(d_xin_, 1, d_pos_, /*prefill=*/false, 0, d_hidden_, stream_);
    cuda::rmsnorm(d_hidden_, final_norm_, d_normed_, 1, h, cfg_.rms_norm_eps, stream_);
    cu_check(cudaStreamEndCapture(stream_, &graph_), "end capture");
    cu_check(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0), "instantiate graph");
    graph_ready_ = true;
    graph_prefill_len_ = cache_.prefill_len();
}

void GpuDecoder::capture_attn_dense_graphs() {
    const int h = cfg_.hidden_size;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    ensure_scratch(1);
    ensure_router_scratch(1);
    attn_graphs_.assign(cfg_.num_hidden_layers, nullptr);
    attn_graph_execs_.assign(cfg_.num_hidden_layers, nullptr);
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        // Layers alternate between the ping/pong buffers (layer 0 reads ping).
        const float* in = (li % 2 == 0) ? d_ping_ : d_pong_;
        float* out = (li % 2 == 0) ? d_pong_ : d_ping_;
        LayerScratch s = layer_scratch(scratch_, 1, h, mi);
        cu_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                 "begin attn capture");
        attention_block(li, in, 1, d_pos_, /*prefill=*/false, 0, s.h1, s.normed2, stream_);
        if (!layers_[li].is_moe)
            mlp_block(li, s.h1, s.normed2, 1, /*dev_moe=*/false, out, stream_);
        cu_check(cudaStreamEndCapture(stream_, &attn_graphs_[li]), "end attn capture");
        cu_check(cudaGraphInstantiate(&attn_graph_execs_[li], attn_graphs_[li], nullptr, nullptr, 0),
                 "instantiate attn graph");
    }
    graph_ready_ = true;
    graph_prefill_len_ = cache_.prefill_len();
}

void GpuDecoder::run_graph_decode_attn_dense(int token, int pos, std::vector<float>& logits) {
    const int h = cfg_.hidden_size;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    ensure_scratch(1);
    ensure_router_scratch(1);
    cu_check(cudaMemcpyAsync(d_ping_, h_embed_pinned_, static_cast<std::size_t>(h) * sizeof(float),
                             cudaMemcpyHostToDevice, stream_), "ad embed");
    cu_check(cudaMemcpyAsync(d_pos_, h_pos_pinned_, sizeof(int), cudaMemcpyHostToDevice, stream_),
             "ad pos");
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        cu_check(cudaGraphLaunch(attn_graph_execs_[li], stream_), "attn graph launch");
        if (layers_[li].is_moe) {
            // MoE is intentionally outside the captured graph.
            LayerScratch s = layer_scratch(scratch_, 1, h, mi);
            float* out = (li % 2 == 0) ? d_pong_ : d_ping_;
            mlp_block(li, s.h1, s.normed2, 1, /*dev_moe=*/true, out, stream_);
        }
    }
    const float* fin = (cfg_.num_hidden_layers % 2 == 0) ? d_ping_ : d_pong_;
    cu_check(cudaMemcpyAsync(d_hidden_, fin, static_cast<std::size_t>(h) * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream_), "ad out");
    cuda::rmsnorm(d_hidden_, final_norm_, d_normed_, 1, h, cfg_.rms_norm_eps, stream_);
    cu_check(cudaStreamSynchronize(stream_), "attn_dense sync");
    (void)token;
    final_logits_from_normed(logits);
}

void GpuDecoder::run_graph_decode(int token, int pos, std::vector<float>& logits) {
    host_weights_->embed_tokens.row(token, h_embed_pinned_);
    *h_pos_pinned_ = pos;
    if (!graph_ready_) {
        if (graph_scope_ == GraphScope::kAttnDense)
            capture_attn_dense_graphs();
        else
            capture_decode_graph();
    }
    if (graph_scope_ == GraphScope::kAttnDense) {
        run_graph_decode_attn_dense(token, pos, logits);
        return;
    }
    cu_check(cudaGraphLaunch(graph_exec_, stream_), "graph launch");
    cu_check(cudaStreamSynchronize(stream_), "graph sync");
    final_logits_from_normed(logits);
}

void GpuDecoder::prefill_embeds(const float* host_embeds, int seq, std::vector<float>& logits) {
    if (graph_ready_ && graph_prefill_len_ != seq) invalidate_graph();
    cache_.reset(seq);
    ensure_scratch(seq);
    const int h = cfg_.hidden_size;
    cu_check(cudaMemcpy(d_xin_, host_embeds, static_cast<std::size_t>(seq) * h * sizeof(float),
                        cudaMemcpyHostToDevice), "embeds H2D");
    std::vector<int> pos(seq);
    for (int i = 0; i < seq; ++i) pos[i] = i;
    cu_check(cudaMemcpy(d_pos_, pos.data(), seq * sizeof(int), cudaMemcpyHostToDevice), "pos H2D");
    forward(d_xin_, seq, d_pos_, /*prefill=*/true, 0, d_hidden_, 0);
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
    if (use_graph_) {
        run_graph_decode(token, pos, logits);
        return;
    }
    const int h = cfg_.hidden_size;
    std::vector<float> embed(h);
    host_weights_->embed_tokens.row(token, embed.data());
    ensure_scratch(1);
    cu_check(cudaMemcpy(d_xin_, embed.data(), h * sizeof(float), cudaMemcpyHostToDevice), "embed H2D");
    cu_check(cudaMemcpy(d_pos_, &pos, sizeof(int), cudaMemcpyHostToDevice), "pos H2D");
    forward(d_xin_, 1, d_pos_, /*prefill=*/false, pos, d_hidden_, 0);
    final_logits(d_hidden_, 1, logits);
}

}  // namespace cuda
}  // namespace uocr
