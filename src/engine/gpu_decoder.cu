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

// Dense projection dispatch: a single row (decode) uses the matvec kernel,
// which reads each weight once instead of the tiled GEMM's redundant reads.
inline void linear_forward(const float* x, const std::uint16_t* w, const float* bias, float* y,
                           int seq, int n, int k, cudaStream_t stream) {
    if (seq == 1)
        matvec_bf16(x, w, bias, y, n, k, stream);
    else
        matmul_t_bf16(x, w, bias, y, seq, n, k, stream);
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
    for (const DevLayer& L : layers_)
        if (L.is_moe) {
            int4_experts_ = L.int4_experts;
            break;
        }
    cu_check(cudaMalloc(&final_norm_, h * sizeof(float)), "final_norm");
    cu_check(cudaMemcpy(final_norm_, weights.final_norm.data(), h * sizeof(float),
                        cudaMemcpyHostToDevice), "final_norm copy");
    {
        Linear lh;
        lh.weight = weights.lm_head;
        upload_linear(lh, lm_head_);
    }
    cu_check(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float)),
             "d_logits");
    cu_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    cu_check(cudaMallocHost(&h_embed_pinned_, static_cast<std::size_t>(h) * sizeof(float)),
             "pinned embed");
    cu_check(cudaMallocHost(&h_pos_pinned_, sizeof(int)), "pinned pos");
    *h_pos_pinned_ = 0;
    for (int i = 0; i < h; ++i) h_embed_pinned_[i] = 0.0f;
    cu_check(cudaEventCreate(&ev_a_), "event a");
    cu_check(cudaEventCreate(&ev_b_), "event b");
    cu_check(cudaEventCreate(&ev_c_), "event c");
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
    f(lm_head_.w);
    f(lm_head_.bias);
    f(d_logits_);
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
    f(d_act_);
    for (float* p : batch_k_)
        if (p) cudaFree(p);
    for (float* p : batch_v_)
        if (p) cudaFree(p);
    f(d_batch_len_);
    f(d_batch_ring_);
    f(d_batch_prefill_);
    f(d_batch_slots_);
    f(d_batch_row_slot_);
    f(d_batch_last_idx_);
    f(d_logits_batch_);
    invalidate_batch_graphs();
    invalidate_graph();
    if (ev_a_) cudaEventDestroy(ev_a_);
    if (ev_b_) cudaEventDestroy(ev_b_);
    if (ev_c_) cudaEventDestroy(ev_c_);
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
    // The scratch base changes, so every captured batched graph (which baked
    // these addresses) must be discarded.
    invalidate_batch_graphs();
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
    cu_check(cudaMalloc(&d_normed_, static_cast<std::size_t>(seq) * h * sizeof(float)), "normed");
    cu_check(cudaMalloc(&d_hidden_, static_cast<std::size_t>(seq) * h * sizeof(float)), "hidden");
}

void GpuDecoder::ensure_router_scratch(int seq) {
    if (seq <= router_seq_ && d_count_ != nullptr) return;
    // `router_cap_` is baked into the captured MoE kernels, so a resize
    // invalidates the batched graphs as well.
    invalidate_batch_graphs();
    auto f = [](void* p) { if (p) cudaFree(p); };
    f(d_router_ids_);
    f(d_router_w_);
    f(d_assign_token_);
    f(d_assign_w_);
    f(d_count_);
    f(d_act_);
    d_router_ids_ = nullptr;
    d_router_w_ = nullptr;
    d_assign_token_ = nullptr;
    d_assign_w_ = nullptr;
    d_count_ = nullptr;
    d_act_ = nullptr;

    router_seq_ = seq;
    router_cap_ = seq < 1 ? 1 : seq;
    const int ne = cfg_.n_routed_experts;
    const int kt = cfg_.num_experts_per_tok;
    const int inter = cfg_.moe_intermediate_size;
    cu_check(cudaMalloc(&d_router_ids_, static_cast<std::size_t>(seq) * kt * sizeof(int)),
             "router ids");
    cu_check(cudaMalloc(&d_router_w_, static_cast<std::size_t>(seq) * kt * sizeof(float)),
             "router weights");
    cu_check(cudaMalloc(&d_assign_token_, static_cast<std::size_t>(ne) * router_cap_ * sizeof(int)),
             "assign token");
    cu_check(cudaMalloc(&d_assign_w_, static_cast<std::size_t>(ne) * router_cap_ * sizeof(float)),
             "assign weight");
    cu_check(cudaMalloc(&d_count_, static_cast<std::size_t>(ne) * sizeof(int)), "expert count");
    cu_check(cudaMalloc(&d_act_,
                        static_cast<std::size_t>(ne) * router_cap_ * inter * sizeof(float)),
             "expert act");
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
    linear_forward(s.normed, L.q.w, L.q.bias, s.q, seq, h, h, stream);
    linear_forward(s.normed, L.k.w, L.k.bias, s.k, seq, h, h, stream);
    linear_forward(s.normed, L.v.w, L.v.bias, s.v, seq, h, h, stream);
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
    linear_forward(s.ctx, L.o.w, L.o.bias, s.attn, seq, h, h, stream);

    cu_check(cudaMemcpyAsync(h1, x, static_cast<std::size_t>(seq) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "h1 copy");
    cuda::add_scaled(h1, s.attn, 1.0f, seq * h, stream);
    cuda::rmsnorm(h1, L.post_ln, normed2, seq, h, cfg_.rms_norm_eps, stream);
}

void GpuDecoder::mlp_block(int li, const float* h1, const float* normed2, int seq, bool dev_moe,
                           bool grouped, float* out, cudaStream_t stream) {
    const int h = cfg_.hidden_size;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    LayerScratch s = layer_scratch(scratch_, seq, h, mi);
    DevLayer& L = layers_[li];

    if (!L.is_moe) {
        const int inter = cfg_.intermediate_size;
        linear_forward(normed2, L.dense_gate.w, L.dense_gate.bias, s.gate, seq, inter, h, stream);
        linear_forward(normed2, L.dense_up.w, L.dense_up.bias, s.up, seq, inter, h, stream);
        cuda::silu_mul(s.gate, s.up, s.act, seq * inter, stream);
        linear_forward(s.act, L.dense_down.w, L.dense_down.bias, s.mlp, seq, h, inter, stream);
        cu_check(cudaMemcpyAsync(out, h1, static_cast<std::size_t>(seq) * h * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream), "out copy");
        cuda::add_scaled(out, s.mlp, 1.0f, seq * h, stream);
        return;
    }

    const int ne = cfg_.n_routed_experts;
    const int kt = cfg_.num_experts_per_tok;
    const int inter = cfg_.moe_intermediate_size;

    if (dev_moe) {
        linear_forward(normed2, L.router, nullptr, s.router, seq, ne, h, stream);
        cu_check(cudaMemsetAsync(d_count_, 0, static_cast<std::size_t>(ne) * sizeof(int), stream),
                 "count zero");
        cuda::moe_router_topk(s.router, seq, ne, kt, cfg_.norm_topk_prob,
                              cfg_.routed_scaling_factor, cfg_.scoring_func == "sigmoid",
                              d_router_ids_, d_router_w_, d_assign_token_, d_assign_w_, d_count_,
                              router_cap_, stream);
        cu_check(cudaMemsetAsync(s.moe_out, 0, static_cast<std::size_t>(seq) * h * sizeof(float),
                                 stream), "moe_out zero");
        if (grouped && L.int4_experts) {
            // One grouped launch per projection: every expert reads its token
            // list from the device grouping table, so there is no host loop and
            // no per-layer D2H/H2D.
            ++grouped_moe_calls_;
            cuda::moe_grouped_gate_up_int4(
                normed2, L.gate_i4.packed, L.gate_i4.scales, L.gate_i4.zeros, L.gate_i4.pstride,
                L.gate_i4.sstride, L.gate_i4.ng, L.up_i4.packed, L.up_i4.scales, L.up_i4.zeros,
                L.up_i4.pstride, L.up_i4.sstride, L.up_i4.ng, d_assign_token_, d_count_, ne,
                router_cap_, h, inter, L.gate_i4.group, d_act_, grouped_bn_, grouped_bm_, stream);
            cuda::moe_grouped_down_int4(
                d_act_, L.down_i4.packed, L.down_i4.scales, L.down_i4.zeros, L.down_i4.pstride,
                L.down_i4.sstride, L.down_i4.ng, d_assign_token_, d_assign_w_, d_count_, ne,
                router_cap_, h, inter, L.down_i4.group, s.moe_out, grouped_bn_, grouped_bm_,
                stream);
        } else if (L.int4_experts) {
            cuda::moe_experts_masked_int4(
                normed2, seq, L.gate_i4.packed, L.gate_i4.scales, L.gate_i4.zeros,
                L.gate_i4.pstride, L.gate_i4.sstride, L.gate_i4.ng, L.up_i4.packed,
                L.up_i4.scales, L.up_i4.zeros, L.up_i4.pstride, L.up_i4.sstride, L.up_i4.ng,
                L.down_i4.packed, L.down_i4.scales, L.down_i4.zeros, L.down_i4.pstride,
                L.down_i4.sstride, L.down_i4.ng, d_assign_token_, d_assign_w_, d_count_, ne,
                router_cap_, h, inter, L.gate_i4.group, d_act_, s.moe_out, stream);
        } else {
            cuda::moe_experts_masked(normed2, seq, L.gate_ptrs, L.up_ptrs, L.down_ptrs,
                                     d_assign_token_, d_assign_w_, d_count_, ne, router_cap_, h,
                                     inter, d_act_, s.moe_out, stream);
        }
    } else {
        linear_forward(normed2, L.router, nullptr, s.router, seq, ne, h, stream);
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
                linear_forward(s.xg, g.w, g.bias, s.gate, rows, inter, h, stream);
                linear_forward(s.xg, u.w, u.bias, s.up, rows, inter, h, stream);
                cuda::silu_mul(s.gate, s.up, s.act, rows * inter, stream);
                linear_forward(s.act, d.w, d.bias, s.yg, rows, h, inter, stream);
            }
            cuda::scatter_add_scaled(s.moe_out, s.yg, d_row_idx_, d_row_w_, rows, h, stream);
        }
    }

    const int sinter = cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1);
    linear_forward(normed2, L.shared_gate.w, L.shared_gate.bias, s.gate, seq, sinter, h, stream);
    linear_forward(normed2, L.shared_up.w, L.shared_up.bias, s.up, seq, sinter, h, stream);
    cuda::silu_mul(s.gate, s.up, s.act, seq * sinter, stream);
    linear_forward(s.act, L.shared_down.w, L.shared_down.bias, s.shared_out, seq, h, sinter,
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
    const bool dev_moe = (use_graph_ && !prefill) || (prefill && prefill_dev_moe_);
    attention_block(li, x, seq, positions, prefill, q_start, s.h1, s.normed2, stream);
    // Single-request prefill / decode keep the masked or host path; grouped is
    // only wired for the ragged multi-request prefill (forward_ragged).
    mlp_block(li, s.h1, s.normed2, seq, dev_moe, /*grouped=*/false, out, stream);
}

void GpuDecoder::forward(const float* x_dev, int seq, const int* positions_dev, bool prefill,
                         int q_start, float* out_dev, cudaStream_t stream) {
    ensure_scratch(seq);
    if (prefill && prefill_dev_moe_) ensure_router_scratch(seq);
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

void GpuDecoder::final_logits_from_normed(std::vector<float>& logits, cudaStream_t stream) {
    const int h = cfg_.hidden_size;
    const int V = cfg_.vocab_size;
    // Device lm_head keeps the 129k-vocab projection off the critical host path.
    cuda::matvec_bf16(d_normed_, lm_head_.w, lm_head_.bias, d_logits_, V, h, stream);
    // The graph stream is non-blocking, so the legacy-stream memcpy below does
    // not order against it; wait explicitly before the synchronous readback.
    if (stream != nullptr) cu_check(cudaStreamSynchronize(stream), "logits stream sync");
    logits.resize(V);
    cu_check(cudaMemcpy(logits.data(), d_logits_, static_cast<std::size_t>(V) * sizeof(float),
                        cudaMemcpyDeviceToHost), "logits D2H");
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

void GpuDecoder::invalidate_batch_graphs() {
    for (cudaGraphExec_t e : batch_graph_execs_)
        if (e) cudaGraphExecDestroy(e);
    for (cudaGraph_t g : batch_graphs_)
        if (g) cudaGraphDestroy(g);
    batch_graph_execs_.clear();
    batch_graphs_.clear();
}

int GpuDecoder::batch_graph_count() const {
    int n = 0;
    for (cudaGraphExec_t e : batch_graph_execs_)
        if (e) ++n;
    return n;
}

void GpuDecoder::capture_batch_graph(int batch) {
    UOCR_CHECK(batch > 0, "capture_batch_graph: invalid batch size");
    // Allocate every buffer the captured kernels address first: `ensure_*` may
    // invalidate existing graphs (and clear the vectors below), so it must run
    // before the resize/lookup.
    ensure_scratch(batch);
    ensure_router_scratch(batch);
    if (static_cast<int>(batch_graph_execs_.size()) < batch) {
        batch_graphs_.resize(batch, nullptr);
        batch_graph_execs_.resize(batch, nullptr);
    }
    if (batch_graph_execs_[batch - 1] != nullptr) return;

    const int h = cfg_.hidden_size;
    cudaGraph_t graph = nullptr;
    cu_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
             "begin batch capture");
    forward_batch(d_xin_, batch, d_pos_, d_batch_slots_, d_hidden_, stream_);
    cuda::rmsnorm(d_hidden_, final_norm_, d_normed_, batch, h, cfg_.rms_norm_eps, stream_);
    cu_check(cudaStreamEndCapture(stream_, &graph), "end batch capture");

    cudaGraphExec_t exec = nullptr;
    cu_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
             "instantiate batch graph");
    batch_graphs_[batch - 1] = graph;
    batch_graph_execs_[batch - 1] = exec;
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
            mlp_block(li, s.h1, s.normed2, 1, /*dev_moe=*/false, /*grouped=*/false, out,
                      stream_);
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
            mlp_block(li, s.h1, s.normed2, 1, /*dev_moe=*/true, /*grouped=*/false, out,
                      stream_);
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
    cu_check(cudaEventRecord(ev_a_, stream_), "ev a");
    cu_check(cudaGraphLaunch(graph_exec_, stream_), "graph launch");
    cu_check(cudaEventRecord(ev_b_, stream_), "ev b");
    cu_check(cudaStreamSynchronize(stream_), "graph sync");
    cu_check(cudaEventElapsedTime(&last_forward_ms_, ev_a_, ev_b_), "elapsed fwd");
    final_logits_from_normed(logits, stream_);
    cu_check(cudaEventRecord(ev_c_, stream_), "ev c");
    cu_check(cudaStreamSynchronize(stream_), "ev c sync");
    cu_check(cudaEventElapsedTime(&last_logits_ms_, ev_b_, ev_c_), "elapsed logits");
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

// ---------------------------------------------------------------------------
// Continuous batching
// ---------------------------------------------------------------------------

void GpuDecoder::batch_configure(int slots, int capacity) {
    // Reuse the current allocation (and any captured batched graphs) when the
    // shape is unchanged: only the per-slot state must be cleared.  Graphs are
    // keyed by batch size and read the slot map from device memory, so they
    // stay valid across requests as long as the buffers are not reallocated.
    const bool reuse = slots == batch_slots_ && capacity == batch_cap_ &&
                       !batch_k_.empty() && batch_k_[0] != nullptr && d_batch_len_ != nullptr;
    if (reuse) {
        const std::size_t per_slot =
            static_cast<std::size_t>(batch_cap_) * batch_stride_ * sizeof(float);
        for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
            cu_check(cudaMemset(batch_k_[l], 0, per_slot * slots), "batch k zero");
            cu_check(cudaMemset(batch_v_[l], 0, per_slot * slots), "batch v zero");
        }
        cu_check(cudaMemset(d_batch_len_, 0, cfg_.num_hidden_layers * slots * sizeof(int)),
                 "batch len zero");
        cu_check(cudaMemset(d_batch_ring_, 0, cfg_.num_hidden_layers * slots * sizeof(int)),
                 "batch ring zero");
        cu_check(cudaMemset(d_batch_prefill_, 0, slots * sizeof(int)), "batch prefill zero");
        slot_prefill_len_.assign(slots, 0);
        return;
    }
    // The per-slot KV buffers and the length/slot tables are (re)allocated, so
    // any captured batched-decode graph points at stale memory.
    invalidate_batch_graphs();
    for (float* p : batch_k_)
        if (p) cudaFree(p);
    for (float* p : batch_v_)
        if (p) cudaFree(p);
    if (d_batch_len_) { cudaFree(d_batch_len_); d_batch_len_ = nullptr; }
    if (d_batch_ring_) { cudaFree(d_batch_ring_); d_batch_ring_ = nullptr; }
    if (d_batch_prefill_) { cudaFree(d_batch_prefill_); d_batch_prefill_ = nullptr; }
    if (d_batch_slots_) { cudaFree(d_batch_slots_); d_batch_slots_ = nullptr; }
    if (d_batch_row_slot_) { cudaFree(d_batch_row_slot_); d_batch_row_slot_ = nullptr; }
    if (d_batch_last_idx_) { cudaFree(d_batch_last_idx_); d_batch_last_idx_ = nullptr; }
    batch_slots_cap_ = 0;
    ragged_slots_cap_ = 0;
    batch_slots_ = slots;
    batch_cap_ = capacity;
    batch_stride_ = cfg_.num_key_value_heads * cfg_.head_dim();
    const std::size_t per_slot =
        static_cast<std::size_t>(batch_cap_) * batch_stride_ * sizeof(float);
    batch_k_.assign(cfg_.num_hidden_layers, nullptr);
    batch_v_.assign(cfg_.num_hidden_layers, nullptr);
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        cu_check(cudaMalloc(&batch_k_[l], per_slot * slots), "batch k");
        cu_check(cudaMalloc(&batch_v_[l], per_slot * slots), "batch v");
        cu_check(cudaMemset(batch_k_[l], 0, per_slot * slots), "batch k zero");
        cu_check(cudaMemset(batch_v_[l], 0, per_slot * slots), "batch v zero");
    }
    cu_check(cudaMalloc(&d_batch_len_, cfg_.num_hidden_layers * slots * sizeof(int)), "batch len");
    cu_check(cudaMalloc(&d_batch_ring_, cfg_.num_hidden_layers * slots * sizeof(int)),
             "batch ring");
    cu_check(cudaMalloc(&d_batch_prefill_, slots * sizeof(int)), "batch prefill");
    cu_check(cudaMalloc(&d_batch_slots_, slots * sizeof(int)), "batch slots");
    cu_check(cudaMalloc(&d_batch_last_idx_, slots * sizeof(int)), "batch last idx");
    batch_slots_cap_ = slots;
    cu_check(cudaMemset(d_batch_len_, 0, cfg_.num_hidden_layers * slots * sizeof(int)),
             "batch len zero");
    cu_check(cudaMemset(d_batch_ring_, 0, cfg_.num_hidden_layers * slots * sizeof(int)),
             "batch ring zero");
    cu_check(cudaMemset(d_batch_prefill_, 0, slots * sizeof(int)), "batch prefill zero");
    slot_prefill_len_.assign(slots, 0);
}

void GpuDecoder::batch_import_prefill(int slot, int prefill_len) {
    UOCR_CHECK(slot < batch_slots_, "batch slot out of range");
    const int stride = batch_stride_;
    const std::size_t bytes = static_cast<std::size_t>(prefill_len) * stride * sizeof(float);
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        float* dst_k = batch_k_[l] + static_cast<std::size_t>(slot) * batch_cap_ * stride;
        float* dst_v = batch_v_[l] + static_cast<std::size_t>(slot) * batch_cap_ * stride;
        cu_check(cudaMemcpy(dst_k, cache_.keys(l), bytes, cudaMemcpyDeviceToDevice), "batch k imp");
        cu_check(cudaMemcpy(dst_v, cache_.values(l), bytes, cudaMemcpyDeviceToDevice),
                 "batch v imp");
        const int idx = l * batch_slots_ + slot;
        const int zero = 0;
        cu_check(cudaMemcpy(d_batch_len_ + idx, &prefill_len, sizeof(int),
                            cudaMemcpyHostToDevice), "batch len imp");
        cu_check(cudaMemcpy(d_batch_ring_ + idx, &zero, sizeof(int), cudaMemcpyHostToDevice),
                 "batch ring imp");
    }
    cu_check(cudaMemcpy(d_batch_prefill_ + slot, &prefill_len, sizeof(int),
                        cudaMemcpyHostToDevice),
             "batch prefill imp");
    slot_prefill_len_[slot] = prefill_len;
}

void GpuDecoder::attention_block_batch(int li, int batch, const float* x, const int* positions,
                                       const int* slots, float* h1, float* normed2,
                                       cudaStream_t stream) {
    const int h = cfg_.hidden_size;
    const int heads = cfg_.num_attention_heads;
    const int kv_heads = cfg_.num_key_value_heads;
    const int hd = cfg_.head_dim();
    DevLayer& L = layers_[li];
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    LayerScratch s = layer_scratch(scratch_, batch, h, mi);

    cuda::rmsnorm(x, L.in_ln, s.normed, batch, h, cfg_.rms_norm_eps, stream);
    linear_forward(s.normed, L.q.w, L.q.bias, s.q, batch, h, h, stream);
    linear_forward(s.normed, L.k.w, L.k.bias, s.k, batch, h, h, stream);
    linear_forward(s.normed, L.v.w, L.v.bias, s.v, batch, h, h, stream);
    cuda::rope(s.q, s.k, positions, batch, heads, kv_heads, hd, cfg_.rope_theta, stream);

    // One append + one attention launch for the whole batch (was O(batch)).
    const int len_off = li * batch_slots_;
    cuda::rswa_append_decode_batch(s.k, s.v, batch_k_[li], batch_v_[li], d_batch_len_ + len_off,
                                   d_batch_ring_ + len_off, d_batch_prefill_, slots, batch,
                                   batch_cap_, cfg_.sliding_window, kv_heads, hd, stream);
    cuda::rswa_attention_batch(s.q, batch_k_[li], batch_v_[li], d_batch_len_ + len_off, slots,
                               batch, batch_cap_, heads, kv_heads, hd, s.ctx, stream);
    linear_forward(s.ctx, L.o.w, L.o.bias, s.attn, batch, h, h, stream);

    cu_check(cudaMemcpyAsync(h1, x, static_cast<std::size_t>(batch) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "batch h1 copy");
    cuda::add_scaled(h1, s.attn, 1.0f, batch * h, stream);
    cuda::rmsnorm(h1, L.post_ln, normed2, batch, h, cfg_.rms_norm_eps, stream);
}

void GpuDecoder::forward_batch(const float* x, int batch, const int* positions, const int* slots,
                               float* out, cudaStream_t stream) {
    ensure_scratch(batch);
    ensure_router_scratch(batch);
    const int h = cfg_.hidden_size;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    const float* cur = x;
    float* next = d_ping_;
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        LayerScratch s = layer_scratch(scratch_, batch, h, mi);
        attention_block_batch(li, batch, cur, positions, slots, s.h1, s.normed2, stream);
        mlp_block(li, s.h1, s.normed2, batch, /*dev_moe=*/true, /*grouped=*/false, next, stream);
        cur = next;
        next = (next == d_ping_) ? d_pong_ : d_ping_;
    }
    cu_check(cudaMemcpyAsync(out, cur, static_cast<std::size_t>(batch) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "batch out copy");
}

void GpuDecoder::attention_block_ragged(int li, int total, const float* x, const int* positions,
                                        const int* slots, float* h1, float* normed2,
                                        cudaStream_t stream) {
    const int h = cfg_.hidden_size;
    const int heads = cfg_.num_attention_heads;
    const int kv_heads = cfg_.num_key_value_heads;
    const int hd = cfg_.head_dim();
    DevLayer& L = layers_[li];
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    LayerScratch s = layer_scratch(scratch_, total, h, mi);

    cuda::rmsnorm(x, L.in_ln, s.normed, total, h, cfg_.rms_norm_eps, stream);
    linear_forward(s.normed, L.q.w, L.q.bias, s.q, total, h, h, stream);
    linear_forward(s.normed, L.k.w, L.k.bias, s.k, total, h, h, stream);
    linear_forward(s.normed, L.v.w, L.v.bias, s.v, total, h, h, stream);
    cuda::rope(s.q, s.k, positions, total, heads, kv_heads, hd, cfg_.rope_theta, stream);
    cuda::rswa_write_prefill_ragged(s.k, s.v, batch_k_[li], batch_v_[li], slots, positions, total,
                                    batch_cap_, kv_heads, hd, stream);
    cuda::rswa_attention_ragged(s.q, batch_k_[li], batch_v_[li], slots, positions, total,
                                batch_cap_, heads, kv_heads, hd, s.ctx, stream);
    linear_forward(s.ctx, L.o.w, L.o.bias, s.attn, total, h, h, stream);

    cu_check(cudaMemcpyAsync(h1, x, static_cast<std::size_t>(total) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "ragged h1 copy");
    cuda::add_scaled(h1, s.attn, 1.0f, total * h, stream);
    cuda::rmsnorm(h1, L.post_ln, normed2, total, h, cfg_.rms_norm_eps, stream);
}

void GpuDecoder::forward_ragged(const float* x, int total, const int* positions, const int* slots,
                                float* out, cudaStream_t stream) {
    ensure_scratch(total);
    ensure_router_scratch(total);
    const int h = cfg_.hidden_size;
    const int mi = std::max(cfg_.intermediate_size,
                            cfg_.moe_intermediate_size * std::max(cfg_.n_shared_experts, 1));
    // With few rows per expert the masked matvec wins (weights are read once
    // per token but the kernel is well occupied); once several tokens share an
    // expert, the per-expert tensor-core GEMM amortises the weight read and
    // becomes much faster.  Crossover is around 2 tokens/expert.
    // The grouped INT4 kernel removes the per-layer router D2H and the
    // per-expert host loop, and (unlike the masked kernel) reads every expert
    // weight exactly once per row tile, so it wins at small *and* large totals.
    // It needs float4-addressable activations, hence the % 4 guards.
    const bool grouped =
        int4_experts_ && (h % 4 == 0) && (cfg_.moe_intermediate_size % 4 == 0);
    // Without the grouped path, fall back to the masked device kernel for small
    // batches and the host per-expert path for large ones.
    const bool dev_moe = grouped || total <= 2 * cfg_.n_routed_experts;
    const float* cur = x;
    float* next = d_ping_;
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        LayerScratch s = layer_scratch(scratch_, total, h, mi);
        attention_block_ragged(li, total, cur, positions, slots, s.h1, s.normed2, stream);
        mlp_block(li, s.h1, s.normed2, total, dev_moe, grouped, next, stream);
        cur = next;
        next = (next == d_ping_) ? d_pong_ : d_ping_;
    }
    cu_check(cudaMemcpyAsync(out, cur, static_cast<std::size_t>(total) * h * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream), "ragged out copy");
}

void GpuDecoder::batch_prefill_embeds(const float* host_embeds, const std::vector<int>& starts,
                                      const std::vector<int>& lengths,
                                      const std::vector<int>& slots,
                                      std::vector<std::vector<float>>& logits) {
    const int requests = static_cast<int>(lengths.size());
    UOCR_CHECK(requests > 0, "batch_prefill_embeds: no requests");
    UOCR_CHECK(static_cast<int>(starts.size()) == requests &&
                   static_cast<int>(slots.size()) == requests,
               "batch_prefill_embeds: array size mismatch");
    const int h = cfg_.hidden_size;
    const int V = cfg_.vocab_size;
    int total = 0;
    for (int r = 0; r < requests; ++r) {
        UOCR_CHECK(lengths[r] > 0, "batch_prefill_embeds: empty prompt");
        UOCR_CHECK(starts[r] == total, "batch_prefill_embeds: non-contiguous starts");
        UOCR_CHECK(slots[r] >= 0 && slots[r] < batch_slots_, "batch_prefill_embeds: slot out of range");
        total += lengths[r];
    }

    ensure_scratch(total);
    ensure_router_scratch(total);

    std::vector<int> pos(static_cast<std::size_t>(total));
    std::vector<int> row_slot(static_cast<std::size_t>(total));
    for (int r = 0; r < requests; ++r)
        for (int p = 0; p < lengths[r]; ++p) {
            pos[static_cast<std::size_t>(starts[r] + p)] = p;
            row_slot[static_cast<std::size_t>(starts[r] + p)] = slots[r];
        }
    if (total > ragged_slots_cap_) {
        if (d_batch_row_slot_) cudaFree(d_batch_row_slot_);
        cu_check(cudaMalloc(&d_batch_row_slot_, static_cast<std::size_t>(total) * sizeof(int)),
                 "ragged row slot");
        ragged_slots_cap_ = total;
    }
    cu_check(cudaMemcpy(d_xin_, host_embeds, static_cast<std::size_t>(total) * h * sizeof(float),
                        cudaMemcpyHostToDevice), "ragged embeds H2D");
    cu_check(cudaMemcpy(d_pos_, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice),
             "ragged pos H2D");
    cu_check(cudaMemcpy(d_batch_row_slot_, row_slot.data(), row_slot.size() * sizeof(int),
                        cudaMemcpyHostToDevice), "ragged slots H2D");

    forward_ragged(d_xin_, total, d_pos_, d_batch_row_slot_, d_hidden_, 0);

    // Publish the per-slot prefill state so subsequent batch_decode can run.
    std::vector<int> starts_dev(static_cast<std::size_t>(requests));
    for (int r = 0; r < requests; ++r) starts_dev[r] = starts[r] + lengths[r] - 1;
    cu_check(cudaMemcpy(d_batch_last_idx_, starts_dev.data(), starts_dev.size() * sizeof(int),
                        cudaMemcpyHostToDevice), "ragged last idx");
    for (int r = 0; r < requests; ++r) {
        const int slot = slots[r];
        const int prefill_len = lengths[r];
        const int zero = 0;
        for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
            const int idx = l * batch_slots_ + slot;
            cu_check(cudaMemcpy(d_batch_len_ + idx, &prefill_len, sizeof(int),
                                cudaMemcpyHostToDevice), "ragged len");
            cu_check(cudaMemcpy(d_batch_ring_ + idx, &zero, sizeof(int), cudaMemcpyHostToDevice),
                     "ragged ring");
        }
        cu_check(cudaMemcpy(d_batch_prefill_ + slot, &prefill_len, sizeof(int),
                            cudaMemcpyHostToDevice), "ragged prefill len");
        slot_prefill_len_[slot] = prefill_len;
    }

    // Last-token logits for each request.
    cu_check(cudaMemcpy(d_row_idx_, starts_dev.data(), starts_dev.size() * sizeof(int),
                        cudaMemcpyHostToDevice), "ragged last idx copy");
    cuda::gather_rows(d_ping_, d_hidden_, d_row_idx_, requests, h, 0);
    cuda::rmsnorm(d_ping_, final_norm_, d_normed_, requests, h, cfg_.rms_norm_eps, 0);
    if (requests > batch_logits_cap_) {
        if (d_logits_batch_) cudaFree(d_logits_batch_);
        cu_check(cudaMalloc(&d_logits_batch_,
                            static_cast<std::size_t>(requests) * V * sizeof(float)),
                 "ragged logits");
        batch_logits_cap_ = requests;
    }
    if (requests == 1)
        cuda::matvec_bf16(d_normed_, lm_head_.w, nullptr, d_logits_batch_, V, h, 0);
    else
        cuda::matmul_t_bf16(d_normed_, lm_head_.w, nullptr, d_logits_batch_, requests, V, h, 0);
    cu_check(cudaStreamSynchronize(0), "ragged sync");
    std::vector<float> buf(static_cast<std::size_t>(requests) * V);
    cu_check(cudaMemcpy(buf.data(), d_logits_batch_, buf.size() * sizeof(float),
                        cudaMemcpyDeviceToHost), "ragged logits D2H");
    logits.resize(requests);
    for (int r = 0; r < requests; ++r)
        logits[r].assign(buf.begin() + static_cast<std::size_t>(r) * V,
                         buf.begin() + static_cast<std::size_t>(r + 1) * V);
}

void GpuDecoder::batch_decode(const std::vector<int>& tokens, const std::vector<int>& positions,
                              const std::vector<int>& slots,
                              std::vector<std::vector<float>>& logits) {
    const int batch = static_cast<int>(tokens.size());
    UOCR_CHECK(batch > 0 && batch <= batch_slots_, "batch_decode size out of range");
    UOCR_CHECK(static_cast<int>(positions.size()) == batch, "positions size mismatch");
    UOCR_CHECK(static_cast<int>(slots.size()) == batch, "slots size mismatch");
    UOCR_CHECK(batch <= batch_slots_cap_, "batch slot scratch not configured");
    const int h = cfg_.hidden_size;
    const int V = cfg_.vocab_size;

    ensure_scratch(batch);
    ensure_router_scratch(batch);

    std::vector<float> embeds(static_cast<std::size_t>(batch) * h);
    for (int b = 0; b < batch; ++b)
        host_weights_->embed_tokens.row(tokens[b], embeds.data() + static_cast<std::size_t>(b) * h);
    cu_check(cudaMemcpy(d_xin_, embeds.data(), embeds.size() * sizeof(float),
                        cudaMemcpyHostToDevice), "batch embeds");
    cu_check(cudaMemcpy(d_pos_, positions.data(), batch * sizeof(int), cudaMemcpyHostToDevice),
             "batch pos");
    cu_check(cudaMemcpy(d_batch_slots_, slots.data(), batch * sizeof(int), cudaMemcpyHostToDevice),
             "batch slots");

    cu_check(cudaEventRecord(ev_a_, 0), "batch ev a");
    // A captured batched graph is valid for a fixed row count; the row -> slot
    // map, positions and embeddings are staged into persistent device buffers
    // just above and read at replay time, so the same graph serves any active
    // slot set.  The lm_head projection stays outside the graph (it needs a
    // logits buffer that may be reallocated).
    const bool graph_ok = use_graph_ && graph_scope_ == GraphScope::kFull;
    if (graph_ok) {
        capture_batch_graph(batch);
        cu_check(cudaGraphLaunch(batch_graph_execs_[batch - 1], 0), "batch graph launch");
    } else {
        forward_batch(d_xin_, batch, d_pos_, d_batch_slots_, d_hidden_, 0);
        cuda::rmsnorm(d_hidden_, final_norm_, d_normed_, batch, h, cfg_.rms_norm_eps, 0);
    }
    if (batch > batch_logits_cap_) {
        if (d_logits_batch_) cudaFree(d_logits_batch_);
        cu_check(cudaMalloc(&d_logits_batch_, static_cast<std::size_t>(batch) * V * sizeof(float)),
                 "batch logits");
        batch_logits_cap_ = batch;
    }
    if (batch == 1)
        cuda::matvec_bf16(d_normed_, lm_head_.w, nullptr, d_logits_batch_, V, h, 0);
    else
        cuda::matmul_t_bf16(d_normed_, lm_head_.w, nullptr, d_logits_batch_, batch, V, h, 0);
    cu_check(cudaEventRecord(ev_b_, 0), "batch ev b");
    cu_check(cudaStreamSynchronize(0), "batch sync");
    cu_check(cudaEventElapsedTime(&last_forward_ms_, ev_a_, ev_b_), "batch elapsed");

    std::vector<float> buf(static_cast<std::size_t>(batch) * V);
    cu_check(cudaMemcpy(buf.data(), d_logits_batch_, buf.size() * sizeof(float),
                        cudaMemcpyDeviceToHost), "batch logits D2H");
    logits.resize(batch);
    for (int b = 0; b < batch; ++b)
        logits[b].assign(buf.begin() + static_cast<std::size_t>(b) * V,
                         buf.begin() + static_cast<std::size_t>(b + 1) * V);
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
    cu_check(cudaEventRecord(ev_a_, 0), "ev a");
    forward(d_xin_, 1, d_pos_, /*prefill=*/false, pos, d_hidden_, 0);
    cu_check(cudaEventRecord(ev_b_, 0), "ev b");
    cu_check(cudaStreamSynchronize(0), "plain sync");
    cu_check(cudaEventElapsedTime(&last_forward_ms_, ev_a_, ev_b_), "elapsed fwd");
    final_logits(d_hidden_, 1, logits);
    cu_check(cudaEventRecord(ev_c_, 0), "ev c");
    cu_check(cudaStreamSynchronize(0), "plain logits sync");
    cu_check(cudaEventElapsedTime(&last_logits_ms_, ev_b_, ev_c_), "elapsed logits");
}

}  // namespace cuda
}  // namespace uocr
