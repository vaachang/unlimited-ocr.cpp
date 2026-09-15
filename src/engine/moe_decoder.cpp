#include "uocr/moe_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "uocr/log.h"
#include "uocr/ops.h"

namespace uocr {

MoEDecoder::MoEDecoder(ModelConfig cfg, DecoderWeights weights)
    : cfg_(std::move(cfg)), w_(std::move(weights)) {
    UOCR_CHECK(static_cast<int>(w_.layers.size()) == cfg_.num_hidden_layers,
               "decoder weight/layer count mismatch");
}

Tensor MoEDecoder::embed(const std::vector<int>& tokens) const {
    const int h = cfg_.hidden_size;
    Tensor out({static_cast<i64>(tokens.size()), h});
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        w_.embed_tokens.row(tokens[t], out.data() + t * h);
    }
    return out;
}

namespace {

void add_rows(float* a, const float* b, i64 n) {
    for (i64 i = 0; i < n; ++i) a[i] += b[i];
}

void rmsnorm_rows(const float* x, const float* w, float* out, int rows, int cols, float eps) {
    ops::rmsnorm(x, w, out, rows, cols, eps);
}

}  // namespace

void MoEDecoder::mlp_forward(const MLPWeights& mlp, const Tensor& x, Tensor& out) {
    const int seq = static_cast<int>(x.dim(0));
    const int hidden = cfg_.hidden_size;
    const int inter = mlp.gate.out_features();

    std::vector<float> gate(static_cast<std::size_t>(seq) * inter);
    std::vector<float> up(static_cast<std::size_t>(seq) * inter);
    std::vector<float> act(static_cast<std::size_t>(seq) * inter);
    mlp.gate.forward(x.data(), gate.data(), seq);
    mlp.up.forward(x.data(), up.data(), seq);
    ops::silu_mul(gate.data(), up.data(), act.data(), static_cast<i64>(seq) * inter);

    if (out.numel() != static_cast<i64>(seq) * hidden)
        out = Tensor({seq, hidden});
    mlp.down.forward(act.data(), out.data(), seq);
}

void MoEDecoder::routed_expert(const LayerWeights& L, const float* x, float* out,
                               int expert_idx) {
    const int hidden = cfg_.hidden_size;
    const int inter = cfg_.moe_intermediate_size;
    const ExpertWeights& e = L.experts[expert_idx];

    std::vector<float> gate(inter), up(inter), act(inter);
    e.gate.weight.matvec(x, gate.data());
    e.up.weight.matvec(x, up.data());
    ops::silu_mul(gate.data(), up.data(), act.data(), inter);
    e.down.weight.matvec(act.data(), out);
    (void)hidden;
}

void MoEDecoder::moe_forward(const LayerWeights& L, const Tensor& x, Tensor& out) {
    const int seq = static_cast<int>(x.dim(0));
    const int hidden = cfg_.hidden_size;
    out = Tensor({seq, hidden}, 0.0f);

    std::vector<int> ids;
    std::vector<float> weights;
    moe_detail::moe_gate(x.data(), L.router, cfg_, seq, ids, weights);

    const int k = cfg_.num_experts_per_tok;
    std::vector<float> acc(hidden);
    std::vector<RouterTrace> layer_traces(trace_router_ ? seq : 0);
    for (int t = 0; t < seq; ++t) {
        const float* xt = x.data() + static_cast<std::size_t>(t) * hidden;
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int j = 0; j < k; ++j) {
            const int e = ids[static_cast<std::size_t>(t) * k + j];
            const float wgt = weights[static_cast<std::size_t>(t) * k + j];
            routed_expert(L, xt, acc.data(), e);
            for (int d = 0; d < hidden; ++d)
                out.data()[static_cast<std::size_t>(t) * hidden + d] += wgt * acc[d];
        }
        if (trace_router_) {
            for (int j = 0; j < k; ++j) {
                layer_traces[static_cast<std::size_t>(t)].experts.push_back(
                    ids[static_cast<std::size_t>(t) * k + j]);
                layer_traces[static_cast<std::size_t>(t)].weights.push_back(
                    weights[static_cast<std::size_t>(t) * k + j]);
            }
        }
    }
    if (trace_router_) router_trace_.push_back(std::move(layer_traces));

    // shared experts
    Tensor shared;
    mlp_forward(L.shared, x, shared);
    add_rows(out.data(), shared.data(), out.numel());
}

void MoEDecoder::layer_forward(int layer_idx, const Tensor& x, const std::vector<int>& positions,
                               RSWACache& cache, bool prefill, int q_start, Tensor& out) {
    const LayerWeights& L = w_.layers[layer_idx];
    const int seq = static_cast<int>(x.dim(0));
    const int hidden = cfg_.hidden_size;
    const int heads = cfg_.num_attention_heads;
    const int kv_heads = cfg_.num_key_value_heads;
    const int hd = cfg_.head_dim();

    // ---- attention block ----
    Tensor normed({seq, hidden});
    rmsnorm_rows(x.data(), L.input_layernorm.data(), normed.data(), seq, hidden,
                 cfg_.rms_norm_eps);

    std::vector<float> q(static_cast<std::size_t>(seq) * hidden);
    std::vector<float> k(static_cast<std::size_t>(seq) * hidden);
    std::vector<float> v(static_cast<std::size_t>(seq) * hidden);
    L.q_proj.forward(normed.data(), q.data(), seq);
    L.k_proj.forward(normed.data(), k.data(), seq);
    L.v_proj.forward(normed.data(), v.data(), seq);

    ops::rope(q.data(), k.data(), positions.data(), seq, heads, kv_heads, hd, cfg_.rope_theta);

    if (prefill) {
        cache.write_prefill(layer_idx, k.data(), v.data(), seq);
    } else {
        cache.append_decode(layer_idx, k.data(), v.data());
    }

    std::vector<float> ctx(static_cast<std::size_t>(seq) * hidden);
    cache.attention(layer_idx, q.data(), seq, q_start, ctx.data(), heads, prefill);

    Tensor attn_out({seq, hidden});
    L.o_proj.forward(ctx.data(), attn_out.data(), seq);

    Tensor h1({seq, hidden});
    for (i64 i = 0; i < h1.numel(); ++i) h1[i] = x[i] + attn_out[i];

    // ---- MLP / MoE block ----
    Tensor normed2({seq, hidden});
    rmsnorm_rows(h1.data(), L.post_attention_layernorm.data(), normed2.data(), seq, hidden,
                 cfg_.rms_norm_eps);

    Tensor mlp_out;
    if (L.is_moe)
        moe_forward(L, normed2, mlp_out);
    else
        mlp_forward(L.dense, normed2, mlp_out);

    out = Tensor({seq, hidden});
    for (i64 i = 0; i < out.numel(); ++i) out[i] = h1[i] + mlp_out[i];
}

void MoEDecoder::forward(const Tensor& inputs, const std::vector<int>& positions, RSWACache& cache,
                         bool prefill, int q_start, bool final_norm, std::vector<float>& logits,
                         std::vector<Tensor>* layer_outputs) {
    const int seq = static_cast<int>(inputs.dim(0));
    const int hidden = cfg_.hidden_size;
    UOCR_CHECK(static_cast<int>(positions.size()) == seq, "positions length mismatch");

    if (layer_outputs) layer_outputs->clear();
    Tensor h = inputs;
    Tensor next;
    for (int li = 0; li < cfg_.num_hidden_layers; ++li) {
        layer_forward(li, h, positions, cache, prefill, q_start, next);
        h = std::move(next);
        if (layer_outputs) layer_outputs->push_back(h);
    }

    // final norm on the last row only
    std::vector<float> normed(static_cast<std::size_t>(hidden));
    const float* last = h.data() + static_cast<std::size_t>(seq - 1) * hidden;
    if (final_norm) {
        rmsnorm_rows(last, w_.final_norm.data(), normed.data(), 1, hidden, cfg_.rms_norm_eps);
    } else {
        std::memcpy(normed.data(), last, hidden * sizeof(float));
    }

    logits.assign(cfg_.vocab_size, 0.0f);
    w_.lm_head.matvec(normed.data(), logits.data());
}

void MoEDecoder::prefill(RSWACache& cache, const std::vector<int>& tokens, int start_pos,
                         std::vector<float>& logits) {
    const int seq = static_cast<int>(tokens.size());
    cache.reset(start_pos + seq);
    Tensor inputs = embed(tokens);
    std::vector<int> positions(seq);
    for (int i = 0; i < seq; ++i) positions[i] = start_pos + i;
    forward(inputs, positions, cache, true, start_pos, true, logits);
}

void MoEDecoder::decode(RSWACache& cache, int token, int pos, std::vector<float>& logits) {
    Tensor inputs = embed({token});
    std::vector<int> positions{pos};
    forward(inputs, positions, cache, false, pos, true, logits);
}

// ---------------------------------------------------------------------------
// Standalone helpers
// ---------------------------------------------------------------------------
namespace moe_detail {

void rmsnorm(const float* x, const float* w, float* out, int rows, int cols, float eps) {
    ops::rmsnorm(x, w, out, rows, cols, eps);
}

void moe_gate(const float* x, const WeightMatrix& router, const ModelConfig& cfg, int rows,
              std::vector<int>& expert_ids, std::vector<float>& expert_weights) {
    const int n_experts = cfg.n_routed_experts;
    const int k = cfg.num_experts_per_tok;
    expert_ids.assign(static_cast<std::size_t>(rows) * k, 0);
    expert_weights.assign(static_cast<std::size_t>(rows) * k, 0.0f);

    std::vector<float> logits(static_cast<std::size_t>(rows) * n_experts);
    // router weights are stored [n_experts, hidden]
    router.matmul(x, logits.data(), rows);

    for (int r = 0; r < rows; ++r) {
        float* row = logits.data() + static_cast<std::size_t>(r) * n_experts;
        if (cfg.scoring_func == "sigmoid") {
            for (int e = 0; e < n_experts; ++e) row[e] = 1.0f / (1.0f + std::exp(-row[e]));
        } else {
            ops::softmax(row, 1, n_experts);
        }

        // greedy top-k (order irrelevant for the weighted sum)
        for (int j = 0; j < k; ++j) {
            int best = -1;
            float bestv = -1e30f;
            for (int e = 0; e < n_experts; ++e) {
                if (row[e] > bestv) {
                    bestv = row[e];
                    best = e;
                }
            }
            expert_ids[static_cast<std::size_t>(r) * k + j] = best;
            expert_weights[static_cast<std::size_t>(r) * k + j] = bestv;
            row[best] = -1e30f;  // remove from selection
        }

        float* wrow = expert_weights.data() + static_cast<std::size_t>(r) * k;
        if (k > 1 && cfg.norm_topk_prob) {
            float sum = 0.0f;
            for (int j = 0; j < k; ++j) sum += wrow[j];
            const float denom = sum + 1e-20f;
            for (int j = 0; j < k; ++j) wrow[j] = wrow[j] / denom * cfg.routed_scaling_factor;
        } else {
            for (int j = 0; j < k; ++j) wrow[j] *= cfg.routed_scaling_factor;
        }
    }
}

}  // namespace moe_detail

}  // namespace uocr
