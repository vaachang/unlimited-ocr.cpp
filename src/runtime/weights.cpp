#include "uocr/weights.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "uocr/log.h"
#include "uocr/ops.h"

namespace uocr {

// ---------------------------------------------------------------------------
// WeightMatrix
// ---------------------------------------------------------------------------
void WeightMatrix::to_f32(std::vector<float>& out) const {
    out.resize(static_cast<std::size_t>(rows) * cols);
    switch (fmt) {
        case WeightFormat::F32_OWNED:
            std::copy(f32.begin(), f32.end(), out.begin());
            break;
        case WeightFormat::F32_EXT:
            std::memcpy(out.data(), ext, out.size() * sizeof(float));
            break;
        case WeightFormat::BF16_EXT: {
            const auto* p = static_cast<const std::uint16_t*>(ext);
            for (std::size_t i = 0; i < out.size(); ++i) out[i] = bf16_to_f32(p[i]);
            break;
        }
        case WeightFormat::INT4:
            q.dequantize(out);
            break;
    }
}

void WeightMatrix::matvec(const float* x, float* y) const { matmul(x, y, 1); }

void WeightMatrix::matmul(const float* x, float* y, int m) const {
    switch (fmt) {
        case WeightFormat::F32_OWNED:
            ops::matmul_t(x, f32.data(), nullptr, y, m, rows, cols);
            break;
        case WeightFormat::F32_EXT:
            ops::matmul_t(x, static_cast<const float*>(ext), nullptr, y, m, rows, cols);
            break;
        case WeightFormat::BF16_EXT:
            ops::matmul_t_bf16(x, static_cast<const std::uint16_t*>(ext), nullptr, y, m, rows, cols);
            break;
        case WeightFormat::INT4: {
            const int ng = q.n_groups();
            const int packed_row = (cols + 1) / 2;
            for (int i = 0; i < m; ++i) {
                const float* xi = x + static_cast<std::size_t>(i) * cols;
                float* yi = y + static_cast<std::size_t>(i) * rows;
                for (int r = 0; r < rows; ++r) {
                    const std::uint8_t* prow = q.packed.data() + static_cast<std::size_t>(r) * packed_row;
                    const float* srow = q.scales.data() + static_cast<std::size_t>(r) * ng;
                    const float* zrow = q.zeros.data() + static_cast<std::size_t>(r) * ng;
                    float acc = 0.0f;
                    for (int c = 0; c < cols; ++c) {
                        const std::uint8_t byte = prow[c / 2];
                        const int qv = (c & 1) ? (byte >> 4) : (byte & 0x0f);
                        const int g = c / q.group_size;
                        acc += xi[c] * (static_cast<float>(qv) - zrow[g]) * srow[g];
                    }
                    yi[r] = acc;
                }
            }
            break;
        }
    }
}

void WeightMatrix::row(int r, float* out) const {
    UOCR_CHECK(r >= 0 && r < rows, "WeightMatrix::row out of range");
    switch (fmt) {
        case WeightFormat::F32_OWNED:
            std::memcpy(out, f32.data() + static_cast<std::size_t>(r) * cols, cols * sizeof(float));
            break;
        case WeightFormat::F32_EXT:
            std::memcpy(out, static_cast<const float*>(ext) + static_cast<std::size_t>(r) * cols,
                        cols * sizeof(float));
            break;
        case WeightFormat::BF16_EXT: {
            const auto* p = static_cast<const std::uint16_t*>(ext) + static_cast<std::size_t>(r) * cols;
            for (int c = 0; c < cols; ++c) out[c] = bf16_to_f32(p[c]);
            break;
        }
        case WeightFormat::INT4:
            for (int c = 0; c < cols; ++c) out[c] = q.at(r, c);
            break;
    }
}

void Linear::forward(const float* x, float* y, int m) const {
    weight.matmul(x, y, m);
    if (has_bias) {
        for (int i = 0; i < m; ++i) {
            float* yi = y + static_cast<std::size_t>(i) * weight.rows;
            for (int j = 0; j < weight.rows; ++j) yi[j] += bias[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------
namespace {

WeightMatrix view_bf16(const SafetensorsFile& st, const std::string& name) {
    if (!st.contains(name)) UOCR_THROW("missing tensor: " + name);
    const auto& in = st.info(name);
    UOCR_CHECK(in.shape.size() == 2, "expected 2-D tensor: " + name);
    WeightMatrix w;
    w.rows = static_cast<int>(in.shape[0]);
    w.cols = static_cast<int>(in.shape[1]);
    w.fmt = (in.dtype == DType::BF16) ? WeightFormat::BF16_EXT : WeightFormat::F32_EXT;
    w.ext = in.data;
    return w;
}

std::vector<float> vec_f32(const SafetensorsFile& st, const std::string& name) {
    if (!st.contains(name)) UOCR_THROW("missing tensor: " + name);
    return st.read_f32(name);
}

WeightMatrix make_quantized(const SafetensorsFile& st, const std::string& name, int group) {
    std::vector<float> tmp = st.read_f32(name);
    const auto& in = st.info(name);
    UOCR_CHECK(in.shape.size() == 2, "expected 2-D tensor: " + name);
    WeightMatrix w;
    w.rows = static_cast<int>(in.shape[0]);
    w.cols = static_cast<int>(in.shape[1]);
    w.fmt = WeightFormat::INT4;
    w.q = quantize_int4_awq(tmp.data(), w.rows, w.cols, group);
    return w;
}

Linear load_linear(const SafetensorsFile& st, const std::string& name, bool quantize, int group) {
    Linear l;
    l.weight = quantize ? make_quantized(st, name, group) : view_bf16(st, name);
    return l;
}

float rng_normal(std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, 0.02f);
    return d(rng);
}

Linear random_linear(int out, int in, std::mt19937& rng) {
    Linear l;
    l.weight.rows = out;
    l.weight.cols = in;
    l.weight.fmt = WeightFormat::F32_OWNED;
    l.weight.f32.resize(static_cast<std::size_t>(out) * in);
    for (auto& v : l.weight.f32) v = rng_normal(rng);
    return l;
}

}  // namespace

DecoderWeights DecoderWeights::load(const std::string& path, const ModelConfig& cfg,
                                    bool quantize_experts_int4, int int4_group_size) {
    DecoderWeights d;
    d.checkpoint = std::make_shared<SafetensorsFile>(path);
    const SafetensorsFile& st = *d.checkpoint;

    d.embed_tokens = view_bf16(st, "model.embed_tokens.weight");
    d.lm_head = st.contains("lm_head.weight") ? view_bf16(st, "lm_head.weight")
                                              : d.embed_tokens;
    d.final_norm = vec_f32(st, "model.norm.weight");
    d.image_newline = vec_f32(st, "model.image_newline");
    d.view_seperator = vec_f32(st, "model.view_seperator");

    d.projector.weight = view_bf16(st, "model.projector.layers.weight");
    d.projector.bias = vec_f32(st, "model.projector.layers.bias");
    d.projector.has_bias = true;

    d.layers.resize(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        LayerWeights& L = d.layers[i];
        L.input_layernorm = vec_f32(st, p + "input_layernorm.weight");
        L.post_attention_layernorm = vec_f32(st, p + "post_attention_layernorm.weight");
        L.q_proj.weight = view_bf16(st, p + "self_attn.q_proj.weight");
        L.k_proj.weight = view_bf16(st, p + "self_attn.k_proj.weight");
        L.v_proj.weight = view_bf16(st, p + "self_attn.v_proj.weight");
        L.o_proj.weight = view_bf16(st, p + "self_attn.o_proj.weight");

        L.is_moe = i >= cfg.first_k_dense_replace;
        if (!L.is_moe) {
            L.dense.gate = load_linear(st, p + "mlp.gate_proj.weight", false, int4_group_size);
            L.dense.up = load_linear(st, p + "mlp.up_proj.weight", false, int4_group_size);
            L.dense.down = load_linear(st, p + "mlp.down_proj.weight", false, int4_group_size);
        } else {
            L.router = view_bf16(st, p + "mlp.gate.weight");
            L.experts.resize(cfg.n_routed_experts);
            for (int e = 0; e < cfg.n_routed_experts; ++e) {
                const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
                L.experts[e].gate =
                    load_linear(st, ep + "gate_proj.weight", quantize_experts_int4, int4_group_size);
                L.experts[e].up =
                    load_linear(st, ep + "up_proj.weight", quantize_experts_int4, int4_group_size);
                L.experts[e].down =
                    load_linear(st, ep + "down_proj.weight", quantize_experts_int4, int4_group_size);
            }
            const std::string sp = p + "mlp.shared_experts.";
            L.shared.gate = load_linear(st, sp + "gate_proj.weight", false, int4_group_size);
            L.shared.up = load_linear(st, sp + "up_proj.weight", false, int4_group_size);
            L.shared.down = load_linear(st, sp + "down_proj.weight", false, int4_group_size);
        }
        UOCR_DEBUG("loaded layer %d", i);
    }
    UOCR_INFO("decoder weights loaded from %s", path.c_str());
    return d;
}

DecoderWeights DecoderWeights::random(const ModelConfig& cfg, std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    DecoderWeights d;

    auto rand_vec = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = rng_normal(rng);
        return v;
    };
    auto rand_ones = [&](int n) {
        std::vector<float> v(n);
        for (auto& x : v) x = 1.0f + 0.01f * rng_normal(rng);
        return v;
    };

    d.embed_tokens.rows = cfg.vocab_size;
    d.embed_tokens.cols = cfg.hidden_size;
    d.embed_tokens.fmt = WeightFormat::F32_OWNED;
    d.embed_tokens.f32 = rand_vec(cfg.vocab_size * cfg.hidden_size);
    d.lm_head = d.embed_tokens;  // sharing storage is fine for tests

    d.final_norm = rand_ones(cfg.hidden_size);
    d.image_newline = rand_vec(cfg.hidden_size);
    d.view_seperator = rand_vec(cfg.hidden_size);

    d.projector = random_linear(cfg.projector_n_embed, cfg.projector_input_dim, rng);
    d.projector.bias = rand_vec(cfg.projector_n_embed);
    d.projector.has_bias = true;

    d.layers.resize(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        LayerWeights& L = d.layers[i];
        L.input_layernorm = rand_ones(cfg.hidden_size);
        L.post_attention_layernorm = rand_ones(cfg.hidden_size);
        L.q_proj = random_linear(cfg.hidden_size, cfg.hidden_size, rng);
        L.k_proj = random_linear(cfg.hidden_size, cfg.hidden_size, rng);
        L.v_proj = random_linear(cfg.hidden_size, cfg.hidden_size, rng);
        L.o_proj = random_linear(cfg.hidden_size, cfg.hidden_size, rng);
        L.is_moe = i >= cfg.first_k_dense_replace;
        if (!L.is_moe) {
            L.dense.gate = random_linear(cfg.intermediate_size, cfg.hidden_size, rng);
            L.dense.up = random_linear(cfg.intermediate_size, cfg.hidden_size, rng);
            L.dense.down = random_linear(cfg.hidden_size, cfg.intermediate_size, rng);
        } else {
            L.router.rows = cfg.n_routed_experts;
            L.router.cols = cfg.hidden_size;
            L.router.fmt = WeightFormat::F32_OWNED;
            L.router.f32 = rand_vec(cfg.n_routed_experts * cfg.hidden_size);
            const int shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts;
            L.experts.resize(cfg.n_routed_experts);
            for (auto& e : L.experts) {
                e.gate = random_linear(cfg.moe_intermediate_size, cfg.hidden_size, rng);
                e.up = random_linear(cfg.moe_intermediate_size, cfg.hidden_size, rng);
                e.down = random_linear(cfg.hidden_size, cfg.moe_intermediate_size, rng);
            }
            L.shared.gate = random_linear(shared_inter, cfg.hidden_size, rng);
            L.shared.up = random_linear(shared_inter, cfg.hidden_size, rng);
            L.shared.down = random_linear(cfg.hidden_size, shared_inter, rng);
        }
    }
    return d;
}

}  // namespace uocr
