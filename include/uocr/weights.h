#pragma once

// Weight containers for the MoE decoder plus helpers to obtain them either
// from a safetensors checkpoint or from a deterministic random initializer
// (used by tests with a tiny configuration).

#include <memory>
#include <string>
#include <vector>

#include "uocr/common.h"
#include "uocr/config.h"
#include "uocr/quant.h"
#include "uocr/safetensors.h"

namespace uocr {

enum class WeightFormat { F32_OWNED, F32_EXT, BF16_EXT, INT4 };

// A 2-D weight matrix [rows=out_features, cols=in_features].
class WeightMatrix {
public:
    int rows = 0;
    int cols = 0;
    WeightFormat fmt = WeightFormat::F32_OWNED;

    const void* ext = nullptr;       // non-owning view (F32_EXT / BF16_EXT)
    std::vector<float> f32;          // owned F32_OWNED
    QuantizedMatrix q;               // INT4

    bool empty() const { return rows == 0 || cols == 0; }

    // Dequantize/materialize the whole matrix (debug & tests).
    void to_f32(std::vector<float>& out) const;

    // y[rows] = W * x[cols]  (single activation vector)
    void matvec(const float* x, float* y) const;

    // Y[m,rows] = X[m,cols] * W^T
    void matmul(const float* x, float* y, int m) const;

    // Copy row `r` (length cols) into out, handling every storage format.
    void row(int r, float* out) const;
};

struct Linear {
    WeightMatrix weight;
    std::vector<float> bias;
    bool has_bias = false;

    int out_features() const { return weight.rows; }
    int in_features() const { return weight.cols; }

    // x: [m, in], y: [m, out]
    void forward(const float* x, float* y, int m) const;
};

struct ExpertWeights {
    Linear gate, up, down;
};

struct MLPWeights {
    Linear gate, up, down;
};

struct LayerWeights {
    std::vector<float> input_layernorm;
    std::vector<float> post_attention_layernorm;

    Linear q_proj, k_proj, v_proj, o_proj;

    bool is_moe = false;

    // dense MLP (first_k_dense_replace)
    MLPWeights dense;

    // MoE (when is_moe)
    WeightMatrix router;  // [n_routed_experts, hidden]
    std::vector<ExpertWeights> experts;
    MLPWeights shared;
};

struct DecoderWeights {
    std::shared_ptr<SafetensorsFile> checkpoint;

    WeightMatrix embed_tokens;  // [vocab, hidden]
    WeightMatrix lm_head;       // [vocab, hidden]
    std::vector<float> final_norm;
    std::vector<float> image_newline;    // [hidden]
    std::vector<float> view_seperator;   // [hidden]

    Linear projector;  // [n_embed, projector_input]

    std::vector<LayerWeights> layers;

    // Convenience: load decoder weights from a safetensors checkpoint.
    static DecoderWeights load(const std::string& safetensors_path,
                               const ModelConfig& cfg,
                               bool quantize_experts_int4 = false,
                               int int4_group_size = 128);

    // Deterministic random weights for a (small) config.  Skips vision.
    static DecoderWeights random(const ModelConfig& cfg, std::uint64_t seed = 1234);
};

}  // namespace uocr
