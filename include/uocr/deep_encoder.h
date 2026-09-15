#pragma once

// DeepEncoder: SAM-ViT-B (windowed attention + decomposed relative position)
// followed by a CLIP-L transformer that consumes the SAM feature map, then a
// linear projector into the decoder embedding space.
//
// The forward path (see modeling_unlimitedocr.py / deepencoder.py):
//   1. image -> SAM ViT-B -> neck/upscale -> [1024, 64, 64] feature map
//   2. SAM features treated as tokens: CLIP-L runs over them
//   3. concat(CLIP output, flattened SAM features) -> linear projector
//   4. append `image_newline` embeddings, flatten -> visual token sequence
//
// This implementation targets the 1024x1024 input used by Unlimited-OCR.

#include <string>
#include <utility>
#include <vector>

#include "uocr/config.h"
#include "uocr/safetensors.h"
#include "uocr/tensor.h"
#include "uocr/weights.h"

namespace uocr {

struct SAMBlockWeights {
    std::vector<float> norm1_w, norm1_b;
    WeightMatrix qkv_w;
    std::vector<float> qkv_b;
    WeightMatrix proj_w;
    std::vector<float> proj_b;
    std::vector<float> rel_pos_h, rel_pos_w;  // [2*window-1, head_dim]
    std::vector<float> norm2_w, norm2_b;
    WeightMatrix mlp_lin1_w;
    std::vector<float> mlp_lin1_b;
    WeightMatrix mlp_lin2_w;
    std::vector<float> mlp_lin2_b;
};

struct SAMWeights {
    std::vector<float> patch_w;  // [768, 3, 16, 16]
    std::vector<float> patch_b;
    std::vector<float> pos_embed;  // [64*64*768]
    std::vector<SAMBlockWeights> blocks;
    std::vector<float> neck0_w;  // [256, 768, 1, 1]
    std::vector<float> neck1_w, neck1_b;
    std::vector<float> neck2_w;  // [256, 256, 3, 3]
    std::vector<float> neck3_w, neck3_b;
    std::vector<float> net2_w;   // [512, 256, 3, 3]
    std::vector<float> net3_w;   // [1024, 512, 3, 3]
};

struct CLIPLayerWeights {
    std::vector<float> ln1_w, ln1_b;
    WeightMatrix qkv_w;
    std::vector<float> qkv_b;
    WeightMatrix out_w;
    std::vector<float> out_b;
    std::vector<float> ln2_w, ln2_b;
    WeightMatrix fc1_w;
    std::vector<float> fc1_b;
    WeightMatrix fc2_w;
    std::vector<float> fc2_b;
};

struct CLIPWeights {
    std::vector<float> class_embedding;  // [1024]
    std::vector<float> pos_embed;        // [257*1024]
    std::vector<float> pre_ln_w, pre_ln_b;
    std::vector<CLIPLayerWeights> layers;
};

struct VisionWeights {
    SAMWeights sam;
    CLIPWeights clip;
    static VisionWeights load(const SafetensorsFile& st, const ModelConfig& cfg);
};

class DeepEncoder {
public:
    DeepEncoder(ModelConfig cfg, VisionWeights w, DecoderWeights projector_weights);

    // image: CHW float32, normalized, size cfg.image_size x cfg.image_size.
    // Returns [num_visual_tokens, hidden_size] row-major embeddings.
    // Optional debug outputs expose the SAM feature map ([1024,16,16] CHW) and
    // the CLIP output without the class token ([256,1024]) for alignment.
    void encode(const float* image_chw, int height, int width, Tensor& out,
                std::vector<float>* sam_debug = nullptr,
                std::vector<float>* clip_debug = nullptr) const;

    // Alignment helper: additionally records named intermediate tensors.
    using Stage = std::pair<std::string, std::vector<float>>;
    void encode_stages(const float* image_chw, int height, int width, Tensor& out,
                       std::vector<Stage>* stages, std::vector<float>* sam_debug = nullptr,
                       std::vector<float>* clip_debug = nullptr) const;

    int num_tokens() const { return num_tokens_; }

private:
    std::vector<float> sam_forward(const float* image_chw, int h, int w,
                                   std::vector<Stage>* stages = nullptr) const;
    std::vector<float> clip_forward(const std::vector<float>& sam_tokens, int tokens,
                                    std::vector<Stage>* stages = nullptr) const;

    ModelConfig cfg_;
    VisionWeights vw_;
    DecoderWeights pw_;
    mutable int num_tokens_ = 0;
};

}  // namespace uocr
