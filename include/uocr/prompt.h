#pragma once

// OCR prompt / image-token layout.
//
// Reproduces `UnlimitedOCRForCausalLM.infer()`'s token construction:
// the plain SFT prompt is split on `<image>` and each placeholder is replaced
// by the exact number of `<image>` token ids that the vision tower will emit.
// `images_seq_mask` marks those positions so the engine can scatter the visual
// embeddings in (E3).

#include <cstdint>
#include <vector>

#include "uocr/config.h"
#include "uocr/tokenizer.h"

namespace uocr {

struct ImageSpatialCrop {
    int width_crop_num = 1;
    int height_crop_num = 1;
};

struct PromptLayout {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> images_seq_mask;  // 1 = visual token position
    int num_image_tokens = 0;
};

// `text_splits` has exactly `crops.size() + 1` entries (as produced by
// splitting the prompt on `<image>`).  `crops[i]` describes the dynamic-
// resolution grid of the i-th image.
PromptLayout build_ocr_prompt(const Tokenizer& tok, const std::vector<std::string>& text_splits,
                              const std::vector<ImageSpatialCrop>& crops, const ModelConfig& cfg,
                              bool crop_mode = true);

// Number of `<image>` token ids the layout contains for one image, following the
// reference `infer()` formula:
//   crop mode, fits        : num_queries_base * (num_queries_base + 1) + 1
//   crop mode, local views : above + (num_queries*wcn + 1) * (num_queries*hcn)
//   non-crop mode          : num_queries * (num_queries + 1) + 1
// `image_embeddings()` asserts its visual-embedding count equals this, which
// keeps the prompt layout and the vision path in lock-step.
int image_token_count(const ModelConfig& cfg, bool crop_mode, int width_crop_num,
                      int height_crop_num);

}  // namespace uocr
