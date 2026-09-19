#include "uocr/prompt.h"

#include <cmath>

#include "uocr/log.h"

namespace uocr {

PromptLayout build_ocr_prompt(const Tokenizer& tok, const std::vector<std::string>& text_splits,
                              const std::vector<ImageSpatialCrop>& crops, const ModelConfig& cfg,
                              bool crop_mode) {
    UOCR_CHECK(text_splits.size() == crops.size() + 1,
               "build_ocr_prompt: text_splits must have crops.size()+1 entries");

    const int patch_size = cfg.patch_size;              // 16
    const int downsample_ratio = cfg.downsample_ratio;  // 4
    const int image_size = cfg.candidate_image_size;    // 640 (local view)
    const int base_size = cfg.base_size;                // 1024 (global view)
    const int id = cfg.image_token_id;

    const int num_queries = static_cast<int>(
        std::ceil(static_cast<double>(image_size / patch_size) / downsample_ratio));
    const int num_queries_base = static_cast<int>(
        std::ceil(static_cast<double>(base_size / patch_size) / downsample_ratio));

    PromptLayout out;
    auto push_text = [&](const std::string& s) {
        std::vector<int> ids = tok.encode(s, false, false);
        out.input_ids.insert(out.input_ids.end(), ids.begin(), ids.end());
        out.images_seq_mask.insert(out.images_seq_mask.end(), ids.size(), 0);
    };

    for (std::size_t i = 0; i < crops.size(); ++i) {
        push_text(text_splits[i]);

        std::vector<int> img;
        if (crop_mode) {
            // global view: ([id]*nqb + [id]) * nqb + [id]
            for (int r = 0; r < num_queries_base; ++r) {
                img.insert(img.end(), static_cast<std::size_t>(num_queries_base + 1), id);
            }
            img.push_back(id);

            const int wcn = crops[i].width_crop_num;
            const int hcn = crops[i].height_crop_num;
            if (wcn > 1 || hcn > 1) {
                const int row = num_queries * wcn + 1;
                for (int r = 0; r < num_queries * hcn; ++r)
                    img.insert(img.end(), static_cast<std::size_t>(row), id);
            }
        } else {
            // single global view at `image_size`
            for (int r = 0; r < num_queries; ++r)
                img.insert(img.end(), static_cast<std::size_t>(num_queries + 1), id);
            img.push_back(id);
        }

        out.input_ids.insert(out.input_ids.end(), img.begin(), img.end());
        out.images_seq_mask.insert(out.images_seq_mask.end(), img.size(), 1);
        out.num_image_tokens += static_cast<int>(img.size());
    }
    push_text(text_splits.back());

    // prepend bos
    out.input_ids.insert(out.input_ids.begin(), cfg.bos_token_id);
    out.images_seq_mask.insert(out.images_seq_mask.begin(), 0);

    return out;
}

}  // namespace uocr
