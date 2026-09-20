#pragma once

// GPU DeepEncoder (SAM-ViT + CLIP-L + projector), CUDA builds only.
//
// Weights are uploaded once (linear weights as bf16, everything else f32);
// a whole image view is encoded on the device.  The output layout matches the
// CPU `DeepEncoder::encode` exactly (grid rows + image_newline + view_seperator
// + final view_seperator token handled by the caller).

#if defined(UOCR_CUDA_ENABLED)

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "uocr/common.h"
#include "uocr/config.h"
#include "uocr/deep_encoder.h"
#include "uocr/tensor.h"
#include "uocr/weights.h"

namespace uocr {
namespace cuda {

class GpuEncoder {
public:
    GpuEncoder(ModelConfig cfg, const VisionWeights& vw, const DecoderWeights& pw);
    ~GpuEncoder();

    GpuEncoder(const GpuEncoder&) = delete;
    GpuEncoder& operator=(const GpuEncoder&) = delete;

    // image: CHW f32, normalized, height x width (assumed square).  Produces
    // [rows*cols+1, hidden] with image_newline rows and the trailing
    // view_seperator (same as the CPU encoder).
    void encode(const float* image_chw, int height, int width, Tensor& out,
                std::vector<float>* sam_debug = nullptr,
                std::vector<float>* clip_debug = nullptr,
                std::vector<std::pair<std::string, std::vector<float>>>* stages = nullptr);

    int num_tokens() const { return num_tokens_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int num_tokens_ = 0;
};

}  // namespace cuda
}  // namespace uocr

#endif  // UOCR_CUDA_ENABLED
