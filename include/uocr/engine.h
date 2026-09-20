#pragma once

// Top-level engine: ties the model configuration, weights, decoder, block
// manager and continuous-batching scheduler together.
//
// The CPU reference path executes one request at a time (host GEMMs); the CUDA
// path is intended to run the batched kernels.  `generate()` is the simplest
// entry point and is what the benchmarks/tests use.

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "uocr/block_manager.h"
#include "uocr/config.h"
#include "uocr/continuous_batch.h"
#include "uocr/deep_encoder.h"
#include "uocr/image.h"
#include "uocr/moe_decoder.h"
#include "uocr/sampler.h"
#include "uocr/tokenizer.h"
#include "uocr/weights.h"

namespace uocr {

namespace cuda {
class GpuDecoder;
#if defined(UOCR_CUDA_ENABLED)
class GpuEncoder;
#endif
}

enum class Backend { CPU, CUDA };

struct GenerationResult {
    std::vector<int> tokens;      // newly generated tokens
    double ttft_ms = 0.0;         // time to first token (prefill)
    double tpot_ms = 0.0;         // average time per output token
    double decode_ms = 0.0;       // total decode time
    int prefill_tokens = 0;
    int decode_tokens = 0;
};

class Engine {
public:
    Engine(ModelConfig mcfg, EngineConfig ecfg, DecoderWeights weights, Backend backend = Backend::CPU);
    ~Engine();

    // Convenience: load config + weights from `ecfg.model_dir`.
    static std::unique_ptr<Engine> load(const EngineConfig& ecfg, Backend backend = Backend::CPU);

    MoEDecoder& decoder() { return *decoder_; }
    const MoEDecoder& decoder() const { return *decoder_; }
    BlockManager& block_manager() { return *block_mgr_; }
    ContinuousBatchScheduler& scheduler() { return *scheduler_; }
    Sampler& sampler() { return *sampler_; }
    const ModelConfig& model_config() const { return mcfg_; }
    const EngineConfig& engine_config() const { return ecfg_; }
    Backend backend() const { return backend_; }

    // Single-request greedy/nucleus generation.  `prompt` is already tokenized.
    // `doc_key` enables reference-region sharing across requests.
    GenerationResult generate(const std::vector<int>& prompt, int max_new_tokens = 64,
                              const std::string& doc_key = "");

    // Like generate(), but `images_seq_mask` marks positions in `prompt` that
    // must be filled with rows of `visual_embeddings` ([V, hidden] row-major)
    // instead of token embeddings (E3).
    GenerationResult generate_from_image(const std::vector<int>& prompt,
                                         const std::vector<std::uint8_t>& images_seq_mask,
                                         const std::vector<float>& visual_embeddings, int hidden,
                                         int max_new_tokens = 64, const std::string& doc_key = "");

    // Continuous batching: run several text prompts concurrently, driving the
    // scheduler and the device batched decoder.  Returns one result per prompt
    // (in input order).  On the CPU backend this falls back to sequential
    // `generate` calls.
    std::vector<GenerationResult> generate_batch(const std::vector<std::vector<int>>& prompts,
                                                 int max_new_tokens = 64);

    // Assemble the visual embeddings for one image exactly like the reference:
    // global view, plus Gundam local crops when crop_mode and the dynamic grid
    // is larger than 1x1.  Requires `set_vision()`.
    std::vector<float> image_embeddings(const ImageRGB& image, bool crop_mode = true,
                                        int base_size = 0, int image_size = 0);

    // Text prompt -> embeddings (embed_tokens only).  Images are scattered by
    // the caller / generate_from_image().
    Tensor embed_tokens(const std::vector<int>& tokens) const { return decoder_->embed(tokens); }

    // Encode an image into visual embeddings (requires vision weights).
    void set_vision(std::shared_ptr<DeepEncoder> enc) { vision_ = std::move(enc); }
    const DeepEncoder* vision() const { return vision_.get(); }

#if defined(UOCR_CUDA_ENABLED)
    // Optional GPU vision encoder.  When set on a CUDA engine,
    // `image_embeddings` runs the DeepEncoder on the device instead of the CPU.
    void set_vision_gpu(std::shared_ptr<cuda::GpuEncoder> enc) {
        gpu_vision_ = std::move(enc);
    }
    const cuda::GpuEncoder* vision_gpu() const { return gpu_vision_.get(); }
#endif

    // Insert visual embeddings into `tokens`/`inputs` at image-token positions.
    // Returns the indices that were replaced.
    static std::vector<int> find_image_token_positions(const std::vector<int>& tokens,
                                                       int image_token_id);

    std::size_t kv_bytes_per_request(int prefill_len) const;

private:
    ModelConfig mcfg_;
    EngineConfig ecfg_;
    DecoderWeights weights_;
    Backend backend_;
    std::unique_ptr<MoEDecoder> decoder_;
    std::unique_ptr<BlockManager> block_mgr_;
    std::unique_ptr<ContinuousBatchScheduler> scheduler_;
    std::unique_ptr<Sampler> sampler_;
    std::shared_ptr<DeepEncoder> vision_;
#if defined(UOCR_CUDA_ENABLED)
    std::unique_ptr<cuda::GpuDecoder> gpu_decoder_;
    std::shared_ptr<cuda::GpuEncoder> gpu_vision_;
#endif
};

}  // namespace uocr
