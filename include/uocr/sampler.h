#pragma once

// Token sampling and the sliding-window no-repeat-ngram processor used by the
// reference OCR pipeline (README: no_repeat_ngram_size=35, ngram_window=1024).

#include <random>
#include <vector>

#include "uocr/common.h"

namespace uocr {

struct SamplingParams {
    float temperature = 0.0f;  // 0 => greedy
    float top_p = 1.0f;
    int top_k = 0;  // 0 => disabled
    int seed = 0;
};

class Sampler {
public:
    explicit Sampler(SamplingParams p = {}) : params_(p), rng_(p.seed) {}

    void set_params(const SamplingParams& p) {
        params_ = p;
        if (p.seed != 0) rng_.seed(static_cast<std::uint32_t>(p.seed));
    }
    const SamplingParams& params() const { return params_; }

    int sample(const float* logits, int n);

    static int greedy(const float* logits, int n);

    // Ban tokens that would repeat an n-gram of size `ngram` found in the last
    // `window` tokens of `history`.  `logits` is modified in place.
    static void apply_no_repeat_ngram(float* logits, int vocab, const std::vector<int>& history,
                                      int ngram, int window);

private:
    SamplingParams params_{};
    std::mt19937 rng_{0};
};

}  // namespace uocr
