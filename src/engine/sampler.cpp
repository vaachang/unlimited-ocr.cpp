#include "uocr/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace uocr {

int Sampler::greedy(const float* logits, int n) {
    UOCR_CHECK(n > 0, "empty logits");
    int best = 0;
    float bestv = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > bestv) {
            bestv = logits[i];
            best = i;
        }
    }
    return best;
}

int Sampler::sample(const float* logits, int n) {
    if (params_.temperature <= 0.0f && params_.top_k <= 0 && params_.top_p >= 1.0f)
        return greedy(logits, n);

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    // top-k prefilter
    int keep = n;
    if (params_.top_k > 0 && params_.top_k < n) {
        std::partial_sort(idx.begin(), idx.begin() + params_.top_k, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        keep = params_.top_k;
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    }

    const float temp = params_.temperature > 0.0f ? params_.temperature : 1.0f;
    std::vector<float> probs(keep);
    float maxl = logits[idx[0]];
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) {
        probs[i] = std::exp((logits[idx[i]] - maxl) / temp);
        sum += probs[i];
    }
    for (int i = 0; i < keep; ++i) probs[i] /= sum;

    // top-p
    int cutoff = keep;
    if (params_.top_p < 1.0f) {
        float c = 0.0f;
        for (int i = 0; i < keep; ++i) {
            c += probs[i];
            if (c >= params_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < cutoff; ++i) norm += probs[i];
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    float u = d(rng_) * norm;
    float c = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        c += probs[i];
        if (u <= c) return idx[i];
    }
    return idx[cutoff - 1];
}

void Sampler::apply_no_repeat_ngram(float* logits, int vocab, const std::vector<int>& history,
                                    int ngram, int window) {
    // Mirrors `SlidingWindowNoRepeatNgramProcessor`: ban the continuation of any
    // (ngram-1)-prefix within the sliding window that matches the current
    // suffix (the reference processor is called with the full input_ids).
    if (ngram <= 0) return;
    const int n = static_cast<int>(history.size());
    if (n < ngram) return;
    const int search_start = std::max(0, n - window);
    const int search_end = n - ngram + 1;
    if (search_end <= search_start) return;

    const int plen = ngram - 1;  // current prefix length
    const std::size_t base = static_cast<std::size_t>(n - plen);
    for (int idx = search_start; idx < search_end; ++idx) {
        bool match = true;
        for (int j = 0; j < plen; ++j) {
            if (history[static_cast<std::size_t>(idx + j)] != history[base + static_cast<std::size_t>(j)]) {
                match = false;
                break;
            }
        }
        if (!match) continue;
        const int banned = history[static_cast<std::size_t>(idx + ngram - 1)];
        if (banned >= 0 && banned < vocab) logits[banned] = -std::numeric_limits<float>::infinity();
    }
}

}  // namespace uocr
