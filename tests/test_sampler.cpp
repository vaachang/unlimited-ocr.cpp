#include "test_main.h"
#include "uocr/sampler.h"

#include <limits>
#include <vector>

using namespace uocr;

namespace {
bool is_banned(float v) { return v == -std::numeric_limits<float>::infinity(); }
}  // namespace

UOCR_TEST(no_repeat_ngram_prefix_match) {
    // Reference `SlidingWindowNoRepeatNgramProcessor`: ban the continuation of
    // any (ngram-1)-prefix in the window matching the current suffix.
    std::vector<int> history = {1, 2, 3, 4, 1, 2};
    std::vector<float> logits(8, 0.0f);
    Sampler::apply_no_repeat_ngram(logits.data(), 8, history, 3, 1024);
    CHECK(is_banned(logits[3]));
    for (int i : {0, 1, 2, 4, 5, 6, 7}) CHECK(!is_banned(logits[static_cast<std::size_t>(i)]));
}

UOCR_TEST(no_repeat_ngram_window) {
    std::vector<int> history = {9, 5, 6, 7};
    std::vector<float> logits(10, 0.0f);
    // window=3 keeps only [5,6,7]; ngram=1 bans all tokens in the window.
    Sampler::apply_no_repeat_ngram(logits.data(), 10, history, 1, 3);
    CHECK(is_banned(logits[5]));
    CHECK(is_banned(logits[6]));
    CHECK(is_banned(logits[7]));
    CHECK(!is_banned(logits[9]));
}

UOCR_TEST(no_repeat_ngram_too_short) {
    std::vector<int> history = {1, 2};
    std::vector<float> logits(5, 0.0f);
    Sampler::apply_no_repeat_ngram(logits.data(), 5, history, 3, 1024);
    for (float v : logits) CHECK(!is_banned(v));
}

UOCR_TEST(sampler_greedy) {
    std::vector<float> logits = {0.1f, -2.0f, 3.5f, 0.0f};
    CHECK_EQ(Sampler::greedy(logits.data(), 4), 2);
}
