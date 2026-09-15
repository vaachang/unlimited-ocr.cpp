#include "test_main.h"
#include "uocr/kv_cache.h"

#include <cmath>
#include <random>
#include <vector>

using namespace uocr;

namespace {

std::vector<float> rand_vec(std::mt19937& rng, std::size_t n) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// Naive attention over the first `limit` slots of a cache layer (no causal
// mask), matching RSWACache::attention(..., causal=false).
void naive_attention(const RSWACache& cache, int layer, const std::vector<float>& q, int heads,
                     int q_pos, std::vector<float>& out) {
    const int hd = cache.head_dim();
    const int kv_heads = cache.kv_heads();
    const int limit = cache.length(layer);
    const int group = heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const float* kbuf = cache.keys(layer);
    const float* vbuf = cache.values(layer);
    out.assign(static_cast<std::size_t>(heads) * hd, 0.0f);
    (void)q_pos;
    for (int h = 0; h < heads; ++h) {
        const int kvh = h / group;
        const float* qh = q.data() + static_cast<std::size_t>(h) * hd;
        float mx = -1e30f;
        std::vector<float> scores(static_cast<std::size_t>(limit));
        for (int t = 0; t < limit; ++t) {
            const float* kt = kbuf + (static_cast<std::size_t>(t) * kv_heads + kvh) * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) dot += qh[d] * kt[d];
            scores[static_cast<std::size_t>(t)] = dot * scale;
            mx = std::max(mx, scores[static_cast<std::size_t>(t)]);
        }
        float sum = 0.0f;
        for (int t = 0; t < limit; ++t) {
            scores[static_cast<std::size_t>(t)] = std::exp(scores[static_cast<std::size_t>(t)] - mx);
            sum += scores[static_cast<std::size_t>(t)];
        }
        for (int t = 0; t < limit; ++t) {
            const float p = scores[static_cast<std::size_t>(t)] / sum;
            const float* vt = vbuf + (static_cast<std::size_t>(t) * kv_heads + kvh) * hd;
            for (int d = 0; d < hd; ++d) out[static_cast<std::size_t>(h) * hd + d] += p * vt[d];
        }
    }
}

}  // namespace

UOCR_TEST(ring_buffer_lifecycle) {
    const int layers = 1, kv_heads = 2, hd = 4, W = 4, P = 3;
    RSWACache cache(layers, kv_heads, hd, W);
    cache.reset(P);
    CHECK_EQ(cache.capacity(), P + W);
    CHECK_EQ(cache.prefill_len(), P);
    CHECK_EQ(cache.length(0), P);

    std::mt19937 rng(42);
    auto pre_k = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
    auto pre_v = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
    cache.write_prefill(0, pre_k.data(), pre_v.data(), P);

    // copy of the reference region, must never change
    std::vector<float> ref_k(pre_k), ref_v(pre_v);

    // warmup: append W tokens, cache grows linearly
    for (int i = 0; i < W; ++i) {
        auto k = rand_vec(rng, kv_heads * hd);
        auto v = rand_vec(rng, kv_heads * hd);
        cache.append_decode(0, k.data(), v.data());
    }
    CHECK_EQ(cache.length(0), P + W);
    CHECK(cache.ring_started(0));

    // steady state: overwrite ring slots
    for (int i = 0; i < 5; ++i) {
        auto k = rand_vec(rng, kv_heads * hd);
        auto v = rand_vec(rng, kv_heads * hd);
        cache.append_decode(0, k.data(), v.data());
    }
    CHECK_EQ(cache.length(0), P + W);
    CHECK_EQ(cache.ring_pos(0), 5 % W);

    // reference region intact
    for (std::size_t i = 0; i < ref_k.size(); ++i) {
        CHECK_NEAR(cache.keys(0)[i], ref_k[i], 0.0);
        CHECK_NEAR(cache.values(0)[i], ref_v[i], 0.0);
    }
}

UOCR_TEST(ring_attention_matches_naive) {
    const int layers = 1, kv_heads = 2, heads = 2, hd = 4, W = 4, P = 5;
    RSWACache cache(layers, kv_heads, hd, W);
    cache.reset(P);
    std::mt19937 rng(7);

    auto pre_k = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
    auto pre_v = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
    cache.write_prefill(0, pre_k.data(), pre_v.data(), P);

    for (int i = 0; i < 13; ++i) {
        auto k = rand_vec(rng, kv_heads * hd);
        auto v = rand_vec(rng, kv_heads * hd);
        cache.append_decode(0, k.data(), v.data());
    }

    auto q = rand_vec(rng, static_cast<std::size_t>(heads) * hd);
    std::vector<float> got(static_cast<std::size_t>(heads) * hd);
    cache.attention(0, q.data(), 1, cache.length(0) - 1, got.data(), heads, false);

    std::vector<float> want;
    naive_attention(cache, 0, q, heads, cache.length(0) - 1, want);

    for (std::size_t i = 0; i < got.size(); ++i) CHECK_NEAR(got[i], want[i], 1e-4);
}

UOCR_TEST(prefill_causal_attention) {
    const int layers = 1, kv_heads = 1, heads = 1, hd = 4, W = 4, P = 6;
    RSWACache cache(layers, kv_heads, hd, W);
    cache.reset(P);
    std::mt19937 rng(3);

    auto k = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
    auto v = rand_vec(rng, static_cast<std::size_t>(P) * kv_heads * hd);
    cache.write_prefill(0, k.data(), v.data(), P);

    auto q = rand_vec(rng, static_cast<std::size_t>(P) * heads * hd);
    std::vector<float> got(static_cast<std::size_t>(P) * heads * hd);
    cache.attention(0, q.data(), P, 0, got.data(), heads, true);

    // manual causal attention
    for (int s = 0; s < P; ++s) {
        float mx = -1e30f;
        std::vector<float> sc(s + 1);
        for (int t = 0; t <= s; ++t) {
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) dot += q[s * hd + d] * k[t * hd + d];
            sc[t] = dot / std::sqrt(static_cast<float>(hd));
            mx = std::max(mx, sc[t]);
        }
        float sum = 0.0f;
        for (int t = 0; t <= s; ++t) {
            sc[t] = std::exp(sc[t] - mx);
            sum += sc[t];
        }
        for (int d = 0; d < hd; ++d) {
            float acc = 0.0f;
            for (int t = 0; t <= s; ++t) acc += sc[t] / sum * v[t * hd + d];
            CHECK_NEAR(got[s * hd + d], acc, 1e-4);
        }
    }
}
