#include "uocr/kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace uocr {

RSWACache::RSWACache(int num_layers, int kv_heads, int head_dim, int window) {
    configure(num_layers, kv_heads, head_dim, window);
}

void RSWACache::configure(int num_layers, int kv_heads, int head_dim, int window) {
    num_layers_ = num_layers;
    kv_heads_ = kv_heads;
    head_dim_ = head_dim;
    window_ = window;
    k_.assign(num_layers, {});
    v_.assign(num_layers, {});
    len_.assign(num_layers, 0);
    ring_pos_.assign(num_layers, 0);
    ring_started_.assign(num_layers, false);
    prefill_len_ = 0;
    capacity_ = 0;
    configured_ = true;
}

void RSWACache::reset(int prefill_len) {
    UOCR_CHECK(configured_, "RSWACache not configured");
    UOCR_CHECK(prefill_len >= 0, "prefill_len must be >= 0");
    prefill_len_ = prefill_len;
    capacity_ = prefill_len + window_;
    const std::size_t per_layer =
        static_cast<std::size_t>(capacity_) * kv_heads_ * head_dim_;
    for (int l = 0; l < num_layers_; ++l) {
        k_[l].assign(per_layer, 0.0f);
        v_[l].assign(per_layer, 0.0f);
        len_[l] = prefill_len;
        ring_pos_[l] = 0;
        ring_started_[l] = false;
    }
}

void RSWACache::write_prefill(int layer, const float* k, const float* v, int seq) {
    UOCR_CHECK(seq <= capacity_, "prefill larger than cache capacity");
    const std::size_t stride = static_cast<std::size_t>(kv_heads_) * head_dim_;
    std::memcpy(k_[layer].data(), k, static_cast<std::size_t>(seq) * stride * sizeof(float));
    std::memcpy(v_[layer].data(), v, static_cast<std::size_t>(seq) * stride * sizeof(float));
    len_[layer] = seq;
    prefill_len_ = seq;
}

void RSWACache::append_decode(int layer, const float* k, const float* v) {
    const std::size_t stride = static_cast<std::size_t>(kv_heads_) * head_dim_;
    float* kbuf = k_[layer].data();
    float* vbuf = v_[layer].data();
    const int len = len_[layer];

    if (len < prefill_len_ + window_) {
        // warmup: linear append
        std::memcpy(kbuf + static_cast<std::size_t>(len) * stride, k, stride * sizeof(float));
        std::memcpy(vbuf + static_cast<std::size_t>(len) * stride, v, stride * sizeof(float));
        len_[layer] = len + 1;
        if (len + 1 >= prefill_len_ + window_) {
            ring_started_[layer] = true;
            ring_pos_[layer] = 0;
        }
        return;
    }

    // steady state: circular overwrite
    if (!ring_started_[layer]) {
        ring_started_[layer] = true;
        ring_pos_[layer] = 0;
    }
    const int rp = ring_pos_[layer];
    const int slot = prefill_len_ + rp;
    std::memcpy(kbuf + static_cast<std::size_t>(slot) * stride, k, stride * sizeof(float));
    std::memcpy(vbuf + static_cast<std::size_t>(slot) * stride, v, stride * sizeof(float));
    ring_pos_[layer] = (rp + 1) % window_;
}

void RSWACache::attention(int layer, const float* q, int seq, int q_start, float* out,
                          int heads, bool causal) const {
    const int hd = head_dim_;
    const int kv_len = len_[layer];
    const int group = heads / kv_heads_;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    const float* kbuf = k_[layer].data();
    const float* vbuf = v_[layer].data();

    for (int s = 0; s < seq; ++s) {
        const int qpos = q_start + s;
        for (int h = 0; h < heads; ++h) {
            const float* qh = q + (static_cast<std::size_t>(s) * heads + h) * hd;
            const int kvh = h / group;
            const float* kh = kbuf + static_cast<std::size_t>(kvh) * hd;
            const float* vh = vbuf + static_cast<std::size_t>(kvh) * hd;

            const int limit = causal ? std::min(kv_len, qpos + 1) : kv_len;

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            float acc[512];
            UOCR_CHECK(hd <= 512, "head_dim too large for stack accumulator");
            std::memset(acc, 0, sizeof(float) * static_cast<std::size_t>(hd));

            for (int t = 0; t < limit; ++t) {
                const float* kt = kh + static_cast<std::size_t>(t) * kv_heads_ * hd;
                float dot = 0.0f;
                for (int d = 0; d < hd; ++d) dot += qh[d] * kt[d];
                dot *= scale;

                const float m_new = std::max(m, dot);
                const float alpha = std::exp(m - m_new);
                const float beta = std::exp(dot - m_new);
                l = l * alpha + beta;
                const float* vt = vh + static_cast<std::size_t>(t) * kv_heads_ * hd;
                for (int d = 0; d < hd; ++d) acc[d] = acc[d] * alpha + beta * vt[d];
                m = m_new;
            }

            float* oh = out + (static_cast<std::size_t>(s) * heads + h) * hd;
            const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            for (int d = 0; d < hd; ++d) oh[d] = acc[d] * inv;
        }
    }
}

std::size_t RSWACache::bytes() const {
    std::size_t total = 0;
    for (const auto& kv : k_) total += kv.size() * sizeof(float);
    for (const auto& vv : v_) total += vv.size() * sizeof(float);
    return total;
}

}  // namespace uocr
