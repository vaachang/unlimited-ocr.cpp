#pragma once

// R-SWA (Reference Sliding Window Attention) KV cache.
//
// Layout per layer (mirrors the reference implementation in
// modeling_deepseekv2.SlidingWindowLlamaAttention):
//
//   [ reference region: 0 .. P-1 ][ ring region: P .. P+W-1 ]
//        fixed, never overwritten     overwritten in a circular fashion
//
// `P` is the length of the first prefill (image + prompt tokens).  During
// decode the cache grows linearly until P+W, then the ring overwrites slot
// P + ring_pos.  Attention always runs over the whole cache (P+W slots), which
// is what makes R-SWA "reference tokens + last W generated tokens".

#include <memory>
#include <vector>

#include "uocr/common.h"

namespace uocr {

class RSWACache {
public:
    RSWACache() = default;
    RSWACache(int num_layers, int kv_heads, int head_dim, int window);

    // Configure the cache for a request whose first prefill has length P.
    // Allocates P + W slots per layer.  Resets counters.
    void reset(int prefill_len);

    void configure(int num_layers, int kv_heads, int head_dim, int window);

    int num_layers() const { return num_layers_; }
    int kv_heads() const { return kv_heads_; }
    int head_dim() const { return head_dim_; }
    int window() const { return window_; }
    int prefill_len() const { return prefill_len_; }

    // Number of valid slots in the cache for `layer` (P .. P+W).
    int length(int layer) const { return len_[layer]; }
    int ring_pos(int layer) const { return ring_pos_[layer]; }
    bool ring_started(int layer) const { return ring_started_[layer]; }

    // cap = P + W
    int capacity() const { return capacity_; }

    // Write `seq` keys/values at the beginning (prefill).  k/v layout:
    // [seq, kv_heads * head_dim] (contiguous, row-major).
    void write_prefill(int layer, const float* k, const float* v, int seq);

    // Append one decoded token.  Handles warmup vs. ring overwrite exactly like
    // the reference: grow until P+W, then circular overwrite.
    // k/v layout: [kv_heads * head_dim].
    void append_decode(int layer, const float* k, const float* v);

    // Attention for `seq` queries starting at global position `q_start`.
    // q/out layout: [seq, heads * head_dim].
    // `causal` must be true for a prefill; decode steps pass false.
    void attention(int layer, const float* q, int seq, int q_start, float* out,
                   int heads, bool causal) const;

    // Raw accessors (used by tests / CUDA upload).
    int len(int layer) const { return len_[layer]; }
    const float* keys(int layer) const { return k_[layer].data(); }
    const float* values(int layer) const { return v_[layer].data(); }
    float* mutable_keys(int layer) { return k_[layer].data(); }
    float* mutable_values(int layer) { return v_[layer].data(); }

    std::size_t bytes() const;

private:
    int num_layers_ = 0;
    int kv_heads_ = 0;
    int head_dim_ = 0;
    int window_ = 128;
    int prefill_len_ = 0;
    int capacity_ = 0;
    bool configured_ = false;

    std::vector<std::vector<float>> k_;
    std::vector<std::vector<float>> v_;
    std::vector<int> len_;
    std::vector<int> ring_pos_;
    std::vector<bool> ring_started_;
};

}  // namespace uocr
