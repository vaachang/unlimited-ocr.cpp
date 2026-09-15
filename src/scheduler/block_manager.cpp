#include "uocr/block_manager.h"

#include "uocr/log.h"

namespace uocr {

BlockManager::BlockManager(std::size_t prefix_pool_bytes, std::size_t ring_pool_bytes)
    : prefix_pool_(prefix_pool_bytes), ring_pool_(ring_pool_bytes) {}

PrefixEntry* BlockManager::acquire_prefix(const std::string& key, int prefill_len,
                                          std::size_t bytes) {
    auto it = prefixes_.find(key);
    if (it != prefixes_.end()) {
        it->second.refcount += 1;
        recompute_shared();
        return &it->second;
    }

    i64 off = prefix_pool_.allocate(bytes);
    UOCR_CHECK(off >= 0, "prefix pool exhausted");

    PrefixEntry e;
    e.key = key;
    e.prefill_len = prefill_len;
    e.bytes = bytes;
    e.refcount = 1;
    e.offset = off;
    auto res = prefixes_.emplace(key, std::move(e));
    recompute_shared();
    UOCR_DEBUG("prefix allocated key=%s prefill_len=%d bytes=%zu", key.c_str(), prefill_len, bytes);
    return &res.first->second;
}

void BlockManager::release_prefix(const std::string& key) {
    auto it = prefixes_.find(key);
    if (it == prefixes_.end()) return;
    PrefixEntry& e = it->second;
    e.refcount -= 1;
    if (e.refcount <= 0) {
        prefix_pool_.free(e.offset, e.bytes);
        prefixes_.erase(it);
    }
    recompute_shared();
}

void BlockManager::recompute_shared() {
    shared_bytes_ = 0;
    for (const auto& kv : prefixes_) {
        if (kv.second.refcount >= 2) shared_bytes_ += kv.second.bytes;
    }
    peak_shared_bytes_ = std::max(peak_shared_bytes_, shared_bytes_);
}

i64 BlockManager::allocate_ring(std::size_t bytes) { return ring_pool_.allocate(bytes); }

void BlockManager::free_ring(i64 offset, std::size_t bytes) { ring_pool_.free(offset, bytes); }

std::size_t BlockManager::live_prefixes() const { return prefixes_.size(); }

std::size_t BlockManager::total_prefix_bytes() const {
    std::size_t total = 0;
    for (const auto& kv : prefixes_) total += kv.second.bytes;
    return total;
}

BlockManager::Stats BlockManager::stats() const {
    Stats s;
    s.prefix_live_bytes = prefix_pool_.used();
    s.prefix_peak_bytes = prefix_pool_.peak();
    s.prefix_shared_bytes = shared_bytes_;
    s.ring_live_bytes = ring_pool_.used();
    s.ring_peak_bytes = ring_pool_.peak();
    s.prefix_fragmentation = prefix_pool_.fragmentation();
    s.ring_fragmentation = ring_pool_.fragmentation();
    s.live_prefixes = prefixes_.size();
    return s;
}

}  // namespace uocr
