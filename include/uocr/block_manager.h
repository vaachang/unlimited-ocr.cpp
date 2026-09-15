#pragma once

// R-SWA aware block manager.
//
// Unlike PagedAttention's homogeneous block table, R-SWA has two regions with
// different lifetimes:
//
//   * reference region (vision + prompt): read-only after prefill, shareable
//     across requests that decode the same document.  Managed by a
//     reference-counted prefix pool.
//   * ring region: per-request, fixed W slots, overwritten circularly.  No
//     block-table bookkeeping is needed because the slot address only depends
//     on (request, layer, ring_pos).

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "uocr/memory_pool.h"

namespace uocr {

struct PrefixEntry {
    std::string key;
    int prefill_len = 0;
    std::size_t bytes = 0;
    int refcount = 0;
    i64 offset = -1;  // offset into the prefix arena
};

class BlockManager {
public:
    BlockManager(std::size_t prefix_pool_bytes, std::size_t ring_pool_bytes);

    // Acquire (or re-acquire) a shared reference-region allocation for `key`.
    // Returns the entry.  `bytes` is the size of the prefix KV across all
    // layers.  If an entry already exists, its refcount is bumped.
    PrefixEntry* acquire_prefix(const std::string& key, int prefill_len, std::size_t bytes);

    // Drop one reference; frees the arena block when the count reaches zero.
    void release_prefix(const std::string& key);

    // Allocate/free the ring region for one request (one contiguous block
    // holding W slots for every layer).
    i64 allocate_ring(std::size_t bytes);
    void free_ring(i64 offset, std::size_t bytes);

    // Live prefix allocations.
    std::size_t live_prefixes() const;
    std::size_t total_prefix_bytes() const;
    std::size_t shared_prefix_bytes() const { return shared_bytes_; }
    std::size_t peak_prefix_bytes() const { return peak_shared_bytes_; }

    const MemoryPool& prefix_pool() const { return prefix_pool_; }
    const MemoryPool& ring_pool() const { return ring_pool_; }

    struct Stats {
        std::size_t prefix_live_bytes = 0;
        std::size_t prefix_peak_bytes = 0;
        std::size_t prefix_shared_bytes = 0;
        std::size_t ring_live_bytes = 0;
        std::size_t ring_peak_bytes = 0;
        double prefix_fragmentation = 0.0;
        double ring_fragmentation = 0.0;
        std::size_t live_prefixes = 0;
    };
    Stats stats() const;

private:
    void recompute_shared();

    MemoryPool prefix_pool_;
    MemoryPool ring_pool_;
    std::unordered_map<std::string, PrefixEntry> prefixes_;
    std::size_t shared_bytes_ = 0;
    std::size_t peak_shared_bytes_ = 0;
};

}  // namespace uocr
