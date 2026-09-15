#pragma once

// A simple offset-based arena allocator with coalescing free list.  The KV
// cache and activation workspace are carved out of it so the engine can reason
// about a fixed memory budget and report fragmentation instead of relying on
// cudaMalloc/malloc during steady-state decoding.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "uocr/common.h"

namespace uocr {

class MemoryPool {
public:
    struct Block {
        std::size_t offset = 0;
        std::size_t size = 0;
    };

    explicit MemoryPool(std::size_t bytes = 0, std::size_t alignment = 256);

    void reserve(std::size_t bytes);
    void reset();

    // Returns the byte offset inside the arena, or -1 on failure.
    i64 allocate(std::size_t bytes, std::size_t alignment = 0);
    void free(i64 offset, std::size_t bytes);

    std::size_t total() const { return total_; }
    std::size_t used() const { return used_; }
    std::size_t peak() const { return peak_; }
    std::size_t alignment() const { return alignment_; }

    // 1 - (largest_free / total_free).  Zero when fully allocated.
    double fragmentation() const;
    std::size_t num_free_blocks() const { return free_list_.size(); }

    std::uint8_t* base() { return buf_.data(); }
    const std::uint8_t* base() const { return buf_.data(); }

private:
    std::size_t total_ = 0;
    std::size_t used_ = 0;
    std::size_t peak_ = 0;
    std::size_t alignment_ = 256;
    std::vector<std::uint8_t> buf_;
    std::vector<Block> free_list_;  // sorted by offset
};

}  // namespace uocr
