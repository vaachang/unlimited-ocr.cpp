#include "uocr/memory_pool.h"

#include <algorithm>

namespace uocr {

MemoryPool::MemoryPool(std::size_t bytes, std::size_t alignment) : alignment_(alignment) {
    if (bytes > 0) reserve(bytes);
}

void MemoryPool::reserve(std::size_t bytes) {
    buf_.assign(bytes, 0);
    total_ = bytes;
    used_ = 0;
    peak_ = 0;
    free_list_.clear();
    free_list_.push_back({0, bytes});
}

void MemoryPool::reset() {
    used_ = 0;
    peak_ = 0;
    free_list_.clear();
    free_list_.push_back({0, total_});
}

i64 MemoryPool::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) return 0;
    std::size_t align = alignment == 0 ? alignment_ : alignment;
    if (align == 0) align = 1;

    // First fit, preserving alignment of the returned offset.
    for (std::size_t i = 0; i < free_list_.size(); ++i) {
        const std::size_t start = free_list_[i].offset;
        const std::size_t size = free_list_[i].size;
        const std::size_t pad = (align - (start % align)) % align;
        if (size < pad + bytes) continue;

        const std::size_t off = start + pad;
        free_list_.erase(free_list_.begin() + static_cast<long>(i));
        if (pad > 0)
            free_list_.insert(free_list_.begin() + static_cast<long>(i), {start, pad});
        if (size > pad + bytes)
            free_list_.insert(free_list_.begin() + static_cast<long>(i) + (pad > 0 ? 1 : 0),
                              {off + bytes, size - pad - bytes});

        used_ += pad + bytes;
        peak_ = std::max(peak_, used_);
        return static_cast<i64>(off);
    }
    return -1;
}

void MemoryPool::free(i64 offset, std::size_t bytes) {
    if (bytes == 0) return;
    Block b{static_cast<std::size_t>(offset), bytes};
    auto it = std::lower_bound(free_list_.begin(), free_list_.end(), b,
                               [](const Block& a, const Block& x) { return a.offset < x.offset; });
    it = free_list_.insert(it, b);
    if (used_ >= bytes) used_ -= bytes;

    // coalesce with next then previous
    auto coalesce = [&](std::size_t idx) {
        if (idx + 1 < free_list_.size() &&
            free_list_[idx].offset + free_list_[idx].size == free_list_[idx + 1].offset) {
            free_list_[idx].size += free_list_[idx + 1].size;
            free_list_.erase(free_list_.begin() + static_cast<long>(idx) + 1);
            return true;
        }
        return false;
    };
    const std::size_t idx = static_cast<std::size_t>(it - free_list_.begin());
    coalesce(idx);
    if (idx > 0) coalesce(idx - 1);
}

double MemoryPool::fragmentation() const {
    std::size_t free_total = 0, largest = 0;
    for (const auto& b : free_list_) {
        free_total += b.size;
        largest = std::max(largest, b.size);
    }
    if (free_total == 0) return 0.0;
    return 1.0 - static_cast<double>(largest) / static_cast<double>(free_total);
}

}  // namespace uocr
