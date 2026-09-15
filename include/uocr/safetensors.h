#pragma once

// Read-only memory-mapped access to HuggingFace safetensors files.
//
// The format is a little-endian u64 header length, a JSON header describing
// every tensor and a flat data section.  We mmap the file so multi-GB
// checkpoints can be inspected without copying.

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "uocr/common.h"

namespace uocr {

class SafetensorsFile {
public:
    struct Info {
        DType dtype = DType::F32;
        std::vector<i64> shape;
        const void* data = nullptr;   // pointer into the mapping
        std::size_t nbytes = 0;
        std::size_t begin = 0;        // offset inside the data section
        std::size_t end = 0;
    };

    explicit SafetensorsFile(const std::string& path);
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    bool contains(const std::string& name) const;
    const Info& info(const std::string& name) const;
    const std::vector<std::string>& names() const { return names_; }

    std::size_t file_size() const { return file_size_; }
    const std::string& path() const { return path_; }

    // Decode a tensor to row-major float32.
    std::vector<float> read_f32(const std::string& name) const;

private:
    std::string path_;
    void* map_ = nullptr;
    std::size_t file_size_ = 0;
    std::size_t data_start_ = 0;
    std::unordered_map<std::string, Info> tensors_;
    std::vector<std::string> names_;
};

}  // namespace uocr
