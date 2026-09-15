#include "uocr/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

#include <nlohmann/json.hpp>

namespace uocr {

namespace {

DType parse_dtype(const std::string& s) {
    if (s == "F32" || s == "FLOAT") return DType::F32;
    if (s == "BF16" || s == "BFLOAT16") return DType::BF16;
    if (s == "F16" || s == "FLOAT16") return DType::F16;
    if (s == "I32" || s == "INT32") return DType::I32;
    if (s == "I8" || s == "INT8") return DType::I8;
    if (s == "U8" || s == "UINT8") return DType::U8;
    if (s == "I4") return DType::I4;
    if (s == "U4") return DType::U4;
    UOCR_THROW("unsupported safetensors dtype: " + s);
    return DType::F32;  // unreachable
}

}  // namespace

SafetensorsFile::SafetensorsFile(const std::string& path) : path_(path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    UOCR_CHECK(fd >= 0, "cannot open safetensors file: " + path);

    struct stat st{};
    UOCR_CHECK(::fstat(fd, &st) == 0, "fstat failed: " + path);
    file_size_ = static_cast<std::size_t>(st.st_size);

    map_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    UOCR_CHECK(map_ != MAP_FAILED, "mmap failed: " + path);

    std::uint64_t header_len = 0;
    std::memcpy(&header_len, map_, sizeof(header_len));
    const std::size_t data_start = 8 + static_cast<std::size_t>(header_len);
    UOCR_CHECK(data_start <= file_size_, "safetensors header out of range");

    std::string header(static_cast<const char*>(map_) + 8, static_cast<std::size_t>(header_len));
    auto root = nlohmann::json::parse(header);

    data_start_ = data_start;
    for (auto it = root.begin(); it != root.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        Info info;
        info.dtype = parse_dtype(it.value().at("dtype").get<std::string>());
        for (const auto& d : it.value().at("shape")) info.shape.push_back(d.get<i64>());
        const auto& offs = it.value().at("data_offsets");
        info.begin = offs[0].get<std::size_t>();
        info.end = offs[1].get<std::size_t>();
        info.nbytes = info.end - info.begin;
        UOCR_CHECK(data_start_ + info.end <= file_size_, "tensor beyond file end: " + it.key());
        info.data = static_cast<const char*>(map_) + data_start_ + info.begin;
        names_.push_back(it.key());
        tensors_.emplace(it.key(), std::move(info));
    }
}

SafetensorsFile::~SafetensorsFile() {
    if (map_ && map_ != MAP_FAILED) ::munmap(map_, file_size_);
}

bool SafetensorsFile::contains(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const SafetensorsFile::Info& SafetensorsFile::info(const std::string& name) const {
    auto it = tensors_.find(name);
    UOCR_CHECK(it != tensors_.end(), "tensor not found: " + name);
    return it->second;
}

std::vector<float> SafetensorsFile::read_f32(const std::string& name) const {
    const Info& in = info(name);
    std::size_t n = 1;
    for (i64 d : in.shape) n *= static_cast<std::size_t>(d);
    std::vector<float> out(n);

    switch (in.dtype) {
        case DType::F32:
            std::memcpy(out.data(), in.data, n * sizeof(float));
            break;
        case DType::BF16: {
            const auto* p = static_cast<const std::uint16_t*>(in.data);
            for (std::size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
            break;
        }
        case DType::F16: {
            const auto* p = static_cast<const std::uint16_t*>(in.data);
            for (std::size_t i = 0; i < n; ++i) out[i] = f16_to_f32(p[i]);
            break;
        }
        case DType::I32: {
            const auto* p = static_cast<const std::int32_t*>(in.data);
            for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DType::I8: {
            const auto* p = static_cast<const std::int8_t*>(in.data);
            for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        case DType::U8: {
            const auto* p = static_cast<const std::uint8_t*>(in.data);
            for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(p[i]);
            break;
        }
        default:
            UOCR_THROW("read_f32: unsupported dtype for " + name);
    }
    return out;
}

}  // namespace uocr
