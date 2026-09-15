#pragma once

// Common definitions shared by every layer of the engine.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace uocr {

using i32 = std::int32_t;
using i64 = std::int64_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

enum class DType : std::uint8_t {
    F32 = 0,
    BF16 = 1,
    F16 = 2,
    I32 = 3,
    I8 = 4,
    U8 = 5,
    I4 = 6,   // packed 2 per byte
    U4 = 7,
};

std::size_t dtype_size(DType t);
const char* dtype_name(DType t);

// A small, dependency-free exception type so the error message always carries
// the originating file/line.
class Error : public std::runtime_error {
public:
    Error(const std::string& what, const char* file, int line)
        : std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + what) {}
};

#define UOCR_THROW(msg) ::uocr::Error((msg), __FILE__, __LINE__)
#define UOCR_CHECK(cond, msg)                        \
    do {                                             \
        if (!(cond)) UOCR_THROW(msg);                \
    } while (0)

// ---------------------------------------------------------------------------
// bfloat16 helpers (host side).  We deliberately avoid <cuda_bf16.h> here so
// that the header can be included from plain C++ translation units.
// ---------------------------------------------------------------------------
inline float bf16_to_f32(std::uint16_t h) {
    u32 bits = static_cast<u32>(h) << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

inline std::uint16_t f32_to_bf16(float f) {
    u32 bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    // round-to-nearest-even
    u32 lsb = (bits >> 16) & 1u;
    u32 rounding = 0x7fffu + lsb;
    bits = (bits + rounding) >> 16;
    return static_cast<std::uint16_t>(bits);
}

inline float f16_to_f32(std::uint16_t h) {
    const u32 sign = (h & 0x8000u) << 16;
    const u32 exp = (h >> 10) & 0x1fu;
    const u32 man = h & 0x3ffu;
    u32 bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            // subnormal
            int e = -1;
            u32 m = man;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            m &= 0x3ffu;
            bits = sign | static_cast<u32>(127 - 15 - e) << 23 | (m << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float dtype_to_f32(const void* p, DType t) {
    switch (t) {
        case DType::F32: {
            float v;
            __builtin_memcpy(&v, p, sizeof(v));
            return v;
        }
        case DType::BF16: {
            std::uint16_t v;
            __builtin_memcpy(&v, p, sizeof(v));
            return bf16_to_f32(v);
        }
        case DType::F16: {
            std::uint16_t v;
            __builtin_memcpy(&v, p, sizeof(v));
            return f16_to_f32(v);
        }
        default:
            UOCR_THROW("dtype_to_f32: unsupported dtype");
    }
}

}  // namespace uocr
