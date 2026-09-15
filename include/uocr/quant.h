#pragma once

// Group-wise affine INT4 quantization (AWQ-style: per-group scale + zero point).
//
// Layout: weight matrix [rows, cols] is quantized in row-major order with
// `group_size` consecutive columns sharing one scale/zero.  Packed values are
// stored two per byte (low nibble = even column).

#include <cstdint>
#include <vector>

#include "uocr/common.h"

namespace uocr {

struct QuantizedMatrix {
    int rows = 0;
    int cols = 0;
    int group_size = 128;
    std::vector<std::uint8_t> packed;  // rows * ceil(cols/2) bytes
    std::vector<float> scales;         // rows * n_groups
    std::vector<float> zeros;          // rows * n_groups

    int n_groups() const { return (cols + group_size - 1) / group_size; }
    std::size_t packed_bytes() const;

    // Decode a single value (slow, for reference/debug).
    float at(int r, int c) const;

    // Decode the full matrix into row-major floats.
    void dequantize(std::vector<float>& out) const;
};

// Symmetric/asymmetric affine group quantization of `w` ([rows, cols] row-major
// floats) into 4-bit values.
QuantizedMatrix quantize_int4_awq(const float* w, int rows, int cols, int group_size = 128);

// Purely symmetric variant (zero point fixed to 8) useful for kernels that do
// not support zero points.
QuantizedMatrix quantize_int4_symmetric(const float* w, int rows, int cols, int group_size = 128);

}  // namespace uocr
