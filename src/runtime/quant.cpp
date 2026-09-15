#include "uocr/quant.h"

#include <algorithm>
#include <cmath>

namespace uocr {

std::size_t QuantizedMatrix::packed_bytes() const {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>((cols + 1) / 2);
}

float QuantizedMatrix::at(int r, int c) const {
    const int g = c / group_size;
    const int packed_row = (cols + 1) / 2;
    const std::size_t idx = static_cast<std::size_t>(r) * packed_row + c / 2;
    const std::uint8_t byte = packed[idx];
    const int q = (c & 1) ? (byte >> 4) : (byte & 0x0f);
    const std::size_t gi = static_cast<std::size_t>(r) * n_groups() + g;
    return (static_cast<float>(q) - zeros[gi]) * scales[gi];
}

void QuantizedMatrix::dequantize(std::vector<float>& out) const {
    out.resize(static_cast<std::size_t>(rows) * cols);
    const int packed_row = (cols + 1) / 2;
    const int ng = n_groups();
    for (int r = 0; r < rows; ++r) {
        const std::uint8_t* prow = packed.data() + static_cast<std::size_t>(r) * packed_row;
        float* orow = out.data() + static_cast<std::size_t>(r) * cols;
        const float* srow = scales.data() + static_cast<std::size_t>(r) * ng;
        const float* zrow = zeros.data() + static_cast<std::size_t>(r) * ng;
        for (int c = 0; c < cols; ++c) {
            const std::uint8_t byte = prow[c / 2];
            const int q = (c & 1) ? (byte >> 4) : (byte & 0x0f);
            orow[c] = (static_cast<float>(q) - zrow[c / group_size]) * srow[c / group_size];
        }
    }
}

QuantizedMatrix quantize_int4_awq(const float* w, int rows, int cols, int group_size) {
    UOCR_CHECK(group_size > 0, "group_size must be > 0");
    QuantizedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.group_size = group_size;
    const int ng = q.n_groups();
    q.scales.assign(static_cast<std::size_t>(rows) * ng, 0.0f);
    q.zeros.assign(static_cast<std::size_t>(rows) * ng, 0.0f);
    q.packed.assign(q.packed_bytes(), 0);

    const int packed_row = (cols + 1) / 2;
    for (int r = 0; r < rows; ++r) {
        const float* wrow = w + static_cast<std::size_t>(r) * cols;
        std::uint8_t* prow = q.packed.data() + static_cast<std::size_t>(r) * packed_row;
        for (int g = 0; g < ng; ++g) {
            const int c0 = g * group_size;
            const int c1 = std::min(c0 + group_size, cols);
            float mn = wrow[c0], mx = wrow[c0];
            for (int c = c0 + 1; c < c1; ++c) {
                mn = std::min(mn, wrow[c]);
                mx = std::max(mx, wrow[c]);
            }
            float scale = (mx - mn) / 15.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            float zero = std::round(-mn / scale);
            zero = std::clamp(zero, 0.0f, 15.0f);
            const std::size_t gi = static_cast<std::size_t>(r) * ng + g;
            q.scales[gi] = scale;
            q.zeros[gi] = zero;
            for (int c = c0; c < c1; ++c) {
                int v = static_cast<int>(std::lround(wrow[c] / scale + zero));
                v = std::clamp(v, 0, 15);
                if (c & 1)
                    prow[c / 2] = static_cast<std::uint8_t>((prow[c / 2] & 0x0f) | (v << 4));
                else
                    prow[c / 2] = static_cast<std::uint8_t>((prow[c / 2] & 0xf0) | v);
            }
        }
    }
    return q;
}

QuantizedMatrix quantize_int4_symmetric(const float* w, int rows, int cols, int group_size) {
    UOCR_CHECK(group_size > 0, "group_size must be > 0");
    QuantizedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.group_size = group_size;
    const int ng = q.n_groups();
    q.scales.assign(static_cast<std::size_t>(rows) * ng, 0.0f);
    q.zeros.assign(static_cast<std::size_t>(rows) * ng, 8.0f);  // fixed midpoint
    q.packed.assign(q.packed_bytes(), 0);

    const int packed_row = (cols + 1) / 2;
    for (int r = 0; r < rows; ++r) {
        const float* wrow = w + static_cast<std::size_t>(r) * cols;
        std::uint8_t* prow = q.packed.data() + static_cast<std::size_t>(r) * packed_row;
        for (int g = 0; g < ng; ++g) {
            const int c0 = g * group_size;
            const int c1 = std::min(c0 + group_size, cols);
            float amax = 0.0f;
            for (int c = c0; c < c1; ++c) amax = std::max(amax, std::fabs(wrow[c]));
            float scale = amax / 7.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            q.scales[static_cast<std::size_t>(r) * ng + g] = scale;
            for (int c = c0; c < c1; ++c) {
                int v = static_cast<int>(std::lround(wrow[c] / scale)) + 8;
                v = std::clamp(v, 0, 15);
                if (c & 1)
                    prow[c / 2] = static_cast<std::uint8_t>((prow[c / 2] & 0x0f) | (v << 4));
                else
                    prow[c / 2] = static_cast<std::uint8_t>((prow[c / 2] & 0xf0) | v);
            }
        }
    }
    return q;
}

}  // namespace uocr
