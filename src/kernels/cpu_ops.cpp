#include "uocr/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace uocr {
namespace ops {

void matmul_t(const float* x, const float* w, const float* bias, float* y,
              int m, int n, int k) {
#pragma omp parallel for schedule(static) if (static_cast<long long>(m) * n * k > 1 << 18)
    for (int i = 0; i < m; ++i) {
        const float* xrow = x + static_cast<std::size_t>(i) * k;
        float* yrow = y + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) {
            const float* wrow = w + static_cast<std::size_t>(j) * k;
            float acc = bias ? bias[j] : 0.0f;
            for (int t = 0; t < k; ++t) acc += xrow[t] * wrow[t];
            yrow[j] = acc;
        }
    }
}

void matmul_t_bf16(const float* x, const std::uint16_t* w, const float* bias, float* y,
                   int m, int n, int k) {
#pragma omp parallel for schedule(static) if (static_cast<long long>(m) * n * k > 1 << 18)
    for (int i = 0; i < m; ++i) {
        const float* xrow = x + static_cast<std::size_t>(i) * k;
        float* yrow = y + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) {
            const std::uint16_t* wrow = w + static_cast<std::size_t>(j) * k;
            float acc = bias ? bias[j] : 0.0f;
            for (int t = 0; t < k; ++t) acc += xrow[t] * bf16_to_f32(wrow[t]);
            yrow[j] = acc;
        }
    }
}

void matmul_nn(const float* x, const float* w, const float* bias, float* y,
               int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        float* yrow = y + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) yrow[j] = bias ? bias[j] : 0.0f;
        for (int t = 0; t < k; ++t) {
            const float xt = x[static_cast<std::size_t>(i) * k + t];
            const float* wrow = w + static_cast<std::size_t>(t) * n;
            for (int j = 0; j < n; ++j) yrow[j] += xt * wrow[j];
        }
    }
}

void add_inplace(float* a, const float* b, i64 n) {
    for (i64 i = 0; i < n; ++i) a[i] += b[i];
}

void rmsnorm(const float* x, const float* weight, float* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xrow = x + static_cast<std::size_t>(r) * cols;
        float ss = 0.0f;
        for (int c = 0; c < cols; ++c) ss += xrow[c] * xrow[c];
        const float inv = 1.0f / std::sqrt(ss / cols + eps);
        float* orow = out + static_cast<std::size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) orow[c] = xrow[c] * inv * weight[c];
    }
}

void layernorm(const float* x, const float* weight, const float* bias, float* out,
               int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* xrow = x + static_cast<std::size_t>(r) * cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += xrow[c];
        mean /= cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            const float d = xrow[c] - mean;
            var += d * d;
        }
        var /= cols;
        const float inv = 1.0f / std::sqrt(var + eps);
        float* orow = out + static_cast<std::size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            float v = (xrow[c] - mean) * inv;
            if (weight) v *= weight[c];
            if (bias) v += bias[c];
            orow[c] = v;
        }
    }
}

void softmax(float* x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float* row = x + static_cast<std::size_t>(r) * cols;
        float mx = row[0];
        for (int c = 1; c < cols; ++c) mx = std::max(mx, row[c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - mx);
            sum += row[c];
        }
        const float inv = 1.0f / sum;
        for (int c = 0; c < cols; ++c) row[c] *= inv;
    }
}

void silu_mul(const float* gate, const float* up, float* out, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

void gelu(float* x, i64 n) {
    // Exact (erf-based) GELU, matching PyTorch nn.GELU() with the default
    // approximate="none" used by the reference SAM ViT.
    constexpr float kInvSqrt2 = 0.7071067811865476f;
#pragma omp parallel for if (n > 4096)
    for (i64 i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::erff(v * kInvSqrt2));
    }
}

void quick_gelu(float* x, i64 n) {
    for (i64 i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = v / (1.0f + std::exp(-1.702f * v));
    }
}

void rope_tables(int head_dim, int max_seq, float theta, float* cos_out, float* sin_out) {
    const int half = head_dim / 2;
    for (int p = 0; p < max_seq; ++p) {
        for (int i = 0; i < half; ++i) {
            const float freq = std::pow(theta, -2.0f * i / static_cast<float>(head_dim));
            const float angle = p * freq;
            cos_out[static_cast<std::size_t>(p) * half + i] = std::cos(angle);
            sin_out[static_cast<std::size_t>(p) * half + i] = std::sin(angle);
        }
    }
}

namespace {
void rope_row(float* v, const float* cos, const float* sin, int head_dim) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float x0 = v[i];
        const float x1 = v[i + half];
        v[i] = x0 * cos[i] - x1 * sin[i];
        v[i + half] = x0 * sin[i] + x1 * cos[i];
    }
}
}  // namespace

void rope(float* q, float* k, const int* positions, int seq, int n_heads, int n_kv_heads,
          int head_dim, float theta) {
    const int half = head_dim / 2;
    std::vector<float> cbuf(static_cast<std::size_t>(half)), sbuf(static_cast<std::size_t>(half));
    for (int s = 0; s < seq; ++s) {
        const int p = positions ? positions[s] : s;
        for (int i = 0; i < half; ++i) {
            const float freq = std::pow(theta, -2.0f * i / static_cast<float>(head_dim));
            const float angle = p * freq;
            cbuf[i] = std::cos(angle);
            sbuf[i] = std::sin(angle);
        }
        float* qrow = q + static_cast<std::size_t>(s) * n_heads * head_dim;
        for (int h = 0; h < n_heads; ++h)
            rope_row(qrow + static_cast<std::size_t>(h) * head_dim, cbuf.data(), sbuf.data(), head_dim);
        if (k) {
            float* krow = k + static_cast<std::size_t>(s) * n_kv_heads * head_dim;
            for (int h = 0; h < n_kv_heads; ++h)
                rope_row(krow + static_cast<std::size_t>(h) * head_dim, cbuf.data(), sbuf.data(),
                         head_dim);
        }
    }
}

}  // namespace ops
}  // namespace uocr
