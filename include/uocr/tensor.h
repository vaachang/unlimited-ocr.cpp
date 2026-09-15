#pragma once

// A minimal row-major N-dimensional float tensor used by the CPU reference
// backend.  It intentionally mirrors the subset of torch semantics the engine
// needs: shape(), numel(), indexing and strided views.  The CUDA backend keeps
// the same logical layout, only the storage lives on the device.

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "uocr/common.h"

namespace uocr {

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(std::vector<i64> shape, float fill = 0.0f)
        : shape_(std::move(shape)) {
        i64 n = 1;
        for (i64 d : shape_) n *= d;
        numel_ = n;
        data_.assign(static_cast<std::size_t>(n), fill);
    }

    static Tensor zeros(std::vector<i64> shape) { return Tensor(std::move(shape), 0.0f); }
    static Tensor empty(std::vector<i64> shape);
    static Tensor from_data(std::vector<i64> shape, std::vector<float> data) {
        Tensor t;
        t.shape_ = std::move(shape);
        i64 n = 1;
        for (i64 d : t.shape_) n *= d;
        UOCR_CHECK(static_cast<i64>(data.size()) == n, "Tensor::from_data size mismatch");
        t.numel_ = n;
        t.data_ = std::move(data);
        return t;
    }

    const std::vector<i64>& shape() const { return shape_; }
    i64 dim(int i) const { return shape_[static_cast<std::size_t>(i)]; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    i64 numel() const { return numel_; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    std::vector<float>& vec() { return data_; }
    const std::vector<float>& vec() const { return data_; }

    float& operator[](i64 i) { return data_[static_cast<std::size_t>(i)]; }
    float operator[](i64 i) const { return data_[static_cast<std::size_t>(i)]; }

    bool empty() const { return numel_ == 0; }

private:
    std::vector<i64> shape_{};
    i64 numel_ = 0;
    std::vector<float> data_{};
};

// A non-owning typed view used to exchange data with kernels (CPU or CUDA).
struct TensorView {
    float* data = nullptr;
    std::vector<i64> shape;

    i64 numel() const {
        i64 n = 1;
        for (i64 d : shape) n *= d;
        return n;
    }
};

}  // namespace uocr
