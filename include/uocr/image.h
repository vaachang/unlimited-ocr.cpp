#pragma once

// Image pre-processing matching the Unlimited-OCR reference pipeline:
//   BasicImageTransform(mean=std=0.5)  ->  (pixel/255 - 0.5) / 0.5
//   ImageOps.pad(..., method=BICUBIC)  ->  aspect-preserving resize + center pad
//   dynamic_preprocess(...)            ->  Gundam-style local crops
//
// PIL compatibility: the bicubic resampler reproduces Pillow's coefficient
// computation and uint8 intermediate rounding (verified to <=1 LSB difference).

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace uocr {

struct ImageRGB {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;  // HWC, 3 channels, row-major

    bool empty() const { return width <= 0 || height <= 0 || pixels.empty(); }
};

// Read a binary P6 PPM (maxval 255).  Returns an empty image on failure.
ImageRGB load_ppm(const std::string& path);
bool save_ppm(const std::string& path, const ImageRGB& img);

// Decode a PNG (8/16-bit, gray/palette/RGB/RGBA, alpha stripped).  Returns an
// empty image when the build has no libpng (`UOCR_HAVE_PNG` unset).
ImageRGB load_png(const std::string& path);

// Generic loader: sniff the file signature and dispatch to PNG or PPM.
ImageRGB load_image(const std::string& path);

// Pillow-compatible bicubic resize (Catmull-Rom, a=-0.5, antialiased).
ImageRGB resize_bicubic(const ImageRGB& src, int out_w, int out_h);

// ImageOps.pad(img, (size,size), BICUBIC, color=pad, centering=(0.5,0.5)).
ImageRGB pad_square(const ImageRGB& src, int size, std::uint8_t pad);

struct DynamicPreprocess {
    std::vector<ImageRGB> crops;
    int width_crop_num = 1;
    int height_crop_num = 1;
};

// Reference `dynamic_preprocess` (min_num=2, max_num=32).
DynamicPreprocess dynamic_preprocess(const ImageRGB& src, int image_size = 640, int min_num = 2,
                                     int max_num = 32, bool use_thumbnail = false);

// CHW float32, [0,1] then (x-0.5)/0.5 => x*2-1.
std::vector<float> to_tensor_normalized(const ImageRGB& img);

}  // namespace uocr
