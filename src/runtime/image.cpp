#include "uocr/image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <vector>

#if defined(UOCR_HAVE_PNG)
#include <png.h>
#endif

#include "uocr/log.h"

namespace uocr {

namespace {

// Python's round() uses round-half-to-even; C++ lround rounds half away from
// zero.  Match Python to keep `ImageOps.pad` centering identical.
int py_round(double v) { return static_cast<int>(std::nearbyint(v)); }

double bicubic_filter(double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

struct Coeff {
    int start = 0;
    std::vector<double> w;  // normalized weights for consecutive inputs
};

// Pillow `precompute_coeffs` (bicubic, support=2).
std::vector<Coeff> precompute_coeffs(int in_size, int out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 2.0 * filterscale;
    std::vector<Coeff> out(static_cast<std::size_t>(out_size));
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        Coeff c;
        c.start = xmin;
        double sum = 0.0;
        for (int x = xmin; x < xmax; ++x) {
            const double w = bicubic_filter((x + 0.5 - center) / filterscale) / filterscale;
            if (w != 0.0) {
                c.w.push_back(w);
                sum += w;
            } else {
                c.w.push_back(0.0);
            }
        }
        if (sum != 0.0)
            for (double& w : c.w) w /= sum;
        out[static_cast<std::size_t>(xx)] = std::move(c);
    }
    return out;
}

std::uint8_t clip8(double v) {
    const int r = static_cast<int>(std::floor(v + 0.5));
    if (r < 0) return 0;
    if (r > 255) return 255;
    return static_cast<std::uint8_t>(r);
}

}  // namespace

ImageRGB load_ppm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    ImageRGB img;
    if (!f.good()) return img;
    std::string magic;
    f >> magic;
    if (magic != "P6") return img;
    auto skip = [&]() {
        while (f.good()) {
            int c = f.peek();
            if (c == '#') {
                std::string line;
                std::getline(f, line);
            } else if (std::isspace(c)) {
                f.get();
            } else {
                break;
            }
        }
    };
    int w = 0, h = 0, maxv = 0;
    skip();
    f >> w;
    skip();
    f >> h;
    skip();
    f >> maxv;
    f.get();  // single whitespace after maxval
    if (!f.good() || w <= 0 || h <= 0 || maxv != 255) return ImageRGB{};
    img.width = w;
    img.height = h;
    img.pixels.resize(static_cast<std::size_t>(w) * h * 3);
    f.read(reinterpret_cast<char*>(img.pixels.data()),
           static_cast<std::streamsize>(img.pixels.size()));
    if (!f) return ImageRGB{};
    return img;
}

bool save_ppm(const std::string& path, const ImageRGB& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f.good() || img.empty()) return false;
    f << "P6\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.pixels.data()),
            static_cast<std::streamsize>(img.pixels.size()));
    return f.good();
}

ImageRGB load_png(const std::string& path) {
#if defined(UOCR_HAVE_PNG)
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return ImageRGB{};
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { std::fclose(fp); return ImageRGB{}; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); std::fclose(fp); return ImageRGB{}; }
    ImageRGB img;
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return ImageRGB{};
    }
    png_init_io(png, fp);
    png_read_info(png, info);
    const int w = static_cast<int>(png_get_image_width(png, info));
    const int h = static_cast<int>(png_get_image_height(png, info));
    const png_byte color = png_get_color_type(png, info);
    const png_byte depth = png_get_bit_depth(png, info);
    if (depth == 16) png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    png_set_strip_alpha(png);
    png_read_update_info(png, info);
    if (w <= 0 || h <= 0) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return ImageRGB{};
    }
    img.width = w;
    img.height = h;
    img.pixels.resize(static_cast<std::size_t>(w) * h * 3);
    std::vector<png_bytep> rows(static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y)
        rows[static_cast<std::size_t>(y)] =
            img.pixels.data() + static_cast<std::size_t>(y) * w * 3;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    return img;
#else
    (void)path;
    return ImageRGB{};
#endif
}

ImageRGB load_image(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return ImageRGB{};
    unsigned char sig[8] = {};
    f.read(reinterpret_cast<char*>(sig), sizeof(sig));
    f.close();
    static const unsigned char kPng[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    if (std::memcmp(sig, kPng, sizeof(kPng)) == 0) return load_png(path);
    return load_ppm(path);
}

ImageRGB resize_bicubic(const ImageRGB& src, int out_w, int out_h) {
    if (src.empty() || out_w <= 0 || out_h <= 0) return ImageRGB{};
    if (out_w == src.width && out_h == src.height) return src;

    const int in_w = src.width, in_h = src.height;
    const int ch = 3;

    // horizontal pass: [in_h, out_w, ch], rounded to uint8 (Pillow temp image)
    std::vector<Coeff> hc = precompute_coeffs(in_w, out_w);
    std::vector<std::uint8_t> tmp(static_cast<std::size_t>(in_h) * out_w * ch);
    for (int y = 0; y < in_h; ++y) {
        const std::uint8_t* row = src.pixels.data() + static_cast<std::size_t>(y) * in_w * ch;
        for (int ox = 0; ox < out_w; ++ox) {
            const Coeff& c = hc[static_cast<std::size_t>(ox)];
            double acc[3] = {0, 0, 0};
            for (std::size_t k = 0; k < c.w.size(); ++k) {
                const double w = c.w[k];
                if (w == 0.0) continue;
                const std::uint8_t* p = row + static_cast<std::size_t>(c.start + static_cast<int>(k)) * ch;
                acc[0] += p[0] * w;
                acc[1] += p[1] * w;
                acc[2] += p[2] * w;
            }
            std::uint8_t* o = tmp.data() + (static_cast<std::size_t>(y) * out_w + ox) * ch;
            o[0] = clip8(acc[0]);
            o[1] = clip8(acc[1]);
            o[2] = clip8(acc[2]);
        }
    }

    // vertical pass
    ImageRGB dst;
    dst.width = out_w;
    dst.height = out_h;
    dst.pixels.resize(static_cast<std::size_t>(out_w) * out_h * ch);
    std::vector<Coeff> vc = precompute_coeffs(in_h, out_h);
    for (int oy = 0; oy < out_h; ++oy) {
        const Coeff& c = vc[static_cast<std::size_t>(oy)];
        for (int x = 0; x < out_w; ++x) {
            double acc[3] = {0, 0, 0};
            for (std::size_t k = 0; k < c.w.size(); ++k) {
                const double w = c.w[k];
                if (w == 0.0) continue;
                const std::uint8_t* p = tmp.data() +
                                        (static_cast<std::size_t>(c.start + static_cast<int>(k)) * out_w + x) * ch;
                acc[0] += p[0] * w;
                acc[1] += p[1] * w;
                acc[2] += p[2] * w;
            }
            std::uint8_t* o = dst.pixels.data() + (static_cast<std::size_t>(oy) * out_w + x) * ch;
            o[0] = clip8(acc[0]);
            o[1] = clip8(acc[1]);
            o[2] = clip8(acc[2]);
        }
    }
    return dst;
}

ImageRGB pad_square(const ImageRGB& src, int size, std::uint8_t pad) {
    if (src.empty() || size <= 0) return ImageRGB{};
    // contain(): preserve aspect, fit inside (size,size)
    const double im_ratio = static_cast<double>(src.width) / src.height;
    const double dest_ratio = 1.0;
    int rw = src.width, rh = src.height;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            int new_h = static_cast<int>(py_round(static_cast<double>(src.height) / src.width * size));
            if (new_h != size) {
                rw = size;
                rh = new_h;
            }
        } else {
            int new_w = static_cast<int>(py_round(static_cast<double>(src.width) / src.height * size));
            if (new_w != size) {
                rw = new_w;
                rh = size;
            }
        }
    } else {
        rw = size;
        rh = size;
    }
    ImageRGB resized = resize_bicubic(src, rw, rh);

    ImageRGB out;
    out.width = size;
    out.height = size;
    out.pixels.assign(static_cast<std::size_t>(size) * size * 3, pad);
    int x0 = 0, y0 = 0;
    if (rw != size)
        x0 = static_cast<int>(py_round((size - rw) * 0.5));
    else
        y0 = static_cast<int>(py_round((size - rh) * 0.5));
    for (int y = 0; y < rh; ++y) {
        const std::uint8_t* srow = resized.pixels.data() + static_cast<std::size_t>(y) * rw * 3;
        std::uint8_t* drow = out.pixels.data() + (static_cast<std::size_t>(y + y0) * size + x0) * 3;
        std::memcpy(drow, srow, static_cast<std::size_t>(rw) * 3);
    }
    return out;
}

namespace {

std::pair<int, int> find_closest_aspect_ratio(double aspect_ratio,
                                              const std::vector<std::pair<int, int>>& ratios,
                                              int width, int height, int image_size) {
    double best_diff = std::numeric_limits<double>::infinity();
    std::pair<int, int> best{1, 1};
    const double area = static_cast<double>(width) * height;
    for (const auto& r : ratios) {
        const double target = static_cast<double>(r.first) / r.second;
        const double diff = std::fabs(aspect_ratio - target);
        if (diff < best_diff) {
            best_diff = diff;
            best = r;
        } else if (diff == best_diff) {
            if (area > 0.5 * image_size * image_size * r.first * r.second) best = r;
        }
    }
    return best;
}

}  // namespace

DynamicPreprocess dynamic_preprocess(const ImageRGB& src, int image_size, int min_num, int max_num,
                                     bool use_thumbnail) {
    DynamicPreprocess out;
    if (src.empty()) return out;

    const double aspect_ratio = static_cast<double>(src.width) / src.height;
    std::set<std::pair<int, int>> ratio_set;
    for (int n = min_num; n <= max_num; ++n)
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                if (i * j <= max_num && i * j >= min_num) ratio_set.insert({i, j});
    std::vector<std::pair<int, int>> target_ratios(ratio_set.begin(), ratio_set.end());
    std::stable_sort(target_ratios.begin(), target_ratios.end(),
                     [](const auto& a, const auto& b) { return a.first * a.second < b.first * b.second; });

    auto ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, src.width, src.height, image_size);
    const int target_w = image_size * ratio.first;
    const int target_h = image_size * ratio.second;
    const int blocks = ratio.first * ratio.second;

    ImageRGB resized = resize_bicubic(src, target_w, target_h);
    const int cols = target_w / image_size;
    for (int i = 0; i < blocks; ++i) {
        const int bx = (i % cols) * image_size;
        const int by = (i / cols) * image_size;
        ImageRGB crop;
        crop.width = image_size;
        crop.height = image_size;
        crop.pixels.resize(static_cast<std::size_t>(image_size) * image_size * 3);
        for (int y = 0; y < image_size; ++y) {
            const std::uint8_t* s = resized.pixels.data() +
                                    (static_cast<std::size_t>(by + y) * target_w + bx) * 3;
            std::uint8_t* d = crop.pixels.data() + static_cast<std::size_t>(y) * image_size * 3;
            std::memcpy(d, s, static_cast<std::size_t>(image_size) * 3);
        }
        out.crops.push_back(std::move(crop));
    }
    if (use_thumbnail && out.crops.size() != 1) out.crops.push_back(resize_bicubic(src, image_size, image_size));

    out.width_crop_num = ratio.first;
    out.height_crop_num = ratio.second;
    return out;
}

std::vector<float> to_tensor_normalized(const ImageRGB& img) {
    std::vector<float> out;
    if (img.empty()) return out;
    const int h = img.height, w = img.width, ch = 3;
    out.resize(static_cast<std::size_t>(w) * h * ch);
    for (int c = 0; c < ch; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float v = img.pixels[(static_cast<std::size_t>(y) * w + x) * ch + c] / 255.0f;
                out[(static_cast<std::size_t>(c) * h + y) * w + x] = (v - 0.5f) / 0.5f;
            }
    return out;
}

}  // namespace uocr
