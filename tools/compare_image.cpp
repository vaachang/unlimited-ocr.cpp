// compare_image -- checks the C++ image pre-processing against the Pillow
// reference exported by tools/reference/export_image_cases.py (E2).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "uocr/common.h"
#include "uocr/image.h"

using namespace uocr;
using nlohmann::json;

namespace {

std::vector<float> load_f32(const std::string& path, std::size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    UOCR_CHECK(f.good(), "cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return v;
}

ImageRGB load_rgb(const std::string& path, int w, int h) {
    ImageRGB img;
    img.width = w;
    img.height = h;
    img.pixels.resize(static_cast<std::size_t>(w) * h * 3);
    std::ifstream f(path, std::ios::binary);
    UOCR_CHECK(f.good(), "cannot open " + path);
    f.read(reinterpret_cast<char*>(img.pixels.data()),
           static_cast<std::streamsize>(img.pixels.size()));
    return img;
}

void report(const std::string& name, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::printf("  %-16s SIZE MISMATCH ours=%zu ref=%zu\n", name.c_str(), a.size(), b.size());
        return;
    }
    double max_abs = 0, num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
        max_abs = std::max(max_abs, e);
        num += e * e;
        den += static_cast<double>(b[i]) * b[i];
    }
    std::printf("  %-16s max_abs=%.6g rel_l2=%.6g\n", name.c_str(), max_abs,
                den > 0 ? std::sqrt(num / den) : std::sqrt(num));
}

}  // namespace

int main(int argc, char** argv) {
    std::string ref_dir = "/tmp/opencode/ref_image";
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_dir = argv[++i];

    std::ifstream mf(ref_dir + "/manifest.json");
    UOCR_CHECK(mf.good(), "cannot open " + ref_dir + "/manifest.json");
    json manifest = json::parse(mf);
    const int base_size = manifest.value("base_size", 1024);
    const int image_size = manifest.value("image_size", 640);
    const std::uint8_t pad = 127;

    bool ok = true;
    for (const auto& c : manifest.at("cases")) {
        const std::string name = c.at("name").get<std::string>();
        const int w = c.at("width").get<int>();
        const int h = c.at("height").get<int>();
        ImageRGB src = load_rgb(ref_dir + "/" + name + "_image.bin", w, h);
        std::printf("case %s (%dx%d):\n", name.c_str(), w, h);

        // crop_mode=True global
        ImageRGB g = pad_square(src, base_size, pad);
        std::vector<float> g_ours = to_tensor_normalized(g);
        auto gc = c.at("global_crop_shape");
        std::vector<float> g_ref = load_f32(ref_dir + "/" + name + "_global_crop.bin",
                                            static_cast<std::size_t>(gc[0].get<int>()) *
                                                gc[1].get<int>() * gc[2].get<int>());
        report("global_crop", g_ours, g_ref);

        // crop_mode=False global
        ImageRGB sq = resize_bicubic(src, image_size, image_size);
        ImageRGB g2 = pad_square(sq, image_size, pad);
        std::vector<float> g2_ours = to_tensor_normalized(g2);
        auto gn = c.at("global_nocrop_shape");
        std::vector<float> g2_ref = load_f32(ref_dir + "/" + name + "_global_nocrop.bin",
                                             static_cast<std::size_t>(gn[0].get<int>()) *
                                                 gn[1].get<int>() * gn[2].get<int>());
        report("global_nocrop", g2_ours, g2_ref);

        // dynamic_preprocess local crops
        DynamicPreprocess dp = dynamic_preprocess(src, image_size);
        std::printf("  crop_ratio      ours=(%d,%d) ref=(%d,%d)\n", dp.width_crop_num,
                    dp.height_crop_num, c.at("crop_ratio")[0].get<int>(),
                    c.at("crop_ratio")[1].get<int>());
        if (dp.width_crop_num != c.at("crop_ratio")[0].get<int>() ||
            dp.height_crop_num != c.at("crop_ratio")[1].get<int>())
            ok = false;

        auto ls = c.at("local_shape");
        const std::size_t n_crops = ls[0].get<std::size_t>();
        const std::size_t per = static_cast<std::size_t>(ls[1].get<int>()) * ls[2].get<int>() *
                                ls[3].get<int>();
        std::vector<float> l_ref = load_f32(ref_dir + "/" + name + "_local.bin", n_crops * per);
        std::vector<float> l_ours;
        for (const ImageRGB& crop : dp.crops) {
            std::vector<float> t = to_tensor_normalized(crop);
            l_ours.insert(l_ours.end(), t.begin(), t.end());
        }
        report("local", l_ours, l_ref);
    }
    std::printf("\nimage preprocessing: %s\n", ok ? "ratios ok" : "RATIO MISMATCH");
    return ok ? 0 : 1;
}
