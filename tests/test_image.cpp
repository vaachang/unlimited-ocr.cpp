#include "test_main.h"
#include "uocr/image.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace uocr;

namespace {

// Minimal PNGs generated with Pillow, embedded so the loader is tested without
// depending on an external file or on writing PNG (we only have a PPM writer).
// Palette PNG, 3x2, palette [red, green, blue], data [0,1,2, 2,1,0].
const unsigned char kPalPng[] = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 3, 0, 0, 0, 2, 2, 3, 0, 0,
    0, 224, 26, 142, 137, 0, 0, 0, 9, 80, 76, 84, 69, 255, 0, 0, 0, 255, 0, 0, 0, 255, 45, 74, 205,
    138, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 144, 96, 152, 0, 0, 0, 220, 0, 169, 108, 231,
    192, 35, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
};
// Grayscale PNG, 2x2, values [[0, 85], [170, 255]].
const unsigned char kGrayPng[] = {
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 0, 0, 0,
    0, 87, 221, 82, 248, 0, 0, 0, 14, 73, 68, 65, 84, 120, 156, 99, 96, 8, 101, 88, 245, 31, 0, 3,
    173, 1, 255, 103, 251, 202, 9, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
};

std::filesystem::path write_temp(const char* name, const unsigned char* data, std::size_t n) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    return p;
}

}  // namespace

UOCR_TEST(image_ppm_roundtrip) {
    ImageRGB img;
    img.width = 3;
    img.height = 2;
    img.pixels = {10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::filesystem::path p =
        std::filesystem::temp_directory_path() / "uocr_test_roundtrip.ppm";
    CHECK(save_ppm(p.string(), img));
    ImageRGB got = load_image(p.string());
    CHECK_EQ(got.width, 3);
    CHECK_EQ(got.height, 2);
    CHECK(got.pixels == img.pixels);
    std::filesystem::remove(p);
}

UOCR_TEST(image_load_png_palette) {
    const std::filesystem::path p = write_temp("uocr_test_pal.png", kPalPng, sizeof(kPalPng));
    ImageRGB got = load_image(p.string());
    std::filesystem::remove(p);
    if (got.empty()) return;  // PNG support not compiled in
    CHECK_EQ(got.width, 3);
    CHECK_EQ(got.height, 2);
    const std::uint8_t exp[18] = {255, 0,  0,   0,   255, 0,   0,   0,   255,
                                  0,   0,  255, 0,   255, 0,   255, 0,   0};
    for (int i = 0; i < 18; ++i) CHECK_EQ(static_cast<int>(got.pixels[i]), static_cast<int>(exp[i]));
}

UOCR_TEST(image_load_png_grayscale) {
    const std::filesystem::path p = write_temp("uocr_test_gray.png", kGrayPng, sizeof(kGrayPng));
    ImageRGB got = load_image(p.string());
    std::filesystem::remove(p);
    if (got.empty()) return;  // PNG support not compiled in
    CHECK_EQ(got.width, 2);
    CHECK_EQ(got.height, 2);
    // gray value v -> (v, v, v)
    const std::uint8_t gray[4] = {0, 85, 170, 255};
    for (int i = 0; i < 4; ++i)
        for (int c = 0; c < 3; ++c)
            CHECK_EQ(static_cast<int>(got.pixels[i * 3 + c]), static_cast<int>(gray[i]));
}
