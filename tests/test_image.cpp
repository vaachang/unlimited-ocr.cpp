#include "test_main.h"
#include "uocr/common.h"
#include "uocr/engine.h"
#include "uocr/image.h"
#include "uocr/prompt.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace uocr;

namespace {

// Tiny decoder config for engine-level tests; keeps the image-layout fields
// (patch_size/downsample_ratio/base_size/candidate_image_size) at their real
// defaults so the token-count math still reflects the production model.
ModelConfig tiny_engine_config() {
    ModelConfig c;
    c.vocab_size = 64;
    c.hidden_size = 16;
    c.intermediate_size = 32;
    c.moe_intermediate_size = 16;
    c.num_hidden_layers = 1;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 2;
    c.first_k_dense_replace = 1;
    c.n_routed_experts = 2;
    c.n_shared_experts = 1;
    c.num_experts_per_tok = 1;
    c.projector_input_dim = 64;
    c.projector_n_embed = 16;
    return c;
}

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

// --- layout / vision consistency regressions -------------------------------

// Regression: `UOCR_CHECK` must actually throw.  The macro previously expanded
// to a discarded `Error` temporary, silently swallowing every failed check.
UOCR_TEST(uocr_check_throws) {
    bool threw = false;
    try {
        UOCR_CHECK(1 == 2, "expected failure");
    } catch (const Error& e) {
        threw = std::string(e.what()).find("expected failure") != std::string::npos;
    }
    CHECK(threw);
    bool passed = true;
    try {
        UOCR_CHECK(1 == 1, "must not fire");
    } catch (...) {
        passed = false;
    }
    CHECK(passed);
}

// Reference `infer()` image-token counts (base 1024, local 640, patch 16, ds 4).
UOCR_TEST(image_token_count_reference_formula) {
    ModelConfig cfg;
    CHECK_EQ(image_token_count(cfg, true, 1, 1), 273);   // global only
    CHECK_EQ(image_token_count(cfg, false, 1, 1), 111);  // single 640 view
    CHECK_EQ(image_token_count(cfg, true, 2, 1), 273 + (10 * 2 + 1) * (10 * 1));
    CHECK_EQ(image_token_count(cfg, true, 7, 4), 273 + (10 * 7 + 1) * (10 * 4));
}

// Regression: the crop grid chosen for the prompt layout must be the same grid
// the vision path uses.  Images larger than `candidate_image_size` take the
// Gundam dynamic grid; fitting images and crop_mode=false use a single view.
UOCR_TEST(engine_image_crops_matches_dynamic_preprocess) {
    ModelConfig cfg = tiny_engine_config();
    EngineConfig ecfg;
    ecfg.max_batch_size = 2;
    ecfg.max_seq_len = 64;
    ecfg.memory_pool_bytes = 1u << 20;
    Engine engine(cfg, ecfg, DecoderWeights::random(cfg, 11), Backend::CPU);

    auto make = [](int w, int h) {
        ImageRGB im;
        im.width = w;
        im.height = h;
        im.pixels.assign(static_cast<std::size_t>(w) * h * 3, 128);
        return im;
    };

    {  // fits -> single global view
        auto c = engine.image_crops(make(600, 400), true);
        CHECK_EQ(c.size(), static_cast<std::size_t>(1));
        CHECK_EQ(c[0].width_crop_num, 1);
        CHECK_EQ(c[0].height_crop_num, 1);
        CHECK_EQ(image_token_count(cfg, true, 1, 1), 273);
    }
    {  // crop_mode off -> always a single view, regardless of size
        auto c = engine.image_crops(make(2000, 1000), false);
        CHECK_EQ(c.size(), static_cast<std::size_t>(1));
        CHECK_EQ(c[0].width_crop_num, 1);
        CHECK_EQ(c[0].height_crop_num, 1);
    }
    {  // larger than 640 -> exactly the dynamic_preprocess grid
        auto c = engine.image_crops(make(800, 400), true);
        DynamicPreprocess dp = dynamic_preprocess(make(800, 400), cfg.candidate_image_size);
        CHECK_EQ(c.size(), static_cast<std::size_t>(1));
        CHECK_EQ(c[0].width_crop_num, dp.width_crop_num);
        CHECK_EQ(c[0].height_crop_num, dp.height_crop_num);
        CHECK(image_token_count(cfg, true, c[0].width_crop_num, c[0].height_crop_num) > 273);
    }
}
