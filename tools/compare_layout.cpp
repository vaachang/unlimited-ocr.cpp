// compare_layout -- checks build_ocr_prompt() against the reference token
// layout exported by tools/reference/export_layout_cases.py (E1).

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "uocr/config.h"
#include "uocr/prompt.h"
#include "uocr/tokenizer.h"

using namespace uocr;
using nlohmann::json;

namespace {

std::vector<std::string> split_on(const std::string& s, const std::string& sep) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (true) {
        const std::size_t p = s.find(sep, pos);
        if (p == std::string::npos) {
            out.push_back(s.substr(pos));
            break;
        }
        out.push_back(s.substr(pos, p - pos));
        pos = p + sep.size();
    }
    return out;
}

std::string first_diff(const std::vector<int>& a, const std::vector<int>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
        if (a[i] != b[i])
            return "index " + std::to_string(i) + ": got " + std::to_string(a[i]) + " want " +
                   std::to_string(b[i]);
    if (a.size() != b.size())
        return "length got " + std::to_string(a.size()) + " want " + std::to_string(b.size());
    return "identical";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models";
    std::string ref_file = "/tmp/opencode/ref_layout/layout_cases.json";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_file = argv[++i];
    }

    std::ifstream f(ref_file);
    UOCR_CHECK(f.good(), "cannot open " + ref_file);
    json manifest = json::parse(f);

    Tokenizer tok = Tokenizer::from_file(model_dir + "/tokenizer.json");
    ModelConfig cfg = ModelConfig::from_json_file(model_dir + "/config.json");

    int total = 0, id_ok = 0, mask_ok = 0;
    for (const auto& c : manifest.at("cases")) {
        ++total;
        const std::string prompt = c.at("prompt").get<std::string>();
        const bool crop_mode = c.at("crop_mode").get<bool>();

        std::vector<ImageSpatialCrop> crops;
        for (const auto& cr : c.at("crop_ratios")) {
            ImageSpatialCrop sc;
            sc.width_crop_num = cr[0].get<int>();
            sc.height_crop_num = cr[1].get<int>();
            crops.push_back(sc);
        }

        std::vector<std::string> splits = split_on(prompt, "<image>");
        PromptLayout got = build_ocr_prompt(tok, splits, crops, cfg, crop_mode);

        std::vector<int> want_ids;
        for (const auto& x : c.at("input_ids")) want_ids.push_back(x.get<int>());
        std::vector<std::uint8_t> want_mask;
        for (const auto& x : c.at("images_seq_mask")) want_mask.push_back(static_cast<std::uint8_t>(x.get<int>()));

        const bool ids_match = got.input_ids == want_ids;
        const bool mask_match = got.images_seq_mask == want_mask;
        id_ok += ids_match;
        mask_ok += mask_match;

        if (!ids_match || !mask_match) {
            std::printf("case %d MISMATCH (ids %s, mask %s) prompt=%s crop_mode=%d\n", total,
                        ids_match ? "ok" : "BAD", mask_match ? "ok" : "BAD", prompt.c_str(),
                        static_cast<int>(crop_mode));
            if (!ids_match)
                std::printf("  ids : %s\n", first_diff(got.input_ids, want_ids).c_str());
            if (!mask_match) {
                std::string m;
                const std::size_t n = std::min(got.images_seq_mask.size(), want_mask.size());
                for (std::size_t i = 0; i < n; ++i)
                    if (got.images_seq_mask[i] != want_mask[i]) {
                        m = "index " + std::to_string(i) + ": got " +
                            std::to_string(got.images_seq_mask[i]) + " want " +
                            std::to_string(want_mask[i]);
                        break;
                    }
                if (m.empty()) m = "length mismatch";
                std::printf("  mask: %s\n", m.c_str());
            }
        }
    }

    std::printf("\nlayout alignment: ids %d/%d, mask %d/%d\n", id_ok, total, mask_ok, total);
    return (id_ok == total && mask_ok == total) ? 0 : 1;
}
