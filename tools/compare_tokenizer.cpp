// compare_tokenizer -- checks uocr::Tokenizer against the HuggingFace reference
// cases exported by tools/reference/export_tokenizer_cases.py (E0 alignment).
//
// Compares both the DeepSeek BPE pre-tokenized pieces and the final token ids.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "uocr/tokenizer.h"

using namespace uocr;
using nlohmann::json;

namespace {

std::string join_ids(const std::vector<int>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ' ';
        s += std::to_string(v[i]);
    }
    return s;
}

std::string join_strs(const std::vector<std::string>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += v[i];
    }
    s += "]";
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models";
    std::string ref_file = "/tmp/opencode/ref_tokenizer/tokenizer_cases.json";
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_file = argv[++i];
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
    }

    std::ifstream f(ref_file);
    UOCR_CHECK(f.good(), "cannot open " + ref_file);
    json manifest = json::parse(f);

    Tokenizer tok = Tokenizer::from_file(model_dir + "/tokenizer.json");

    int id_ok = 0, pre_ok = 0, total = 0;
    int shown = 0;
    for (const auto& c : manifest.at("cases")) {
        const std::string text = c.at("text").get<std::string>();
        ++total;

        std::vector<int> want_ids;
        for (const auto& x : c.at("ids")) want_ids.push_back(x.get<int>());
        std::vector<std::string> want_pre;
        for (const auto& x : c.at("pretokens")) want_pre.push_back(x.get<std::string>());

        std::vector<int> got_ids = tok.encode(text, false, false);
        std::vector<std::string> got_pre = tok.pretokenize(text);

        const bool ids_match = got_ids == want_ids;
        const bool pre_match = got_pre == want_pre;
        id_ok += ids_match ? 1 : 0;
        pre_ok += pre_match ? 1 : 0;

        if (!ids_match || !pre_match || verbose) {
            std::printf("case %d: %s (ids %s, pretok %s)\n", total,
                        ids_match && pre_match ? "OK" : "MISMATCH",
                        ids_match ? "ok" : "BAD", pre_match ? "ok" : "BAD");
            if (!pre_match && shown < 20) {
                std::printf("  text      : %s\n", text.c_str());
                std::printf("  pretok want: %s\n", join_strs(want_pre).c_str());
                std::printf("  pretok got : %s\n", join_strs(got_pre).c_str());
            }
            if (!ids_match && shown < 20) {
                std::printf("  text   : %s\n", text.c_str());
                std::printf("  ids want: %s\n", join_ids(want_ids).c_str());
                std::printf("  ids got : %s\n", join_ids(got_ids).c_str());
            }
            ++shown;
        }
    }

    std::printf("\ntokenizer alignment: pretok %d/%d, ids %d/%d\n", pre_ok, total, id_ok, total);
    return (id_ok == total && pre_ok == total) ? 0 : 1;
}
