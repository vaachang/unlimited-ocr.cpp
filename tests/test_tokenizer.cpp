#include "test_main.h"
#include "uocr/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace uocr;

namespace {

std::string find_tokenizer() {
    if (const char* env = std::getenv("UOCR_TOKENIZER")) return env;
    const char* candidates[] = {"models/tokenizer.json", "../models/tokenizer.json",
                                "../../models/tokenizer.json", "../../../models/tokenizer.json"};
    for (const char* c : candidates) {
        if (FILE* f = std::fopen(c, "rb")) {
            std::fclose(f);
            return c;
        }
    }
    return "";
}

}  // namespace

UOCR_TEST(tokenizer_roundtrip_ascii) {
    const std::string path = find_tokenizer();
    if (path.empty()) {
        std::printf("         (skipped: models/tokenizer.json not found)\n");
        return;
    }
    Tokenizer tok = Tokenizer::from_file(path);
    CHECK(tok.vocab_size() > 100000);

    const std::string text = "Hello world, this is a test 12345!";
    std::vector<int> ids = tok.encode(text, false, false);
    CHECK(!ids.empty());
    for (int id : ids) CHECK(id >= 0 && id < tok.vocab_size());
    std::string back = tok.decode(ids, true);
    CHECK_EQ(back, text);
}

UOCR_TEST(tokenizer_deepseek_pretokenizer) {
    const std::string path = find_tokenizer();
    if (path.empty()) {
        std::printf("         (skipped: models/tokenizer.json not found)\n");
        return;
    }
    Tokenizer tok = Tokenizer::from_file(path);

    // Expected ids produced by HuggingFace `tokenizers 0.22.2` with
    // `encode(text, add_special_tokens=False)`; see
    // tools/reference/export_tokenizer_cases.py.
    struct Case { const char* text; std::vector<int> ids; };
    const std::vector<Case> cases = {
        {"Hello world, this is a test 12345!",
         {19923, 2058, 14, 566, 344, 260, 1950, 223, 6895, 1883, 3}},
        {"1234", {6895, 22}},                      // \p{N}{1,3} -> "123","4"
        {"3.14159", {21, 16, 9926, 3318}},         // "3",".","141","59"
        {"你好，世界！", {30594, 303, 3427, 1175}},
        {"第1页 2024年", {1056, 19, 5357, 223, 939, 22, 695}},
        {"你好world123", {30594, 29616, 6895}},
        {"  leading spaces", {223, 6646, 13564}},
        {"a\n\nb", {67, 271, 68}},
        {"<image>hello", {128815, 33310}},
    };
    for (const auto& c : cases) {
        std::vector<int> got = tok.encode(c.text, false, false);
        if (got != c.ids) {
            std::printf("         mismatch for %s\n", c.text);
            std::printf("         got:");
            for (int id : got) std::printf(" %d", id);
            std::printf("\n         want:");
            for (int id : c.ids) std::printf(" %d", id);
            std::printf("\n");
        }
        CHECK(got == c.ids);
    }
}

UOCR_TEST(tokenizer_special_tokens) {
    const std::string path = find_tokenizer();
    if (path.empty()) {
        std::printf("         (skipped: models/tokenizer.json not found)\n");
        return;
    }
    Tokenizer tok = Tokenizer::from_file(path);
    std::vector<int> ids = tok.encode("<image>hello", false, false);
    CHECK(!ids.empty());
    // the <image> token must map to 128815 for Unlimited-OCR
    bool found = false;
    for (int id : ids)
        if (id == 128815) found = true;
    CHECK(found);
}
