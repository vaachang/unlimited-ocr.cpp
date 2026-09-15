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
