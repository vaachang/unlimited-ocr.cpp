#pragma once

// Pure C++ byte-level BPE tokenizer (DeepSeek/Unlimited-OCR vocabulary).
//
// Reads HuggingFace `tokenizer.json` (vocab + merges + special tokens) at
// construction time, caching the merge table so encoding avoids regex and
// allocations where possible.

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "uocr/common.h"

namespace uocr {

class Tokenizer {
public:
    Tokenizer() = default;

    static Tokenizer from_file(const std::string& tokenizer_json_path);

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

    std::vector<int> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;
    std::string decode(const std::vector<int>& ids, bool skip_special = false) const;

    int token_to_id(const std::string& token) const;
    const std::string& id_to_token(int id) const;
    bool is_special(int id) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> ranks_;  // "a b" -> merge rank
    std::vector<std::pair<std::string, std::string>> merges_;
    std::vector<std::pair<std::string, int>> specials_;  // content -> id
    std::vector<bool> special_flag_;
    int bos_id_ = 0;
    int eos_id_ = 1;

    std::vector<int> encode_pretoken(const std::string& mapped) const;
};

}  // namespace uocr
