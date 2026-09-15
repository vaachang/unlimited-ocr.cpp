#include "uocr/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "uocr/log.h"

namespace uocr {

namespace {

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::vector<std::uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<std::uint32_t> out;
    std::size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::uint32_t cp = 0;
        int extra = 0;
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            extra = 1;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            extra = 2;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            extra = 3;
        } else {
            cp = c;
            extra = 0;
        }
        ++i;
        for (int k = 0; k < extra && i < s.size(); ++k, ++i)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
        out.push_back(cp);
    }
    return out;
}

// GPT-2 bytes_to_unicode table.
void build_byte_maps(std::string byte_to_char[256], std::unordered_map<std::uint32_t, int>& cp_to_byte) {
    std::vector<int> bs;
    for (int b = 0x21; b <= 0x7E; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (std::size_t i = 0; i < bs.size(); ++i) {
        std::string enc;
        append_utf8(enc, static_cast<std::uint32_t>(cs[i]));
        byte_to_char[bs[i]] = enc;
        cp_to_byte[static_cast<std::uint32_t>(cs[i])] = bs[i];
    }
}

const std::string kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

// DeepSeek special tokens (fullwidth vertical bar U+FF5C, lower one eighth
// block U+2581).  Kept as UTF-8 literals to avoid hex-escape parsing pitfalls.
const std::string kBosToken = u8"<｜begin▁of▁sentence｜>";
const std::string kEosToken = u8"<｜end▁of▁sentence｜>";

bool starts_with(const std::string& s, std::size_t pos, const std::string& p) {
    return pos + p.size() <= s.size() && std::equal(p.begin(), p.end(), s.begin() + static_cast<long>(pos));
}

}  // namespace

Tokenizer Tokenizer::from_file(const std::string& path) {
    std::ifstream f(path);
    UOCR_CHECK(f.good(), "cannot open tokenizer file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    auto root = nlohmann::json::parse(ss.str());

    Tokenizer t;

    const auto& vocab = root.at("model").at("vocab");
    int max_id = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const int id = it.value().get<int>();
        t.token_to_id_[it.key()] = id;
        max_id = std::max(max_id, id);
    }
    t.id_to_token_.assign(static_cast<std::size_t>(max_id) + 1, std::string());
    for (const auto& kv : t.token_to_id_) t.id_to_token_[static_cast<std::size_t>(kv.second)] = kv.first;

    for (const auto& m : root.at("model").at("merges")) {
        std::string a, b;
        if (m.is_array()) {
            a = m[0].get<std::string>();
            b = m[1].get<std::string>();
        } else {
            const std::string s = m.get<std::string>();
            const std::size_t sp = s.find(' ');
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        }
        t.merges_.emplace_back(a, b);
        t.ranks_[a + " " + b] = static_cast<int>(t.ranks_.size());
    }

    // special / added tokens
    t.special_flag_.assign(t.id_to_token_.size(), false);
    if (root.contains("added_tokens")) {
        for (const auto& a : root["added_tokens"]) {
            const int id = a.at("id").get<int>();
            const std::string content = a.at("content").get<std::string>();
            t.specials_.emplace_back(content, id);
            if (static_cast<std::size_t>(id) < t.special_flag_.size()) t.special_flag_[id] = true;
            if (id >= static_cast<int>(t.id_to_token_.size())) t.id_to_token_.resize(id + 1);
            t.id_to_token_[id] = content;
            t.token_to_id_[content] = id;
        }
        std::sort(t.specials_.begin(), t.specials_.end(),
                  [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });
    }

    // bos/eos from tokenizer_config (best effort)
    t.bos_id_ = t.token_to_id(kBosToken);
    if (t.bos_id_ < 0) t.bos_id_ = 0;
    t.eos_id_ = t.token_to_id(kEosToken);
    if (t.eos_id_ < 0) t.eos_id_ = 1;

    UOCR_INFO("tokenizer loaded: %d tokens, %zu merges, %zu specials", t.vocab_size(),
              t.merges_.size(), t.specials_.size());
    if (!t.token_to_id_.count(kBosToken)) UOCR_WARN("tokenizer: bos token not found by literal name");
    return t;
}

int Tokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

const std::string& Tokenizer::id_to_token(int id) const {
    static const std::string empty;
    if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return empty;
    return id_to_token_[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_special(int id) const {
    return id >= 0 && id < static_cast<int>(special_flag_.size()) && special_flag_[static_cast<std::size_t>(id)];
}

std::vector<int> Tokenizer::encode_pretoken(const std::string& mapped) const {
    std::vector<std::uint32_t> cps = utf8_codepoints(mapped);
    std::vector<std::string> syms;
    syms.reserve(cps.size());
    for (std::uint32_t cp : cps) {
        std::string s;
        append_utf8(s, cp);
        syms.push_back(std::move(s));
    }

    while (syms.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        int best_i = -1;
        for (std::size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = ranks_.find(syms[i] + " " + syms[i + 1]);
            if (it != ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = static_cast<int>(i);
            }
        }
        if (best_i < 0) break;
        const std::string merged = syms[best_i] + syms[best_i + 1];
        std::vector<std::string> next;
        next.reserve(syms.size());
        for (std::size_t i = 0; i < syms.size();) {
            if (i + 1 < syms.size() && syms[i] + " " + syms[i + 1] == syms[best_i] + " " + syms[best_i + 1]) {
                next.push_back(syms[i] + syms[i + 1]);
                i += 2;
            } else {
                next.push_back(syms[i]);
                ++i;
            }
        }
        syms.swap(next);
    }

    std::vector<int> ids;
    ids.reserve(syms.size());
    for (const auto& s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) ids.push_back(it->second);
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    static std::string byte_to_char[256];
    static std::unordered_map<std::uint32_t, int> cp_to_byte;
    static bool init = false;
    if (!init) {
        build_byte_maps(byte_to_char, cp_to_byte);
        init = true;
    }
    (void)cp_to_byte;

    std::vector<int> ids;
    if (add_bos) ids.push_back(bos_id_);

    // Split into special / normal segments.
    std::size_t i = 0;
    std::string normal;
    auto flush = [&]() {
        if (normal.empty()) return;
        // pretokenize on raw bytes
        std::size_t p = 0;
        const std::size_t n = normal.size();
        while (p < n) {
            const unsigned char c = static_cast<unsigned char>(normal[p]);
            std::size_t end = p;
            if (std::isalpha(c)) {
                while (end < n && std::isalpha(static_cast<unsigned char>(normal[end]))) ++end;
            } else if (std::isdigit(c)) {
                while (end < n && std::isdigit(static_cast<unsigned char>(normal[end]))) ++end;
            } else if (c == ' ') {
                if (p + 1 < n && std::isalpha(static_cast<unsigned char>(normal[p + 1]))) {
                    end = p + 2;
                    while (end < n && std::isalpha(static_cast<unsigned char>(normal[end]))) ++end;
                } else if (p + 1 < n && std::isdigit(static_cast<unsigned char>(normal[p + 1]))) {
                    end = p + 2;
                    while (end < n && std::isdigit(static_cast<unsigned char>(normal[end]))) ++end;
                } else {
                    end = p + 1;
                    if (p + 1 < n && static_cast<unsigned char>(normal[p + 1]) == ' ') {
                        while (end < n && static_cast<unsigned char>(normal[end]) == ' ') ++end;
                    }
                }
            } else if (c == '\'' && p + 1 < n) {
                end = p + 1;
                for (const auto& con : kContractions) {
                    if (starts_with(normal, p, con)) {
                        end = p + con.size();
                        break;
                    }
                }
            } else if (std::isspace(c)) {
                end = p + 1;
                while (end < n && std::isspace(static_cast<unsigned char>(normal[end]))) ++end;
            } else {
                // punctuation / non-ascii run, optionally followed by letters
                end = p + 1;
                while (end < n) {
                    const unsigned char d = static_cast<unsigned char>(normal[end]);
                    if (std::isalnum(d) || std::isspace(d)) break;
                    ++end;
                }
                while (end < n && std::isalpha(static_cast<unsigned char>(normal[end]))) ++end;
            }
            // map bytes
            std::string mapped;
            mapped.reserve((end - p) * 2);
            for (std::size_t k = p; k < end; ++k)
                mapped += byte_to_char[static_cast<unsigned char>(normal[k])];
            std::vector<int> sub = encode_pretoken(mapped);
            ids.insert(ids.end(), sub.begin(), sub.end());
            p = end;
        }
        normal.clear();
    };

    while (i < text.size()) {
        bool matched = false;
        for (const auto& sp : specials_) {
            if (starts_with(text, i, sp.first)) {
                flush();
                ids.push_back(sp.second);
                i += sp.first.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            normal.push_back(text[i]);
            ++i;
        }
    }
    flush();

    if (add_eos) ids.push_back(eos_id_);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::unordered_map<std::uint32_t, int> cp_to_byte;
    {
        std::string btc[256];
        build_byte_maps(btc, cp_to_byte);
    }
    std::string out;
    for (int id : ids) {
        if (is_special(id)) {
            if (!skip_special) out += id_to_token(id);
            continue;
        }
        const std::string& tok = id_to_token(id);
        for (std::uint32_t cp : utf8_codepoints(tok)) {
            auto it = cp_to_byte.find(cp);
            if (it != cp_to_byte.end())
                out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}

}  // namespace uocr
