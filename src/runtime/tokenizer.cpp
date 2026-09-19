#include "uocr/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "uocr/log.h"
#include "uocr/unicode_tables.h"

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

// DeepSeek special tokens (fullwidth vertical bar U+FF5C, lower one eighth
// block U+2581).  Kept as UTF-8 literals to avoid hex-escape parsing pitfalls.
const std::string kBosToken = u8"<｜begin▁of▁sentence｜>";
const std::string kEosToken = u8"<｜end▁of▁sentence｜>";

bool starts_with(const std::string& s, std::size_t pos, const std::string& p) {
    return pos + p.size() <= s.size() && std::equal(p.begin(), p.end(), s.begin() + static_cast<long>(pos));
}

// ---------------------------------------------------------------------------
// Unicode general-category classification (generated tables).
// ---------------------------------------------------------------------------

enum : std::uint8_t { kCatL = 1, kCatM = 2, kCatN = 4, kCatP = 8, kCatS = 16 };

std::uint8_t unicode_cat_mask(std::uint32_t cp) {
    const UocrUnicodeRange* tab = kUnicodeCategoryRanges;
    std::size_t lo = 0, hi = sizeof(kUnicodeCategoryRanges) / sizeof(UocrUnicodeRange);
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (cp < tab[mid].lo)
            hi = mid;
        else if (cp > tab[mid].hi)
            lo = mid + 1;
        else
            return tab[mid].mask;
    }
    return 0;
}

bool unicode_is_whitespace(std::uint32_t cp) {
    const UocrUnicodeRange* tab = kWhitespaceRanges;
    std::size_t lo = 0, hi = sizeof(kWhitespaceRanges) / sizeof(UocrUnicodeRange);
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (cp < tab[mid].lo)
            hi = mid;
        else if (cp > tab[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

bool is_cjk(std::uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) ||
           (cp >= 0x30A0 && cp <= 0x30FF);
}

bool is_ascii_alpha(std::uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

// First alternative of the third Split regex:
//   [!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
bool is_pattern_punct(std::uint32_t cp) {
    if (cp > 0x7F) return false;
    static const char kPunct[] = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return std::strchr(kPunct, static_cast<int>(cp)) != nullptr;
}

// Decode UTF-8 into code points + byte offsets (offs has n+1 entries).
void decode_utf8_offsets(const std::string& s, std::vector<std::uint32_t>& cps,
                         std::vector<std::size_t>& offs) {
    cps.clear();
    offs.clear();
    std::size_t i = 0;
    const std::size_t n = s.size();
    while (i < n) {
        offs.push_back(i);
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
        }
        ++i;
        for (int k = 0; k < extra && i < n; ++k, ++i)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
        cps.push_back(cp);
    }
    offs.push_back(n);
}

using CpVec = std::vector<std::uint32_t>;

// Match `\p{N}{1,3}` at code-point index i, returns length (0 = no match).
int match_numbers(const CpVec& cp, int n, int i) {
    if (i >= n || !(unicode_cat_mask(cp[i]) & kCatN)) return 0;
    int j = i + 1;
    while (j < n && (j - i) < 3 && (unicode_cat_mask(cp[j]) & kCatN)) ++j;
    return j - i;
}

// Match `[一-龥぀-ゟ゠-ヿ]+`.
int match_cjk(const CpVec& cp, int n, int i) {
    if (i >= n || !is_cjk(cp[i])) return 0;
    int j = i + 1;
    while (j < n && is_cjk(cp[j])) ++j;
    return j - i;
}

// Match the third (and most complex) Split regex.  See the HuggingFace
// `tokenizer.json` pre_tokenizer.  Alternatives are tried in order and use
// greedy quantifiers, matching the Rust `regex` crate semantics.
int match_pattern3(const CpVec& cp, int n, int i) {
    // A: [punct][A-Za-z]+
    if (i < n && is_pattern_punct(cp[i])) {
        int j = i + 1;
        while (j < n && is_ascii_alpha(cp[j])) ++j;
        if (j > i + 1) return j - i;
    }
    // B: [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    {
        if (i < n && cp[i] != '\r' && cp[i] != '\n' &&
            !(unicode_cat_mask(cp[i]) & (kCatL | kCatP | kCatS))) {
            int j = i + 1;
            while (j < n && (unicode_cat_mask(cp[j]) & (kCatL | kCatM))) ++j;
            if (j > i + 1) return j - i;
        }
        if (i < n && (unicode_cat_mask(cp[i]) & (kCatL | kCatM))) {
            int j = i + 1;
            while (j < n && (unicode_cat_mask(cp[j]) & (kCatL | kCatM))) ++j;
            return j - i;
        }
    }
    // C: ' '?[\p{P}\p{S}]+[\r\n]*
    if (i < n && (unicode_cat_mask(cp[i]) & (kCatP | kCatS))) {
        int j = i + 1;
        while (j < n && (unicode_cat_mask(cp[j]) & (kCatP | kCatS))) ++j;
        while (j < n && (cp[j] == '\r' || cp[j] == '\n')) ++j;
        return j - i;
    }
    if (i + 1 < n && cp[i] == ' ' && (unicode_cat_mask(cp[i + 1]) & (kCatP | kCatS))) {
        int j = i + 2;
        while (j < n && (unicode_cat_mask(cp[j]) & (kCatP | kCatS))) ++j;
        while (j < n && (cp[j] == '\r' || cp[j] == '\n')) ++j;
        return j - i;
    }
    // D: \s*[\r\n]+
    {
        int j = i;
        while (j < n && unicode_is_whitespace(cp[j])) ++j;
        if (j > i) {
            int last_nl = -1;
            for (int k = i; k < j; ++k)
                if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
            if (last_nl >= 0) return last_nl - i + 1;
        }
    }
    // E: \s+(?!\S)
    {
        int j = i;
        while (j < n && unicode_is_whitespace(cp[j])) ++j;
        if (j > i) {
            if (j == n) return j - i;      // trailing whitespace: consume all
            if (j - i >= 2) return j - i - 1;  // leave the last ws for the next piece
        }
    }
    // F: \s+
    {
        int j = i;
        while (j < n && unicode_is_whitespace(cp[j])) ++j;
        if (j > i) return j - i;
    }
    return 0;
}

// Apply one `Split(..., behavior="isolated")` pass to `s`.
template <typename MatchFn>
std::vector<std::string> split_isolated(const std::string& s, MatchFn match) {
    std::vector<std::string> out;
    std::vector<std::uint32_t> cps;
    std::vector<std::size_t> offs;
    decode_utf8_offsets(s, cps, offs);
    const int n = static_cast<int>(cps.size());
    std::size_t seg_start = 0;
    int i = 0;
    while (i < n) {
        const int len = match(cps, n, i);
        if (len > 0) {
            const std::size_t ms = offs[i];
            const std::size_t me = offs[i + len];
            if (ms > seg_start) out.push_back(s.substr(seg_start, ms - seg_start));
            out.push_back(s.substr(ms, me - ms));
            i += len;
            seg_start = me;
        } else {
            ++i;
        }
    }
    if (seg_start < s.size()) out.push_back(s.substr(seg_start));
    return out;
}

struct ByteMaps {
    std::string byte_to_char[256];
    std::unordered_map<std::uint32_t, int> cp_to_byte;
    ByteMaps() { build_byte_maps(byte_to_char, cp_to_byte); }
};

const ByteMaps& byte_maps() {
    static const ByteMaps maps;
    return maps;
}

std::string map_bytes(const std::string& s) {
    const ByteMaps& m = byte_maps();
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) out += m.byte_to_char[c];
    return out;
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

    // Index specials by their first byte to keep encode() linear-ish.
    t.special_by_first_.assign(256, {});
    for (std::size_t k = 0; k < t.specials_.size(); ++k) {
        if (t.specials_[k].first.empty()) continue;
        t.special_by_first_[static_cast<unsigned char>(t.specials_[k].first[0])].push_back(
            static_cast<int>(k));
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

// Apply the three `Split` pre-tokenizers in sequence and ByteLevel-map the
// resulting pieces.  `text` must not contain special/added tokens (the caller
// extracts those first).
std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> stage1 = split_isolated(text, match_numbers);
    std::vector<std::string> out;
    for (const std::string& a : stage1) {
        std::vector<std::string> stage2 = split_isolated(a, match_cjk);
        for (const std::string& b : stage2) {
            std::vector<std::string> stage3 = split_isolated(b, match_pattern3);
            for (const std::string& c : stage3) out.push_back(map_bytes(c));
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int> ids;
    if (add_bos) ids.push_back(bos_id_);

    // Split into special / normal segments, then pre-tokenize the normal runs.
    std::size_t i = 0;
    std::string normal;
    auto flush = [&]() {
        if (normal.empty()) return;
        for (const std::string& piece : pretokenize(normal)) {
            std::vector<int> sub = encode_pretoken(piece);
            ids.insert(ids.end(), sub.begin(), sub.end());
        }
        normal.clear();
    };

    while (i < text.size()) {
        bool matched = false;
        const unsigned char first = static_cast<unsigned char>(text[i]);
        if (first < special_by_first_.size()) {
            for (int idx : special_by_first_[first]) {
                const auto& sp = specials_[static_cast<std::size_t>(idx)];
                if (starts_with(text, i, sp.first)) {
                    flush();
                    ids.push_back(sp.second);
                    i += sp.first.size();
                    matched = true;
                    break;
                }
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
