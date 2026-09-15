#include "uocr/config.h"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace uocr {

using nlohmann::json;

namespace {

int jint(const json& j, const char* key, int def) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<int>();
    return def;
}

float jfloat(const json& j, const char* key, float def) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<float>();
    return def;
}

bool jbool(const json& j, const char* key, bool def) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<bool>();
    return def;
}

std::string jstr(const json& j, const char* key, const std::string& def) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<std::string>();
    return def;
}

}  // namespace

ModelConfig ModelConfig::from_json_string(const std::string& text) {
    json root = json::parse(text);
    ModelConfig c;

    c.model_type = jstr(root, "model_type", c.model_type);

    // The language model fields may live either at the top level (as in the
    // baidu/Unlimited-OCR config.json) or inside `language_config`.
    json lang = root;
    if (root.contains("language_config")) {
        // top level overrides nested
        lang = root["language_config"];
        for (auto it = root.begin(); it != root.end(); ++it) {
            if (it.key() == "language_config" || it.key() == "vision_config" ||
                it.key() == "projector_config")
                continue;
            lang[it.key()] = it.value();
        }
    }

    c.vocab_size = jint(lang, "vocab_size", c.vocab_size);
    c.hidden_size = jint(lang, "hidden_size", c.hidden_size);
    c.intermediate_size = jint(lang, "intermediate_size", c.intermediate_size);
    c.moe_intermediate_size = jint(lang, "moe_intermediate_size", c.moe_intermediate_size);
    c.num_hidden_layers = jint(lang, "num_hidden_layers", c.num_hidden_layers);
    c.num_attention_heads = jint(lang, "num_attention_heads", c.num_attention_heads);
    c.num_key_value_heads = jint(lang, "num_key_value_heads", c.num_attention_heads);
    c.first_k_dense_replace = jint(lang, "first_k_dense_replace", c.first_k_dense_replace);
    c.max_position_embeddings = jint(lang, "max_position_embeddings", c.max_position_embeddings);
    c.n_routed_experts = jint(lang, "n_routed_experts", c.n_routed_experts);
    c.n_shared_experts = jint(lang, "n_shared_experts", c.n_shared_experts);
    c.num_experts_per_tok = jint(lang, "num_experts_per_tok", c.num_experts_per_tok);
    c.n_group = jint(lang, "n_group", c.n_group);
    c.topk_group = jint(lang, "topk_group", c.topk_group);
    c.norm_topk_prob = jbool(lang, "norm_topk_prob", c.norm_topk_prob);
    c.routed_scaling_factor = jfloat(lang, "routed_scaling_factor", c.routed_scaling_factor);
    c.scoring_func = jstr(lang, "scoring_func", c.scoring_func);
    c.topk_method = jstr(lang, "topk_method", c.topk_method);
    c.hidden_act = jstr(lang, "hidden_act", c.hidden_act);
    c.rms_norm_eps = jfloat(lang, "rms_norm_eps", c.rms_norm_eps);
    c.rope_theta = jfloat(lang, "rope_theta", c.rope_theta);
    c.use_mla = jbool(lang, "use_mla", c.use_mla);
    c.tie_word_embeddings = jbool(lang, "tie_word_embeddings", c.tie_word_embeddings);

    c.bos_token_id = jint(lang, "bos_token_id", c.bos_token_id);
    c.eos_token_id = jint(lang, "eos_token_id", c.eos_token_id);
    c.image_token_id = jint(lang, "image_token_id", c.image_token_id);

    // R-SWA window
    c.sliding_window = jint(lang, "sliding_window_size", c.sliding_window);
    c.sliding_window = jint(lang, "sliding_window", c.sliding_window);

    // ---- vision ----
    if (root.contains("vision_config")) {
        const json& v = root["vision_config"];
        c.image_size = jint(v, "image_size", c.image_size);
        if (v.contains("width") && v["width"].is_object()) {
            if (v["width"].contains("sam_vit_b")) {
                const json& sam = v["width"]["sam_vit_b"];
                c.sam_embed_dim = jint(sam, "width", c.sam_embed_dim);
                c.sam_depth = jint(sam, "layers", c.sam_depth);
                c.sam_heads = jint(sam, "heads", c.sam_heads);
            }
            if (v["width"].contains("clip-l-14-224")) {
                const json& clip = v["width"]["clip-l-14-224"];
                c.clip_hidden_size = jint(clip, "width", c.clip_hidden_size);
                c.clip_layers = jint(clip, "layers", c.clip_layers);
                c.clip_heads = jint(clip, "heads", c.clip_heads);
                c.clip_image_size = jint(clip, "image_size", c.clip_image_size);
                c.clip_patch_size = jint(clip, "patch_size", c.clip_patch_size);
            }
        }
        c.clip_ffn_size = 4 * c.clip_hidden_size;
    }

    if (root.contains("projector_config")) {
        const json& p = root["projector_config"];
        c.projector_input_dim = jint(p, "input_dim", c.projector_input_dim);
        c.projector_n_embed = jint(p, "n_embed", c.projector_n_embed);
    }

    c.global_view_pos_head = jstr(root, "global_view_pos", "head") == "head";

    if (root.contains("candidate_resolutions") && root["candidate_resolutions"].is_array()) {
        c.candidate_resolutions.clear();
        for (const auto& r : root["candidate_resolutions"]) {
            if (r.is_array() && r.size() == 2)
                c.candidate_resolutions.emplace_back(r[0].get<int>(), r[1].get<int>());
        }
    }

    // default image_token_id for Unlimited-OCR
    if (c.image_token_id == 0) c.image_token_id = 128815;

    return c;
}

ModelConfig ModelConfig::from_json_file(const std::string& path) {
    std::ifstream f(path);
    UOCR_CHECK(f.good(), "cannot open config file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return from_json_string(ss.str());
}

std::string ModelConfig::to_string() const {
    std::ostringstream o;
    o << "ModelConfig{\n"
      << "  hidden_size=" << hidden_size << " layers=" << num_hidden_layers
      << " heads=" << num_attention_heads << "/" << num_key_value_heads
      << " head_dim=" << head_dim() << "\n"
      << "  vocab=" << vocab_size << " intermediate=" << intermediate_size
      << " moe_intermediate=" << moe_intermediate_size << "\n"
      << "  experts=" << n_routed_experts << " topk=" << num_experts_per_tok
      << " shared=" << n_shared_experts << " first_k_dense=" << first_k_dense_replace << "\n"
      << "  norm_topk_prob=" << (norm_topk_prob ? "true" : "false")
      << " routed_scaling=" << routed_scaling_factor << " scoring=" << scoring_func
      << " topk_method=" << topk_method << "\n"
      << "  rms_eps=" << rms_norm_eps << " rope_theta=" << rope_theta
      << " sliding_window=" << sliding_window << "\n"
      << "  image_size=" << image_size << " patch=" << patch_size
      << " base=" << base_size << " cand=" << candidate_image_size << "\n"
      << "  sam{emb=" << sam_embed_dim << ",depth=" << sam_depth << ",heads=" << sam_heads
      << "} clip{hidden=" << clip_hidden_size << ",layers=" << clip_layers
      << ",heads=" << clip_heads << "}\n"
      << "}";
    return o.str();
}

}  // namespace uocr
