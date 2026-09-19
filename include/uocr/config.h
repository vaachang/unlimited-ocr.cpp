#pragma once

// Model / engine configuration.  Populated from a HuggingFace-style
// config.json (the file shipped with baidu/Unlimited-OCR).

#include <string>
#include <vector>

#include "uocr/common.h"

namespace uocr {

struct ModelConfig {
    std::string model_type = "unlimited-ocr";

    // ---- language decoder (DeepSeek-V2 style MoE) ----
    int vocab_size = 129280;
    int hidden_size = 1280;
    int intermediate_size = 6848;       // dense MLP (layer 0)
    int moe_intermediate_size = 896;    // per routed expert
    int num_hidden_layers = 12;
    int num_attention_heads = 10;
    int num_key_value_heads = 10;
    int first_k_dense_replace = 1;
    int max_position_embeddings = 32768;
    int n_routed_experts = 64;
    int n_shared_experts = 2;
    int num_experts_per_tok = 6;
    int n_group = 1;
    int topk_group = 1;
    bool norm_topk_prob = false;
    float routed_scaling_factor = 1.0f;
    std::string scoring_func = "softmax";
    std::string topk_method = "greedy";
    std::string hidden_act = "silu";
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool use_mla = false;
    bool tie_word_embeddings = false;

    // ---- R-SWA ----
    // Reference token region (vision + prompt) is kept in full; the generated
    // tail is stored in a ring buffer of `sliding_window` slots.
    int sliding_window = 128;

    // ---- special tokens ----
    int bos_token_id = 0;
    int eos_token_id = 1;
    int image_token_id = 128815;

    // ---- vision tower ----
    int image_size = 1024;
    int patch_size = 16;
    int downsample_ratio = 4;
    int base_size = 1024;           // global view resolution
    int candidate_image_size = 640; // gundam local view resolution
    int sam_embed_dim = 768;
    int sam_depth = 12;
    int sam_heads = 12;
    int sam_global_attn_interval = 3;  // global attention every N blocks (2,5,8,11)
    int sam_window_size = 14;
    int sam_out_chans = 256;
    int clip_hidden_size = 1024;
    int clip_layers = 24;
    int clip_heads = 16;
    int clip_ffn_size = 4096;
    int clip_image_size = 224;
    int clip_patch_size = 14;
    int projector_input_dim = 2048;
    int projector_n_embed = 1280;
    bool global_view_pos_head = true;  // concatenate global view before local views

    std::vector<std::pair<int, int>> candidate_resolutions{{1024, 1024}};

    // derived
    int head_dim() const { return hidden_size / num_attention_heads; }
    int kv_head_dim() const { return head_dim(); }

    static ModelConfig from_json_file(const std::string& path);
    static ModelConfig from_json_string(const std::string& text);
    std::string to_string() const;
};

struct EngineConfig {
    std::string model_dir = "models";
    std::string weights_file;  // optional explicit path to a converted .uocr file

    // batching
    int max_batch_size = 16;
    int min_batch_size = 4;
    int max_seq_len = 32768;
    int max_new_tokens = 4096;

    // memory
    std::size_t memory_pool_bytes = 0;  // 0 -> derive from device

    // execution
    bool use_cuda_graph = true;
    // CUDA-Graph capture scope: "full" captures the whole decode step (device
    // routing + fused expert kernel); "attn_dense" captures only the attention
    // subgraphs and issues the MoE outside the graph (ablation).
    std::string graph_scope = "full";
    bool use_int4_experts = true;
    int int4_group_size = 128;

    // sampling defaults
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int no_repeat_ngram_size = 35;
    int ngram_window = 1024;
};

}  // namespace uocr
