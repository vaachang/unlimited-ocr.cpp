// compare_reference -- numerically compares the C++ engine against reference
// activations exported from PyTorch by tools/reference/export_reference.py.
//
// Usage:
//   compare_reference --model models --ref /tmp/opencode/ref_decoder [--traces]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "uocr/config.h"
#include "uocr/moe_decoder.h"

using namespace uocr;
using nlohmann::json;

namespace {

std::vector<float> load_bin(const std::string& dir, const json& entry) {
    std::vector<i64> shape;
    for (const auto& d : entry.at("shape")) shape.push_back(d.get<i64>());
    i64 n = 1;
    for (i64 d : shape) n *= d;
    std::vector<float> v(static_cast<std::size_t>(n));
    const std::string path = dir + "/" + entry.at("file").get<std::string>();
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) UOCR_THROW("cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return v;
}

Tensor to_tensor(const std::vector<float>& v, const json& entry) {
    std::vector<i64> shape;
    for (const auto& d : entry.at("shape")) shape.push_back(d.get<i64>());
    return Tensor::from_data(shape, v);
}

struct Diff {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    double ref_max = 0.0;
    double ref_rms = 0.0;
};

Diff diff(const float* a, const float* b, i64 n) {
    Diff d;
    double num = 0, den = 0;
    for (i64 i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - b[i]);
        d.max_abs = std::max(d.max_abs, e);
        d.ref_max = std::max(d.ref_max, std::fabs(static_cast<double>(b[i])));
        num += e * e;
        den += static_cast<double>(b[i]) * b[i];
    }
    d.rel_l2 = std::sqrt(num / (den + 1e-30));
    d.ref_rms = std::sqrt(den / static_cast<double>(n));
    return d;
}

struct RouterDiff {
    int total = 0;
    int set_mismatch = 0;
    double weight_max_abs = 0.0;
    std::vector<int> mismatch_per_layer;
};

RouterDiff compare_routers(const std::vector<std::vector<MoEDecoder::RouterTrace>>& trace,
                           const std::string& dir, const json& tensors, const std::string& prefix,
                           const ModelConfig& cfg) {
    RouterDiff rd;
    const int first = cfg.first_k_dense_replace;
    for (std::size_t mi = 0; mi < trace.size(); ++mi) {
        const int li = first + static_cast<int>(mi);
        std::vector<float> ids = load_bin(dir, tensors.at(prefix + "_router_ids_" + std::to_string(li)));
        std::vector<float> ws = load_bin(dir, tensors.at(prefix + "_router_w_" + std::to_string(li)));
        const int k = cfg.num_experts_per_tok;
        int layer_mismatch = 0;
        for (std::size_t t = 0; t < trace[mi].size(); ++t) {
            std::vector<int> ref_ids;
            for (int j = 0; j < k; ++j)
                ref_ids.push_back(static_cast<int>(ids[t * k + j]));
            std::vector<int> got = trace[mi][t].experts;
            std::sort(ref_ids.begin(), ref_ids.end());
            std::sort(got.begin(), got.end());
            ++rd.total;
            if (ref_ids != got) {
                ++rd.set_mismatch;
                ++layer_mismatch;
            }
            for (int j = 0; j < k; ++j)
                rd.weight_max_abs = std::max(
                    rd.weight_max_abs,
                    std::fabs(static_cast<double>(ws[t * k + j]) - trace[mi][t].weights[j]));
        }
        rd.mismatch_per_layer.push_back(layer_mismatch);
    }
    return rd;
}

void compare_attn(const std::vector<MoEDecoder::AttnTrace>& tr, const std::string& dir,
                  const json& tensors, const std::string& prefix, int n_layers,
                  std::vector<double>& per_layer_q) {
    static const char* names[4] = {"q", "k", "v", "o"};
    double worst[4] = {0, 0, 0, 0};
    for (int li = 0; li < n_layers && li < static_cast<int>(tr.size()); ++li) {
        const MoEDecoder::AttnTrace& t = tr[li];
        const std::vector<float>* got[4] = {&t.q, &t.k, &t.v, &t.o};
        double layer_q = 0;
        for (int ki = 0; ki < 4; ++ki) {
            const std::string key = prefix + "_attn_" + names[ki] + "_" + std::to_string(li);
            if (!tensors.contains(key)) continue;
            std::vector<float> ref = load_bin(dir, tensors.at(key));
            if (ref.size() != got[ki]->size()) continue;
            Diff d = diff(got[ki]->data(), ref.data(), static_cast<i64>(ref.size()));
            worst[ki] = std::max(worst[ki], d.rel_l2);
            if (ki == 0) layer_q = d.rel_l2;
        }
        per_layer_q.push_back(layer_q);
    }
    std::printf("== %s attention projections: worst rel_l2 q=%.5f k=%.5f v=%.5f o=%.5f\n",
                prefix.c_str(), worst[0], worst[1], worst[2], worst[3]);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models";
    std::string ref_dir = "/tmp/opencode/ref_decoder";
    bool show = false;
    bool bf16 = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--traces")) show = true;
        else if (!std::strcmp(argv[i], "--bf16")) bf16 = true;
    }

    std::ifstream mf(ref_dir + "/manifest.json");
    UOCR_CHECK(mf.good(), "cannot open " + ref_dir + "/manifest.json");
    json manifest = json::parse(mf);
    UOCR_CHECK(manifest.at("mode") == "decoder", "manifest is not a decoder export");
    const json& tensors = manifest.at("tensors");

    ModelConfig cfg = ModelConfig::from_json_file(model_dir + "/config.json");
    // Use the exported window (manifest) so cache geometry matches.
    cfg.sliding_window = manifest.at("window").get<int>();

    std::printf("loading decoder weights (BF16 views) ...\n");
    DecoderWeights weights = DecoderWeights::load(
        model_dir + "/model-00001-of-000001.safetensors", cfg, false, 128);
    MoEDecoder dec(cfg, weights);
    dec.set_bf16_rounding(bf16);
    std::printf("bf16 activation rounding: %s\n", bf16 ? "on" : "off");

    // ---- prefill ----
    const int seq = manifest.at("seq").get<int>();
    std::vector<float> emb = load_bin(ref_dir, tensors.at("hidden_0"));
    Tensor inputs = to_tensor(emb, tensors.at("hidden_0"));
    std::vector<int> positions(seq);
    for (int i = 0; i < seq; ++i) positions[i] = i;

    RSWACache cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim(),
                    cfg.sliding_window);
    cache.reset(seq);

    dec.set_trace_router(true);
    dec.set_trace_attn(true);
    std::vector<float> logits;
    std::vector<Tensor> trace;
    dec.forward(inputs, positions, cache, /*prefill=*/true, 0, /*final_norm=*/true, logits, &trace);

    if (tensors.contains("prefill_attn_q_0")) {
        std::vector<double> per_layer_q;
        compare_attn(dec.attn_trace(), ref_dir, tensors, "prefill", cfg.num_hidden_layers,
                     per_layer_q);
        std::printf("   per-layer q rel_l2 (layer0..):");
        for (double v : per_layer_q) std::printf(" %.4f", v);
        std::printf("\n");
    }
    dec.clear_attn_trace();

    std::printf("\n== prefill: per-layer hidden states (torch hidden_%d) ==\n", 0);
    double worst = 0;
    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const std::string key = "hidden_" + std::to_string(li + 1);
        std::vector<float> ref = load_bin(ref_dir, tensors.at(key));
        Diff d = diff(trace[li].data(), ref.data(), static_cast<i64>(ref.size()));
        worst = std::max(worst, d.rel_l2);
        if (show || li < 3 || li == cfg.num_hidden_layers - 1)
            std::printf("  layer %2d: max_abs=%.5f rel_l2=%.5f ref_rms=%.3f ref_max=%.2f\n", li,
                        d.max_abs, d.rel_l2, d.ref_rms, d.ref_max);
    }
    std::printf("  worst prefill rel_l2 = %.5f\n", worst);

    {
        std::vector<float> ref = load_bin(ref_dir, tensors.at("prefill_logits"));

        Diff d = diff(logits.data(), ref.data(), static_cast<i64>(ref.size()));
        std::printf("== prefill logits: max_abs=%.4f rel_l2=%.5f\n", d.max_abs, d.rel_l2);
    }

    if (tensors.contains("prefill_router_ids_1")) {
        RouterDiff rd = compare_routers(dec.router_trace(), ref_dir, tensors, "prefill", cfg);
        std::printf("== prefill router: %d tokens, set_mismatch=%d, weight_max_abs=%.4f\n",
                    rd.total, rd.set_mismatch, rd.weight_max_abs);
        std::printf("   per-layer mismatches (layer1..layer11):");
        for (int m : rd.mismatch_per_layer) std::printf(" %d", m);
        std::printf("\n");
    }
    dec.clear_router_trace();

    // ---- cache comparison (prefill region) ----
    {
        const int kv_heads = cfg.num_key_value_heads;
        const int hd = cfg.head_dim();
        int checked = 0;
        double worst_cache = 0;
        for (int li = 0; li < cfg.num_hidden_layers; ++li) {
            std::vector<float> rk = load_bin(ref_dir, tensors.at("prefill_k_" + std::to_string(li)));
            // ref layout [kv_heads, len, hd]; ours [len, kv_heads, hd]
            const int len = static_cast<int>(rk.size()) / (kv_heads * hd);
            std::vector<float> ours(static_cast<std::size_t>(len) * kv_heads * hd);
            const float* kcache = cache.keys(li);
            for (int t = 0; t < len; ++t)
                for (int h = 0; h < kv_heads; ++h)
                    for (int d = 0; d < hd; ++d)
                        ours[(static_cast<std::size_t>(h) * len + t) * hd + d] =
                            kcache[(static_cast<std::size_t>(t) * kv_heads + h) * hd + d];
            Diff d = diff(ours.data(), rk.data(), static_cast<i64>(rk.size()));
            worst_cache = std::max(worst_cache, d.rel_l2);
            std::printf("  K layer %2d: max_abs=%.4f rel_l2=%.5f\n", li, d.max_abs, d.rel_l2);
            ++checked;
        }
        std::printf("== prefill K cache: %d layers, worst rel_l2=%.5f\n", checked, worst_cache);
    }

    // ---- decode steps ----
    std::printf("\n== decode steps ==\n");
    const json& steps = manifest.at("decode_steps");
    int pos = seq;
    double worst_dec_hidden = 0, worst_dec_logits = 0;
    std::size_t worst_dec_step = 0;
    for (std::size_t s = 0; s < steps.size(); ++s) {
        const int tok = steps[s].at("token").get<int>();
        Tensor in = dec.embed({tok});
        std::vector<int> p{pos};
        std::vector<float> lg;
        std::vector<Tensor> tr;
        dec.forward(in, p, cache, /*prefill=*/false, pos, true, lg, &tr);
        ++pos;

        std::vector<float> refh = load_bin(ref_dir, tensors.at("decode_hidden_" + std::to_string(s)));

        Diff dh = diff(tr.back().data(), refh.data(), static_cast<i64>(refh.size()));
        std::vector<float> refl = load_bin(ref_dir, tensors.at("decode_logits_" + std::to_string(s)));
        Diff dl = diff(lg.data(), refl.data(), static_cast<i64>(refl.size()));
        if (dh.rel_l2 > worst_dec_hidden) { worst_dec_hidden = dh.rel_l2; worst_dec_step = s; }
        worst_dec_logits = std::max(worst_dec_logits, dl.rel_l2);
        std::printf("  step %zu tok=%6d pos=%3d  hidden rel_l2=%.5f max_abs=%.4f | "
                    "logits rel_l2=%.5f\n",
                    s, tok, steps[s].at("position").get<int>(), dh.rel_l2, dh.max_abs, dl.rel_l2);
        if (s < 2 && tensors.contains("decode" + std::to_string(s) + "_router_ids_1")) {
            RouterDiff rd = compare_routers(dec.router_trace(), ref_dir, tensors,
                                            "decode" + std::to_string(s), cfg);
            std::printf("           router: tokens=%d set_mismatch=%d weight_max_abs=%.4f\n",
                        rd.total, rd.set_mismatch, rd.weight_max_abs);
        }
        if (s < 2 && tensors.contains("decode" + std::to_string(s) + "_attn_q_0")) {
            std::vector<double> pq;
            compare_attn(dec.attn_trace(), ref_dir, tensors, "decode" + std::to_string(s),
                         cfg.num_hidden_layers, pq);
        }
        dec.clear_router_trace();
        dec.clear_attn_trace();
    }
    std::printf("== decode worst: hidden rel_l2=%.5f (step %zu), logits rel_l2=%.5f\n",
                worst_dec_hidden, worst_dec_step, worst_dec_logits);

    // ---- final cache comparison (ring wraparound) ----
    if (tensors.contains("final_k_0")) {
        const int kv_heads = cfg.num_key_value_heads;
        const int hd = cfg.head_dim();
        double worst_k = 0, worst_v = 0;
        for (int li = 0; li < cfg.num_hidden_layers; ++li) {
            for (int which = 0; which < 2; ++which) {
                std::vector<float> ref = load_bin(
                    ref_dir, tensors.at((which ? "final_v_" : "final_k_") + std::to_string(li)));
                const int len = static_cast<int>(ref.size()) / (kv_heads * hd);
                std::vector<float> ours(static_cast<std::size_t>(len) * kv_heads * hd);
                const float* cc = which ? cache.values(li) : cache.keys(li);
                for (int t = 0; t < len; ++t)
                    for (int h = 0; h < kv_heads; ++h)
                        for (int d = 0; d < hd; ++d)
                            ours[(static_cast<std::size_t>(h) * len + t) * hd + d] =
                                cc[(static_cast<std::size_t>(t) * kv_heads + h) * hd + d];
                Diff d = diff(ours.data(), ref.data(), static_cast<i64>(ref.size()));
                if (which)
                    worst_v = std::max(worst_v, d.rel_l2);
                else
                    worst_k = std::max(worst_k, d.rel_l2);
            }
        }
        std::printf("== final cache: worst K rel_l2=%.5f, worst V rel_l2=%.5f (len=%d)\n",
                    worst_k, worst_v, cache.len(0));
    }

    return 0;
}
