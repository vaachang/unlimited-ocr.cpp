// inspect_model -- reads a Unlimited-OCR checkpoint and reports architecture
// facts, tensor inventory and memory estimates.  Optionally measures AWQ INT4
// quantization error on the expert weights.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "uocr/config.h"
#include "uocr/quant.h"
#include "uocr/safetensors.h"
#include "uocr/weights.h"

using namespace uocr;

namespace {
std::string norm_key(const std::string& k) {
    std::string out;
    bool in_num = false;
    for (char c : k) {
        if (c >= '0' && c <= '9') {
            if (!in_num) out.push_back('#');
            in_num = true;
        } else {
            in_num = false;
            out.push_back(c);
        }
    }
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    std::string dir = "models";
    bool quant = false;
    bool load = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!std::strcmp(argv[i], "--quant-check")) quant = true;
        else if (!std::strcmp(argv[i], "--load")) load = true;
    }

    ModelConfig cfg = ModelConfig::from_json_file(dir + "/config.json");
    std::printf("%s\n\n", cfg.to_string().c_str());

    SafetensorsFile st(dir + "/model-00001-of-000001.safetensors");
    std::map<std::string, int> counts;
    std::size_t total_bytes = 0;
    for (const auto& name : st.names()) {
        const auto& in = st.info(name);
        counts[norm_key(name)] += 1;
        total_bytes += in.nbytes;
    }
    std::printf("checkpoint: %zu tensors, %.2f GB\n", st.names().size(),
                total_bytes / (1024.0 * 1024.0 * 1024.0));
    const int expert_tensors = counts["model.layers.#.mlp.experts.#.gate_proj.weight"] +
                               counts["model.layers.#.mlp.experts.#.up_proj.weight"] +
                               counts["model.layers.#.mlp.experts.#.down_proj.weight"];
    std::printf("per-layer expert tensors: %d (expect 3*64*11=%d)\n", expert_tensors,
                3 * cfg.n_routed_experts * (cfg.num_hidden_layers - cfg.first_k_dense_replace));

    if (load) {
        std::printf("\nloading decoder weights (BF16 views)...\n");
        DecoderWeights w = DecoderWeights::load(dir + "/model-00001-of-000001.safetensors", cfg,
                                                false, 128);
        std::printf("loaded: %zu layers, embed=[%d,%d], router=[%d,%d], experts=%zu\n",
                    w.layers.size(), w.embed_tokens.rows, w.embed_tokens.cols,
                    w.layers[1].router.rows, w.layers[1].router.cols, w.layers[1].experts.size());
        // quick sanity: decode the first embedding row
        std::vector<float> row(cfg.hidden_size);
        w.embed_tokens.row(0, row.data());
        std::printf("embed row0[:4] = %.4f %.4f %.4f %.4f\n", row[0], row[1], row[2], row[3]);
    }

    if (quant) {
        // Sample a spread of expert matrices (gate/up/down across layers) and
        // compare the round-trip weight error of the two quantizers over a group
        // size sweep.  This is the "AWQ vs naive INT4" ablation (task 2.7).
        std::vector<std::string> names;
        const char* projs[3] = {"gate_proj", "up_proj", "down_proj"};
        for (int l = cfg.first_k_dense_replace; l < cfg.num_hidden_layers && names.size() < 12; ++l)
            for (int e = 0; e < cfg.n_routed_experts && names.size() < 12; ++e)
                for (int p = 0; p < 3 && names.size() < 12; ++p) {
                    const std::string name = "model.layers." + std::to_string(l) +
                                             ".mlp.experts." + std::to_string(e) + "." + projs[p] +
                                             ".weight";
                    if (st.contains(name)) names.push_back(name);
                }
        std::printf("\nINT4 weight quantization error (mean rel-L2 over %zu sampled expert matrices):\n",
                    names.size());
        const int groups[] = {32, 64, 128, 256};
        std::printf("%-14s", "scheme");
        for (int g : groups) std::printf("%12s", ("g=" + std::to_string(g)).c_str());
        std::printf("\n");
        auto sweep = [&](const char* label, bool symmetric) {
            std::printf("%-14s", label);
            for (int g : groups) {
                double rel_sum = 0;
                double bpw = 0;
                for (const std::string& nm : names) {
                    std::vector<float> wf = st.read_f32(nm);
                    const std::vector<i64>& shp = st.info(nm).shape;
                    const int rows = static_cast<int>(shp[0]);
                    const int cols = shp.size() > 1 ? static_cast<int>(shp[1]) : 1;
                    QuantizedMatrix q = symmetric
                                            ? quantize_int4_symmetric(wf.data(), rows, cols, g)
                                            : quantize_int4_awq(wf.data(), rows, cols, g);
                    std::vector<float> deq;
                    q.dequantize(deq);
                    double num = 0, den = 0;
                    for (std::size_t i = 0; i < wf.size(); ++i) {
                        const double d = static_cast<double>(wf[i]) - deq[i];
                        num += d * d;
                        den += static_cast<double>(wf[i]) * wf[i];
                    }
                    rel_sum += std::sqrt(num / (den + 1e-12));
                    const double ng = q.n_groups();
                    const double bytes = q.packed_bytes() + 4.0 * rows * ng + 4.0 * rows * ng;
                    bpw += bytes * 8.0 / (static_cast<double>(rows) * cols);
                }
                std::printf("%7.4f(%4.2f)", rel_sum / names.size(), bpw / names.size());
            }
            std::printf("   [rel-L2(bits/weight)]\n");
        };
        sweep("awq/asym", false);
        sweep("symmetric", true);
    }
    return 0;
}
