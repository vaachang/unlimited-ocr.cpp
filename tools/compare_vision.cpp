// compare_vision -- compares DeepEncoder visual embeddings against the PyTorch
// reference exported by tools/reference/export_reference.py --mode vision.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "uocr/config.h"
#include "uocr/deep_encoder.h"
#include "uocr/weights.h"

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
    UOCR_CHECK(f.good(), "cannot open " + path);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models";
    std::string ref_dir = "/tmp/opencode/ref_vision";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) ref_dir = argv[++i];
    }

    std::ifstream mf(ref_dir + "/manifest.json");
    UOCR_CHECK(mf.good(), "cannot open " + ref_dir + "/manifest.json");
    json manifest = json::parse(mf);
    const json& tensors = manifest.at("tensors");

    ModelConfig cfg = ModelConfig::from_json_file(model_dir + "/config.json");
    const std::string weights_path = model_dir + "/model-00001-of-000001.safetensors";

    std::printf("loading vision + projector weights ...\n");
    SafetensorsFile st(weights_path);
    VisionWeights vw = VisionWeights::load(st, cfg);
    DecoderWeights dw = DecoderWeights::load(weights_path, cfg, false, 128);
    DeepEncoder enc(cfg, std::move(vw), std::move(dw));

    std::vector<float> image = load_bin(ref_dir, tensors.at("image"));
    const auto& ishape = tensors.at("image").at("shape");
    const int H = ishape[1].get<int>();
    const int W = ishape[2].get<int>();
    std::printf("image %dx%d, running DeepEncoder (CPU reference, may take a while) ...\n", H, W);

    Tensor out;
    enc.encode(image.data(), H, W, out);

    std::vector<float> ref = load_bin(ref_dir, tensors.at("visual_embeddings"));
    std::printf("ours: [%lld,%lld]  ref: [%d,%d]  tokens expected=%d\n", static_cast<long long>(out.dim(0)),
                static_cast<long long>(out.dim(1)), manifest.at("num_visual_tokens").get<int>(),
                manifest.at("hidden").get<int>(), enc.num_tokens());

    if (out.numel() != static_cast<i64>(ref.size())) {
        std::printf("SHAPE MISMATCH: ours=%lld ref=%zu\n", static_cast<long long>(out.numel()),
                    ref.size());
        return 1;
    }

    double num = 0, den = 0, max_abs = 0;
    for (i64 i = 0; i < out.numel(); ++i) {
        const double e = std::fabs(static_cast<double>(out[i]) - ref[i]);
        max_abs = std::max(max_abs, e);
        num += e * e;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    std::printf("visual embeddings: max_abs=%.5f rel_l2=%.5f\n", max_abs,
                std::sqrt(num / (den + 1e-30)));
    return 0;
}
