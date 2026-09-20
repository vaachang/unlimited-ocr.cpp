// compare_vision_gpu -- validates the CUDA DeepEncoder against the CPU
// reference encoder on the real vision weights.  The CPU encoder itself is
// already aligned to the PyTorch f32 reference (see ALIGNMENT.md 4), so a
// small GPU-vs-CPU rel_l2 is the acceptance criterion for the port.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "uocr/config.h"
#include "uocr/cuda_ops.h"
#include "uocr/deep_encoder.h"
#include "uocr/gpu_encoder.h"
#include "uocr/safetensors.h"
#include "uocr/weights.h"

using namespace uocr;

namespace {

void report(const char* name, const std::vector<float>& ours, const std::vector<float>& ref,
            std::size_t begin = 0) {
    if (ours.size() + begin != ref.size()) {
        std::printf("%-20s SIZE MISMATCH ours=%zu ref=%zu begin=%zu\n", name, ours.size(),
                    ref.size(), begin);
        return;
    }
    double num = 0, den = 0, maxabs = 0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
        const double e = static_cast<double>(ours[i]) - ref[begin + i];
        num += e * e;
        den += static_cast<double>(ref[begin + i]) * ref[begin + i];
        maxabs = std::max(maxabs, std::fabs(e));
    }
    std::printf("%-20s max_abs=%.5f rel_l2=%.6f\n", name, maxabs, std::sqrt(num / (den + 1e-30)));
}

float* dev_up(const std::vector<float>& v) {
    float* p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(float));
    cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    return p;
}
void dev_down(const float* p, std::vector<float>& v) {
    cudaMemcpy(v.data(), p, v.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

// Independent CPU reference for the decomposed-rel-pos attention, matching
// src/engine/deep_encoder.cpp `sam_attention`.
std::vector<float> cpu_sam_attention(const std::vector<float>& qkv,
                                     const std::vector<float>& relh,
                                     const std::vector<float>& relw, int B, int H, int W,
                                     int heads) {
    const int C = heads * 64, hd = 64;
    std::vector<float> out(static_cast<std::size_t>(B) * H * W * C, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < heads; ++h)
            for (int qi = 0; qi < H * W; ++qi) {
                const int qh = qi / W, qw = qi % W;
                const float* qp = qkv.data() + (static_cast<std::size_t>(b) * H * W + qi) * 3 * C + h * hd;
                float m = -1e30f;
                std::vector<float> sc(H * W);
                for (int ki = 0; ki < H * W; ++ki) {
                    const int kh = ki / W, kw = ki % W;
                    const float* kp = qkv.data() + (static_cast<std::size_t>(b) * H * W + ki) * 3 * C + C + h * hd;
                    float dot = 0;
                    for (int d = 0; d < hd; ++d) dot += qp[d] * kp[d];
                    float s = dot * scale;
                    const float* rhp = relh.data() + (static_cast<std::size_t>(qh - kh + H - 1) * hd);
                    const float* rwp = relw.data() + (static_cast<std::size_t>(qw - kw + W - 1) * hd);
                    for (int d = 0; d < hd; ++d) s += qp[d] * (rhp[d] + rwp[d]);
                    sc[ki] = s;
                    m = std::max(m, s);
                }
                float l = 0;
                std::vector<float> acc(hd, 0.0f);
                for (int ki = 0; ki < H * W; ++ki) {
                    const float e = std::exp(sc[ki] - m);
                    l += e;
                    const float* vp = qkv.data() + (static_cast<std::size_t>(b) * H * W + ki) * 3 * C + 2 * C + h * hd;
                    for (int d = 0; d < hd; ++d) acc[d] += e * vp[d];
                }
                float* op = out.data() + (static_cast<std::size_t>(b) * H * W + qi) * C + h * hd;
                for (int d = 0; d < hd; ++d) op[d] = acc[d] / l;
            }
    return out;
}

const std::vector<float>* stage_of(const std::vector<DeepEncoder::Stage>& s, const std::string& n) {
    for (const auto& x : s)
        if (x.first == n) return &x.second;
    return nullptr;
}

std::vector<float> host_window_partition(const std::vector<float>& x, int H, int W, int C, int ws,
                                         int& nh, int& nw) {
    const int pad_h = (ws - H % ws) % ws, pad_w = (ws - W % ws) % ws;
    const int Hp = H + pad_h, Wp = W + pad_w;
    nh = Hp / ws;
    nw = Wp / ws;
    std::vector<float> out(static_cast<std::size_t>(nh * nw) * ws * ws * C, 0.0f);
    for (int wi = 0; wi < nh; ++wi)
        for (int wj = 0; wj < nw; ++wj) {
            float* dst = out.data() + (static_cast<std::size_t>(wi * nw + wj) * ws * ws) * C;
            for (int i = 0; i < ws; ++i)
                for (int j = 0; j < ws; ++j) {
                    const int ih = wi * ws + i, iw = wj * ws + j;
                    if (ih >= H || iw >= W) continue;
                    std::memcpy(dst + (static_cast<std::size_t>(i) * ws + j) * C,
                                x.data() + (static_cast<std::size_t>(ih) * W + iw) * C,
                                C * sizeof(float));
                }
        }
    return out;
}

std::vector<float> host_window_unpartition(const std::vector<float>& win, int ws, int nh, int nw,
                                           int H, int W, int C) {
    std::vector<float> x(static_cast<std::size_t>(H) * W * C, 0.0f);
    for (int wi = 0; wi < nh; ++wi)
        for (int wj = 0; wj < nw; ++wj) {
            const float* src = win.data() + (static_cast<std::size_t>(wi * nw + wj) * ws * ws) * C;
            for (int i = 0; i < ws; ++i)
                for (int j = 0; j < ws; ++j) {
                    const int ih = wi * ws + i, iw = wj * ws + j;
                    if (ih >= H || iw >= W) continue;
                    std::memcpy(x.data() + (static_cast<std::size_t>(ih) * W + iw) * C,
                                src + (static_cast<std::size_t>(i) * ws + j) * C,
                                C * sizeof(float));
                }
        }
    return x;
}

void check_block0(const VisionWeights& vw, const std::vector<DeepEncoder::Stage>& gs,
                  const std::vector<DeepEncoder::Stage>& cs, int grid, int dim, int heads) {
    const auto* x0 = stage_of(gs, "gpu_sam_pos");
    const auto* gnorm = stage_of(gs, "gpu_norm1");
    const auto* gqkv = stage_of(gs, "gpu_qkv");
    const auto* gattn = stage_of(gs, "gpu_attn0");
    if (!x0 || !gnorm || !gqkv || !gattn) {
        std::printf("check_block0: missing stages\n");
        return;
    }
    const int N = grid * grid, C3 = 3 * dim;
    const SAMBlockWeights& b0 = vw.sam.blocks[0];
    // layernorm
    std::vector<float> norm(static_cast<std::size_t>(N) * dim);
    for (int r = 0; r < N; ++r) {
        float mean = 0;
        for (int c = 0; c < dim; ++c) mean += (*x0)[r * dim + c];
        mean /= dim;
        float var = 0;
        for (int c = 0; c < dim; ++c) var += ((*x0)[r * dim + c] - mean) * ((*x0)[r * dim + c] - mean);
        var /= dim;
        const float inv = 1.0f / std::sqrt(var + 1e-6f);
        for (int c = 0; c < dim; ++c)
            norm[r * dim + c] = ((*x0)[r * dim + c] - mean) * inv * b0.norm1_w[c] + b0.norm1_b[c];
    }
    char lbl[64];
    std::snprintf(lbl, sizeof(lbl), "b0 layernorm");
    report(lbl, *gnorm, norm);
    // qkv
    std::vector<float> qkv(static_cast<std::size_t>(N) * C3);
    b0.qkv_w.matmul(norm.data(), qkv.data(), N);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < C3; ++c) qkv[r * C3 + c] += b0.qkv_b[c];
    std::snprintf(lbl, sizeof(lbl), "b0 qkv");
    report(lbl, *gqkv, qkv);
    // attention (windowed)
    int nh = 0, nw = 0;
    std::vector<float> win = host_window_partition(qkv, grid, grid, C3, 14, nh, nw);
    std::vector<float> attn_win = cpu_sam_attention(win, b0.rel_pos_h, b0.rel_pos_w, nh * nw, 14, 14, heads);
    std::vector<float> attn = host_window_unpartition(attn_win, 14, nh, nw, grid, grid, dim);
    std::snprintf(lbl, sizeof(lbl), "b0 attn");
    report(lbl, *gattn, attn);
    const auto* gproj = stage_of(gs, "gpu_proj0");
    const auto* gnorm2 = stage_of(gs, "gpu_norm2");
    const auto* gmlp1 = stage_of(gs, "gpu_mlp1");
    const auto* gmlp2 = stage_of(gs, "gpu_mlp2");
    const auto* gblock0 = stage_of(gs, "gpu_sam_block0");
    std::vector<float> proj(static_cast<std::size_t>(N) * dim);
    b0.proj_w.matmul(attn.data(), proj.data(), N);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < dim; ++c) proj[r * dim + c] += b0.proj_b[c];
    report("b0 proj", *gproj, proj);
    std::vector<float> x1(static_cast<std::size_t>(N) * dim);
    for (std::size_t i = 0; i < x1.size(); ++i) x1[i] = (*x0)[i] + proj[i];
    // norm2
    std::vector<float> norm2(static_cast<std::size_t>(N) * dim);
    for (int r = 0; r < N; ++r) {
        float mean = 0;
        for (int c = 0; c < dim; ++c) mean += x1[r * dim + c];
        mean /= dim;
        float var = 0;
        for (int c = 0; c < dim; ++c) var += (x1[r * dim + c] - mean) * (x1[r * dim + c] - mean);
        var /= dim;
        const float inv = 1.0f / std::sqrt(var + 1e-6f);
        for (int c = 0; c < dim; ++c)
            norm2[r * dim + c] = (x1[r * dim + c] - mean) * inv * b0.norm2_w[c] + b0.norm2_b[c];
    }
    report("b0 norm2", *gnorm2, norm2);
    const int I4 = 4 * dim;
    std::vector<float> mlp1(static_cast<std::size_t>(N) * I4);
    b0.mlp_lin1_w.matmul(norm2.data(), mlp1.data(), N);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < I4; ++c) {
            float v = mlp1[r * I4 + c] + b0.mlp_lin1_b[c];
            mlp1[r * I4 + c] = 0.5f * v * (1.0f + std::erff(v * 0.7071067811865476f));
        }
    report("b0 mlp1", *gmlp1, mlp1);
    std::vector<float> mlp2(static_cast<std::size_t>(N) * dim);
    b0.mlp_lin2_w.matmul(mlp1.data(), mlp2.data(), N);
    std::vector<float> x2(static_cast<std::size_t>(N) * dim);
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < dim; ++c)
            x2[r * dim + c] = x1[r * dim + c] + mlp2[r * dim + c] + b0.mlp_lin2_b[c];
    std::snprintf(lbl, sizeof(lbl), "b0 block");
    report(lbl, *gblock0, x2);
    const auto* cblock0 = stage_of(cs, "sam_block0");
    if (cblock0) {
        report("b0 our-vs-cpu", x2, *cblock0);
        report("b0 gpu-vs-cpu", *gblock0, *cblock0);
    }
    const auto* cx0 = stage_of(cs, "sam_pos");
    if (cx0) report("b0 x0 gpu-vs-cpu", *x0, *cx0);
    if (cx0) {
        // CPU layernorm of the CPU x0
        std::vector<float> n2(static_cast<std::size_t>(N) * dim);
        for (int r = 0; r < N; ++r) {
            float mean = 0;
            for (int c = 0; c < dim; ++c) mean += (*cx0)[r * dim + c];
            mean /= dim;
            float var = 0;
            for (int c = 0; c < dim; ++c)
                var += ((*cx0)[r * dim + c] - mean) * ((*cx0)[r * dim + c] - mean);
            var /= dim;
            const float inv = 1.0f / std::sqrt(var + 1e-6f);
            for (int c = 0; c < dim; ++c)
                n2[r * dim + c] =
                    ((*cx0)[r * dim + c] - mean) * inv * b0.norm1_w[c] + b0.norm1_b[c];
        }
        report("b0 ln(cpu x0)", *gnorm, n2);
    }
}

int selftest() {
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    // --- plain attention (CLIP-like: S=257, heads=16, relpos=false) ---
    {
        const int S = 257, heads = 16, hd = 64, C = heads * hd;
        std::vector<float> qkv(static_cast<std::size_t>(S) * 3 * C);
        for (auto& v : qkv) v = dist(rng);
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        std::vector<float> ref(static_cast<std::size_t>(S) * C, 0.0f);
        for (int h = 0; h < heads; ++h)
            for (int qi = 0; qi < S; ++qi) {
                const float* qp = qkv.data() + static_cast<std::size_t>(qi) * 3 * C + h * hd;
                float m = -1e30f;
                std::vector<float> sc(S);
                for (int ki = 0; ki < S; ++ki) {
                    const float* kp = qkv.data() + static_cast<std::size_t>(ki) * 3 * C + C + h * hd;
                    float dot = 0;
                    for (int d = 0; d < hd; ++d) dot += qp[d] * kp[d];
                    sc[ki] = dot * scale;
                    m = std::max(m, sc[ki]);
                }
                float l = 0;
                std::vector<float> acc(hd, 0.0f);
                for (int ki = 0; ki < S; ++ki) {
                    const float e = std::exp(sc[ki] - m);
                    l += e;
                    const float* vp = qkv.data() + static_cast<std::size_t>(ki) * 3 * C + 2 * C + h * hd;
                    for (int d = 0; d < hd; ++d) acc[d] += e * vp[d];
                }
                float* op = ref.data() + static_cast<std::size_t>(qi) * C + h * hd;
                for (int d = 0; d < hd; ++d) op[d] = acc[d] / l;
            }
        float* d_qkv = dev_up(qkv);
        float* d_ref = dev_up(ref);
        cuda::attention_flash(d_qkv, nullptr, nullptr, d_ref, 1, S, 1, 1, heads, false);
        std::vector<float> got(ref.size());
        dev_down(d_ref, got);
        report("selftest attn full", got, ref);
        cudaFree(d_qkv);
        cudaFree(d_ref);
    }
    // --- layernorm ---
    {
        const int R = 64, C = 768;
        std::vector<float> x(R * C), w(C), b(C);
        for (auto& v : x) v = dist(rng);
        for (auto& v : w) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        std::vector<float> ref(R * C);
        for (int r = 0; r < R; ++r) {
            float mean = 0;
            for (int c = 0; c < C; ++c) mean += x[r * C + c];
            mean /= C;
            float var = 0;
            for (int c = 0; c < C; ++c) var += (x[r * C + c] - mean) * (x[r * C + c] - mean);
            var /= C;
            const float inv = 1.0f / std::sqrt(var + 1e-6f);
            for (int c = 0; c < C; ++c)
                ref[r * C + c] = (x[r * C + c] - mean) * inv * w[c] + b[c];
        }
        float *dx = dev_up(x), *dw = dev_up(w), *db = dev_up(b), *dy = dev_up(ref);
        cuda::layernorm_rows(dx, dw, db, dy, R, C, 1e-6f);
        std::vector<float> got(R * C);
        dev_down(dy, got);
        report("selftest layernorm", got, ref);
        cudaFree(dx);
        cudaFree(dw);
        cudaFree(db);
        cudaFree(dy);
    }
    // --- attention ---
    {
        const int H = 14, W = 14, heads = 12, hd = 64, S = H * W, B = 9;
        const int C = heads * hd;
        std::vector<float> qkv(static_cast<std::size_t>(B) * S * 3 * C);
        for (auto& v : qkv) v = dist(rng);
        std::vector<float> relh(static_cast<std::size_t>(2 * H - 1) * hd);
        std::vector<float> relw(static_cast<std::size_t>(2 * W - 1) * hd);
        for (auto& v : relh) v = dist(rng);
        for (auto& v : relw) v = dist(rng);
        std::vector<float> ref = cpu_sam_attention(qkv, relh, relw, B, H, W, heads);
        float *d_qkv = dev_up(qkv), *d_rh = dev_up(relh), *d_rw = dev_up(relw);
        float* d_out = dev_up(ref);
        cuda::attention_flash(d_qkv, d_rh, d_rw, d_out, B, S, H, W, heads, true);
        std::vector<float> got(ref.size());
        dev_down(d_out, got);
        report("selftest attention", got, ref);
        cudaFree(d_qkv);
        cudaFree(d_rh);
        cudaFree(d_rw);
        cudaFree(d_out);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = "models";
    int size = 640;
    bool run_cpu = true;
    unsigned seed = 1234;
    bool do_selftest = false;
    bool want_stages = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--size") && i + 1 < argc) size = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-cpu")) run_cpu = false;
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--selftest")) do_selftest = true;
        else if (!std::strcmp(argv[i], "--stages")) want_stages = true;
    }
    if (do_selftest) return selftest();

    ModelConfig cfg = ModelConfig::from_json_file(model_dir + "/config.json");
    const std::string weights_path = model_dir + "/model-00001-of-000001.safetensors";
    std::printf("loading vision + projector weights ...\n");
    SafetensorsFile st(weights_path);
    VisionWeights vw = VisionWeights::load(st, cfg);
    DecoderWeights dw = DecoderWeights::load(weights_path, cfg, false, 128);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> image(static_cast<std::size_t>(3) * size * size);
    for (auto& v : image) v = (dist(rng) - 0.5f);  // normalized-ish input

    cuda::GpuEncoder gpu(cfg, vw, dw);
    Tensor gout;
    std::vector<float> gsam, gclip;
    std::vector<DeepEncoder::Stage> gstages;
    {
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        gpu.encode(image.data(), size, size, gout, &gsam, &gclip, nullptr);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("GPU encode time: %.1f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (want_stages) {
        gpu.encode(image.data(), size, size, gout, &gsam, &gclip, &gstages);
    }
    std::printf("GPU encode done: [%lld,%lld] tokens=%d\n", static_cast<long long>(gout.dim(0)),
                static_cast<long long>(gout.dim(1)), gpu.num_tokens());

    if (!run_cpu) return 0;

    DeepEncoder cpu(cfg, vw, dw);
    Tensor cout;
    std::vector<float> csam, cclip;
    std::vector<DeepEncoder::Stage> cstages;
    cpu.encode_stages(image.data(), size, size, cout, &cstages, &csam, &cclip);
    std::printf("CPU encode done: [%lld,%lld] tokens=%d\n", static_cast<long long>(cout.dim(0)),
                static_cast<long long>(cout.dim(1)), cpu.num_tokens());

    if (want_stages)
        check_block0(vw, gstages, cstages, size / cfg.patch_size, cfg.sam_embed_dim, cfg.sam_heads);

    // Per-stage comparison (GPU stage names carry a gpu_ prefix).
    for (const auto& gs : gstages) {
        std::string cn = gs.first;
        if (cn.rfind("gpu_", 0) == 0) cn = cn.substr(4);
        const std::vector<float>* ref = nullptr;
        for (const auto& cs : cstages)
            if (cs.first == cn) ref = &cs.second;
        if (!ref) continue;
        char label[64];
        std::snprintf(label, sizeof(label), "stage %s", cn.c_str());
        report(label, gs.second, *ref);
    }

    report("sam_features", gsam, csam);
    report("clip_features", gclip, cclip);
    std::vector<float> gv(gout.data(), gout.data() + gout.numel());
    std::vector<float> cv(cout.data(), cout.data() + cout.numel());
    report("visual_embeddings", gv, cv);
    return 0;
}
