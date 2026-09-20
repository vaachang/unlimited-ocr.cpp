// GPU DeepEncoder implementation.  Mirrors src/engine/deep_encoder.cpp stage by
// stage; see that file for the reference semantics.

#include "uocr/gpu_encoder.h"

#if defined(UOCR_CUDA_ENABLED)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "uocr/cuda_ops.h"
#include "uocr/log.h"

namespace uocr {
namespace cuda {
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        UOCR_CHECK(false, std::string("GpuEncoder: ") + what + ": " + cudaGetErrorString(e));
    }
}

inline std::size_t bytes_f32(std::size_t n) { return n * sizeof(float); }

// ---------------------------------------------------------------------------
// Host bicubic resize (align_corners=false), used for the CLIP position
// embedding when the token grid differs from the checkpoint's 16x16.
// ---------------------------------------------------------------------------
float cubic_weight(float t) {
    const float a = -0.75f;
    t = std::fabs(t);
    if (t <= 1.0f) return ((a + 2) * t - (a + 3)) * t * t + 1.0f;
    if (t < 2.0f) return (((t - 5) * t + 8) * t - 4) * a;
    return 0.0f;
}

std::vector<float> bicubic_resize(const std::vector<float>& src, int Hi, int Wi, int C, int Ho,
                                  int Wo) {
    std::vector<float> out(static_cast<std::size_t>(Ho) * Wo * C);
    const float sh = static_cast<float>(Hi) / Ho;
    const float sw = static_cast<float>(Wi) / Wo;
    const float fh = sh > 1.0f ? sh : 1.0f;
    const float fw = sw > 1.0f ? sw : 1.0f;
    for (int oh = 0; oh < Ho; ++oh) {
        const float sy = (oh + 0.5f) * sh - 0.5f;
        const int y0 = static_cast<int>(std::floor(sy));
        for (int ow = 0; ow < Wo; ++ow) {
            const float sx = (ow + 0.5f) * sw - 0.5f;
            const int x0 = static_cast<int>(std::floor(sx));
            float wsum = 0.0f;
            float wacc[4][4];
            for (int j = 0; j < 4; ++j) {
                const float wy = cubic_weight((sy - (y0 - 1 + j)) / fh);
                for (int i = 0; i < 4; ++i) {
                    const float wx = cubic_weight((sx - (x0 - 1 + i)) / fw);
                    wacc[j][i] = wx * wy;
                    wsum += wx * wy;
                }
            }
            float* po = out.data() + (static_cast<std::size_t>(oh) * Wo + ow) * C;
            if (wsum != 0.0f) {
                for (int c = 0; c < C; ++c) po[c] = 0.0f;
                for (int j = 0; j < 4; ++j) {
                    const int iy = std::clamp(y0 - 1 + j, 0, Hi - 1);
                    for (int i = 0; i < 4; ++i) {
                        const int ix = std::clamp(x0 - 1 + i, 0, Wi - 1);
                        const float wgt = wacc[j][i] / wsum;
                        const float* ps = src.data() + (static_cast<std::size_t>(iy) * Wi + ix) * C;
                        for (int c = 0; c < C; ++c) po[c] += wgt * ps[c];
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace

struct GpuEncoder::Impl {
    ModelConfig cfg;
    std::vector<void*> owned;

    template <typename T>
    T* alloc(std::size_t n) {
        T* p = nullptr;
        check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
        owned.push_back(p);
        return p;
    }

    void upload_f32(const std::vector<float>& v, float** out) {
        *out = alloc<float>(v.size());
        check(cudaMemcpy(*out, v.data(), bytes_f32(v.size()), cudaMemcpyHostToDevice), "H2D f32");
    }

    void to_bf16_dev(const float* data, std::size_t n, const std::uint16_t** out) {
        std::vector<std::uint16_t> t(n);
        for (std::size_t i = 0; i < n; ++i) t[i] = f32_to_bf16(data[i]);
        std::uint16_t* p = alloc<std::uint16_t>(n);
        check(cudaMemcpy(p, t.data(), n * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
              "H2D bf16");
        *out = p;
    }

    void upload_mat(const WeightMatrix& w, const std::uint16_t** out) {
        std::vector<float> f;
        w.to_f32(f);
        to_bf16_dev(f.data(), f.size(), out);
    }

    ~Impl() {
        for (void* p : owned) cudaFree(p);
    }

    // ---- device weights ----
    struct DBlock {
        float *n1w = nullptr, *n1b = nullptr;
        const std::uint16_t* qkvw = nullptr;
        float* qkvb = nullptr;
        const std::uint16_t* projw = nullptr;
        float* projb = nullptr;
        float *rh = nullptr, *rw = nullptr;
        float *n2w = nullptr, *n2b = nullptr;
        const std::uint16_t* m1w = nullptr;
        float* m1b = nullptr;
        const std::uint16_t* m2w = nullptr;
        float* m2b = nullptr;
    };
    struct DCLayer {
        float *ln1w = nullptr, *ln1b = nullptr;
        const std::uint16_t* qkvw = nullptr;
        float* qkvb = nullptr;
        const std::uint16_t* outw = nullptr;
        float* outb = nullptr;
        float *ln2w = nullptr, *ln2b = nullptr;
        const std::uint16_t* fc1w = nullptr;
        float* fc1b = nullptr;
        const std::uint16_t* fc2w = nullptr;
        float* fc2b = nullptr;
    };

    const std::uint16_t* patch_w = nullptr;
    float* patch_b = nullptr;
    float* sam_pos = nullptr;
    std::vector<DBlock> blocks;
    const std::uint16_t *neck0 = nullptr, *neck2 = nullptr, *net2w = nullptr, *net3w = nullptr;
    float *neck1w = nullptr, *neck1b = nullptr, *neck3w = nullptr, *neck3b = nullptr;

    float *class_emb = nullptr, *clip_pos = nullptr, *pre_ln_w = nullptr, *pre_ln_b = nullptr;
    std::vector<DCLayer> clayers;
    const std::uint16_t* proj_w = nullptr;
    float* proj_b = nullptr;

    std::vector<float> image_newline_host, view_sep_host;
    int clip_pos_rows = 16;

    void load(const VisionWeights& vw, const DecoderWeights& pw) {
        const std::string sp;
        (void)sp;
        upload_mat_f32(vw.sam.patch_w, &patch_w);
        upload_f32(vw.sam.patch_b, &patch_b);
        upload_f32(vw.sam.pos_embed, &sam_pos);
        blocks.resize(vw.sam.blocks.size());
        for (std::size_t i = 0; i < vw.sam.blocks.size(); ++i) {
            const SAMBlockWeights& b = vw.sam.blocks[i];
            DBlock& d = blocks[i];
            upload_f32(b.norm1_w, &d.n1w);
            upload_f32(b.norm1_b, &d.n1b);
            upload_mat(b.qkv_w, &d.qkvw);
            upload_f32(b.qkv_b, &d.qkvb);
            upload_mat(b.proj_w, &d.projw);
            upload_f32(b.proj_b, &d.projb);
            upload_f32(b.rel_pos_h, &d.rh);
            upload_f32(b.rel_pos_w, &d.rw);
            upload_f32(b.norm2_w, &d.n2w);
            upload_f32(b.norm2_b, &d.n2b);
            upload_mat(b.mlp_lin1_w, &d.m1w);
            upload_f32(b.mlp_lin1_b, &d.m1b);
            upload_mat(b.mlp_lin2_w, &d.m2w);
            upload_f32(b.mlp_lin2_b, &d.m2b);
        }
        upload_mat_f32(vw.sam.neck0_w, &neck0);
        upload_f32(vw.sam.neck1_w, &neck1w);
        upload_f32(vw.sam.neck1_b, &neck1b);
        upload_mat_f32(vw.sam.neck2_w, &neck2);
        upload_f32(vw.sam.neck3_w, &neck3w);
        upload_f32(vw.sam.neck3_b, &neck3b);
        upload_mat_f32(vw.sam.net2_w, &net2w);
        upload_mat_f32(vw.sam.net3_w, &net3w);

        upload_f32(vw.clip.class_embedding, &class_emb);
        upload_f32(vw.clip.pos_embed, &clip_pos);
        upload_f32(vw.clip.pre_ln_w, &pre_ln_w);
        upload_f32(vw.clip.pre_ln_b, &pre_ln_b);
        clayers.resize(vw.clip.layers.size());
        for (std::size_t i = 0; i < vw.clip.layers.size(); ++i) {
            const CLIPLayerWeights& L = vw.clip.layers[i];
            DCLayer& d = clayers[i];
            upload_f32(L.ln1_w, &d.ln1w);
            upload_f32(L.ln1_b, &d.ln1b);
            upload_mat(L.qkv_w, &d.qkvw);
            upload_f32(L.qkv_b, &d.qkvb);
            upload_mat(L.out_w, &d.outw);
            upload_f32(L.out_b, &d.outb);
            upload_f32(L.ln2_w, &d.ln2w);
            upload_f32(L.ln2_b, &d.ln2b);
            upload_mat(L.fc1_w, &d.fc1w);
            upload_f32(L.fc1_b, &d.fc1b);
            upload_mat(L.fc2_w, &d.fc2w);
            upload_f32(L.fc2_b, &d.fc2b);
        }
        upload_mat(pw.projector.weight, &proj_w);
        upload_f32(pw.projector.bias, &proj_b);
        image_newline_host = pw.image_newline;
        view_sep_host = pw.view_seperator;
        const int hd = cfg.clip_hidden_size;
        const int n = static_cast<int>(vw.clip.pos_embed.size() / hd) - 1;
        clip_pos_rows = static_cast<int>(std::lround(std::sqrt(static_cast<double>(n))));
    }

    void upload_mat_f32(const std::vector<float>& v, const std::uint16_t** out) {
        to_bf16_dev(v.data(), v.size(), out);
    }
};

GpuEncoder::GpuEncoder(ModelConfig cfg, const VisionWeights& vw, const DecoderWeights& pw)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = std::move(cfg);
    impl_->load(vw, pw);
    UOCR_INFO("GpuEncoder: vision weights uploaded");
}

GpuEncoder::~GpuEncoder() = default;

void GpuEncoder::encode(const float* image_chw, int height, int width, Tensor& out,
                        std::vector<float>* sam_debug, std::vector<float>* clip_debug,
                        std::vector<std::pair<std::string, std::vector<float>>>* stages) {
    (void)sam_debug;
    (void)clip_debug;
    Impl& I = *impl_;
    const ModelConfig& cfg = I.cfg;
    const int patch = cfg.patch_size;
    UOCR_CHECK(height == width, "GpuEncoder expects a square image");
    const int grid = height / patch;  // 64 for 1024
    const int N = grid * grid;
    const int dim = cfg.sam_embed_dim;  // 768

    std::vector<void*> tmp;
    auto f32buf = [&](std::size_t n) {
        float* p = nullptr;
        check(cudaMalloc(&p, n * sizeof(float)), "tmp malloc");
        tmp.push_back(p);
        return p;
    };
    struct FreeGuard {
        std::vector<void*>& v;
        ~FreeGuard() {
            for (void* p : v) cudaFree(p);
        }
    } guard{tmp};

    auto record = [&](const std::string& name, const float* dev, std::size_t n) {
        if (!stages) return;
        std::vector<float> h(n);
        check(cudaMemcpy(h.data(), dev, bytes_f32(n), cudaMemcpyDeviceToHost), "stage D2H");
        stages->emplace_back(name, std::move(h));
    };

    // host image -> device
    float* img = I.alloc<float>(static_cast<std::size_t>(3) * height * width);
    check(cudaMemcpy(img, image_chw, bytes_f32(static_cast<std::size_t>(3) * height * width),
                     cudaMemcpyHostToDevice),
          "H2D image");

    // ---- patch embed + pos ----
    float* x = f32buf(static_cast<std::size_t>(N) * dim);
    {
        const int k = 3 * patch * patch;
        float* col = f32buf(static_cast<std::size_t>(N) * k);
        cuda::im2col_chw(img, col, 3, height, width, patch, patch, patch, 0, grid, grid);
        cuda::matmul_t_split_bf16(col, I.patch_w, I.patch_b, x, N, dim, k);
        cuda::add_inplace(x, I.sam_pos, N * dim);
    }
    record("gpu_sam_pos", x, static_cast<std::size_t>(N) * dim);

    // ---- SAM blocks ----
    const int sam_heads = cfg.sam_heads;
    float* normed = f32buf(static_cast<std::size_t>(N) * dim);
    float* qkv = f32buf(static_cast<std::size_t>(N) * 3 * dim);
    float* attn = f32buf(static_cast<std::size_t>(N) * dim);
    float* proj = f32buf(static_cast<std::size_t>(N) * dim);
    float* mlp1 = f32buf(static_cast<std::size_t>(N) * 4 * dim);
    float* mlp2 = f32buf(static_cast<std::size_t>(N) * dim);
    for (int bi = 0; bi < cfg.sam_depth; ++bi) {
        const Impl::DBlock& d = I.blocks[bi];
        const bool is_global = (bi >= 2 && (bi - 2) % 3 == 0);
        cuda::layernorm_rows(x, d.n1w, d.n1b, normed, N, dim, 1e-6f);
        if (is_global) {
            // Global attention runs over the whole grid: QKV then attention.
            cuda::matmul_t_split_bf16(normed, d.qkvw, d.qkvb, qkv, N, 3 * dim, dim);
            cuda::attention_flash(qkv, d.rh, d.rw, attn, 1, N, grid, grid, sam_heads, true);
        } else {
            // Windowed attention mirrors the reference: partition the normed
            // activations (zero padding) *then* apply QKV, so the padded window
            // tokens carry the QKV bias rather than zero.
            const int ws = cfg.sam_window_size;
            const int nh = (grid + ws - 1) / ws;
            const int nw = (grid + ws - 1) / ws;
            const int nwin = nh * nw;
            const int sw = ws * ws;
            const int rows = nwin * sw;
            float* normed_win = f32buf(static_cast<std::size_t>(rows) * dim);
            float* qkv_win = f32buf(static_cast<std::size_t>(rows) * 3 * dim);
            float* attn_win = f32buf(static_cast<std::size_t>(rows) * dim);
            cuda::window_partition(normed, normed_win, grid, grid, dim, ws, nh, nw);
            cuda::matmul_t_split_bf16(normed_win, d.qkvw, d.qkvb, qkv_win, rows, 3 * dim, dim);
            cuda::attention_flash(qkv_win, d.rh, d.rw, attn_win, nwin, sw, ws, ws, sam_heads, true);
            cuda::window_unpartition(attn_win, attn, grid, grid, dim, ws, nh, nw);
        }
        cuda::matmul_t_split_bf16(attn, d.projw, d.projb, proj, N, dim, dim);
        cuda::add_inplace(x, proj, N * dim);

        cuda::layernorm_rows(x, d.n2w, d.n2b, normed, N, dim, 1e-6f);
        cuda::matmul_t_split_bf16(normed, d.m1w, d.m1b, mlp1, N, 4 * dim, dim);
        cuda::gelu_inplace(mlp1, N * 4 * dim);
        cuda::matmul_t_split_bf16(mlp1, d.m2w, d.m2b, mlp2, N, dim, 4 * dim);
        cuda::add_inplace(x, mlp2, N * dim);
        if (bi == 0 && stages) record("gpu_sam_block0", x, static_cast<std::size_t>(N) * dim);
        if (bi == 1) record("gpu_sam_block1", x, static_cast<std::size_t>(N) * dim);
        if (bi == 2) record("gpu_sam_block2", x, static_cast<std::size_t>(N) * dim);
        if (bi == 3) record("gpu_sam_block3", x, static_cast<std::size_t>(N) * dim);
        if (bi == cfg.sam_depth - 1)
            record("gpu_sam_block" + std::to_string(bi), x, static_cast<std::size_t>(N) * dim);
    }

    // ---- neck ----
    constexpr int kNC = 256;
    float* hwc = f32buf(static_cast<std::size_t>(N) * kNC);
    cuda::matmul_t_split_bf16(x, I.neck0, nullptr, hwc, N, kNC, dim);
    cuda::layernorm_rows(hwc, I.neck1w, I.neck1b, hwc, N, kNC, 1e-6f);
    {
        const int k = kNC * 9;
        float* col = f32buf(static_cast<std::size_t>(N) * k);
        cuda::im2col_hwc(hwc, col, kNC, grid, grid, 3, 3, 1, 1, grid, grid);
        cuda::matmul_t_split_bf16(col, I.neck2, nullptr, hwc, N, kNC, k);
        cuda::layernorm_rows(hwc, I.neck3w, I.neck3b, hwc, N, kNC, 1e-6f);
    }
    record("gpu_sam_neck", hwc, static_cast<std::size_t>(N) * kNC);
    const int h2 = (grid + 2 * 1 - 3) / 2 + 1;
    const int N2 = h2 * h2;
    float* x2 = f32buf(static_cast<std::size_t>(N2) * 512);
    {
        const int k = kNC * 9;
        float* col = f32buf(static_cast<std::size_t>(N2) * k);
        cuda::im2col_hwc(hwc, col, kNC, grid, grid, 3, 3, 2, 1, h2, h2);
        cuda::matmul_t_split_bf16(col, I.net2w, nullptr, x2, N2, 512, k);
    }
    record("gpu_sam_net2", x2, static_cast<std::size_t>(N2) * 512);
    const int h3 = (h2 + 2 * 1 - 3) / 2 + 1;
    const int N3 = h3 * h3;
    float* sam_tokens = f32buf(static_cast<std::size_t>(N3) * 1024);
    {
        const int k = 512 * 9;
        float* col = f32buf(static_cast<std::size_t>(N3) * k);
        cuda::im2col_hwc(x2, col, 512, h2, h2, 3, 3, 2, 1, h3, h3);
        cuda::matmul_t_split_bf16(col, I.net3w, nullptr, sam_tokens, N3, 1024, k);
    }
    record("gpu_sam_net3", sam_tokens, static_cast<std::size_t>(N3) * 1024);

    // ---- CLIP ----
    const int chd = cfg.clip_hidden_size;  // 1024
    const int Nc = N3 + 1;
    float* clip = f32buf(static_cast<std::size_t>(Nc) * chd);
    check(cudaMemcpy(clip, I.class_emb, bytes_f32(chd), cudaMemcpyDeviceToDevice), "class emb");
    check(cudaMemcpy(clip + chd, sam_tokens, bytes_f32(static_cast<std::size_t>(N3) * chd),
                     cudaMemcpyDeviceToDevice),
          "sam tokens");
    // position embedding (interpolate the 16x16 grid when needed)
    {
        const int src_grid = I.clip_pos_rows;
        std::vector<float> pos_host(static_cast<std::size_t>(Nc) * chd);
        if (src_grid != h3) {
            std::vector<float> pos_full(static_cast<std::size_t>(src_grid) * src_grid * chd + chd);
            check(cudaMemcpy(pos_full.data(), I.clip_pos, bytes_f32(pos_full.size()),
                             cudaMemcpyDeviceToHost),
                  "clip pos D2H");
            std::vector<float> cls(pos_full.begin(), pos_full.begin() + chd);
            std::vector<float> gpos(pos_full.begin() + chd, pos_full.end());
            std::vector<float> resized = bicubic_resize(gpos, src_grid, src_grid, chd, h3, h3);
            std::memcpy(pos_host.data(), cls.data(), bytes_f32(chd));
            std::memcpy(pos_host.data() + chd, resized.data(), bytes_f32(resized.size()));
        } else {
            check(cudaMemcpy(pos_host.data(), I.clip_pos, bytes_f32(pos_host.size()),
                             cudaMemcpyDeviceToHost),
                  "clip pos D2H");
        }
        float* dpos = f32buf(pos_host.size());
        check(cudaMemcpy(dpos, pos_host.data(), bytes_f32(pos_host.size()), cudaMemcpyHostToDevice),
              "clip pos H2D");
        cuda::add_inplace(clip, dpos, Nc * chd);
    }
    record("gpu_clip_embeds", clip, static_cast<std::size_t>(Nc) * chd);
    cuda::layernorm_rows(clip, I.pre_ln_w, I.pre_ln_b, clip, Nc, chd, 1e-5f);
    record("gpu_clip_preln", clip, static_cast<std::size_t>(Nc) * chd);

    const int clip_heads = cfg.clip_heads;
    float* cqkv = f32buf(static_cast<std::size_t>(Nc) * 3 * chd);
    float* cout = f32buf(static_cast<std::size_t>(Nc) * chd);
    float* residual = f32buf(static_cast<std::size_t>(Nc) * chd);
    float* fc1 = f32buf(static_cast<std::size_t>(Nc) * cfg.clip_ffn_size);
    for (const Impl::DCLayer& d : I.clayers) {
        check(cudaMemcpy(residual, clip, bytes_f32(static_cast<std::size_t>(Nc) * chd),
                         cudaMemcpyDeviceToDevice),
              "residual");
        cuda::layernorm_rows(clip, d.ln1w, d.ln1b, clip, Nc, chd, 1e-5f);
        cuda::matmul_t_split_bf16(clip, d.qkvw, d.qkvb, cqkv, Nc, 3 * chd, chd);
        cuda::attention_flash(cqkv, nullptr, nullptr, cout, 1, Nc, 1, 1, clip_heads, false);
        cuda::matmul_t_split_bf16(cout, d.outw, d.outb, clip, Nc, chd, chd);
        cuda::add_inplace(clip, residual, Nc * chd);

        check(cudaMemcpy(residual, clip, bytes_f32(static_cast<std::size_t>(Nc) * chd),
                         cudaMemcpyDeviceToDevice),
              "residual2");
        cuda::layernorm_rows(clip, d.ln2w, d.ln2b, clip, Nc, chd, 1e-5f);
        cuda::matmul_t_split_bf16(clip, d.fc1w, d.fc1b, fc1, Nc, cfg.clip_ffn_size, chd);
        cuda::quick_gelu_inplace(fc1, Nc * cfg.clip_ffn_size);
        // clip = fc2 + residual (overwrite the layer-normed value, like the CPU).
        cuda::matmul_t_split_bf16(fc1, d.fc2w, d.fc2b, clip, Nc, chd, cfg.clip_ffn_size);
        cuda::add_inplace(clip, residual, Nc * chd);
    }

    // ---- projector: concat(clip[1:], sam) -> linear ----
    const int proj_in = chd + 1024;
    float* cat = f32buf(static_cast<std::size_t>(N3) * proj_in);
    check(cudaMemcpy2D(cat, proj_in * sizeof(float), clip + chd, chd * sizeof(float),
                       chd * sizeof(float), N3, cudaMemcpyDeviceToDevice),
          "concat clip");
    check(cudaMemcpy2D(cat + chd, proj_in * sizeof(float), sam_tokens, 1024 * sizeof(float),
                       1024 * sizeof(float), N3, cudaMemcpyDeviceToDevice),
          "concat sam");
    const int hidden = cfg.projector_n_embed;
    float* projected = f32buf(static_cast<std::size_t>(N3) * hidden);
    cuda::matmul_t_split_bf16(cat, I.proj_w, I.proj_b, projected, N3, hidden, proj_in);

    check(cudaDeviceSynchronize(), "encode sync");
    if (sam_debug) {
        sam_debug->resize(static_cast<std::size_t>(N3) * 1024);
        check(cudaMemcpy(sam_debug->data(), sam_tokens, bytes_f32(sam_debug->size()),
                         cudaMemcpyDeviceToHost),
              "sam_debug D2H");
    }
    if (clip_debug) {
        clip_debug->resize(static_cast<std::size_t>(N3) * chd);
        check(cudaMemcpy(clip_debug->data(), clip + chd, bytes_f32(clip_debug->size()),
                         cudaMemcpyDeviceToHost),
              "clip_debug D2H");
    }
    std::vector<float> host_proj(static_cast<std::size_t>(N3) * hidden);
    check(cudaMemcpy(host_proj.data(), projected, bytes_f32(host_proj.size()),
                     cudaMemcpyDeviceToHost),
          "projected D2H");

    const int rows = h3, cols = h3 + 1;
    out = Tensor({rows * cols + 1, hidden});
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < h3; ++c) {
            std::memcpy(out.data() + static_cast<std::size_t>(r * cols + c) * hidden,
                        host_proj.data() + static_cast<std::size_t>(r * h3 + c) * hidden,
                        bytes_f32(hidden));
        }
        std::memcpy(out.data() + static_cast<std::size_t>(r * cols + h3) * hidden,
                    I.image_newline_host.data(), bytes_f32(hidden));
    }
    std::memcpy(out.data() + static_cast<std::size_t>(rows * cols) * hidden,
                I.view_sep_host.data(), bytes_f32(hidden));
    num_tokens_ = rows * cols + 1;
}

}  // namespace cuda
}  // namespace uocr

#endif  // UOCR_CUDA_ENABLED
