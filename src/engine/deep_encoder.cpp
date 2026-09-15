#include "uocr/deep_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "uocr/log.h"
#include "uocr/ops.h"

namespace uocr {

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------
namespace {

std::vector<float> read_any(const SafetensorsFile& st, const std::string& name) {
    return st.read_f32(name);
}

}  // namespace

VisionWeights VisionWeights::load(const SafetensorsFile& st, const ModelConfig& cfg) {
    VisionWeights v;
    const std::string sp = "model.sam_model.";

    v.sam.patch_w = read_any(st, sp + "patch_embed.proj.weight");
    v.sam.patch_b = read_any(st, sp + "patch_embed.proj.bias");
    v.sam.pos_embed = read_any(st, sp + "pos_embed");
    v.sam.neck0_w = read_any(st, sp + "neck.0.weight");
    v.sam.neck1_w = read_any(st, sp + "neck.1.weight");
    v.sam.neck1_b = read_any(st, sp + "neck.1.bias");
    v.sam.neck2_w = read_any(st, sp + "neck.2.weight");
    v.sam.neck3_w = read_any(st, sp + "neck.3.weight");
    v.sam.neck3_b = read_any(st, sp + "neck.3.bias");
    v.sam.net2_w = read_any(st, sp + "net_2.weight");
    v.sam.net3_w = read_any(st, sp + "net_3.weight");

    v.sam.blocks.resize(cfg.sam_depth);
    for (int i = 0; i < cfg.sam_depth; ++i) {
        const std::string p = sp + "blocks." + std::to_string(i) + ".";
        SAMBlockWeights& b = v.sam.blocks[i];
        b.norm1_w = read_any(st, p + "norm1.weight");
        b.norm1_b = read_any(st, p + "norm1.bias");
        b.qkv_w.rows = 3 * cfg.sam_embed_dim;
        b.qkv_w.cols = cfg.sam_embed_dim;
        b.qkv_w.fmt = WeightFormat::BF16_EXT;
        b.qkv_w.ext = st.info(p + "attn.qkv.weight").data;
        b.qkv_b = read_any(st, p + "attn.qkv.bias");
        b.proj_w.rows = cfg.sam_embed_dim;
        b.proj_w.cols = cfg.sam_embed_dim;
        b.proj_w.fmt = WeightFormat::BF16_EXT;
        b.proj_w.ext = st.info(p + "attn.proj.weight").data;
        b.proj_b = read_any(st, p + "attn.proj.bias");
        b.rel_pos_h = read_any(st, p + "attn.rel_pos_h");
        b.rel_pos_w = read_any(st, p + "attn.rel_pos_w");
        b.norm2_w = read_any(st, p + "norm2.weight");
        b.norm2_b = read_any(st, p + "norm2.bias");
        b.mlp_lin1_w.rows = 4 * cfg.sam_embed_dim;
        b.mlp_lin1_w.cols = cfg.sam_embed_dim;
        b.mlp_lin1_w.fmt = WeightFormat::BF16_EXT;
        b.mlp_lin1_w.ext = st.info(p + "mlp.lin1.weight").data;
        b.mlp_lin1_b = read_any(st, p + "mlp.lin1.bias");
        b.mlp_lin2_w.rows = cfg.sam_embed_dim;
        b.mlp_lin2_w.cols = 4 * cfg.sam_embed_dim;
        b.mlp_lin2_w.fmt = WeightFormat::BF16_EXT;
        b.mlp_lin2_w.ext = st.info(p + "mlp.lin2.weight").data;
        b.mlp_lin2_b = read_any(st, p + "mlp.lin2.bias");
    }

    const std::string cp = "model.vision_model.";
    v.clip.class_embedding = read_any(st, cp + "embeddings.class_embedding");
    v.clip.pos_embed = read_any(st, cp + "embeddings.position_embedding.weight");
    v.clip.pre_ln_w = read_any(st, cp + "pre_layrnorm.weight");
    v.clip.pre_ln_b = read_any(st, cp + "pre_layrnorm.bias");
    v.clip.layers.resize(cfg.clip_layers);
    for (int i = 0; i < cfg.clip_layers; ++i) {
        const std::string p = cp + "transformer.layers." + std::to_string(i) + ".";
        CLIPLayerWeights& L = v.clip.layers[i];
        L.ln1_w = read_any(st, p + "layer_norm1.weight");
        L.ln1_b = read_any(st, p + "layer_norm1.bias");
        L.qkv_w.rows = 3 * cfg.clip_hidden_size;
        L.qkv_w.cols = cfg.clip_hidden_size;
        L.qkv_w.fmt = WeightFormat::BF16_EXT;
        L.qkv_w.ext = st.info(p + "self_attn.qkv_proj.weight").data;
        L.qkv_b = read_any(st, p + "self_attn.qkv_proj.bias");
        L.out_w.rows = cfg.clip_hidden_size;
        L.out_w.cols = cfg.clip_hidden_size;
        L.out_w.fmt = WeightFormat::BF16_EXT;
        L.out_w.ext = st.info(p + "self_attn.out_proj.weight").data;
        L.out_b = read_any(st, p + "self_attn.out_proj.bias");
        L.ln2_w = read_any(st, p + "layer_norm2.weight");
        L.ln2_b = read_any(st, p + "layer_norm2.bias");
        L.fc1_w.rows = cfg.clip_ffn_size;
        L.fc1_w.cols = cfg.clip_hidden_size;
        L.fc1_w.fmt = WeightFormat::BF16_EXT;
        L.fc1_w.ext = st.info(p + "mlp.fc1.weight").data;
        L.fc1_b = read_any(st, p + "mlp.fc1.bias");
        L.fc2_w.rows = cfg.clip_hidden_size;
        L.fc2_w.cols = cfg.clip_ffn_size;
        L.fc2_w.fmt = WeightFormat::BF16_EXT;
        L.fc2_w.ext = st.info(p + "mlp.fc2.weight").data;
        L.fc2_b = read_any(st, p + "mlp.fc2.bias");
    }
    UOCR_INFO("vision weights loaded");
    return v;
}

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------
namespace {

// x: [Cin,H,W] row-major, w: [Cout,Cin,Kh,Kw], bias [Cout].
void conv2d(const float* x, int Cin, int H, int W, const float* w, const float* bias, int Cout,
            int Kh, int Kw, int stride, int pad, std::vector<float>& out, int& Ho, int& Wo) {
    Ho = (H + 2 * pad - Kh) / stride + 1;
    Wo = (W + 2 * pad - Kw) / stride + 1;
    out.assign(static_cast<std::size_t>(Cout) * Ho * Wo, 0.0f);
#pragma omp parallel for schedule(static) if (static_cast<long long>(Cout) * Ho * Wo * Cin * Kh * Kw > 1 << 20)
    for (int co = 0; co < Cout; ++co) {
        const float b = bias ? bias[co] : 0.0f;
        for (int oh = 0; oh < Ho; ++oh) {
            for (int ow = 0; ow < Wo; ++ow) {
                float acc = b;
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int kh = 0; kh < Kh; ++kh) {
                        const int ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < Kw; ++kw) {
                            const int iw = ow * stride - pad + kw;
                            if (iw < 0 || iw >= W) continue;
                            acc += x[(static_cast<std::size_t>(ci) * H + ih) * W + iw] *
                                   w[((static_cast<std::size_t>(co) * Cin + ci) * Kh + kh) * Kw + kw];
                        }
                    }
                }
                out[(static_cast<std::size_t>(co) * Ho + oh) * Wo + ow] = acc;
            }
        }
    }
}

// LayerNorm over the channel dimension of an HWC tensor.
void layernorm2d(std::vector<float>& x, int C, int H, int W, const float* w, const float* b,
                 float eps) {
    for (int h = 0; h < H; ++h) {
        for (int ww = 0; ww < W; ++ww) {
            float* px = x.data() + (static_cast<std::size_t>(h) * W + ww) * C;
            float mean = 0.0f;
            for (int c = 0; c < C; ++c) mean += px[c];
            mean /= C;
            float var = 0.0f;
            for (int c = 0; c < C; ++c) {
                const float d = px[c] - mean;
                var += d * d;
            }
            var /= C;
            const float inv = 1.0f / std::sqrt(var + eps);
            for (int c = 0; c < C; ++c) {
                float v = (px[c] - mean) * inv;
                if (w) v *= w[c];
                if (b) v += b[c];
                px[c] = v;
            }
        }
    }
}

// Fused LayerNorm over rows of a [N,C] row-major matrix.
void layernorm_rows(std::vector<float>& x, int N, int C, const float* w, const float* b, float eps) {
    for (int n = 0; n < N; ++n) {
        float* px = x.data() + static_cast<std::size_t>(n) * C;
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) mean += px[c];
        mean /= C;
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            const float d = px[c] - mean;
            var += d * d;
        }
        var /= C;
        const float inv = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < C; ++c) {
            float v = (px[c] - mean) * inv;
            if (w) v *= w[c];
            if (b) v += b[c];
            px[c] = v;
        }
    }
}

inline float cubic_weight(float t) {
    const float a = -0.75f;
    t = std::fabs(t);
    if (t <= 1.0f) return ((a + 2) * t - (a + 3)) * t * t + 1.0f;
    if (t < 2.0f) return (((t - 5) * t + 8) * t - 4) * a;
    return 0.0f;
}

// Bicubic resize of an HWC image (align_corners=false).  `antialias` is
// approximated by clamping the filter support when downsampling.
std::vector<float> bicubic_resize(const std::vector<float>& src, int Hi, int Wi, int C, int Ho,
                                  int Wo) {
    std::vector<float> out(static_cast<std::size_t>(Ho) * Wo * C);
    const float sh = static_cast<float>(Hi) / Ho;
    const float sw = static_cast<float>(Wi) / Wo;
    const float fh = sh > 1.0f ? sh : 1.0f;  // filter scale for antialiasing
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

// window_partition for HWC: returns windows [B*num, ws, ws, C] and padded Hp,Wp.
std::vector<float> window_partition(const std::vector<float>& x, int H, int W, int C, int ws,
                                    int& Hp, int& Wp, int& num_windows) {
    const int pad_h = (ws - H % ws) % ws;
    const int pad_w = (ws - W % ws) % ws;
    Hp = H + pad_h;
    Wp = W + pad_w;
    const int nh = Hp / ws, nw = Wp / ws;
    num_windows = nh * nw;
    std::vector<float> windows(static_cast<std::size_t>(num_windows) * ws * ws * C, 0.0f);
    for (int wi = 0; wi < nh; ++wi) {
        for (int wj = 0; wj < nw; ++wj) {
            float* dst = windows.data() +
                         (static_cast<std::size_t>(wi * nw + wj) * ws * ws) * C;
            for (int i = 0; i < ws; ++i) {
                const int ih = wi * ws + i;
                for (int j = 0; j < ws; ++j) {
                    const int iw = wj * ws + j;
                    if (ih >= H || iw >= W) continue;
                    const float* ps = x.data() + (static_cast<std::size_t>(ih) * W + iw) * C;
                    std::memcpy(dst + (static_cast<std::size_t>(i) * ws + j) * C, ps,
                                C * sizeof(float));
                }
            }
        }
    }
    return windows;
}

std::vector<float> window_unpartition(const std::vector<float>& windows, int ws, int Hp, int Wp,
                                      int H, int W, int C) {
    const int nh = Hp / ws, nw = Wp / ws;
    std::vector<float> x(static_cast<std::size_t>(H) * W * C, 0.0f);
    for (int wi = 0; wi < nh; ++wi) {
        for (int wj = 0; wj < nw; ++wj) {
            const float* src =
                windows.data() + (static_cast<std::size_t>(wi * nw + wj) * ws * ws) * C;
            for (int i = 0; i < ws; ++i) {
                const int ih = wi * ws + i;
                if (ih >= H) continue;
                for (int j = 0; j < ws; ++j) {
                    const int iw = wj * ws + j;
                    if (iw >= W) continue;
                    float* pd = x.data() + (static_cast<std::size_t>(ih) * W + iw) * C;
                    std::memcpy(pd, src + (static_cast<std::size_t>(i) * ws + j) * C,
                                C * sizeof(float));
                }
            }
        }
    }
    return x;
}

// get_rel_pos for the q_size == k_size case used by SAM.
void build_rel_pos(const std::vector<float>& rel, int q, int k, int hd, std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(q) * k * hd, 0.0f);
    const float scale_q = std::max(static_cast<float>(k) / q, 1.0f);
    const float scale_k = std::max(static_cast<float>(q) / k, 1.0f);
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < k; ++j) {
            int idx = static_cast<int>(std::lround((i * scale_q - j * scale_k) + (k - 1) * scale_k));
            idx = std::clamp(idx, 0, static_cast<int>(rel.size()) / hd - 1);
            const float* ps = rel.data() + static_cast<std::size_t>(idx) * hd;
            std::memcpy(out.data() + (static_cast<std::size_t>(i) * k + j) * hd, ps,
                        hd * sizeof(float));
        }
    }
}

// Multi-head attention over a batch of HxW token grids with decomposed rel pos.
std::vector<float> sam_attention(const std::vector<float>& x, int B, int H, int W, int C,
                                 int heads, const SAMBlockWeights& bw) {
    const int hd = C / heads;
    std::vector<float> qkv(static_cast<std::size_t>(B) * H * W * 3 * C);
    bw.qkv_w.matmul(x.data(), qkv.data(), B * H * W);

    std::vector<float> q(static_cast<std::size_t>(B) * H * W * C);
    std::vector<float> k(static_cast<std::size_t>(B) * H * W * C);
    std::vector<float> v(static_cast<std::size_t>(B) * H * W * C);
    const int N = B * H * W;
    for (int n = 0; n < N; ++n) {
        const float* src = qkv.data() + static_cast<std::size_t>(n) * 3 * C;
        for (int c = 0; c < C; ++c) {
            q[static_cast<std::size_t>(n) * C + c] = src[c] + bw.qkv_b[c];
            k[static_cast<std::size_t>(n) * C + c] = src[C + c] + bw.qkv_b[C + c];
            v[static_cast<std::size_t>(n) * C + c] = src[2 * C + c] + bw.qkv_b[2 * C + c];
        }
    }

    std::vector<float> rh, rw;
    build_rel_pos(bw.rel_pos_h, H, H, hd, rh);
    build_rel_pos(bw.rel_pos_w, W, W, hd, rw);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> out(static_cast<std::size_t>(B) * H * W * C, 0.0f);

#pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int qi = 0; qi < H * W; ++qi) {
                const int qh = qi / W, qw = qi % W;
                const float* qp = q.data() + ((static_cast<std::size_t>(b) * H * W + qi) * C) + h * hd;
                float m = -std::numeric_limits<float>::infinity();
                std::vector<float> scores(static_cast<std::size_t>(H * W));
                for (int ki = 0; ki < H * W; ++ki) {
                    const int kh = ki / W, kw = ki % W;
                    const float* kp =
                        k.data() + ((static_cast<std::size_t>(b) * H * W + ki) * C) + h * hd;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; ++d) dot += qp[d] * kp[d];
                    float s = dot * scale;
                    const float* rhp = rh.data() + (static_cast<std::size_t>(qh) * H + kh) * hd;
                    const float* rwp = rw.data() + (static_cast<std::size_t>(qw) * W + kw) * hd;
                    // Decomposed relative position: bias = q . (Rh + Rw), i.e. a
                    // scalar per (query,key) pair, NOT a sum of raw components.
                    for (int d = 0; d < hd; ++d) s += qp[d] * (rhp[d] + rwp[d]);
                    scores[static_cast<std::size_t>(ki)] = s;
                    m = std::max(m, s);
                }
                float l = 0.0f;
                std::vector<float> acc(hd, 0.0f);
                for (int ki = 0; ki < H * W; ++ki) {
                    const float e = std::exp(scores[static_cast<std::size_t>(ki)] - m);
                    l += e;
                    const float* vp =
                        v.data() + ((static_cast<std::size_t>(b) * H * W + ki) * C) + h * hd;
                    for (int d = 0; d < hd; ++d) acc[d] += e * vp[d];
                }
                float* op = out.data() + ((static_cast<std::size_t>(b) * H * W + qi) * C) + h * hd;
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                for (int d = 0; d < hd; ++d) op[d] = acc[d] * inv;
            }
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// DeepEncoder
// ---------------------------------------------------------------------------
DeepEncoder::DeepEncoder(ModelConfig cfg, VisionWeights w, DecoderWeights projector_weights)
    : cfg_(std::move(cfg)), vw_(std::move(w)), pw_(std::move(projector_weights)) {}

std::vector<float> DeepEncoder::sam_forward(const float* image_chw, int h, int w,
                                            std::vector<Stage>* stages) const {
    auto record = [&](const std::string& name, const std::vector<float>& data) {
        if (stages) stages->emplace_back(name, data);
    };
    const int dim = cfg_.sam_embed_dim;  // 768

    // patch embed
    int Ho = 0, Wo = 0;
    std::vector<float> feat;
    conv2d(image_chw, 3, h, w, vw_.sam.patch_w.data(), vw_.sam.patch_b.data(), dim,
           cfg_.patch_size, cfg_.patch_size, cfg_.patch_size, 0, feat, Ho, Wo);
    // [dim, Ho, Wo] -> HWC [Ho*Wo, dim]
    std::vector<float> x(static_cast<std::size_t>(Ho) * Wo * dim);
    for (int hh = 0; hh < Ho; ++hh)
        for (int ww = 0; ww < Wo; ++ww)
            for (int c = 0; c < dim; ++c)
                x[(static_cast<std::size_t>(hh) * Wo + ww) * dim + c] =
                    feat[(static_cast<std::size_t>(c) * Ho + hh) * Wo + ww];
    record("sam_patch", x);

    // absolute positional embedding (already at the target 64x64 grid)
    for (int i = 0; i < Ho * Wo; ++i)
        for (int c = 0; c < dim; ++c)
            x[static_cast<std::size_t>(i) * dim + c] += vw_.sam.pos_embed[static_cast<std::size_t>(i) * dim + c];
    record("sam_pos", x);

    const int heads = cfg_.sam_heads;
    for (int bi = 0; bi < cfg_.sam_depth; ++bi) {
        const SAMBlockWeights& bw = vw_.sam.blocks[bi];
        // global_attn_indexes = {2,5,8,11}
        const bool is_global = (bi >= 2 && (bi - 2) % 3 == 0);

        std::vector<float> residual = x;
        std::vector<float> normed = x;
        layernorm_rows(normed, Ho * Wo, dim, bw.norm1_w.data(), bw.norm1_b.data(), 1e-6f);

        std::vector<float> attn_out;
        if (is_global) {
            normed.resize(static_cast<std::size_t>(Ho) * Wo * dim);
            attn_out = sam_attention(normed, 1, Ho, Wo, dim, heads, bw);
        } else {
            int Hp = 0, Wp = 0, nwin = 0;
            std::vector<float> windows =
                window_partition(normed, Ho, Wo, dim, cfg_.sam_window_size, Hp, Wp, nwin);
            std::vector<float> wo =
                sam_attention(windows, nwin, cfg_.sam_window_size, cfg_.sam_window_size, dim, heads, bw);
            attn_out = window_unpartition(wo, cfg_.sam_window_size, Hp, Wp, Ho, Wo, dim);
        }
        // proj + residual
        std::vector<float> proj(static_cast<std::size_t>(Ho) * Wo * dim);
        bw.proj_w.matmul(attn_out.data(), proj.data(), Ho * Wo);
        for (int i = 0; i < Ho * Wo; ++i) {
            for (int c = 0; c < dim; ++c) {
                const std::size_t idx = static_cast<std::size_t>(i) * dim + c;
                proj[idx] += bw.proj_b[c];
                x[idx] = residual[idx] + proj[idx];
            }
        }
        // mlp
        std::vector<float> normed2 = x;
        layernorm_rows(normed2, Ho * Wo, dim, bw.norm2_w.data(), bw.norm2_b.data(), 1e-6f);
        std::vector<float> mlp1(static_cast<std::size_t>(Ho) * Wo * 4 * dim);
        bw.mlp_lin1_w.matmul(normed2.data(), mlp1.data(), Ho * Wo);
        for (std::size_t i = 0; i < mlp1.size(); ++i) {
            const int c = static_cast<int>(i % (4 * dim));
            mlp1[i] += bw.mlp_lin1_b[c];
        }
        ops::gelu(mlp1.data(), static_cast<i64>(mlp1.size()));
        std::vector<float> mlp2(static_cast<std::size_t>(Ho) * Wo * dim);
        bw.mlp_lin2_w.matmul(mlp1.data(), mlp2.data(), Ho * Wo);
        for (int i = 0; i < Ho * Wo; ++i)
            for (int c = 0; c < dim; ++c)
                x[static_cast<std::size_t>(i) * dim + c] = x[static_cast<std::size_t>(i) * dim + c] +
                                                           mlp2[static_cast<std::size_t>(i) * dim + c] +
                                                           bw.mlp_lin2_b[c];
        record("sam_block" + std::to_string(bi), x);
    }

    // neck: x [Ho*Wo, dim] -> CHW
    std::vector<float> chw(static_cast<std::size_t>(dim) * Ho * Wo);
    for (int c = 0; c < dim; ++c)
        for (int i = 0; i < Ho * Wo; ++i)
            chw[static_cast<std::size_t>(c) * Ho * Wo + i] = x[static_cast<std::size_t>(i) * dim + c];

    int nH = 0, nW = 0;
    std::vector<float> neck;
    conv2d(chw.data(), dim, Ho, Wo, vw_.sam.neck0_w.data(), nullptr, 256, 1, 1, 1, 0, neck, nH, nW);
    // neck output is CHW -> HWC for layernorm2d
    std::vector<float> hwc(static_cast<std::size_t>(nH) * nW * 256);
    for (int c = 0; c < 256; ++c)
        for (int i = 0; i < nH * nW; ++i)
            hwc[static_cast<std::size_t>(i) * 256 + c] = neck[static_cast<std::size_t>(c) * nH * nW + i];
    layernorm2d(hwc, 256, nH, nW, vw_.sam.neck1_w.data(), vw_.sam.neck1_b.data(), 1e-6f);
    std::vector<float> neck2chw(static_cast<std::size_t>(256) * nH * nW);
    for (int c = 0; c < 256; ++c)
        for (int i = 0; i < nH * nW; ++i)
            neck2chw[static_cast<std::size_t>(c) * nH * nW + i] = hwc[static_cast<std::size_t>(i) * 256 + c];

    std::vector<float> neck3;
    conv2d(neck2chw.data(), 256, nH, nW, vw_.sam.neck2_w.data(), nullptr, 256, 3, 3, 1, 1, neck3, nH, nW);
    std::vector<float> hwc3(static_cast<std::size_t>(nH) * nW * 256);
    for (int c = 0; c < 256; ++c)
        for (int i = 0; i < nH * nW; ++i)
            hwc3[static_cast<std::size_t>(i) * 256 + c] = neck3[static_cast<std::size_t>(c) * nH * nW + i];
    layernorm2d(hwc3, 256, nH, nW, vw_.sam.neck3_w.data(), vw_.sam.neck3_b.data(), 1e-6f);
    std::vector<float> neck4chw(static_cast<std::size_t>(256) * nH * nW);
    for (int c = 0; c < 256; ++c)
        for (int i = 0; i < nH * nW; ++i)
            neck4chw[static_cast<std::size_t>(c) * nH * nW + i] = hwc3[static_cast<std::size_t>(i) * 256 + c];

    record("sam_neck", hwc3);

    int h2 = 0, w2 = 0, h3 = 0, w3 = 0;
    std::vector<float> x2, x3;
    conv2d(neck4chw.data(), 256, nH, nW, vw_.sam.net2_w.data(), nullptr, 512, 3, 3, 2, 1, x2, h2, w2);
    record("sam_net2", x2);
    conv2d(x2.data(), 512, h2, w2, vw_.sam.net3_w.data(), nullptr, 1024, 3, 3, 2, 1, x3, h3, w3);
    record("sam_net3", x3);

    // [1024, h3, w3] -> [h3*w3, 1024]
    std::vector<float> tokens(static_cast<std::size_t>(h3) * w3 * 1024);
    for (int c = 0; c < 1024; ++c)
        for (int i = 0; i < h3 * w3; ++i)
            tokens[static_cast<std::size_t>(i) * 1024 + c] = x3[static_cast<std::size_t>(c) * h3 * w3 + i];

    // stash the grid size for clip_forward via cfg_.image_size convention
    return tokens;
}

std::vector<float> DeepEncoder::clip_forward(const std::vector<float>& sam_tokens, int tokens,
                                             std::vector<Stage>* stages) const {
    auto record = [&](const std::string& name, const std::vector<float>& data) {
        if (stages) stages->emplace_back(name, data);
    };
    const int hd = cfg_.clip_hidden_size;  // 1024
    const int grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(tokens))));
    UOCR_CHECK(grid * grid == tokens, "SAM token count is not a perfect square");

    // patch_embeds = sam_tokens; prepend class token
    const int N = tokens + 1;
    std::vector<float> x(static_cast<std::size_t>(N) * hd);
    std::memcpy(x.data(), vw_.clip.class_embedding.data(), hd * sizeof(float));
    std::memcpy(x.data() + hd, sam_tokens.data(), static_cast<std::size_t>(tokens) * hd * sizeof(float));

    // position embedding: interpolate 16x16 -> grid x grid (plus class token)
    const int src_grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(vw_.clip.pos_embed.size() / hd - 1))));
    std::vector<float> pos_new;
    if (src_grid != grid) {
        std::vector<float> cls(vw_.clip.pos_embed.begin(), vw_.clip.pos_embed.begin() + hd);
        std::vector<float> gridpos(vw_.clip.pos_embed.begin() + hd, vw_.clip.pos_embed.end());
        std::vector<float> resized = bicubic_resize(gridpos, src_grid, src_grid, hd, grid, grid);
        pos_new.resize(static_cast<std::size_t>(N) * hd);
        std::memcpy(pos_new.data(), cls.data(), hd * sizeof(float));
        std::memcpy(pos_new.data() + hd, resized.data(), resized.size() * sizeof(float));
    } else {
        pos_new = vw_.clip.pos_embed;
    }
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += pos_new[i];
    record("clip_embeds", x);
    layernorm_rows(x, N, hd, vw_.clip.pre_ln_w.data(), vw_.clip.pre_ln_b.data(), 1e-5f);
    record("clip_preln", x);

    const int heads = cfg_.clip_heads;
    const int chd = hd / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(chd));

    int clip_li = 0;
    for (const CLIPLayerWeights& L : vw_.clip.layers) {
        // attention
        std::vector<float> residual = x;
        std::vector<float> normed = x;
        layernorm_rows(normed, N, hd, L.ln1_w.data(), L.ln1_b.data(), 1e-5f);
        std::vector<float> qkv(static_cast<std::size_t>(N) * 3 * hd);
        L.qkv_w.matmul(normed.data(), qkv.data(), N);
        for (int n = 0; n < N; ++n) {
            float* qr = qkv.data() + static_cast<std::size_t>(n) * 3 * hd;
            for (int c = 0; c < 3 * hd; ++c) qr[c] += L.qkv_b[c];
        }
        std::vector<float> out(static_cast<std::size_t>(N) * hd, 0.0f);
#pragma omp parallel for schedule(static)
        for (int h = 0; h < heads; ++h) {
            for (int qi = 0; qi < N; ++qi) {
                const float* qp = qkv.data() + static_cast<std::size_t>(qi) * 3 * hd + h * chd;
                float m = -std::numeric_limits<float>::infinity();
                std::vector<float> scores(static_cast<std::size_t>(N));
                for (int ki = 0; ki < N; ++ki) {
                    const float* kp = qkv.data() + static_cast<std::size_t>(ki) * 3 * hd + hd + h * chd;
                    float dot = 0.0f;
                    for (int d = 0; d < chd; ++d) dot += qp[d] * kp[d];
                    scores[static_cast<std::size_t>(ki)] = dot * scale;
                    m = std::max(m, scores[static_cast<std::size_t>(ki)]);
                }
                float l = 0.0f;
                std::vector<float> acc(chd, 0.0f);
                for (int ki = 0; ki < N; ++ki) {
                    const float e = std::exp(scores[static_cast<std::size_t>(ki)] - m);
                    l += e;
                    const float* vp = qkv.data() + static_cast<std::size_t>(ki) * 3 * hd + 2 * hd + h * chd;
                    for (int d = 0; d < chd; ++d) acc[d] += e * vp[d];
                }
                float* op = out.data() + static_cast<std::size_t>(qi) * hd + h * chd;
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                for (int d = 0; d < chd; ++d) op[d] = acc[d] * inv;
            }
        }
        L.out_w.matmul(out.data(), x.data(), N);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < hd; ++c)
                x[static_cast<std::size_t>(n) * hd + c] =
                    residual[static_cast<std::size_t>(n) * hd + c] + x[static_cast<std::size_t>(n) * hd + c] +
                    L.out_b[c];

        // mlp
        residual = x;
        layernorm_rows(x, N, hd, L.ln2_w.data(), L.ln2_b.data(), 1e-5f);
        std::vector<float> fc1(static_cast<std::size_t>(N) * cfg_.clip_ffn_size);
        L.fc1_w.matmul(x.data(), fc1.data(), N);
        for (std::size_t i = 0; i < fc1.size(); ++i)
            fc1[i] += L.fc1_b[i % cfg_.clip_ffn_size];
        ops::quick_gelu(fc1.data(), static_cast<i64>(fc1.size()));
        std::vector<float> fc2(static_cast<std::size_t>(N) * hd);
        L.fc2_w.matmul(fc1.data(), fc2.data(), N);
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < hd; ++c)
                x[static_cast<std::size_t>(n) * hd + c] =
                    residual[static_cast<std::size_t>(n) * hd + c] + fc2[static_cast<std::size_t>(n) * hd + c] +
                    L.fc2_b[c];
        record("clip_layer" + std::to_string(clip_li++), x);
    }

    // drop the class token
    return std::vector<float>(x.begin() + hd, x.end());
}

void DeepEncoder::encode(const float* image_chw, int height, int width, Tensor& out,
                         std::vector<float>* sam_debug, std::vector<float>* clip_debug) const {
    encode_stages(image_chw, height, width, out, nullptr, sam_debug, clip_debug);
}

void DeepEncoder::encode_stages(const float* image_chw, int height, int width, Tensor& out,
                                std::vector<Stage>* stages, std::vector<float>* sam_debug,
                                std::vector<float>* clip_debug) const {
    std::vector<float> sam = sam_forward(image_chw, height, width, stages);
    if (sam_debug) *sam_debug = sam;
    const int sam_dim = 1024;
    const int tokens = static_cast<int>(sam.size() / sam_dim);

    std::vector<float> clip = clip_forward(sam, tokens, stages);
    if (clip_debug) *clip_debug = clip;
    UOCR_CHECK(static_cast<int>(clip.size()) == tokens * cfg_.clip_hidden_size, "clip size mismatch");

    // concat(clip, sam) -> projector
    const int proj_in = cfg_.clip_hidden_size + sam_dim;
    UOCR_CHECK(proj_in == cfg_.projector_input_dim, "projector input dim mismatch");
    std::vector<float> cat(static_cast<std::size_t>(tokens) * proj_in);
    for (int t = 0; t < tokens; ++t) {
        std::memcpy(cat.data() + static_cast<std::size_t>(t) * proj_in,
                    clip.data() + static_cast<std::size_t>(t) * cfg_.clip_hidden_size,
                    cfg_.clip_hidden_size * sizeof(float));
        std::memcpy(cat.data() + static_cast<std::size_t>(t) * proj_in + cfg_.clip_hidden_size,
                    sam.data() + static_cast<std::size_t>(t) * sam_dim, sam_dim * sizeof(float));
    }
    const int hidden = cfg_.projector_n_embed;
    std::vector<float> projected(static_cast<std::size_t>(tokens) * hidden);
    pw_.projector.forward(cat.data(), projected.data(), tokens);

    // append image_newline per row, then view_seperator
    const int grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(tokens))));
    const int rows = grid;
    const int cols = grid + 1;
    out = Tensor({rows * cols + 1, hidden});
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < grid; ++c) {
            std::memcpy(out.data() + static_cast<std::size_t>(r * cols + c) * hidden,
                        projected.data() + static_cast<std::size_t>(r * grid + c) * hidden,
                        hidden * sizeof(float));
        }
        std::memcpy(out.data() + static_cast<std::size_t>(r * cols + grid) * hidden,
                    pw_.image_newline.data(), hidden * sizeof(float));
    }
    std::memcpy(out.data() + static_cast<std::size_t>(rows * cols) * hidden,
                pw_.view_seperator.data(), hidden * sizeof(float));
    num_tokens_ = rows * cols + 1;
}

}  // namespace uocr
