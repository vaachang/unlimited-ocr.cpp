#!/usr/bin/env python3
"""Activation-aware (GPTQ) INT4 quantization of the Unlimited-OCR MoE experts.

The engine's built-in INT4 path is group-wise round-to-nearest (RTN), which
ignores the activation distribution and leaves ~10% weight error / divergent
end-to-end output (see ALIGNMENT.md 5.2).  This tool instead:

  1. runs a small calibration set (text prompts + optional images) through the
     BF16 reference model while hooking the routed experts, accumulating the
     per-layer input second-moment (Hessian) H = X^T X for the gate/up input
     (same for both) and for the down input;
  2. runs block-wise GPTQ with the engine's group-wise asymmetric affine
     quantization, using that Hessian to compensate quantization error.

Because MoE routing makes per-expert calibration data very scarce, the Hessian
is *pooled over all experts of a layer* (all experts see the same input space),
which is far better conditioned, and GPTQ is batched over the experts.

Output is a safetensors file with exactly the layout `DecoderWeights` loads:
  <weight_name>.packed   U8  [rows, ceil(cols/2)]
  <weight_name>.scales   F32 [rows, n_groups]
  <weight_name>.zeros    F32 [rows, n_groups]
where <weight_name> is e.g. model.layers.2.mlp.experts.7.gate_proj.weight.

Point the engine at the result with `--int4-quant <file>` (or copy it to
models/int4_gptq.safetensors to auto-load with `--int4`).

No third-party quantization library is required: GPTQ is plain PyTorch.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# calibration input construction
# ---------------------------------------------------------------------------

def reference_dynamic_preprocess(model):
    try:
        from importlib import import_module
        mod = import_module(type(model).__module__)
        return getattr(mod, "dynamic_preprocess", None)
    except Exception:
        return None


def image_case(args, model, image_path):
    """Build (input_ids, images_crop, images_ori, mask, crop_ratio) for one image."""
    from PIL import Image, ImageOps
    from tokenizers import Tokenizer

    base_size, image_size = 1024, 640
    crop_mode = args.crop_mode
    if not os.path.exists(image_path):
        return None

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    image_token_id, bos_id = 128815, 0
    prompt = args.prompt.strip()
    text_splits = prompt.split("<image>")
    assert len(text_splits) == 2, "calibration prompt must contain exactly one <image>"

    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    pad_color = (127, 127, 127)

    def transform(x):
        a = np.asarray(x).astype(np.float32) / 255.0
        return (a.transpose(2, 0, 1) - mean) / std

    crop_ratio = [1, 1]
    images_crop_list = []
    if crop_mode:
        global_view = ImageOps.pad(im, (base_size, base_size), color=pad_color)
        images_ori = transform(global_view)
        if w > image_size or h > image_size:
            dp = reference_dynamic_preprocess(model)
            if dp is not None:
                images_crop_raw, crop_ratio = dp(im)
                images_crop_list = [transform(c) for c in images_crop_raw]
    else:
        sq = im.resize((image_size, image_size))
        global_view = ImageOps.pad(sq, (image_size, image_size), color=pad_color)
        images_ori = transform(global_view)
    if images_crop_list:
        images_crop = np.stack(images_crop_list).astype(np.float32)
    else:
        images_crop = np.zeros((1, 3, base_size, base_size), dtype=np.float32)

    width_crop_num, height_crop_num = int(crop_ratio[0]), int(crop_ratio[1])
    patch_size, downsample_ratio = 16, 4
    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

    ids, mask = [], []
    for text_sep in text_splits[:-1]:
        t = tok.encode(text_sep, add_special_tokens=False).ids
        ids += t
        mask += [0] * len(t)
        if crop_mode:
            img = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
            img += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                img += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                    num_queries * height_crop_num)
        else:
            img = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            img += [image_token_id]
        ids += img
        mask += [1] * len(img)
    t = tok.encode(text_splits[-1], add_special_tokens=False).ids
    ids += t
    mask += [0] * len(t)
    ids = [bos_id] + ids
    mask = [0] + mask
    return ids, images_crop, images_ori, mask, crop_ratio


def text_case(args, seed, length):
    # Real-ish token ids from the tokenizer over a text document when available.
    if args.calib_text and os.path.exists(args.calib_text):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
            txt = open(args.calib_text, "r", encoding="utf-8", errors="ignore").read()
            ids = tok.encode(txt, add_special_tokens=False).ids
            if len(ids) >= length:
                start = (seed * length) % max(1, len(ids) - length)
                return [0] + ids[start:start + length - 1]
        except Exception:
            pass
    return [0] + [int(x) for x in (np.arange(length - 1, dtype=np.int64) * 7919 % 120000) + 3]


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------

def rtn_batch(Ws, group):
    """Round-to-nearest baseline for [E, R, C] weights."""
    E, R, C = Ws.shape
    ng = (C + group - 1) // group
    codes = torch.zeros(E, R, C, dtype=torch.int64)
    scales = torch.zeros(E, R, ng)
    zeros = torch.zeros(E, R, ng)
    for gi, c0 in enumerate(range(0, C, group)):
        c1 = min(c0 + group, C)
        wg = Ws[:, :, c0:c1]
        mn = wg.amin(dim=2)
        mx = wg.amax(dim=2)
        scale = torch.clamp((mx - mn) / 15.0, min=1e-8)
        zero = torch.clamp(torch.round(-mn / scale), 0, 15)
        codes[:, :, c0:c1] = torch.clamp(torch.round(wg / scale[:, :, None] + zero[:, :, None]),
                                         0, 15)
        scales[:, :, gi] = scale
        zeros[:, :, gi] = zero
    return codes, scales, zeros


def gptq_batch(Ws, Hinv, group, blocksize):
    """Block-wise GPTQ over a batch of experts sharing Hessian inverse."""
    E, R, C = Ws.shape
    ng = (C + group - 1) // group
    W = Ws.clone().float()
    codes = torch.zeros(E, R, C, dtype=torch.int64)
    scales = torch.zeros(E, R, ng)
    zeros = torch.zeros(E, R, ng)
    for gi, g0 in enumerate(range(0, C, group)):
        g1 = min(g0 + group, C)
        wg = W[:, :, g0:g1]
        mn = wg.amin(dim=2)
        mx = wg.amax(dim=2)
        scale = torch.clamp((mx - mn) / 15.0, min=1e-8)
        zero = torch.clamp(torch.round(-mn / scale), 0, 15)
        scales[:, :, gi] = scale
        zeros[:, :, gi] = zero
        for j in range(g0, g1, blocksize):
            j1 = min(j + blocksize, g1)
            W1 = W[:, :, j:j1].clone()
            Err = torch.zeros(E, R, j1 - j)
            for c in range(j, j1):
                w = W1[:, :, c - j].clone()
                d = Hinv[c, c]
                q = torch.clamp(torch.round(w / scale + zero), 0, 15)
                dq = (q - zero) * scale
                W1[:, :, c - j] = dq
                codes[:, :, c] = q.to(torch.int64)
                if d.abs() > 1e-12:
                    Err[:, :, c - j] = (w - dq) / d
            if j1 < C:
                W[:, :, j1:] -= torch.einsum("erb,bc->erc", Err, Hinv[j:j1, j1:])
    return codes, scales, zeros


def pack_codes(codes):
    c = codes.to(torch.int64).numpy().astype(np.uint8)
    even = c[:, 0::2]
    odd = c[:, 1::2]
    packed = even.copy()
    packed[:, : odd.shape[1]] |= (odd << 4)
    return packed


def h_metric(W, Wq, H):
    """sqrt(tr(M H M^T) / tr(W H W^T)) for M = W - Wq (activation-weighted)."""
    M = (W - Wq).float()
    num = (M * (M @ H)).sum()
    den = (W * (W @ H)).sum()
    return float(torch.sqrt(num / (den + 1e-30)))


def weight_rel(W, Wq):
    num = ((W - Wq) ** 2).sum().sqrt()
    den = (W ** 2).sum().sqrt()
    return float(num / (den + 1e-12))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--blocksize", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01, help="fraction of mean diag(H)")
    ap.add_argument("--text-seq", type=int, default=512)
    ap.add_argument("--text-cases", type=int, default=8)
    ap.add_argument("--calib-text", default="", help="UTF-8 text file tokenized for calibration")
    ap.add_argument("--images", default="")
    ap.add_argument("--prompt", default="<image>\nFree OCR.")
    ap.add_argument("--crop-mode", action="store_true", default=True)
    ap.add_argument("--no-crop-mode", dest="crop_mode", action="store_false")
    ap.add_argument("--layers", default="")
    ap.add_argument("--metrics", action="store_true",
                    help="compute (slow) weight/activation error diagnostics")
    args = ap.parse_args()

    from transformers import AutoModel

    print(f"loading {args.model} (bf16, cuda)...", flush=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().cuda()

    cfg = model.config
    cfg._ring_window = 128
    cfg.sliding_window = None
    first = cfg.first_k_dense_replace
    n_layers = len(model.model.layers)
    n_experts = cfg.n_routed_experts
    hidden = cfg.hidden_size
    inter = cfg.moe_intermediate_size

    layer_filter = None
    if args.layers:
        layer_filter = {int(x) for x in args.layers.split(",") if x != ""}

    # ---- accumulate pooled per-layer Hessians ----
    Hg = {}  # li -> [hidden, hidden] (gate/up input)
    Hd = {}  # li -> [inter, inter] (down input)
    handles = []

    def gate_hook(li):
        def hook(module, inp):
            x = inp[0].detach().to(torch.float32).cpu()
            H = x.T @ x
            Hg[li] = H if li not in Hg else Hg[li] + H
        return hook

    def down_hook(li):
        def hook(module, inp):
            x = inp[0].detach().to(torch.float32).cpu()
            H = x.T @ x
            Hd[li] = H if li not in Hd else Hd[li] + H
        return hook

    for li in range(first, n_layers):
        if layer_filter is not None and li not in layer_filter:
            continue
        for e in range(n_experts):
            handles.append(model.model.layers[li].mlp.experts[e].gate_proj
                           .register_forward_pre_hook(gate_hook(li)))
            handles.append(model.model.layers[li].mlp.experts[e].down_proj
                           .register_forward_pre_hook(down_hook(li)))

    cases = [("text", text_case(args, i, args.text_seq)) for i in range(args.text_cases)]
    for path in [p for p in args.images.split(",") if p]:
        built = image_case(args, model, path)
        if built is None:
            print(f"warning: image not found: {path}", flush=True)
        else:
            cases.append(("image", built))

    print(f"calibrating on {len(cases)} cases...", flush=True)
    for kind, data in cases:
        if kind == "text":
            input_ids = torch.tensor([data], dtype=torch.long, device=model.device)
            with torch.no_grad():
                model(input_ids=input_ids, images=None, use_cache=False, return_dict=True)
        else:
            ids, images_crop, images_ori, mask, crop_ratio = data
            input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
            images_seq_mask = torch.tensor([mask], dtype=torch.bool, device=model.device)
            spatial = torch.tensor([crop_ratio], dtype=torch.long, device=model.device)
            ic = torch.from_numpy(images_crop).to(model.device, torch.bfloat16)
            io = torch.from_numpy(images_ori)[None].to(model.device, torch.bfloat16)
            with torch.no_grad():
                model(input_ids=input_ids, images=[(ic, io)], images_seq_mask=images_seq_mask,
                      images_spatial_crop=spatial, use_cache=False, return_dict=True)
    for h in handles:
        h.remove()
    torch.cuda.empty_cache()

    # ---- batched GPTQ per (layer, projection) ----
    out_tensors = {}
    stats = []
    projs = (("gate", "gate_proj", Hg), ("up", "up_proj", Hg), ("down", "down_proj", Hd))
    for li in range(first, n_layers):
        if layer_filter is not None and li not in layer_filter:
            continue
        experts = model.model.layers[li].mlp.experts
        for proj, pname, hmap in projs:
            H = hmap.get(li)
            Ws = torch.stack([getattr(experts[e], pname).weight.detach().to(torch.float32).cpu()
                              for e in range(n_experts)])
            if H is None:
                codes, scales, zeros = rtn_batch(Ws, args.group)
            else:
                Hc = H.clone()
                Hc[range(Hc.shape[0]), range(Hc.shape[0])] += args.damp * torch.diag(Hc).mean()
                Hinv = torch.linalg.inv(Hc)
                codes, scales, zeros = gptq_batch(Ws, Hinv, args.group, args.blocksize)
            for e in range(n_experts):
                name = f"model.layers.{li}.mlp.experts.{e}.{pname}.weight"
                out_tensors[name + ".packed"] = torch.from_numpy(pack_codes(codes[e]))
                out_tensors[name + ".scales"] = scales[e].contiguous()
                out_tensors[name + ".zeros"] = zeros[e].contiguous()
            if H is not None and args.metrics:
                rc, rs, rz = rtn_batch(Ws, args.group)
                diff = (codes != rc).float().mean().item()
                print(f"    L{li}/{proj}: code-diff vs RTN = {diff:.4f}", flush=True)
                Wq = torch.zeros_like(Ws)
                for gi, c0 in enumerate(range(0, Ws.shape[2], args.group)):
                    c1 = min(c0 + args.group, Ws.shape[2])
                    Wq[:, :, c0:c1] = (codes[:, :, c0:c1].float() - zeros[:, :, gi][:, :, None]) \
                        * scales[:, :, gi][:, :, None]
                Wr = torch.zeros_like(Ws)
                for gi, c0 in enumerate(range(0, Ws.shape[2], args.group)):
                    c1 = min(c0 + args.group, Ws.shape[2])
                    Wr[:, :, c0:c1] = (rc[:, :, c0:c1].float() - rz[:, :, gi][:, :, None]) \
                        * rs[:, :, gi][:, :, None]
                # sample a few experts for the (expensive) Hessian metric
                sel = torch.randperm(n_experts)[:8]
                hm_r = np.mean([h_metric(Ws[i], Wr[i], H) for i in sel])
                hm_g = np.mean([h_metric(Ws[i], Wq[i], H) for i in sel])
                stats.append((li, proj, weight_rel(Ws, Wr), weight_rel(Ws, Wq), hm_r, hm_g))
        print(f"  layer {li} done", flush=True)

    meta = {"group": args.group, "scheme": "gptq-asym-pooled", "calib_cases": len(cases)}
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out_tensors, args.out, metadata={"format": "pt", "uocr_int4": json.dumps(meta)})

    if stats:
        wr = np.mean([s[2] for s in stats])
        wg = np.mean([s[3] for s in stats])
        hr = np.mean([s[4] for s in stats])
        hg = np.mean([s[5] for s in stats])
        print(f"per-(layer,proj) means over {len(stats)} groups:")
        print(f"  weight rel-L2 : RTN={wr:.4f} GPTQ={wg:.4f}")
        print(f"  act-weighted  : RTN={hr:.4f} GPTQ={hg:.4f}")
    print(f"wrote {args.out} ({len(out_tensors)} tensors, {len(cases)} calib cases)")


if __name__ == "__main__":
    main()
