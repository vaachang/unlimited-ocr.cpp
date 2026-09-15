#!/usr/bin/env python3
"""Export per-stage DeepEncoder activations for C++ alignment.

This mirrors the intermediate tensors recorded by
``DeepEncoder::encode_stages`` (see tools/compare_vision.cpp --dump-stages) so
that a mismatch can be localised to SAM patch/block/neck/net or CLIP layer.

Usage:
  python tools/reference/export_vision_stages.py --model models --out /tmp/dbg
"""

import argparse
import os

import numpy as np
import torch

from export_reference import deterministic_image


def scalar(name, arr):
    np.asarray(arr, dtype="<f4").tofile(os.path.join(OUT, name + ".bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = ap.parse_args()

    global OUT
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoModel

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, use_safetensors=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().cuda()
    m = model.model

    img = deterministic_image(args.image_size)
    img_t = torch.from_numpy(img)[None].cuda()
    img_t = ((img_t - 0.5) / 0.5).to(dtype)

    def hook_sam_patch(_mod, _inp, out):
        # PatchEmbed already returns B,H,W,C.
        scalar("sam_patch", out[0].reshape(-1, out.shape[-1]).float().cpu().numpy())

    def make_sam_attn(li):
        def hook(_mod, _inp, out):
            # out is [B*num_windows, ws*ws, C] for windowed blocks, [B, H*W, C] else.
            scalar(f"sam_attn{li}", out.reshape(-1, out.shape[-1]).float().cpu().numpy())
        return hook

    def hook_sam_pos(_mod, inp):
        x = inp[0]
        scalar("sam_pos", x[0].reshape(-1, x.shape[-1]).float().cpu().numpy())

    def hook_neck(_mod, _inp, out):
        scalar("sam_neck", out.permute(0, 2, 3, 1).reshape(-1, out.shape[1]).float().cpu().numpy())

    def hook_net(_mod, _inp, out):
        scalar("sam_net" + ("2" if out.shape[1] == 512 else "3"), out[0].float().cpu().numpy())

    def hook_clip_embeds(_mod, _inp, out):
        scalar("clip_embeds", out[0].float().cpu().numpy())

    def hook_clip_preln(_mod, _inp, out):
        scalar("clip_preln", out[0].float().cpu().numpy())

    def make_sam_block(li):
        def hook(_mod, _inp, out):
            scalar(f"sam_block{li}", out[0].reshape(-1, out.shape[-1]).float().cpu().numpy())
        return hook

    def make_clip_layer(li):
        def hook(_mod, _inp, out):
            scalar(f"clip_layer{li}", out[0].float().cpu().numpy())
        return hook

    handles = [
        m.sam_model.patch_embed.register_forward_hook(hook_sam_patch),
        m.sam_model.blocks[0].register_forward_pre_hook(hook_sam_pos),
        m.sam_model.neck.register_forward_hook(hook_neck),
        m.sam_model.net_2.register_forward_hook(hook_net),
        m.sam_model.net_3.register_forward_hook(hook_net),
        m.vision_model.embeddings.register_forward_hook(hook_clip_embeds),
        m.vision_model.pre_layrnorm.register_forward_hook(hook_clip_preln),
    ]
    for i, blk in enumerate(m.sam_model.blocks):
        handles.append(blk.register_forward_hook(make_sam_block(i)))
        handles.append(blk.attn.register_forward_hook(make_sam_attn(i)))
    for i, layer in enumerate(m.vision_model.transformer.layers):
        handles.append(layer.register_forward_hook(make_clip_layer(i)))

    with torch.no_grad():
        f1 = m.sam_model(img_t)
        m.vision_model(img_t, f1)

    for h in handles:
        h.remove()
    print(f"wrote stages to {OUT}")


if __name__ == "__main__":
    main()
