#!/usr/bin/env python3
"""Export reference activations from baidu/Unlimited-OCR for C++ alignment.

Modes:
  decoder : text-only forward, exports inputs_embeds, per-layer hidden states,
            logits, R-SWA KV cache and subsequent greedy decode steps.
  vision  : exports the DeepEncoder visual embeddings for a deterministic image.

The tensors are written as .npy plus a manifest.json.  See
tools/compare_reference.cpp for the consumer.
"""

import argparse
import json
import os

import numpy as np
import torch


def save_np(out_dir, name, tensor):
    """Write both .npy (numpy) and raw little-endian f32 .bin; return manifest entry."""
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    np.save(os.path.join(out_dir, name + ".npy"), arr)
    arr.astype("<f4").tofile(os.path.join(out_dir, name + ".bin"))
    return {"file": name + ".bin", "shape": list(arr.shape), "dtype": "<f4"}


def get_layer_cache(cache, i):
    """Return (keys, values) for layer i across transformers versions."""
    if hasattr(cache, "layers"):
        layer = cache.layers[i]
        return layer.keys, layer.values
    return cache.key_cache[i], cache.value_cache[i]


def run_decoder(args, model):
    manifest = {"mode": "decoder", "tensors": {}}
    out = args.out
    os.makedirs(out, exist_ok=True)

    seq = args.seq
    torch.manual_seed(args.seed)
    # fixed token ids inside the vocab range, starting with bos
    ids = [0] + [int(x) for x in (np.arange(seq - 1, dtype=np.int64) * 7919 % 120000) + 3]
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)

    model.config._ring_window = args.window
    model.config.sliding_window = None

    # Capture MoE router decisions per layer.
    gate_out = {}
    first_dense = model.config.first_k_dense_replace
    n_layers = len(model.model.layers)
    handles = []

    def make_hook(li):
        def hook(module, inp, output):
            idx, weight, _aux = output
            gate_out[li] = (idx.detach().to(torch.float32).cpu(),
                            weight.detach().to(torch.float32).cpu())
        return hook

    for li in range(first_dense, n_layers):
        handles.append(model.model.layers[li].mlp.gate.register_forward_hook(make_hook(li)))

    def save_routers(prefix):
        for li in sorted(gate_out):
            idx, w = gate_out[li]
            manifest["tensors"][f"{prefix}_router_ids_{li}"] = save_np(
                args.out, f"{prefix}_router_ids_{li}", idx)
            manifest["tensors"][f"{prefix}_router_w_{li}"] = save_np(
                args.out, f"{prefix}_router_w_{li}", w)

    with torch.no_grad():
        out = model(input_ids=input_ids, images=None, use_cache=True,
                    output_hidden_states=True, return_dict=True)
    save_routers("prefill")

    manifest["window"] = args.window
    manifest["tokens"] = ids
    manifest["seq"] = seq

    manifest["tensors"]["prefill_input_ids"] = save_np(args.out, "prefill_input_ids", input_ids[0])
    hs = out.hidden_states
    manifest["num_hidden_states"] = len(hs)
    for i, h in enumerate(hs):
        manifest["tensors"][f"hidden_{i}"] = save_np(args.out, f"hidden_{i}", h[0])
    manifest["tensors"]["prefill_logits"] = save_np(args.out, "prefill_logits", out.logits[0, -1])

    cache = out.past_key_values
    manifest["cache_layers"] = len(model.model.layers)
    for i in range(len(model.model.layers)):
        k, v = get_layer_cache(cache, i)
        manifest["tensors"][f"prefill_k_{i}"] = save_np(args.out, f"prefill_k_{i}", k[0])
        manifest["tensors"][f"prefill_v_{i}"] = save_np(args.out, f"prefill_v_{i}", v[0])
    kv_len = get_layer_cache(cache, 0)[0].shape[-2]
    manifest["prefill_cache_len"] = int(kv_len)

    # greedy decode steps
    steps = []
    pos = seq
    for step in range(args.decode_steps):
        logits = out.logits[0, -1]
        tok = int(torch.argmax(logits).item())
        cur = torch.tensor([[tok]], dtype=torch.long, device=model.device)
        gate_out.clear()
        with torch.no_grad():
            out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                        output_hidden_states=True, return_dict=True)
        if step < 2:
            save_routers(f"decode{step}")
        cache = out.past_key_values
        k0 = get_layer_cache(cache, 0)[0]
        rec = {
            "token": tok,
            "position": pos,
            "cache_len": int(k0.shape[-2]),
        }
        manifest["tensors"][f"decode_token_{step}"] = save_np(args.out, f"decode_token_{step}",
                                                             torch.tensor([tok]))
        manifest["tensors"][f"decode_hidden_{step}"] = save_np(
            args.out, f"decode_hidden_{step}", out.hidden_states[-1][0, 0])
        manifest["tensors"][f"decode_logits_{step}"] = save_np(
            args.out, f"decode_logits_{step}", out.logits[0, -1])
        steps.append(rec)
        pos += 1
    manifest["decode_steps"] = steps

    # final cache snapshot (ring may have wrapped)
    for i in range(len(model.model.layers)):
        k, v = get_layer_cache(cache, i)
        manifest["tensors"][f"final_k_{i}"] = save_np(args.out, f"final_k_{i}", k[0])
        manifest["tensors"][f"final_v_{i}"] = save_np(args.out, f"final_v_{i}", v[0])
    manifest["final_cache_len"] = int(get_layer_cache(cache, 0)[0].shape[-2])

    for h in handles:
        h.remove()

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[decoder] wrote {out if False else args.out}/manifest.json, "
          f"prefill_cache_len={manifest['prefill_cache_len']}, "
          f"final_cache_len={manifest['final_cache_len']}")


def deterministic_image(size):
    # smooth 2-D gradient plus a checkerboard; CHW in [0,1] then normalized
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, size, dtype=np.float32)[None, :]
    base = (x + y) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    checker = (((xx // 32) + (yy // 32)) % 2).astype(np.float32)
    gray = 0.7 * base + 0.3 * checker
    img = np.stack([gray, checker, base], axis=0)  # [3,H,W]
    return img


def run_vision(args, model):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"mode": "vision", "tensors": {}}

    size = args.image_size
    img = deterministic_image(size)  # [3,H,W] in [0,1]
    manifest["tensors"]["image"] = save_np(out_dir, "image", torch.from_numpy(img))

    # normalize with mean=std=0.5
    img_t = torch.from_numpy(img)[None].to(model.device)
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.to(torch.bfloat16)

    m = model.model
    with torch.no_grad():
        f1 = m.sam_model(img_t)
        f2 = m.vision_model(img_t, f1)
        feat = torch.cat((f2[:, 1:], f1.flatten(2).permute(0, 2, 1)), dim=-1)
        feat = m.projector(feat)
        _, hw, n_dim = feat.shape
        h = w = int(hw ** 0.5)
        gf = feat.view(h, w, n_dim)
        gf = torch.cat([gf, m.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
        gf = gf.view(-1, n_dim)
        emb = torch.cat([gf, m.view_seperator[None, :]], dim=0)

    manifest["num_visual_tokens"] = int(emb.shape[0])
    manifest["hidden"] = int(emb.shape[1])
    manifest["tensors"]["sam_features"] = save_np(out_dir, "sam_features", f1[0])
    manifest["tensors"]["clip_features"] = save_np(out_dir, "clip_features", f2[0])
    manifest["tensors"]["visual_embeddings"] = save_np(out_dir, "visual_embeddings", emb)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[vision] visual tokens={emb.shape[0]} dim={emb.shape[1]} -> {out_dir}/manifest.json")


def det_rgb(w, h, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx / max(w - 1, 1)) * 180 + (yy / max(h - 1, 1)) * 60).astype(np.int32)
    noise = rng.integers(-20, 21, (h, w)).astype(np.int32)
    r = np.clip(base + noise, 0, 255).astype(np.uint8)
    g = np.clip(base * 2 // 3 + noise, 0, 255).astype(np.uint8)
    b = np.clip(255 - base // 2 + noise, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def run_ocr(args, model):
    """End-to-end image+prompt case (E3/E4)."""
    import math

    from PIL import Image, ImageOps
    from tokenizers import Tokenizer

    out = args.out
    os.makedirs(out, exist_ok=True)
    manifest = {"mode": "ocr", "tensors": {}}
    base_size, image_size = 1024, 640
    crop_mode = args.crop_mode

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    image_token_id, bos_id = 128815, 0
    prompt = args.prompt.strip()
    text_splits = prompt.split("<image>")
    assert len(text_splits) == 2, "run_ocr expects exactly one <image>"

    w, h = args.ocr_width, args.ocr_height
    arr = det_rgb(w, h, args.seed)
    arr.tofile(os.path.join(out, "ocr_image.bin"))
    manifest["image_hw"] = [h, w]
    manifest["prompt"] = prompt
    manifest["crop_mode"] = crop_mode

    im = Image.fromarray(arr)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
    pad_color = tuple(int(x * 255) for x in mean.reshape(-1))

    def transform(x):
        a = np.asarray(x).astype(np.float32) / 255.0
        return (a.transpose(2, 0, 1) - mean) / std

    crop_ratio = [1, 1]
    if crop_mode:
        global_view = ImageOps.pad(im, (base_size, base_size), color=pad_color)
        images_ori = transform(global_view)
    else:
        sq = im.resize((image_size, image_size))
        global_view = ImageOps.pad(sq, (image_size, image_size), color=pad_color)
        images_ori = transform(global_view)
    images_crop = np.zeros((1, 3, base_size, base_size), dtype=np.float32)

    # token layout
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

    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    images_seq_mask = torch.tensor([mask], dtype=torch.bool, device=model.device)
    spatial = torch.tensor([crop_ratio], dtype=torch.long, device=model.device)

    model.config._ring_window = args.window
    model.config.sliding_window = None

    images_ori_t = torch.from_numpy(images_ori)[None].to(model.device, torch.bfloat16)
    images_crop_t = torch.from_numpy(images_crop).to(model.device, torch.bfloat16)
    gate_out = {}
    first_dense = model.config.first_k_dense_replace
    handles = []

    def make_hook(li):
        def hook(module, inp, output):
            idx, weight, _aux = output
            gate_out[li] = (idx.detach().to(torch.float32).cpu(),
                            weight.detach().to(torch.float32).cpu())
        return hook

    for li in range(first_dense, len(model.model.layers)):
        handles.append(model.model.layers[li].mlp.gate.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        res = model(input_ids=input_ids, images=[(images_crop_t, images_ori_t)],
                    images_seq_mask=images_seq_mask, images_spatial_crop=spatial,
                    use_cache=True, output_hidden_states=True, return_dict=True)
    for hh in handles:
        hh.remove()

    hidden0 = res.hidden_states[0][0]
    mask_t = torch.tensor(mask, dtype=torch.bool, device=hidden0.device)
    visual = hidden0[mask_t]

    manifest["tensors"]["input_ids"] = save_np(out, "input_ids", torch.tensor(ids))
    manifest["tensors"]["images_seq_mask"] = save_np(out, "images_seq_mask",
                                                     torch.tensor(mask, dtype=torch.float32))
    manifest["tensors"]["image_global"] = save_np(out, "image_global", torch.from_numpy(images_ori))
    manifest["tensors"]["inputs_embeds_scattered"] = save_np(out, "inputs_embeds_scattered", hidden0)
    manifest["tensors"]["visual_scattered"] = save_np(out, "visual_scattered", visual)
    manifest["tensors"]["prefill_logits"] = save_np(out, "prefill_logits", res.logits[0, -1])
    manifest["num_visual_tokens"] = int(visual.shape[0])
    manifest["hidden"] = int(visual.shape[1])

    # greedy decode
    cache = res.past_key_values
    steps = []
    pos = len(ids)
    for step in range(args.decode_steps):
        logits = res.logits[0, -1]
        tokid = int(torch.argmax(logits).item())
        cur = torch.tensor([[tokid]], dtype=torch.long, device=model.device)
        with torch.no_grad():
            res = model(input_ids=cur, past_key_values=cache, use_cache=True,
                        output_hidden_states=True, return_dict=True)
        cache = res.past_key_values
        manifest["tensors"][f"decode_token_{step}"] = save_np(
            out, f"decode_token_{step}", torch.tensor([tokid]))
        manifest["tensors"][f"decode_logits_{step}"] = save_np(
            out, f"decode_logits_{step}", res.logits[0, -1])
        steps.append({"token": tokid, "position": pos})
        pos += 1
    manifest["decode_steps"] = steps

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ocr] ids={len(ids)} visual_tokens={visual.shape[0]} "
          f"prefill_tok={steps[0]['token'] if steps else -1} -> {out}/manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["decoder", "vision", "ocr", "all"], default="decoder")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="<image>\nFree OCR.")
    ap.add_argument("--crop-mode", action="store_true", default=True)
    ap.add_argument("--no-crop-mode", dest="crop_mode", action="store_false")
    ap.add_argument("--ocr-width", type=int, default=500)
    ap.add_argument("--ocr-height", type=int, default=400)
    args = ap.parse_args()

    from transformers import AutoModel

    print(f"loading {args.model} (bf16, cuda)...")
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().cuda()

    if args.mode in ("decoder", "all"):
        run_decoder(args, model)
    if args.mode in ("vision", "all"):
        run_vision(args, model)
    if args.mode in ("ocr", "all"):
        run_ocr(args, model)


if __name__ == "__main__":
    main()
