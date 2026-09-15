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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["decoder", "vision", "all"], default="decoder")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
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


if __name__ == "__main__":
    main()
