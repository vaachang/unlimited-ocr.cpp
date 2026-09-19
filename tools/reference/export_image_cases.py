#!/usr/bin/env python3
"""Export reference image pre-processing cases (E2) for C++ alignment.

For each deterministic RGB image it writes:
  - the raw image as HWC uint8 `.bin`
  - ImageOps.pad(global) normalized tensors for crop_mode True/False
  - dynamic_preprocess local crops normalized
  - `dynamic_preprocess` crop ratio

Usage:
    ./.venv/bin/python tools/reference/export_image_cases.py \
        --out /tmp/opencode/ref_image
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps


def det_image(w, h, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx / max(w - 1, 1)) * 180 + (yy / max(h - 1, 1)) * 60).astype(np.uint8)
    noise = rng.integers(-20, 21, (h, w)).astype(np.int32)
    r = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    g = np.clip(base.astype(np.int32) * 2 // 3 + noise, 0, 255).astype(np.uint8)
    b = np.clip(255 - base.astype(np.int32) // 2 + noise, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)  # HWC


class BasicImageTransform:
    def __init__(self, mean=0.5, std=0.5):
        if np.isscalar(mean):
            mean = [mean] * 3
        if np.isscalar(std):
            std = [std] * 3
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, img):
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
        arr = arr.transpose(2, 0, 1)  # CHW
        return (arr - self.mean) / self.std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/ref_image")
    ap.add_argument("--base-size", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=640)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    transform = BasicImageTransform(0.5, 0.5)
    pad_color = tuple(int(x * 255) for x in transform.mean.reshape(-1))

    specs = [("small", 500, 400, 0), ("large", 1000, 700, 1), ("tall", 400, 900, 2)]
    manifest = {"mode": "image", "base_size": args.base_size,
                "image_size": args.image_size, "tensors": {}, "cases": []}

    for name, w, h, seed in specs:
        arr = det_image(w, h, seed)
        arr.tofile(os.path.join(args.out, f"{name}_image.bin"))

        im = Image.fromarray(arr)
        case = {"name": name, "width": w, "height": h}

        # crop_mode=True global view (base_size)
        gv = ImageOps.pad(im, (args.base_size, args.base_size), color=pad_color)
        t = transform(gv).astype("<f4")
        t.tofile(os.path.join(args.out, f"{name}_global_crop.bin"))
        case["global_crop_shape"] = list(t.shape)

        # crop_mode=False global view (image_size)
        sq = im.resize((args.image_size, args.image_size))
        gv2 = ImageOps.pad(sq, (args.image_size, args.image_size), color=pad_color)
        t2 = transform(gv2).astype("<f4")
        t2.tofile(os.path.join(args.out, f"{name}_global_nocrop.bin"))
        case["global_nocrop_shape"] = list(t2.shape)

        # dynamic_preprocess local crops
        from importlib import util as _util
        # Reimplement locally to avoid importing the model package.
        min_num, max_num = 2, 32
        aspect = w / h
        ratios = sorted({(i, j) for n in range(min_num, max_num + 1)
                         for i in range(1, n + 1) for j in range(1, n + 1)
                         if min_num <= i * j <= max_num},
                        key=lambda x: x[0] * x[1])
        best_diff, best = float("inf"), (1, 1)
        area = w * h
        for r in ratios:
            d = abs(aspect - r[0] / r[1])
            if d < best_diff:
                best_diff, best = d, r
            elif d == best_diff and area > 0.5 * args.image_size ** 2 * r[0] * r[1]:
                best = r
        tw, th = args.image_size * best[0], args.image_size * best[1]
        resized = im.resize((tw, th))
        crops = []
        cols = tw // args.image_size
        for i in range(best[0] * best[1]):
            box = ((i % cols) * args.image_size, (i // cols) * args.image_size,
                   ((i % cols) + 1) * args.image_size, ((i // cols) + 1) * args.image_size)
            crops.append(resized.crop(box))
        local = np.stack([transform(c) for c in crops], axis=0).astype("<f4")
        local.tofile(os.path.join(args.out, f"{name}_local.bin"))
        case["local_shape"] = list(local.shape)
        case["crop_ratio"] = list(best)
        manifest["cases"].append(case)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out}/manifest.json ({len(specs)} cases)")


if __name__ == "__main__":
    main()
