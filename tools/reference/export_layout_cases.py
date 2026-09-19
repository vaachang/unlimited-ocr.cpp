#!/usr/bin/env python3
"""Export reference OCR prompt layouts (E1) for C++ alignment.

Replicates the token-layout half of `UnlimitedOCRForCausalLM.infer()` (plain
SFT template + `<image>` placeholder expansion + `images_seq_mask`) without
running the model, so it only needs the HuggingFace tokenizer.

Usage:
    ./.venv/bin/python tools/reference/export_layout_cases.py \
        --model models --out /tmp/opencode/ref_layout
"""

import argparse
import json
import math
import os


def build_layout(tokenizer, prompt, crop_ratios, base_size=1024, image_size=640,
                 crop_mode=True, patch_size=16, downsample_ratio=4,
                 image_token_id=128815, bos_id=0):
    prompt = prompt.strip()
    text_splits = prompt.split("<image>")
    assert len(text_splits) == len(crop_ratios) + 1

    num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
    num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

    tokenized_str, images_seq_mask = [], []

    def text_encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    for text_sep, crop in zip(text_splits, crop_ratios):
        ids = text_encode(text_sep)
        tokenized_str += ids
        images_seq_mask += [False] * len(ids)

        width_crop_num, height_crop_num = crop
        if crop_mode:
            tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [image_token_id] * (num_queries * width_crop_num) + [image_token_id]
                ) * (num_queries * height_crop_num)
        else:
            tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            tokenized_image += [image_token_id]

        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

    ids = text_encode(text_splits[-1])
    tokenized_str += ids
    images_seq_mask += [False] * len(ids)

    tokenized_str = [bos_id] + tokenized_str
    images_seq_mask = [False] + images_seq_mask
    return tokenized_str, images_seq_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models")
    ap.add_argument("--out", default="/tmp/opencode/ref_layout")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    os.makedirs(args.out, exist_ok=True)

    cases = [
        # (prompt, crop_ratios, crop_mode)
        ("Free OCR.", [], True),
        ("<image>\nFree OCR.", [[1, 1]], True),
        ("<image>\nExtract the text in the image.", [[1, 1]], True),
        ("<image>\nConvert to markdown.", [[2, 1]], True),
        ("<image>\nConvert to markdown.", [[1, 2]], True),
        ("<image>\nConvert to markdown.", [[2, 2]], True),
        ("<image>\nConvert to markdown.", [[3, 2]], True),
        ("<image>\n<image>\nCompare the two pages.", [[1, 1], [2, 2]], True),
        ("<image>\nFree OCR.", [[1, 1]], False),
        ("<image>\nParse the figure.", [[1, 1]], False),
        ("中文提示：<image>\n提取图中所有文字。", [[1, 1]], True),
    ]

    out_cases = []
    for prompt, crops, crop_mode in cases:
        ids, mask = build_layout(tok, prompt, crops, crop_mode=crop_mode)
        out_cases.append({
            "prompt": prompt,
            "crop_ratios": crops,
            "crop_mode": crop_mode,
            "input_ids": ids,
            "images_seq_mask": [1 if m else 0 for m in mask],
        })

    manifest = {"count": len(out_cases), "cases": out_cases}
    path = os.path.join(args.out, "layout_cases.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"wrote {path}: {len(out_cases)} cases, "
          f"{sum(len(c['input_ids']) for c in out_cases)} tokens")


if __name__ == "__main__":
    main()
