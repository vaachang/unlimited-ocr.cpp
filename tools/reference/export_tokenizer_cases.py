#!/usr/bin/env python3
"""Export DeepSeek BPE tokenizer reference cases for C++ alignment (E0).

Writes a JSON manifest with, for each corpus string, the HuggingFace token ids
(without special tokens) and the byte-level pre-tokenized pieces.  The C++
`tools/compare_tokenizer` tool checks `uocr::Tokenizer::encode` against it.

Usage:
    ./.venv/bin/python tools/reference/export_tokenizer_cases.py \
        --model models --out /tmp/opencode/ref_tokenizer
"""

import argparse
import json
import os


def build_corpus() -> list[str]:
    cases = [
        # plain ASCII / words / contractions
        "Hello world, this is a test 12345!",
        "The quick brown fox jumps over the lazy dog.",
        "don't can't we'll they're I'm you'd",
        "a b  c\ttab\tseparated",
        "  leading spaces",
        "trailing spaces   ",
        "a\n\nb",
        "line one\nline two\n\nline three",
        "\n\n\n",
        " \n \n ",
        # numbers / versions / units
        "1234",
        "0 1 2 3 4 5 6 7 8 9 10 100 1000 10000",
        "3.14159",
        "2024-09-15",
        "1,234,567.89",
        "100% and $50 and #42",
        "v1.2.3-beta+build",
        "12:34:56.789",
        # punctuation / symbols / code-ish
        "foo_bar-baz.qux/quux",
        "arr[0] = {'k': \"v\"}",
        "if (x <= y && y != z) { return; }",
        "!@#$%^&*()_+{}|:\"<>?~`",
        "a@b.com  http://example.com/path?q=1&r=2",
        # CJK / mixed
        "你好，世界！",
        "第1页 2024年",
        "你好world123",
        "这是一段中文文本，包含标点符号。",
        "中文English混排with数字456and符号！",
        "日本語のテキストとカタカナ",
        "한국어 텍스트는 어떻게?",
        "①②③ Ⅷ Ⅻ ½ ¾",
        # special-ish / chat template fragments
        "<image>hello",
        "<image>" * 3,
        "User: <image>\nWhat is in the image?\nAssistant:",
        "<|User|>hi<|Assistant|>",
        "text with\r\nCRLF endings\r\n",
        # emoji / symbols (category So)
        "emoji 😀 and ✨ mixed 中文",
        "→ ← ↑ ↓  ←→",
        # whitespace edge cases
        "a\u00a0b",
        "x\u3000y",
        "a\u2003\u2003b",
        "end with newline\n",
    ]

    # A moderately long OCR-like page (deterministic) to stress the scanner.
    lines = []
    for i in range(1, 41):
        lines.append(
            f"{i}. Item {i}: value = {i * 37}, ratio {i / 7:.3f}, "
            f"note 中文{i}---ok"
        )
    cases.append("\n".join(lines))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models")
    ap.add_argument("--out", default="/tmp/opencode/ref_tokenizer")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    os.makedirs(args.out, exist_ok=True)

    cases = build_corpus()
    out_cases = []
    for text in cases:
        ids = tok.encode(text, add_special_tokens=False).ids
        pretokens = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(text)]
        out_cases.append({"text": text, "ids": ids, "pretokens": pretokens})

    manifest = {
        "tokenizer": os.path.join(args.model, "tokenizer.json"),
        "count": len(out_cases),
        "cases": out_cases,
    }
    path = os.path.join(args.out, "tokenizer_cases.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    total = sum(len(c["ids"]) for c in out_cases)
    print(f"wrote {path}: {len(out_cases)} cases, {total} reference tokens")


if __name__ == "__main__":
    main()
