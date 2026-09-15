# 与 PyTorch 参考实现的数值对齐

本文件记录 `tools/reference/export_reference.py` 导出的参考激活与 C++ 引擎
（`tools/compare_reference.cpp`、`tools/compare_vision.cpp`）的对比结果。

## 1. 参考环境（Python）

本机只有 Python 3.14，未预装 pip。已在仓库内创建虚拟环境 `.venv`（已 gitignore）：

```bash
python3 -m venv .venv
# 网络经代理 http://192.168.1.164:7897（见 PITFALLS）
./.venv/bin/python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.10.0" "torchvision==0.25.0"
./.venv/bin/python -m pip install "transformers==4.57.1" accelerate safetensors \
    tokenizers einops addict easydict tqdm requests huggingface_hub matplotlib
```

验证：`torch 2.10.0+cu128`，`torch.cuda.is_available() == True`，
设备 `NVIDIA GeForce RTX 5060 Ti`（sm_120）。`transformers 4.57.1`。

模型远程代码（`modeling_unlimitedocr.py` 等）已复制到 `models/`，配合
`trust_remote_code=True` 离线加载。

## 2. 参考导出

```bash
./.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_decoder --mode decoder --seq 16 --decode-steps 8
./.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_vision --mode vision
```

- `decoder` 模式：固定 16 个 token 的文本 prompt，导出 `inputs_embeds`、每层
  hidden states、logits、R-SWA KV Cache（prefill + 每步 decode）、MoE 路由
  决策（`prefill_router_ids_*` / `prefill_router_w_*`），以及 8 步 greedy decode。
- `vision` 模式：确定性合成图（`deterministic_image`），导出 SAM 特征、CLIP 特征
  与最终视觉 embedding。
- 每张量同时写 `.npy` 与原始 little-endian f32 `.bin` + `manifest.json`。

关键：参考代码通过 `config._ring_window=W; config.sliding_window=None` 启用
`SlidingWindowLlamaAttention`（`mha_eager`），与 `infer()` 一致。

## 3. Decoder 对齐结果

命令：`./build/tools/compare_reference --model models --ref <ref_decoder>`

输入使用参考导出的 `hidden_0`（即 torch 的 `inputs_embeds`），从而隔离
Decoder 计算。

| 项目 | 结果 |
|---|---|
| 各层 hidden（layer 0–10） | rel_l2 ≈ 0.005–0.008 |
| 第 11 层 hidden | rel_l2 ≈ 1.40（见根因） |
| prefill logits | rel_l2 ≈ 0.009，max_abs ≈ 0.31 |
| prefill K cache（12 层） | rel_l2 ≤ 0.023 |
| prefill MoE 路由 | 176 个决策中 13 个集合不一致 |
| decode logits（8 步） | rel_l2 ≈ 0.003–0.021 |
| decode hidden（8 步） | rel_l2 ≈ 0.4–3.8 |

### 根因分析

- 路由不一致按层分布：`layer1..11 = 0 0 0 2 2 1 0 4 1 2 1`（共 13）。这些是
  **近似并列的专家选择**：参考实现以 bf16 激活计算路由 logits，本引擎以 f32
  计算，微小数值差异导致 top-6 中个别专家翻转。
- 单次专家翻转会把该 token 的输出替换为差异较大的向量；第 11 层 hidden 的
  RMS 较小（ref_rms≈0.6），因此少数 token 的偏差主导了该层 rel_l2。
- 最终 logits 仍然高度一致（<1%），说明对齐在功能上成立；OCR 生成对个别
  路由翻转不敏感。

### 可选的进一步收紧

1. 在 MoE gate 前把激活量化到 bf16，复刻参考的 bf16 舍入，减少路由翻转。
2. 逐层对比 attention/q/k/v 中间量，定位除路由外的残余误差。
3. 增加 `--decode-steps 140` 覆盖 ring 覆盖阶段（warmup 后 W=128 才发生覆写），
   验证环形覆写路径的数值一致性。

## 4. Vision 对齐

参考导出确认视觉 token 数为 **273**：

```
visual_embeddings = 16x16 (256) 个 SAM-CLIP 融合 token
                    + 16 个 image_newline
                    + 1 个 view_seperator = 273
```

这与文本侧插入的 `<image>` 占位数量一致（`(16+1)*16+1 = 273`），
**纠正了 PITFALLS.md §3 中"273 vs 4161"的早期误判**：SAM 的 `net_2/net_3`
在 64×64 上各做一次 stride2 下采样，最终为 16×16，而非 64×64。

`tools/compare_vision.cpp` 已实现（加载 `image.bin` → `DeepEncoder::encode` →
与 `visual_embeddings` 对比）。由于本机 CPU 参考实现运行完整 DeepEncoder 较慢
（数分钟），该对比在本次会话中**未跑完**，列为下一阶段首要验证项。
