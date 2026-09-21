# 与 PyTorch 参考实现的数值对齐

> 开发入口见 `tAgent.md`；文档导航见 `README.md`。

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

## 4. Vision 对齐（已完成）

参考导出确认视觉 token 数为 **273**：

```
visual_embeddings = 16x16 (256) 个 SAM-CLIP 融合 token
                    + 16 个 image_newline
                    + 1 个 view_seperator = 273
```

这与文本侧插入的 `<image>` 占位数量一致（`(16+1)*16+1 = 273`），
**纠正了 PITFALLS.md §3 中"273 vs 4161"的早期误判**：SAM 的 `net_2/net_3`
在 64×64 上各做一次 stride2 下采样，最终为 16×16，而非 64×64。

### 4.1 运行方式

```bash
# 参考（bf16，GPU）
./.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_vision --mode vision
# 参考（f32，用于隔离 bf16 舍入）
./.venv/bin/python tools/reference/export_vision_stages.py --model models \
    --out /tmp/opencode/dbg_sam_f32 --dtype float32
# C++（CPU，OpenMP 并行，约 1–2 分钟）
OMP_NUM_THREADS=8 ./build/tools/compare_vision --model models \
    --ref /tmp/opencode/ref_vision [--dump-stages DIR]
```

`--dump-stages` 会把 `DeepEncoder::encode_stages` 记录的所有中间张量写到
`DIR/*.bin`，配合 `export_vision_stages.py` 可逐段定位误差。

### 4.2 结果

C++ 输出的 `visual_embeddings` 为 `[273,1280]`，与参考形状一致。

对 **f32** 参考（`--dtype float32`）：

| 段 | rel_l2 |
|---|---|
| `sam_patch` / `sam_pos` | ~1e-6 |
| SAM blocks 0–11 | ≤ 1e-5 |
| `sam_neck` / `net_2` / `net_3` | ~3e-4 |
| CLIP layers 0–23 | ~3e-4 … 6e-4 |
| `visual_embeddings` | **≤ 5.8e-4** |

对 **bf16** 参考（真实推理精度）：

| 段 | max_abs | rel_l2 |
|---|---|---|
| `sam_features` (net_3) | 0.0031 | 0.0080 |
| `clip_features` | 0.263 | 0.0377 |
| `visual_embeddings` | 0.226 | **0.0324** |

（`tools/compare_vision` 一次运行约 2 分钟，8 线程。）

逐段（对 f32 参考）误差：`sam_patch/sam_pos ≈ 1e-6`，SAM 各 block `≤ 1e-5`，
`sam_neck/net_2/net_3 ≈ 3e-4`，CLIP 各层 `≈ 3e-4…6e-4`。也就是说剩下的
3.2% 完全来自参考端的 bf16 激活舍入，而 **C++ 的 f32 实现与 f32 参考在数值上
等价**。

### 4.3 修复的两个关键 bug（详见 PITFALLS §9）

1. **SAM 分解式相对位置**：C++ 把 `sum_d (Rh_d + Rw_d)` 当成了偏置，正确为
   `q · (Rh + Rw) = sum_d q_d * (Rh_d + Rw_d)`，漏乘 query。
2. **CLIP attention 漏加 QKV 投影 bias**：`NoTPAttention.qkv_proj` 带 bias，
   C++ 只做了 `qkv_w.matmul`，未加 `qkv_b`。

修复后，Decode 阶段仍保持 <1% 的既有对齐（见第 3 节）。

## 5. 端到端图像 OCR 对齐（E0–E5，已完成）

参考导出与 C++ 对比，全部工具在 `tools/`：

```bash
# 参考（GPU）
./.venv/bin/python tools/reference/export_tokenizer_cases.py --model models \
    --out /tmp/opencode/ref_tokenizer
./.venv/bin/python tools/reference/export_layout_cases.py --model models \
    --out /tmp/opencode/ref_layout
./.venv/bin/python tools/reference/export_image_cases.py --out /tmp/opencode/ref_image
./.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_ocr --mode ocr --prompt $'<image>\nFree OCR.' \
    --crop-mode --ocr-width 500 --ocr-height 400 --decode-steps 24 \
    --ngram-size 35 --ngram-window 1024
# C++
./build/tools/compare_tokenizer --model models --ref /tmp/opencode/ref_tokenizer/tokenizer_cases.json
./build/tools/compare_layout    --model models --ref /tmp/opencode/ref_layout/layout_cases.json
./build/tools/compare_image     --ref /tmp/opencode/ref_image
OMP_NUM_THREADS=8 ./build/tools/compare_ocr --model models --ref /tmp/opencode/ref_ocr
```

| 阶段 | 项目 | 结果 |
|---|---|---|
| E0 | 预分词 / ids（43 用例） | 43/43 |
| E1 | prompt/`<image>` 布局 ids + mask（11 用例） | 11/11 |
| E2 | 预处理 vs Pillow | ≤1 LSB，crop ratio 全对 |
| E3/E4 | `image_global` | rel_l2 6.6e-5 |
| E3/E4 | `visual_embeddings`（273 token） | rel_l2 **0.0419** |
| E4 | `prefill_logits` | rel_l2 0.0637，top-1 一致（3051） |
| E5 | greedy（ngram 35/1024） | **24/24** |

说明：视觉/首 token 的 4–6% 误差与第 4 节一致，来源是参考端 bf16 激活舍入
（C++ 为 f32）；参考 prefill logits 本身用 bf16 计算。**greedy token 序列完全一致**，
说明该误差不影响生成结果。

### 5.1 修复的 bug

1. **`crop_ratio` 判定**：参考 `infer()` 对 `w,h <= 640` 的图直接设
   `crop_ratio=[1,1]`，不调用 `dynamic_preprocess`。C++ 最初无条件调用，导致
   视觉 token 由 273 变 2473。已在 `Engine::image_embeddings` 修复。
2. **`ImageOps.pad` 中心对齐**：Python `round()` 为 round-half-to-even，
   C++ `std::lround` 为 round-half-away-from-zero，半像素时偏 1 行/列。
   已改用 `std::nearbyint`（`py_round`）。
3. **no-repeat-ngram 语义**：参考/SGLang 是"匹配 `ngram-1` 前缀并禁用其延续词、
   在 `[len-window, len-ngram+1)` 内搜索"，原实现按完整 n-gram 匹配。
   已在 `Sampler::apply_no_repeat_ngram` 修正。

### 5.2 CUDA 端端到端回归（2.11，2026-09-21）

`compare_ocr` 新增 `--gpu`（CUDA 解码器）/`--int4`/`--int4-group N`，可对同一
`ref_ocr` 跑 CPU、CUDA-BF16、CUDA-INT4 三条路径：

```bash
./build-cuda/tools/compare_ocr --model models --ref /tmp/opencode/ref_ocr --gpu --gpu-vision            # CUDA/BF16
./build-cuda/tools/compare_ocr --model models --ref /tmp/opencode/ref_ocr --gpu --gpu-vision --int4     # CUDA/INT4
./build-cuda/tools/compare_ocr --model models --ref /tmp/opencode/ref_ocr --gpu-vision --int4           # CPU/INT4
```

| 路径 | visual rel_l2 | prefill logits rel_l2 | top-1 | greedy |
|---|---|---|---|---|
| CPU/f32（旧基线） | 0.0419 | 0.0637 | 3051 ✓ | **24/24** |
| CUDA/BF16 + GPU vision | 0.0418 | 0.0601 | 3051 ✓ | **24/24** |
| CPU/INT4（group 128） | 0.0418 | **0.3623** | 8227 ✗ | 0/24 |
| CUDA/INT4（group 128） | 0.0418 | **0.3620** | 8227 ✗ | 0/24 |
| CUDA/INT4（group 32） | 0.0418 | 0.3067 | 3051 ✓ | 1/24 |

结论：
- **CUDA 解码器正确**——CUDA/INT4 与 CPU/INT4 的 logits/贪心 token 完全一致
  （8227/28/54678…），所以差异不是 CUDA 移植 bug。
- **BF16（f32 激活）两条路径都与参考 24/24**，视觉精度的 bf16 漂移不影响生成。
- **INT4（当前 group-128 非对称 min/max，非真正 AWQ）精度不足**：单专家权重
  round-trip rel-L2 ≈ 0.10（`PROGRESS.md` §5），12 层 MoE 后 prefill logits 误差
  ~0.36，翻转 top-1；这个合成图（随机 RGB 梯度、非真实文字）本身输出是退化重复
  序列，任何扰动都会级联，故 0/24。group=32 把误差降到 0.31、top-1 恢复但第 2 步
  仍分叉。**真实 OCR 精度需 OmniDocBench 评估（任务 2.7）**；量化器本身是
  round-to-nearest，`quantize_int4_awq` 的名字并不名副其实（未用激活统计）。

### 5.3 多 crop（>640px）端到端对齐（2.12，2026-09-21）

`>640px` 的图片按参考 `infer` 走 Gundam 动态切图（最多 32 个 640 局部 crop + 1024
全局视图）。为验证该路径，临时脚本（`/tmp/opencode/export_ocr_large.py`，仿
`export_reference.py --mode ocr` 但实现 `dynamic_preprocess` 与动态 token 布局）
导出了 800×400 参考（crop_ratio `(2,1)`，2 个局部 crop，488 ids / 483 visual）：

```bash
./build-cuda/tools/compare_ocr --model models --ref /tmp/opencode/ref_ocr_large --gpu --gpu-vision
# 以及 CPU vision：... --gpu
```

| 路径 | visual rel_l2 | prefill logits rel_l2 | top-1 | greedy |
|---|---|---|---|---|
| 修复前（CPU/GPU vision） | 0.430 / 0.431 | — | 3051 ✓（退化） | 1/16 |
| 修复后 CPU vision | **0.0572** | 0.0247 | 1 ✓ | **16/16** |
| 修复后 CUDA/BF16 + GPU vision | **0.0572** | 0.0244 | 1 ✓ | **16/16** |

- 布局 `ids=488 mask_true=483` 与参考完全一致。
- 修复内容：见 `PITFALLS.md` §21（`UOCR_THROW` 缺 `throw`；`Engine::image_crops`
  统一布局/视觉 crop；`image_embeddings` 多 crop 拼接按参考重写；SAM
  `pos_embed`/`rel_pos` 对非 1024 输入插值）。修复后误差与 1×1 路径同量级
  （0.057 vs 0.042），属 bf16/f32 正常漂移。
- **已入库（2.13，2026-09-21）**：`export_reference.py --mode ocr` 现支持 `--image-file`
  与参考自身的 `dynamic_preprocess`；`compare_ocr --strict` 作为回归判定；CTest 注册
  `compare_ocr_large`。复核（CUDA/BF16+GPU vision）：
  `ids=488 / mask=483 / crop_ratio=(2,1) / local_crops=2`，visual rel_l2 **0.0572**、
  prefill rel_l2 **0.0241**、greedy **16/16**；`ctest -R compare_ocr` 两条均 Passed
  （`--strict` 下 `--visual-tol 0.001` 会正确 FAIL，证明判定有效）。

## 6. R-SWA 环形覆写验证（P1，已完成）

用 `--decode-steps 140` 跑过 `P+W=16+128=144` 的覆写拐点（`ref_decoder140`）：

```bash
./.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_decoder140 --mode decoder --seq 16 --decode-steps 140
./build/tools/compare_reference --model models --ref /tmp/opencode/ref_decoder140
```

| 项目 | 结果 |
|---|---|
| decode logits rel_l2（ring 前，step ≤128） | ≤ 0.084 |
| decode logits rel_l2（ring 后，step ≥129） | ≤ 0.036 |
| final K cache rel_l2（144 槽，12 层） | **≤ 0.015** |
| final V cache rel_l2 | **≤ 0.030** |

结论：C++ `RSWACache::append_decode` 的 warmup/ring 指针与参考完全一致。

### 6.1 一个导出侧陷阱（不是引擎 bug）

`export_reference.py` 最初在直接调用 `model(...)` 做 teacher-forced 解码时**没有显式
传 `position_ids`**。此时模型用 `arange(past_key_values.get_seq_length(), ...)` 生成
位置，而环形 cache 写满后 `get_seq_length()` 恒为 `P+W`，导致**位置编码从第 145 步起
被钉在 144**，参考 logits 出现阶跃误差（rel_l2 1.19）。真实 `infer()`/`generate` 用
不断增长的 `attention_mask.cumsum` 得到递增位置，不存在该问题。导出脚本已显式传
`position_ids`，修正后环后 logits rel_l2 回到 ≤0.036。详见 `PITFALLS.md` §10。

### 6.2 bf16 舍入尝试（未生效）

`MoEDecoder::set_bf16_rounding(true)` 会把 MLP/MoE 前的 RMSNorm 输出舍入到 bf16，
以模仿参考 gate 的 `hidden_states.type(torch.float32)`。实测 prefill
`set_mismatch` 仍为 13/176（per-layer 分布不变）。根因是 router 输入已经因
attention/expert 的 f32 累积而偏离参考 bf16 激活超过 1 ulp，仅舍入最后一步无效。
保留为可选开关，默认关闭。

## 7. 逐层 attention q/k/v/O 对比（P1，2026-09-19）

`export_reference.py` 增加对每层 `self_attn.{q,k,v,o}_proj` 的 hook，导出
`{prefill,decode0,decode1}_attn_{q,k,v,o}_{li}`；`MoEDecoder::set_trace_attn(true)` 在
C++ 侧记录同样的张量；`compare_reference` 输出逐层 rel_l2：

| 段 | q | k | v | o |
|---|---|---|---|---|
| prefill worst | 0.0245 | 0.0233 | 0.0437 | 0.0567 |
| decode0 worst | 0.0222 | 0.0250 | 0.0473 | 0.0414 |
| decode1 worst | 0.0123 | 0.0126 | 0.0216 | 0.0291 |

prefill 逐层 q rel_l2（layer0→11）：
`0.0018 0.0047 0.0043 0.0066 0.0064 0.0121 0.0188 0.0199 0.0191 0.0245 0.0232 0.0195`。

**结论**：误差随层数单调累积（第 0 层 ~0.2% → 第 9 层 ~2.5%），v/O 投影比 q/k 稍大
（v 无 RoPE、误差不会因旋转而部分抵消）。这是 **bf16 激活累积漂移**，不是实现错误：
C++ 侧 `MoEDecoder` 全程 f32 累积，参考端为 bf16 autocast。它与第 3 节的 13/176 路由
翻转同源，也解释了第 11 层 hidden 的 1.40（该层 RMS 小，少数 token 偏差占主导）。
在最后 logits 上仍收敛到 <1%，greedy 完全一致。

## 8. 复现

```bash
./.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_attn --mode decoder --seq 16 --decode-steps 4
./build/tools/compare_reference --model models --ref /tmp/opencode/ref_attn
```
