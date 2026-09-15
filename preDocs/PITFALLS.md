# 踩过的坑与注意事项

本文件记录实现过程中遇到的非显然问题，避免后续重复踩坑。

## 1. R-SWA 的真实语义（最关键的坑）

`prj.md` 的文字描述与 HuggingFace 参考实现（`modeling_deepseekv2.py` 的
`SlidingWindowLlamaAttention`）有细微但重要的差别，**必须以参考实现为准**：

参考实现实际是：

```
cache 布局： [ prefill 区 (长度 P) ][ ring 区 (W=128) ]
```

- **P = 第一次 prefill 的全部 token 数**（视觉 token + prompt + 特殊 token），
  P 区始终完整保留、永不覆写。它不是"只有视觉 token"。
- **ring 区只保存 Decode 阶段生成的 token**，不是从 prefill 尾部截取。
- 解码开始时先 **warmup**：`cur_len < P + W` 期间按普通追加增长；
  当达到 `P + W` 时置 `ring_pos = 0`；之后再写入就覆盖 `P + ring_pos` 槽位，
  `ring_pos = (ring_pos + 1) % W`。
- 每步 attention 都在**整个 P+W 个槽位**上做（decode 时 q_len=1，无需 causal mask）。

`prj.md` 第 5.1 节提出的"prefill 时把 gap 区 KV 直接丢弃、只保留最后 W 个"
是一个**尚未被参考实现采用的优化**。本项目按参考语义实现（`RSWACache`），
在文档中保留该优化作为后续方向。

对应的环形 buffer 容量为 `P + W`，而不是 `W`。`prj.md` 的显存表格按 batch=16
估算 KV cache ~3 GB 时需注意这一点（P 较大时 P 区占主导）。

## 2. 配置字段的默认值陷阱

`baidu/Unlimited-OCR/config.json` 的**顶层**缺少若干字段，落到 `DeepseekV2Config`
的默认值：

| 字段 | 实际值 | 影响 |
|---|---|---|
| `norm_topk_prob` | `false` | MoE 路由权重**不做归一化**，只乘 `routed_scaling_factor` |
| `routed_scaling_factor` | `1.0` | 路由权重就是 softmax 原始概率 |
| `scoring_func` | `softmax` | |
| `topk_method` | `greedy` | 无 `e_score_correction_bias` |
| `hidden_act` | `silu` | |
| `rms_norm_eps` | `1e-6` | |
| `rope_theta` | `10000` | |
| `sliding_window_size` | `128` | |

字段同时出现在顶层和 `language_config` 里，解析时以顶层为准（`config.cpp` 中做了合并）。

## 3. DeepEncoder 的图像 token 数量矛盾（待解决）

参考实现的 `infer()` 对 base(1024) 模式：

- 文本中插入的 `<image>` 占位 token 数为 `(16+1)*16 + 1 = 273`；
- 但 `UnlimitedOCRModel.forward` 实际生成的视觉 embedding 为
  `64*65 + 1 = 4161`（SAM 4096 + 每行 newline + view_seperator）。

`masked_scatter_` 在 source 元素多于 mask 时**静默截断**，不会报错。这暗示：
要么上游 `infer` 预处理与 `forward` 不一致（bug），要么还有我们未发现的降采样步骤。

**本项目当前按实际 embedding 数量生成视觉 token**（`DeepEncoder::encode` 返回
`rows*(grid+1)+1`），并在后续计划中通过 PyTorch 逐层对齐来厘清。这是端到端 OCR
正确性尚未验证的主要原因。

## 4. DeepEncoder 结构（容易理解错的地方）

- CLIP-L 并不是对原始像素做 patch embedding，而是把 **SAM 的输出特征图**
  （`[B,1024,64,64]`）当作 `patch_embeds` 传入，即 `sam_model(x)` 的结果喂给
  `vision_model(x, sam_features)`。
- CLIP 会 prepend 一个 class token，处理后 `[:, 1:]` 去掉，得到 4096 个 token。
- 拼接顺序是 `cat(CLIP[:,1:], SAM.flatten)` → `[4096, 2048]` → `projector(2048→1280)`。
- SAM 的 window=14，`global_attn_indexes={2,5,8,11}`，对应 `rel_pos_h/w` 长度分别为
  27（window，2*14-1）与 127（global，2*64-1）。窗口分区会把 64×64 pad 到 70×70。
- 位置编码插值：CLIP 的 `position_embedding` 是 16×16+1=257，需 bicubic
  `align_corners=False` 插值到目标网格。

## 5. CUDA / 构建

- **`CMAKE_CUDA_ARCHITECTURES` 默认值**：`enable_language(CUDA)` 会把该变量初始化成
  编译器默认架构（本机得到 75）。必须在 enable 之后显式 `FORCE` 为 120
  （见 `cmake/CUDAArch.cmake`），否则会为错误架构编译。
- `project(LANGUAGES CXX)` 不要写 CUDA，否则 CPU-only 构建也会强制要求 nvcc；
  应先用 `check_language(CUDA)` 再 `enable_language(CUDA)`。
- g++ 16 + nvcc 13.4 实测可编译，不需要降级到 g++-15。
- CUDA 侧需要显式链接 `CUDA::cudart`，仅链接 `uocr_core` 不够。

## 6. C++ / 杂项

- `"\x9Cbegin"`：`\x` 转义会一直吃后续十六进制字符，`b`/`e` 是十六进制字符，
  导致 `\x9Cbe...` 越界。解决方法：用 UTF-8 字面量
  `u8"<｜begin▁of▁sentence｜>"`，或用相邻字符串字面量断开。
- `MemoryPool::used()` **包含对齐 padding**（这是符合真实显存占用的），
  测试期望值需要考虑 padding。
- safetensors 头部是大端无关的 little-endian u64 长度 + JSON，`mmap` 后
  `info.data = base + 8 + header_len + begin`。
- bf16→f32 用移位即可（`bits << 16`）；f32→bf16 做 round-to-nearest-even。
- 本机访问 `huggingface.co` 超时，**使用 `hf-mirror.com`** 镜像下载模型。
- `models/`、`build*/` 已加入 `.gitignore`，不要提交 6.67 GB 权重。
- 真实模型 FP32 权重约 12 GB，超过本机 15 GB 内存，因此：
  - 测试使用合成小模型；
  - 真实权重以 BF16 mmap 零拷贝访问，`DecoderWeights` 只保存指针/视图。

## 7. 连续批处理的一个行为

当前调度器在 batch 已满（`max_batch_size`）时**优先填满 decode**，因此等待队列
要等有请求结束腾出名额后才会被接纳。这是有意的（decode 优先、避免抢占），
测试 `scheduler_continuous_batching` 按此语义编写。`min_batch_size` 目前仅作
配置项保存，未强制合并小 batch。
