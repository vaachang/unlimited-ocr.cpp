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

## 3. 视觉 token 数量（已澄清）

早期判断认为"文本侧 273 个 `<image>` 占位 vs 视觉侧 4161 个 embedding"存在矛盾。
经 PyTorch 参考导出核实，**双方都是 273**，并不存在矛盾：

- SAM 的 `net_2` / `net_3` 在 64×64 特征图上各做一次 stride2 下采样，
  最终输出 `[1,1024,16,16]` = **256** 个 token（不是 4096）。
- CLI-L 在 256 个 token 上运行，`[:,1:]` 得 256，与 SAM flatten 拼接后
  经 projector 得到 `[256,1280]`。
- 每行加 `image_newline`（16 行）→ 256+16=272，再加 `view_seperator` → **273**。
- 文本侧 `(num_queries_base+1)*num_queries_base + 1 = (16+1)*16+1 = 273`。

结论：`DeepEncoder::encode` 返回 `rows*(grid+1)+1 = 16*17+1 = 273`，与参考一致。
详见 `ALIGNMENT.md`。

（早期文档中"4161"是按 64×64=4096 计算所致，属误判。）

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

## 8. 网络与磁盘（安装 PyTorch 的坑）

- 本机 `huggingface.co` 直连超时；`hf-mirror.com` 可用。
  `pypi.org` 的 `simple` 索引可达，但大文件（torch wheel、nvidia-* 依赖）经常
  中断，`mirrors.aliyun.com` 的 wheel 链接也慢。
- **可用 HTTP 代理 `http://192.168.1.164:7897`**（用户提供）；经代理后
  pypi 约 13 MB/s、download.pytorch.org 约 3.8 MB/s。安装 Python 依赖时需
  设置 `https_proxy` / `http_proxy`。
- PyTorch CUDA 版来自 `--index-url https://download.pytorch.org/whl/cu128`
  （torch 2.10.0+cu128 能识别 sm_120）。
- **`/tmp` 是 7.9 GB tmpfs（`usrquota`）**。pip 默认在 `/tmp` 解包大 wheel，
  会填满 tmpfs 并报 `[Errno 122] Disk quota exceeded`。解决：设置
  `TMPDIR=/home/admin/.pip-tmp`（位于 `/dev/sda2`），并清理 `/tmp/pip-*` 残留。
- 本机仅 Python 3.14，`torch`/`torchvision` 需选 cp314 wheel；`tokenizers`
  用 `cp39-abi3` wheel 可在 3.14 上工作。

## 9. Vision 对齐中踩的坑（2026-09-15）

用 `tools/compare_vision` + `tools/reference/export_vision_stages.py` 逐段对比
时发现并修复了 3 个问题，记录以备后人：

1. **输入归一化遗漏**：参考 `run_vision` 在喂入 `sam_model` 前做
   `(img-0.5)/0.5`（对应 `BasicImageTransform(mean=std=0.5)`），而
   `compare_vision` 最初直接把 `[0,1]` 原图传给 `encode`。`DeepEncoder::encode`
   期望已归一化输入，务必在调用方完成归一化。
2. **SAM 分解式相对位置漏乘 query**：`add_decomposed_rel_pos` 里
   `rel_h = q · Rh` 是标量（每个 query-key 对），而最初的 C++ 写成
   `s += Rh_d + Rw_d`（对 head 维求和），少了 `q_d *`。修复为
   `s += q_d*(Rh_d + Rw_d)`。对窗口块（14×14）影响较小、对全局块（64×64）
   影响巨大。
3. **CLIP attention 漏加 QKV bias**：`NoTPAttention.qkv_proj` 是带 bias 的
   `nn.Linear`，C++ 只写了 `qkv_w.matmul`，忘了 `qkv_b`。SAM 侧的
   `sam_attention` 加了这个 bias，所以只有 CLIP 出错。

诊断经验：
- **先跑 f32 参考**（`export_vision_stages.py --dtype float32`）可以把
  "实现错误" 与 "bf16 舍入" 分开：f32 下若还差很多，就是实现 bug。
- `--dump-stages` 导出的中间张量要保证**布局一致**再比较：例如 C++
  `record("sam_attn")` 是 **pre-proj** 输出，而 Python hook `blk.attn` 抓的是
  **post-proj** 输出，一开始误比得出"注意力全错"的结论。后来直接对
  `block` 级输出（含 proj/residual/MLP）比较才定位准确。
- C++ 的 `DeepEncoder` CPU 前向较慢，务必用 `OMP_NUM_THREADS=8`
  （CMake 已启用 OpenMP），否则 1024×1024 编码要十几分钟。

## 10. R-SWA 参考导出的 position_ids 陷阱（2026-09-19）

用 `tools/reference/export_reference.py` 做 teacher-forced 解码（直接调
`model(input_ids=..., past_key_values=cache)`，不经过 `generate`）时，若不显式传
`position_ids`，模型会走：

```python
past_key_values_length = past_key_values.get_seq_length()
position_ids = arange(past_key_values_length, seq_length + past_key_values_length)
```

环形 cache 一旦写满（`P+W`），参考的 `SlidingWindowLlamaAttention` 是**原地覆写**
`kcache[:, :, slot]`，`get_seq_length()` 恒等于 `P+W`，于是**所有后续 decode 的位置
编码都被钉在 `P+W`**。表现：`--decode-steps > P+W` 时，从 `P+W+1` 步起参考 logits
相对 C++ 阶跃偏大（rel_l2 从 0.005 跳到 0.3，最终到 1.19），但 `final K/V` 差异只有
几十个最近槽位——极易误判成"环形覆写实现错误"。

真实推理路径（`infer()` → `generate()`）在 `prepare_inputs_for_generation` 里用
`attention_mask.cumsum(-1)-1` 得到**递增**的 `position_ids`，不存在该现象。因此：
- 对比/导出脚本必须显式传 `position_ids`（见 `export_reference.py` decode 循环）；
- C++ `Engine` 递增 `pos` 的语义是正确的，不要为了迁就该导出而去 clamp 位置。

修正后 `--decode-steps 140`：ring 后 logits rel_l2 ≤ 0.036，final K/V ≤ 0.03。
