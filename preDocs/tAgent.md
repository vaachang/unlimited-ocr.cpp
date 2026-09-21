# Unlimited-OCR 引擎 — 开发入口（tAgent）

> **下次继续开发请从本文件开始。** 顶部是当前状态与下一步任务；历史明细、实现细节、
> 数据、踩坑分别在 `PROGRESS.md` / `CORE_TECH.md` / `BENCHMARKS.md` / `PITFALLS.md` /
> `ALIGNMENT.md`。文档导航见 `README.md`；原始需求见 `prj.md`。

## 0. 项目要求（原始任务）

1. 按 `prj.md` 实现项目。
2. 有不明白/不清楚的地方先问；安装第三方依赖需先征得同意，不要擅自安装。
3. 中间进度文档、重要性能测试结果、踩过的坑、核心技术实现保留在 `preDocs/`。
4. 使用 g++ 作为 C++ 编译器。
5. 用 git 保存项目。

---

## 1. 当前状态（2026-09-21 快照）

- **后端**：CPU 参考（OpenMP）+ CUDA 生产（sm_120），`-DENGINE_BACKEND=CPU|CUDA` 双构建。
- **完成度**：M1–M6、P0（端到端 OCR 对齐 E0–E5）、P1（R-SWA 环形覆写 / CUDA 设备端
  decoder）、P2（TC INT4 GEMM、CUDA Graph、device INT4 权重、连续批处理、batched graph、
  bf16 TC GEMM）、P3（cp.async 流水线 TC GEMM、DeepEncoder CUDA 移植、**grouped INT4
  专家 GEMM + device router**、**device embedding 查表**、**INT4 量化并行加载**、
  **视觉 split-bf16 TC GEMM + relpos 因式分解**、**视觉 tensor-core flash attention**）
  均已完成（2.6 分析后判定不适用/不实现，见该节）。**约 98%**（对照 `prj.md`；
  仅剩 2.7 精度评测/2.8 profiling 受外部工具与权限限制，2.14 为纯性能项）。
- **收尾决定（2026-09-21）**：项目判定为**可用、完整**，本轮不再改功能代码。
  2.14（BF16 grouped ragged prefill）保留为性能可选项；2.7（OmniDocBench）需先安装
  外部工具链（Docker/TeX Live 等）；2.8 受本机 `ncu` 权限限制。若后续要做，按 §2 各节
  的验收口径实施。
- **入口/可用性**：新增独立 CLI `tools/ocr_image`（`--image page.png` → 打印识别文本，
  默认 CUDA + GPU 视觉 + **BF16 专家**；`--cpu`/`--int4`/`--no-crop-mode` 可选；图片支持
  PNG（libpng）与 PPM）。`EngineConfig::use_int4_experts` 默认改为 **false（BF16）**，
  INT4 作为显存/速度受限时的显式选项；`generate_from_image` 增加单测。
- **多尺寸修复（2026-09-21）**：修复 `UOCR_THROW` 缺 `throw`（此前所有 `UOCR_CHECK`
  静默失效）、`>640px` 图片 prompt 布局与视觉 embedding 数量不一致（`ocr_image`
  写死 1×1 crop 而 `image_embeddings` 动态切图），以及视觉编码器对非 1024 输入的
  SAM `pos_embed`/global `rel_pos` 插值缺失。新增 `Engine::image_crops()` 作为
  「布局/视觉使用同一套 crop」的唯一来源，`image_token_count()` 与
  `image_embeddings` 计数互相校验。验证：800×400 多 crop 参考 visual rel_l2
  **0.057**、prefill 0.024、greedy **16/16**；1×1 参考对齐不变（0.0418、24/24）；
  `uocr_tests` **27/27**。现在 **>640px 图片可直接用默认 crop mode**。
- **多 crop 回归入库（2.13，2026-09-21）**：`export_reference.py --mode ocr` 支持
  `--image-file` 与参考自身 `dynamic_preprocess`；`compare_ocr --strict` 作为回归判定；
  CTest 注册 `compare_ocr`（1×1）与 `compare_ocr_large`（800×400，(2,1) 双 crop），
  两条均 **Passed**。真实 800×1000 页面（4×5=20 crop，2323 visual tokens）经
  `ocr_image` 识别正确。
- **可用性**：新增根目录 `README.md`（构建 / 权重 / 用法 / 对齐 / 目录结构入口）。
- **回归**：`ocr_image` 在参考图上与 PyTorch 参考输出**逐字一致**；`compare_ocr`
  greedy **24/24**（CPU/f32 与 **CUDA/BF16+GPU vision** 两条路径），多 crop
  16/16（见上）；`uocr_tests` **27/27**（含 PNG/PPM 图像加载 + 多尺寸布局/异常校验
  回归）；`uocr_cuda_tests` **全过**；`compare_vision_gpu_selftest` **全过**。
  **INT4（group-128 RTN）** 端到端与参考分叉（top-1 翻转、0/24），CPU/INT4 与
  CUDA/INT4 **完全一致**，属量化精度问题（见 2.11 / `ALIGNMENT.md` §5.2），故默认
  不再用 INT4。
- **性能**（真实 `baidu/Unlimited-OCR`，prompt=64, steps=16, max_batch=16, warmup 稳态）：

  | 指标 | 值 |
  |---|---|
  | BF16 batch=16 吞吐 | **~520 tok/s**（显存峰值 10.4GB） |
  | INT4 batch=16 吞吐 | **~518 tok/s**（显存峰值 3.37GB，含 331MB bf16 embedding 表） |
  | 整波 prefill（16 请求） | **~165 ms** BF16 / **120 ms** INT4 |
  | batch=1 吞吐 | 155.8 BF16 / 138.2 INT4 tok/s |
  | lm_head m=16（n=129280） | **1174 µs**（原 2652，2.3×） |
  | 单图视觉编码（1024, GPU） | **238–246 ms**（CPU 参考 ~2 min；rel_l2 1.44e-4；TC attention + 8-warp n-split） |

- 性能数据明细见 `BENCHMARKS.md` §2.5–2.10；历史进度见 `PROGRESS.md` §4。

**快速验证**：

```bash
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8
ctest --test-dir build-cuda --output-on-failure
# 端到端 OCR（CUDA + GPU 视觉 + BF16 专家；--int4 可切 INT4）
./build-cuda/tools/ocr_image --model models --image page.png
./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
```

---

## 2. 下一步任务（开发从这里开始）

按收益/成本排序。每项给出目标、方案、验收与涉及文件；完成后在本文档勾掉并更新 §1。

> **下一阶段计划（2026-09-21 收尾后）**：项目已判定**可用、完整**（见 §1）。
> 以下为后续 **可选 / 受外部条件限制** 的工作，按收益/成本排序；开工前先确认范围。
>
> 1. **2.14 BF16 grouped ragged prefill（性能，首选）**
>    - 现状：默认 `use_int4_experts=false`，BF16 大批量 ragged prefill 仍走
>      host router D2H + 逐专家 `linear_forward`（每层 64×3 次 launch）。
>    - 方案：仿 `moe_grouped_gate_up_int4` / `moe_grouped_down_int4`，新增
>      `moe_grouped_gate_up_bf16` / `moe_grouped_down_bf16`（gate+up+SiLU 融合、down
>      scatter-add，各一次 launch，读 device grouping 表；权重 bf16，激活走
>      `matmul_t_split_bf16` 或等价 TC 路径）。`forward_ragged` 里 BF16 也走 grouped。
>    - 验收：整波 prefill（16 请求）**165ms → ≤120ms**、B=16 吞吐 **≥550 tok/s**；
>      `test_rswa_cuda.cu` 新增 grouped-BF16 vs host 回归 rel_l2 同量级；greedy 不变。
>    - 涉及：`src/kernels/cuda/moe_device.cu`（或新 `moe_gemm_bf16.cu`）、
>      `include/uocr/cuda_ops.h`、`src/engine/gpu_decoder.cu`、`tests/test_rswa_cuda.cu`、
>      `BENCHMARKS.md`。
> 2. **2.7 精度评测（OmniDocBench v1.6）** — 需外部工具链，**安装前必须先征得同意**。
>    - 依赖：官方 Docker 镜像 `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`
>      （或本机 TeX Live 2025 + ImageMagick 7 + Ghostscript + Python 3.10，~7GB+）。
>    - 接入后按 `configs/end2end.yaml` 出 text Edit / TEDS / CDM 综合分；本机当前无
>      Docker，该项未接入。
> 3. **真正的 AWQ / GPTQ 量化** — 依赖校准前向（可用 2.7 数据或自建校准集）。
>    当前 `quantize_int4_awq` 实为 group-wise RTN（权重误差 ~10%、端到端 logits ~0.36，
>    top-1 翻转），改用激活感知 per-channel scaling 预计降到 ~3–5%，INT4 端到端才有意义。
>    涉及 `src/runtime/quant.cpp`、`weights.cpp`。
> 4. **2.8 profiling 补全（受限）** — 本机 `ncu` 报 `ERR_NVGPUCTRPERM`（PITFALLS §17），
>    只能用 `nsys` + 消融；需要 SM/DRAM 峰值利用率时先解决权限。
> 5. **非阻塞优化**：`rswa_attn_ragged`（B=16 ~21ms，grouped 之后的第二大头）；
>    整栈 CUDA Graph（encoder+cache 一起捕获）；视觉 GEMM cp.async；单请求 prefill
>    的 host embedding gather。
> 6. ⛔ **2.6 Prefill KV 分区**：分析后判定与参考实现不兼容、且本负载 `V+W>P` 无
>    可丢弃 gap，**不实现**（见该节）。
>
> 以下 ✅/🔶/⛔ 是 2026-09-20/21 的完成快照。✅ 已完成；🔶 部分完成；⛔ 分析后不实现。

### ✅ 2.12 可用性修复：异常校验 + 多 crop 端到端（2026-09-21 完成）

- **已做**：
  1. `include/uocr/common.h` 的 `UOCR_THROW` 此前漏了 `throw`，导致全项目
     `UOCR_CHECK` 静默失效；已补 `throw`。
  2. 新增 `Engine::image_crops()` 作为「prompt 布局 / 视觉 embedding 使用哪套 crop」
     的唯一来源；`ocr_image`、`compare_ocr` 均改用它；`image_token_count()` 与
     `image_embeddings`、`build_ocr_prompt` 三方计数互相校验。
  3. `image_embeddings()` 多 crop 拼接按参考重写：各 crop 先拼成
     `(hcn·h2)×(wcn·w2)` 大网格、**整体**每行加一次 `image_newline`，再接 global
     视图与 `view_seperator`（原来逐个 crop 各带 newline 再首尾相接，数量/顺序都错）。
  4. 视觉编码器（CPU `deep_encoder.cpp` + GPU `gpu_encoder.cu`）对非 1024 输入做
     SAM `pos_embed`（bicubic）与 global `rel_pos`（linear, `align_corners=False`）
     插值，对齐参考 `get_abs_pos_sam` / `get_rel_pos`。
- **验收**：`uocr_tests` **27/27**；1×1 参考对齐不变（visual 0.0418、greedy 24/24）；
  新增 800×400 多 crop 参考 visual rel_l2 **0.057**、prefill logits 0.024、
  greedy **16/16**（修复前 0.43/解码发散）；`1280×720` 图片默认 crop mode 可正确识别。
- **涉及**：`common.h`、`engine.{h,cpp}`、`prompt.{h,cpp}`、`deep_encoder.cpp`、
  `gpu_encoder.cu`、`tools/{ocr_image,compare_ocr}.cpp`、`tests/test_image.cpp`。
  细节见 `PITFALLS.md` §21。

### ✅ 2.13 多 crop 参考回归入库（2026-09-21 完成）

- **已做**：
  1. `tools/reference/export_reference.py --mode ocr` 支持 `--image-file`，并用**参考
     模型自身的 `dynamic_preprocess`**（`importlib` 取 `modeling_unlimitedocr` 同款
     函数，保证切图网格/平局规则逐位一致）生成 `images_crop`、`crop_ratio` 与多 crop
     token 布局。
  2. `tools/compare_ocr` 增加 `--strict`（默认容差 visual/prefill rel_l2 ≤0.15、
     greedy 全中）；不达标退出码非 0，并顺带修掉原 summary 里 `ref_visual` 越界读的
     隐患（改为复用 `report()` 的返回值）。
  3. `CMakeLists.txt` 在 `ENGINE_REFERENCE_DIR` 分支注册 `compare_ocr_large`
     （`ref_ocr_large`，800×400，crop_ratio `(2,1)`，488 ids / 483 visual），两条
     `compare_ocr` 测试都加 `--strict`。
- **验收**：导出 800×400 → `ids=488 visual_tokens=483 crop_ratio=(2,1) local_crops=2`；
  CTest `compare_ocr`（1×1，24/24）与 `compare_ocr_large`（多 crop，16/16）均
  **Passed**；CUDA/GPU vision 直跑：多 crop visual rel_l2 **0.0572**、prefill
  **0.0241**、greedy **16/16**；`--strict --visual-tol 0.001` 能正确 FAIL(exit 1)。
- **涉及**：`tools/reference/export_reference.py`、`tools/compare_ocr.cpp`、
  `CMakeLists.txt`、`README.md`、`ALIGNMENT.md` §5.3。

### ✅ 2.1 高性能 TC GEMM 重写（2026-09-20 完成）

- **已做**：`matmul_t_bf16` 换成 cp.async 多级 ring + bank-conflict-free padding 的
  `matmul_t_bf16_tc_pipe_kernel`（m≤16 的 n 拆分小 tile / m>16 的 64×64 tile）；
  INT4 专家 GEMM 激活 staging 向量化并把默认 tile 改为 `bn=8/bk=64`，另加 cp.async
  备选变体。详见 `CORE_TECH.md` §5.5 / §5.5b，数据见 `BENCHMARKS.md` §2.5/§2.7。
- **结果**：lm_head m=8/16 2.3×；BF16 batch=16 416→**522** tok/s、整波 prefill
  212→**161** ms；INT4 batch=16 387→**429** tok/s。
- **剩余**：m>16 的 bf16 kernel 每 n-tile 重读 A（未处理）；`moe_gemm_int4_tc`
  的逐专家发射已被 2.2 的 grouped GEMM 取代。

### ✅ 2.3 DeepEncoder CUDA 移植（2026-09-20 完成）

- **已做**：新增 `src/engine/gpu_encoder.cu`（`cuda::GpuEncoder`）与
  `src/kernels/cuda/vision_ops.cu`（layernorm / gelu / im2col / window partition /
  flash attention）。patch+pos、12 层 SAM、neck、net2/net3、CLIP×24、projector
  全部设备执行；线性权重 bf16 常驻，但用 **f32 tiled GEMM**（`matmul_t_f32w`）以
  匹配 f32 参考（bf16 激活在深层栈里会漂到 ~11%）。`Engine::set_vision_gpu()` +
  `image_embeddings` 分派，`tools/compare_ocr --gpu-vision` 可端到端验证。
- **结果**：`tools/compare_vision_gpu` visual rel_l2 **1.1e-5**（验收 ≤6e-4）；
  单图 1024 编码 **612 ms**（CPU ~2 min）、640 153 ms、224 34 ms。
  详见 `CORE_TECH.md` §5.8、`BENCHMARKS.md` §2.9、`PITFALLS.md` §18。
- **剩余**：见 2.10。

### ✅ 2.4 device router + 2.2 grouped expert GEMM（2026-09-20 完成）

- **已做**：ragged 大批量 prefill 不再走「router D2H + host top-k + 逐专家 GEMM」。
  `forward_ragged` 里 INT4 专家一律改走 device router（`moe_router_topk` 产出
  `(assign_token, assign_w, count)`，`cap=total`）与两个 grouped 内核：
  `moe_grouped_gate_up_int4`（grid=(ceil(inter/BN), n_experts)，融合 gate+up+SiLU）
  和 `moe_grouped_down_int4`（grid=(ceil(hidden/BN), n_experts)，含 scatter-add）。
  每层从 ~192 次 launch 降到 2 次、无 D2H/H2D 同步。tile 可调
  （`EngineConfig::grouped_moe_bn/bm`，默认 `bn=32/bm=128`；`BM=128` 是最大收益点）。
  见 `CORE_TECH.md` §5.9、`PITFALLS.md` §19。
- **结果**：INT4 batch=16 整波 prefill **213→120 ms**、tok/s **429→488–516**；
  B=1/2/4/8 prefill 全线下降。`tests/test_rswa_cuda.cu` 新增 grouped vs 逐专家 host
  回归（600 token，多 m-tile），rel_l2 **1.7e-3**（device vs host router 同量级）。
- **剩余**：BF16 专家的 ragged 大批量 prefill 仍走 host 逐专家路径（未加 grouped
  BF16 内核）；`rswa_attn_ragged` 成为新的第二大头（B=16 ~21 ms），未优化。

### ✅ 2.5 device embedding 查表（2026-09-20 完成）

- **已做**：`upload_embedding` 把 `embed_tokens` 常驻设备（源为 BF16 时存 bf16，
  否则 f32，保证与 host `row()` 逐位一致）；新增 `embed_gather` 内核按下标 gather。
  单请求 Graph 内用 gather 取代原来 `hidden` 长度的 H2D，`batch_decode` 只 H2D
  `batch` 个 token id（4B/行）。`d_token_ids_` 容量变化会使已捕获 Graph 失效
  （`ensure_token_ids`），`batch_configure` 预分配到 slot 数以免反复失效。
  见 `CORE_TECH.md` §5.10。
- **结果**：每步不再有 `hidden`/`batch*hidden` 的 embedding H2D；真实模型
  BF16/INT4 batch=16 约 **520/518 tok/s**（原 522/488–516，噪声内），greedy 不变；
  显存 +331MB（bf16 表）。
- **剩余**：单请求 `prefill_tokens` 仍在 host 查表（每请求一次，不是每步）；
  position 仍是 4B pinned H2D（本就是 int，未做成表）。

### ✅ 2.9 权重加载优化（2026-09-20 完成）

- **已做**：负载实测的主要成本不是 H2D（~2.1GB / 2s），而是 **INT4 AWQ 量化**
  （2112 个专家张量、单线程 ~18s）。量化按专家完全独立且 `quantize_int4_awq` 是
  纯函数、`read_f32` 只读 mmap，故在 `weights.cpp` 的专家循环加
  `#pragma omp parallel for schedule(dynamic)`（仅 INT4 时）。
- **结果**：真实模型 `--real --int4` 进程墙钟 **20.5 → 7.4s**（8 核；H2D ~2s 不变），
  量化结果确定，显存不变。
- **未做（受环境限制）**：`mmap + cudaHostRegister` pinned 直通——本机 `ulimit -l`
  只有 **8MB**，无法 pin 6.67GB 权重；分块 pinned 流水意义有限（H2D 仅 ~2s）。
  nsys 的 “H2D 占 host API 76%” 是 host-track 占比，不代表墙钟。
- **涉及**：`src/runtime/weights.cpp`。

### 2.8 性能记录补全

- **方案**：用 ncu 采 SM/DRAM 峰值利用率；KV Cache 碎片率；纯 decode 的
  TTFT/TPOT 分解。**注意**：本机 `ncu` 报 `ERR_NVGPUCTRPERM`（PITFALLS §17），
  该子项暂时做不了，只能用 nsys + 消融。

### 2.6 Prefill KV 分区写入优化（prj.md 创新点三）— 分析后判定不适用/不实现

- **prj.md 设想**：prefill 时按位置分区（视觉区 / 环形区 / gap 丢弃），省 ~70% KV
  写入带宽。
- **与参考冲突（关键）**：参考实现（`modeling_deepseekv2.py`
  `SlidingWindowLlamaAttention`）**prefill 写全部 P 个 KV**，且 **decode 在完整
  `P+W` cache 上做 attention、没有任何 window mask**（`_attn_forward` 只在真 prefill
  用 causal mask；ring decode 路径对全 cache 做 softmax）。因此 **P 区每个槽位 decode
  都会读到**，丢弃任何 gap 都会改变输出；要自洽就必须同时把 decode 改成 windowed
  mask，那是与参考不同的新设计。
- **对本工作负载无收益**：真实 OCR 用例 `P=278`（273 视觉 + 5 文本/special）、
  `W=128`，`V + W = 401 > P`，**没有可丢弃的 gap**（丢弃比例 0%）。视觉区 KV 的
  跨请求共享（创新点一）已由 block manager 实现。
- **结论**：不实现该分区（会偏离参考且无收益）。若将来要支持「超长文本 prompt +
  windowed decode」的新语义，需要单独的 design + `EngineConfig` 开关，并与参考路径
  隔离评测。**涉及**：`src/kernels/cuda/rswa_attention.cu`、`GpuRSWACache`。

### 🔶 2.7 精度评测（部分完成：本地量化消融已做；OmniDocBench 待外部工具链）

- **已完成（本地量化消融，2026-09-21）**：`inspect_model --quant-check` 扩展为
  scheme × group 扫描（12 个采样专家矩阵，weight round-trip rel-L2 / 有效 bit）：

  | scheme | g=32 | g=64 | g=128 | g=256 |
  |---|---|---|---|---|
  | awq/asym（当前实现） | **0.0809**(6.00b) | 0.0913(5.00b) | 0.1010(4.50b) | 0.1096(4.26b) |
  | symmetric | 0.0974(6.00b) | 0.1082(5.00b) | 0.1180(4.50b) | 0.1268(4.26b) |

  即当前 `quantize_int4_awq` 实为 **RTN（round-to-nearest）group 量化**，未用激活
  统计；即使 group=32 也有 ~8% 权重误差，端到端 prefill logits rel_l2 0.31–0.36
  （见 `ALIGNMENT.md` §5.2）。对称量化更差。
- **未做（受外部工具链限制）**：OmniDocBench v1.6 端到端评测需要 1651 页数据集
  + TeX Live 2025(~7GB) + ImageMagick 7 + Ghostscript + Python 3.10 环境（官方推荐
  Docker 镜像 `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`），本机无
  Docker/该工具链，**未接入**。接入后可按 `configs/end2end.yaml` 跑 text Edit /
  TEDS / CDM 综合分。
- **改进方向（需单独确认）**：真正 AWQ（激活感知 per-channel scaling，需校准前向）
  或 GPTQ 误差补偿，预计把权重误差从 ~10% 降到 ~3–5%；或在显存允许时直接用
  BF16 专家（3B/6.2GB，16GB 卡可放下，端到端已 24/24）。

### ✅ 2.10 视觉编码器性能收尾（完成；TC attention 已调通 2026-09-21）

- **已完成（`f678751`，1024 612→366 ms，rel_l2 8.2e-5）**
  1. **split-bf16 tensor-core GEMM**：`matmul_t_split_bf16`（激活拆 `hi=bf16(x)`、
     `lo=bf16(x-hi)`，权重本就是精确 bf16，每个权重面板 2 次 mma、f32 累加），
     视觉线性/卷积全用它；`k%8≠0` 回退 `matmul_t_f32w`。GEMM 178 → 98 ms。
  2. **SAM attention relpos 因式分解**：`q·(Rh[ih]+Rw[iw])` 拆成
     `q·Rh[ih] + q·Rw[iw]`，每 query 在 block 起始算好 `H+W` 长度的查找表放 shared，
     内层循环由「每 key 4 次 global load + 两次 warp 归约」变成 2 次 shared 查表；
     另加每 warp 4 query、`float2` shared 读、`__expf`。attention 404 → 239 ms。
- **已完成（2026-09-21）：tensor-core flash attention 调通**
  - **根因是共享内存布局 bug**（不是 mma fragment 语义）：`sVhi/sVlo` 是
    `[head_dim][key]`（64 行）却按 `kTcBN*RS`（32 行）分配/寻址，`d≥32` 越界写进
    `sVlo`/`sPhi`；launcher 的 shared 字节也少算。表现：QK^T/sS/softmax-L 全对、
    仅 PV 错，且结果跨运行不稳定。修复：V 按 `kAttnHD` 行分配，V/P 用窄 stride
    `kTcRS2=kTcBN+8`，launcher 重算（1024 时 89856 B < 101376 上限）。详见
    `CORE_TECH.md` §5.11、`PITFALLS.md` §20。
  - **验收**：`compare_vision_gpu --selftest` 新增 `H=W=32/S=1024`、**输出置零**的
    TC case，rel_l2 **9e-6**，已注册 ctest；真实模型 1024 编码 **238–246 ms**
    （原 366）、visual rel_l2 **1.44e-4**（原 8.2e-5，仍 ≪6e-4）。640 75 ms、
    224 20 ms。
  - **已做（occupancy）**：block 改 8 warp，把 mma 的 n 维对半拆（`kTcSplit=2`），
    每 warp tensor-core 工作量减半、每调度器 2 warp 掩盖延迟；1024 全局 attention
    4 块合计 81 ms。nsys 显示当前最大头已是 **`matmul_t_split_bf16` 98 ms**（视觉
    GEMM），windowed f32 attention 30 ms。
  - **剩余（非阻塞）**：仍 1 block/SM（`sRel` 32 KB）；进一步需 register-resident
    flash attention / 缩减 sRel / 优化视觉 GEMM（cp.async）。整栈 CUDA Graph 捕获、
    非方形尺寸支持未做。
- **环境约束（重要）**：sm_120 的 `cudaDevAttrMaxSharedMemoryPerBlockOptin` 只有
  **101376 B（~99 KB）**、每 SM shared 102400 B；tile shared 超过上限时自动回退
  f32 kernel（正确性不变）。
- **涉及**：`src/kernels/cuda/backend.cu`、`src/kernels/cuda/vision_ops.cu`、
  `src/engine/gpu_encoder.cu`、`tools/compare_vision_gpu.cpp`。详见 `CORE_TECH.md`
  §5.8/§5.11、`BENCHMARKS.md` §2.9、`PITFALLS.md` §20。

### ✅ 2.11 CUDA 端真实 OCR 端到端回归（2026-09-21 完成）

- **已做**：`compare_ocr` 新增 `--gpu`（CUDA 解码器）、`--int4`、`--int4-group N`；
  `Engine` 暴露 `gpu_decoder()`；导出 `ref_ocr` 后跑通 CPU / CUDA-BF16 / CUDA-INT4
  三条路径（命令与数据见 `ALIGNMENT.md` §5.2）。
- **结果**：CUDA/BF16+GPU vision **greedy 24/24**（与 CPU 基线一致）；CUDA/INT4 与
  CPU/INT4 的 logits/token **完全一致**（证明 CUDA 解码器无 bug），但当前
  group-128 非对称 min/max INT4 的 prefill logits rel_l2 ≈ **0.36**、top-1 翻转、
  greedy 0/24。group=32 降到 0.31、top-1 恢复但仍在第 2 步分叉。
- **结论**：差异来自**量化精度**而非移植；`quantize_int4_awq` 实为
  round-to-nearest（未用激活统计），单专家权重误差 ~0.10。真实 OCR 精度待
  OmniDocBench（任务 2.7）评估，或改用真正 AWQ / 更小 group。
- **涉及**：`tools/compare_ocr.cpp`、`include/uocr/engine.h`、`ALIGNMENT.md` §5.2。

> 2.4–2.5 是「能上 GPU 但仍在 CPU」的模型部分（见 §3）；tokenizer、图像预处理、
> 采样/ngram、调度留 CPU 属设计选择。

---

## 3. GPU 卸载现状（2026-09-20 审计）

| 模型/计算部分 | 执行位置 | 备注 |
|---|---|---|
| 文本解码器（12 层 MoE：RMSNorm/QKV/O/RoPE/R-SWA/dense/shared/专家） | **GPU** | `GpuDecoder`，权重常驻 |
| lm_head | **GPU** | 设备副本 + `matvec`/`matmul_t_bf16` |
| MoE router / top-k（decode、prefill） | **GPU** | `moe_router_topk`（INT4 expert 的 ragged 路径也走设备端，无 D2H） |
| MoE 专家 MLP（INT4，ragged prefill） | **GPU** | grouped gate+up+SiLU / down 各一次 launch（§5.9） |
| MoE 专家 MLP（BF16，ragged 大批量 prefill） | **CPU 调度 + GPU GEMM** | 仍 host top-k + 逐专家 GEMM（未加 grouped BF16） |
| token embedding 查表 | **GPU** | embedding 表设备常驻 + `embed_gather`；单请求 prefill 仍 host gather |
| **DeepEncoder 视觉编码器（SAM+CLIP+projector）** | **GPU** | `GpuEncoder`（split-bf16 TC GEMM + vision 内核；SAM global attention 走 **split-bf16 TC flash attention**，小 S/超限时回退 f32）；`set_vision_gpu()` 后启用 |
| 采样 / no-repeat-ngram | CPU | 每步 D2H logits 后采样，属设计选择 |
| tokenizer / 图像预处理 / 调度 | CPU | 设计如此 |

---

## 4. 已完成里程碑（简表）

> 结果为该阶段完成时的快照（非当前最优）；**当前性能与回归见 §1**。
> 历程见 `PROGRESS.md` §4，实现细节见 `CORE_TECH.md`。

| 阶段 | 内容 | 结果（阶段快照） |
|---|---|---|
| M1 调研 | 读参考 modeling，确定 R-SWA 精确语义 | `CORE_TECH.md` §1 |
| M2 骨架 | CMake 双后端、Tensor/ops、config、safetensors | 可编译 |
| M3 调度 | ring KV cache、block manager（引用计数）、memory pool、连续批处理 | `CORE_TECH.md` §1/§4 |
| M4 引擎 | MoEDecoder、DeepEncoder、Sampler、Engine | `CORE_TECH.md` §2 |
| M5 CUDA | R-SWA attn / INT4 GEMM / rmsnorm / rope 内核 + 对照测试 | `CORE_TECH.md` §5 |
| M6 验证 | 真实权重 mmap、量化误差、基准 | `PROGRESS.md` §5、`BENCHMARKS.md` |
| P0 E0 | DeepSeek BPE 预分词复刻 | 43/43 |
| P0 E1 | prompt / `<image>` 布局 | 11/11 |
| P0 E2 | PIL 兼容图像预处理 | ≤1 LSB |
| P0 E3–E4 | 视觉注入 + 首 token logits | visual 4.2%、logits 6.4%、top-1 一致 |
| P0 E5 | `no_repeat_ngram` 采样 | greedy **24/24** |
| P0 Vision | DeepEncoder 分段对齐 | f32 rel_l2 ≤ 5.8e-4；修 2 个 bug |
| P1 路由/attention | bf16 舍入尝试、逐层 q/k/v/o 对比 | 确认 bf16 漂移非 bug |
| P1 R-SWA ring | `--decode-steps 140` 覆盖环形覆写 | final K/V ≤0.03 |
| P1 CUDA 主路径 | device KV + `GpuDecoder` + Engine CUDA 分支 | 显存 9.2GB → 2.2GB |
| P2 CUDA Graph | device router + 固定调度掩码 + 单请求整步图 | TPOT 26.7 → 5.4 ms |
| P2 TC INT4 GEMM | W4A16 `ldmatrix` 路径 | rel_l2 0.0024 |
| P2 连续批处理 | batched attention + ragged prefill + slot 修复 | 298/311 tok/s |
| P2 batched graph | 按行数 B 缓存图，活跃集合变化免重捕获 | graph vs plain rel_l2 = 0 |
| P2 bf16 TC GEMM | dense/shared + lm_head + 小 m 变体 | BF16 batch16 → 415.9 tok/s |
| P2 INT4 专家调优 | BN 模板 + staging 重写 | 整波 prefill −10% |
| P3 TC GEMM 重写 | bf16 cp.async 流水线 + padding；INT4 向量化 staging + bn8/bk64；cp.async 备选变体 | BF16 batch16 416→**522** tok/s、prefill 212→**161** ms；lm_head m≤16 **2.3×**；INT4 batch16→**429** |
| P3 DeepEncoder CUDA | SAM+CLIP+projector 设备内核 + f32 GEMM；`GpuEncoder` + Engine 分派 | 单图 1024 **612 ms**（CPU ~2 min）；visual rel_l2 **1.1e-5** |
| P3 grouped MoE | device router + grouped INT4 gate_up/down（一层 2 次 launch、无 D2H）；tile `bn=32/bm=128` | INT4 整波 prefill 213→**120 ms**、B=16 **~518** tok/s；ragged rel_l2 1.7e-3 vs host |
| P3 device embedding | bf16/f32 表设备常驻 + `embed_gather`；单请求 Graph 内 gather、batch 只传 token id | 每步无隐含 H2D；greedy 不变；显存 +331MB |
| P3 权重加载 | INT4 专家量化改为 OpenMP 并行（`weights.cpp` 专家循环） | 真实模型加载墙钟 **20.5 → 7.4s**；显存不变 |
| P3 视觉编码器 | split-bf16 TC GEMM（激活 hi/lo）+ SAM attention relpos 因式分解 | 1024 编码 **612→366ms**、640 153→95ms；rel_l2 8.2e-5（未达 150–250 目标） |
| P3 视觉编码器（TC attn） | tensor-core flash attention（Q/K/P/V hi/lo 拆分；`attention_flash_tc_kernel`）；修 V^T 面板越界 + V/P 窄 stride；8-warp n-split | 1024 **366→238–246ms**（rel_l2 1.44e-4）；640 75ms |

---

## 5. 已知遗留 / 技术债

- **视觉编码器已进入 150–250ms 目标区间**（1024：238–246ms）。nsys 显示最大头
  已是视觉 GEMM（`matmul_t_split_bf16` 98ms），其次全局 TC attention 81ms、
  windowed f32 attention 30ms；TC attention 受 ~90KB shared → 1 block/SM（`sRel`
  表 32KB）限制。整栈 CUDA Graph 未做。**任意长宽比的输入图片可用**（`ocr_image`
  经 `image_embeddings` 做 aspect-preserving resize + 居中 pad 成正方形后再喂
  编码器）；只有直接调用 `GpuEncoder::encode` 时才要求正方形。见任务 2.10。
- 视觉编码器的 **非 1024 输入**（Gundam 640 局部 crop）现已做 SAM `pos_embed` /
  global `rel_pos` 插值（2.12）；多 crop 端到端已通过临时参考验证（visual rel_l2
  0.057），但**参考回归尚未入库**（任务 2.13）。
- 单请求 `GpuDecoder::prefill_tokens` 仍在 host 查 embedding（每请求一次，非每步）；
  position 仍是 4B pinned H2D（本就是 int，未做成表）。
- BF16 专家的 ragged 大批量 prefill 仍是 host top-k + 逐专家 GEMM（grouped 只做了
  INT4）；`rswa_attn_ragged`（B=16 ~21 ms）是 grouped 之后的第二大头。
- 视觉编码器 f32 GEMM 已由 **split-bf16 tensor-core GEMM** 取代（激活 hi/lo 两片
  补偿误差），视觉栈保持 6e-4；f32 CUDA-core 版本仅作 `k%8≠0` 回退。
- **INT4 量化精度不足**：当前 `quantize_int4_awq` 实为 group-wise RTN（非激活感知
  AWQ），单专家权重 rel-L2 ≈ 0.10、端到端 prefill logits ≈ 0.36，合成用例 top-1
  翻转。CUDA/INT4 与 CPU/INT4 一致（移植无误）。真实 OCR 精度待 OmniDocBench
  （2.7）或真正 AWQ 校准（2.7 消融）。
- `matmul_t_bf16_ref` 仅用于单测 A/B，release 构建保留。

---

## 6. 环境与依赖备注

- GPU：RTX 5060 Ti 16GB（sm_120, 36 SM）；CUDA 13.4；g++ 16.2.1；CMake 4.4.3。
- 唯一第三方 C++ 依赖：系统 `nlohmann/json`（已安装）；未用 `spdlog`（自研 log）。
  OpenMP 用于 CPU 内核（视觉编码器）。
- Python 参考环境在 `.venv`（torch 2.10.0+cu128 / transformers 4.57.1）；
  安装依赖需经代理 `http://192.168.1.164:7897`。
- **新增系统依赖或 Python 包前先询问确认。**

---

## 7. 工作方式（维护约定）

- 每完成一项任务：更新本文件 §1「当前状态」与 §2「下一步任务」；把细节写入
  `CORE_TECH.md`（实现）/ `PITFALLS.md`（坑）/ `BENCHMARKS.md`（数据，原始输出进
  `bench/`）/ `ALIGNMENT.md`（对齐）/ `PROGRESS.md`（历史），跑回归后 `git commit` 并 push。
- 「下一步计划」只以本文件 §2 为准，其他文档不重复维护计划。
- 改了 CUDA 内核/引擎，务必重跑 `ctest --test-dir build-cuda` 与相关 benchmark。
