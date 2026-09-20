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

## 1. 当前状态（2026-09-20 快照）

- **后端**：CPU 参考（OpenMP）+ CUDA 生产（sm_120），`-DENGINE_BACKEND=CPU|CUDA` 双构建。
- **完成度**：M1–M6、P0（端到端 OCR 对齐 E0–E5）、P1（R-SWA 环形覆写 / CUDA 设备端
  decoder）、P2（TC INT4 GEMM、CUDA Graph、device INT4 权重、连续批处理、batched graph、
  bf16 TC GEMM）、P3（cp.async 流水线 TC GEMM、DeepEncoder CUDA 移植、**grouped INT4
  专家 GEMM + device router**、**device embedding 查表**）均已完成。**约 95–97%**
  （对照 `prj.md`）。
- **回归**：`compare_ocr` greedy **24/24**（CPU 参考路径）；`uocr_tests` **20/20**；
  `uocr_cuda_tests` **全过**（batched 排列 / ragged prefill / **grouped INT4 ragged
  prefill** / slot 复用 / batched graph / TC vs ref）。
- **性能**（真实 `baidu/Unlimited-OCR`，prompt=64, steps=16, max_batch=16, warmup 稳态）：

  | 指标 | 值 |
  |---|---|
  | BF16 batch=16 吞吐 | **~520 tok/s**（显存峰值 10.4GB） |
  | INT4 batch=16 吞吐 | **~518 tok/s**（显存峰值 3.37GB，含 331MB bf16 embedding 表） |
  | 整波 prefill（16 请求） | **~165 ms** BF16 / **120 ms** INT4 |
  | batch=1 吞吐 | 155.8 BF16 / 138.2 INT4 tok/s |
  | lm_head m=16（n=129280） | **1174 µs**（原 2652，2.3×） |
  | 单图视觉编码（1024, GPU） | **612 ms**（CPU 参考 ~2 min；rel_l2 1.1e-5） |

- 性能数据明细见 `BENCHMARKS.md` §2.5–2.10；历史进度见 `PROGRESS.md` §4。

**快速验证**：

```bash
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8
ctest --test-dir build-cuda --output-on-failure
./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
```

---

## 2. 下一步任务（开发从这里开始）

按收益/成本排序。每项给出目标、方案、验收与涉及文件；完成后在本文档勾掉并更新 §1。

> **2026-09-20 优先级**：2.4 device router + 2.2 grouped expert GEMM、2.5 device
> embedding、2.9 权重加载均已完成（见下）→ 2.10 视觉编码器性能收尾 / 2.11 CUDA 端
> OCR 端到端回归（2.11 需先导出参考张量）；2.6/2.7 需先确认；2.8 受 ncu 权限限制。
> ✅ 表示已完成（保留一段背景说明）。

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

### 2.6 Prefill KV 分区写入优化（prj.md 创新点三）

- **方案**：prefill 时按位置分区（视觉区/环形区/gap 丢弃），省 ~70% KV 写入带宽。
- **注意**：参考实现并不丢弃 gap（`PITFALLS.md` §1），改动会偏离参考数值，**先确认是否接受**。
- **涉及**：`src/kernels/cuda/rswa_attention.cu`（写出逻辑）、`GpuRSWACache`。

### 2.7 精度评测

- **方案**：接入 OmniDocBench v1.6（AWQ vs BF16 综合分）、AWQ vs 朴素 INT4 消融。
- **注意**：需下载外部工具链/数据，**先询问确认**。

### 2.10 视觉编码器性能收尾（P3 遗留）

- **现状**：`GpuEncoder` 1024 输入 612 ms，其中 f32 tiled GEMM 约 1.8 TFLOPS，
  是主要成本；权重 bf16、激活/累加 f32。
- **方案**：① bf16 tensor-core + 误差补偿（A 拆成 hi/lo 两片 bf16 做 2~3 次 mma，
  保持 ~16-bit 有效尾数）；② CUDA Graph 捕获整个视觉栈；③ 支持任意非方形尺寸。
- **验收**：1024 编码降到 ~150–250 ms；rel_l2 仍 ≤6e-4；`compare_vision_gpu` 通过。
- **涉及**：`src/engine/gpu_encoder.cu`、`src/kernels/cuda/vision_ops.cu`、
  `src/kernels/cuda/backend.cu`。

### 2.11 CUDA 端真实 INT4 OCR 端到端回归（技术债）

- **问题**：当前 OCR 对齐（greedy 24/24）走 CPU 参考路径；CUDA 后端（`GpuDecoder`
  + `GpuEncoder`）尚未与参考做端到端 greedy 回归。
- **方案**：导出/复用 `ref_ocr`，用 `compare_ocr --gpu-vision` + CUDA 解码器跑
  teacher-forced greedy，对比 token 与 logits。
- **注意**：需要 `ref_ocr` 参考张量（当前仓库无，需重新导出，**先确认**）。
- **涉及**：`tools/compare_ocr.cpp`、`tools/reference/export_reference.py`。

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
| **DeepEncoder 视觉编码器（SAM+CLIP+projector）** | **GPU** | `GpuEncoder`（f32 GEMM + vision 内核）；`set_vision_gpu()` 后启用 |
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

---

## 5. 已知遗留 / 技术债

- 单请求 `GpuDecoder::prefill_tokens` 仍在 host 查 embedding（每请求一次，非每步）；
  position 仍是 4B pinned H2D（本就是 int，未做成表）。
- BF16 专家的 ragged 大批量 prefill 仍是 host top-k + 逐专家 GEMM（grouped 只做了
  INT4）；`rswa_attn_ragged`（B=16 ~21 ms）是 grouped 之后的第二大头。
- 视觉编码器 f32 GEMM 目前是 CUDA-core tiled（~1.8 TFLOPS）；bf16 tensor-core 需
  误差补偿才能在视觉栈里保持 6e-4，尚未做。
- `GpuDecoder::mlp_block_batch` 为未定义的空声明，可删除。
- `GpuDecoder::batch_import_prefill` 已无调用者，可删除。
- 真实 INT4 模型下 CUDA 端 greedy 尚未与参考 OCR 做端到端回归（当前 OCR 对齐走
  CPU 参考路径）。
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
