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
  bf16 TC GEMM）、P3（cp.async 流水线 TC GEMM、DeepEncoder CUDA 移植）均已完成。
  **约 92–95%**（对照 `prj.md`）。
- **回归**：`compare_ocr` greedy **24/24**（CPU 参考路径）；`uocr_tests` **20/20**；
  `uocr_cuda_tests` **全过**（batched 排列 / ragged prefill / slot 复用 / batched graph /
  TC vs ref）。
- **性能**（真实 `baidu/Unlimited-OCR`，prompt=64, steps=16, max_batch=16, warmup 稳态）：

  | 指标 | 值 |
  |---|---|
  | BF16 batch=16 吞吐 | **521.6 tok/s** |
  | INT4 batch=16 吞吐 | **429.3 tok/s**（显存 2.2GB） |
  | 整波 prefill（16 请求） | **161 ms** BF16 / **213 ms** INT4 |
  | batch=1 吞吐 | 155.8 BF16 / 132.9 INT4 tok/s |
  | lm_head m=16（n=129280） | **1174 µs**（原 2652，2.3×） |
  | 单图视觉编码（1024, GPU） | **612 ms**（CPU 参考 ~2 min；rel_l2 1.1e-5） |

- 性能数据明细见 `BENCHMARKS.md` §2.5–2.9；历史进度见 `PROGRESS.md` §4。

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

> **2026-09-20 优先级**：2.4 device router → 2.2 grouped expert GEMM → 2.5 device
> embedding → 2.9 权重加载；2.10/2.11 为已完成项收尾；2.6/2.7 需先确认；2.8 受
> ncu 权限限制。✅ 表示已完成（保留一段背景说明）。

### ✅ 2.1 高性能 TC GEMM 重写（2026-09-20 完成）

- **已做**：`matmul_t_bf16` 换成 cp.async 多级 ring + bank-conflict-free padding 的
  `matmul_t_bf16_tc_pipe_kernel`（m≤16 的 n 拆分小 tile / m>16 的 64×64 tile）；
  INT4 专家 GEMM 激活 staging 向量化并把默认 tile 改为 `bn=8/bk=64`，另加 cp.async
  备选变体。详见 `CORE_TECH.md` §5.5 / §5.5b，数据见 `BENCHMARKS.md` §2.5/§2.7。
- **结果**：lm_head m=8/16 2.3×；BF16 batch=16 416→**522** tok/s、整波 prefill
  212→**161** ms；INT4 batch=16 387→**429** tok/s。
- **剩余**（并入 2.2）：m>16 的 bf16 kernel 每 n-tile 重读 A；`moe_gemm_int4_tc`
  仍是逐专家发射，grouped GEMM 未做。

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

### 2.4 大批量 prefill 的 device router（建议先做，解锁 2.2）

- **问题**：`GpuDecoder::mlp_block(dev_moe=false)` 分支（ragged 大批量 prefill 走此分支）
  每层 router logits **D2H** + host top-k/分组 + 每专家索引 **H2D**，形成同步点，
  阻塞 CUDA Graph 且拖慢整波 prefill。
- **方案**：把 softmax/sigmoid + top-k 下放设备端（复用 `moe_router_topk`），直接产出
  设备端 `(assign_token, assign_w, count)` 分组表；`forward_ragged` 的逐专家循环改为
  读设备分组表（或直接接 2.2 的 grouped GEMM）。注意 `moe_router_topk` 当前假定
  `count` 已清零、`cap` 固定，ragged 下需按 `total`/`top_k` 重新推导容量。
- **验收**：ragged prefill 每层无 D2H/H2D 同步（nsys 上无 `cudaMemcpy` 同步点）；
  logits 与 host 路由路径 rel_l2 ≤ 1e-4；`bench_cuda_batch` 整波 prefill 不退化。
- **涉及**：`src/engine/gpu_decoder.cu`（`mlp_block` / `forward_ragged` / scratch）、
  `src/kernels/cuda/moe_device.cu`。

### 2.2 Grouped expert GEMM（依赖 2.4；prefill 最大剩余项）

- **问题**：`forward_ragged` 每层逐专家发 3×64 个 `moe_gemm_int4_tc`
  （整轮约 12096 次、avg ~50µs），nsys 上占 GPU kernel 时间 >50%。
- **方案**：一层一次 launch：block 映射到 `(expert, m-tile, n-tile)`，设备端按专家
  token 分组表取 A 行；gate/up/down 各一次（或合并 gate+up）。M 很小时回退到现有
  masked matvec。
- **验收**：整波 prefill 明显下降（目标 <150 ms INT4）；每层 `moe_gemm_int4_tc`
  launch 数从 ~192 降到 3；all-close logits（rel_l2 ≤ 2e-3）。
- **涉及**：`src/kernels/cuda/moe_gemm_int4.cu`、`src/engine/gpu_decoder.cu`
  （`mlp_block` / `forward_ragged`）。

### 2.5 device embedding / position 查表

- **问题**：每步 `host_weights_->embed_tokens.row()` 在 host 查表再 H2D
  （`h_embed_pinned_`），graph replay 前需写 pinned staging。
- **方案**：embedding 表常驻设备（bf16/f32），token id 在设备端 gather；position
  表同样驻留。注意 vocab 129280×1280 的 bf16 表约 331MB，需纳入显存预算（INT4
  路径当前 2.2GB，可接受）。
- **验收**：每步无 embedding H2D；batch 越大收益越明显；greedy token 不变。
- **涉及**：`src/engine/gpu_decoder.cu`（`decode_token`/`batch_decode`/`run_graph_decode`）。

### 2.9 权重加载优化（prj.md 6.1）

- **问题**：nsys 显示权重上传（2.08GB H2D）占 host API 76%，首次加载慢。
- **方案**：`mmap` + `cudaHostRegister` pinned 直通，或 `cudaMemcpyAsync` 分块流水
  上传；INT4 量化与上传重叠。
- **验收**：模型加载墙钟明显下降；显存不变。
- **涉及**：`src/runtime/weights.cpp`、`src/engine/gpu_decoder.cu`、`GpuEncoder` 构造。

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
| MoE router / top-k（decode、小批量 prefill） | **GPU** | `moe_router_topk` |
| MoE router / top-k（ragged 大批量 prefill） | **CPU** | 每层 router D2H + host top-k（任务 2.4） |
| token embedding 查表 | **CPU** | host gather + 每步 H2D（任务 2.5） |
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

---

## 5. 已知遗留 / 技术债

- `GpuDecoder::mlp_block` 的 host 路由分支（ragged 大批量 prefill）每层 router D2H +
  host top-k；见任务 2.4。
- **embedding 查表在 host**（lm_head 已在设备）；见任务 2.5。
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
