# Unlimited-OCR 引擎 — 开发进度与历程

> 本文件记录**已完成的内容与历史**（环境、完成度、目录、构建、实测数据、里程碑明细）。
> **下一步计划以开发入口 `tAgent.md` 为准**；文档导航见 `README.md`，原始需求见 `prj.md`。

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB（Blackwell GB206，sm_120，36 SM） |
| CUDA | 13.4（计划要求 ≥ 12.8） |
| 编译器 | g++ 16.2.1（CUDA 主机编译器） |
| CMake | 4.4.3 |
| 依赖 | `nlohmann/json`（系统已安装，唯一第三方依赖；`spdlog` 未安装，改用自研 `uocr/log.h`）；OpenMP（g++ 自带，用于 CPU 内核并行，`find_package(OpenMP)`） |
| 模型 | `baidu/Unlimited-OCR`（经 hf-mirror 下载，6.67 GB safetensors） |

> 说明：`tAgent.md` 要求"需要安装第三方依赖时先询问"。本项目只用到系统已有的
> `nlohmann/json`，未安装任何新依赖。若后续需要 `spdlog`/`gtest` 等会先确认。

## 2. 当前完成度

```
✅ 可编译骨架（CPU + CUDA 双后端，sm_120）
✅ CPU 参考实现（R-SWA / MoE / 调度 / 分词 / 视觉编码器）
✅ CUDA 内核（R-SWA attention / INT4 MoE GEMM / RMSNorm / RoPE）
✅ 单元测试 15/15 通过（CPU），CUDA 测试 2/2 通过
✅ 真实权重 mmap 加载验证
✅ 基准测试可运行
✅ PyTorch 参考环境（.venv, torch 2.10+cu128, transformers 4.57.1）
✅ Decoder 数值对齐（11/12 层 hidden <1%，logits <1%）
✅ Vision 对齐完成（对 f32 参考全链路 rel_l2 ≤ 6e-4；对 bf16 参考 3.2%）
✅ 端到端图像 OCR 对齐（E0–E5：布局、预处理、视觉注入、greedy 24/24）
✅ CUDA 设备端 decoder + Engine CUDA 分支（GpuDecoder，bf16 权重常驻）
✅ Tensor Core W4A16 INT4 MoE GEMM
✅ CUDA Graph 捕获（device router + 全专家固定调度 + 掩码跳过；全图/attn_dense 两种范围）
✅ device 端 INT4 专家权重（显存 9.2GB → 2.2GB）
✅ 合并式 decode 内核（warp-per-output matvec / expert MLP）+ 分块 prefill GEMM
✅ Tensor Core W4A16 `ldmatrix` + shared-memory staging
✅ 连续批处理接入 device decoder（`Engine::generate_batch`，每 slot 独立 R-SWA KV）
✅ batched R-SWA attention/append 内核（每步 attention kernel 数 O(B·L)→O(L)）
✅ prefill 走 device masked MoE（prefill/请求 88→53ms INT4、35ms BF16）
✅ ragged 多请求 prefill（整波一次前向，batch=16 prefill 串行 16×88ms→279ms）
✅ batched decode 纳入 CUDA Graph（按行数缓存图；活跃 slot 变化无需重捕获）
✅ bf16 tensor-core GEMM（dense/shared + lm_head，`ldmatrix`/`mma`）
✅ INT4 专家 TC GEMM staging/bn 调优（整波 prefill −10%）
```

> 性能原始数据与汇总见 `BENCHMARKS.md` 和 `bench/` 目录。

### CUDA Graph 与性能（P2，2026-09-19）

- **device router**：`moe_router_topk` 在设备端做 softmax/top-k + 按专家分组，去掉了每层
  router D2H + host top-k 同步点。
- **全专家固定调度 + 掩码跳过**：`moe_experts_masked` 以 `grid=(n_experts, rows)` 启动，
  每个 block/warp 先读 `count[e]`，无 token 立即返回；网格只依赖静态形状，可被 Graph 捕获。
- **device 环形指针**：`rswa_append_decode` 在设备端推进 `len/ring_pos`，写入槽位在 replay
  时解析；`rswa_attention_devlen` 读取 `*d_len` 做掩码，因此**同一个 Graph 覆盖 warmup→ring**。
- **持久化 buffer**：KV cache、router 分组 buffer、ping/pong、embed/pos pinned staging 全部
  预分配，replay 地址稳定。
- **Graph 范围**：`EngineConfig.graph_scope = "full" | "attn_dense"`。full 捕获整个 decode
  step；attn_dense 只捕获 attention 子图，MoE 在图外发射（消融用）。
- **device INT4 专家权重**：`Linear.weight` 为 INT4 时上传打包 AWQ 权重（连续
  `[n_experts, rows, cols]` 布局），`moe_experts_masked_int4` 内核内反量化。

实测（真实 `baidu/Unlimited-OCR`，prefill=128，batch=1，RTX 5060 Ti）：

| 配置 | TTFT | TPOT(稳态) | 显存 |
|---|---|---|---|
| BF16 + plain（无 Graph，host 路由） | 193 ms | 6.76 ms | 9.2 GB |
| BF16 + full Graph | 192 ms | **5.45 ms** | 9.2 GB |
| INT4 + plain | 127 ms | 14.75 ms | 2.2 GB |
| INT4 + full Graph | 127 ms | **5.43 ms** | 2.2 GB |

> 合并式内核（warp-per-output、向量化合并访存）与 device lm_head 之前，BF16 full Graph 的
> TPOT 为 26.7ms（其中 host lm_head matvec 占 ~20ms）。详见 `PITFALLS.md` §11。

### 端到端 OCR 对齐（E0–E5，2026-09-19）

| 阶段 | 工具 | 结果 |
|---|---|---|
| E0 BPE 预分词 | `compare_tokenizer` | pretok/ids 43/43 |
| E1 prompt/`<image>` 布局 | `compare_layout` | ids/mask 11/11 |
| E2 图像预处理 | `compare_image` | 与 Pillow ≤1 LSB，crop ratio 全对 |
| E3/E4 视觉注入 + 首 token | `compare_ocr` | visual rel_l2 4.2%、logits rel_l2 6.4%、top-1 一致 |
| E5 no-repeat-ngram | `compare_ocr` | greedy 24/24 |

典型结果（500×400，`<image>\nFree OCR.`，crop_mode=True）：
`layout OK → visual 273 tokens, rel_l2 0.0419 → prefill_logits rel_l2 0.0637
(top-1=3051) → greedy 24/24`（`no_repeat_ngram_size=35, ngram_window=1024`）。

## 3. 目录结构

```
unlimited-ocr.cpp/
├── CMakeLists.txt              # 双后端构建（-DENGINE_BACKEND=CPU|CUDA）
├── cmake/CUDAArch.cmake        # sm_120 / CUDA≥12.8 检查
├── include/uocr/               # 公共头文件
│   ├── common.h tensor.h ops.h
│   ├── config.h safetensors.h tokenizer.h quant.h weights.h
│   ├── unicode_tables.h image.h prompt.h
│   ├── kv_cache.h block_manager.h memory_pool.h continuous_batch.h
│   ├── moe_decoder.h deep_encoder.h sampler.h engine.h
│   └── cuda_ops.h
├── src/
│   ├── runtime/    config.cpp safetensors.cpp tokenizer.cpp quant.cpp weights.cpp log.cpp
│   │                prompt.cpp image.cpp
│   ├── scheduler/  kv_cache.cpp block_manager.cpp memory_pool.cpp continuous_batch.cpp
│   ├── engine/     moe_decoder.cpp deep_encoder.cpp sampler.cpp engine.cpp
│   │                gpu_decoder.cu（device decoder + CUDA Graph 捕获）
│   └── kernels/
│       ├── cpu_ops.cpp
│       └── cuda/   rmsnorm.cu rope_fused.cu rswa_attention.cu moe_gemm_int4.cu
│                    backend.cu gpu_ops.cu gpu_cache.cu moe_device.cu
├── tests/          test_main.* test_kv_ring_buffer.cpp test_memory_budget.cpp
│                   test_scheduler.cpp test_decoder.cpp test_moe_gate.cpp
│                   test_tokenizer.cpp test_rswa_cuda.cu
├── benchmarks/     bench_decode.cpp bench_throughput.cpp bench_cuda_decode.cu
│                   bench_cuda_batch.cu bench_int4_gemm.cu bench_bf16_gemm.cu
├── tools/          inspect_model.cpp compare_reference.cpp compare_vision.cpp
│                   compare_tokenizer.cpp compare_layout.cpp compare_image.cpp
│                   compare_ocr.cpp gen_unicode_tables.py
│                   reference/ export_reference.py export_vision_stages.py
│                              export_tokenizer_cases.py export_layout_cases.py
│                              export_image_cases.py
├── .venv/          （gitignore，PyTorch 参考环境）
└── models/         （.gitignore，包含下载的真实权重与远程代码）
```

## 4. 构建与运行

```bash
# CPU 参考后端
cmake -S . -B build -DENGINE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
./build/benchmarks/bench_decode --prefill 128 --steps 64

# CUDA 后端（本机验证通过）
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8
./build-cuda/tests/uocr_cuda_tests

# 真实权重检查
./build/tools/inspect_model --model models --load
./build/tools/inspect_model --model models --quant-check
```

## 5. 实测数据（本机）

### 5.1 真实权重（baidu/Unlimited-OCR）

- checkpoint：2710 个张量，6.21 GiB（BF16）。
- MoE 专家张量 2112 个（3 × 64 experts × 11 MoE 层），与 `config.json` 完全一致。
- AWQ INT4（group=128）在采样专家上 rel-L2 ≈ 0.101，单矩阵打包后 0.55 MB
  （BF16 为 2.19 MB，压缩到 ~25%）。

### 5.2 CPU 参考性能（合成分层模型，仅用于回归，不代表 3B 真机性能）

| 指标 | 值 |
|---|---|
| prefill 128 token | ~387 ms |
| TPOT（首步） | 7.5 ms |
| TPOT（稳态） | 3.7–3.9 ms |
| KV cache（P=128, W=128, 12 层） | 2.00 MB |
| batch=8 合成吞吐 | ~105 tok/s |

> 说明：CPU 参考实现的目的是**数值正确性与调度验证**，不是性能目标。
> 3B 模型 CPU 前向极慢，且 FP32 权重无法放入本机 15 GB 内存，因此基准默认使用
> 合成小模型；`bench_decode --real` 会加载真实 BF16 权重（mmap，按需分页）。

### 5.3 CUDA 内核正确性

| 测试 | 结果 |
|---|---|
| R-SWA decode attention（kv_len=307, heads=10, hd=128） | max_err = 0.000000 |
| `GpuRSWACache` 环形覆写（W=8, P=4, 24 步） | cache K/V err = 0；decode err = 1e-6 |
| `GpuRSWACache` prefill causal attention | max_err = 0.000001 |
| Batched R-SWA（B=3 变长 prefill, W=8, 24 步, 环形覆写，非恒等 slot 排列） | attn err = 0；cache err = 0 |
| ragged 多请求 prefill（3 请求，非恒等 slot）vs 逐请求 prefill logits | rel_l2 = 0 |
| INT4 MoE GEMM 标量（8×64×256, group=128） | max_err = 0.000010 |
| INT4 MoE GEMM 张量核 W4A16（8×64×256） | rel_l2 = 0.0024 |
| INT4 MoE GEMM 张量核 ragged（M=5,K=48） | rel_l2 = 0.0026 |
| `GpuDecoder` vs CPU `MoEDecoder`（tiny，prefill） | rel_l2 = 0.0022 |
| `GpuDecoder` vs CPU（tiny，6 步 decode） | worst rel_l2 = 0.0025 |
| `Engine` CUDA vs CPU（tiny，greedy 4 步） | token 一致 |
| Graph decode vs plain（W=4，20 步，含 ring 覆写） | rel_l2 = 0.00000（完全一致） |
| attn_dense Graph vs plain | rel_l2 = 0.00000 |
| device INT4 专家 vs CPU（plain / graph） | rel_l2 = 0.0025 / 0.0025 |
| Engine batch（4 个不同长度 prompt）vs sequential / CPU | token 完全一致 |

### 5.4 DeepEncoder (Vision) 对齐

`tools/compare_vision` 加载 `models/image.bin`（1024×1024，mean=std=0.5）跑
`DeepEncoder::encode`，输出 `[273,1280]`：

| 对比对象 | max_abs | rel_l2 |
|---|---|---|
| 对 f32 参考（逐段：patch/pos/12 blocks/neck/net2/net3/CLIP×24） | ≤ 2.5e-3 | **≤ 5.8e-4** |
| 对 bf16 参考（真实推理） | 0.226 | **0.0324** |

结论：C++ 的 f32 实现与 f32 参考等价；对 bf16 参考的 3.2% 差异来自参考端
bf16 激活舍入。详见 `ALIGNMENT.md` §4、`PITFALLS.md` §9。CPU 完整编码约
1–2 分钟（OpenMP 8 线程）。

## 6. 里程碑

1. **M1 调研**：从 HuggingFace 下载并阅读 `modeling_unlimitedocr.py` /
   `modeling_deepseekv2.py` / `deepencoder.py`，确定 R-SWA 精确语义（见 `CORE_TECH.md`）。
2. **M2 骨架**：CMake 双后端、Tensor/ops、config、safetensors。
3. **M3 调度**：R-SWA ring KV cache、block manager（前缀引用计数）、memory pool、连续批处理。
4. **M4 引擎**：MoE decoder、DeepEncoder、sampler、Engine。
5. **M5 CUDA**：四个内核 + 对照测试。
6. **M6 文档与验证**：真实权重加载、量化误差、基准、本文档。

## 7. 状态与后续

已完成：P0（E0–E5 端到端 OCR 24/24）、P1（R-SWA 环形覆写、attention 逐层对比、
CUDA 设备端 decoder/Graph）、P2（device router + 固定调度掩码 kernel + CUDA Graph、
device INT4 权重、合并访存内核、连续批处理 + batched attention + ragged prefill、
batched decode CUDA Graph、bf16 tensor-core GEMM + 小 m 变体、INT4 专家 GEMM bn 调优/
staging 重写、指标表、nsys kernel 分解、可选 reference CTest）。
最终回归：`compare_ocr` layout OK、visual rel_l2 0.04187、greedy **24/24**；
`uocr_tests` 20/20；`uocr_cuda_tests` 全过。

### 2026-09-19 第四轮（本轮）

对照 `BENCHMARKS.md` §2.7（真实模型 prompt=64, steps=16, max_batch=16, warmup 稳态）：

| 配置 | 第三轮末 | 本轮末 |
|---|---|---|
| BF16 batch=16 tok/s | 298.0 | **415.9** |
| INT4 batch=16 tok/s | 307.6 | **386.8** |
| BF16 batch=16 整波 prefill | 367 ms | **212 ms** |
| INT4 batch=16 整波 prefill | 279 ms | **204 ms** |
| BF16 / INT4 batch=1 tok/s | 99.4 / 91.4 | **149.4 / 130.9** |

1. **batched decode 纳入 CUDA Graph**：`forward_batch + final rmsnorm` 整步捕获，
   按行数 B 缓存图；行→slot/pos/embedding 写入地址固定的设备 buffer，活跃 slot 变化
   无需重捕获。`batch_configure` 形状不变时复用分配与图。单测 graph vs plain
   logits rel_l2 = 0。`CORE_TECH.md` §5.7、`PITFALLS.md` §14。
2. **bf16 tensor-core GEMM**：`matmul_t_bf16`（dense/shared + lm_head）改
   `ldmatrix`/`mma.m16n8k16`；m<64 越界 warp 跳过 mma；另加 m≤16 的 n 方向拆 warp
   小 m 变体。n=k=1280 最高 18.4 TFLOPS，单测 TC vs CUDA-core ref rel_l2 ≤ 0.0018。
   `CORE_TECH.md` §5.5、`BENCHMARKS.md` §2.7–2.8。
3. **INT4 专家 TC GEMM 调优**：`BN` 模板化 + 扫描，专家形状（N=896/1280, M≈96）
   **bn=8 最优**；反量化 staging 改为 128 线程逐元素后 kernel 快 ~20%，整波 prefill
   −10%。`moe_gemm_int4_tc_n` 可扫 bn。`BENCHMARKS.md` §2.5。
4. **nsys kernel 分解**：prefill 逐专家 `moe_gemm_int4_tc` 占 55% kernel 时间
   （12096 次小 GEMM）；decode masked INT4 专家 ~19%；TC dense ~16%；ragged attention
   7%；graph launch 30 次 vs 普通路径 27584 次 `cudaLaunchKernel`。`BENCHMARKS.md` §2.9。

> **后续计划以 `tAgent.md` 为准**（该文件是开发入口）；本文件只记录"做过什么"。

> GPU 卸载审计（2026-09-19）：文本解码器（含 lm_head、decode router）已在 GPU；
> 仍在 CPU 的模型部分是 **DeepEncoder**、**大批量 prefill 的 MoE 路由**、
> **token embedding 查表**；采样/tokenizer/预处理/调度留 CPU 属设计选择。
> 详见 `tAgent.md` §3。

---

## 8. 历史里程碑明细

> 这些是各阶段的原始 checklist，保留作实现/验证记录；当前计划见 `tAgent.md`。

### P0 视觉编码器数值对齐（2026-09-15）
- [x] `tools/compare_vision.cpp`：`DeepEncoder::encode` 输出 273×1280 与参考
      `visual_embeddings`，对 f32 参考 rel-L2 ≤ 5.8e-4，对 bf16 参考 3.2%。
- [x] 分段对比：`DeepEncoder::encode_stages` + `--dump-stages` +
      `tools/reference/export_vision_stages.py`，定位并修复 SAM 相对位置漏乘 query、
      CLIP 漏加 QKV bias 两个 bug。CPU 侧 OpenMP（8 线程约 2 分钟）。
      详见 `ALIGNMENT.md` §4、`PITFALLS.md` §9。

### P0 端到端图像 OCR 对齐（2026-09-19，E0–E5）
1. [x] **DeepSeek BPE 预分词**（E0）：`tokenizer.json` 的 3 条 `Split` 正则复刻；
     Unicode 类别表 `tools/gen_unicode_tables.py` → `include/uocr/unicode_tables.h`；
     43 个用例 pretok/ids 43/43。
2. [x] **prompt 与 `<image>` 布局**（E1）：`prompt.h`/`prompt.cpp::build_ocr_prompt`；
     11 个用例 ids/mask 11/11（含 crop `[1,1]/[2,1]/[1,2]/[2,2]/[3,2]`）。
3. [x] **图像预处理**（E2）：PIL 兼容 bicubic、`ImageOps.pad`（Python round-half-even
     中心对齐）、`dynamic_preprocess`、`BasicImageTransform`；与 Pillow 逐像素 ≤1 LSB。
4. [x] **Engine 注入视觉 embedding**（E3）：`MoEDecoder::prefill_embeds`、
     `Engine::generate_from_image`、`Engine::image_embeddings`。
5. [x] **端到端参考导出与对比**（E4）：`export_reference.py --mode ocr` +
     `tools/compare_ocr`；500×400 图 visual rel_l2 4.2%、prefill logits 6.4%、
     top-1 一致、greedy 24/24。
6. [x] **采样与输出文本**（E5）：修正 `Sampler::apply_no_repeat_ngram`（匹配 `ngram-1`
     前缀 + 滑动窗口）；24 步 greedy **24/24**。

### P1 降低路由翻转 / R-SWA 环形覆写（2026-09-19）
- [x] MoE gate 前 bf16 舍入尝试（`MoEDecoder::set_bf16_rounding`，默认关）：
      实测 set_mismatch 仍为 13，未降低翻转；翻转源于 f32-vs-bf16 累积漂移，暂缓。
- [x] 逐层对比 attention q/k/v/o（`set_trace_attn` + `compare_reference`）：
      q/k/v/o 分别 ≤0.024/0.023/0.044/0.057，误差随层累积，确认非 bug。
      `ALIGNMENT.md` §7。
- [x] `--decode-steps 140` 覆盖环形覆写：final K/V rel_l2 ≤ 0.03、decode logits ≤ 0.084。
      `ALIGNMENT.md` §6。

### P1 CUDA 主路径接入（2026-09-19）
- [x] `GpuRSWACache` + `cuda::rswa_attention`（causal prefill / decode）；环形覆写
      cache 完全一致、decode/prefill 误差 ≤1e-6。
- [x] `GpuDecoder`：RMSNorm/QKV/O/RoPE/R-SWA/dense/MoE 全放 GPU；权重常驻。
      vs CPU `MoEDecoder`：prefill rel_l2 0.0019、decode 0.0021。
- [x] `Engine(..., Backend::CUDA)` 走 `GpuDecoder`，`generate`/`generate_from_image`
      自动分派；CPU/CUDA greedy token 一致。
- [x] device INT4 专家权重：`upload_expert_table` + `moe_experts_masked_int4`；
      显存 9.2GB → 2.2GB。

### P2 CUDA Graph 与创新点（2026-09-19）
- [x] device `moe_router_topk`；"全 expert 固定调度 + 掩码跳过"（网格静态）。
- [x] 单请求整步 Graph：plain 6.8ms → full graph 5.4ms/token（INT4）。
- [x] `rswa_append_decode` 设备端推进 `len/ring_pos`，同一 Graph 覆盖 warmup→ring。
- [x] `EngineConfig.graph_scope = full | attn_dense`（消融）。
- [x] `batch_prefill_embeds` + `forward_ragged` + `rswa_write_prefill_ragged` /
      `rswa_attention_ragged`；与逐请求 prefill logits rel_l2 = 0。
- [x] **slot 映射修复**：`batch_decode` 显式接收"行→slot"映射（`PITFALLS.md` §13）。
- [x] batched CUDA Graph：按行数 B 缓存图；graph vs plain rel_l2 = 0。

### P2 Tensor Core / 连续批处理（2026-09-19）
- [x] INT4 W4A16 `ldmatrix` 路径；保留标量 `moe_gemm_int4` 做基线；rel_l2 ≤ 0.0026。
      注：`prj.md` 的 `s4.s4.s32`（W4A4）与"激活 BF16 输入"矛盾，最终采用 W4A16。
- [x] `moe_gemm_int4_tc` `BN` 模板 + bn 扫描：专家形状 bn=8 最优；staging 重写
      kernel +20%，整波 prefill −10%。`moe_gemm_int4_tc_n` 可扫 bn。
- [x] bf16 TC GEMM `matmul_t_bf16`（dense/shared + lm_head）：64×64 tile、`ldmatrix`、
      小 m 越界 warp 跳过 mma；小 m 变体（m≤16 沿 n 拆 warp）；`matmul_t_bf16_ref`
      做 A/B，rel_l2 ≤ 0.0018；n=k=1280 最高 18.4 TFLOPS。
- [x] nsys kernel 分解（`BENCHMARKS.md` §2.9）。
- [x] 分词器对齐提前到 E0（43/43）；可选 reference CTest。


