# Unlimited-OCR 引擎 — 开发进度

> 本文件记录实现过程、已完成模块、当前状态与后续计划。对应任务见 `preDocs/tAgent.md`。

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
⬜ CUDA Graph 捕获（当前为普通 kernel 启动）
⬜ CUDA 版 MoE/decoder 调度接入（当前 CUDA 仅提供内核与测试，主推理走 CPU）
```

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
│   └── kernels/
│       ├── cpu_ops.cpp
│       └── cuda/   rmsnorm.cu rope_fused.cu rswa_attention.cu moe_gemm_int4.cu backend.cu
├── tests/          test_main.* test_kv_ring_buffer.cpp test_memory_budget.cpp
│                   test_scheduler.cpp test_decoder.cpp test_moe_gate.cpp
│                   test_tokenizer.cpp test_rswa_cuda.cu
├── benchmarks/     bench_decode.cpp bench_throughput.cpp
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
| INT4 MoE GEMM（8×64×256, group=128） | max_err = 0.000010 |

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

## 7. 后续计划

> **P0 端到端图像 OCR 对齐（E0–E5）已于 2026-09-19 完成**，详见上文与
> `tAgent.md`。下一步按 `tAgent.md` 的 P1/P2 推进。

1. **P1 降低路由翻转**：MoE gate 前把激活按 bf16 舍入以复刻参考数值；
   逐层对比 attention q/k/v、O 投影；用 `--decode-steps 140` 覆盖环形覆写。
2. **P1 CUDA 主路径**：把 `MoEDecoder` 的 `Linear::forward` / attention 分派到
   `uocr::cuda::*`，实现 device 上的 KV cache（当前 `RSWACache` 为 host 内存）。
3. **P2 CUDA Graph**：按 prj.md 的"路由在 Graph 外、expert 计算在 Graph 内全调度 + 掩码跳过"
   方案捕获解码稳态。
4. **P2 Tensor Core INT4 GEMM**：把 `moe_gemm_int4.cu` 的标量版替换为
   `mma.sync.aligned.m16n8k32.s4.s4.s32`。
5. **性能记录表**：补齐 `prj.md` 第 7.2 节所有指标（设备端 TTFT/TPOT/吞吐/显存）。
