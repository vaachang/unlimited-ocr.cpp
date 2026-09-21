# Unlimited-OCR 原生推理引擎（C++17 / CUDA）

面向百度 **Unlimited-OCR**（3B，MoE + R-SWA + DeepEncoder）的 C++17/CUDA 原生推理引擎：
从零实现 R-SWA 注意力、AWQ/INT4 MoE 解码、DeepEncoder 视觉编码与连续批处理调度，
**不依赖 Python 运行时**。目标硬件为消费级 Blackwell（RTX 5060 Ti 16GB，sm_120）。

- 原始需求：`preDocs/prj.md`
- **开发入口 / 当前状态 / 下一步任务：`preDocs/tAgent.md`**
- 文档导航：`preDocs/README.md`

## 特性

| 模块 | 说明 |
|---|---|
| R-SWA 注意力 | 固定视觉区 + 环形滑动窗口 KV，CPU 参考内核与 CUDA 内核 |
| MoE 解码器 | 64 专家 top-6 + shared，BF16 或 INT4（W4A16，`mma.sync`/`ldmatrix`） |
| DeepEncoder | SAM-ViT + CLIP-L + projector，CPU 与 CUDA 实现 |
| 调度 | 视觉前缀引用计数共享、环形 buffer、连续批处理（ragged prefill） |
| 运行时 | mmap 权重、纯 C++ BPE tokenizer、PNG/PPM 图像加载 |
| 加速 | tensor-core GEMM、device router + grouped INT4 GEMM、CUDA Graph、device embedding |

## 构建

需要 CMake ≥ 3.24、C++17、g++；可选 CUDA ≥ 12.8（sm_120）与系统 `nlohmann/json`、`libpng`。
CUDA 生产构建：

```bash
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8
ctest --test-dir build-cuda --output-on-failure
```

CPU 参考构建（仅 OpenMP）：

```bash
cmake -S . -B build -DENGINE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 && ctest --test-dir build --output-on-failure
```

## 模型权重

把 HuggingFace [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) 的
以下文件放到 `models/`（默认 `--model models`，目录已在 `.gitignore` 中）：

```
models/
├── model-00001-of-000001.safetensors
├── config.json
├── tokenizer.json
└── tokenizer_config.json
```

## 用法

端到端 OCR：给一张图片，输出识别文本（默认 CUDA + GPU DeepEncoder + BF16 专家）：

```bash
./build-cuda/tools/ocr_image --model models --image page.png
```

可选参数：`--cpu`（纯 CPU 参考路径）、`--cpu-vision`（CUDA 解码 + CPU 视觉）、
`--int4`（更省显存、但当前 group-128 RTN 精度下降）、`--no-crop-mode`、
`--prompt "<image>\nFree OCR."`、`--max-new-tokens N`。图片支持 PNG（需 libpng）或 P6 PPM；
>640px 的图片默认走 Gundam 动态切图。

真实模型连续批处理吞吐：

```bash
./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
```

## 与 PyTorch 参考对齐

参考张量由 `tools/reference/export_*.py`（`.venv` 中的 PyTorch）导出，随后作为 CTest
注册（默认跳过，除非提供 `ENGINE_REFERENCE_DIR`）：

```bash
.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_decoder --mode decoder --seq 16 --decode-steps 8
.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_ocr --mode ocr --ocr-width 500 --ocr-height 400 --decode-steps 24
# multi-crop path (>640px -> Gundam dynamic grid)
.venv/bin/python tools/reference/export_reference.py --model models \
    --out /tmp/opencode/ref_ocr_large --mode ocr --ocr-width 800 --ocr-height 400 --decode-steps 16
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA \
    -DENGINE_REFERENCE_DIR=/tmp/opencode -DCMAKE_BUILD_TYPE=Release
```

`compare_ocr --strict` 会用 `ref_ocr`（1×1 crop）与 `ref_ocr_large`（>640px 多 crop）
校验布局、视觉 embedding、prefill logits 与 greedy 解码。

## 目录结构

```
CMakeLists.txt          # 双后端构建（-DENGINE_BACKEND=CPU|CUDA）
cmake/CUDAArch.cmake    # sm_120 / CUDA≥12.8 检查
include/uocr/           # 公共头文件
src/runtime/            # config / safetensors / tokenizer / quant / weights / image / prompt
src/scheduler/          # kv_cache / block_manager / memory_pool / continuous_batch
src/engine/             # moe_decoder / deep_encoder / sampler / engine（+ gpu_*.cu）
src/kernels/            # cpu_ops.cpp + cuda/ 内核
tests/                  # CPU 单测 + CUDA 内核测试
benchmarks/             # 微基准与端到端吞吐
tools/                  # inspect_model / compare_* / ocr_image + reference 导出脚本
preDocs/                # 设计、进度、性能、对齐、踩坑文档
```
