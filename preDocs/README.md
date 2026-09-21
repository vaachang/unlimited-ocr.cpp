# preDocs 文档导航

本目录是项目的设计、进度与验证文档。**下次继续开发时，先读 `tAgent.md`。**
`prj.md` 是原始需求（只读参考）。

## 文档职责

| 文件 | 作用 | 何时读/写 |
|---|---|---|
| **`tAgent.md`** | **开发入口**：当前状态速览、下一步任务（含验收口径）、GPU 卸载审计、技术债、环境备注 | **每次开发先读；完成任务后更新** |
| `prj.md` | 原始项目需求与架构/创新点/指标定义（需求方文档） | 需要对照需求时 |
| `PROGRESS.md` | 开发历程、完成度清单、目录结构、构建命令、模型与权重数据 | 了解"做过什么/怎么做" |
| `CORE_TECH.md` | 核心技术实现说明（R-SWA、MoE、量化、CUDA 内核、Graph、连续批处理等） | 改代码前理解实现 |
| `BENCHMARKS.md` | 所有性能/微基准数据的汇总（原始输出在 `bench/`） | 记录或核对性能 |
| `ALIGNMENT.md` | 与 PyTorch 参考的数值对齐方法与结果 | 关心正确性/对齐时 |
| `PITFALLS.md` | 踩过的坑、隐性 bug、语义陷阱、捕获前提等 | 改相关模块前必读 |
| `bench/` | 基准与对齐工具的原始输出（可复核） | 需要原始数据时 |

## 推荐阅读顺序

首次接手：`tAgent.md`（状态+任务）→ `prj.md`（需求）→ `CORE_TECH.md`（实现）
→ `PROGRESS.md`（历史）→ `PITFALLS.md`（避坑）→ `BENCHMARKS.md` / `ALIGNMENT.md`（数据）。

只想知道"现在怎么样、接下来做什么"：`tAgent.md` 即可。

## 快速上手

```bash
# CUDA 生产构建（本机 sm_120）
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8
ctest --test-dir build-cuda --output-on-failure          # uocr_tests / uocr_cuda_tests / vision selftest

# 模型权重：把 HuggingFace `baidu/Unlimited-OCR` 的
#   model-00001-of-000001.safetensors + config.json + tokenizer.json
# 放到 `models/` 目录（默认 `--model models`）。

# CPU 参考构建
cmake -S . -B build -DENGINE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 && ctest --test-dir build --output-on-failure

# 端到端 OCR：给一张图片输出文本（默认 CUDA + GPU 视觉 + BF16 专家）
./build-cuda/tools/ocr_image --model models --image page.png
# 其他用法：--cpu（纯 CPU 参考路径）、--int4（更省显存但精度下降）、
#           --no-crop-mode、--prompt "<image>\nFree OCR."、--max-new-tokens N
# 输入图片：PNG（需 libpng，系统一般自带）或二进制 P6 PPM。

# 真实模型连续批处理吞吐
./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
```

可选参考对齐 CTest（需先导出参考张量，默认跳过）：

```bash
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DENGINE_REFERENCE_DIR=/path/to/ref
```

## 文档维护约定

见 `tAgent.md` §7（工作方式）。核心原则：**计划只在 `tAgent.md` 维护一处**；每完成
一项更新 `tAgent.md` 并沉淀到对应文档；原始基准输出放 `bench/`；然后 commit + push。
