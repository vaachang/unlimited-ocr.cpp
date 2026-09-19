# preDocs 文档导航

本目录是项目的设计、进度与验证文档。**下次继续开发时，先读 `tAgent.md`。**
`prj.md` 是原始需求（只读参考）。

## 文档职责

| 文件 | 作用 | 何时读/写 |
|---|---|---|
| **`tAgent.md`** | **开发入口**：当前状态速览、下一步任务（含验收口径）、GPU 卸载审计、技术债、环境备注 | **每次开发先读；完成任务后更新** |
| `prj.md` | 原始项目需求与架构/创新点/指标定义（需求方文档） | 需要对照需求时 |
| `PROGRESS.md` | 开发历程与完成度、目录结构、构建命令、真实权重数据、历史里程碑明细 | 了解"做过什么/怎么做" |
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
ctest --test-dir build-cuda --output-on-failure          # uocr_tests / uocr_cuda_tests

# CPU 参考构建
cmake -S . -B build -DENGINE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 && ctest --test-dir build --output-on-failure

# 真实模型连续批处理吞吐
./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
```

可选参考对齐 CTest（需先导出参考张量，默认跳过）：

```bash
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DENGINE_REFERENCE_DIR=/path/to/ref
```

## 文档维护约定

- **每完成一项任务**：更新 `tAgent.md` 的「当前状态」与「下一步任务」；细节写入
  `CORE_TECH.md`（实现）/ `PITFALLS.md`（坑）/ `BENCHMARKS.md`（数据）/
  `ALIGNMENT.md`（对齐）/ `PROGRESS.md`（历史），然后 `git commit` 并 push。
- 原始基准输出统一放 `bench/`，正文只放汇总表。
- 不要把"下一步计划"散落到多处，统一以 `tAgent.md` 为准。
