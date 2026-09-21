# Unlimited-OCR 引擎 — 开发进度与历程

> **开发入口见 `tAgent.md`**；文档导航见 `README.md`。本文件只记录"做过什么"。
> 性能数据见 `BENCHMARKS.md`，数值对齐见 `ALIGNMENT.md`，实现细节见 `CORE_TECH.md`，
> 踩坑见 `PITFALLS.md`，原始需求见 `prj.md`。

## 1. 构建与运行

```bash
# CUDA 生产后端（本机 sm_120，已验证）
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8
ctest --test-dir build-cuda --output-on-failure      # uocr_tests + uocr_cuda_tests

# CPU 参考后端
cmake -S . -B build -DENGINE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
./build/benchmarks/bench_decode --prefill 128 --steps 64

# 真实权重检查 / 量化误差
./build/tools/inspect_model --model models --load
./build/tools/inspect_model --model models --quant-check
```

可选参考对齐 CTest（需先导出参考张量，默认跳过）：

```bash
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DENGINE_REFERENCE_DIR=/path/to/ref
```

## 2. 目录结构

```
unlimited-ocr.cpp/
├── CMakeLists.txt              # 双后端构建（-DENGINE_BACKEND=CPU|CUDA）
├── cmake/CUDAArch.cmake        # sm_120 / CUDA≥12.8 检查
├── include/uocr/               # 公共头文件（common/tensor/ops、config/safetensors/
│   │                           #   tokenizer/quant/weights、image/prompt、kv_cache/
│   │                           #   block_manager/memory_pool/continuous_batch、
│   │                           #   moe_decoder/deep_encoder/sampler/engine、
│   │                           #   gpu_cache/gpu_decoder/cuda_ops 等）
├── src/
│   ├── runtime/    config safetensors tokenizer quant weights log prompt image
│   ├── scheduler/  kv_cache block_manager memory_pool continuous_batch
│   ├── engine/     moe_decoder deep_encoder sampler engine
│   │                gpu_decoder.cu（device decoder + CUDA Graph）
│   └── kernels/    cpu_ops.cpp
│                   cuda/ rmsnorm rope_fused rswa_attention moe_gemm_int4
│                         backend gpu_ops gpu_cache moe_device
├── tests/          CPU 单测 + test_rswa_cuda.cu（CUDA）
├── benchmarks/     bench_decode bench_throughput bench_cuda_decode
│                   bench_cuda_batch bench_int4_gemm bench_bf16_gemm
├── tools/          inspect_model + compare_{reference,vision,tokenizer,layout,image,ocr}
│                   gen_unicode_tables.py
│                   reference/ export_{reference,vision_stages,tokenizer_cases,
│                              layout_cases,image_cases}.py
├── .venv/          （gitignore，PyTorch 参考环境）
└── models/         （gitignore，真实权重）
```

## 3. 完成度

```
✅ 可编译骨架（CPU + CUDA 双后端，sm_120）
✅ CPU 参考实现（R-SWA / MoE / 调度 / 分词 / 视觉编码器）
✅ CUDA 内核（R-SWA attention / INT4 MoE GEMM / RMSNorm / RoPE）
✅ 单元测试 20/20（CPU）；uocr_cuda_tests 全过
✅ 真实权重 mmap 加载验证
✅ 基准测试可运行
✅ PyTorch 参考环境（.venv, torch 2.10+cu128, transformers 4.57.1）
✅ Decoder 数值对齐（<1%）
✅ Vision 对齐（对 f32 参考 ≤ 6e-4）
✅ 端到端图像 OCR 对齐（E0–E5，greedy 24/24）
✅ CUDA device decoder + Engine CUDA 分支（GpuDecoder，权重常驻）
✅ Tensor Core W4A16 INT4 MoE GEMM（ldmatrix + bn 调优 + staging 重写）
✅ CUDA Graph（单请求整步 + batched decode，按行数缓存）
✅ device INT4 专家权重（显存 9.2GB → 2.2GB）
✅ 合并访存内核 + 分块/TC prefill GEMM
✅ 连续批处理（batched attention + ragged 多请求 prefill + slot 映射修复）
✅ bf16 tensor-core GEMM（dense/shared + lm_head + 小 m 变体）
✅ cp.async 流水线 TC GEMM（bf16 + INT4 专家，bank-conflict-free padding；P3）
✅ DeepEncoder CUDA 移植（SAM+CLIP+projector 设备内核 + f32 GEMM；P3）
✅ ragged prefill device router + grouped INT4 专家 GEMM（一层 2 次 launch、无 D2H；P3）
✅ device embedding 查表（表设备常驻 + `embed_gather`，每步无 embedding H2D；P3）
✅ 权重加载：INT4 量化 OpenMP 并行（真实模型加载 20.5→7.4s；P3）
```

## 4. 开发历程

按时间顺序的轮次记录；括号内为主要产出文档。

| 时间 | 内容 | 结果 |
|---|---|---|
| 2026-09-15 | M1–M6：调研/骨架/调度/引擎/CUDA 内核/文档验证；DeepEncoder 视觉对齐 | 见 `CORE_TECH.md`、`ALIGNMENT.md` §4 |
| 2026-09-19 P0 | 端到端图像 OCR 对齐 E0–E5（分词 / 布局 / 预处理 / 视觉注入 / 采样） | greedy **24/24**，`ALIGNMENT.md` §5 |
| 2026-09-19 P1 | R-SWA 环形覆写验证、逐层 attention q/k/v/o 对比、CUDA 设备端 decoder | `ALIGNMENT.md` §6/§7、`CORE_TECH.md` §5.1 |
| 2026-09-19 P2-① | device router + 固定调度掩码 + 单请求 CUDA Graph + device INT4 专家权重 | TPOT 26.7→5.4ms、显存 2.2GB，`CORE_TECH.md` §5.2/§5.3 |
| 2026-09-19 P2-② | 合并访存内核 + INT4 TC `ldmatrix` + 分块 prefill GEMM | `CORE_TECH.md` §5.4/§5.3、`BENCHMARKS.md` §2.5 |
| 2026-09-19 P2-③ | 连续批处理接入 device decoder、batched attention、ragged prefill、slot 修复 | `CORE_TECH.md` §5.6、`BENCHMARKS.md` §2.6 |
| 2026-09-19 P2-④ | batched decode CUDA Graph、bf16 TC GEMM（+小 m 变体）、INT4 专家 GEMM 调优、nsys 分解 | `CORE_TECH.md` §5.5/§5.7、`BENCHMARKS.md` §2.5/§2.6–2.8 |
| 2026-09-20 P3 | cp.async 流水线 bf16 TC GEMM（padding 去 bank 冲突）、INT4 专家 GEMM 向量化 staging + `bn=8/bk=64`、cp.async 备选变体 | `CORE_TECH.md` §5.5/§5.5b、`BENCHMARKS.md` §2.5/§2.7、`PITFALLS.md` §16/§17 |
| 2026-09-20 P3 | DeepEncoder CUDA 移植（`GpuEncoder` + vision 内核 + f32 tiled GEMM），Engine `set_vision_gpu` 分派 | `CORE_TECH.md` §5.8、`BENCHMARKS.md` §2.9、`PITFALLS.md` §18、`tools/compare_vision_gpu` |
| 2026-09-20 P3 | ragged prefill 的 device router + grouped INT4 专家 GEMM（一层 2 次 launch、无 D2H；`bn=32/bm=128`） | `CORE_TECH.md` §5.9、`BENCHMARKS.md` §2.10、`PITFALLS.md` §19；INT4 整波 prefill 213→120 ms |
| 2026-09-20 P3 | device embedding 查表（表设备常驻 bf16/f32 + `embed_gather`；单请求 Graph 内 gather、batch 只传 token id） | `CORE_TECH.md` §5.10、`BENCHMARKS.md` §2.11；每步无 embedding H2D，greedy 不变 |
| 2026-09-20 P3 | 权重加载：INT4 专家量化 OpenMP 并行化（量化是加载墙钟主要成本，非 H2D） | `CORE_TECH.md` §5.11、`BENCHMARKS.md` §2.12；加载 20.5→7.4s |
| 2026-09-20 P3 | 视觉编码器性能：split-bf16 TC GEMM（激活 hi/lo 补偿）+ SAM attention relpos 因式分解 | `CORE_TECH.md` §5.8、`BENCHMARKS.md` §2.9；1024 编码 612→366ms，rel_l2 8.2e-5（未达 150–250 目标） |
| 2026-09-21 P3 | 视觉 tensor-core flash attention（WIP）：Q/K/P/V hi/lo 拆分 + shared 分数/softmax，~250ms 但数值错误，dispatch 关闭 | `CORE_TECH.md` §5.11、`PITFALLS.md` §20；下一步调试见 `tAgent.md` 2.10 |
| 2026-09-21 P3 | 修通视觉 TC attention：真因是 V^T 面板按错误行数分配/寻址（越界写坏 sV/sPhi），改为按 `kAttnHD` 分配 + V/P 窄 stride `kTcRS2`；selftest 加输出置零的 S=1024 case 并注册 ctest | `CORE_TECH.md` §5.11、`PITFALLS.md` §20、`BENCHMARKS.md` §2.9；1024 编码 366→248–254ms，rel_l2 1.44e-4；640 95→75ms |
| 2026-09-21 P3 | TC attention occupancy：block 8 warp、mma n 维对半拆（`kTcSplit=2`），每调度器 2 warp 掩盖延迟；nsys 复核（GEMM 98ms 成最大头、TC attention 81ms、windowed f32 30ms） | `CORE_TECH.md` §5.11、`BENCHMARKS.md` §2.9；1024 编码 248→238–246ms，rel_l2 不变 |
| 2026-09-21 P3 | CUDA/INT4 端到端 OCR 回归（2.11）：`compare_ocr --gpu/--int4`；CUDA/BF16 24/24、CUDA/INT4 与 CPU/INT4 一致但量化精度不足 | `ALIGNMENT.md` §5.2、`tAgent.md` 2.11 |
| 2026-09-21 P3 | 2.6 分析：参考 decode 在完整 P+W 上 attention、无 window mask，且本负载 V+W>P，无可丢弃 gap → 判定不实现 | `tAgent.md` 2.6、`CORE_TECH.md` §6 |
| 2026-09-21 P3 | 2.7 量化消融：`inspect_model --quant-check` 扩展为 scheme×group 扫描；OmniDocBench 因缺外部工具链未接入 | `BENCHMARKS.md` §2.13、`tAgent.md` 2.7 |
| 2026-09-21 P3 | 可用性收尾：独立 OCR CLI `tools/ocr_image`（PNG/PPM → 文本，默认 CUDA+GPU 视觉+BF16）；`EngineConfig::use_int4_experts` 默认改 BF16；`generate_from_image` 单测 | `tools/ocr_image.cpp`、`include/uocr/image.h`；单元测试 20→21 |

> 每一步的实现/坑/数据分别沉淀在 `CORE_TECH.md` / `PITFALLS.md` / `BENCHMARKS.md`；
> 当前性能与回归见 `tAgent.md` §1。

## 5. 模型与权重（真实 `baidu/Unlimited-OCR`）

| 项 | 值 |
|---|---|
| config | hidden 1280、intermediate 6848、moe_intermediate 896、12 层、heads 10、64 路由专家、top-6、shared 2、首层 dense、vocab 129280 |
| checkpoint | 2710 个张量，6.21 GiB（BF16）；MoE 专家张量 2112 个（3 × 64 × 11 层） |
| AWQ INT4（group=128） | 采样专家 rel-L2 ≈ 0.101；单矩阵 0.55 MB（BF16 2.19 MB，约 25%） |
| INT4 消融（group 32/64/128/256） | 非对称 RTN：0.081/0.091/0.101/0.110；对称：0.097/0.108/0.118/0.127（`BENCHMARKS.md` §2.13） |
| 显存（CUDA，INT4） | 权重常驻约 2.2 GB；batch=16 峰值约 3.37 GB（含 331MB 设备 embedding 表） |
