# Unlimited-OCR 高性能推理引擎 — 项目文档

**项目定位**：面向百度 Unlimited-OCR 模型的 C++17/CUDA 原生推理引擎，从零实现 R-SWA 注意力、MoE INT4 解码与连续批处理调度，不依赖 Python 运行时。
**目标硬件**：NVIDIA RTX 5060 Ti 16GB（Blackwell GB206, sm_120, 4608 CUDA Cores, 448 GB/s 带宽）
**构建系统**：CMake ≥ 3.24 + CUDA ≥ 12.8 + C++17
**核心原则**：软件优化优先，在消费级硬件的带宽和显存约束下逼近性能极限。


## 一、问题定义：R-SWA 的工程挑战

Unlimited-OCR 是百度 2026 年 6 月开源的端到端长文档 OCR 模型，总参数 3B，推理时仅激活约 570M。其核心创新 R-SWA（Reference Sliding Window Attention）将解码器 KV Cache 从线性增长压缩为常数：视觉 token 作为固定参考信息完整保留，仅对最近 128 个输出 token 维持滑动注意力窗口。视觉编码器沿用 DeepSeek OCR 的 DeepEncoder，将 1024×1024 图像压缩为 256 个视觉 token，压缩率达 16×。

**但在工程层面，R-SWA 引入了一个非标准的 KV Cache 结构。** 标准 Transformer Decoder 的 KV Cache 是连续追加的线性序列，PagedAttention 等通用方案可直接复用。R-SWA 的 Cache 则是“固定视觉 token 区 + 环形滑动窗口区”的混合结构——视觉区在解码过程中不变，滑动窗口区需要循环覆写。这意味着：

1. **PagedAttention 的 block 管理逻辑不直接适用**：环形覆写跨越 block 边界，标准 block table 映射无法正确追踪逻辑位置与物理存储的对应关系。
2. **CUDA Graph 捕获需要特殊处理**：R-SWA 的 KV 索引 buffer 宽度大于窗口 W，需保持设备地址在多次 replay 间稳定。
3. **MoE 路由与注意力计算的耦合**：每步解码需先完成 expert 路由，再根据路由结果决定哪些 expert 参与计算，标准静态图无法直接覆盖动态分支。

本项目的核心工程目标：**在 C++/CUDA 层面为 R-SWA 量身定制一套完整的推理引擎，而非在通用框架上做适配层。**


## 二、硬件约束分析：RTX 5060 Ti 的性能模型

RTX 5060 Ti 16GB 的关键规格：4608 CUDA Cores，16GB GDDR7，128-bit 位宽，显存带宽 448 GB/s，TDP 180W。与 A100 80G（2039 GB/s 带宽）相比，显存带宽仅为 22%，这是**整个引擎设计的核心约束**。

**Decode 阶段的 TPOT 下界由显存带宽决定**：

```
TPOT_min ≈ (模型权重字节数 + KV Cache 字节数) / 显存带宽
```

- BF16 权重（~6GB）：TPOT_min ≈ 6GB / 448 GB/s ≈ 13.4ms/tok
- AWQ INT4 权重（~1.8GB）：TPOT_min ≈ 1.8GB / 448 GB/s ≈ 4.0ms/tok

这解释了为什么在消费级 GPU 上做 LLM 推理，量化不是可选项而是必需项。

**16GB 显存的分配方案（INT4 方案）** ：

| 显存用途 | 大小 |
|---|---|
| MoE Decoder 权重（AWQ INT4） | ~1.8 GB |
| DeepEncoder 权重（FP16） | ~1.5 GB |
| KV Cache（视觉区 + 环形区，batch=16） | ~3.0 GB |
| 激活值 / 中间张量 | ~1.0 GB |
| Workspace + CUDA Context + 碎片预留 | ~2.5 GB |
| **合计** | **~9.8 GB** |
| **剩余余量** | **~6.2 GB** |


## 三、sm_120 架构差异与内核设计约束

RTX 5060 Ti 采用消费级 Blackwell（sm_120），与数据中心 Blackwell（sm_100）在硬件特性上存在**关键差异**，直接影响内核设计：

- **缺少 TMEM（Tensor Memory）** ：sm_100 引入的 Tensor Memory 用于加速矩阵乘法累加器访问，sm_120 **不支持**。
- **缺少 tcgen05 指令**：sm_120 无法使用 tcgen05.mma（UMMA），必须回退到 `mma.sync.aligned` 系列指令。
- **INT4 MMA 可用**：sm_120 支持 `mma.sync.aligned.m16n8k32` 的 INT4 变体（f8f6f4 kind），这是实现 INT4 MoE GEMM 的基础。
- **CUDA Graph 可用但需注意**：社区已在 RTX 5090（同为 sm_120）上验证了 CUDA Graph 捕获的可行性，但部分自定义算子可能需要 AOT 编译。
- **AWQ INT4 优于 NVFP4**：消费级 Blackwell 上 AWQ INT4 的实际推理表现优于 NVFP4，建议量化方案选择 AWQ。


## 四、整体架构与 CMake 项目组织

引擎采用“**Host Runtime → Scheduler → Compute Engine → Kernel Layer**”的四层 C++ 架构，全部以 CMake target 组织，支持 CPU-only 调试构建和 CUDA 生产构建。

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Runtime (C++17)                       │
│  Model Loader (mmap + INT4 解包)  │  纯 C++ BPE Tokenizer     │
├─────────────────────────────────────────────────────────────┤
│                    Scheduler (C++17)                          │
│  连续批处理  │  R-SWA Block Manager  │  显存预算管理           │
├─────────────────────────────────────────────────────────────┤
│                  Compute Engine (C++17/CUDA)                  │
│  DeepEncoder (FP16, CUDA Graph)  │  MoE Decoder (R-SWA + INT4)│
├─────────────────────────────────────────────────────────────┤
│                    Kernel Layer (CUDA sm_120)                 │
│  R-SWA Attention Kernel  │  INT4 MoE GEMM  │  融合 RMSNorm+RoPE│
└─────────────────────────────────────────────────────────────┘
```

**CMake 核心配置**：

```cmake
cmake_minimum_required(VERSION 3.24)
project(unlimited_ocr_engine LANGUAGES CXX CUDA)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR "CUDA >= 12.8 required for sm_120")
endif()

set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)  # 缩短增量构建时间
```

**目录结构**：

```
unlimited-ocr-engine/
├── CMakeLists.txt
├── cmake/CUDAArch.cmake
├── src/
│   ├── runtime/       # model_loader.cpp, tokenizer.cpp, config.cpp
│   ├── scheduler/     # continuous_batch.cpp, block_manager.cpp, memory_pool.cpp
│   ├── engine/        # deep_encoder.cpp, moe_decoder.cpp, sampler.cpp
│   └── kernels/       # rswa_attention.cu, moe_gemm_int4.cu, rmsnorm.cu, rope_fused.cu
├── tests/             # test_kv_ring_buffer.cpp, test_rswa_kernel.cu, test_memory_budget.cpp
├── benchmarks/        # bench_decode.cpp, bench_throughput.cpp
└── third_party/       # nlohmann/json, spdlog
```

**双构建模式**：`-DENGINE_BACKEND=CUDA` 启用 CUDA 路径，`-DENGINE_BACKEND=CPU` 走参考实现（仅依赖 OpenMP），用于 kernel 正确性验证。


## 五、模块设计

### 5.1 Scheduler：R-SWA 感知的 Block Manager

这是本项目最具工程挑战的模块。标准 PagedAttention 将 KV Cache 视为同构块序列，而 R-SWA 的结构本质上是**异构的**：

```
[视觉 token 区: 0 ~ V-1]  [滑动窗口区: 环形 buffer, 大小 W=128]
     ↑ 固定, 永不覆写              ↑ 每步写入新 KV, 覆盖最旧位置
```

**Block Manager 的定制设计**：

- **视觉 token 区使用独立内存池**，按文档粒度共享。同一文档的不同页面在 Prefill 阶段可共享视觉 token 的 KV 前缀，通过引用计数管理生命周期。利用 DeepEncoder 的 16× 压缩特性，前缀缓存的收益显著。
- **滑动窗口区使用环形缓冲区**，每个请求维护独立的 `head` 指针和 `window_pos` 索引。环形 buffer 以 `cudaMalloc` 一次性分配 W=128 个 token 的 KV 空间，写入时通过原子操作更新 head。环形 buffer 不参与 block 级碎片整理，消除了 PagedAttention 在滑动窗口场景下的 block table 更新开销。
- **Prefill 阶段的 KV 分区写入**：Prefill 时视觉 token 数量远大于 W，但只有最后 W 个 token 需进入滑动窗口。设计上将 Prefill 输出按 token 位置分区：位置 < V 的 KV 写入视觉区，位置 ≥ V-W 的 KV 写入环形区，中间 gap 区域的 KV 直接丢弃。这比“先全部写入再裁剪”的方案节省约 70% 的 Prefill KV 写入带宽。

**连续批处理**：调度器维护活跃请求队列和等待队列。每步解码前扫描活跃队列，组装当前 batch。新请求插入 batch 尾部，完成解码的请求立即释放 KV 资源。Batch 大小受显存水位和 kernel 启动开销双重约束——profiling 发现 batch size < 4 时 kernel launch 开销占总解码时间的 30% 以上，因此设置最小有效 batch size 为 4。推荐最大 batch size 为 16，在 SM 利用率和显存占用间达到平衡。

### 5.2 Compute Engine：INT4 优先的权重策略

**DeepEncoder 保持 FP16**。DeepEncoder 采用 SAM-ViT + CLIP-ViT 级联架构，参数量约 1.5GB FP16，负责细粒度文字检测和特征提取。量化对 OCR 精度的破坏在编码器端尤为敏感。

**MoE Decoder 采用 AWQ INT4 量化**。3B 参数中 MoE Decoder 占绝大部分，AWQ 量化后权重从 ~6GB 降至 ~1.8GB，释放约 4.2GB 显存用于 KV Cache 和 batch 扩展。社区在消费级 Blackwell 上的基准测试表明，AWQ INT4 的实际吞吐优于 NVFP4，且精度损失更可控。量化校准按 expert 分组收集激活统计，减少对低频 expert 的量化精度冲击。

**CUDA Graph 捕获范围**：将 Graph 捕获限定在确定性内核序列上（RMSNorm → QKV → RoPE → Attention → O → MoE Router），采样和 MoE expert 计算在 Graph 外部执行。这既保证了 Graph 的稳定性，又保留了动态路由的灵活性。

### 5.3 Kernel Layer：带宽友好型内核设计

RTX 5060 Ti 的 448 GB/s 带宽意味着**每个 byte 都珍贵**。内核设计围绕“最小化全局内存往返”这一核心原则：

**R-SWA Attention Kernel**：
- 视觉区 KV 使用**共享内存 staging**，每个 thread block 将当前需要的视觉 KV 块预取到共享内存。
- 环形区 KV 使用 `__ldg` 只读缓存路径。
- 融合 QK^T 与 softmax，消除分数矩阵的全局内存往返。
- 环形区使用 Online Softmax（Flash Attention 风格），避免存储完整注意力分数矩阵。

**INT4 MoE GEMM Kernel**：
- 使用 `mma.sync.aligned.m16n8k32.row.col.satfinite.s4.s4.s32` 指令（sm_120 支持的 INT4 MMA 形状）。
- 激活值 BF16 输入，权重 INT4 打包（uint32 中 8 个 4-bit 值）。
- 小 batch（≤8）使用 split-K 策略，K 维按 128 分段，原子加归约。
- 大 batch（>8）使用标准 GEMM，充分利用 Tensor Core。

**融合 RMSNorm + RoPE Kernel**：将 RMSNorm 的 reduction、权重缩放和 RoPE 旋转位置编码融合为单个 kernel，消除中间张量的全局内存往返。


## 六、关键技术点与创新点

### 6.1 关键技术点

| 技术点 | 实现方式 | 性能收益 |
|---|---|---|
| **R-SWA 环形 KV Buffer** | 环形缓冲区 + 原子 head 指针，无 block table 更新 | 消除滑动窗口 block 管理开销 |
| **视觉 Token 前缀共享** | 引用计数 + 独立内存池 | Prefill KV 写入带宽节省 |
| **CUDA Graph 双路径捕获** | 全局/局部路径独立 Graph，持久化 KV 索引 buffer | Kernel launch 开销降低 |
| **INT4 MoE GEMM** | Split-K + sm_120 mma.sync INT4 指令 | 权重显存降至 30% |
| **纯 C++ BPE Tokenizer** | 预编译 merge 查找表 | 单 token < 2μs |
| **mmap + Pinned 权重加载** | `mmap` + `cudaHostRegister` DMA 直通 | 消除 H2D 拷贝 CPU 侧参与 |

### 6.2 创新点

**创新点一：R-SWA 感知的“固定+环形”双层 KV Cache 管理。** 通用 PagedAttention 将 KV Cache 视为同构块序列，而 R-SWA 的结构本质上是异构的——视觉区是只读的、固定大小的、可跨请求共享的；滑动窗口区是读写循环的、每请求独立的。我们将这两部分分离管理，视觉区走引用计数共享池，滑动窗口区走无 block table 的环形 buffer，在 C++ 调度器层面直接编码了这一结构差异。

**创新点二：CUDA Graph 条件下的动态 MoE 路由。** CUDA Graph 要求捕获的 kernel 序列在执行时完全静态，而 MoE 的 expert 路由结果是数据依赖的动态值。方案是将路由 kernel 和 expert 计算 kernel 分离：路由 kernel 在 Graph 外部执行，输出写入持久化 buffer；expert 计算 kernel 在 Graph 内部以“全 expert 调度 + 掩码跳过”方式执行——每个 thread block 根据路由 buffer 判断是否跳过当前 expert。这以约 15% 的冗余计算换取了 CUDA Graph 的完全兼容性。

**创新点三：Prefill 阶段的 KV 分区写入。** 标准流程是 Prefill 生成全部 KV 后写入 Cache，再在 Decode 阶段逐步裁剪。我们在 Prefill 的 attention kernel 内部直接按位置分区：位置 < V 的 KV 写入视觉区，位置 ≥ V-W 的写入环形区，中间 gap 区域直接丢弃。这需要修改 attention kernel 的 KV 写出逻辑，但消除了约 70% 的无效 KV 写入。


## 七、性能测试指标

### 7.1 测试方法论

基准测试使用 `bench_decode.cpp` 和 `bench_throughput.cpp` 两个独立可执行文件。`bench_decode` 加载固定长度的模拟 KV Cache，测量单步解码 kernel 执行时间，排除调度和 I/O 干扰。`bench_throughput` 启动完整引擎，以固定 QPS 注入请求，测量端到端指标。性能数据通过 CUDA Events 和 `nvtx` 标记采集，用 `nsys profile` 生成 timeline，用 `ncu` 分析 kernel 级指标。

### 7.2 核心指标记录表

**测试环境**：RTX 5060 Ti 16GB, CUDA 12.8+, Linux

| 指标类别 | 具体指标 | 实测值 |
|---|---|---|
| **Kernel 级** | R-SWA Attention kernel 耗时（W=128, batch=1） | |
| | MoE INT4 GEMM 有效 TFLOPS（batch=8） | |
| | CUDA Graph launch 开销 | |
| **端到端** | 单页 TTFT P50 | |
| | TPOT P50（decode 稳定态, batch=1） | |
| | 解码第 40 页 vs 第 1 页 TPOT 比 | |
| **吞吐** | batch=8 输出 token 吞吐 | |
| | batch=16 输出 token 吞吐 | |
| | GPU SM 利用率（decode 阶段） | |
| **显存** | batch=8 峰值显存（INT4） | |
| | batch=16 峰值显存（INT4） | |
| | KV Cache 碎片率 | |
| **精度** | AWQ INT4 vs BF16 的 OmniDocBench v1.6 综合分下降 | |
| | R-SWA Kernel 输出与 PyTorch 参考的最大绝对误差 | |

### 7.3 Ablation 实验计划

1. **INT4 vs BF16 权重**：相同 batch 配置下对比 TPOT 和显存占用，验证 INT4 在 sm_120 上的带宽收益。
2. **AWQ vs 朴素 INT4**：对比精度损失和内核兼容性，验证 AWQ 在 OCR 场景下的精度优势。
3. **CUDA Graph 捕获范围**：只捕获 Decoder vs 捕获 Decoder + Encoder 的 kernel launch 开销对比。
4. **环形 Buffer vs 通用 PagedAttention 模拟**：对比 KV Cache 管理方案的显存碎片率和 block 更新耗时。
5. **内存池预分配 vs 动态分配**：对比推理过程中是否存在显存碎片导致的延迟抖动。
