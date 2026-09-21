# 核心技术实现说明

> 开发入口见 `tAgent.md`；文档导航见 `README.md`。

本文说明本项目关键模块的设计与实现位置。代码路径均为相对仓库根目录。

## 1. R-SWA KV Cache（`include/uocr/kv_cache.h`, `src/scheduler/kv_cache.cpp`）

R-SWA 的缓存是"固定参考区 + 环形窗口区"的异构结构。`RSWACache` 精确复刻了
参考实现的状态机：

```
状态：prefill_len P、capacity P+W、len[layer]、ring_pos[layer]、ring_started[layer]
```

- `reset(P)`：为每层分配 `(P+W) × kv_heads × head_dim` 的 K/V，`len=P`。
- `write_prefill(layer, k, v, seq)`：把 prefill 的 seq 个 KV 写进 `[0, seq)`，
  同时令 `prefill_len = seq`。
- `append_decode(layer, k, v)`：
  - `len < P+W`：线性追加，达到 `P+W` 时 `ring_started=true, ring_pos=0`（warmup）；
  - 否则覆写槽位 `P + ring_pos`，`ring_pos = (ring_pos+1) % W`（稳态）。
- `attention(layer, q, seq, q_start, out, heads, causal)`：
  在 `[0, len)` 上做注意力，prefill 传 `causal=true`，decode 传 `false`。
  使用 **online softmax**（Flash-Attention 风格），不落地 score 矩阵；
  GQA 通过 `group = heads / kv_heads` 映射。

关键点：**参考区永不覆写**，因此同一文档的多个请求可以共享它（见第 4 节）。
缓存大小在 warmup 后恒定，使解码第 N 步与第 1 步成本相同（由
`test_decoder.cpp::decoder_rswa_cache_bounded` 验证）。

## 2. MoE Decoder（`include/uocr/moe_decoder.h`, `src/engine/moe_decoder.cpp`）

12 层，`hidden=1280`，`first_k_dense_replace=1`：第 0 层为 dense MLP
（`6848` 中间维），第 1–11 层为 MoE。

每层：
```
x1 = x + Attention(RMSNorm(x))
x2 = x1 + MLP/MoE(RMSNorm(x1))
```

Attention 为普通 MHA（`use_mla=false`）：q/k/v/o 均为 1280→1280，10 头，
`head_dim=128`。RoPE 在写入 cache 之前施加。

MoE 路由（`moe_detail::moe_gate`）：
1. `logits = x @ router^T`（`router` 形状 `[64,1280]`），全程 FP32；
2. softmax（`scoring_func`）；
3. greedy top-6；
4. `norm_topk_prob=false` → 直接 `weight *= routed_scaling_factor(1.0)`；
5. 每个被选专家做 `down(silu(gate(x)) * up(x))`，按权重累加；
6. 再加 **2 个共享专家**（`moe_intermediate_size × 2 = 1792` 中间维）。

权重存储与计算解耦：`WeightMatrix` 支持 `F32`、`BF16_EXT`（零拷贝 mmap 视图）、
`INT4`（AWQ group 量化）三种格式，`matmul` 自动分派（`src/runtime/weights.cpp`）。

## 3. AWQ INT4 量化（`include/uocr/quant.h`, `src/runtime/quant.cpp`）

逐行、按 `group_size`（默认 128）做仿射量化：

```
scale = (max - min) / 15
zero  = round(-min / scale)        // 量化值 q = round(w/scale + zero) ∈ [0,15]
w'    = (q - zero) * scale
```

打包为每字节 2 个值（低半字节=偶数列）。另有对称版本（zero 固定为 8）。
真实模型采样专家上 rel-L2 ≈ 0.101，压缩到 BF16 的 ~25%。

MoE 权重在 `DecoderWeights::load(..., quantize_experts_int4=true)` 时按需量化；
默认保持 BF16 零拷贝视图以避免不必要的 CPU 开销与内存。

**加载耗时**：真实模型 INT4 加载墙钟主要由 2112 个专家张量的量化决定（单线程 ~18s），
H2D 上传只有 ~2s。量化按专家独立、`quantize_int4_awq` 为纯函数、`read_f32` 只读
mmap，因此在 `weights.cpp` 的专家循环加 `#pragma omp parallel for schedule(dynamic)`
（仅在量化时并行；`_OPENMP` 未定义时自动退化为串行），实测进程墙钟 20.5→7.4s。
`cudaHostRegister` 直通方案受本机 `ulimit -l = 8MB` 限制无法实施（见 `tAgent.md` §2.9）。

## 4. Block Manager 与显存池（`include/uocr/block_manager.h`, `src/scheduler/`）

- `MemoryPool`：偏移式 arena + 空闲链表 + 相邻合并，可统计 `used/peak/fragmentation`。
- `BlockManager`：
  - **参考区**走 `acquire_prefix(key, P, bytes)` / `release_prefix(key)`，
    按 `doc_key` 引用计数共享；`shared_prefix_bytes` 统计因共享节省的字节。
  - **环形区**走独立 `ring_pool`，每请求一块，无需 block table。
- `ContinuousBatchScheduler`：等待队列 + 运行集合，`build_batch()` 每步填满
  batch（优先 decode，空位再接纳新请求），请求结束即释放 KV。

## 5. CUDA 内核（`src/kernels/cuda/`）

| 文件 | 内核 | 说明 |
|---|---|---|
| `rswa_attention.cu` | `rswa_decode_kernel` | 每 head 一个 block，在线 softmax 遍历 `kv_len`，`__ldg` 读 cache，无 score 矩阵 |
| `moe_gemm_int4.cu` | `moe_gemm_int4_kernel` | AWQ 反量化 + FP32 累加（标量正确性基线） |
| `moe_gemm_int4.cu` | `moe_gemm_int4_tc_kernel` | **W4A16**：寄存器内 INT4→BF16 反量化 + `mma.m16n8k16.bf16`（f32 累加） |
| `moe_gemm_int4.cu` | `moe_grouped_(gate_up|down)_int4_kernel` | **grouped expert GEMM**：一层一次 launch，block 映射 (expert, n-tile)，按设备分组表循环 token（见 §5.9） |
| `rmsnorm.cu` | `rmsnorm_kernel` | 块内归约 |
| `rope_fused.cu` | `rope_kernel`, `rmsnorm_head_kernel` | 融合逐头 RMSNorm + RoPE |
| `backend.cu` | `matmul_t(_bf16)`, `matmul_t_split_bf16`, `device_info` | 稠密 GEMM（含视觉用 split-bf16 高精度 TC）、设备信息 |
| `gpu_ops.cu` | `silu_mul`, `add_scaled`, `gather_rows`, `scatter_add_scaled`, `embed_gather` | 设备端逐元素 / 分组 MoE 辅助 + embedding 查表（§5.10） |
| `gpu_cache.cu` / `gpu_cache.h` | `GpuRSWACache` | device-resident 固定+环形 KV cache |
| `gpu_decoder.cu` / `gpu_decoder.h` | `GpuDecoder` | 完整设备端 MoE decoder（bf16 权重常驻；路由 top-k 在 host 调度） |

CUDA 构建与 CPU 构建共用 `RSWACache`/`WeightMatrix` 的数据布局，因此
`tests/test_rswa_cuda.cu` 可以直接把 host cache 上传后与 CPU 参考逐元素对比
（当前 max_err < 1e-5）。

### 5.1 设备端 decoder（`GpuDecoder`）

每层流程全部在 GPU 上：`rmsnorm → q/k/v matmul(bf16) → rope → GpuRSWACache
(readonly/causal prefill 或 ring decode) → o matmul → residual → rmsnorm2 →
dense MLP 或 MoE`。MoE 里 router logits 由 GPU 算出后拷回 host 做 top-k 与
“按专家分组”，再对每个专家在 GPU 上批量 `gate/up/silu/down`，用
`scatter_add_scaled` 按路由权重累加，最后加共享专家。权重在构造时上传一次
（bf16，含全部专家），decode 每步只上传一行 embedding 与 position。

构建上 CUDA 源文件并入 `uocr_core`（`UOCR_ENGINE_LIB` 恒为 `uocr_core`），
`Engine(..., Backend::CUDA)` 内部持有 `GpuDecoder`，`generate` 与
`generate_from_image` 自动分派；CPU 构建不编译 `.cu`、不包含该成员。

### 5.2 CUDA Graph 捕获（P2）

decode step 现在可以整体捕获成一个 CUDA Graph：

- **device router**（`moe_device.cu::moe_router_topk`）：softmax/sigmoid → greedy top-k →
  用 `atomicAdd` 把 `(token, expert, weight)` 分组写入 `[n_experts, cap]`，去掉 host 同步。
- **全专家固定调度 + 掩码跳过**（`moe_experts_masked` / `..._int4`）：`grid=(n_experts,
  ceil(rows/warps))`，每个 warp 先检查 `count[e]`，为 0 直接返回。网格只依赖静态形状。
- **设备端环形指针**（`rswa_append_decode`）：`len`/`ring_pos` 常驻 device，写入槽位在
  kernel 内解析；`rswa_attention_devlen` 从 `*d_len` 读有效长度并掩码，避免把
  warmup 期长度烘焙进图。
- **Graph 范围**：`GraphScope::kFull`（默认）捕获整个 step；`kAttnDense`
  逐层捕获 attention 子图、MoE 在图外发射（`EngineConfig.graph_scope="attn_dense"`）。
- 每次 decode 前把一行 embedding 与 position 写入 pinned staging，Graph replay 时由
  录制好的 H2D 读取；replay 后同步一次、跑 device lm_head、D2H logits。

### 5.3 device INT4 专家权重（P2）

`DecoderWeights::load(..., quantize_experts_int4=true)` 时专家权重是 AWQ INT4。
`GpuDecoder` 把它们按 `[n_experts, rows, cols]` 连续上传（packed uint8 + per-group
scale/zero），显存从 9.2GB 降到 2.2GB。`moe_experts_masked_int4` 在 kernel 内
按 group 反量化，权重读取同样是 warp-per-output 合并访存。

prefill 的大批量专家计算走 `moe_gemm_int4_tc`（W4A16 TC GEMM，详见
`BENCHMARKS.md` §2.5 与 `CORE_TECH.md` §5.5b）：模板参数 `BN`/`BK` 控制每 block 的
n 列宽与 k 步长，专家 GEMM 的 N=896/1280 下经验最优是 **BN=8 / BK=64**（更宽的 BN
会让 block 数掉到 36 个 SM 以下；更大的 BK 摊薄同步开销）。权重 panel 的反量化
staging 由 128 线程完成；`moe_gemm_int4_tc_nk` 可扫 bn∈{8,16,32,64}×bk∈{16,32,64}，
另提供 cp.async 变体 `moe_gemm_int4_tc_pipe`。

### 5.4 合并访存内核（P2 性能）

`matvec_bf16`（warp-per-output，每 lane 向量化读 2 个 bf16）与 expert MLP 的
gate_up/down 内核修复了“相邻线程按行 stride 读权重”导致的 ~16× 带宽浪费。
`matmul_t_bf16` 的 CUDA-core 参考版保留为 64×64 分块 + shared memory staging
（panel 补 1 列避免 bank conflict），见 §5.5 的 tensor-core 取代。

### 5.5 bf16 tensor-core GEMM（P2，P3 重写 2026-09-20）

`matmul_t_bf16`（dense/shared 投影与 lm_head 的主力）用
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，P3 起统一改用带
`cp.async` 流水线的 `matmul_t_bf16_tc_pipe_kernel`：

- **cp.async 多级 ring + bank-conflict-free padding**（`src/kernels/cuda/backend.cu`）：
  权重面板以 `cp.async` 预取进共享内存 ring（m≤16 用 STAGES=4，m>16 用 STAGES=3），
  在处理当前面板时下一面板已在传输；共享行尾 padding 8 个 bf16，使 BK=32 时
  `ldmatrix`（x4 取 A、x2 取 B）无 bank 冲突。每 k-step 只需 **一次**
  `__syncthreads`：环形下一内存槽由该同步保证已释放，尾部用 `wait_group(0)` 排空。
- **两种 tile**：`SMALL=true`（m≤16，decode/batched lm_head）BM=16、4 warp 沿 n
  拆分（BN=64，每 warp 16 列 = 2 个 n8）；`SMALL=false`（m>16，prefill）BM=64、
  4 warp 各 16 行、遍历 8 个 n8。激活面板在计算前用 vectorized `float4` 载入并转
  bf16（L2 命中），权重走 cp.async。
- **回退/对照**：`matmul_t_bf16_ref`（原 CUDA-core 分块内核）保留，供单测 A/B；
  单测在 `m∈{3,16,40,64,273}`、`k∈{48,64,128,160,256}` 上 TC vs ref
  rel_l2 ≤ 0.0018。
- `batch_decode`/ragged prefill 的 lm_head 在行数 ==1 时仍走 `matvec_bf16`
  （避免用 GEMM 处理单行）。
- **残留**：m>16 的 kernel 每个 n-tile 重读一次 A、权重按 tile 读一次；更大 tile
  （BM=128）与 grouped expert GEMM（任务 2.2）是后续方向。
- 微基准（`bench/bench_bf16_gemm_*.txt`，RTX 5060 Ti）：
  - n=k=1280：m=8 31.3→**16.7µs**（1.9×）、m=16 31.7→**17.2µs**（1.8×）、
    m=273 52.2→**48.5µs**（18.5 TFLOPS）。
  - lm_head n=129280、k=1280：m=8 2633→**1147µs**（2.3×）、m=16 2651→**1174µs**
    （2.3×）——m≤16 的 lm_head 是本次最大收益点。
- 真实模型连续批处理（`BENCHMARKS.md` §2.6）：BF16 batch=16 416→**~520** tok/s、
  整波 prefill 212→**~164 ms**；INT4 batch=16 387→**~431** tok/s。

### 5.5b INT4 专家 GEMM 的 P3 调优（2026-09-20）

`moe_gemm_int4_tc`（prefill 逐专家 W4A16）在 P3 做了两处改动：
- **激活 staging 向量化**：原按元素标量 `x[gm*k+gk]` 改为 `float4` 载入 + 转 bf16
  写入带 padding 的 `sA`，每线程指令数从 8 降到 2。
- **默认 tile 改为 `bn=8 / bk=64`**：`bk` 从 16 增到 64 把 k-step 从 80 降到 20，
  同步/发射开销大幅下降；`bn=8` 保留高 block 数（延迟隐藏）。`moe_gemm_int4_tc_nk`
  可扫 `bn∈{8,16,32,64}×bk∈{16,32,64}`。
- **备选 cp.async 变体**：`moe_gemm_int4_tc_pipe`（packed 权重异步预取 + 激活驻留
  寄存器）在 m≤32 时最优（m=1 30.2→26.1µs），但 m≈96（prefill 平均每专家 token 数）
  时不如 `bk=64` 的向量化同步版，故仅保留供 `bench_int4_gemm` 调优对照。
- 微基准：n=896/k=1280 时 m=96 51.1→**49.1µs**、m=273 111.7→**97.3µs**（`bn=8/bk=64`）。

### 5.6 连续批处理（P2 收尾，2026-09-19）

- **per-slot R-SWA KV**：`GpuDecoder::batch_configure(slots, capacity)` 为每个 slot
  分配独立 `[capacity, kv_heads, head_dim]` K/V 与 `d_len/d_ring_pos`（`[layers*slots]`
  扁平数组，索引 `layer*slots + slot`）。
- **batched attention**：`rswa_append_decode_batch`（grid=B，每行一个 block 推进
  对应 slot 的环形游标）与 `rswa_attention_batch`（grid=(B, heads)，每 block 走
  online softmax）。行→slot 由 `d_slots[b]` 指定（请求可占用任意 slot），cache
  地址与 `d_len/d_ring_pos/d_prefill_len` 均按 slot 索引，激活按行索引，因此变长
  prompt 的固定区/环形区边界正确。每步 attention 发射数从 `2·B·L` 降到 `2·L`。
- **batch decode 流程**：`forward_batch` 逐层 `attention_block_batch` +
  `mlp_block(dev_moe=true)`（device router + masked 专家），最后一次性
  `matmul_t_bf16` lm_head 得到 `[B, V]` logits 并 D2H。
- **ragged 多请求 prefill**：`GpuDecoder::batch_prefill_embeds(embeds, starts, lengths,
  slots, logits)` 把一步内新请求的全部 token 打包成一个 `[total, h]` 前向。
  - `rswa_write_prefill_ragged` 把第 t 行 K/V 直接散写到
    `batch_k[l][slots[t]*cap + pos[t]]`；`rswa_attention_ragged`（grid=`(total, heads)`）
    让第 t 行只 causal attend 自己 slot 的 `[0, pos[t]]`。
  - 前向结束后按请求取最后一行做 final norm + lm_head，得到每个请求的首 token logits。
  - MoE 选择：每专家 token 数 ≤2 时用 masked matvec（`dev_moe=true`），否则用逐专家
    tensor-core GEMM（`moe_gemm_int4_tc`，M 即该专家 token 数）——大批量时权重只读
    一次，prefill 从"16 请求逐串行"变成整波一次（当前耗时见 `BENCHMARKS.md` §2.6）。
  - `EngineConfig::use_device_moe_prefill`（默认 true）控制单请求 prefill
    （`prefill_embeds`，OCR 路径）走 masked device MoE。

### 5.7 Batched CUDA Graph（P2，2026-09-19）

单请求 decode 早已整步捕获，但连续批处理的每一步仍由 host 逐 kernel 发射
（每步约 `13 × ~20` 个 launch）。本项把 batched decode 也纳入 Graph：

- **按 batch size 缓存 Graph**：`GpuDecoder` 维护 `batch_graphs_[B-1]` /
  `batch_graph_execs_[B-1]`，第一次以某个行数 `B` 解码时捕获
  `forward_batch + final rmsnorm`，之后同 `B` 直接 replay。不同步数只对应
  有限个 `B`（`1..max_batch`），因此最多缓存 `max_batch` 张图。
- **活跃集合变化不需要重捕获**：`B` 只决定 kernel 的 grid 维度；每步变化的
  行→slot 映射、position、token embedding 都写入**地址固定**的
  `d_batch_slots_ / d_pos_ / d_xin_`，在 replay 时被录制好的 kernel 读取。
  因此请求进出、slot 复用、非恒等 slot 排列都不会使 Graph 失效——这与
  tAgent 里“活跃集合变化即重捕获”的备选方案相比更省。
- **失效条件**：只要 `ensure_scratch` / `ensure_router_scratch` 因更大的
  `seq` 重新分配（scratch 基址或 `router_cap_` 改变），或
  `batch_configure` 重分配 per-slot KV，就销毁全部 batched graph
  （`invalidate_batch_graphs()`），下次解码重新捕获。
- **lm_head 仍在图外**：`matmul_t_bf16` 的输出 `d_logits_batch_` 会按需扩容，
  且每步都要 D2H 供采样，放在图外避免把可变量烘焙进去。
- `graph_scope=="attn_dense"` 时 batched decode 回退到非 Graph 路径（逐层图仅
  为单请求 seq=1 设计）。
- 测试：`uocr_cuda_tests` 新增“batched decode graph vs plain”——固定 token/pos
  脚本、非恒等 slot 排列、窗口 W=8 跑 14 步覆盖环形覆写，两者 logits
  rel_l2 = 0（逐位一致），且 graph 路径确认捕获到 1 张图、plain 路径 0 张。

### 5.8 DeepEncoder CUDA 移植（P3，2026-09-20）

视觉编码器（SAM-ViT-B + CLIP-L + projector）从纯 CPU/OpenMP 移植到设备
（`src/engine/gpu_encoder.cu`、`src/kernels/cuda/vision_ops.cu`）：

- **权重**：线性层（SAM qkv/proj/mlp、CLIP qkv/out/fc1/fc2、projector）以 bf16
  常驻设备；norm/bias/pos/rel 为 f32，conv 权重转 bf16 供 GEMM。
- **精度（关键）**：朴素 bf16 激活舍入经 12 层 SAM + 24 层 CLIP 累积放大到 ~11%，
  远超 6e-4。改为 **split-bf16 tensor-core GEMM**（`matmul_t_split_bf16`，P3 收尾）：
  激活拆成 `hi=bf16(x)`、`lo=bf16(x-hi)`，权重本就是精确 bf16，因此每个权重面板
  做 2 次 mma（`hi·W + lo·W`），恢复 ~16-bit 有效尾数；累加仍是 f32。staging 复用
  既有 `cp.async` 权重流水线，`k%8≠0` 时回退 `matmul_t_f32w`（f32 激活、f32 累加）。
  实测 1024 输入 visual rel_l2 **8.2e-5**（验收 6e-4）。
- **内核**：`layernorm_rows`、`gelu_inplace`/`quick_gelu_inplace`、`add_bias_rows`/
  `add_inplace`、`im2col_chw`/`im2col_hwc`（conv → im2col + GEMM）、
  `window_partition`/`window_unpartition`、`attention_flash`（online softmax、
  warp-per-query、K/V 分块入 shared；`RELPOS` 模板给 SAM 加 decomposed relative
  position bias，CLIP 不加）。
- **attention 优化（P3 收尾）**：1024 输入时 SAM global attention 曾是最大头（~404ms）。
  两处改动：① **relpos 因式分解**——`q·(Rh[ih]+Rw[iw])` 拆成 `q·Rh[ih] + q·Rw[iw]`，
  每 query 在 block 起始处算好长度 `H+W` 的查找表存入 shared，内层循环由「每 key
  4 次 global load + 两次 warp 归约」变成「2 次 shared 查表」，省 ~120ms；② 每 warp
  处理 `kAttnQP=4` 个 query（K/V tile 复用 4×）、`float2` 读 shared、softmax 用
  `__expf`。核心仍是 f32 CUDA-core（~1 TFLOPS），要再上一档需 tensor-core
  attention（未做）。
- **两个语义坑**（见 `PITFALLS.md` §18）：windowed SAM 必须**先 partition 再 QKV**
  （padding token 要拿到 QKV bias）；CLIP MLP 的残差要用 `fc2` 覆盖 LN 结果再
  `+= residual`。
- **集成**：`Engine::set_vision_gpu()` 后 `image_embeddings` 在 CUDA 后端整条视觉
  路径走设备；`tools/compare_ocr --gpu-vision` 可端到端验证。
- **验收**（`tools/compare_vision_gpu`，GPU vs CPU 参考）：
  `visual_embeddings` rel_l2 **8.2e-5**、`clip_features` 9.9e-5、`sam_features`
  1.7e-5（验收 ≤6e-4）；单图 1024 编码 **366 ms**（CPU ~2 min）、640 **95 ms**、
  224 **18 ms**。`--selftest` 覆盖 layernorm / relpos attention / full attention。
  相对 f32 GEMM 版本（612/153/34 ms）提升 **1.67×**；剩余瓶颈是 f32 CUDA-core
  SAM attention（BENCHMARKS §2.9），达到 150–250ms 目标需 tensor-core attention。

### 5.9 Grouped INT4 expert GEMM + device router（P3 收尾，2026-09-20）

ragged 大批量 prefill 原来走 `mlp_block(dev_moe=false)`：每层 router logits
**D2H → host top-k → 每专家索引 H2D**，再对 64 专家各发 gate/up/down 三个
`moe_gemm_int4_tc`（外加 gather/silu/scatter），一层 ~192 次 launch 且带同步点。
本项把它替换成 device router + 一次 grouped launch：

- **device 分组表**：直接复用 `moe_router_topk`，产出 `d_assign_token/d_assign_w/
  d_count`（布局 `[n_experts, cap]`，`cap = total`）。`mlp_block` 的 dev_moe 分支
  不再把 router logits 拷回 host。
- **`moe_grouped_gate_up_int4`**：grid=`(ceil(inter/BN), n_experts)`，每 block 负责
  (expert, n-tile)，在设备端按 `count[e]` 循环该专家的 token（`assign_token[e, :]`），
  一次算出 gate+up 并融合 SiLU 写进 `act[e, cap_slot, inter]`。
- **`moe_grouped_down_int4`**：grid=`(ceil(hidden/BN), n_experts)`，从 `act` 读回，
  一次算出 down 并按 `assign_w` scatter-add 进输出。down 的 k 维即 `inter`。
- **固定网格**：grid 只依赖静态形状；token 列表、count、权重全部设备驻留，因此一层
  只有 2 次 launch（原来是 ~192），且 ragged prefill 每层**无 D2H/H2D 同步**。
- **`forward_ragged` 选择**：专家为 INT4 且 `hidden % 4 == 0`、`inter % 4 == 0`
  时一律走 grouped（小批量也比 masked 快，因为权重只读一次）；否则保持原逻辑
  （小批量 masked / 大批量 host）。grouped 不用于单请求 prefill 与 decode。
- **tile 形状**（`EngineConfig::grouped_moe_bn/bm`，默认 `bn=32/bm=128`）：
  `BM=128` 用 8 warp×16 行，把每专家的 m-tile 数从 2 降到 1（count≈96 时权重
  只 stage 一次），这是最大收益点（gate_up 7.1→4.0 ms/layer）；`BN=32` 在占用率与
  每 block 权重面板复用间取平衡（`BN` 越大 A 面板复用越多，但 block 数减少会掉占用）。
  helper 的 staging 循环改成按 `blockDim.x` 步进以支持 128/256 线程两种 block。
- **验收**（`tests/test_rswa_cuda.cu` 新增 “Grouped INT4 ragged prefill”）：3 个
  请求共 600 token（每专家 > 128，覆盖 BM=128 多 m-tile），grouped vs 逐专家 host
  路径 logits rel_l2 **1.7e-3**（与既有 device-router vs host-router 的 ragged 回归
  同量级，阈值 1e-2）；`grouped_moe_calls()>0` 确认真的走了 grouped 内核。
- **性能**（真实模型，见 `BENCHMARKS.md` §2.10）：INT4 batch=16 整波 prefill
  **213 → 120 ms**、tok/s **429 → 488–516**；B=1/2/4/8 的 prefill 也全线下降
  （原 per-expert host 路径的同步与发射开销被消除）。
- **剩余**：`rswa_attn_ragged`（prefill attention，B=16 时 ~21 ms）成为新的第二大头，
  未在本项处理；BF16 专家的大批量 prefill 仍走 host 路径（未加 grouped BF16 内核）。

### 5.10 device embedding 查表（P3 收尾，2026-09-20）

原来每一步 decode 都在 host 上 `embed_tokens.row(token, h_embed_pinned_)` 再
`cudaMemcpyAsync` 整个 hidden（1280 float = 5KB）到 `d_xin_`；单请求 Graph 还把这次
H2D 录制进图。改为：

- **表常驻设备**：`upload_embedding` 把 `embed_tokens` 拷到 `d_embed_`。源是
  `BF16_EXT`（真实 checkpoint）时直接存 bf16（331MB），gather 时转 f32——与 host
  `row()` 逐位一致；源是 F32 时存 f32。避免了一律升 f32 的 662MB 开销。
- **`embed_gather`**（`src/kernels/cuda/gpu_ops.cu`）：`out[r,:] = table[ids[r],:]`，
  bf16/f32 两个模板分支。
- **单请求 decode**：只把 token id 写入 pinned 再 H2D（4B），Graph 内执行
  `embed_gather`（不再录 H2D）。
- **batched decode**：H2D `batch` 个 token id（`d_token_ids_`），`embed_gather` 在
  Graph 外写入 `d_xin_` 后 replay；Graph 只需按行数缓存，不受 token 变化影响。
- **失效规则**：`d_token_ids_` 重新分配会使已捕获 Graph 失效
  （`ensure_token_ids` 调 `invalidate_graph`/`invalidate_batch_graphs`）；`batch_configure`
  按 slot 数预分配，避免 batch 增长时反复失效。
- **`attn_dense` scope**：逐层图读 `d_ping_`，gather 在图外写 `d_ping_` 后 launch。
- **验收**：`uocr_cuda_tests` 全部通过、greedy 不变；真实模型 batch=16 吞吐在噪声内
  （BF16/INT4 ~520/518 tok/s），显存 +331MB（bf16 表）。剩余：单请求 prefill 仍
  host 查表（每请求一次）。

### 5.11 tensor-core flash attention（P3 收尾，2026-09-21 调通）

`attention_flash_tc_kernel<RELPOS>`（`src/kernels/cuda/vision_ops.cu`）是给 SAM
global attention（S≥512）准备的 tensor-core 版本，`attention_flash` 在
`relpos && S>=512` 且 tile shared 放得下时启用，否则回退 f32 kernel。

- **设计**：Block = **256 线程（8 warp）** × 64 query，per-head；tile `BM=64/BN=32`。
  8 个 warp 把 **mma 的 n 维对半拆**（`kTcSplit=2`）：`wq=warp&3` 选 16 个 query 行，
  `wn=warp>>2` 选 n 的一半（S 的 2/4 个 n-tile、PV 的 4/8 个 n-tile），每个 warp 的
  tensor-core 工作量减半；因为 shared 不变仍是 1 block/SM，这样每个调度器有 2 个
  warp 来掩盖 `ldmatrix`/shared 延迟（1024 全局 attention 4 个块合计 **81 ms**）。
  - Q 常驻 shared（`sQhi/sQlo`，split-bf16，stride `RS=HD+8=72`）；每个 k-tile
    stage K（`[key][hd]`、split、stride RS）与 V（**转置** `[hd][key]`、split）——
    转置是因为 PV 的 B 操作数是 `V^T`（mma `row.col` 要求 B 为 `[n][k]` 行主序）。
  - `S = Q·K^T`：3 次 mma/面板（`QhiKhi + QhiKlo + QloKhi`），写出到 f32 shared
    `sS[64][32]`，同时按 `sRel` 查表加 relpos、把越界 key 置 `-inf`。
  - online softmax 用 64 个线程（一线程一行）在 `sS` 上算 max/exp，`P` 以 split-bf16
    写回 `sPhi/sPlo`；`sAlpha` 记录每行 `exp(m_old-m_new)`。
  - `O = O*alpha + P·V`：先按 `sAlpha` 缩放寄存器里的 `cO` fragment，再 3 次 mma
    （`PhiVhi + PhiVlo + PloVhi`）。`sRel` 是每 query 的 `[H+W]` relpos 查找表，
    在 block 起始算一次。
- **共享内存**：V/P 只有 `kTcBN` 列，用更窄的 stride `kTcRS2 = kTcBN+8 = 40`
  （仍 16B 对齐、ldmatrix 无 bank conflict）；bf16 面板共 `(2*BM + 2*BN)*RS +
  (2*HD + 2*BM)*RS2`。1024px（S=4096, H=W=64）时合计 **89856 B** < sm_120 的
  101376 B opt-in；更大的 S 自动回退 f32。
- **踩的坑（已修，详见 `PITFALLS.md` §20）**：`sVhi/sVlo` 是 `[HD][BN]` 的转置矩阵，
  却按 `kTcBN*RS`（32 行）分配/寻址，`d≥32` 的行越界写进 `sVlo`/`sPhi`，导致 V
  被破坏、输出 rel_l2≈1.1；同时 launcher 的 shared 字节数也少算了 V 的另外
  `(HD-BN)` 行。修复为按 `kAttnHD*RS2` 分配并改用 RS2。**这类错误在 f32 kernel 里
  不存在**（它不转置 V）。
- **验收**（`compare_vision_gpu --selftest` 新增 `H=W=32/S=1024`、输出**置零**的
  TC case）rel_l2 **9e-6**；真实模型 `--size 1024` visual rel_l2 **1.44e-4**
  （sam 4.6e-5 / clip 1.77e-4），编码 **366 → 238–240 ms**。见 `BENCHMARKS.md`
  §2.9。
- **剩余（非阻塞）**：每个 block ~90 KB shared → **1 block/SM**；8-warp n-split 已
  改善延迟掩盖，但 nsys 显示当前最大头已是 **`matmul_t_split_bf16`（98 ms）**，
  全局 attention 81 ms、windowed f32 attention 30 ms。进一步提速需缩减 `sRel`
  （32 KB，最大项）/改 register-resident flash attention，或优化视觉 GEMM
  （cp.async）。整栈 CUDA Graph 捕获与非方形尺寸支持仍未做。


## 6. 与 `prj.md` 三大创新点的对应

| prj.md 创新点 | 本项目实现 | 状态 |
|---|---|---|
| 固定+环形双层 KV 管理 | `RSWACache` + `BlockManager`（前缀引用计数 + 环形池） | 已实现 |
| CUDA Graph 下的动态 MoE 路由 | 路由在 Graph 内以 device kernel 求值、expert 以"全调度 + 掩码跳过"执行；单请求与 batched decode 均整步捕获 | 已实现（含连续批处理） |
| Prefill 的 KV 分区写入 | 参考实现并不丢弃 gap（见 PITFALLS §1），当前按参考语义；分区丢弃作为优化 TODO | 未实现优化 |

## 7. 端到端数据流（当前）

```
图像 → DeepEncoder(SAM+CLIP+projector) → 视觉 embedding [V,1280]
文本 token → embed_tokens → 在 <image> 位置替换为视觉 embedding
          → MoEDecoder 12 层（R-SWA attention + MoE）→ final RMSNorm
          → lm_head → logits → Sampler（greedy / no-repeat-ngram）
```

`Engine::generate()`（`src/engine/engine.cpp`）是单请求入口：
`prefill → 循环 decode → 采样 → 写回 cache`，并统计 TTFT/TPOT。
