# 核心技术实现说明

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
| `rmsnorm.cu` | `rmsnorm_kernel` | 块内归约 |
| `rope_fused.cu` | `rope_kernel`, `rmsnorm_head_kernel` | 融合逐头 RMSNorm + RoPE |
| `backend.cu` | `matmul_t(_bf16)`, `device_info` | 稠密 GEMM、设备信息 |
| `gpu_ops.cu` | `silu_mul`, `add_scaled`, `gather_rows`, `scatter_add_scaled` | 设备端逐元素 / 分组 MoE 辅助 |
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
`BENCHMARKS.md` §2.5）：模板参数 `BN` 控制每 block 的 n 列宽，专家 GEMM 的
N=896/1280 下经验最优是 **BN=8**（更宽的 tile 会让 block 数掉到 36 个 SM 以下）。
权重 panel 的反量化 staging 由 128 线程按元素循环完成；`moe_gemm_int4_tc_n` 可扫
bn∈{8,16,32,64}。

### 5.4 合并访存内核（P2 性能）

`matvec_bf16`（warp-per-output，每 lane 向量化读 2 个 bf16）与 expert MLP 的
gate_up/down 内核修复了“相邻线程按行 stride 读权重”导致的 ~16× 带宽浪费。
`matmul_t_bf16` 的 CUDA-core 参考版保留为 64×64 分块 + shared memory staging
（panel 补 1 列避免 bank conflict），见 §5.5 的 tensor-core 取代。

### 5.5 bf16 tensor-core GEMM（P2，2026-09-19）

`matmul_t_bf16`（dense/shared 投影与 lm_head 的主力）从 CUDA-core 分块 GEMM
换成 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`：

- block = 4 warps 计算 64(m)×64(n) tile，`kTCM=64/kTCN=64/kTCK=16`。激活值 staging
  时 `__float2bfloat16`；权重已是 bf16 直接搬。A 用 `ldmatrix.x4`、B（`W[n][k]`
  行主序 = mma 的 col-major）用 `ldmatrix.x2`，每个 warp 负责 16 行 m、遍历 8 个
  n8 子块。grid = `(ceil(n/64), ceil(m/64))`，block=128 线程。
- **小 m 不浪费**：m 不足 64 时，16 行切片完全越界的 warp 跳过 mma 循环（仍参与
  `__syncthreads`），因此 decode 的 m=16 不会为 64 行 tile 付出 4× 计算。
- **小 m 变体**（m ≤ 16）：另一个 BM=16/BN=64 的内核让 4 个 warp **沿 n 方向**
  各算 16 列（各 2 个 n8 子块），避免上面 64 行内核在 m=16 时只有 1 个 warp 做
  mma。`matmul_t_bf16` 在 m ≤ 16 时分派到它；n=k=1280、m=16 时
  36.1→30.1µs（1.74 TFLOPS，+20%）。
- **回退/对照**：`matmul_t_bf16_ref`（原 CUDA-core 分块内核）保留，供单测 A/B；
  单测在 `m∈{3,16,40,64,273}`、`k∈{48,64,128,160,256}` 上 TC vs ref
  rel_l2 ≤ 0.0018。
- `batch_decode`/ragged prefill 的 lm_head 在行数 ==1 时走 `matvec_bf16`
  （避免用 GEMM 处理单行）。
- 微基准（`bench/bench_bf16_gemm_*.txt`，RTX 5060 Ti）：
  - n=k=1280：m=16 时 84.7→30.1µs（2.8×），m=273 时 217.4→48.7µs（4.5×，18.4 TFLOPS）。
  - lm_head n=129280、k=1280：m=16 时 3832→2652µs（1.44×，2.0 TFLOPS）——即使 4 个
    warp 全用上仍受权重读取/同步限制，是后续调优点（split-K、更大 BN、`cp.async`）。
- 真实模型连续批处理（`BENCHMARKS.md` §2.7）：BF16 batch=16 298→**417** tok/s，
  整波 prefill 367→207ms；INT4 batch=16 308→**374** tok/s。

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
    一次，prefill 从串行 16×88ms 降到整波 279ms(INT4)。
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
