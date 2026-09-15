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
| `moe_gemm_int4.cu` | `moe_gemm_int4_kernel` | AWQ 反量化 + FP32 累加（tensor-core mma 留作后续） |
| `rmsnorm.cu` | `rmsnorm_kernel` | 块内归约 |
| `rope_fused.cu` | `rope_kernel`, `rmsnorm_head_kernel` | 融合逐头 RMSNorm + RoPE |
| `backend.cu` | `matmul_t(_bf16)`, `device_info` | 稠密 GEMM、设备信息 |

CUDA 构建与 CPU 构建共用 `RSWACache`/`WeightMatrix` 的数据布局，因此
`tests/test_rswa_cuda.cu` 可以直接把 host cache 上传后与 CPU 参考逐元素对比
（当前 max_err < 1e-5）。

## 6. 与 `prj.md` 三大创新点的对应

| prj.md 创新点 | 本项目实现 | 状态 |
|---|---|---|
| 固定+环形双层 KV 管理 | `RSWACache` + `BlockManager`（前缀引用计数 + 环形池） | 已实现 |
| CUDA Graph 下的动态 MoE 路由 | 路由在 Graph 外求值、expert 在 Graph 内"全调度 + 掩码跳过" | 设计保留，未接入 Graph |
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
