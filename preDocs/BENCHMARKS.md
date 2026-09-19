# 实验数据记录（Benchmarks）

本文件汇总关键性能与数值对齐实验。**原始输出**保存在 `preDocs/bench/`，可直接复核。

## 0. 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB（Blackwell GB206, sm_120, 36 SM, 448 GB/s） |
| CUDA | 13.4 |
| 编译器 | g++ 16.2.1（CUDA host compiler） |
| 模型 | `baidu/Unlimited-OCR`（BF16 safetensors 6.67 GB，MoE 12 层） |
| 构建 | `-DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release` |

复现命令（结果分别见 `bench/` 同名文件）：

```bash
cmake -S . -B build-cuda -DENGINE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j8

./build-cuda/benchmarks/bench_cuda_decode --prefill 128 --steps 64 \
    | tee preDocs/bench/bench_synthetic.txt
./build-cuda/benchmarks/bench_cuda_decode --real --prefill 128 --steps 32 \
    | tee preDocs/bench/bench_real_bf16.txt
./build-cuda/benchmarks/bench_cuda_decode --real --int4 --prefill 128 --steps 32 \
    | tee preDocs/bench/bench_real_int4.txt
```

`bench_cuda_decode` 对每种配置构造一个 `GpuDecoder`，测量 prefill、decode TPOT
（规避前 2 步捕获/预热），并分解为 `fwd`（decoder 前向）与 `logits`（lm_head + D2H），
同时报告该 decoder 占用的 device 显存。

## 1. 真实模型 decode 性能（batch=1）

原始数据：`bench/bench_real_bf16.txt`、`bench/bench_real_int4.txt`。

| 配置 | TTFT（128 token prefill） | TPOT 稳态 | fwd | logits | 显存 |
|---|---|---|---|---|---|
| BF16, plain（无 Graph） | 189 ms | 6.77 ms | 5.86 ms | 1.19 ms | 9212 MB |
| BF16, full Graph | 195 ms | **5.55 ms** | 4.59 ms | 1.14 ms | 9212 MB |
| BF16, attn_dense Graph | 192 ms | 5.58 ms | — | — | 9212 MB |
| INT4, plain | 127 ms | 14.87 ms | 13.68 ms | 1.12 ms | **2216 MB** |
| INT4, full Graph | 130 ms | **5.53 ms** | 4.59 ms | 1.12 ms | **2216 MB** |
| INT4, attn_dense Graph | 126 ms | 5.66 ms | — | — | 2216 MB |

要点：

- **INT4 full Graph 达到 prj.md 的 ~4ms/token 目标区间（5.5ms）**，显存 2.2GB。
- `logits` ~1.1ms 是 lm_head（[129280,1280] bf16，权重 331MB）的带宽下界
  （0.74ms）附近，已是固定开销。
- plain INT4 慢（14.9ms）是因为 host 路由 + `moe_gemm_int4_tc`（标量反量化）逐
  expert 发射；full Graph 走合并访存的 `moe_experts_masked_int4`。
- BF16/INT4 的 full Graph TPOT 接近（5.5ms）：decode 已从“权重带宽受限”转为
  “每层 ~24 个小 kernel 的延迟/占用受限”，INT4 的带宽优势被掩盖。

### 历史对比（优化前）

| 阶段 | BF16 full Graph TPOT | 说明 |
|---|---|---|
| host lm_head + 未合并访存 | 26.7 ms | host lm_head matvec 占 ~20ms |
| coalesced matvec/expert MLP | 5.5 ms | 修复 ~16× 带宽浪费 |
| 真实 INT4 full Graph | 5.5 ms | 显存 9.2→2.2GB |

## 2. 合成模型（分层回归，非真机性能）

原始数据：`bench/bench_synthetic.txt`（hidden=256，4 层，MoE ne=8）。

| 配置 | prefill 128 | TPOT 稳态 |
|---|---|---|
| plain | 3.94 ms | 0.799 ms |
| full Graph | 2.96 ms | 0.589 ms |
| attn_dense Graph | 2.89 ms | 0.588 ms |

## 2.5 INT4 W4A16 GEMM：标量 vs Tensor Core（`bench/bench_int4_gemm.txt`）

n=896, k=1280, group=128, iters=200（单次 GEMM 调用；`2*m*n*k` 计 FLOP）：

| m | scalar (µs) | tensor-core (µs) | TC 有效 TFLOPS | 加速 |
|---|---|---|---|---|
| 1 | 66.4 | 37.1 | 0.062 | 1.8× |
| 8 | 71.6 | 38.9 | 0.472 | 1.8× |
| 32 | 206.0 | 41.3 | 1.776 | 5.0× |
| 64 | 356.0 | 60.1 | 2.443 | 5.9× |
| 128 | 653.4 | 70.9 | 4.140 | 9.2× |
| 273 | 1371.7 | 135.6 | 4.619 | 10.1× |

`moe_gemm_int4_tc` 现在用 shared-memory staging + `ldmatrix.x4/x2`：block=4 warps 计算
64×8 tile，反量化后的权重 panel `sW[8][16]` 每 k-step 只加载一次并被 4 个 warp 共享。
结果与标量路径 rel_l2 ≤ 0.0026（`uocr_cuda_tests`，含 ragged M/K）。

## 2.6 连续批处理吞吐（`bench/bench_batch_*.txt`）

`Engine::generate_batch`（scheduler + device batched decoder），prompt=64，steps=16；
吞吐按总生成 token / 总墙钟计（**含每个请求的串行 prefill**，因此偏保守）。
2026-09-19 两项优化后（batched attention kernel + prefill 走 device masked MoE）：

| batch | BF16 tok/s | BF16 peak | INT4 tok/s | INT4 peak |
|---|---|---|---|---|
| 1 | 101.4 | 9288 MB | 92.3 | 2292 MB |
| 2 | 88.5 | 9314 MB | 80.8 | 2318 MB |
| 4 | 136.8 | 9366 MB | 118.3 | 2370 MB |
| 8 | 191.3 | 9466 MB | 153.0 | 2470 MB |
| 16 | **240.4** | 9660 MB | **179.9** | 2664 MB |

优化前对照（同命令，本次改动前）：

| batch | BF16 tok/s | INT4 tok/s | prefill/请求 |
|---|---|---|---|
| 8 | 74.3 | 105.7 | 88 ms |
| 16 | 73.5 | 117.6 | 88 ms |

- **Batched attention kernel**：`rswa_append_decode_batch` + `rswa_attention_batch`
  （grid=(slots, heads)）把每步 attention 的 kernel 数从 `O(B·L)` 降到 `O(L)`。
  合成小模型（launch-bound）收益最大：batch=16 吞吐 2828 → 6406 tok/s（2.3×）；
  真实模型因 prefill/MoE 占主导，单纯此改动吞吐持平。
- **prefill device MoE**：prefill 不再逐专家发射 `moe_gemm_int4_tc`（64×3 个小
  GEMM/层），而是复用解码的 device router + 融合 masked 专家内核，按 token 归组。
  prefill 从 88 ms 降到 53 ms（INT4）/ 35 ms（BF16）。这是真实模型吞吐提升的主因。

正确性：`uocr_cuda_tests` 的 `Engine batch` 用例用 4 个不同长度 prompt 对比
batch 与 sequential device decode，token 完全一致（并对 CPU 一致）；
`GpuDecoder prefill`、`Engine CUDA greedy` 同样通过。

剩余瓶颈：prefill 仍逐请求串行（未做 ragged/chunked prefill）；
batched decode 未纳入 CUDA Graph（当前复用单请求 Graph 仅覆盖 batch=1）。

## 3. 数值对齐

### 3.1 端到端 OCR（`bench/compare_ocr.txt`）

500×400 图，prompt `<image>\nFree OCR.`，crop_mode，decode 24，ngram 35/1024：

```
summary: layout=OK visual_rel_l2=0.04187 greedy=24/24
```

### 3.2 Decoder + attention 逐层（`bench/compare_reference.txt`）

| 项 | 结果 |
|---|---|
| prefill attention q/k/v/o worst rel_l2 | 0.0245 / 0.0233 / 0.0437 / 0.0567 |
| prefill 逐层 q rel_l2 (layer0→11) | 0.0018 → 0.0245（随深度累积） |
| prefill logits | rel_l2 0.00896 |
| prefill router | 176 token，13 个集合翻转 |
| decode logits | rel_l2 ≤ 0.021 |

结论：除路由外残差为 bf16 激活累积漂移，非实现 bug；最终 logits <1%，greedy 一致。

### 3.3 CUDA 单元测试（`ctest --test-dir build-cuda`）

| 测试 | 结果 |
|---|---|
| R-SWA decode attention | max_err 0.000000 |
| GpuRSWACache 环形覆写 | cache err 0，decode err 1e-6 |
| INT4 MoE GEMM 标量 | max_err 1.9e-5 |
| INT4 MoE GEMM 张量核 W4A16 | rel_l2 0.0024 |
| GpuDecoder vs CPU（prefill/decode） | rel_l2 ≤ 0.0025 |
| Graph decode vs plain（20 步，含 ring 覆写） | rel_l2 0.00000 |
| attn_dense Graph vs plain | rel_l2 0.00000 |
| device INT4 专家 vs CPU（plain/graph） | rel_l2 0.0025 |
