# 实验数据记录（Benchmarks）

> 开发入口见 `tAgent.md`；文档导航见 `README.md`。

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

## 2.5 INT4 W4A16 GEMM：标量 vs Tensor Core（P3 更新）
（`bench/bench_int4_gemm.txt`、`bench/bench_int4_gemm_down.txt`）

`bench_int4_gemm` 对每个 m 扫 `bn∈{8,16,32} × bk∈{16,32,64}`，再加 4 个 cp.async
`pipe` 变体。gate/up 形状 n=896, k=1280, group=128, iters=300（`2*m*n*k` 计 FLOP）：

| m | scalar (µs) | bn8/bk16（旧默认） | **bn8/bk64（新默认）** | pipe8 |
|---|---|---|---|---|
| 1 | 66.4 | 29.1 | **25.2** | 26.1 |
| 8 | 71.6 | 29.8 | **28.0** | 26.9 |
| 32 | 205.6 | 34.8 | 36.3 | 36.1 |
| 64 | 354.5 | 50.0 | **47.9** | 47.4 |
| 96 | 502.5 | 54.0 | **49.1** | 58.8 |
| 128 | 651.4 | 53.5 | **50.4** | 69.4 |
| 273 | 1369.3 | 115.3 | **97.3** | 136.5 |

down 形状 n=1280, k=896 结论一致。要点：

- **激活 staging 从逐元素标量改为 `float4` 向量化**（每线程 8→2 条载入指令，写入
  带 padding 的 bf16 panel），`bn=8/bk64` 在 m=273 从 112 → **97µs**。
- **`bk` 从 16 提到 64** 把每个 GEMM 的 k-step 从 80 降到 20，同步/发射开销摊薄。
- **`bn=8` 仍最优**：更宽的 tile 会让 block 数掉到 36 个 SM 以下，延迟隐藏变差
  （例如 n=896、bn=64 只有 14 个 block）。
- **cp.async `pipe` 变体只在 m≤32 略优**（m=1 26.1 vs 25.2µs），在 m≈96（prefill
  平均每专家 token 数）反而更慢（58.8µs），故不作默认；保留供对照。
- 结果与标量路径 rel_l2 ≤ 0.0026（`uocr_cuda_tests`，含 ragged M/K）。

## 2.6 连续批处理吞吐（真实模型）

`Engine::generate_batch`（scheduler + device batched decoder），prompt=64、steps=16、
max_batch=16。下面表格的 tok/s 含整波 prefill，括号内为 `decode_ms` 减去 prefill 墙钟
得到的纯 decode 吞吐；`bench_cuda_batch` 先跑一次不计时的 `generate_batch` 做 warmup
（捕获 Graph 并复用分配），因此是**稳态**数字。原始输出见
`bench/bench_batch_real_*.txt`。

设计与优化（按加入顺序）：

1. **batched attention**：`rswa_append_decode_batch` + `rswa_attention_batch`
   （grid=`(B, heads)`，行→slot 由 `d_slots` 指定）把每步 attention 的 kernel 数从
   `O(B·L)` 降到 `O(L)`。
2. **ragged 多请求 prefill**：一步内新请求打包成一次 `forward_ragged`，K/V 用
   `rswa_write_prefill_ragged` 散写到各自 slot，`rswa_attention_ragged` 逐行 causal；
   K/V 按 slot 索引、激活按行索引（`CORE_TECH.md` §5.6）。
3. **prefill MoE 路径选择**：每专家 token 少时走 masked device MoE，多时走逐专家
   tensor-core GEMM（`moe_gemm_int4_tc`）。
4. **bf16 TC GEMM**（`CORE_TECH.md` §5.5）：dense/shared + lm_head 走上 `ldmatrix`/
   `mma`；行数 1 的 lm_head 走 `matvec_bf16`。
5. **Batched CUDA Graph**（`CORE_TECH.md` §5.7）：`forward_batch + final rmsnorm`
   按行数 B 缓存；活跃 slot 变化靠设备端映射免重捕获。
6. **INT4 专家 GEMM 调优**（§2.5）：BN 模板 + staging 重写。

当前结果（`--no-graph` 对照差异在 ±3% 噪声内；括号内为 `decode_ms − prefill_ms`
换算的纯 decode 吞吐）：

| batch | BF16 tok/s | INT4 tok/s（grouped, 2026-09-20） |
|---|---|---|
| 1 | 155.8 (221) | 138.2 |
| 2 | 179.3 (253) | 180.6 |
| 4 | 253.0 (386) | 292.4 |
| 8 | 379.2 (578) | 394.3 |
| 16 | **521.6 (776)** | **487.6**（多次 488–516） |

> INT4 列在 P3 grouped GEMM（§2.10）后更新；同一命令连续运行 tok/s 波动约 ±5%
> （decode 占主导）。原始输出：`bench/bench_batch_real_int4_grouped.txt`。

累计提升（同一命令、warmup 稳态）：

| 配置 | 起始值 | 现在 |
|---|---|---|
| BF16 batch=16 tok/s | 73.5 | **521.6** |
| INT4 batch=16 tok/s | 117.6 | **429.3** |
| 整波 prefill（16 请求） | 16 × 88 ms 串行 | **161 ms** BF16 / **213 ms** INT4 |
| BF16 / INT4 batch=1 tok/s | 99.4 / 91.4 | 155.8 / 132.9 |

要点：

- 收益主要来自 **TC GEMM**（prefill 与 decode 的 dense/shared/投影 + lm_head 全线提速）
  与 **INT4 专家 GEMM 调优**；**Graph 本身在带宽受限负载下收益在噪声内**，价值是每步
  host 侧只剩 1 次 graph launch + 1 次 lm_head launch（正确性 graph vs plain 逐位一致）。
- **P3**：bf16 TC GEMM 换 cp.async 流水线后 BF16 batch=16 **416→521.6** tok/s、
  整波 prefill **212→161 ms**；INT4 默认 `bn=8/bk=64` + 向量化 staging 后 batch=16
  **387→429.3** tok/s。lm_head 小 m 收益最大（§2.7）。
- 复现：
  ```bash
  ./build-cuda/benchmarks/bench_cuda_batch --real      --prompt 64 --steps 16 --max-batch 16
  ./build-cuda/benchmarks/bench_cuda_batch --real      --no-graph --prompt 64 --steps 16 --max-batch 16
  ./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
  ./build-cuda/benchmarks/bench_cuda_batch --real --int4 --no-graph --prompt 64 --steps 16 --max-batch 16
  ```

## 2.7 bf16 tensor-core GEMM 微基准（`bench/bench_bf16_gemm_*.txt`）

`bench_bf16_gemm` 对比 `matmul_t_bf16_ref`（CUDA-core tiled）与 `matmul_t_bf16`
（P3：cp.async 流水线），iters=300/20（`2*m*n*k` 计 FLOP）：

n=k=1280（dense 投影形状）：

| m | tiled (µs) | TC (µs) | TC TFLOPS | 加速 |
|---|---|---|---|---|
| 1 | 3.2 | 3.2 (matvec) | 1.01 | 1.00 |
| 8 | 83.2 | **16.8** | 1.56 | 4.96 |
| 16 | 84.7 | **17.2** | 3.06 | 4.94 |
| 64 | 86.3 | 45.6 | 4.60 | 1.89 |
| 128 | 150.3 | 47.8 | 8.78 | 3.15 |
| 273 | 217.5 | 50.9 | 17.59 | 4.28 |

n=129280, k=1280（lm_head 形状）：

| m | tiled (µs) | TC (µs) | TC TFLOPS | 加速 |
|---|---|---|---|---|
| 1 | 780.1 | 780.1 (matvec) | 0.42 | 1.00 |
| 8 | 3806.1 | **1147.5** | 2.31 | 3.32 |
| 16 | 3846.6 | **1173.9** | 4.51 | 3.28 |
| 64 | 3966.4 | **1309.0** | 16.18 | 3.03 |
| 128 | 7853.5 | 2653.2 | 15.97 | 2.96 |
| 273 | 19403.6 | 6411.4 | 14.09 | 3.03 |

要点：

- **m≤16**（decode / batched lm_head）走 n 拆分小 tile + cp.async：n=k=1280 从 ~31 →
  **17µs**，lm_head 从 ~2650 → **~1150µs（2.3×）**。
- **m>16** 走 cp.async 64×64：n=k=1280 的 m=64–273 与旧同步内核相当（±10%），但**宽 n**
  的 lm_head 全线 **~2×**（m=273 12963 → 6411µs）——权重带宽受限时流水收益最大。
- `matmul_t_bf16` 在 `k % 8 ≠ 0` 时回退到 `matmul_t_bf16_ref`（cp.async 16B 拷贝的
  对齐要求）；`m==1` 走 `matvec_bf16`。

## 2.8 nsys kernel 分解（`bench/bench_nsys_batch_int4.txt`）

命令（INT4，`steps=4` 以缩小 trace；包含 B=1..16 与 warmup，故是 prefill+decode 的
混合统计）：

```bash
nsys profile --stats=true -o nsys_batch ./build-cuda/benchmarks/bench_cuda_batch \
    --real --int4 --prompt 64 --steps 4 --max-batch 16
```

GPU kernel 时间占比（top）：

| kernel | 占比 | instances | avg |
|---|---|---|---|
| `moe_gemm_int4_tc_kernel` | **55.4%** | 12096 | 55 µs |
| `expert_gate_up_int4_kernel`（decode masked） | 12.6% | 44 | 3.44 ms |
| `matmul_t_bf16_tc_kernel` | 9.3% | 950 | 117 µs |
| `matmul_t_bf16_tc_small_kernel` | 7.0% | 32 | 2.64 ms |
| `rswa_attn_ragged_kernel<128>`（ragged prefill） | 7.0% | 120 | 0.70 ms |
| `expert_down_int4_kernel`（decode masked） | 6.3% | 44 | 1.71 ms |
| 其余（scatter/gather/silu/rmsnorm/rope） | <2% | — | — |

host API：`cudaMemcpy` 76%（2.46s，主要是权重上传 2.08GB）、`cudaStreamSynchronize`
13%（427ms）、`cudaLaunchKernel` 27584 次 / 99ms；`cudaGraphLaunch` 仅 30 次——说明
CUDA Graph 确实把逐步发射压到常数级。

解读与下一步：

- **prefill 的逐专家 `moe_gemm_int4_tc` 是最大头**：12096 次小 GEMM（avg 55µs）。
  每个 prefill 波对 11 层 × 64 专家 × {gate,up,down} 各发一次，M≈tokens/expert。
  **已做**：staging 重写 + bn=8 使 kernel 快 ~20%（§2.5/§2.6）；随后 **grouped
  GEMM（§2.10）** 把一层 ~192 次 launch 压到 2 次、去掉每层 router D2H，整波
  prefill 213→**120 ms**。**本节表格为 grouped 之前的快照。**
- decode 的 masked INT4 专家（gate_up+down 18.9%）是第二块；其权重读取受带宽限制，
  但每专家仅 1–2 token 时也偏延迟受限，可考虑小批量 grouped GEMM。
- `rswa_attn_ragged` 7%：prefill attention 的 grid=(total,heads)，total 大时较可观。


## 2.9 DeepEncoder (vision) CUDA 移植（`bench/` 无固定文件；用 `compare_vision_gpu`）

`tools/compare_vision_gpu` 在真实视觉权重上对比 GPU 编码器与 CPU 参考编码器
（CPU 编码器已对齐 PyTorch f32 参考，见 `ALIGNMENT.md` §4），随机归一化图像：

| size | GPU 编码 | CPU 编码 | visual rel_l2 | clip rel_l2 | sam rel_l2 |
|---|---|---|---|---|---|
| 224 | **34 ms** | — | — | — | — |
| 640 | **153 ms** | ~43 s | 9e-6 | 1.1e-5 | 2e-6 |
| 1024 | **612 ms** | ~2 min | **1.1e-5** | 1.3e-5 | 2e-6 |

验收线为 rel_l2 ≤ 6e-4（tAgent 2.3），实测约 **50×** 余量。GPU 端用 f32 GEMM
（`matmul_t_f32w`）而非 bf16 tensor core：bf16 激活舍入在深层视觉栈里累积到 ~11%。
`--selftest` 的 layernorm / relpos attention / full attention rel_l2 ≤ 1e-6。

## 2.10 grouped INT4 专家 GEMM（2026-09-20，`bench/bench_batch_real_int4_grouped.txt`）

device router + grouped expert GEMM（`CORE_TECH.md` §5.9）。命令同 §2.6：

```bash
./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
```

整波 prefill（16 请求，改变前后）：

| 指标 | 优化前（逐专家 host 路径） | grouped（bn=32/bm=128） |
|---|---|---|
| 整波 prefill (B=16) | **213 ms** | **120 ms** |
| B=1 / 2 / 4 / 8 prefill | 48 / 88 / 76 / 113 ms | 43 / 49 / 59 / 80 ms |
| INT4 B=16 tok/s | 429 | 488–516 |

tile 扫描（B=16 prefill ms，`--only-batch 16`，`preDocs/bench/bench_grouped_tuning.txt`）：

| bn\bm | 64 | 128 |
|---|---|---|
| 8 | 152.7 | 147.3 |
| 16 | 145.5 | 124.2 |
| 32 | 157.5 | **121.2** |
| 64 | 199.5 | 138.3 |

要点：**BM=128 是最大收益**（8 warp×16 行，每专家的 m-tile 数减半，权重不再重复
stage）；`BN` 太小则 A 面板复用差、太大则 block 数/占用下降，32 是拐点。

B=16 单次 prefill 的 kernel 分解（`bench/bench_nsys_batch_int4_grouped.txt`，
两次 forward，故除以 2）：

| kernel | 每次 prefill | 说明 |
|---|---|---|
| `moe_grouped_gate_up_int4_kernel<32,64,128>` | 44 ms | 一层一次（11 层），含 gate+up+SiLU |
| `rswa_attn_ragged_kernel<128>` | 21 ms | prefill attention（新的第二大头，未优化） |
| `moe_grouped_down_int4_kernel<32,64,128>` | 21 ms | 一层一次，含 scatter |
| `matmul_t_bf16_tc_pipe_kernel<64,3,false>` | 20 ms | q/k/v/o + shared 投影 |
| `matmul_t_bf16_tc_pipe_kernel<64,4,true>` | 5 ms | 16 请求的 lm_head |
| `moe_router_topk_kernel` | 0.4 ms | device router |

host API 侧不再有每层 router 的 D2H：整段 trace 只有 8 次 D2H（都是最后一次
`[requests, vocab]` logits 回读，66 MB）。权重上传（2.07 GB H2D）仍占 host API 大头
（任务 2.9）。

## 3. 回归快照

数值对齐的方法、逐项结果与根因分析统一记录在 **`ALIGNMENT.md`**（端到端 OCR
greedy 24/24、视觉对 f32 参考 rel_l2 ≤ 5.8e-4、逐层 attention q/k/v/o、R-SWA
环形覆写、decoder logits 等）；原始输出在 `bench/compare_*.txt`。此处只保留
CUDA 单元测试一览。

CUDA 单元测试（`ctest --test-dir build-cuda` 的 `uocr_cuda_tests`）：

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
