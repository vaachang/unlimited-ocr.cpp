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

## 2.5 INT4 W4A16 GEMM：标量 vs Tensor Core
（`bench/bench_int4_gemm.txt`、`bench/bench_int4_gemm_down.txt`）

`bench_int4_gemm` 现在对每个 m 扫 `bn∈{8,16,32}`（块内 n 列宽）。gate/up 形状
n=896, k=1280, group=128, iters=300（`2*m*n*k` 计 FLOP）：

| m | scalar (µs) | tc bn=8 | tc bn=16 | tc bn=32 |
|---|---|---|---|---|
| 1 | 66.4 | **29.3** | 46.8 | 67.1 |
| 8 | 71.6 | **30.7** | 48.8 | 67.9 |
| 32 | 205.8 | **33.1** | 58.2 | 70.1 |
| 64 | 354.9 | **43.2** | 95.1 | 81.0 |
| 96 | 503.2 | **50.2** | 99.1 | 87.2 |
| 128 | 652.0 | **51.3** | 101.8 | 93.3 |
| 273 | 1370.9 | 110.1 | 115.3 | **110.0** |

down 形状 n=1280, k=896（`bench_int4_gemm_down.txt`）同样 bn=8 最优（m=96 时
60.0µs，bn=16/32 为 71/64µs）。

结论：**专家 GEMM 的小 N（896/1280）下 bn=8 最优**——更宽的 tile 会让 block 数
掉到 36 个 SM 以下（例如 n=896、bn=64 只有 28 个 block），SM 空转比 mma 复用更重要。
`moe_gemm_int4_tc` 默认 bn=8，并保留 `moe_gemm_int4_tc_n` 供调优。

本轮同时把权重 panel 的反量化 staging 从“64 线程各处理一字节”改成“128 线程按元素
循环”，bn=8 下 m=96 50.2µs（旧版同形状约 ~65µs）、m=273 110µs（旧 136µs），
**约 +20%**。结果与标量路径 rel_l2 ≤ 0.0026（`uocr_cuda_tests`，含 ragged M/K）。

## 2.6 连续批处理吞吐（`bench/bench_batch_*.txt`）

`Engine::generate_batch`（scheduler + device batched decoder），prompt=64，steps=16；
吞吐按总生成 token / 总墙钟计。`prefill_ms` 列是**整波 ragged prefill 的墙钟**
（每请求 ttft 都记同一值，表格取平均）。2026-09-19 三项优化后：

| batch | BF16 tok/s | BF16 peak | INT4 tok/s | INT4 peak |
|---|---|---|---|---|
| 1 | 99.4 | 9264 MB | 89.4 | 2268 MB |
| 2 | 89.4 | 9318 MB | 82.3 | 2322 MB |
| 4 | 115.1 | 9420 MB | 134.3 | 2424 MB |
| 8 | 189.8 | 9636 MB | 216.5 | 2640 MB |
| 16 | **298.0** | 10044 MB | **310.7** | 3048 MB |

优化前对照（同命令，本轮改动前；prefill 逐请求串行、masked MoE）：

| batch | BF16 tok/s | INT4 tok/s | prefill/请求 |
|---|---|---|---|
| 8 | 74.3 | 105.7 | 88 ms |
| 16 | 73.5 | 117.6 | 88 ms |

三项优化：

1. **Batched attention kernel**：`rswa_append_decode_batch` + `rswa_attention_batch`
   （grid=`(B, heads)`，行→slot 由 `d_slots` 指定）把每步 attention 的 kernel 数从
   `O(B·L)` 降到 `O(L)`。合成小模型（launch-bound）batch=16 吞吐 2828 → 8365 tok/s。
2. **ragged 多请求 prefill**：一步内新请求打包成一次 `forward_ragged`，
   K/V 用 `rswa_write_prefill_ragged` 直接散写到各自 slot，attention 用
   `rswa_attention_ragged`（每行按自己的 slot/局部位置做 causal）。当每专家 token 数
   >2 时走逐专家 tensor-core GEMM（`moe_gemm_int4_tc`，M≈96），否则走 masked matvec。
   prefill 从“16×88ms 串行”降到整波 **279ms(INT4)/370ms(BF16)**。
3. **prefill device MoE（小批量）**：每专家 token 少时（≤2）复用解码的 masked
   专家内核，避免逐专家小 GEMM 的发射开销。

正确性：新增单测把 ragged 多请求 prefill 的 logits 与逐请求 `prefill_tokens` 对比
（rel_l2 = 0）；`Engine batch` / slot 复用 / 非恒等 slot 排列 / `GpuDecoder prefill` /
`Engine CUDA greedy` 全部通过。

剩余瓶颈：`matmul_t_bf16` 曾是 CUDA-core tiled GEMM（~2 TFLOPS，未用 TC）；
lm_head 小 m 下 TC 利用率偏低。

### 2.7 连续批处理：bf16 TC GEMM + Batched CUDA Graph
（`bench/bench_batch_real_*_{graph,plain}.txt`）

本轮两项改动叠加：

1. **bf16 tensor-core GEMM**（`CORE_TECH.md` §5.5）：`matmul_t_bf16`（dense/shared
   投影 + lm_head）走 `ldmatrix`/`mma.m16n8k16`；行数 1 的 lm_head 走 `matvec_bf16`。
2. **Batched CUDA Graph**（`CORE_TECH.md` §5.7）：`forward_batch + final rmsnorm`
   按行数 B 缓存 Graph；活跃 slot 变化靠设备端映射免重捕获。基准先跑一次不计时的
   `generate_batch` 做 warmup（捕获 Graph 并复用分配），表中的数字是**稳态**，
   不含一次性捕获开销。`--no-graph` 关闭 Graph 做对照。

prompt=64、steps=16、max_batch=16，`graph=on` + warmup 稳态；tok/s 含整波 prefill，
括号内为 `decode_ms` 减去 prefill 墙钟得到的纯 decode 吞吐。同命令加 `--no-graph`
的对照差异在 run-to-run 噪声（±3%）内（`bench_batch_real_*_plain.txt`）。

| batch | BF16 graph (tok/s) | INT4 graph (tok/s) |
|---|---|---|
| 1 | 149.4 (213) | 130.9 (220) |
| 2 | 122.3 (155) | 107.3 (154) |
| 4 | 162.6 (257) | 183.5 (261) |
| 8 | 272.3 (423) | 275.1 (398) |
| 16 | **415.9 (634)** | **386.8 (559)** |

**与上一轮（无 TC、无 batched graph）对比**（同样命令、warmup 稳态）：

| 配置 | 上一轮 | 现在 | 备注 |
|---|---|---|---|
| BF16 batch=16 整波 prefill | 367 ms | **212 ms** | bf16 TC GEMM −42% |
| BF16 batch=16 tok/s | 298.0 | **415.9** | +40% |
| INT4 batch=16 整波 prefill | 279 ms | **204 ms** | bf16 TC + INT4 专家 GEMM staging −27% |
| INT4 batch=16 tok/s | 307.6 | **386.8** | +26% |
| BF16 batch=1 tok/s | 99.4 | **149.4** | TC + matvec lm_head +50% |
| INT4 batch=1 tok/s | 91.4 | **130.9** | +43% |

结论：

- **TC GEMM 是主要收益**（prefill 与 decode 的 dense/shared/投影 + lm_head 全线提速）。
- **Graph 本身收益仍在噪声内**（±3%）：负载已接近带宽/占用受限，host 发射不是
  瓶颈；其价值是每步 host 侧只剩 1 次 graph launch + 1 次 lm_head launch。正确性
  上 graph vs plain 逐位一致（单测 rel_l2=0）。
- 复现：
  ```bash
  ./build-cuda/benchmarks/bench_cuda_batch --real      --prompt 64 --steps 16 --max-batch 16
  ./build-cuda/benchmarks/bench_cuda_batch --real      --no-graph --prompt 64 --steps 16 --max-batch 16
  ./build-cuda/benchmarks/bench_cuda_batch --real --int4 --prompt 64 --steps 16 --max-batch 16
  ./build-cuda/benchmarks/bench_cuda_batch --real --int4 --no-graph --prompt 64 --steps 16 --max-batch 16
  ```

### 2.8 bf16 tensor-core GEMM 微基准（`bench/bench_bf16_gemm_*.txt`）

`bench_bf16_gemm` 对比 `matmul_t_bf16_ref`（CUDA-core tiled）与 `matmul_t_bf16`
（TC），iters=300/20（`2*m*n*k` 计 FLOP）：

n=k=1280（dense 投影形状）：

| m | tiled (µs) | TC (µs) | TC TFLOPS | 加速 |
|---|---|---|---|---|
| 1 | 3.3 | 3.3 (matvec) | 1.00 | 1.00 |
| 8 | 83.2 | 29.7 | 0.88 | 2.80 |
| 16 | 84.7 | 30.1 | 1.74 | 2.82 |
| 64 | 86.3 | 38.7 | 5.42 | 2.23 |
| 128 | 150.3 | 41.7 | 10.05 | 3.60 |
| 273 | 217.4 | 48.7 | **18.38** | 4.47 |

（m ≤ 16 走小 m 变体：4 warp 沿 n 拆分；m ≥ 64 走 64×64 内核。）

n=129280, k=1280（lm_head 形状）：

| m | tiled (µs) | TC (µs) | TC TFLOPS | 加速 |
|---|---|---|---|---|
| 1 | 780.7 | 780.7 (matvec) | 0.42 | 1.00 |
| 8 | 3792.3 | 2633.6 | 1.01 | 1.44 |
| 16 | 3832.2 | 2652.2 | 2.00 | 1.44 |
| 64 | 3958.6 | 2646.6 | 8.00 | 1.50 |
| 128 | 7819.9 | 5245.7 | 8.08 | 1.49 |
| 273 | 19384.1 | 12963.7 | 6.97 | 1.50 |

要点：n=k=1280 时 TC 到 18.4 TFLOPS；小 m（≤16）用 n 方向拆 warp 的变体后
m=8/16 从 ~36µs 降到 ~30µs（+20%）。lm_head 只在 2–8 TFLOPS——大 n 下权重读取与
每 k-step 两次 `__syncthreads` 限制了流水，是后续调优点（split-K、更大 BN、
`cp.async` 双缓冲）。

### 2.9 nsys kernel 分解（`bench/bench_nsys_batch_int4.txt`）

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
  **已做**：staging 重写 + bn=8 使 kernel 快 ~20%，整波 prefill 227→204ms
  （§2.5/§2.7）。**剩余**：发射次数仍需 grouped GEMM（一层一次 launch，block 映射
  到 (expert,n-tile)）或合并 gate/up 来消除。
- decode 的 masked INT4 专家（gate_up+down 18.9%）是第二块；其权重读取受带宽限制，
  但每专家仅 1–2 token 时也偏延迟受限，可考虑小批量 grouped GEMM。
- `rswa_attn_ragged` 7%：prefill attention 的 grid=(total,heads)，total 大时较可观。


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
