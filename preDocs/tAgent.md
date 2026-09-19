按照preDocs/prj.md中的内容实现这个项目。
实现这个项目的过程中，有不明白、不清楚的地方问我。需要安装第三方依赖时请让我做决定，不要擅自安装。
保留中间进度文档、重要性能测试结果、踩过的坑、核心技术实现等内容，放在preDocs目录下。
使用g++作为C++编译器。
记得使用git保存项目。

---

## 下一阶段任务计划（2026-09-19 四次更新）

进度与结果见 `PROGRESS.md`、`ALIGNMENT.md`、`PITFALLS.md`、`CORE_TECH.md`、
`BENCHMARKS.md`（含 `bench/` 原始输出）。

**已完成里程碑**：M1 调研 → M2 骨架 → M3 调度 → M4 引擎 → M5 CUDA 内核
→ M6 视觉对齐 → P0 端到端 OCR 对齐（E0–E5）→ P1 R-SWA 环形覆写验证
→ P1 CUDA 设备端 decoder + Engine CUDA 分支 → P2 Tensor Core W4A16 GEMM
→ P2 CUDA Graph（device router + 固定调度掩码跳过）→ P2 device INT4 专家权重
→ P2 合并访存内核与分块 prefill GEMM → P1 attention 逐层残差定位
→ P2 TC `ldmatrix`/shared-memory staging → 连续批处理接入 device decoder
→ batched R-SWA attention/append 内核 → ragged 多请求 prefill → slot 映射 bug 修复
→ **batched decode 纳入 CUDA Graph** → **bf16 tensor-core GEMM（dense/shared + lm_head）**
→ **bf16 小 m 变体** → **INT4 专家 GEMM bn 调优 / staging 重写**。

最终回归：`compare_ocr` greedy **24/24**（CPU 参考路径）；`uocr_cuda_tests` 全过
（含 batched 排列 / ragged prefill / slot 复用 / batched graph / TC vs ref）；
`uocr_tests` 20/20。

**连续批处理性能（真实模型，prompt=64，steps=16，max_batch=16，warmup 稳态）**：

| 配置 | 起始值 | 现在 |
|---|---|---|
| BF16 tok/s（batch=16） | 73.5 | **415.9** |
| INT4 tok/s（batch=16） | 117.6 | **386.8** |
| 整波 prefill（16 请求） | 16 × 88 ms 串行 | **204 ms** INT4 / **212 ms** BF16 |
| BF16 / INT4 batch=1 tok/s | 99.4 / 91.4 | 149.4 / 130.9 |

> 累计改动：① batched attention（O(B·L)→O(L)）；② ragged prefill；③ 按专家规模选
> masked matvec / TC GEMM；④ batched decode CUDA Graph；⑤ bf16 TC GEMM（dense/shared
> + lm_head，含小 m 变体）；⑥ INT4 专家 GEMM bn 调优 + staging 重写。
> 详见 `BENCHMARKS.md` §2.5–2.9、`CORE_TECH.md` §5.5–5.7。

**下一步优先级**（按收益/成本排序）：

1. **高性能 TC GEMM 重写**（当前最大剩余项）：现有 TC 内核每个 warp 每 k-step 只发
   1 个 mma + 2 次 `__syncthreads`，流水太短，实测仅 4–18 TFLOPS（硬件上限 ~80）。
   做更大 tile（128×128 / 256×128）+ register tiling + `cp.async` 双缓冲 + swizzle。
   收益覆盖：prefill 专家 GEMM（nsys 占 55% kernel 时间）、bf16 dense/shared、
   lm_head（大 n 仅 1.5–8 TFLOPS）。可与第 2 项合并做。
2. **Grouped expert GEMM**：`forward_ragged` 每层逐专家发 3×64 个
   `moe_gemm_int4_tc`（12096 次/整轮、avg 55µs）。改为“一层一次 launch，block 映射到
   (expert, m-tile, n-tile)”可去除 ~2000 次小发射；与第 1 项一起做收益最大。
3. **Prefill KV 分区写入优化**（prj.md 创新点三）：按位置分区（视觉区/环形区/gap
   丢弃），省 ~70% prefill KV 写入带宽。**注意**：参考实现并不丢弃 gap
   （`PITFALLS.md` §1），改动会偏离参考数值，需先确认是否接受。
4. **精度评测**：OmniDocBench v1.6（AWQ vs BF16 综合分）、AWQ vs 朴素 INT4 消融。
5. **性能记录补全**：ncu 采 SM/DRAM 峰值利用率（nsys kernel 分解已完成，见
   `BENCHMARKS.md` §2.9）、KV Cache 碎片率、纯 decode 的 TTFT/TPOT 分解。
6. **权重加载优化**（prj.md 6.1）：`mmap` + `cudaHostRegister` pinned DMA 直通
   （nsys 显示权重上传 2.08GB H2D 占 host API 76%）。

**已知遗留/技术债**：
- `GpuDecoder::mlp_block` 的 host 路由分支（`dev_moe=false`，ragged 大批量 prefill
  走此分支）每层做一次 router D2H + host top-k；可与 grouped GEMM 一起下放设备端。
- `mlp_block_batch` 为未定义的空声明，可删除。
- `batch_import_prefill` 已无调用者，可删除。
- 真实 INT4 模型下 CUDA 端 greedy 尚未与参考 OCR 做端到端回归（当前 OCR 对齐
  走 CPU 参考路径）。
- release 构建下 `matmul_t_bf16_ref` 仅用于单测 A/B，保留。

### P0 视觉编码器数值对齐（已完成 2026-09-15）
- [x] 跑完 `tools/compare_vision.cpp`。`DeepEncoder::encode` 输出 273×1280 与
      参考 `visual_embeddings`：对 f32 参考 rel-L2 ≤ 5.8e-4，对 bf16 参考 3.2%。
- [x] 分段对比：已加 `DeepEncoder::encode_stages` + `--dump-stages` +
      `tools/reference/export_vision_stages.py`，定位并修复了 SAM 相对位置漏乘
      query、CLIP 漏加 QKV bias 两个 bug。CPU 侧启用 OpenMP（8 线程约 2 分钟）。
      详见 `ALIGNMENT.md` §4、`PITFALLS.md` §9。

### P0 端到端图像 OCR 对齐（已完成 2026-09-19）
目标：C++ `Engine` 输入单张真实页面图 + 文本 prompt，输出与参考
`UnlimitedOCRForCausalLM.infer()` 一致的 token 序列；先对齐首 token logits，
再逐步覆盖 greedy。**E0–E5 全部完成，端到端 greedy 24/24 与参考一致。**

1. [x] **DeepSeek BPE 预分词对齐**（E0）
   - `src/runtime/tokenizer.cpp` 已改为复刻 `tokenizer.json` 的 3 条 `Split`
     正则（`\p{N}{1,3}`、CJK、主标点/单词正则），Unicode 类别表由
     `tools/gen_unicode_tables.py` 生成到 `include/uocr/unicode_tables.h`。
   - 回归：`tools/compare_tokenizer` + `tools/reference/export_tokenizer_cases.py`
     在 43 个用例上 pretok/ids **43/43**；单测 `tokenizer_deepseek_pretokenizer`。
2. [x] **prompt 与 `<image>` 布局**（E1）
   - `include/uocr/prompt.h` + `src/runtime/prompt.cpp` 的 `build_ocr_prompt`。
   - `tools/compare_layout` + `tools/reference/export_layout_cases.py`：
     11 个用例 ids/mask **11/11**（含 crop `[1,1]/[2,1]/[1,2]/[2,2]/[3,2]`）。
3. [x] **图像预处理**（E2）
   - `include/uocr/image.h` + `src/runtime/image.cpp`：PIL 兼容 bicubic、
     `ImageOps.pad`（含 Python round-half-even 中心对齐）、`dynamic_preprocess`、
     `BasicImageTransform`。
   - `tools/compare_image` + `tools/reference/export_image_cases.py`：
     与 Pillow 逐像素 ≤1 LSB，crop ratio 全对。
4. [x] **Engine 注入视觉 embedding**（E3）
   - `MoEDecoder::prefill_embeds`、`Engine::generate_from_image`、
     `Engine::image_embeddings`（含 `<= image_size` 时 crop_ratio=[1,1] 规则）。
5. [x] **端到端参考导出与对比**（E4）
   - `export_reference.py --mode ocr` 导出 `input_ids`/`images_seq_mask`/
     `image_global`/`visual_scattered`/`prefill_logits`/greedy；
     `tools/compare_ocr` 逐项对比。500×400 图：visual rel_l2 4.2%、
     prefill logits rel_l2 6.4% 且 top-1 一致、greedy 24/24。
6. [x] **采样与输出文本**（E5）
   - 按参考/SGLang 语义修正 `Sampler::apply_no_repeat_ngram`
     （匹配 `ngram-1` 前缀 + 滑动窗口），单测 `test_sampler`；
     24 步 greedy 在 `no_repeat_ngram_size=35, ngram_window=1024` 下 **24/24**。

### P1 降低路由翻转（提升对齐精度）
- [x] 在 MoE gate 前把激活按 bf16 舍入，复刻参考数值。（2026-09-19 尝试：
      `MoEDecoder::set_bf16_rounding`，默认关闭；实测 set_mismatch 仍为 13，
      **未能降低翻转**。翻转来自 attention/expert 的 f32-vs-bf16 累积漂移，
      需要整链路 bf16 才可能对齐，暂缓。）
- [x] 逐层对比 attention 的 q/k/v、O 投影输出（2026-09-19）：`export_reference.py`
      增加 q/k/v/o hook，`MoEDecoder::set_trace_attn` + `compare_reference` 输出逐层
      rel_l2，q/k/v/o 分别 ≤0.024/0.023/0.044/0.057。误差随层数累积，确认除路由外
      的残差是 **bf16 激活累积漂移**（非 bug），详见 `ALIGNMENT.md` §7。
- [x] 增加 `--decode-steps 140`，覆盖 ring 真正发生覆写的阶段，验证 R-SWA
      环形覆写路径与参考一致。（已验证：ring 完成后 final K/V rel_l2
      ≤ 0.03，decode logits rel_l2 ≤ 0.084。见 `ALIGNMENT.md` §6。）

### P1 CUDA 主路径接入（prj.md 要求 CUDA 生产构建）
- [x] device KV cache + CUDA attention 调度（2026-09-19）：新增
      `GpuRSWACache`（`include/uocr/gpu_cache.h`, `src/kernels/cuda/gpu_cache.cu`）
      与通用 `cuda::rswa_attention`（causal prefill / decode）。
      `tests/test_rswa_cuda.cu` 验证：环形覆写 cache K/V 完全一致、
      decode 与 prefill causal 误差 ≤1e-6。
- [x] 设备端完整前向（2026-09-19）：`GpuDecoder`
      （`include/uocr/gpu_decoder.h`, `src/engine/gpu_decoder.cu`）把 RMSNorm、
      QKV/O 投影、RoPE、R-SWA attention、dense/MoE 全部放到 GPU；MoE 路由
      在 GPU 算 logits，top-k/按专家分组在 host 调度。
- [x] 权重常驻 device（当前 bf16，加载时上传一次）；embedding 与 lm_head
      仍在 host（每步仅取/算一行）。`tests/test_rswa_cuda.cu` 验证
      `GpuDecoder` vs CPU `MoEDecoder`：prefill rel_l2 0.0019、decode 0.0021。
- [x] `Engine` 的 CUDA 执行分支：`Engine(..., Backend::CUDA)` 构造 `GpuDecoder`，
      `generate` / `generate_from_image` 自动走设备路径；测试中 CPU/CUDA
      greedy token 一致。
- [x] device 端 INT4 专家权重（2026-09-19）：`upload_expert_table` 上传 AWQ
      打包权重（连续 `[n_experts,rows,cols]`），`moe_experts_masked_int4` 内核内
      反量化。真实模型显存 9.2GB → **2.2GB**。
- [x] resident KV / CUDA Graph（见下）。

> 构建结构调整：CUDA 源文件直接并入 `uocr_core`（`UOCR_ENGINE_LIB` 恒为
> `uocr_core`），避免 Engine 与 CUDA 静态库的循环依赖；CPU 构建不编译 `.cu`。

### P2 CUDA Graph 与创新点落地（已完成 2026-09-19）
- [x] device 端 router + top-k kernel（`moe_router_topk`），去掉每层 D2H/host top-k。
- [x] “全 expert 固定调度 + 路由掩码跳过”（`moe_experts_masked[_int4]`），网格静态。
- [x] 捕获解码稳态：`GpuDecoder` 首次 decode 时捕获，之后 replay；对比见
      `bench_cuda_decode`（plain 6.8ms → full graph 5.4ms / token，INT4）。
- [x] 持久化 R-SWA KV 索引：`rswa_append_decode` 设备端推进 `len/ring_pos`，
      `rswa_attention_devlen` 读设备长度，同一 Graph 覆盖 warmup→ring。
- [x] `EngineConfig.graph_scope` 支持 `full` / `attn_dense` 两种范围（消融）。
- [ ] Prefill KV 分区写入优化（当前按参考语义保留全部 prefill KV）。
- 说明：`matvec_bf16` / expert MLP 改为 warp-per-output 合并访存后，TPOT 从
  26.7ms 降到 5.4ms；详见 `PITFALLS.md` §11-12。

### P2 Tensor Core INT4 GEMM
- [x] 补充张量核 W4A16 路径（2026-09-19）：`moe_gemm_int4_tc` 在寄存器内把
      INT4 权重反量化为 bf16，用 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
      计算，保留标量版 `moe_gemm_int4` 做正确性基线。
      测试：`[8,64]x[256,256]` rel_l2 0.0024、ragged `M/K` rel_l2 0.0026。
- [ ] **注意**：`prj.md` 写的 `s4.s4.s32` 要求 A/B 都是 INT4（即 W4A4），与同段的
      “激活值 BF16 输入”矛盾。经确认采用 W4A16（内核内 INT4→BF16 反量化 +
      bf16 tensor core），既保留权重带宽收益又保留 BF16 激活精度。

### P2 Tensor Core 性能优化（部分完成 2026-09-19）
- [x] `moe_gemm_int4_tc` 改用 shared-memory staging + `ldmatrix.x4/x2`：block=4
      warps 计算 64×BN tile。**bn 调优结论：专家 GEMM 用 bn=8**（n=896/1280 时
      block 数足够；bn≥16 会让 block 数掉到 SM 数以下）。反量化 staging 改为
      128 线程按元素循环后，bn=8 下 m=96 50.2µs、m=273 110µs（旧版 ~65/~136µs，
      **约 +20%**）。微基准 `bench/bench_int4_gemm{,_down}.txt`，`moe_gemm_int4_tc_n`
      可扫 bn∈{8,16,32,64}。
- [x] **prefill INT4 专家 GEMM 提速**（2026-09-19）：整波 prefill（B=16）
      227→204ms（−10%），INT4 batch=16 374→387 tok/s。见 `BENCHMARKS.md` §2.5/§2.7。
- [ ] 剩余：prefill 每层仍逐专家发 3×64 个 GEMM（nsys 12096 次、占 55% kernel
      时间）——下一步做 grouped GEMM（一层一次 launch，block 映射到
      (expert,n-tile)），或合并 gate/up。另 `swizzle`/split-K/`cp.async`。
- [x] **bf16 tensor-core GEMM**（2026-09-19）：`matmul_t_bf16` 改为 block=4 warps、
      64(m)×64(n) tile、`ldmatrix.x4/x2` + `mma.m16n8k16.bf16`；m 不足 64 时越界
      warp 跳过 mma（小 m 不浪费）。保留 `matmul_t_bf16_ref` 做 A/B，单测
      rel_l2 ≤ 0.0018。微基准 n=k=1280 最高 18.4 TFLOPS（m=273，4.5×）；真实模型
      BF16 batch=16 298→411 tok/s、整波 prefill 367→216ms。见 `CORE_TECH.md` §5.5、
      `BENCHMARKS.md` §2.7–2.8。
- [x] 小 m 变体（2026-09-19）：m ≤ 16 时 4 warp 沿 n 拆分（BM=16/BN=64），
      n=k=1280 m=16 36.1→30.1µs；真实模型各 batch 再 +2~5%。
- [ ] 剩余：lm_head（大 n）仍 2–8 TFLOPS，受权重读取/同步限制；`swizzle`、
      split-K、`cp.async` 双缓冲。

### P2 连续批处理（已完成 2026-09-19）
- [x] `GpuDecoder` per-slot R-SWA KV cache + `attention_block_batch` /
      `forward_batch` / `batch_decode`。
- [x] `Engine::generate_batch` 由 `ContinuousBatchScheduler` 驱动：admit/prefill →
      所有活跃 slot 一起 decode → 完成即释放 slot。
- [x] 修复 CUDA 下 host `MemoryPool` 按 `max_seq_len×max_batch` 预分配数十 GB 的
      问题（CUDA 后端改用小 arena）。
- [x] `bench_cuda_batch`（2026-09-19 三次更新后）：INT4 batch=16
      **310.7 tok/s / 3048MB**；BF16 **298.0 tok/s / 10044MB**
      （batched attention + ragged prefill + 按专家规模选 MoE 路径，见 §2.6）。
      测试 `Engine batch`（4 个不同长度 prompt）batch == sequential == CPU。
- [x] batched attention kernel（`rswa_append_decode_batch` / `rswa_attention_batch`）。
- [x] **ragged 多请求 prefill**：`batch_prefill_embeds` + `forward_ragged`，
      K/V 用 `rswa_write_prefill_ragged` 按 slot 散写，attention 用
      `rswa_attention_ragged` 逐行 causal；一步内新请求一次前向完成。
      单测与逐请求 prefill logits rel_l2 = 0。
- [x] **slot 映射修复**：`batch_decode(tokens, positions, slots, logits)` 显式传入
      “行→slot”映射（此前按行号当 slot，请求顺序与 slot 顺序不一致时会读错
      KV）。见 `PITFALLS.md` §13。
- [x] **待优化**：dense/shared 投影上 tensor core（已于 2026-09-19 完成，见上）。

### P2 Batched CUDA Graph（已完成 2026-09-19）
- [x] `forward_batch + final rmsnorm` 整步捕获；按行数 B 缓存
      `batch_graphs_[B-1]`，第一次遇到该 B 时捕获、之后 replay。
- [x] 活跃 slot 集合 / 行→slot 排列 / position / embedding 每步写入地址固定的
      `d_batch_slots_/d_pos_/d_xin_`，replay 时读取 → **活跃集合变化无需重捕获**。
- [x] 失效条件：scratch 扩容、`router_cap_` 扩容、`batch_configure` 重分配；统一
      `invalidate_batch_graphs()`。`batch_configure` 形状不变时复用分配与图。
- [x] 测试：batched graph vs plain（非恒等 slot、W=8、14 步含环形覆写）
      logits rel_l2=0；`Engine batch` 二次调用复用图后 token 不变。
- [x] 基准：`bench_cuda_batch --no-graph` 对照 + warmup 稳态计时，
      原始数据 `bench/bench_batch_real_*_{graph,plain}.txt`，汇总见
      `BENCHMARKS.md` §2.7。带宽受限下收益有限（BF16 batch=16 纯 decode +6.7%）。

### P2 分词器与性能记录
- [x] 分词器对齐已提前到 P0/E0 执行（43/43），此处只保留性能与回归项。
- [x] 补齐 prj.md §7.2 大部分指标（TTFT/TPOT/吞吐/峰值显存），见 `BENCHMARKS.md`
      与 `prj.md §7.2`；§7.3 完成 INT4 vs BF16、Graph 范围消融。
- [x] `compare_reference` / `compare_vision` 等纳入可选 CTest
      （`-DENGINE_REFERENCE_DIR=...`，默认跳过）。
- [ ] 仍缺：GPU SM 利用率（已安装 ncu/nsys）、KV Cache 碎片率、AWQ vs 朴素 INT4。

### 环境与依赖备注
- 仅第三方 C++ 依赖：系统 `nlohmann/json`（已安装）；未用 `spdlog`（自研 log）。
- Python 参考环境在 `.venv`（torch 2.10.0+cu128 / transformers 4.57.1），
  安装依赖前需经代理 `http://192.168.1.164:7897`。
- 若需新增系统依赖或 Python 包，先询问确认后再安装。
