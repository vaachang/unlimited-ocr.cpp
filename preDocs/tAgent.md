按照preDocs/prj.md中的内容实现这个项目。
实现这个项目的过程中，有不明白、不清楚的地方问我。需要安装第三方依赖时请让我做决定，不要擅自安装。
保留中间进度文档、踩过的坑、核心技术实现等内容，放在preDocs目录下。
使用g++作为C++编译器。
记得使用git保存项目。

---

## 下一阶段任务计划（2026-09-19 更新）

进度与结果见 `PROGRESS.md`、`ALIGNMENT.md`、`PITFALLS.md`、`CORE_TECH.md`。

**已完成里程碑**：M1 调研 → M2 骨架 → M3 调度 → M4 引擎 → M5 CUDA 内核
→ M6 视觉对齐 → P0 端到端 OCR 对齐（E0–E5）→ P1 R-SWA 环形覆写验证
→ P1 CUDA 设备端 decoder + Engine CUDA 分支 → P2 Tensor Core W4A16 GEMM。

**下一步优先级**（详见文末各节）：

1. **P2 Tensor Core 性能**：`moe_gemm_int4_tc` 仍按标量读寄存器装入 mma，需
   `ldmatrix`/shared-memory staging + split-K；prefill 已用 64×64 shared-memory
   分块 GEMM，但 INT4 TC 路径未做。
2. **P1 残留**：逐层对比 attention 的 q/k/v、O 投影输出，定位路由以外的残差
   （需先扩展 `export_reference.py` 导出 attention 内部张量）。
3. **批处理**：已完成（2026-09-19）。`Engine::generate_batch` 用
   `ContinuousBatchScheduler` 驱动 device batched decoder（每 slot 独立 R-SWA KV）。
   `bench_cuda_batch` 实测 INT4 batch=16 达 117.6 tok/s、显存 2.66GB；测试
   `Engine batch` 验证 batch 与 sequential token 一致。**待优化**：batched attention
   kernel（当前每 slot 循环发射）、batched/continuous prefill。
4. **精度评测**：OmniDocBench v1.6（AWQ vs BF16）尚未接入。

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

### P2 Tensor Core 性能优化（后续）
- [ ] 目前 TC kernel 每线程标量读取权重/激活；应改用 `ldmatrix` / 共享内存
      staging + swizzle，并做 split-K，才能接近峰值。

### P2 分词器与性能记录
- [ ] 分词器对齐已提前到 P0/E0 执行（见上），此处只保留性能与回归项。
- [ ] 补齐 prj.md §7.2 的所有指标（CUDA 上的 TTFT/TPOT/吞吐/峰值显存/碎片率），
      以及 §7.3 的 ablation（INT4 vs BF16、AWQ vs 朴素 INT4、Graph 范围等）。
- [ ] 把 `compare_reference` / `compare_vision` 纳入可选 CTest（需要 `.venv`
      与权重，默认跳过）。

### 环境与依赖备注
- 仅第三方 C++ 依赖：系统 `nlohmann/json`（已安装）；未用 `spdlog`（自研 log）。
- Python 参考环境在 `.venv`（torch 2.10.0+cu128 / transformers 4.57.1），
  安装依赖前需经代理 `http://192.168.1.164:7897`。
- 若需新增系统依赖或 Python 包，先询问确认后再安装。

