按照preDocs/prj.md中的内容实现这个项目。
实现这个项目的过程中，有不明白、不清楚的地方问我。需要安装第三方依赖时请让我做决定，不要擅自安装。
保留中间进度文档、踩过的坑、核心技术实现等内容，放在preDocs目录下。
使用g++作为C++编译器。
记得使用git保存项目。

---

## 下一阶段任务计划（2026-09-15 更新）

进度与结果见 `PROGRESS.md`、`ALIGNMENT.md`、`PITFALLS.md`、`CORE_TECH.md`。

### P0 视觉编码器数值对齐（已完成 2026-09-15）
- [x] 跑完 `tools/compare_vision.cpp`。`DeepEncoder::encode` 输出 273×1280 与
      参考 `visual_embeddings`：对 f32 参考 rel-L2 ≤ 5.8e-4，对 bf16 参考 3.2%。
- [x] 分段对比：已加 `DeepEncoder::encode_stages` + `--dump-stages` +
      `tools/reference/export_vision_stages.py`，定位并修复了 SAM 相对位置漏乘
      query、CLIP 漏加 QKV bias 两个 bug。CPU 侧启用 OpenMP（8 线程约 2 分钟）。
      详见 `ALIGNMENT.md` §4、`PITFALLS.md` §9。

### P0 端到端图像 OCR 对齐
- [ ] 复刻参考 `infer()` 的预处理：`BasicImageTransform`、`ImageOps.pad`、
      `dynamic_preprocess`（base/gundam 两种模式）、`<image>` 占位与
      `images_seq_mask` 布局。
- [ ] 用真实权重跑单页图像，比较首 token logits 与参考 `generate` 的第 1 步；
      再比较前若干步 greedy 结果。
- [ ] 用 README 的 `no_repeat_ngram_size=35` 逻辑验证输出文本。

### P1 降低路由翻转（提升对齐精度）
- [ ] 在 MoE gate 前把激活按 bf16 舍入，复刻参考数值，减少近似并列的专家翻转。
- [ ] 逐层对比 attention 的 q/k/v、O 投影输出，定位除路由外的残差。
- [ ] 增加 `--decode-steps 140`，覆盖 ring 真正发生覆写的阶段，验证 R-SWA
      环形覆写路径与参考一致。

### P1 CUDA 主路径接入（prj.md 要求 CUDA 生产构建）
- [ ] 新增 `GpuTensor` / device KV cache；将 `MoEDecoder` 的 `Linear::forward`
      与 `RSWACache::attention` 分派到 `uocr::cuda::*`。
- [ ] 权重常驻 device（bf16 + INT4），避免每步 H2D。
- [ ] 提供 `Engine` 的 CUDA 执行分支；CPU 仍作为参考实现。

### P2 CUDA Graph 与创新点落地
- [ ] 按 prj.md 方案实现"路由 kernel 在 Graph 外、expert 计算在 Graph 内全调度 +
      掩码跳过"，捕获解码稳态。
- [ ] 持久化 R-SWA KV 索引 buffer，保证多次 replay 地址稳定。
- [ ] 实现 prj.md 的 Prefill KV 分区写入优化（当前按参考语义保留全部 prefill KV）。

### P2 Tensor Core INT4 GEMM
- [ ] 将 `moe_gemm_int4.cu` 的标量反量化版替换/补充为
      `mma.sync.aligned.m16n8k32.s4.s4.s32`（sm_120），并保留标量版做正确性基线。

### P2 分词器与性能记录
- [ ] 对齐 DeepSeek BPE 预分词正则（`\p{N}{1,3}`、CJK、标点 split），
      当前为 GPT-2 风格近似。
- [ ] 补齐 prj.md §7.2 的所有指标（CUDA 上的 TTFT/TPOT/吞吐/峰值显存/碎片率），
      以及 §7.3 的 ablation（INT4 vs BF16、AWQ vs 朴素 INT4、Graph 范围等）。
- [ ] 把 `compare_reference` / `compare_vision` 纳入可选 CTest（需要 `.venv`
      与权重，默认跳过）。

### 环境与依赖备注
- 仅第三方 C++ 依赖：系统 `nlohmann/json`（已安装）；未用 `spdlog`（自研 log）。
- Python 参考环境在 `.venv`（torch 2.10.0+cu128 / transformers 4.57.1），
  安装依赖前需经代理 `http://192.168.1.164:7897`。
- 若需新增系统依赖或 Python 包，先询问确认后再安装。

