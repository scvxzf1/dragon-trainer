# Anima FP8 块交换传输最终技术报告

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

日期：2026-06-03

## 结论

**不建议**把 `fp8_e4m3` 块交换传输推进为默认训练方案。

结果很清楚：

- **性能上**，FP8 传输确实能把 frozen DiT block 的 H2D 时间削掉大约 42%~45%。
- **数值上**，raw FP8 和保守 scaled FP8 都没达到当前训练路径可接受的误差阈值。
- 因此本次结论是：**保留 `block_swap_transfer_dtype = "bf16"` 为默认；`fp8_e4m3` 只保留实验开关，不进入 preset。**

## 背景

在 `balanced_16g` 的基础上，块交换调度问题已经解决，当前剩余的是 H2D 传输带宽瓶颈。本轮只验证一个问题：

> 能否把 frozen base block 的 CPU master 压成 FP8，从而减少 PCIe 传输时间，同时不把训练数值搞坏。

相关实现已经落地在：

- `library/runtime/offloading.py`
- `library/anima/models.py`
- `library/training/cli_args.py`
- `train.py`
- `configs/base.toml`

WebUI 入口也已同步：

- `web/static/js/config/catalog/labels-options.js`
- `web/static/js/config/catalog/field-help-training.js`
- `web/static/js/config/catalog/form-layout.js`
- `web/static/js/config/catalog/defaults.js`

本报告已合并原 `anima_fp8_blockswap_transfer_ablation_plan.md` 的实验目标、固定条件和验收口径；旧计划文件已归档到 `_archive/docs/findings/`。

## 已实现的实验开关

- `block_swap_transfer_dtype = "bf16"`: 当前稳定路径，默认值。
- `block_swap_transfer_dtype = "fp8_e4m3"`: 实验路径，只量化 frozen base block 的 CPU master。

仍然保持：

- `include_trainable=False`
- LoRA / router / trainable adapter 常驻 GPU
- 不与 `unsloth_offload_checkpointing`、`cpu_offload_checkpointing`、soft tokens、functional loss 混用

## 实验设计留存

原计划只验证一个扩展假设：

> 对 frozen DiT base block 的 CPU master 使用 FP8 e4m3 传输格式，可以减少 PCIe H2D 时间，从而进一步降低 block swap 的 `forward_wait`，但会引入 frozen base 权重量化误差，必须用数值和训练稳定性验证后才能进入默认 preset。

核心目标：

- 验证 FP8 传输是否能把 `h2d_ms` 从约 `14-15ms` 降到约 `7-9ms`。
- 验证 `forward_wait_p95` 是否随之下降，目标是相对 bf16 swap 至少降低 `25%`。
- 验证 step time 是否有可见收益：`blocks_to_swap=12` 至少快 `1%`，或 `blocks_to_swap=16` 至少快 `2%`。
- 验证显存目标不回退。
- 验证 loss 曲线、固定 prompt 样张和 block 输出误差没有明显退化。

固定实验条件：

- GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16GB
- 方法：plain LoRA
- 数据来源：`20260602-233643-training-imported--1` 的 runtime config
- 数据集：`ichika87_style-过拟合测试a1`
- `network_dim = 32`
- `network_alpha = 32`
- `optimizer_type = "CAME"`
- `mixed_precision = "bf16"`
- `attn_mode = "flash"`
- `use_custom_down_autograd = true`
- `torch_compile = true`
- `compile_inductor_mode = default`
- `gradient_checkpointing = false`
- `unsloth_offload_checkpointing = false`
- `selective_checkpoint = "off"`
- `sample_at_first = false`
- `sample_every_n_steps = 0`

原计划的决策口径：

1. 通过并有明确收益：新增实验 preset `balanced_16g_fp8swap`，但不替换 `balanced_16g`。
2. H2D 降低但 step time 无收益：保留 CLI 实验开关，不新增 preset。
3. 数值或稳定性不通过：不开放 preset，只保留 findings 结论，后续改策划 scaled FP8 或停止该方向。

本轮实际落在第 3 类：H2D 收益成立，但数值误差没有过门槛。

## Phase 0: 离线数值检查

### raw FP8

环境：

- DiT：`/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-preview3-base.safetensors`
- 统计对象：28 个 blocks，420 个 swappable frozen tensors

结果：

- source size: `3.6094 GiB`
- fp8 size: `1.8047 GiB`
- `relative_l2` p50 / p95 / max: `0.048486 / 0.088261 / 0.142291`
- `mean_abs_error` p95: `0.000524`
- `max_abs_error` p95: `0.0625`
- `saturated_tensors`: `0`

解释：

- 没有数值饱和。
- 但 raw e4m3 精度不够，`relative_l2` p95 约 `8.8%`，明显高于本轮目标阈值。

### conservative scaled FP8

尝试了 per-tensor scale，并扫了多个保守 padding（`1.01 / 1.05 / 1.1 / 2.0`）。

最佳区间结果基本一致：

- `relative_l2` p95 约 `0.0266`
- `mean_abs_error` p95 约 `0.000237`
- `saturated_tensors`: `0`

解释：

- 规模化确实比 raw FP8 好很多。
- 但仍然停在 `2.65%` 左右，没压到本轮预设的数值门槛之下。
- 说明误差主要来自 e4m3 精度上限，不是单纯 clipping。

## Phase 1: 传输微基准

132MiB 量级测试：

| 路径 | p50 | p95 |
| --- | ---: | ---: |
| bf16 CPU pinned -> bf16 GPU | `13.718ms` | `13.771ms` |
| fp8 CPU pinned -> bf16 GPU direct | `7.592ms` | `7.990ms` |
| fp8 CPU pinned -> fp8 staging + bf16 copy | `7.635ms` | `7.995ms` |

相对 bf16 基线：

- p50 降幅约 `44.6%`
- p95 降幅约 `42.0%`

解释：

- FP8 的带宽收益是真实的。
- 但它只证明“传得更快”，没有证明“训练能安全用”。

## 最终判断

### 推荐级别

- **默认训练：不推荐**
- **实验研究：可保留**
- **WebUI preset：不新增**

### 为什么不推荐

1. 性能收益成立，但不是免费的。
2. raw FP8 数值误差过大。
3. conservative scaled FP8 仍未达到当前路径的数值门槛。
4. 当前 `balanced_16g` 已经在 bf16 路径下达到可接受的速度和显存目标，没有必要为了带宽收益冒数值风险。

### 如果未来还想继续

下一步不应再直接做 raw e4m3，而应考虑：

- 更细粒度的 per-channel / per-row scaling
- 其他压缩格式
- 或者保持 bf16 传输，只继续优化调度窗口

## 验证记录

已通过：

```bash
timeout 60 .venv/bin/python -m py_compile library/runtime/offloading.py library/anima/models.py train.py library/training/cli_args.py
timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py tests/test_config.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py::test_config_form_uses_navigation_search_and_progressive_disclosure -q
timeout 60 .venv/bin/python -m pytest tests/test_training_resume.py -k "block_swap or progress_jsonl" -q
timeout 60 .venv/bin/python -m pytest tests/test_deferred_sample_cleanup.py -q
```

其中前端全量测试里仍有一个与本次改动无关的旧断言偏差，不影响本结论。

## 产物

- Phase 0 raw FP8: `/tmp/anima-fp8-blockswap/phase0_fp8_weight_error.jsonl`
- Phase 0 scaled FP8: `/tmp/anima-fp8-blockswap/phase0_scaled_fp8_weight_error.jsonl`
- Phase 1 microbench: `/tmp/anima-fp8-blockswap/phase1_h2d_microbench.jsonl`

## 结语

这轮实验回答了最关键的问题：

- **FP8 块交换传输能降带宽**
- **但当前 Anima 训练路径上，它还不够稳**

所以结论不是“继续上训练”，而是“把它留在实验开关里，默认仍用 bf16”。
