# Anima FP8 块交换传输消融实验计划

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

日期：2026-06-03

## 当前结论

`balanced_16g` 的基础对照报告已经完成，默认 `blocks_to_swap=12` 方案成立。本计划只验证一个新的扩展假设：

> 对 frozen DiT base block 的 CPU master 使用 FP8 e4m3 传输格式，可以减少 PCIe H2D 时间，从而进一步降低 block swap 的 `forward_wait`，但会引入 frozen base 权重量化误差，必须用数值和训练稳定性验证后才能进入默认 preset。

第一轮微基准已经显示方向有潜力：

- bf16 H2D 约 `132MiB`: p50 约 `13.3ms`
- fp8 e4m3 H2D 约 `66MiB`，并在 GPU 侧转回 bf16: p50 约 `7.4ms`
- 传输时间约降低 `44%`

这不是无损传输优化。FP8 方案会让执行中的 frozen base 权重变成 FP8 近似再还原到 bf16，因此必须单独做消融，默认保留现有 bf16 block swap。

## 实验目标

1. 验证 FP8 传输是否能把 `h2d_ms` 从约 `14-15ms` 降到约 `7-9ms`。
2. 验证 `forward_wait_p95` 是否随之下降，目标是相对当前 bf16 swap 至少降低 `25%`。
3. 验证 step time 是否有可见收益，目标是 `blocks_to_swap=12` 稳态相对 bf16 swap 至少快 `1%`，或在 `blocks_to_swap=16` 下至少快 `2%`。
4. 验证显存目标不回退：`max_reserved` 和 NVML peak 不高于对应 bf16 swap。
5. 验证数值风险可控：loss 曲线无明显漂移，固定 prompt 样张无明显退化，block 输出误差在可接受范围。

## 非目标

- 不改变模型数学结构，不做 Spectrum 式预测 block 输出。
- 不量化 LoRA、router、trainable adapter、optimizer state、gradient state。
- 不替换现有 `balanced_16g` 默认值。
- 不和 `unsloth_offload_checkpointing`、`cpu_offload_checkpointing`、soft tokens、functional loss 混用。
- 不优先追求 CPU 内存节省；本轮核心指标是 PCIe H2D 与训练速度。

## 需要实现的实验开关

新增 CLI/config 参数：

```toml
block_swap_transfer_dtype = "bf16"
```

候选取值：

- `"bf16"`：当前行为，作为默认和对照。
- `"fp8_e4m3"`：只对 frozen CPU master 使用 `torch.float8_e4m3fn`，H2D copy 到 GPU bf16 storage 时还原。

实现要求：

- 只作用于 `library/runtime/offloading.py` 中 swappable frozen weight 的 CPU master。
- 继续使用 `include_trainable=False`，trainable 权重必须常驻 GPU。
- `block_swap_config` profile 事件新增：
  - `transfer_dtype`
  - `fp8_master_bytes`
  - `bf16_master_bytes`
  - `fp8_saturated_tensors`
  - `fp8_max_abs_by_block`
  - `fp8_mean_abs_error_by_block`
- `block_swap` profile 事件继续记录：
  - `phase`
  - `block_idx`
  - `wait_ms`
  - `h2d_ms`
  - `d2h_ms`
  - `event_wait_ms`
  - `enqueue_ms`
  - `submit_lag_ms`

第一版只做 persistent FP8 CPU master。暂不做 per-channel scaling，除非 Phase 0 发现 e4m3 饱和或误差不可接受。

## 固定实验条件

延续已完成 Balanced 16G 报告的固定条件，减少变量漂移：

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
- `block_swap_profile_jsonl` 使用显式路径

短跑配置：

```toml
max_train_steps = 24
log_every_n_steps = 2
```

长跑配置：

```toml
max_train_steps = 300
log_every_n_steps = 10
```

## 指标口径

训练速度：

- 从 `progress.jsonl` 的 step 事件 timestamp 差分统计。
- 丢弃 compile warmup 后的早期间隔。
- 记录 `sec/step` median、p90、p95、max。

显存：

- 训练进程内：
  - `cuda/max_memory_allocated_gb`
  - `cuda/max_memory_reserved_gb`
- 长跑额外采样 NVML：
  - `memory.used`
  - `free at peak`

Block swap profile：

- 按 `phase` 聚合 `forward_wait` / `backward_wait`。
- 统计 `wait_ms` avg/p50/p90/p95/p99/max。
- 统计 `h2d_ms` p50/p90/p95/max。
- 统计 `enqueue_ms`、`submit_lag_ms`，确认性能收益来自 H2D，而不是 profile 写入或线程入队噪声。

数值稳定性：

- loss median 和 p95，不要求 bitwise 一致。
- 相同 seed、相同 batch 下的短跑 loss 曲线差值。
- 可选：固定 prompt sample 的人工检查。
- 可选：block 输出 cosine similarity / relative L2。

## Phase 0: 离线 FP8 可行性检查

目的：确认 raw e4m3 不会因权重 outlier 产生明显饱和或过大误差。

运行内容：

1. 加载 Anima DiT bf16 权重。
2. 遍历 28 个 DiT blocks 中 swappable frozen weights。
3. 对每个 tensor 执行：

   ```python
   fp8 = weight.to(torch.float8_e4m3fn)
   restored = fp8.to(torch.bfloat16)
   ```

4. 统计：
   - 原始 `max_abs`
   - restored `max_abs`
   - `mean_abs_error`
   - `max_abs_error`
   - `relative_l2`
   - 是否出现 inf/nan
   - 估算 FP8 master bytes

通过阈值：

- 无 inf/nan。
- swappable tensor 饱和计数为 0。
- per-block `relative_l2` p95 低于 `2%`。
- 如果超过阈值，先暂停训练消融，改策划 scaled FP8，不直接跑训练。

产物：

- `/tmp/anima-fp8-blockswap/phase0_fp8_weight_error.jsonl`
- 报告表格：per-block 误差 p50/p95/max。

## Phase 1: 拷贝微基准

目的：在真实 block tensor 尺寸附近验证 H2D 降幅。

实验组：

| group | source dtype | target storage | expected |
| ----- | ------------ | -------------- | -------- |
| M1-A  | bf16 pinned CPU | bf16 GPU | 当前基线 |
| M1-B  | fp8 e4m3 pinned CPU | bf16 GPU | 目标路径 |
| M1-C  | fp8 e4m3 pinned CPU | fp8 GPU staging + bf16 copy | 备用路径 |

通过阈值：

- M1-B H2D p50 比 M1-A 至少降低 `35%`。
- M1-B H2D p95 比 M1-A 至少降低 `30%`。
- M1-B 不需要额外 GPU staging buffer，或 staging buffer 明确低于 `200MiB`。

## Phase 2: 短跑主消融

目的：验证真实训练中的 wait 和 step time。

实验矩阵：

| group | blocks_to_swap | transfer dtype | steps | repeats |
| ----- | -------------: | -------------- | ----: | ------: |
| P2-A  | 12 | bf16 | 24 | 2 |
| P2-B  | 12 | fp8_e4m3 | 24 | 2 |
| P2-C  | 16 | bf16 | 24 | 2 |
| P2-D  | 16 | fp8_e4m3 | 24 | 2 |

每组只改变 `blocks_to_swap` 或 `block_swap_transfer_dtype`。如果 GPU 时间紧张，先跑 P2-A/P2-B；只有 P2-B 达标才跑 P2-C/P2-D。

通过阈值：

- P2-B `h2d_ms_p95` 相对 P2-A 降低至少 `30%`。
- P2-B `forward_wait_p95` 相对 P2-A 降低至少 `25%`。
- P2-B `sec/step_med` 不慢于 P2-A；理想目标快 `>=1%`。
- P2-D `backward_wait_p95` 不高于 `80ms`。
- FP8 组 `max_reserved` 不高于 bf16 对照。

失败判定：

- FP8 组 step time 变慢超过 `2%`。
- FP8 组 loss 出现持续偏移或异常 spike。
- FP8 组任一 profile 出现 H2D 长尾高于 bf16 对照。

## Phase 3: 数值路径短验证

目的：区分“速度可行”与“训练可用”。

实验组：

| group | blocks_to_swap | transfer dtype | max_train_steps | seed |
| ----- | -------------: | -------------- | --------------: | ---- |
| P3-A  | 12 | bf16 | 80 | fixed |
| P3-B  | 12 | fp8_e4m3 | 80 | fixed |

统计：

- loss median / p95 / final。
- loss 差值曲线。
- 如启用 sample，固定 prompt、固定 seed、固定 step 输出对比。

通过阈值：

- final loss 差异不超过短跑自然波动。
- 没有 NaN/Inf。
- 样张没有明显结构崩坏或 prompt 跟随退化。

## Phase 4: 300 step 长跑

前置条件：Phase 2 和 Phase 3 均通过。

实验组：

| group | blocks_to_swap | transfer dtype | steps |
| ----- | -------------: | -------------- | ----: |
| P4-A  | 12 | bf16 | 300 |
| P4-B  | 12 | fp8_e4m3 | 300 |

如果 Phase 2 显示 `blocks_to_swap=16 + fp8_e4m3` 明显更好，再追加：

| group | blocks_to_swap | transfer dtype | steps |
| ----- | -------------: | -------------- | ----: |
| P4-C  | 16 | fp8_e4m3 | 300 |

通过阈值：

- 训练正常结束，`run_end.status = "ok"`。
- FP8 组 `sec/step_med` 快于或不慢于 bf16 对照。
- FP8 组 `forward_wait_p95 < 15ms`，或相对 bf16 对照下降至少 `25%`。
- `backward_wait_p95 < 80ms`。
- NVML peak 不高于 bf16 对照。
- loss 曲线稳定。

## Phase 5: 是否进入 preset 的决策

根据结果分三档：

1. 通过并有明确收益

   - 新增实验 preset：`balanced_16g_fp8swap`
   - 不替换 `balanced_16g`
   - WebUI 文案标注为“实验：FP8 交换传输，可能轻微改变 frozen base 数值”
2. H2D 降低但 step time 无收益

   - 保留 CLI 实验开关
   - 不新增 preset
   - 技术报告说明瓶颈已转移到 compute/调度窗口
3. 数值或稳定性不通过

   - 不开放 preset
   - 保留 findings 报告
   - 下一轮改策划 scaled FP8 或不继续该方向

## 最终技术报告结构

最终报告文件建议：

```text
docs/findings/anima_fp8_blockswap_transfer_report.md
```

报告必须包含：

1. 摘要结论

   - 是否推荐 FP8 swap
   - 推荐参数
   - 是否进入 WebUI preset
2. 背景

   - Balanced 16G 已完成结果
   - 当前 `h2d_ms` / `forward_wait` 剩余瓶颈
3. 实现说明

   - `block_swap_transfer_dtype`
   - frozen-only 范围
   - profile 新字段
4. 实验环境

   - GPU
   - 数据集
   - 方法
   - seed
   - runtime config 来源
5. Phase 0-4 结果表

   - FP8 误差
   - 微基准
   - 短跑主消融
   - 数值短验证
   - 300 step 长跑
6. 结论解释

   - 性能收益来自哪里
   - 是否有数值风险
   - 为什么进入或不进入 preset
7. 使用建议

   - WebUI
   - 手动 TOML
   - OOM 时与 `blocks_to_swap=16` / `selective_checkpoint` 的优先级
8. 验证记录

   - pytest
   - 训练命令
   - 产物路径

## 实现验证要求

配置或 CLI 改动后至少跑：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_config.py -q
```

block swap runtime 改动后跑：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_resume.py -k "block_swap or progress_jsonl" -q
timeout 60 .venv/bin/python -m pytest tests/test_deferred_sample_cleanup.py -q
```

如果新增 WebUI 入口或 preset 文案，再跑：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_state.py -q
```

## 当前下一步

1. 实现 `block_swap_transfer_dtype` 实验开关，默认 `bf16`。
2. 加 Phase 0 轻量统计脚本或临时分析命令。
3. 先跑 Phase 0 和 Phase 1。
4. 只有误差与微基准通过后，再启动真实训练短跑。
