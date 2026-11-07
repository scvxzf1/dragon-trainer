# Anima Balanced 16G 预测式块交换技术报告

状态：研究记录 / 阶段快照
适用版本：以文中日期、提交和运行环境为准；不作为当前 main 操作说明

日期：2026-06-03

## 当前结论

`balanced_16g` 的核心效率问题在短跑复现实验中已经解决。当前推荐继续保留：

- `balanced_16g`: `blocks_to_swap = 12`、`torch_compile = true`、`compile_inductor_mode = default`、`selective_checkpoint = "off"`。
- 手动更省显存：`blocks_to_swap = 16`。
- 仍然 OOM 时再启用：`selective_checkpoint = "mlp_only"`，最后才尝试 `every_other` 或旧 `low_vram`。

验证结果显示：

- `blocks_to_swap=12`: 稳态约 `+3.0%`，`max_reserved` 约省 `3.79GB`，`forward_wait_p95=18.46ms`，`backward_wait_p95=0.05ms`。
- `blocks_to_swap=16`: 稳态约 `+6.6%`，`max_reserved` 约省 `4.33GB`，`forward_wait_p95=18.88ms`，`backward_wait_p95=29.79ms`。
- 300 step 长跑的 `balanced_16g` 默认档正常结束，NVML 峰值 `12036MB / 16384MB`，峰值余量约 `4.25GB`。
- 两组都低于速度损失 `+15%~20%` 的目标上限，也低于 profile 门槛 `forward_wait_p95 < 40ms`、`backward_wait_p95 < 80ms`。

这不是 Spectrum 式预测 block 输出。当前方案只做执行调度预测，不改变模型数学路径。

## 2026-06-27 LoKr + full checkpoint 补充观测

一次 WebUI LoKr 任务使用了：

- `use_lokr = true`
- `blocks_to_swap = 26`
- `gradient_checkpointing = true`
- `selective_checkpoint = "off"`
- `torch_compile = true`
- `compile_dynamic_seq = true`

失败点不是 OOM，也不是 block swap 交换路径失效。`progress.jsonl` 记录的真实错误是
`torch.utils.checkpoint.CheckpointError`：checkpoint forward 保存的 LoKr 相关张量
metadata 是 `[6144, 2048] bf16 cuda:0`，recompute 时变成 `[2048, 6144] bf16
cuda:0`。同一日志里 Dynamo 还在 `networks/plugins/lokr/module.py:100 forward`
命中 `recompile_limit`，并报告 LoKr forward 的 dtype guard 从 `BFloat16` 变成
`Float`。

`block_swap_profile.jsonl` 同时显示本次运行已经进入 `block_swap_config`、forward
wait 和 backward wait，H2D 单次搬运约 `13ms~16ms`。因此这次不能归因为
`blocks_to_swap` 和完整 checkpoint 普遍不兼容；block swap 本身在该运行中已经正常
调度。

当前结论：

- `selective_checkpoint = "mlp_only"` 是选择性 DiT block checkpoint，和完整
  `gradient_checkpointing = true` 不能同时开启，配置检查会拒绝。
- `blocks_to_swap` 可以和普通完整 `gradient_checkpointing = true` 叠加。
- 本次 LoKr 失败更准确是 `LoKr + full gradient_checkpointing + torch_compile` 的
  checkpoint recompute compiled graph 不稳定。
- 修复方向不是关闭 `torch_compile`，而是在 `LoKr + full checkpoint + compile`
  路径下根据 LoKr 模块数量和 Kronecker 形状提高 Dynamo `recompile_limit` 和
  `accumulated_recompile_limit`，避免 `LoKrModule.forward` 的多个 specialization
  被默认预算挤出缓存。
- 同时按 PyTorch checkpoint 错误提示，在该组合下关闭 Dynamo LRU graph 重排，避免
  recompute 查到和 forward 不同的 LoKr Kronecker 形状 graph。
- 如果实卡仍复现同类 `CheckpointError`，下一层再考虑重写 LoKr forward 的
  dtype/shape 路径。

## 原始问题复现

历史现象：

- 开启组使用 `blocks_to_swap = 12`，并带 `compile_inductor_mode = "max-autotune-no-cudagraphs"`。
- 用户可见显存只从约 `15.33GB` 降到约 `14.56GB`，只省约 `0.77GB`。
- 旧 `block_swap_profile.jsonl` 显示等待偏高：
  - `forward_wait_p95` 约 `112ms`
  - `backward_wait_p95` 约 `284ms`
  - `backward_wait block=11` 平均约 `285ms`

初始判断：不是单纯 preset 参数问题，而是 offloader 交换路径没有充分利用 frozen base weight 的性质，反传预取也太晚。

## 已实施修复

1. Frozen CPU master

   每个可交换的 frozen base weight 保留一份 CPU master。训练时 frozen base 权重不更新，所以交换时只需要从 CPU master 做 H2D restore，不再把 GPU 权重 D2H 回写。
2. 保留 trainable 常驻 GPU

   继续使用 `include_trainable=False`。LoRA、router、trainable adapter、optimizer/grad 状态不进入 CPU block swap 路径。
3. Backward next-use 预取

   Anima DiT block 顺序固定为 `0 -> 27`。反传时 tail block 完成后立刻恢复下一次要用的 early block：

   - `blocks_to_swap=12`: `27 -> 11`、`26 -> 10`、...、`16 -> 0`
4. 异步 CUDA copy 入队

   后台线程只负责把 H2D copy 入队到独立 CUDA stream，不再在 worker 内 `stream.synchronize()`。真正等待只发生在目标 block 执行前的 `_wait_blocks_move()`。
5. Profile JSONL 扩展

   `block_swap_profile.jsonl` 现在记录：

   - `phase`
   - `block_idx`
   - `wait_ms`
   - `h2d_ms`
   - `d2h_ms`
   - `event_wait_ms`
   - `transfer_ms`
   - `enqueue_ms`
   - `submit_lag_ms`
   - `queued_at`
   - `enqueued_at`
   - `ready_at`

   `block_swap_config` 事件记录：

   - `frozen_weight_master_bytes`
   - `frozen_weight_bytes_by_block`
   - `h2d_only = true`
6. 训练日志显存指标

   `progress.jsonl` 的 step 事件新增：

   - `cuda/memory_allocated_gb`
   - `cuda/memory_reserved_gb`
   - `cuda/max_memory_allocated_gb`
   - `cuda/max_memory_reserved_gb`
7. Compile preset 修正

   `balanced_16g` 不再写死 `compile_inductor_mode = "max-autotune-no-cudagraphs"`。短跑显示该模式没有显存或稳态速度优势，且首步 warmup 更重。

## 实验环境

固定条件：

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
- `sample_at_first = false`
- `sample_every_n_steps = 0`
- `max_train_steps = 24`
- `log_every_n_steps = 2`

统计口径：

- 秒/step 使用 `progress.jsonl` step timestamp 差分。
- 丢弃 warmup 后的早期 log 间隔，降低 torch.compile 首步成本干扰。
- 显存使用训练进程内 `torch.cuda.max_memory_allocated/reserved`。
- `block_swap_profile.jsonl` 统计 `forward_wait` 和 `backward_wait`。

## Phase 0: 复现与 compile mode

| group | swap | compile mode               | sec/step med | max allocated GB | max reserved GB | fwd wait p95 ms | bwd wait p95 ms | frozen master GiB |
| ----- | ---: | -------------------------- | -----------: | ---------------: | --------------: | --------------: | --------------: | ----------------: |
| P0-A  |    0 | default                    |        1.568 |            12.23 |           14.76 |                 |                 |                   |
| P0-B  |   12 | max-autotune-no-cudagraphs |        1.617 |            10.67 |           10.96 |           17.10 |            0.01 |              3.61 |
| P0-C  |   12 | default                    |        1.622 |            10.67 |           10.97 |           17.54 |            0.01 |              3.61 |

结论：

- `max-autotune-no-cudagraphs` 没有显存优势，也没有稳态速度优势。
- `compile_inductor_mode = default` 应继续作为 `balanced_16g` 默认。
- H2D-only CPU master 和 backward next-use 已经把历史 `backward_wait_p95 ~284ms` 降到接近 0。

## Phase 1: blocks_to_swap 曲线

| group    | swap | sec/step med | 相对 baseline | max allocated GB | allocated 省 | max reserved GB | reserved 省 | fwd wait p95 ms | bwd wait p95 ms |
| -------- | ---: | -----------: | ------------: | ---------------: | -----------: | --------------: | ----------: | --------------: | --------------: |
| baseline |    0 |        1.569 |          0.0% |            12.23 |         0.00 |           14.76 |        0.00 |                 |                 |
| P1-A     |    4 |        1.571 |         +0.2% |            11.72 |         0.51 |           12.01 |        2.75 |            0.01 |            0.01 |
| P1-B     |    8 |        1.580 |         +0.7% |            11.19 |         1.04 |           14.14 |        0.62 |            0.01 |            0.01 |
| P0-C     |   12 |        1.624 |         +3.5% |            10.67 |         1.56 |           10.97 |        3.79 |           17.52 |            0.01 |
| P1-D     |   16 |        1.684 |         +7.3% |            10.15 |         2.08 |           10.43 |        4.33 |           17.21 |           31.38 |
| P1-E     |   20 |        1.680 |         +7.1% |             9.61 |         2.62 |           12.55 |        2.21 |           16.76 |           27.00 |

结论：

- `blocks_to_swap=12` 是稳妥默认：速度损失很小，reserved 已接近省 `4GB`。
- `blocks_to_swap=16` 是更强手动档：reserved 省约 `4.33GB`，速度仍远低于 `+20%`。
- `blocks_to_swap=20` allocated 继续下降，但 reserved 波动变大，且不再明显更快，不适合作为默认。

## Phase 2b: 异步 copy 入队与可信 profile

Phase 2 第一轮发现 `event_wait_ms` 在新异步路径里全为 0，原因是依赖 event 没开 timing。修正后复跑 Phase 2b：

- `/tmp/anima-blockswap-phase2b/p2b_timed_swap12.*`
- `/tmp/anima-blockswap-phase2b/p2b_timed_swap16.*`

| group    | swap | sec/step med | 相对 baseline | sec/step p95 | max allocated GB | allocated 省 | max reserved GB | reserved 省 | fwd wait p95 ms | bwd wait p95 ms |
| -------- | ---: | -----------: | ------------: | -----------: | ---------------: | -----------: | --------------: | ----------: | --------------: | --------------: |
| baseline |    0 |        1.569 |          0.0% |        1.598 |            12.23 |         0.00 |           14.76 |        0.00 |                 |                 |
| P2b-A    |   12 |        1.617 |         +3.0% |        1.676 |            10.67 |         1.56 |           10.97 |        3.79 |           18.46 |            0.05 |
| P2b-B    |   16 |        1.673 |         +6.6% |        1.679 |            10.15 |         2.08 |           10.43 |        4.33 |           18.88 |           29.79 |

### Phase 2b profile 细节

| group  | phase    | wait avg ms | wait p95 ms | h2d p95 ms | event_wait p95 ms | enqueue p95 ms |
| ------ | -------- | ----------: | ----------: | ---------: | ----------------: | -------------: |
| swap12 | forward  |       12.09 |       18.46 |      15.07 |             0.012 |           7.54 |
| swap12 | backward |        0.04 |        0.05 |      15.52 |             0.016 |          16.30 |
| swap16 | forward  |       14.00 |       18.88 |      15.31 |             0.013 |           7.62 |
| swap16 | backward |       25.79 |       29.79 |      15.54 |             0.016 |           9.65 |

解释：

- H2D 单次搬运约 `15ms`，这是物理 copy 成本。
- `event_wait_ms` p95 只有约 `0.01ms`，说明 copy stream 等 compute stream 的依赖不是主要瓶颈。
- `wait_ms` 才是目标 block 执行前真正可见等待。
- swap16 的 backward 仍有约 `30ms` p95，但远低于 `80ms` 门槛。

## Phase 3: 300 step 实卡长跑

目的：补齐短跑之外的长尾稳定性和用户可见显存余量验证。

配置：

- 来源：`/tmp/anima-blockswap-phase2b/p2b_timed_swap12.toml`
- 长跑配置：`/tmp/anima-blockswap-longrun/balanced12_300.toml`
- `blocks_to_swap = 12`
- `selective_checkpoint = "off"`
- `gradient_checkpointing = false`
- `unsloth_offload_checkpointing = false`
- `max_train_steps = 300`
- `log_every_n_steps = 10`
- NVML 采样：`/tmp/anima-blockswap-longrun/balanced12_300.gpu_stats.jsonl`

产物：

- `/tmp/anima-blockswap-longrun/balanced12_300.progress.jsonl`
- `/tmp/anima-blockswap-longrun/balanced12_300.block_swap_profile.jsonl`
- `/tmp/anima-blockswap-longrun/balanced12_300.gpu_stats.jsonl`
- `/tmp/anima-blockswap-longrun/balanced12_300/output/balanced12_300.safetensors`

结果：

| group               | swap | sec/step med | sec/step p95 | max allocated GB | max reserved GB | NVML peak GB | free at peak GB |
| ------------------- | ---: | -----------: | -----------: | ---------------: | --------------: | -----------: | --------------: |
| 300-step balanced12 |   12 |        1.608 |        1.633 |            10.69 |           10.97 |        11.75 |            4.25 |

Profile 长尾：

| group               | phase    | wait avg ms | wait p95 ms | wait p99 ms | wait max ms | h2d p95 ms | enqueue p95 ms |
| ------------------- | -------- | ----------: | ----------: | ----------: | ----------: | ---------: | -------------: |
| 300-step balanced12 | forward  |       12.53 |       18.45 |       18.98 |       20.23 |      15.14 |           6.94 |
| 300-step balanced12 | backward |        0.04 |        0.05 |        0.06 |        0.11 |      15.52 |          15.88 |

结论：

- 300 step 正常结束，`run_end.status = "ok"`，最终 step 为 `300`。
- 用户可见峰值显存约 `11.75GB`，16GB 卡峰值余量约 `4.25GB`，明显高于 `700MB~1GB` 门槛。
- 长跑没有出现 reserved 或 wait 长尾抬升。
- 默认 `balanced_16g` 不需要启用 selective checkpoint。

## 为什么“显存几乎没减少”

缺少训练进程内 allocated/reserved 指标，只能从 WebUI 或 nvidia-smi 侧看总占用。旧实现还存在两个混淆项：

1. Frozen 权重做了不必要的 D2H 回写，反传预取来不及，速度损失被放大。
2. `max-autotune-no-cudagraphs` 没有带来显存收益，还增加 warmup 干扰。

修复后从训练进程内看：

- swap12 `max_reserved`: `14.76GB -> 10.97GB`，省约 `3.79GB`。
- swap16 `max_reserved`: `14.76GB -> 10.43GB`，省约 `4.33GB`。

allocated 下降较小是正常的，因为峰值还包含 activation、optimizer 临时区、compiled graph/allocator 行为。用户可见能否“空出 4GB”更接近 reserved/NVML 峰值，而不是单看 allocated。

## 当前推荐

默认 WebUI `Balanced 16G`：

```toml
blocks_to_swap = 12
gradient_checkpointing = false
unsloth_offload_checkpointing = false
torch_compile = true
selective_checkpoint = "off"
block_swap_profile_jsonl = "auto"
```

手动更省显存：

```toml
blocks_to_swap = 16
gradient_checkpointing = false
unsloth_offload_checkpointing = false
torch_compile = true
selective_checkpoint = "off"
block_swap_profile_jsonl = "auto"
```

仍然 OOM 时：

```toml
blocks_to_swap = 12
selective_checkpoint = "mlp_only"
```

不建议默认：

- `blocks_to_swap = 20`: allocated 更低，但 reserved 波动和调度压力更高。
- `gradient_checkpointing = true`: 普通 gradient checkpoint 可以和 block swap 叠加，但全量 checkpoint 会重算过多，违背 Balanced 的速度目标。
- `unsloth_offload_checkpointing = true`: 这是 activation CPU offload 路线，当前和 block swap 互斥；它是 `low_vram` 保命线，不是 Balanced 主方案。
- `compile_inductor_mode = "reduce-overhead"` 或 CUDAGraph 模式：block swap 会移动权重，不适合 CUDAGraph。

## 后续消融实验

已经完成的实验足以说明默认 `balanced_16g` 的效率问题已解决。后续只保留这些扩展验证：

1. `blocks_to_swap=16` 长跑

   - 只在用户希望把手动更省显存档作为常用配置时需要
   - 重点看 backward wait p95 是否长期保持低于 `80ms`
2. 选择性 checkpoint

   - 只在真实任务仍 OOM 时跑
   - 优先 `selective_checkpoint = "mlp_only"`
   - 再试 `every_other`
3. sample/eval 安全性

   - 验证 `pause_block_swap()` / `resume_block_swap()` 后状态恢复正确
   - 确认中途 sample 不导致 block swap 状态错乱
4. low_vram 对比

   - `low_vram`
   - `low_vram_blockswap`
   - `balanced_16g`
   - `balanced_16g + mlp_only`

## 验证记录

已通过：

```bash
timeout 60 .venv/bin/python -m py_compile library/runtime/offloading.py
timeout 180 .venv/bin/python -m pytest tests/test_block_swapping.py tests/test_config.py tests/test_training_frontend_state.py tests/test_progress_sink.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_resume.py -k "block_swap or progress_jsonl" -q
timeout 60 .venv/bin/python -m pytest tests/test_deferred_sample_cleanup.py -q
```

结果：

- `82 passed`
- `4 passed, 83 deselected`
- `5 passed`

## 当前状态

结论：`balanced_16g` 作为默认 16GB 快速省显存方案成立。当前不需要把选择性 checkpoint 默认打开，也不需要退回旧 `low_vram`。如果用户要更大余量，优先手动把 `blocks_to_swap` 从 `12` 调到 `16`。
