# fp16 保护与 NaN 后处理：静态等效判定与待热测

日期：2026-10-21（CPU 静态面）；2026-11-06 补记 CUDA 热测（见「CUDA 热测补记」）。状态：CPU 静态回归完成，E2 与 E5/D1 已在真实 CUDA 复现；H3–H7 仍待热测。
范围：CPU 可判定面（真 `torch.amp.GradScaler("cpu")`）加 2026-11-06 两张 V100 的合成张量热测；不含真实 trainer / 模型族 / 数据链路、跨 rank 归约与长程收敛。
基线：`codex/fp16-nan-guards` @ 69fed4a0（CPU 静态面）、@ dfae1a4d（2026-11-06 CUDA 补记）；`library/`、`networks/` 相对 11b7bef5 无 diff，本文所有 library 行号沿用 t1 在 11b7bef5 的冻结值。
证据：`tests/test_fp16_nan_static_equivalence.py`（1086 行、24 用例、22 passed + 2 xfailed）；数字来源已落盘为 `bench/fp16_precision/run_probe.py`（CPU-only，可离线重跑；P1–P9 对照表见 `bench/fp16_precision/README.md`），P1–P9 编号对应其输出；CUDA 合成探针为 `bench/fp16_precision/run_cuda_probe.py`（H1/H2/H8，需真 CUDA），E2 与 E5/D1 的生产接缝复现见「CUDA 热测补记」；`/tmp/t9_fp16_static/probe_output.txt` 仅本轮临时产物，不可长期引用。
相关代码：`library/training/loop.py`、`library/training/finite_checks.py`、`library/runtime/accelerator.py`、`library/training/precision_report.py`。

## 结论

- **E1（无 scaler 位级不变）成立（L0）**：与不缩放参照路径对照时 loss 对象身份保持、grads/params 逐位相等（P5），4 条用例通过。
- **E4（scaler 状态机）成立（L0 轨迹 + L2 顺序）**：真 scaler 轨迹 8→8→16→16→32、溢出 32→16，state_dict 往返用不同 init/growth_interval 复现同一序列（P1）。
- **E2（跳步语义）部分成立**：executed ≠ applied 在 accelerate 层成立（P2：调用点执行、内层 `optimizer.step` 未被调用、权重逐位不变、scale 1024→512）；但 A 无 `optimizer_step_executed` / `optimizer_update_applied` 双字段，主循环 0 处消费 `optimizer_step_was_skipped`。子判据「grad_norm 账本不被 inf 污染」在 A **不适用**（`loop.py:1194` `mean_grad_norm — not tracked here`），不得记成通过。
- **E3（窗口记账）未实现**：交付形态是可执行规格 + 缺口钉，`strict xfail`，**规格未实现**；不改生产代码无法给出真实的窗口记账断言。
- **E5（所有权判定）原为「未实现且违反（缺陷 D1）」；2026-11-06 已修复，见下方「修复状态」**：`--debug_finite_checks` 打开时 `loop.py:861` 的梯度有限性检查早于 `:898` 的 unscale/clip，读到仍乘 scale 的梯度；CPU 实测（P3）有限 loss + scale=65536 + 真梯度 2.0 → 梯度 inf → `check_trainable_grads_finite` 抛 `FloatingPointError`，同一时刻真 scaler 会 skip 且权重不变。（修复前状态；2026-11-06 已修复。）
- 三条缺口必须分开写：**D1** 诊断工具误报（见下）、**D2** 默认路径根本没有有限性检查、**D3** adaptive 路径把跳步当致命错误。
- 数字来源限定：本文所有 CPU 数字引自 P1–P9；P4 是依赖 `scale=65536` 的**比值**，P8 是**权重增量**，两者分开引，不混用。

### 修复状态（2026-11-06）

上文 E1–E5 的**判据定义与 CPU 证据不变**；以下只更新「A 侧是否已实现」的现状。生产改动集中在 `library/training/scaler_ownership.py`（新增）、`library/training/finite_checks.py`、`library/training/loop.py`、`library/training/batch_step.py`。

| 项 | 修复前 | 现状 |
|---|---|---|
| **D1** debug 检查误报 | 读到仍乘 scale 的梯度 → 例行溢出抛 `FloatingPointError` | **已修复**：判定改为按所有权，`GradScaler` 存在时该读数归 scaler，只告警不中止；`check_trainable_grads_finite(..., scaler_owned=True)` 回传 offenders |
| **D2** 默认路径无保护 | 非有限梯度直接进 `optimizer.step` | **已修复**：无 scaler 时在优化器边界丢弃该步更新（zero_grad + 不 step），`LoopState.nonfinite_grad_skips` 计数并告警；多进程时先 `reduce(max)` 对齐，避免分支发散挂死 |
| **E5** 所有权判定 | 全仓 0 处实现 | **已实现**：`scaled_gradient_overflow_is_scaler_owned(accelerator, grads)` 真值表成立，规格用例已从 `strict xfail` 转正 |
| **D3** adaptive 把跳步当致命 | `training_runtime.py` 抛 `RuntimeError`，第 1 次溢出即终止 worker | **已改（C 增强版，2026-11-06）**：新增 `nonfinite_skip` 结果分类；`--adaptive_max_nonfinite_skips` 提供小额连续跳步预算（默认 0 保持严格契约），给 GradScaler 留 backoff 窗口；跳步不计入 `completed_steps`，也不进入步时采样；超预算或跑完步数上限仍凑不够有效步数，才以 `nonfinite_skip` 终止并交 supervisor 处理 |
| **latent NaN 覆盖** | 只在 VAE 现场编码分支 | **已修复**：`_prepare_latents()` 对缓存与编码两个来源都做 NaN 处置，并区分日志来源 |
| **E3** 窗口记账 | 只有可执行规格 | **仍未实现**：保持 `strict xfail`，本条未动 |

修复后的验证（同一分支，2026-11-06）：

- CPU：`tests/test_fp16_nan_static_equivalence.py` 23 passed + 1 xfailed（仅剩 E3 规格）、`tests/test_scaler_ownership_guard.py` 14 passed、`tests/test_latent_nan_guard.py` 4 passed，以及 `test_distributed_gradient_sync.py` / `test_completed_training_step_graph.py` / `test_adaptive_training_loop.py` / `test_training_loop_runtime.py` 合计 23 passed；总计 64 passed + 1 xfailed。
- CUDA（Tesla V100-SXM2-16GB ×2，各自独占）：有 scaler 时溢出步 `guard_raised=null`、`scaler_owned=true`、`step_was_skipped=true`、权重逐位不变、scale 65536→32768；无 scaler 时 `must_skip=true`、`skip_counter=1`、权重保持有限且逐位不变；有限梯度负控制 `must_skip=false` 且权重确实移动。两卡结果一致。

审查修订（2026-11-06，独立验收提出）：

- **扫描成本**：逐参数 `isfinite().all()` 在每个张量上一次 device sync，400 张量实测 43.4 ms/step、
  1200 张量 131.1 ms/step。改为 `torch._foreach_norm` 批量归约（整组两次 sync）后，同一台 V100 上
  400 张量降到 2.34 ms（18.5×）、1200 张量降到 6.39 ms（20.5×）。另外，**有 scaler 且 debug 关闭时
  直接短路不扫描**（该结果不影响控制流），常见 fp16 路径成本归零。
- **跳过路径账目对齐**：不再用 early return，跳过时仍推进 scheduler、清理梯度、跑探针与 block-swap，
  与 scaler 跳步一致；`precision_report` 新增显式 `skipped` 标记，跳步会计入 `updates.skipped`。
- **返回契约**：跳过路径返回 `loss.detach()`，与本函数既有不变量一致。
- **缓存 latent 处置**改为 out-of-place，避免 `.to(device)` 命中同一存储时就地改写 dataloader 张量。
- **新增 loop 级集成用例**（无 scaler 的 NaN 步：不 step、scheduler 仍推进、precision 记 skipped、返回值已
  detach），补上此前只有 fake 单测的空白。
- **R1 回归已修**：`_foreach_norm` 未指定对象 dtype 时，返回的范数会被写回梯度 dtype；
  160k 个 200.0 的 fp16 梯度范数是 80000 > 65504，结果变 inf，**有限梯度被误报非有限**
  （无 scaler 时会错误跳步）。现显式传 `torch.float32`，并补负控制用例
  `test_large_finite_fp16_grads_are_not_reported`（去掉该参数即失败，已验证非假绿）。
- **已知残留（low，未改）**：① 跳步时 `capture_up_grad_stats` 被跳过，`hydra_up_grad`
  会读到上一步的 stale stash（数值不错误，但可能过期）；② `updates.skipped` 同时统计
  scaler 跳步与 guard 跳步两种成因，未拆分；③ `LoopState.nonfinite_grad_skips` 目前只有
  warning 与计数，尚未对外发布为正式指标。
- **D3 落地细节（C 增强版）**：`NonfiniteSkip` 携带 `attempted/effective/skipped/max_nonfinite_skips`；
  worker 结果状态为 `nonfinite_skip`（与 `error` / `cuda_oom` 区分）；`process._result` 白名单已放行该状态，
  `retry` 记 `stop_reason=nonfinite_skip`，`profile_search` 单独归类；`auto_block_swap` 的 online 控制器与
  AUTO probe 同样跳过采样并在超预算时抛 `NonfiniteSkip`。跳步分支在创建计时事件与 `policy.observe` 之前
  return，因此不污染步时统计；顺带修掉 probe 里 `budget` 变量遮蔽 CUDA 显存预算的缺陷。
  默认值 0 下第一次跳步即终止，既有契约用例（`test_real_loop_scaler_skip_stops_before_scheduler_progress_or_snapshot`）保持原断言。
- **D3 复审补正（2026-11-06）**：实现后自审发现三处口径问题，已修——
  ① **D2 跳步对测量层不可见**：guard 丢弃更新时 `optimizer_step_was_skipped` 仍为 False，
  自适应运行时会把它记成 effective、online/AUTO probe 会把它当有效步时样本。现由主循环发布
  统一标志 `state.update_skipped`（scaler 跳步与 guard 跳步都置位），测量层统一读
  `scaler_ownership.update_skipped(state)`，保留 accelerate 属性作为回退。
  ② **`completed_steps` 是游标身份不是测量口径**：`resume_checkpoint` 要求
  `global_step == completed_steps == step`，故 `completed_steps` 恢复为跟随 `global_step`，
  有效步数改由 `effective_steps` / `skipped_steps` 承载。
  ③ **预算语义统一为「连续」**：AUTO probe 原先按累计跳步对比预算，与 runtime/online 的
  连续语义不一致；现三处统一为连续，真实更新即归零。
  口径确认：有效步数不足时按严格做法终止（不延长 `max_train_steps`）；被跳过的步仍照常
  提交 checkpoint（游标前进、权重未变），以保持 resume 数据游标一致。
- 仍未处理：E3 窗口记账。
- 探针读数随之变化（不是回归）：`bench/fp16_precision/run_probe.py` 的 P6 现在给出 `check_trainable_grads_finite_line=-1`、`order_loss_grads_clip_step_sched=false`（该函数已不在 `loop.py` 中调用）；P3 仍复现「函数默认参数下对 scaled 梯度 fail-closed」，但已不代表生产路径。

## 判据 E1–E5 与等效强度分级

### 判据表

| 判据 | 可判定谓词（摘要） | 强度 | A 侧状态 |
|---|---|---|---|
| E1 位级不变 | `seam(loss) is loss`；每个 grad / step 后参数 `view(torch.uint8)` 与参考路径逐字节相等；无副作用 —— 仅「不创建 scaler」被断言，`requires_grad`/`grad` 集合与 `stride`/`dtype`/`device` 不变**未断言**（见已知边界） | L0 | 成立（数值/身份部分） |
| E2 跳步语义分裂 | executed（调用点正常返回）≠ applied（内层 `optimizer.step` 调用 0 次 + 权重字节不变 + scale 退档）；账本子判据不适用 | L0（权重/scale）+ L2（事件） | 部分成立；无命名双字段 |
| E3 窗口记账 | 一次非有限累积窗口后：optimizer/scheduler/global_step/EMA/ckpt 计数不变，loss 账本回到窗口起点；连续两窗口序列 + 负控制 | L0（快照/计数） | 未实现（规格未实现） |
| E4 scaler 状态机 | 固定 init_scale/growth_interval 的轨迹表逐项相等；state_dict 往返用**不同** init/growth_interval 构造恢复对象后序列一致 | L0（轨迹）+ L2（顺序） | 成立 |
| E5 所有权判定 | 2×2 真值表（scaler 有/无 × grads 有限/非有限）：owned=True → 保护层不动作且真 scaler 随后 skip；owned=False → 原行为异常类型/消息前缀一致 | L2（真值表）+ L0（owned=False 分支） | 未实现，且当前实现违反（D1） |

### 等效强度与容差

| 级别 | 定义 | 容差 | 适用 |
|---|---|---|---|
| L0 位级 | 对象身份相等 + 张量字节相等（`view(torch.uint8)`）；含 NaN 张量改用 isfinite 真值表 + 有限元素字节比较 | 0 | E1、E2 权重/scale、E3 快照、E4 轨迹、E5 owned=False |
| L1 语义 | 允许同一算术表达式内重排（求和顺序、融合乘加），**禁止跨步累积** | fp32：rtol=1e-6、atol=1e-8（元素数 ≤1e6）；fp16 参数档：rtol=1e-3，subnormal（<5.96e-8）直接归零 | E3 仅当实现用 running-sum 减法回滚时降级（≤4 ULP） |
| L2 计数 | 只断言事件次数/顺序，不断言数值 | 无值容差 | E2 executed/applied、E4 轨迹顺序、E5 真值表 |

- 判据成立前提：所有断言必须在**同种子、同构造、同 dtype/device** 的 A/B 双跑上成立；单跑断言视为不可判定。
- 跨步 long-run 比较不得用 L1，只能 L2 或热测（见待热测 H5）。

## 两个数学边界

### (a) full_fp16：unscale 仍在 fp16 内完成

torch 原生对 fp16 梯度直接 `ValueError("Attempting to unscale FP16 gradients.")`；A 在 `library/runtime/accelerator.py:201-217` monkeypatch `GradScaler._unscale_grads_` 强制 `allow_fp16=True`（仅在 `--full_fp16` 时安装，`library/training/bootstrap.py:1093-1094`）。放行后 **unscale 仍在 fp16 内完成**，真梯度可用窗口为 `[2^-24, 65504/scale]`；`scale=65536` 时为 **`[5.960464477539063e-08, 0.99951171875]`**（P9）。

P9 实测（CPU 探针）：

| 真梯度 | scaled fp16 | unscale 后 | 相对误差 |
|---|---|---|---|
| 1e-8 | 0.0 | 0.0 | 1.0（subnormal 被清零） |
| 1e-7 | 0.0078125 | 1.1920928955078125e-07 | 0.19209（≈19%） |
| 5.96e-8 | 0.00390625 | 5.960464477539063e-08 | 7.79e-05 |
| 0.001 | 65.5625 | 0.0010004043579101562 | 4.04e-04 |
| 0.5 | 32768.0 | 0.5 | 0.0 |
| 1.5 | Infinity | Infinity | Infinity（scaled 溢出 → 整步跳过） |

即：小于 5.96e-8 的梯度直接归零，1e-7 量级相对误差仍有 19%，而 ≥0.9995 的真梯度在 scale=65536 下会 scaled 上溢、整步被 skip。该窗口是**表示窗口的解析值 + CPU 探针读数**，不是 CUDA kernel 实测（见 H2）。

### (b) 裁剪顺序：必须先 unscale

`accelerator.clip_grad_norm_` 标准分支无条件先 `self.unscale_gradients()`（accelerate 1.13.0 `accelerator.py:2994`；条件 `:2923`、`scaler.unscale_` 调用 `:2932`），A 在 `library/training/loop.py:894-898` 调用它。若反序（先 clip 后 unscale），`norm ≤ max_norm` 的断言**照样通过**，但更新量小 1/65536：P4 实测 `wrong/correct = 1.52587890625e-05`（scale=65536 时即 1/scale，随 scale 变化、越大越隐蔽）；P8 实测权重增量正确序 `-0.035355567932128906 / -0.03535531461238861` 对反序 `-9.5367431640625e-07 / -5.394796858126938e-07`。这是典型假绿：只断言「范数被限制」无法区分顺序，必须断言原范数 > max_norm 时裁剪后范数 ∈ {max_norm, 0.5·max_norm} 并核对端到端权重增量。

## 静态用例清单与实测数字

### 验证环境与命令

- 环境：`.venv` Python 3.13、torch **2.12.0+cu130**、accelerate **1.13.0**、CPU-only（`CUDA_VISIBLE_DEVICES=''`；探针 `cuda_initialized=False`）。
- 测试文件模块 docstring 已声明绑定 accelerate 1.13.0 的三个内部面：`Accelerator.native_amp`、`AcceleratorState._mixed_precision`、`AcceleratedOptimizer.step_was_skipped`；`TestAccelerateInternalContract` 用「版本断言 + 属性存在 + unscale 门双向语义」做 canary，升级即失败，不退化成 skip。
- 夹具：真实 `Accelerator` + 真实 `torch.amp.GradScaler("cpu")`，手工置 `native_amp=True`；与生产的差异只有「设备是 CPU」和「该标志位需手工置位」。模型**不**进 `prepare`（避免 CPU fp16 autocast 引入无关舍入），进程级 AcceleratorState/PartialState/GradientState 单例保存-恢复。

```bash
CUDA_VISIBLE_DEVICES='' timeout 120 .venv/bin/python -m pytest \
  tests/test_fp16_nan_static_equivalence.py -q
# 22 passed, 2 xfailed in 11.7s（exit 0；两次运行 11.42s / 11.68s）
```

### 用例清单（24 用例：22 passed + 2 xfailed）

| 类 / 用例组 | 数量 | 覆盖 | 结果 |
|---|---|---|---|
| `TestE4RealGradScalerStateMachine` | 3 | 真 scaler 类型与默认值、growth/backoff 轨迹 + 权重不变、state_dict 往返 | passed |
| `TestE1ComparatorMeta` | 2 | 比较器元测试：`torch.equal` 漏判 -0.0/+0.0、NaN payload 区分 | passed |
| `TestE1BitwiseInvariance` | 4 | 无 scaler 位级直通、scaled 精确 ×scale 负控制、unscale 后与参考逐位相同、已知范数 2.0 vs 16.0 | passed |
| `TestE2SkipSemanticSplit` | 4 | executed≠applied、溢出步 clip 返回 inf、A 侧仅计数语义、attempted/effective/skipped 三计数 | passed |
| `TestE5ScalerOwnershipGap` | 4 | 缺口钉（全仓 0 处 scaler_owned）、所有权成对断言（有/无 scaler）、D1 复现 | 3 passed + **1 xfailed（规格未实现）** |
| `TestE3WindowAccountingGap` | 5 | 非有限 loss 只 abort、门控仅 debug、8 个观测点缺席、E2/E3 边界、窗口记账完整规格 | 4 passed + **1 xfailed（规格未实现）** |
| `TestAccelerateInternalContract` | 1 | accelerate 1.13.0 canary（版本/属性/unscale 门双向语义） | passed |
| 模块级 `test_patch_accelerator_for_fp16_training_allows_fp16_unscale` | 1 | A 的 `_unscale_grads_` 补丁使 fp16 参数档可 unscale | passed |

### CPU 探针数字（P1–P9，非真机）

| 探针 | 对应判据 | 关键读数 |
|---|---|---|
| P1 | E4 | `growth_interval=2`：8→8→16→16→32；溢出 32→16（backoff 为减半）；state_dict 往返 scale 一致，键为 `_growth_tracker/backoff_factor/growth_factor/growth_interval/scale` |
| P2 | E2 | clean 步权重增量 `[-0.000500023365020752, -0.0004999637603759766]`；溢出步 `step_was_skipped=True`、scale 1024→512、权重逐位不变；fp16 参数原生 `ValueError: Attempting to unscale FP16 gradients.`，打补丁后可动 |
| P3 | E5 / D1 | loss 有限、scale=65536、`scaled_grad_finite=False` → `FloatingPointError: non-finite trainable gradients after backward`；`real_scaler_would_skip=True` |
| P4 | 裁剪顺序 | 正确序范数 1.0、反序 1.52587890625e-05；`wrong/correct = 1.52587890625e-05`（scale=65536） |
| P5 | E1 | 无 scaler 路径 loss 对象身份保持、grads/params 逐位相等 |
| P6 | 静态行号 | `loop.py`：loss 检查 :805、梯度检查 :861、clip :898、step :906、scheduler :915、global_step :694；`finite_checks_knows_scaler=False`；`bootstrap.py:1053`（prepare_optimizer）、`bootstrap.py:1094`（补丁调用）、`runtime/accelerator.py:201`（补丁定义） |
| P7 | 缺口 grep | `accumulation_window` / `window_invalid` / `reset_scaler_step_state` / `optimizer_update_applied` / `optimizer_step_executed` 等在 `library/` 均 0 命中 |
| P8 | 权重增量 | scaled 范数 92681.8984375、unscaled 1.4142135381698608、clipped 0.49999964237213135；正确序增量 `-0.035355567932128906 / -0.03535531461238861`，反序 `-9.5367431640625e-07 / -5.394796858126938e-07` |
| P9 | fp16 窗口 | 窗口 `[5.960464477539063e-08, 0.99951171875]`；1e-7 相对误差 0.19209；1.5 → Infinity |

### 改坏反向验证（M1–M10，全部如期失败）

所有变异只在 `/tmp` 副本上做（临时产物，不作为长期引用），仓库文件未动；未变异副本同选择集 5 passed 作基线，确认失败由变异而非环境造成。

| ID | 变异 | 目标用例 | 结果 |
|---|---|---|---|
| M1 | clip 先于 unscale（直调 `torch.nn.utils.clip_grad_norm_`） | `test_unscaled_seam_is_bitwise_equal_to_unscaled_reference` | FAILED |
| M2 | 溢出步不注入 inf（变成干净步） | `test_overflow_step_runs_but_does_not_move_weights` | FAILED |
| M3 | `growth_interval` 2→2000 | `test_growth_backoff_and_weight_invariance` | FAILED |
| M4 | `init_scale` 65536→1.0（不再上溢） | `test_e5_defect_d1_debug_grad_check_runs_before_unscale` | FAILED |
| M5b | `_bits` 退化成值比较器（丢 -0.0/NaN payload） | 两条 `TestE1ComparatorMeta` 用例 | FAILED |
| M6 | E5 成对用例正例不再上溢 | `test_e5_overflow_ownership_pair_live_scaler_vs_no_scaler` | FAILED |
| M7 | 溢出步记成 effective | `test_e2_a_side_counts_skipped_and_effective_updates` | FAILED |
| M8 | E3 接缝探测假装已有接缝 | E3 规格用例 | **XPASS(strict)→FAILED** |
| M9 | global_step AST 探测恒返回空 | `test_e3_boundary_window_discard_is_not_single_step_skip` | FAILED |
| M10 | 夹具不置 `native_amp` | `TestAccelerateInternalContract` canary | FAILED |

- M5 首版（`str(list)`）属**变异设计失效**（实测 1 failed / 1 passed）：Python repr 仍保留 -0.0 符号，退化比较器没有真正退化，变异目标不成立；该变异已弃用并改用 M5b，此处如实记录，不构成「测试失败后修正」。
- 文档复核时重跑 M8–M10 副本：分别 4 / 3 / 6 failed（均含目标用例），确认不是历史偶发；M10 因夹具共用而连带其他用例失败。

## A 侧现状与缺口

### 现状冻结（来自 t1 的 file:line 证据）

| # | 事实 | 证据 |
|---|---|---|
| 1 | fp16 判定/scaler 归属：只把 `mixed_precision` 透传给 Accelerator，scaler 由 accelerate 在 fp16 且设备非 CPU 时创建；`library/` 生产代码 0 处 `GradScaler(` | `library/runtime/accelerator.py:159-166`、`:115-172`；`site-packages/accelerate/accelerator.py:563-582` |
| 1b | native/adaptive 档对 scaler 存在性 fail-closed | `library/runtime/accelerator.py:167-170` |
| 2 | backward 唯一接缝：`accelerator.backward(loss)`（缩放发生在 accelerate 内部，先除 grad_accum 再 scaler.scale） | `library/training/loop.py:824-828`；`accelerate/accelerator.py:2828`、`:2833-2834` |
| 3 | clip 接缝：`accelerator.clip_grad_norm_` 标准分支无条件先 unscale | `library/training/loop.py:894-898`；`accelerate/accelerator.py:2934`、`:2994`、`:2899-2932` |
| 4 | step 接缝：`state.optimizer.step()`，optimizer 由 `accelerator.prepare` 包成 `AcceleratedOptimizer`，有 scaler 时走 scaler.step/update 并写 `step_was_skipped` | `library/training/loop.py:904-907`；`library/training/bootstrap.py:1053-1063`；`accelerate/optimizer.py:162-179`、`:187-190` |
| 5 | 主循环不消费跳步：`sync_gradients` 为真即无条件 `global_step += 1`、maybe_save、optimizer、scheduler、zero_grad | `library/training/loop.py:692-699`、`:915`、`:916`；`grep optimizer_step_was_skipped library/training/loop.py` → 0 |
| 6 | 跳步标志现有消费者均为旁路：precision 计数、native 运行时计数、resume 不变式、两处把跳步当硬错误 | `library/training/precision_report.py:200-209`；`adaptive_runtime/native_runtime.py:97-102`；`adaptive_runtime/resume_checkpoint.py:55-57`；`adaptive_runtime/training_runtime.py:71-73` |
| 7 | `--full_fp16` 补丁：monkeypatch `GradScaler._unscale_grads_` 强制 `allow_fp16=True`，仅在 `args.full_fp16` 时安装 | `library/runtime/accelerator.py:201-217`；`library/training/bootstrap.py:1093-1094`；`library/anima/training.py:458-463` |
| 8 | finite_checks fail-fast：loss 检查 + 可训练梯度检查，非有限即抛 `FloatingPointError`；**默认关闭**，门控 `--debug_finite_checks` / `ANIMA_DEBUG_FINITE` / v100 safe | `library/training/finite_checks.py:10-13`、`:16-32`、`:35-48`；`library/training/loop.py:804-808`、`:860-861` |
| 9 | latent NaN 处置只在 VAE 现场编码分支 `torch.isnan` → `nan_to_num`，缓存 latent 分支不做该检查 | `library/training/batch_step.py:85-117`（替换 `:113-117`） |
| 10 | fp32 residual guard（Anima）：fp16 时残差与 FinalLayer 走 fp32，须在 compile 之前 | `library/anima/models.py:2028-2032` |
| 11 | Krea-2 native fp32 guard：非量化 fp16 基座 + 逐参数有限性校验 + fp32 last.linear | `library/models/krea2_raw/native_precision.py:9-14`、`:32-49`、`:52-63`、`:66-88` |
| 12 | 遥测：`precision_report.json` 记 amp 段与 `updates.attempted/effective/skipped` | `library/training/precision_report.py:152-162`、`:200-209`、`:211-219` |
| 13 | **A 无 grad_norm 账本**：`mean_grad_norm` / `mean_combined_norm` 均为 `None` + not tracked here | `library/training/loop.py:1194-1195` |

### 三条缺口（严禁混写）

#### D1 `--debug_finite_checks` 的梯度有限性检查早于 unscale/clip（本分支未修复）

- 位置：`library/training/loop.py:861` 的 `check_trainable_grads_finite` 位于 `:825` backward 之后、`:898` clip（accelerate 先 unscale）之前，读到的是**仍乘了 scale** 的梯度。
- CPU 实测（P3）：loss 有限、`scale=65536`、真梯度 2.0 → 梯度 inf → 抛 `FloatingPointError`；同一时刻真 scaler 会 skip 且权重逐位不变。
- 判定：**当前实现违反 E5，例行溢出会被 fail-fast 判成训练失败。** 影响面是开启 debug 诊断/烟测的 fp16 运行；本分支未修复，生产代码改动不在本轮范围。

#### D2 默认路径根本没有有限性检查（不是 D1）

`--debug_finite_checks` 是 `store_true`、默认 false（`library/anima/training.py:458-463`），门控在 `library/training/finite_checks.py:10-13`；loss 检查与梯度检查（`loop.py:804-808`、`:860-861`）都只在 debug 开启时执行。即默认生产路径既没有误报（D1）、也没有任何梯度非有限处置——NaN 梯度会直接进 `optimizer.step`。D1 是「诊断工具误报」，D2 是「默认路径缺保护」，两件事。

#### D3 adaptive 路径把跳步当致命错误

`library/training/adaptive_runtime/training_runtime.py:71-73` 对 `optimizer_step_was_skipped` 直接 `raise RuntimeError("Adaptive worker stopped: loss scaler skipped optimizer update")`，与「溢出 ≠ 失败」的 scaler 语义相反。这是独立缺口，不并入 D1。

### E3 的交付形态与接缝清单

**结论：不改生产代码无法给出真实的窗口记账断言。** 五个计数器里 optimizer / scheduler / global_step / ckpt 只有「会发生」的观测点、没有「窗口无效时不发生」的判定；模型 EMA 与账本三元组（`total_loss` / `num_steps` / `last_loss`）在 A 侧连字段都没有（`library/training/` 0 处）。因此 E3 只交付可执行规格 + 缺口钉，`strict xfail`，**规格未实现**，不声称支持。

`_WINDOW_SEAM_CHECKLIST`（未来实现接缝，本轮只记录不修复）：

1. `loop.py`：在 accumulate 体内暴露窗口起点快照（optimizer_step / scheduler_step / global_step / ckpt 计数）。
2. `loop.py`：新增 `window_invalid` 标记，由非有限 loss 置位，并在边界块入口短路丢弃整窗。
3. 丢弃路径必须同时跳过 `optimizer.step` / `_step_lr_scheduler` / `global_step` 自增 / `maybe_save_step` / EMA。
4. `batch_step.py:113-117`：`nan_to_num` 只处理 latent，不构成 loss 侧窗口判定，需与 `window_invalid` 接线。
5. loss 账本：新增 `(total_loss, num_steps, last_loss)` 三元组，窗口起点快照、丢弃时逐位回滚。
6. scaler 记账：丢弃窗口时复位该窗口的 scaler 记账（A 无 `reset_scaler_step_state` 等价物，全仓 0 处）。

容差口径：默认 **abs=0（逐位相等）**；只有未来实现选择「同一窗口内 loss 换序重新求和」才允许 abs=1e-12 级误差，本规格不授予该宽松度。

## CUDA 热测补记（2026-11-06）

本节只补事实，不改上文任何 CPU 判据的结论与强度分级。所有实验只用合成张量，未加载模型、未读训练数据、未启动项目训练，也未修改 `library/` 与 `networks/`。

### 环境与设备

- 硬件：2× Tesla V100-SXM2-16GB（sm_70，16144 MiB），UUID `GPU-b4b014b2-953b-28e8-b8c1-425c4a401165`（index 0）、`GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061`（index 1）；两组实验分别固定 `CUDA_VISIBLE_DEVICES` 独占单卡。
- 软件：torch **2.10.0+cu129**、CUDA runtime 12.9、accelerate 1.13.0、python 3.13.9、cudnn 91002；分支 `codex/fp16-nan-guards` @ `dfae1a4d`。
- **版本偏差必须注明**：`pyproject` 要求 `torch>=2.12.0,<2.13`，上文 CPU 面记录的验收环境是 torch 2.12.0+cu130，本轮实测只有 **2.10.0+cu129** 可用（目标 worktree 内的 `.venv` 是指向已不存在路径的悬空软链，实际借用 `krea2-webui/.venv`）。接缝所依赖的 API（`Accelerator.native_amp`、`GradScaler`、`AcceleratedOptimizer.step_was_skipped`）在 2.10 上行为一致，但**如要求严格对齐，仍需在 2.12 环境复跑同一探针**。

### E2：跳步语义分裂（已复现，结论不变）

用真 CUDA fp16 autocast + 真 `torch.amp.GradScaler("cuda")` + `AcceleratedOptimizer`，按 `loop.py` 的 zero_grad → autocast → backward → clip → step 顺序跑小型合成 Linear：

- 溢出步：调用点执行（`step` 正常返回、`scaler.step` 调用 1 次）→ executed；内层 `optimizer.step` 0 次 → **applied**；`step_was_skipped=True`；参数与 SGD momentum `view(torch.uint8)` 逐位不变；scale ×0.5（2048→1024，默认档 65536→32768）。
- 干净步负控制：内层 step 1 次、权重与动量确实变化；`growth_interval=2` 时 scale 1024→2048，recovery 步恢复更新。
- 边界：溢出步 `clip_grad_norm_` 返回 inf（干净步为有限正值），clip 后梯度被压成 0 —— **跳步由 `unscale_` 锁存的 `found_inf` 决定，不是 step 时刻的梯度值**。
- fp16 参数档的必要条件被实测确认：不打 `patch_accelerator_for_fp16_training` 时，真 CUDA 直接 `ValueError: Attempting to unscale FP16 gradients.`。

A 侧仍无命名双语义：`optimizer_step_executed` / `optimizer_update_applied` 在 `library/`、`networks/`、`web/`、`scripts/`、`train.py` 均 0 命中，运行时 `hasattr` 亦为 False；可观测载体只有 accelerate 的 `step_was_skipped` 与 `_accelerate_step_called`。因此本节只加强「executed ≠ applied 在 accelerate 原语层成立」，**不改变 E2「A 侧语义未命名」的部分成立判定**。

### E5 / D1：生产接缝在 CUDA 上复现（缺陷成立）

用真 `Accelerator(mixed_precision="fp16")` + 真 `GradScaler` + 生产 `patch_accelerator_for_fp16_training`，直接调用生产 `library.training.loop._run_step`（`trainer`/`state` 仅作为 `SimpleNamespace` 数据容器，被测接缝不 mock）：

- `debug_finite_checks=True`：抛 `FloatingPointError: non-finite trainable gradients after backward: weight: dtype=torch.float16 shape=(1, 1)`，traceback 落点 `loop.py:861 _run_step` → `finite_checks.py:46`。
- 同构造 `debug_finite_checks=False`：无异常、`step_was_skipped=True`、权重逐位不变、scale 65536→32768 —— 即真 scaler 会正常 skip，而 debug 检查把它判成训练失败。
- 无溢出正例（证明该接缝不是恒真失败）：真梯度 0.5（×65536=32768 有限）时检查放行，权重 1.0→0.9501953125 确实更新。

**归因（重要，避免把 D1 误判成纯排序问题）**：解析模型 `w=1, x=1, y=0`，loss 有限、真梯度 2.0、scale=65536 → `2.0×65536=131072 > 65504` 使 fp16 梯度溢出为 inf。只把 unscale 提前，梯度仍是 inf；把 clip+unscale 提前，`inf×0=NaN` 仍触发检查。**因此调整检查顺序修不好 D1，缺的正是 E5 的 scaler 所有权判定**：必须先知道「这一步的梯度非有限应由 scaler 拥有」，保护层才应当不动作。

前置条件复核：`import library.training.finite_checks` + `library.training.loop` 后 `cuda_initialized=false`、显存分配 0 MiB、`AcceleratorState._shared_state` 为空，说明该复现没有隐式训练副作用。

### H1 / H2 / H8 合成探针（`bench/fp16_precision/run_cuda_probe.py`）

两张卡各自独占运行，`--gpu-label gpu0/gpu1` 输出除标签外**逐字段一致**，三项均 pass：

| 探针 | 断言 | 结果 |
|---|---|---|
| H1 | accelerate `backward → unscale+clip → step` 与直接 `GradScaler` 接缝的 loss/scale/grads/params 逐字节相等，且参数确实相对初值移动（负控制） | pass（两卡一致） |
| H2 | CUDA fp16 unscale 的表示窗口采样，并要求 `found_inf` 与「是否上溢」「是否被清零」语义一致 | pass；窗口 `[5.96e-08, 0.99951171875]`，1e-8 归零、1e-7 保留约 19% 相对误差、1.5 → inf 且 `found_inf=1` |
| H8 | `GradScaler.state_dict()` 保存后用**不同** `init_scale`/`growth_interval` 构造的新对象恢复，再跑同一续跑序列 | pass；saved/restored scale 均为 16.0，续跑 scale 轨迹 `16→16, 16→8, 8→8` 完全一致 |

边界：H1/H2/H8 只覆盖合成张量（`Linear(8,4)` 与 `(1,)` 参数），**不代表**真实 trainer / 模型族 / 数据链路；H8 只验证 scaler 轨迹，不含 checkpoint 文件往返与跨设备搬运。

### 与既有文档的两处事实更正

1. 跳步标志的硬错误消费者不是「两处」。`library/` 实际 5 处引用：`precision_report.py:206`（计数）、`auto_block_swap/probe.py:197`、`auto_block_swap/online.py:214`、`adaptive_runtime/training_runtime.py:72`、`adaptive_runtime/native_runtime.py:100`；其中把跳步当致命错误的路径需按 3 处核对。**主循环 `loop.py` 0 处消费的结论不变。**
2. D1 的两处梯度检查（`loop.py:842` 门控 + `:861` 调用）位于 `accelerator.sync_gradients` 门控内，上文未提及该门控，但结论不变。

## 待热测清单

| # | 要验什么（可判定断言） | CPU 给不出结论的原因 | 需要哪张卡 / 条件 |
|---|---|---|---|
| H1 | 真实 CUDA autocast fp16 下三接缝（backward / unscale+clip / step）的位级回归：固定 seed 跑 3 步 dump grad 字节，与参照路径逐字节比对 | CPU 无 CUDA kernel，fp16 累加顺序与 tensor core 不同 | 1× 支持 fp16 的 NVIDIA 卡（sm75 或 sm80+ 均可，须真 CUDA + autocast） |
| H2 | fp16 参数档 unscale 的 kernel 数值（inf 检测与舍入）：对比 fp32 master 路径，核对落在 P9 窗口内 | 只能算表示窗口（P9），拿不到 CUDA unscale kernel 实际舍入 | 同 H1 的 1× 卡；需能同机对照 fp32 master |
| H3 | 跨 rank `found_inf` 传播：单 rank 注入 inf，断言全员 skip 且 scale 同步 | 需要 ≥2 进程与 NCCL 归约 | ≥2 张同型号 GPU + NCCL（DDP） |
| H4 | 循环级窗口记账（真 DataLoader + gradient_accumulation）：丢弃后 optimizer/scheduler/global_step/EMA/ckpt 全不动、账本回起点 | A 无实现；CPU 只能测规格桩 | 先 CPU mock Trainer 跑规格，再 1× GPU 真机 |
| H5 | fold/merge 后 long-run 数值漂移：fp16 训练 200+ 步比较权重轨迹（只作 L2/热测，不用 L1） | 需要真实训练数百步 | 1× GPU，fp16，200+ 步 |
| H6 | 退档后同批是否重复溢出：记录 scale 与 skip 序列 | 依赖真实 batch 分布与 optimizer 状态 | 1× GPU + 真实数据集 |
| H7 | adaptive skip→`RuntimeError` 的可达性（D3） | 需要 CUDA + adaptive 契约（`accelerator.py:167-170` fail-closed） | 1× GPU + adaptive 配置 |
| H8 | GPU scaler `state_dict` 设备往返 / 断点续训：保存-恢复-续跑 3 步对比 scale 轨迹 | CPU 只测得到 CPU state_dict | 1× GPU；保存/恢复后续跑 3 步 |

2026-11-06 状态更新：H1（合成接缝位级对照）、H2（CUDA unscale 窗口采样）、H8（scaler 轨迹恢复）已由 `bench/fp16_precision/run_cuda_probe.py` 在两张 V100 上跑通，详见上文「CUDA 热测补记」；但都限于合成张量，**H3（跨 rank）、H4（循环级窗口记账）、H5（200+ 步长程漂移）、H6（退档重溢）、H7（D3 可达性）仍未执行**，且 H8 不含 checkpoint 文件往返与跨设备搬运。

## 已知边界

- **CPU 与 CUDA 两个面必须分开引用**：正文 P1–P9 与 24 用例均为 CPU 可判定面；2026-11-06 的 CUDA 结论（E2 复现、E5/D1 复现、H1/H2/H8 探针）只覆盖合成张量，**不含**真实 trainer / 模型族 / 数据链路、跨 rank 归约与长程收敛，见 H3–H7。
- **实测 torch 版本为 2.10.0+cu129**：`pyproject` 要求 `torch>=2.12.0,<2.13`，CPU 面原记录 2.12.0+cu130。接缝依赖的 accelerate 内部面（1.13.0）一致，但严格对齐需在 2.12 环境复跑 `bench/fp16_precision/run_cuda_probe.py` 与静态套件。
- **数字来源**：P1–P9 来自 `bench/fp16_precision/run_probe.py`（CPU-only，可离线重跑；对照表见 `bench/fp16_precision/README.md`），是 CPU 探针而非真机；`/tmp/t9_fp16_static/probe_output.txt` 仅本轮临时产物，不可长期引用。P4 是依赖 `scale=65536` 的比值，P8 是权重增量，两者不混用。不要把 CPU 探针数字外推到其它卡或其它 scale。
- **探针不参与测试套件**：`bench/fp16_precision/run_probe.py` 只再生 P1–P9 读数，不是门禁；契约回归仍在 `tests/test_fp16_nan_static_equivalence.py`（24 用例，22 passed + 2 xfailed）。
- **E1 子谓词 (d) 未完全断言**：只覆盖「不创建 scaler」与 loss 的对象/存储身份；`requires_grad`/`grad` 集合与 `stride`/`dtype`/`device` 不变**未断言**，不得计入已验证面。
- **测试绑定 accelerate 1.13.0**：依赖三个内部面（`native_amp`、`AcceleratorState._mixed_precision`、`step_was_skipped`）；升级 accelerate 时 canary 会失败，必须重新核对内部面后再采信。
- **夹具与生产的差异**：真实 `Accelerator` + 真实 `GradScaler("cpu")`，手工置 `native_amp=True`；模型不进 `prepare`；进程级单例保存-恢复。差异只有「设备是 CPU」与「标志位需手工置位」，但仍是差异。
- **两个 strict xfail = 规格未实现**：E3 窗口记账、E5 所有权判定（含 D1 用例的期望）；不得写成「已支持」。
- **D1/D2/D3 均未修复**：本轮只记录，不改生产代码，也没有提交。
- **测试文件规模**：1086 行 / 24 用例，已接近 AGENTS.md 的 1200 行建议上限；后续新增 E3/E5 内容应按领域拆第二个测试文件。该文件当前是未跟踪状态（`git status` 显示 `??`），交付前需纳入版本控制。
- **未覆盖**：`process_batch_inner` 的 latent `nan_to_num`（需要完整 trainer 上下文）、真实累积窗口的 EMA/ckpt 记账、adaptive 路径的 skip 用例。
