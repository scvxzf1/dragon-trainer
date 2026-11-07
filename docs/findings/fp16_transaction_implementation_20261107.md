# fp16 / E3 窗口事务：统计、多卡一致性与可观测性实施记录（2026-11-07）

状态：实验（实施记录，含未验边界）
适用版本：冻结基线 `a615a9e0`（工作树 `dragon-fp16-transaction-impl-20261107`，分支 `codex/fp16-transaction-impl-20261107`）
相关代码：`library/training/loop.py`、`library/training/window_accounting.py`、`library/training/window_coordination.py`、`library/training/numerical_observation.py`、`library/training/precision_report.py`、`library/training/gradient_sync.py`、`library/training/batch_step.py`、`library/runtime/accelerator.py`、`library/preprocess/text.py`、`library/models/krea2_raw/strategy.py`

## 1. 这次修的是什么

三处**观测/统计**缺陷，不是数值策略改动：

1. **V1（微批级写账本）**：`_log_step` 在每个微批就 `record_loss`，于是一个"有限前缀 + 边界非有限"的
   窗口会把失败尝试的前缀和 NaN 一起写进 `LossRecorder`。E3 事务语义里窗口是**全有或全无**的，
   统计必须随之：只在**提交**时发布该窗口最终生效尝试的完整微批 loss。
2. **V2（无"新提交"门控）**：progress tick 与 step 级 validation 只看 `global_step`，重放/丢窗时
   同一个 `global_step` 会被 tick 两次、validation 跑两遍。改为只在**本次边界是一次新提交**时触发。
3. **V3（单进程谓词）**：`WindowBuffer.failure_reason` 只读本 rank 的 losses 与 `update_skipped`，
   多卡时"一个 rank 重放、其它 rank 提交"会打乱梯度规约的配对，且干净的 rank 已经在判定之前
   调用过 `optimizer.step()`。判定与 apply/drop 必须**先**世界一致再分支。

配套补上可观测性：每次 optimizer 边界的尝试计数（含重放与最终丢窗）、成因三层归因、异步
`begin_step`/`finish_step` 双侧计数、无 scaler 的一次性 warning、缓存/输入非有限值的分类计数。

## 2. 保持不变（用户裁定）

`nan_to_num`/NaN 门控清洗、dtype/缓存字节、E3 策略、普通预算 3（第 4 次连续丢窗熔断）、
adaptive 0（首丢窗熔断）、重放 1 次、commit-only 的 scheduler/`global_step` 时间轴；
不实现饱和转换、A1 跳过 backward、按成因分预算、自动调 LR/回滚/剔样/重建缓存。

## 3. 模块划分（热点文件只接线）

| 模块 | 职责 |
|---|---|
| `window_accounting.py` | 单进程窗口谓词与账本；**新增**整窗发布 `publish_window_losses`、槽位 `slots`、按 L1 原发原因的 `window_discard_causes` 与 `window_discard_propagated`（`failure_reason` 仍是 0 处 `dist.`） |
| `window_coordination.py`（新） | 边界层面的两件事：①世界一致（1 元素 MAX 规约的判定/apply-drop）②每边界尝试记账与三层归因 |
| `numerical_observation.py`（新） | 非有限值观测：`source_nan` / `source_inf` / `cast_overflow` 分类、有界来源身份、计数与节流日志 |
| `precision_report.py` | 新增 `transaction_accounting()` 发布契约字段；legacy `optimizer()` 行为**不变** |
| `loop.py` | 只做接线：整窗发布、提交门控、`agree_*` 两处、`note_attempt(route=…)`、`note_window_decision` |

新增模块均落在实现任务 inScope 的名单内（`window_coordination.py`、`numerical_observation.py`），
没有另造第四套名字；`update_accounting` 的职责并入 `window_coordination.py` 第二节。

## 4. 计数口径（R2 契约 C2）

| 名称 | 定义 |
|---|---|
| `updates.step_attempts` | 每个 optimizer 边界尝试计 1，**含每次重放**、含最终丢窗的尝试（仅窗口路由） |
| `windows.committed` / `discarded` | 逻辑窗口**终态**计数（重放成功只算 1 个窗口） |
| `windows.replayed` | 重放**次数**（`WINDOW_MAX_REPLAYS=1`） |
| `updates.effective` | 被接纳的更新数（未被 scaler/guard/传播拒绝）；E3 路由下 `== windows.committed` |
| `unverified I1` | `step_attempts == committed + discarded + replayed`，**仅当**所有窗口已结束、每次尝试恰好一个边界、直达路由计数为 0 |

legacy `attempted/effective/skipped` 保留基线语义（只走提交路径），在报告 `scope`/`legacy_keys_note` 中
明标 `legacy`；`skipped` 在该路由结构性为 0，跳步/丢弃只能由窗口账本与成因表达。

## 5. 三层归因（R2 契约 C4）

- **L1 原发原因**（本 rank 本次尝试自证）：`nonfinite_loss` / `guard_drop_nonfinite_grad` /
  `scaler_overflow`（**仅当**有活 scaler 且 `optimizer_step_was_skipped` 为真）；
- **L2 世界传播 drop**：本 rank 干净、因世界决定跟随丢弃，单独计数，**不进任何 L1 桶**；
- **L3 世界决定**：`apply/drop` 与 `commit/replay/discard` 直方图。

跨 rank 归约用固定编码 `0/1/2` + `MAX`，注释与文档都写明**这只是线格式约定，不是严重度**，
且**不用于**成因统计（`discard_causes` 只由 L1 构建）。`discard_budget()` 仍只看
`window_consecutive_discards`，不按成因分叉。

## 6. 实测（CPU，`CUDA_VISIBLE_DEVICES=''`，GPU 0 s）

| 命令 | 结果 |
|---|---|
| `git diff --check -- library tests scripts/krea2/preprocess_te_cache.py docs` | rc=0，无输出 |
| 既有 13 文件回归（t2-2） | 见实现记录 §7 收据 |
| 新 CPU 契约 4 文件（t2-3） | 见实现记录 §7 收据 |

覆盖的验收行：M1–M12、M19-R2①②③④⑤、M20–M26、M8-R2①–⑥。新文件：
`tests/test_window_transactional_logging.py`、`tests/test_window_rank_consistency.py`、
`tests/test_nonfinite_observation.py`、`tests/test_fp16_transaction_cuda.py`。
夹具 `tests/window_replay_harness.py` 增加了**默认关闭**的 `stub_step_ticks=False` 模式，
让"tick 是否真的发生"可以在真实 `_log_step` / 真实 validation 上断言（既有 3 个消费者行为不变）。

## 7. 未验边界（不得读成"通过"）

1. **M13–M18 已全部写成真驱动，但只在实现侧自检跑过一次，仍是"未验"**：CUDA 档需要真实设备
   与共享 900 s 预算。`tests/test_fp16_transaction_cuda.py` 六行都是真驱动（M13/M14 单卡
   fp16+scaler；M15/M16 双 rank manual NCCL；**M17 双 rank `accelerate` 原生 DDP reducer**；
   **M18 真实 `AsyncGradientSynchronizer` 健康档 + 单 rank 坏窗档**），两 rank 逻辑在新文件
   `tests/fp16_cuda_driver.py`。2026-11-08 自检一次：`RUN_FP16_TRANSACTION_CUDA=1` 下
   6 passed / 0 failed / 0 skipped（33 s 墙钟）。**自检 ≠ 验收**：独立复跑（t3）之前，
   按契约 §C5 相关质量门仍记**未通过**，不得表述为"已实现并验收"。
2. **多卡 CPU 档仍是同构模拟**（fake `torch.distributed`，固定编码 MAX 语义）；真实 NCCL 只在
   CUDA 门那四条两 rank 行里出现，CPU 回归不覆盖这部分。
3. **async `begin_step`/`finish_step` 的接线约束**：`begin_step()` 必须**每个微批**下发
   （`loop.py::_run_microbatch`），`finish_step()` 每个 optimizer 边界一次；一个窗口只下发一次
   `begin_step()` 会让 `finish_step()` 用未规约的本地 payload（自测里表现为两 rank 梯度 hash
   不同）。因此生产接线下 `steps_pending` 会按微批累积（begin > finish），
   `lifecycle` 的"静止为 0"只在 begin 与 finish 配对平衡时成立——M18 用 `m18_paired`
   （1 微批/窗口）与 4 微批/窗口两档分别核对这两种配对形态。
   真实重叠的性能/时序行为仍未量化。
4. **未覆盖**：窗口中途异常/OOM/中断后的实际训练统计、>2 rank、原生 DDP reducer 的
   reducer 桶顺序、窗口中途 `zero_grad` 的世界一致性、`_run_step` 直达路由与窗口路由混跑的
   端到端训练、CPU 缓存重建后的真实数值一致性。
5. 读取层保持原样：NaN 门控 + `nan_to_num`、Inf-only 不改数值、清洗后仍返回原 dtype；
   任何"是否应该拒训/饱和/自动重建"的策略问题都**未**在本轮改动。

## 8. 只读保护

`krea2-webui` / `krea2-fp16-nan` / `krea2-multigpu` / `lulynx-trainer` 与旧实验产物全程只读；
`krea2-multigpu` 观测到外部在途变更（来源未核），本队未写入、未回滚、未清理。
