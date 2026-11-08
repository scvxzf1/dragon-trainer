# 跨项目 fp16 / NaN 保护对照：lulynx-trainer ↔ dragon-trainer

状态：稳定（对照报告，非实验）
适用版本：本项目代码基线 `a615a9e0`（= `origin/dev`）；本文档基线 `d42cb1c3`（`f375f256` + `d42cb1c3` 均为 docs only）
对照目标：`lulynx-trainer` @ `92f3c07cb`，以其 `guide/fp16_nan_protection_design_2026-11-06.md`（194 行、§1–§10）为纲
相关代码（本项目）：`library/training/loop.py`、`library/training/scaler_ownership.py`、`library/training/window_accounting.py`、`library/training/finite_checks.py`、`library/training/precision_report.py`、`library/training/batch_step.py`、`library/preprocess/text.py`、`library/runtime/accelerator.py`
相关代码（对照目标）：`backend/core/lulynx_trainer/grad_scaler_runtime.py`、`gradient_protection/*`、`precision/fp16_*`、`training_loop/update_execution.py`、`losses/safe_guard.py`、`cache_builder_common.py`、`cache_codec.py`

---

## 1. 范围与方法

- **范围**：两仓 fp16 混合精度、loss/gradient 非有限值处置、累积窗口记账、SafeGuard 等价物，以及训练缓存的 fp16 写入边界。
- **方法**：只读静态阅读两仓生产代码 + 定向 `rg`/`git` 核查；本项目侧另有 captain 团队执行的 CPU 测试实测（见 §8 证据附录）。**未启动训练、未改用户数据、未推送**。
- **证据强度标注**（每条结论必须带其一）：
  - `直读`：直接读到该行代码，语义由代码结构决定；
  - `直读+复核`：本轮独立回读复核过的关键行；
  - `测试`：有断言该行为的用例锚点（不等于本文作者跑过）；
  - `captain 实测`：由 captain 团队在本轮执行并留下命令与输出（见 §8）；
  - `未验证`：对照目标该行为无任何测试覆盖。
- **两个必须分清的口径**：
  1. 对照目标是**可信参照实现，但其 5 条非标准路由零 pytest 覆盖**（§0 覆盖表），因此移植建议一律附「参照实现未验证」，不得当作「已验证能力」引进。
  2. 对照目标文档 §10 实为 **7 条**契约（`guide:188-194`），不是「六条」；本文按 7 条对齐。

## 2. 两份事实表摘要

### 2.1 对照目标（L，lulynx-trainer）

| 主题 | 关键事实 | 锚点 |
|---|---|---|
| scaler 归属 | 自建 provider，仅 CUDA+fp16；`init_scale=65536/growth=2/backoff=0.5/interval=2000` | `grad_scaler_provider.py:34-37,157-170` |
| 启用判据 | 四道拒绝：非 fp16 / 非 CUDA / DeepSpeed / 梯度 dtype 为 bf16；构造失败只 warning 后返回 `None` | `:108-137`、`:141-150` |
| 三条 seam | `scale_loss_for_backward` → `maybe_unscale_gradients_before_clip` → `maybe_step_with_grad_scaler`；无 scaler 全 no-op | `grad_scaler_runtime.py:297,193,333` |
| 双指标 | `optimizer_step_executed` 与 `optimizer_update_applied` 分开维护 | `update_execution.py:403-435`、`grad_scaler_runtime.py:422-438` |
| 跳步语义 | **attempt-indexed**：scheduler 每边界照常 step，显式不以 applied 门控 | `update_execution.py:483-500` |
| 预算 | `consecutive=+1 if skipped else 0`；`budget_count=+1 if (skipped and not retryable) else 0` ⇒ **非连续即清零** | `gradient_protection/runtime.py:142-152`、`:159-162` |
| 边界保护 | `GradientProtection.prepare/finish`，逐边界、不受 interval 门控 | `update_execution.py:704-735`、`runtime.py:56-162` |
| 周期诊断 | `SafeGuard.check` 每 10 步；`record_nonfinite_event` 显式不受 interval 门控 | `safe_guard.py:201`、`:238-262` |
| 恢复层 | 5 值 `SafeGuardAction`（含 `RESET_LR`）+ `max_rollback_count=3` 预算→`STOP` + 健康快照 | `safe_guard.py:37-43,73,315-329` |
| 非标准路由 | 7 个执行路由（closure / DeepSpeed / TurboCore / LOMO-AdaLOMO / 整步 CUDA Graph / GR 降级 / side loops），代码含 9 个 route 标签 | `fp16_route_compatibility.py:1-60` |
| 缓存 | `to_finite_fp16_array` 只判「输出非有限即 raise」；区分「原值非有限 vs cast 溢出」在 `cache_codec` | `cache_builder_common.py:479-500`、`cache_codec.py:151,169` |
| 命名陷阱 | `gradient_protection/gradient_guard.py` 是 **AGC + 梯度集中**优化器包装器，**不是** SafeGuard | `gradient_guard.py:1-30,172` |
| 测试面 | 65 个测试文件、`isfinite` 4 处/3 文件（主题相关 2 文件 3 处）；保护逻辑**无行为级用例** | 见 §2.3 |

### 2.2 本项目（A，dragon-trainer）

| 主题 | 关键事实 | 锚点 |
|---|---|---|
| scaler 归属 | 不自建：只把 `mixed_precision` 透传给 `Accelerator`，scaler 由 accelerate 1.13.0 建（超参即 torch 默认，与 L 同） | `library/runtime/accelerator.py:159-166` |
| 硬门 | 唯一「必须有 scaler」的硬门在 adaptive/native 档 | `accelerator.py:167-170` |
| 时序 | forward `:1097` → debug 检查 `:1103` → 无条件有限布尔 `:1109` → **backward `:1126`（无门控）** → 边界规约（保持 scaled）`1168-1181` → 所有权判定 `:1161` → clip `:1207` → drop `1218-1227` / step `:1228` | `library/training/loop.py` |
| 所有权判定 | `scaler_present` / `scaled_gradient_overflow_is_scaler_owned` / `inspect_grad_ownership` / `_any_rank_bad` | `library/training/scaler_ownership.py:61,84,177,161` |
| 扫描 | `_foreach_norm(..., 2.0, torch.float32)` 一次归约（fp32 累加是回归修复点）；有 scaler 且非 debug 时短路不扫 | `scaler_ownership.py:105-158`、`loop.py:953-989` |
| 窗口事务 | 作废 → **重放 1 次** → 丢弃；只有提交才推进 scheduler/global_step/adaptive/precision | `window_accounting.py`、`loop.py:708-1008` |
| 预算 | 普通训练 `--max_nonfinite_window_discards`（默认 3，连续）；adaptive `--adaptive_max_nonfinite_skips`（默认 0） | `window_accounting.py:198-217` |
| 跳步可见性 | 统一布尔 `state.update_skipped`（scaler 跳步与 guard 丢弃都置位） | `loop.py:260-263,1216-1236` |
| 报告 | `precision_report.updates.{attempted,effective,skipped}`；`skipped` 在循环路径结构性为 0 | `precision_report.py:200-214` |
| latent | 缓存与现场编码两来源都 `nan_to_num(0)`，但判据只 `torch.isnan` | `library/training/batch_step.py:96-107` |
| 缓存 dtype | TE 缓存 dtype = `text_encoder.dtype`（随 `weight_dtype`），**fp16 运行即 fp16 缓存**且无 cast 后检测；latent npz 恒 fp32 | `library/preprocess/text.py:253,118,121`、`model_loading.py:81,91,100,112`、`latents.py:367` |
| 测试面 | 13 文件 **223 passed / 0 xfailed**（captain 实测，命令与清单见 §8） | — |

### 2.3 参照实现覆盖表（移植风险溢价）

| 项 | L 侧覆盖 | 标注 |
|---|---|---|
| A1 非有限 loss 仍 backward | 3 处 `isfinite` 断言（`test_universal_dit_train_step_smoke.py:75,84`、`test_flow_contract_declaration_unit.py:552`；`test_gdlokr_kron_unit.py:79` 属 Kron 单测、与主题无关） | **部分（断言级），行为未验证** |
| A2–A8 | 无行为级用例（`gradient_protection` 仅出现在 `test_package_root_freeze.py:21,25,54,90` 的模块清单冻结断言；跨仓另有 `backend/launcher/api/tests/test_training_recovery_visibility_contract.py:71,78` 解析 SafeGuard rollback 日志字符串、`plugin/pytorch_optimizer-plugin/tests/utils.py:278` 的 `GradScaler(enabled=False)`） | **零覆盖 ⇒ 参照实现未验证** |
| A9 | L 无对应实现（其策略是写入端拒收、训练期不清洗） | **不适用，不得写成「对方已验证」** |

结论：**「抄来即用」项数为 0**，每条移植候选都需我方自建验收用例。

## 3. 语义差异矩阵（41 行，五分类互斥）

分类：**A** 能力缺失（有语义需求、我方未实现）｜**B** 语义一致（实现不同、可证明等价）｜**C** 语义冲突（同一情形处置不同且都非缺陷，需判定）｜**D** 仅命名/组织不同（或需求在我方不成立）｜**E** 对方缺失而我们有。
计数：**A=9 / B=15 / C=3 / D=10 / E=4**。

### 3.1 精度解析与 scaler 启用（L §2）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M01 | fp16 必须 loss scaling，超参一致 | 自建，65536/2/0.5/2000（`grad_scaler_provider.py:34-37`） | accelerate 建，torch 默认同四项（`accelerator.py:159-166` → accelerate `:561-582`） | B | 直读 |
| M02 | bf16 梯度不缩放 | 显式拒绝 bf16 梯度（`:125-131`） | bf16 档不建 scaler | B | 直读 |
| M03 | 用解析后精度决策 | `precision_capability.py:212-240` | `precision_policy.py:19-52`（探测异常 fail-open） | B | 直读 |
| M04 | **路由级 fail-fast gate** | `enforce_fp16_route_compatibility`（`fp16_route_compatibility.py:160-185`），拦截 fused-backward 等路由；**非 CUDA 时 `grad_scaler_required` 返回 False ⇒ `:179-180` 直接 `return []`，对 fp16-on-CPU 无 gate** | 无任何路由级兼容门 | **A3a（A）** | 直读+复核 |
| M05 | 无 scaler 时的降级可观测性 | `logger.warning("fp16 requested but loss scaling was not enabled: ...")` 后继续（**`grad_scaler_provider.py:147`**） | 完全静默（accelerate 不报错、项目无 warning） | **D（warn 语义一致）+ A3b（我方缺 warn，低）** | 直读+复核 |

> **消解记录**：M04 与 M05 曾分别被判 A 与 D，读起来像「同一事实两种分类」。实际是两件事：gate 面 L 有、我方无（A3a）；warn 面两侧降级语义一致、只差可观测性（M05 归 D + 记 A3b）。**不能写成「L 会拦住 fp16-on-CPU，我方不拦」。**

### 3.2 一次更新的保护时序（L §3 / §3.4）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M06 | unscale 先于 clip | 显式 seam（`grad_scaler_runtime.py:193-294`）+ clip 前调用（`update_execution.py:716-726`） | `accelerator.clip_grad_norm_`（`loop.py:1207`），accelerate 内部先 unscale | B | 直读+复核；A 侧有 accelerate 1.13.0 契约 canary 用例（`测试`） |
| M07 | fp16 参数档 unscale 正确 | 手动纪律接管（`grad_scaler_manual_runtime.py`） | monkeypatch `_unscale_grads_(allow_fp16=True)`（`accelerator.py:201-217`），unscale 仍在 fp16 内完成 | B | 直读；A 侧 P9 探针给出可表示窗口 `[5.96e-8, 0.9995]`（文档） |
| M08 | 无 scaler 时非有限梯度先于改参决策 | `runtime.py:74-102` | `loop.py:1161` + `1218-1227`（先 zero_grad、不 step） | B | 直读+复核；`测试`（`test_scaler_ownership_guard.py`） |
| M09 | scaler 溢出归 scaler 所有 | `grad_scaler_runtime.py:97-143` | `scaler_ownership.py:84-102` + `loop.py:967-989` | B | 直读+复核 |
| M10 | 多 rank 一致 skip | `dist.all_reduce(MAX)`（`distributed.py:20-28`） | `accelerator.reduce(max)`（`scaler_ownership.py:161-196`） | B | 直读；`测试`（真 2 进程 gloo：`test_distributed_gradient_sync.py:509`） |
| M11 | 规约保持 scaled | 各 route 保证（`configs_gradient_release_interlock.py:33-35`） | 手写 allreduce 显式保持（`gradient_sync.py:321-331`） | B | 直读 |
| M12 | **非有限 loss 不得继续 backward** | `step_execution.py:624-638` 非有限即 `do_backward=False` | **`:1109` 只算布尔、`:1126` 无条件 `accelerator.backward`** | **A1（A）** | 直读+复核 |
| M13 | 不伪造 detached leaf loss | `loss_finite_check.py:2-7` | 不造假，靠 `drop_update` 阻止提交 | B | 直读 |
| M14 | loss 判定不受 debug 门控 | `training_loop_gradient_mixin.py:118-151` | `loop.py:1109` 无条件（debug 只额外抛异常 `:1103`） | B | 直读+复核 |

### 3.3 跳步与 step 账目（L §3.4 / §5）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M15 | **「执行了 step」与「权重动了」是两个指标** | 双字段 + 质量遥测消费（`update_execution.py:403-435`、`training_quality_telemetry.py:124-160`） | **生产代码 0 处**（`state.update_skipped` 单布尔，`loop.py:260-263,1216-1236`）；且 drop 路径不调 step，故「executed 但未 applied」不可表达。**离线探针已测得该三元组**：`bench/fp16_precision/run_probe.py:138-146` 输出 `call_site_executed` / `inner_optimizer_step_invoked_*` / `optimizer_update_applied` | **A2（A）** | 直读+复核（两侧） |
| M16 | **跳步是否推进进度** | attempt-indexed（`update_execution.py:483-500` 明确不以 applied 门控） | 事务（`loop.py:992-1008`） | **C1（C）** | 直读+复核；`测试`（`test_adaptive_training_loop.py:111-157`） |
| M17 | 失败预算成因分流 | retryable=`scale_before>1.0`，**非连续即清零**（`runtime.py:142-152`），耗尽抛错（`:159-162`） | 只数连续丢窗、不分成因（`window_accounting.py:198-217`） | **A6（A）** | 直读 |
| M18 | step-attempt 账本可观测 | route report 完整（`update_execution.py:403-435`、`grad_scaler_runtime.py:333-389`） | `skipped` 桶结构性永不递增、`attempted` 只数已提交边界（`precision_report.py:200-214`）；测试固化 1/1/0 | **A4（A）** | 直读+复核；`测试` |
| M19 | 同名桶告警 | — | `updates.skipped` 语义已退化，须重命名或拆成因 | D | 直读 |

### 3.4 累积窗口（我方独有结构）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M20 | 非有限 loss → 整窗丢弃、不推进 | `microbatch_execution.py:778-858`、`training_loop_gradient_mixin.py:206-232` | `loop.py:840-905`、`window_accounting.py:163-180` | B | 直读；`测试`（`test_window_replay.py`） |
| M21 | 窗口作废触发集合 | 非有限 loss（+ safeguard 标志） | 非有限 loss **或** 边界 `update_skipped` | **C1 同源（C）** | 直读 |
| M22 | 同窗重放 | 无重放概念 | 重放 1 次、同数据、`_forced_accumulate` 自设边界（`window_accounting.py:46-47`、`loop.py:708-783`） | **E2** | 直读；`测试` |
| M23 | 窗口计数粒度 | 只有丢窗计数 + 非有限微批计数 | `attempts/replays/discards/consecutive` + 账本三元组 | **E（本方更细）** | 直读（但未进报告，见 M18） |
| M24 | 丢窗不污染 loss 账本 | 账目回滚到窗口起点快照（`microbatch_execution.py:783-796`） | 非有限样本不进 `LossRecorder`（`window_accounting.py:220-231`） | B | 直读；`测试`（真实链路 2026-11-07 踩到并修） |
| M25 | 模型 EMA 账本位 | 快照含 `ema_state_dict` | 无 EMA，`ema_observation` 占位槽 | D | 直读 |

### 3.5 非标准路由（L §4）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M26 | 非标准更新路由的 fp16 契约 | 7 个执行路由各自处置（`fp16_route_compatibility.py:1-60` 等） | 全为空（无 closure/TurboCore/整步 CUDAGraph/DeepSpeed/side loop） | **A5（A，义务条件不成立但缺检查表）** | 直读；未验证（L 侧零覆盖） |
| M27 | `--fused_backward_pass` | 有真实预扫路径 | **死开关**：只有 CLI 定义 + 两个断言，未传 `Accelerator`、不进 loop（`cli_args.py:204`、`optimizers.py:59-66`） | **A7（A，潜在风险）** | 直读+复核 |
| M28 | stock fused Adam found_inf 注入 | 有（`runtime.py:93-130`） | 无，但需求不成立（我方从不绕过 `scaler.step`） | D | 直读 |
| M29 | side loop 统一 seam | `side_loop_fp16_scaling.py` | 无 side training loop | D | 直读 |

### 3.6 SafeGuard 与恢复（L §5）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M30 | 边界的梯度有限性判定 | `update_execution.py:704-735` 逐边界 | `loop.py:953-989` 逐边界（有 scaler 且非 debug 短路） | B | 直读；`测试` |
| M31 | 周期 host 诊断扫描 | `safe_guard.py:201`（10 步，采样）；`record_nonfinite_event` 不受 interval（`:238-262`） | 无周期扫描 | D（配置面/诊断层，**不与 M30 比严格度**） | 直读+复核 |
| M32 | 扫描实现模式数 | 4 种（`legacy/batched/foreach/off`）+ triton 地址表 kernel | 1 种（`_foreach_norm` fp32） | D | 直读 |
| M33 | 恢复动作 | 5 值 `SafeGuardAction` + rollback 预算→STOP + 健康快照（`safe_guard.py:37-43,73,315-329`） | 无降 LR / 回滚 / 样本剔除；只有同窗重放 + adaptive 重启 | D（配置面），但**保留语义需求注记**（见 §4.4） | 直读+复核；未验证 |
| M34 | CUDAGraph safeguard profile | `device_assert_only`（`trainer_safeguard_runtime.py:61-88`） | 无整步 CUDAGraph | D | 直读 |
| M35 | 命名陷阱 | `gradient_guard.py` = AGC，非 SafeGuard | 无 AGC | D | 直读+复核 |

### 3.7 局部数值与持久化（L §6 / §10-7）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M36 | fp16 residual 升 fp32 | `precision/residual.py:7-22` | `library/anima/models.py:1483-1496,2028-2032`；krea2_raw 整块 body fp32（`native_precision.py:52-74`） | B | 直读；`测试` |
| M37 | 训练期 corrupt latent 处置 | 写入端拒收、现场不清洗（政策取向见 §4.3） | 现场 `nan_to_num(0)`，且判据只 `torch.isnan`（`batch_step.py:101`） | **C2（C）+ A9（A）** | 直读+复核；`测试` |
| M38 | fp16 cast 持久化量程拒收 | `cache_builder_common.py:479-500`（只判输出非有限）+ `cache_codec.py:151,169`（双条件：`not finite(out) and finite(原值)`） | **无 cast 后检测**；TE 缓存随 `weight_dtype` 写成 fp16（`preprocess/text.py:118,121,253`、`model_loading.py:81,91,100,112`）；latent npz 恒 fp32（`latents.py:367`） | **A8（A，条件适用）** | 直读+复核 |
| M41 | **清洗判据只覆盖 NaN，未覆盖 ±inf** | 无对应实现（对方写入端拒收） | `batch_step.py:101` `if torch.any(torch.isnan(latents))` | **A9（A）** | 直读+复核 |

### 3.8 证据体系（反向优势）

| ID | 维度 | L | A | 类 | 证据 |
|---|---|---|---|---|---|
| M39 | 定向测试覆盖 | 保护逻辑无行为级用例（见 §2.3） | E1–E5 静态套件 + D1/D2/D3/E3 定向用例 | **E1** | 直读（L 的空覆盖）；captain 实测（A） |
| M40 | 真机 / 真实链路证据 | 无（guide §8 自认静态阅读） | 真 Accelerator + 真 scaler；V100×2 合成热测；**真 2 进程 gloo DDP**（`tests/test_distributed_gradient_sync.py:509`）；2026-11-07 真实小规模训练 | **E（三链归属见 §8）** | `测试`（DDP 用例存在）+ captain 实测 |

### 3.9 C 类三条的判定

- **C1｜例行 scaler 溢出跳步的进度语义（M16/M21）**：两侧都刻意且都有书面理由——L 走 attempt-indexed（`update_execution.py:483-500`：跳过步若不计入会把 LR 曲线拉长）；我方走事务语义（未完成真实权重更新的窗口不推进任何进度副作用），且我方这条来自**用户裁定**（重放 1 次 + 退化丢窗 + 预算），不是遗漏。**判定：各自保留，不做单方面替换。** 我方代价必须写明：每次例行溢出多付一整窗 forward+backward、该步不推进 `global_step`。**收敛点应落在 A6（让例行退档不消耗熔断预算与窗口成本），不要改事务语义**——改它会连带重做 `completed_steps` 恢复游标与 precision 计数。
- **C2｜训练期 corrupt latent 处置（M37）**：L 让坏值可见（写入端拒收，现场不清洗），我方 `nan_to_num(0)` 静默清零继续练。**判定：先补可观测性**（warning + 计数 + 来源/文件名），**再**把拒收前移到缓存写入（A8）；不要一步改 fail-closed（既有用户缓存可能已含坏值）。
- **C3｜失败预算的计数单位（M17 附属）**：L 数「不可退档的 skip 事件」，我方数「连续丢窗」；两单位不可换算（一窗可含多微批）。**判定：保留窗口单位，但必须分成因（并入 A6）。**

## 4. 语义冲突与缺口的成文处置

### 4.1 A3 拆两档（消解内部矛盾）

- **A3a｜gate 面**：L 有路由级 fail-fast（`fp16_route_compatibility.py:160-185`），我方无。**范围必须收窄**：L 的 gate 在非 CUDA 时是 no-op（`grad_scaler_required` `:121-122` 返回 `False` ⇒ gate `:179-180` 直接 `return []`），所以它拦的是「路由不可缩放」，不是 fp16-on-CPU。我方若在启动期做 `error`，那是**我方自选的更严策略**，不是移植 L 的行为。
- **A3b｜无-scaler 面**：L 只 `logger.warning("fp16 requested but loss scaling was not enabled: ...")` 后继续（**`grad_scaler_provider.py:147`**）；我方连 warning 都没有 ⇒ 用户只能从 `precision_report` 反推。**这一档才是真正的「移植」**，成本约一行 warning。

### 4.2 A6 的准确语义（不许加强）

L 的预算是**非连续即清零**：`budget_count = budget_count + 1 if (skipped and not retryable) else 0`，retryable 要求 `scale_before > 1.0`。我方只数连续丢窗、不分成因。
**证据强度：代码路径推断，未复现。** 已知代价两条：① 每次例行溢出多付一整窗计算且该步不推进 `global_step`；② 退档与 guard 丢弃在计数上不可分。
**不得**表述为「健康 run 会被误杀」——重放本身会吸收一次退档。
**Backlog `BL-P1`（独立条目，未完成）**：真实 Accelerator + 真 scaler 跑三段窗口序列——(a) 溢出→重放成功提交；(b) 溢出→重放仍溢出（应归 retryable 且**复位**计数）；(c) scale 到底仍溢出（应计入熔断）。**复现后才可升级表述。**
**验收条件（比 L 更严）**：`scaler_backoff` 必须① 区分成因、② **把 `consecutive_discards` 复位为 0**（仅「不加一」会比 L 更松）、③ 给窗口总量上限（如 `--max_scaler_backoff_window_discards` 默认 8），否则「退档↔真 NaN」交替可无限消耗算力。

### 4.3 C2 / A9 的因果链与政策依据

**因果链（只走 text cache 侧）**：`--mixed_precision fp16` → TE 缓存写入 fp16 Inf（`preprocess/text.py:118,121` 直接 cast，无 cast 后检测）→ 训练读入嵌入含 Inf → `batch_step.py:101` 的 `torch.isnan` **不触发** → 下游 NaN → E3 整窗重放/丢窗兜底。**不崩训练，但以「反复丢窗」烧算力且掩盖根因。**
**禁止**把 `latents.py:367`（latent npz 恒 fp32，`.float().numpy()`）拉进链——latent 侧无 fp16 溢出路径。
**政策依据（收窄表述）**：L 的 `newbie/newbie_loader_dit_block_execution.py:82-87` 有书面注释反对在 modulation 注入点用 `nan_to_num`（「turns a NaN — which the trainer watches for and acts on — into a plausible number that propagates silently」）。这是**该注入点的局部原则**，不是全仓准则，引用时须带这一限定。
**计数口径**：L 全仓 `nan_to_num` = **`backend/core/lulynx_trainer` 内 14 个非测试文件 / 23 处**（全仓 23 文件 / 39 处），全部是局部项，不含主 loss。**此前「10 处」的说法已作废。**

### 4.4 D 类中的语义需求注记（不因归 D 而消失）

M33（恢复层）与 M34（CUDAGraph profile）按裁定归 D 类配置面/组织差异，不计入我方能力缺失。但**「我方无回滚」是自身文档已承认的局限**（瞬时故障的重启粒度只有「同窗重放 1 次」与「adaptive 从上次提交点重启」），该需求的成本收益由移植评估判定（结论见 §5：**不移植**，写进已知局限）；且对方该部分零 pytest 覆盖——但注意 `backend/launcher/api/tests/test_training_recovery_visibility_contract.py:71,78` 说明 rollback 事件在**消费侧**有日志契约用例，所以精确说法是「无 SafeGuard 行为用例，事件消费侧有契约用例」。

## 5. 移植建议排序

排序依据：价值 / 成本；「参照实现未验证」已作为风险溢价计入成本。

| # | 项 | 价值 | 成本 | 落点 | 备注 |
|---|---|---|---|---|---|
| 1 | **A3** fp16-无-scaler 启动期检查 | 3 | 1 | `library/training/compat_matrix.py`（纯函数 error/warning，CLI + Web preflight 共用） | 默认只 **warn**；`error` 仅覆盖路由面（将来接 A7 时生效）。理由是「把静默变显式 + 为将来路由留门」。**不要在 `prepare_accelerator` 里 raise**——会打断 `tests/test_fp16_nan_static_equivalence.py:157` 的 CPU fp16 无 scaler 夹具（`test_moss_image_fp16_amp.py:74` 是 hardware 门控 CUDA 用例，打不到，不能当理由锚点） |
| 2 | **A8** fp16 TE 缓存 cast 后量程拒收 | 4 | 2.5 | `library/preprocess/text.py` | 分两步：步骤 1 先 warn + 计数（保持行为）；步骤 2 再 fail-closed（需用户确认）。文案可参照 `cache_codec.py:151,169` 的双条件；**注意 `to_finite_fp16_array` 那一支不区分来源**。参照侧零测试 ⇒ 成本 2.5 |
| 3 | **A6** 失败预算成因分流 | 4 | 3 | `library/training/window_accounting.py` | 复用 `scaler_ownership` 的 scaler-owned 判据；**必须三件套**（区分成因 / 复位 / 总量上限，见 §4.2）；需用户裁定；前置 `BL-P1` |
| 4 | **A4** step-attempt / window 账本发布 | 3 | 2 | `library/training/precision_report.py` | A6 的前置；顺带修我方 F4。**最强动机**：`docs/findings/fp16_nan_static_equivalence_20261021.md:43-47` 明文写「跳步会计入 `updates.skipped`」，该承诺在 E3 落地后已结构性失效 ⇒ 不只是缺诊断，而是文档在陈述不存在的行为，落地时须同步改写该句 |
| 5 | **C2** 保留清洗 + 补 warning/计数/来源 | 3 | 1 | `library/training/batch_step.py` | 不在此步 fail-closed；拒收前移给 A8 |
| 6 | **A1** 非有限 loss 不 backward | 2 | 1.5 | `library/training/loop.py` | 须同时挡 `run_after_backward` 与边界梯度规约。**收益口径**：坏窗内剩余微批的 backward 全免 ⇒ 最小形态 `2/(3·n_acc)`、扩展形态 `2(n_acc−k+1)/(3·n_acc)`（`force_skip` 只挡 step、`loop.py:1126` 仍 backward）；采纳扩展形态价值可到 3。边界微批（`sync_gradients=True` 且 `force_skip`）下 `zero_grad`/committed 判定的交互需逐案确认 |
| 7 | **A5** 路由 fp16 检查表 | 2 | 0（零代码） | 文档 + 评审检查项 | 只搬「新增路由必须回答：grads 可读吗 / 有 abandon 点吗」两问，不搬代码 |
| 8 | **A2** | — | — | — | **不做**：`executed ≡ applied`（drop 路径不调 step），双字段无信息量；A4 落地后可顺带给出两列，成本从 4 降到约 1 |
| 9 | **A7** `--fused_backward_pass` | — | — | `cli_args.py` / 文档 | **清理死开关**，不移植：两个处置选项留用户——① 接线时按手动 unscale 纪律（参照 `grad_scaler_manual_runtime.py` + `fp16_fused_backward_scaling.py`）；② 删掉或标「未实现」。一旦接线，fp16 下更新发生在 backward 内、无可跳步边界，会落进 L 的 LOMO 型坑 |

**隔离区（成本最低 0.5，但修复不在本对照团队范围，需用户批准）**
- **A9**：`batch_step.py:101` 判据 `torch.isnan` → `torch.isfinite` + warning + 计数（含来源标签）。与 A8 构成成本阶梯（**A9 先**），且是 C2 政策的前置。

**不移植清单（附理由）**
- **9 路由 / 7 seam 框架**：我方无 closure / TurboCore / fused-backward / 整步 CUDA Graph / DeepSpeed / side loop，照抄 = 死分支 + 假安全。
- **DeepSpeed 分支**、**整步 CUDA Graph 图内 found_inf**：无对应集成（`dynamo_backend` 强制 `NO`，只有 inductor per-block）。
- **手动缩放纪律全量**（`grad_scaler_manual_runtime.py`）：会引入第二个 scale authority，与既有 scaler 归属冲突。
- **SafeGuard 恢复层**（降 LR / 回滚 / 健康快照 / 样本剔除）：零测试 + 与 E3 窗口事务重叠 + 快照双份真相 ⇒ 不移植，写进已知局限（M33 的语义需求注记保留）。
- **interval 扫描 + 4 种模式 + 地址表 kernel**：我方扫描点已是最廉价合法位置；`legacy` 模式会带回已修的 fp16 归约误报。
- **stock fused Adam found_inf 注入、`so_*` 六项配置、AGC wrapper、Gradient Release / side-loop / EMA**：需求在我方不成立或属正交能力。
- **C1**：不移植（各自保留，收敛点在 A6）。**C3**：并入 A6。

**落地批次**：批次 0 A9（需批准）→ 批次 1 A3 + C2 + A5 → 批次 2 A8 步骤 1 → 批次 3 A4（+F4/F3）→ 批次 4 A6（先 `BL-P1`）→ 批次 5 A1；A7 / A2 / C1 / C3 / 恢复层按非移植项落文档。

## 6. 我方自身文档债（本对照顺带发现）

| # | 问题 | 状态 |
|---|---|---|
| F1 | `docs/findings/fp16_nan_static_equivalence_20261021.md:43-47`、`:92` 仍描述 E3 **之前**的语义（「跳步仍推进 scheduler」「precision 记 skipped」），与同文 `:389-400` 及 `tests/test_adaptive_training_loop.py:111-157` 相反 | **待处理**（A4 落地时同步改写） |
| F2 | 同文 `:181`（P6 行号表）、`:231` 仍引用旧 `loop.py:805/861/898/906`；当前实际为 **`:1103`（debug 门）/`:1109`（无条件布尔）/`:1161`（guard 调用）/`:1207`（clip）/`:1228`（step）**，`_guard_nonfinite_grads` 定义在 `:953` | **待处理**。注：该文档 `:8` 自带「行号沿用 11b7bef5 冻结值」的声明，故属已声明的旧口径，严重度低于「未声明漂移」；但「无 diff」的自述在 E3 落地后已失效 |
| F3 | adaptive 的 `skipped_steps`/`consecutive_skips` 预算在窗口路径**不可达**（`after_optimizer` 只在提交路径发出），实际熔断由丢窗预算完成 ⇒ 两套预算并存、只有一套生效 | **待处理**（随 A4/A6） |
| F4 | `nonfinite_grad_skips` 只有 warning + 计数、未发布；跳步时 `hydra_up_grad` 可能读到上一步 stale stash | **待处理**（随 A4） |
| F5 | 同文测试数字曾多处互相矛盾且全部过期（`:6/:34/:86/:93/:156/:159/:384`），其中 `:86`「唯一 xfail 仍为 E3 规格」与 `:384`「strict xfail 已清零」直接冲突 | **已修复**：captain 分两次提交 `f375f256`（顶部新增当前基线 + 历史快照标注）与 `d42cb1c3`（`:10` 改 1161 行 / 24 用例 / 24 passed / 0 xfailed；`:160/:164` 显式标注 2026-10-21 历史快照） |

## 7. 本对照未验证什么

1. **A6 的现状行为未复现**：§4.2 的代价属代码路径推断，`BL-P1` 三段窗口序列尚未执行；复现前任何「会/不会熔断」的强结论都不成立。
2. **A1 扩展形态未验证**：边界微批（`sync_gradients=True` 且 `force_skip`）下跳过 backward 与 `zero_grad`、committed 判定的交互，需实现前逐案确认。
3. **A8 步骤 1 的真实溢出率未知**：TE 缓存是否真的写出过 Inf，取决于具体模型的 hidden state 量程，未统计。
4. **对照目标的非标准路由行为全部未验证**：其 closure / TurboCore native / LOMO-AdaLOMO / 整步 CUDA Graph / side loops 五条无任何 pytest 覆盖，本文对其「当前处置」的描述只有代码直读，没有运行证据。
5. **跨 rank 的重放分支未验证**：真 2 进程用例覆盖的是梯度同步与跳过决策；`_any_rank_bad` + `_forced_accumulate` 在 `num_processes>1` 下的重放组合、以及 `end_of_dataloader` 窗口的重放边界，仍待热测。
6. **数值与性能均未测**：本文不含任何新的精度/性能数字；引用的 CPU 探针数字（P1–P9）属既有 findings 的 CPU 结论，不可外推到其他卡或其他 scale。
7. **A9 的修复不在本对照团队范围**：本文只登记缺陷与因果链，落地需用户批准。
8. **两侧文档都有陈述未核实项**：`docs/findings/fp16_nan_static_equivalence_20261021.md:43-47` 的承诺已失效（F1）；对照目标 `guide` 的 §8 自认其盘点未跑任何测试。

## 8. 证据附录（captain 团队本轮实测，原文收录）

```text
# Captain 实测证据（供 t6 直接引用，2026-11-07）
# 代码基线: a615a9e0 (= origin/dev)   docs HEAD: d42cb1c3
# 解释器: /mnt/SD_Dataset/训练器相关/krea2-webui/.venv/bin/python (torch 2.10.0+cu129, accelerate 1.13.0, py3.13.9)

## 证据 1：13 文件 223 passed / 0 xfailed
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  tests/test_window_replay.py \
  tests/test_fp16_nan_static_equivalence.py \
  tests/test_adaptive_nonfinite_skip.py \
  tests/test_scaler_ownership_guard.py \
  tests/test_latent_nan_guard.py \
  tests/test_distributed_gradient_sync.py \
  tests/test_adaptive_training_loop.py \
  tests/test_adaptive_training_supervisor.py \
  tests/test_adaptive_retry_policy.py \
  tests/test_adaptive_runtime_process.py \
  tests/test_auto_block_swap_probe.py \
  tests/test_dynamic_block_swap_controller.py \
  tests/test_training_precision_report.py -q
# 输出: 223 passed in 23.74s  （复跑第二次 22.73s，同为 223 passed）

## 证据 2：单文件读数（与 t5 审查方一致）
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_fp16_nan_static_equivalence.py -q
# 输出: 24 passed in 8.48s   （文件 1161 行，pytest.mark.xfail 计数 = 0）

## 口径要求
- 引用时必须附「文件清单 + 命令 + 输出」，不得只写数字。
- t5 审查方自选另一套 13 文件集得 204 passed，属不同集合，不是矛盾。
```

**附录说明**：
- 解释器路径跨 worktree（`krea2-webui/.venv`），依赖面由 `uv.lock` 锁定，**按原样保留**。
- 本附录**只覆盖 CPU 测试面**。以下三条结论**不在本文件内**，归属为「由 captain 团队本轮执行，证据在 findings 文档与相应任务记录中」：① 真 scaler 溢出重放；② 2 进程 NCCL 中间微批不 reduce；③ Krea-2 512 bucket 注入 NaN、整窗重放一次只推进 1 步。**不得并入本附录冒充同一份测试输出。**

## 9. 变更记录与口径订正留痕

| 日期 | 内容 |
|---|---|
| 2026-11-07 | 首版落盘。矩阵 41 行（M01–M41），A=9 / B=15 / C=3 / D=10 / E=4 |
| 2026-11-07 | **口径订正（保留痕迹）**：A8 曾据「我方缓存全 bf16」被判为**不适用**；复核 `preprocess/text.py:253`（`cache_dtype = text_encoder.dtype`）与 `model_loading.py:81,91,100,112`（TE 按 `weight_dtype` 加载）后订正为**条件适用**——fp16 运行下 TE 缓存即 fp16 且无 cast 后检测。仅「latent npz 恒 fp32」那半句成立 |
| 2026-11-07 | **新增缺陷 A9**：`batch_step.py:101` 清洗判据只覆盖 NaN、未覆盖 ±inf；与 A8 构成因果链（只走 text cache 侧） |
| 2026-11-07 | A3 拆 A3a/A3b 并附 M04/M05 消解记录，明写「L 对 fp16-on-CPU 只有 warn、不是 fail-fast」 |
| 2026-11-07 | A2 口径由「全仓 `rg`=0」改为「**生产代码 0 处、离线探针已测得**」 |
| 2026-11-07 | A6 语义由「不计入预算」改为「**非连续即清零**」，并追加复位与总量上限两条验收条件 |
| 2026-11-07 | 参照实现测试覆盖口径统一为「65 文件 / `isfinite` 4 处 3 文件 / 主题相关 2 文件 3 处 / **引用存在但无行为级用例**」；`nan_to_num` 计数为 lulynx_trainer 内 14 文件 23 处（此前「10 处」作废） |
| 2026-11-07 | F5 由 captain 分两次修复（`f375f256`、`d42cb1c3`）；本文代码基线 `a615a9e0` 与文档基线 `d42cb1c3` 分开标注 |
