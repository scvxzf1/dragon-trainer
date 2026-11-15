---
状态：审阅报告（t4 quality review，round-1，已完成）
适用版本：codex/moss-pp-v0-20261112 @ 6610f710 + 未提交工作树（本轮绑定 SHA 见 §1）
相关代码：library/training/pp/trainer_{admission,inputs,resume,session}.py、train_session.py、cli_entry.py、library/runtime/accelerator.py、library/training/pp/{gate,moss_runtime}.py
---

# MOSS 主训练 PP 接线审阅（t4，2026-11-15）

## 1. 审阅对象与字节绑定（防内容变化冒充旧验收）

审阅 = t2 实现的最新字节（含 t10/t13/t14/t15/t16 等修复轮后的现状）；**全部 SHA 本轮实测**，非转录：

| 文件 | SHA256 |
| --- | --- |
| `library/training/pp/trainer_admission.py` | `df4d36ef423c4326446224ac2d6a9253fd0f73794265263ad151e3b47afa6185` |
| `library/training/pp/trainer_inputs.py` | `b1dddfd8f6488b44b56e63d109fcf7f95001d4179af0f39a1e80f2b2f2accd04` |
| `library/training/pp/trainer_resume.py` | `0a556adb8bfe8f976fe0b97bd2a95d6c50ee41344528df9d4aca1b8254c3898c` |
| `library/training/pp/trainer_session.py` | `003772022b34c363b1a8665d52bd1582f357a6bea865ce04f61eb37259f3344d` |
| `library/training/pp/moss_runtime.py` | `243c33140a5b1b6283870c329d1690ad2e277ec0c5ca1309b46bac735f86e09e` |
| `library/training/train_session.py` | `5039caa281db4080be037aa58e832db2db7605a60941466f32909d5f1220a126` |
| `library/training/cli_entry.py` | `540b2453bc1c7706c5d5d82c87a4daca7c78b0b2e05f332d111bdb269067b985` |
| `library/runtime/accelerator.py` | `ca9c075bdb359af78dc76e9e8cf0ed13cb0a15ba2fe0b9e81af4436f6cc6af63` |
| `library/training/pp/gate.py` | `c809b262cf4b46e788fe8ad1483d058ab725e8b89f32809e1e59fbab986db89a` |
| `library/training/pp/checkpoint.py` | `309247a7b28a7646e315f96c9e9947833fe131a6effdde07935eaf2e3113874f` |
| `tests/test_moss_image_pp_trainer_hot.py` | `07da5ddf2d2dc617ee6f56bd8a882303fe5e3bc746eecd394928ffc69a0d156e` |
| `docs/proposal/moss_image_pp_trainer_wiring.md`（v3） | `77b514d8c18c129bd89ab6fc1e8adbd38c818556e74ad9001ec1b2c9bc889de9` |

git HEAD `6610f710`，`git diff --check` 干净（本审阅轮实测）。四处 trainer 小模块合计 **1248 行**，均为新增小文件。

## 2. 主入口真实接线（逐行实读）

- **CLI 预检**（`cli_entry.py:84-100`）：`pipeline_parallel_enabled` 时先 `require_pipeline_runtime`（world_size 来自 `WORLD_SIZE`，非整数即 ValueError）再 `require_moss_trainer_admission`；无 wired 路径的族在 CLI 层即命名拒绝（`pp_trainer_path_unavailable`），**永不回落 DDP**。
- **唯一执行分叉**（`train_session.py:506-540`）：冻结分叉点在共享 cache/model/network setup 之后、`setup_optimizer_and_dataloader` 之前，thin dispatch 进 `run_pp_trainer_session`（handoff 传共享产物，不重建）。全库 `run_pp_trainer_session` 调用点仅此一处（grep 实测，pp/__init__ 仅导出映射）。
- **会话内二次准入**（`trainer_session.py:477-481`）：dispatch 后立即 `require_moss_trainer_admission`（defense in depth），随后 `validate_pp_process_group`（`accelerator.py:256` 校验首 init 必为 `cpu:gloo,cuda:nccl` 多后端组，`accelerator.py:253` 一次性 init）。
- **DDP 路径守卫全部在位**（`gate.py` 实读）：`require_trainer_topology`（:252）、`require_accelerate_checkpoint`（:272）、`require_pipeline_runtime` 全部保留且 `__all__`（:320-323）未删；bootstrap/loop 的 DDP-interior 守卫继续拒绝任何漏进来的 PP run（gate.py:205/231 注释与实现一致）。**没有任何守卫被删除或 no-op。**

## 3. 放行范围与诚实拒绝（admission 实读）

`require_moss_trainer_admission`（`trainer_admission.py:59-171`）逐项 `_reject`：grad_accum≠1、训练 preview、非 cache latents/text、loss≠l2、weighting≠none、mixed_precision≠fp16、stage_schedule、adaptive_oom_retry、epoch 制保存、persistent loader workers、torch_compile——每条拒绝带具名 knob 与原因。范围恰为冻结契约的 MOSS PP=2/1F1B/FP16/plainLoRA/完整缓存，**未拓展未验收族**；Anima 等族维持历史 `pp_trainer_path_unavailable`。

## 4. 数据 / 噪声 / 步进语义（trainer_inputs + trainer_session 实读）

- **同批次对账**：`tensor_digest`（trainer_inputs.py:102-109，SHA-256 over 原始字节，dtype-exact）；`_refuse_mismatched_ranks`（:133-148）双 rank digest 不一致即命名拒绝（防 DDP 式分片静默半数据训练）。
- **prep 失败协商**：`_negotiate_prep`（:116-130）在任何 collective 之前 all_gather 排错，rank 局部失败不会 stranded peer。
- **噪声同步**：rank0 `sample_moss_flow`（抽取自 `library/models.moss_image.family`，MOSS 数学未复制）后 broadcast noise/sigmas（:187-203）；last stage 无组即拒绝——无 RNG 顺序耦合。
- **§4 步进语义**（`pp_train_step`，trainer_session.py:210-235）：global_step 计消耗步（skip 计入）、effective_step 仅成功步递增、scheduler 仅在成功步 `step()`（LR 除数=1：`get_scheduler_fix(args, optimizer, 1)`，:499）。与 t11r9 热证据（global4/eff2/skipped2、LR 恒 0.001 至成功步推进）逐项吻合。
- **loader 契约**：`build_pp_train_dataloader`（:107-133）不分片 + 专用 generator 钉死 `(seed, epoch)`；`run_pp_trainer_session:528` 每 epoch `manual_seed(seed+epoch)`——shuffle 序与 resume 无关 RNG 状态，t11r9 resumed 窗口批次摘要全等实证。

## 5. checkpoint / 恢复语义（trainer_resume + checkpoint 集成实读）

- **读→校验→协商→应用的纯序**（`trainer_resume.py`）：`validate_trainer_payload`（:168-207）为纯校验（format/version/rank/run_id/family/step 五道，逐条具名 code）；`load_trainer_resume`（:210-251）先 load+validate、`_gather_errors`（:81-98）all-rank 零错协商后**才**进入 apply 相位（:242-250 scheduler/RNG torch-cpu+cuda/python/numpy 四源恢复）——**校验前零 mutation，不可能部分恢复后继续**。
- **未知格式不静默读取**：format/version 不符即 `CODE_CHECKPOINT_STATE_MISMATCH`；缺 trainer-state 的旧 PP checkpoint 默认命名拒绝（`require_exact_resume_or_opt_in`，:254-278），weights-only 需显式 opt-in（TOML 键或 `MOSS_PP_RESUME_WEIGHTS_ONLY=1`）并 warning 记录为非 exact——契约 §4 兼容规则落地。
- **identity 与协商**：run_id = `sha256(family|seed|output_name)` 全 64-hex（`derive_run_id`，:136-153），与 v3-①/t7 核实一致；`run_identity` 返回 sha256 content digest 供 `validate_identity`。保存双 rank 各写 `trainer_state_rank{r}.pt`，rank 局部失败经 `_gather_errors` 联合失败（不留半恢复包）。
- **保存产物可用且不覆盖**（`save_pp_bundle`，:242-281）：每步新 root `pp_runs/<run_id>-step<N>`；同 step 周期+结束双保存幂等跳过（marker 判定，v3-③）；`LATEST.json` 指针仅 main process 写；`accelerator.save_state/load_state` 全程未调用（grep 实测仅 `load_state_dict` 模块级用法），`pp_accelerate_checkpoint_unsupported` 守卫保持武装。
- **普通 LoRA 导出**（`save_final_lora`，:334-364）：all_gather 各 stage adapter state 到 rank0 网络重名冲突即拒绝（:324-330），经既有 `CheckpointSaver.save_final` 写普通格式——最终产物可用且走共享保存骨架，无第二保存格式。

## 6. 架构复用（非复制、无新框架）

四处新模块合计 1248 行，全部薄编排：复用 `batch_step._prepare_latents`、`family._prepare_text_conditions`、`models.moss_image.family.sample_moss_flow`、`optimizers.get_optimizer`（stage-local optimizer 经 optimizer_factory 注入 moss runtime）、`schedulers.get_scheduler_fix`、`metadata.*`、`checkpoints.CheckpointSaver`、`pp.checkpoint.save/load_distributed_checkpoint`。旧热点 `train_session.py` 仅 +20 行分叉块（:506-540 注释块+handoff），无新 registry/framework/大一统对象。840 行 session 与 1853 行 loop 未复制（文档声明与行数事实一致）。

## 7. 独立证据复核（不重复 GPU 节点，绑定原始证据）

- **CPU**：t11 第 9 轮四条 verify（19/127/79 passed、collect 1、diffcheck 干净）在本审阅前一日实测（报告 §3），本轮绑定 hot SHA 与 library SHA 均与该轮一致——即当前字节就是通过字节，无内容漂移。
- **GPU**：`docs/findings/moss_pp_trainer_wiring_validation_20261114.md` §6g + `hot_t11r9_run.log` + `t11r9_phase_receipts/`（双 rank 回执 + 显存回执，rc=0，33.35s）：straight 4 步真实更新（global4/eff2/skipped2、双重 backoff）、新进程 resume_step=2、step3 真实 skip、param digest/LR/batch digests 跨 phase 双 rank 全等、显存 22.0/21.0MB ≪2GiB。gate JSON（gate_t11r9.json）如实记录 vLLM 驻留与用户授权。预算（timeout60/内部45s/每卡2GiB/双固定 UUID/禁1050）未降低。
- **不越界结论**：tiny 4 步热测只证明接线/数值/恢复/资源门正确，不证明训练质量或可外推规模——报告 §8 已如实标注。

## 8. 审阅发现（全部非阻塞，无返修要求）

| id | severity | 位置 | 问题 | 处置 |
| --- | --- | --- | --- | --- |
| R1 | low | `trainer_session.py:141-149` | `derive_run_id` docstring 写 "binding seed / run name / step budget"，但 material 实为 `family|seed|output_name`（无 step budget）——**代码符合冻结契约 v3-①（t7 已核实）**，仅注释措辞漂移 | 非必须；后续修注释即可，不改行为 |
| R2 | low | `trainer_session.py:447-449` | `_build_metadata` 的 `total_batch_size` 传 `train_batch_size` 不乘 stage 数，注释已说明 PP 一个 pipeline step 消费一整批（world=stage 数非数据并行复制）——与 DDP metadata 语义不同但属有意且已文档化 | 保持；建议 v3+ 需求文档在 §7.3 附一句以绝歧义 |
| R3 | low | `trainer_session.py:251-258` | 同 step 幂等跳过仅凭 marker 文件；若另一 run 恰有相同 run_id（同 seed/output_name）与 output_dir，其同 step 保存会静默复用旧包。与冻结 v3-③（identity 不含 output_dir、新 root 拒覆盖）一致，属已文档化的语义选择 | 保持；如需更强隔离属需求变更，不属本审阅 |

三项均为 low / 观察级，不构成 needs_revision 依据。

## 9. 判定

**PASS**。MOSS 主训练 PP 接线在正确数值（§4 语义逐项吻合热证据）、诚实恢复（校验先于应用、无静默降级、不覆盖产物）、安全边界（守卫全在、有限放行、不拓展族）与可维护性（薄模块、全复用、无新框架）条件下可交付；CPU + 双 V100 主入口 tiny 原始证据完整且绑定当前字节。
