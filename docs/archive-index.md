# 文档归档索引

状态：索引（历史归档主目录）

这里说明哪些文档已经从主文档路径退下去，以及为什么退下去。

## 归档原则

- 归档不是删除，历史上下文仍保留。
- 归档文档默认不是当前实现说明。
- 当前实现以 `docs/README.md` 指向的 `guidelines/`、`methods/`、`experimental/`、`structure/`、`configuration/`、`features/`、`findings/` 和 `optimizations/` 为准。

## Proposal 归档

历史提案和一次性合并材料放在 [_archive/docs/proposal/README.md](../_archive/docs/proposal/README.md)。

| 文档 | 原因 |
| --- | --- |
| [proposal/README.md](../_archive/docs/proposal/README.md) | 历史提案分区归档索引 |
| [adaptive_precision_delivery_reset_20261001.md](../_archive/docs/proposal/adaptive_precision_delivery_reset_20261001.md) | 2026-10-01 自动精度交付收敛批次计划已执行完毕，结论沉淀于 `docs/findings/` 与 `adaptive-runtime.md` |
| [anima_pp_trainer_wiring.md](../_archive/docs/proposal/anima_pp_trainer_wiring.md) | Anima 主训练 PP 接线 t1 冻结契约已完成落地与验收，当前说明见 [anima_pp_trainer_wiring.md](experimental/anima_pp_trainer_wiring.md) |
| [convrot_90hx_development_validation_plan_20261002.md](../_archive/docs/proposal/convrot_90hx_development_validation_plan_20261002.md) | CMP 90HX 六阶段筛选计划已收尾（S5 按门控跳过），结果沉淀于 [convrot_90hx_validation_20261002.md](findings/convrot_90hx_validation_20261002.md) |
| [convrot_w8a_training_plan.md](../_archive/docs/proposal/convrot_w8a_training_plan.md) | ConvRot W8A16/W8A8 核心训练路径已落地，当前说明见 [convrot_int8_training.md](experimental/convrot_int8_training.md) |
| [dev_push_audit_plan_20260927.md](../_archive/docs/proposal/dev_push_audit_plan_20260927.md) | 2026-09-27 一次性待推送变更隔离与审计计划，基线已过时 |
| [dpdmd.md](../_archive/docs/proposal/dpdmd.md) | DP-DMD 迁移提案已落地，当前说明见 [experimental/dpdmd.md](experimental/dpdmd.md) 与 [structure/dpdmd.md](structure/dpdmd.md) |
| [dragon_config_progressive_disclosure.md](../_archive/docs/proposal/dragon_config_progressive_disclosure.md) | 面向已退役旧静态 Dragon UI 的渐进披露提案，原则已由 Next 训练配置计划承接 |
| [krea2_pipeline_parallel.md](../_archive/docs/proposal/krea2_pipeline_parallel.md) | 早期双卡 PP 阶段规划；Anima / MOSS / Krea-2 三族 1F1B 专用训练会话现已上线，见 [distributed-training-card.md](experimental/distributed-training-card.md) |
| [moss_image_pipeline_parallel_v0.md](../_archive/docs/proposal/moss_image_pipeline_parallel_v0.md) | MOSS-Image 库级 PP v0 冻结需求契约已落地并独立验收，当前说明见 [moss_image_pipeline_parallel_v0.md](experimental/moss_image_pipeline_parallel_v0.md) |
| [moss_image_pp_trainer_wiring.md](../_archive/docs/proposal/moss_image_pp_trainer_wiring.md) | MOSS-Image 主训练 PP 接线冻结契约（保留原始字节指纹），当前说明见 [moss_image_pp_trainer_wiring.md](experimental/moss_image_pp_trainer_wiring.md) |
| [multi_gpu_training_plan.md](../_archive/docs/proposal/multi_gpu_training_plan.md) | `codex/multi-gpu-training` 研究分支已并入 `dev` 收口，逐点验收见 [multigpu_reference_validation_20261107.md](findings/multigpu_reference_validation_20261107.md) |
| [optimization-roadmap.md](../_archive/docs/proposal/optimization-roadmap.md) | 早期优化路线图已被 [optimization-configs-current.md](optimization-configs-current.md) 取代 |
| [qwen_image_2_1_training_block_stage3_20260927.md](../_archive/docs/proposal/qwen_image_2_1_training_block_stage3_20260927.md) | Qwen 2.1 阶段 3 冻结投影融合开发与验收已完成，当前入口见 [qwen_image_2_1_fused_projections.md](experimental/qwen_image_2_1_fused_projections.md) |
| [separation_plan.md](../_archive/docs/proposal/separation_plan.md) | `anima_lora` 可安装包门面与 CLI 分离方案已落地 |
| [soft_tokens_contrastive.md](../_archive/docs/proposal/soft_tokens_contrastive.md) | Soft Tokens B=1 InfoNCE 早期提案已并入 [soft_tokens.md](experimental/soft_tokens.md) |
| [soft_tokens_softrank.md](../_archive/docs/proposal/soft_tokens_softrank.md) | Soft Tokens soft-rank 早期提案已落地并并入 [soft_tokens.md](experimental/soft_tokens.md) |
| [turbo_anima_dmd_lora.md](../_archive/docs/proposal/turbo_anima_dmd_lora.md) | CA-decoupled DMD2 旧蒸馏提案已被 DP-DMD 方案取代 |
| [legacy_ui_retirement_plan.md](../_archive/docs/proposal/legacy_ui_retirement_plan.md) | Classic / Dragon 旧 UI 退役已完成；保留 2026-09-28 至 2026-09-30 的迁移、取舍与验收记录 |
| [krea2_raw_gradient_checkpointing.md](../_archive/docs/proposal/krea2_raw_gradient_checkpointing.md) | Krea-2 检查点已落地；保留早期调查、设计与实现分叉的历史核对 |
| [webui-instrument-panel-reskin/README.md](../_archive/docs/proposal/webui-instrument-panel-reskin/README.md) | 2026-07-12 旧静态 UI 换皮已完成；设计、计划、迭代日志和最终评审成组归档 |
| [anima-app-deglobalization.md](../_archive/docs/proposal/anima-app-deglobalization.md) | 阶段 0-3 已落地（85 轮，`globalThis` 从 1075 降到 270），剩余收尾由 `anima-app-legacy-bridge-cleanup.md` 接续 |
| [dragon_frontend_rebuild_plan.md](../_archive/docs/proposal/dragon_frontend_rebuild_plan.md) | Next 已成为唯一入口；保留 FocusFlow / Zaptix 设计与原阶段决策 |
| [krea2_nf4_blockswap.md](../_archive/docs/proposal/krea2_nf4_blockswap.md) | 方向 A 端到端验证已落地；后续方向见 findings 的 NF4 消融结论 |
| [krea2_raw_migration.md](../_archive/docs/proposal/krea2_raw_migration.md) | 历史迁移计划；阶段 0-6 已落地，状态块原标记「待归档」 |
| [krea2_raw_migration_notes.md](../_archive/docs/proposal/krea2_raw_migration_notes.md) | 配套历史风险登记册 |
| [krea2_nf4_fp16_release_gates.md](../_archive/docs/proposal/krea2_nf4_fp16_release_gates.md) | 提案被实测否决：k 自收敛误差 256² 0.123% → 1024² 5.900% 单调上升，k≤15 上限使窗口在 >512² 关闭；结论与证据见 [findings](findings/krea2_nf4_fp16_grad_underflow_20261108.md) |
| [anima-app-legacy-bridge-cleanup.md](../_archive/docs/proposal/anima-app-legacy-bridge-cleanup.md) | `anima-app` legacy bridge 收尾已完成，仅保留实施上下文 |
| [lora-network-decomposition.md](../_archive/docs/proposal/lora-network-decomposition.md) | `LoRANetwork` 分层拆分核心代码已落地，当前仅保留实施计划和历史上下文 |
| [upstream_high_value_merge_roadmap_2026-06-24.md](../_archive/docs/proposal/upstream_high_value_merge_roadmap_2026-06-24.md) | 上游合并路线图已执行，当前仅保留历史上下文 |
| [upstream_core_merge_plan_2026-07-02.md](../_archive/docs/proposal/upstream_core_merge_plan_2026-07-02.md) | 一次性上游增量合并计划已执行完毕，从根目录移入归档 |
| [configs_external_task_completed_2026-07.md](../_archive/docs/proposal/configs_external_task_completed_2026-07.md) | 配置目录外置完成报告，当前说明见 [外置配置目录](configuration/external-configs.md) |
| [upstream_merge_completion_report_2026-06-24.md](../_archive/docs/proposal/upstream_merge_completion_report_2026-06-24.md) | 上游合并完成报告，已经不是活跃提案 |
| [upstream_merge_completion_report_2026-06-24_audit.md](../_archive/docs/proposal/upstream_merge_completion_report_2026-06-24_audit.md) | 上游合并完成报告审核记录 |
| [upstream_merge_completion_report_fixes_summary.md](../_archive/docs/proposal/upstream_merge_completion_report_fixes_summary.md) | 上游合并报告修复摘要 |
| [upstream_preprocess_robustness_analysis.md](../_archive/docs/proposal/upstream_preprocess_robustness_analysis.md) | 上游预处理健壮性分析，服务旧合并工作 |
| [configs_external_data_root_plan_2026-06-24.md](../_archive/docs/proposal/configs_external_data_root_plan_2026-06-24.md) | 配置外置已落地，当前说明由 `docs/configuration/` 承接 |
| [compile_safety_patches_analysis.md](../_archive/docs/proposal/compile_safety_patches_analysis.md) | Compile safety 分析已转为实现和测试事实 |
| [anima-app-runtime-migration.md](../_archive/docs/proposal/anima-app-runtime-migration.md) | WebUI `anima-app` runtime 迁移阶段计划已执行，当前仅保留历史上下文 |

## Findings 归档

历史实验结论与当前使用说明分开保存：

| 文档 | 说明 |
| --- | --- |
| [findings/README.md](../_archive/docs/findings/README.md) | 历史 findings 归档索引 |
| [dcw-bias-findings-202605.md](../_archive/docs/findings/dcw-bias-findings-202605.md) | DCW CFG=1 和早期 band-mask 实验结论 |
| [smc-cfg-analysis-and-proposal-202605.md](../_archive/docs/findings/smc-cfg-analysis-and-proposal-202605.md) | SMC-CFG α-adaptive 控制器的历史分析 |
| [vr-loss-headroom/README.md](../_archive/docs/findings/vr-loss-headroom/README.md) | Variance-reduction loss headroom 历史实验结论 |
| [vr-loss-headroom/proposal.md](../_archive/docs/findings/vr-loss-headroom/proposal.md) | Variance-reduction loss 历史接入提案 |
| [project_cleanup_long_running_goal_20260705.md](../_archive/docs/findings/project_cleanup_long_running_goal_20260705.md) | 长程清理目标（2026-07-05）；状态块自述已完成归档 |
| [project_cleanup_next_stage_goal_20260705.md](../_archive/docs/findings/project_cleanup_next_stage_goal_20260705.md) | 下一阶段清理目标（2026-07-05）；状态块自述已完成归档 |
| [project_cleanup_sustained_goal_20260705.md](../_archive/docs/findings/project_cleanup_sustained_goal_20260705.md) | 持续清理目标（2026-07-05）；状态块自述已完成归档 |
| [anima_balanced_16g_blockswap_ablation_plan.md](../_archive/docs/findings/anima_balanced_16g_blockswap_ablation_plan.md) | 2026-06-03 阶段快照；Balanced 16G 预测式块交换消融的完成态记录 |
| [anima_fp8_blockswap_transfer_ablation_plan.md](../_archive/docs/findings/anima_fp8_blockswap_transfer_ablation_plan.md) | 消融目标与验收口径已并入 transfer 报告，旧计划转历史记录 |
| [anima_lokr_16g_next_goal.md](../_archive/docs/findings/anima_lokr_16g_next_goal.md) | 目标已由 allocator/blocks/FP8 追加消融收口 |
| [fasterdit_signal_densification_plan.md](../_archive/docs/findings/fasterdit_signal_densification_plan.md) | 2026-06-05 阶段快照；信号加密方案与验证记录 |
| [krea2_3080_speed_stage1..12.md](../_archive/docs/findings/) | Krea-2 速度优化 12 阶阶段原文；判定与数字汇总见 [stage_summary](findings/krea2_3080_speed_stage_summary.md) |

## Superpowers 归档

2026-07 至 2026-09 维护期的设计规格（specs）、执行计划（plans）和五轮自动迭代协议整体归档；
不是当前操作说明，当前事实以 `docs/methods/`、`docs/features/`、`docs/findings/` 和实时源码为准。

| 文档 | 说明 |
| --- | --- |
| [superpowers/README.md](../_archive/docs/superpowers/README.md) | 归档分区索引（含 specs / plans 生命周期说明） |
| [superpowers/plans/](../_archive/docs/superpowers/plans/) | 16 篇 2026-07~09 一次性执行计划与迭代日志 |
| [superpowers/specs/](../_archive/docs/superpowers/specs/) | 16 篇配套设计规格与架构决策记录 |

## Configuration 归档

| 文档 | 说明 |
| --- | --- |
| [configuration/README.md](../_archive/docs/configuration/README.md) | 历史配置快照与实施报告索引 |
| [dragon-training-config-215.md](../_archive/docs/configuration/dragon-training-config-215.md) | 已退役静态 Dragon 训练器 215 项配置快照 |
| [moss-image-training-preflight.md](../_archive/docs/configuration/moss-image-training-preflight.md) | MOSS-Image 早期训练前核对快照（后续重测纠偏见 `docs/findings/moss_historical_measurements_audit_20261020.md` 与 `docs/configuration/moss-image.md`） |
| [implementation-report.md](../_archive/docs/configuration/implementation-report.md) | 2026-06-24 配置根外置的本机实施快照，已脱敏 |

## Features 归档

| 文档 | 说明 |
| --- | --- |
| [features/README.md](../_archive/docs/features/README.md) | WebUI 历史界面与阶段施工记录索引 |
| [dragon-ui.md](../_archive/docs/features/dragon-ui.md) | 已退役静态 Dragon / classic 界面说明与路由重定向背景 |
| [dragon-ui-functional-map.md](../_archive/docs/features/dragon-ui-functional-map.md) | 旧静态 Dragon 八个界面的历史功能盘点图谱 |
| [dragon-next-implementation.md](../_archive/docs/features/dragon-next-implementation.md) | 2026-09 Next React 工作台首轮重建实施与验收对账记录 |
| [dragon-next-uiux-progress.md](../_archive/docs/features/dragon-next-uiux-progress.md) | 2026-09 Next UI/UX 阶段性施工修复流水记录 |

## Guidelines 归档

| 文档 | 说明 |
| --- | --- |
| [guidelines/README.md](../_archive/docs/guidelines/README.md) | 早期单模型与已移除桌面 PySide6 GUI 综合指南归档索引 |
| [指南书.md](../_archive/docs/guidelines/指南书.md) | 中文早期桌面 GUI 综合指南 |
| [guidebook.md](../_archive/docs/guidelines/guidebook.md) | 英文早期桌面 GUI 综合指南 |
| [ガイドブック.md](../_archive/docs/guidelines/ガイドブック.md) | 日文早期桌面 GUI 综合指南 |
| [가이드북.md](../_archive/docs/guidelines/가이드북.md) | 韩文早期桌面 GUI 综合指南 |

## Methods 归档

| 文档 | 说明 |
| --- | --- |
| [methods/README.md](../_archive/docs/methods/README.md) | 已退役或已被替代的方法说明归档索引 |
| [invert.md](../_archive/docs/methods/invert.md) | 早期 prefix/postfix embedding inversion 历史说明（现仅保留实验探针 `exp-invert-directedit`） |
