# 文档归档索引

这里说明哪些文档已经从主文档路径退下去，以及为什么退下去。

## 归档原则

- 归档不是删除，历史上下文仍保留。
- 归档文档默认不是当前实现说明。
- 当前实现以 `docs/README.md` 指向的 `guidelines/`、`methods/`、`experimental/`、`structure/`、`configuration/`、`features/`、`findings/` 和 `optimizations/` 为准。

## Proposal 归档

历史提案和一次性合并材料放在 [_archive/docs/proposal/](../_archive/docs/proposal/)。

| 文档 | 原因 |
| --- | --- |
| [legacy_ui_retirement_plan.md](../_archive/docs/proposal/legacy_ui_retirement_plan.md) | Classic / Dragon 旧 UI 退役已完成；保留 2026-09-28 至 2026-09-30 的迁移、取舍与验收记录 |
| [krea2_raw_gradient_checkpointing.md](../_archive/docs/proposal/krea2_raw_gradient_checkpointing.md) | Krea-2 检查点已落地；保留早期调查、设计与实现分叉的历史核对 |
| [webui-instrument-panel-reskin/README.md](../_archive/docs/proposal/webui-instrument-panel-reskin/README.md) | 2026-07-12 旧静态 UI 换皮已完成；设计、计划、迭代日志和最终评审成组归档 |
| [anima-app-deglobalization.md](../_archive/docs/proposal/anima-app-deglobalization.md) | 阶段 0-3 已落地（85 轮，`globalThis` 从 1075 降到 270），剩余收尾由 `anima-app-legacy-bridge-cleanup.md` 接续 |
| [dragon_frontend_rebuild_plan.md](../_archive/docs/proposal/dragon_frontend_rebuild_plan.md) | Next 已成为唯一入口；保留 FocusFlow / Zaptix 设计与原阶段决策 |
| [krea2_nf4_blockswap.md](../_archive/docs/proposal/krea2_nf4_blockswap.md) | 方向 A 端到端验证已落地；后续方向见 findings 的 NF4 消融结论 |
| [krea2_raw_migration.md](../_archive/docs/proposal/krea2_raw_migration.md) | 历史迁移计划；阶段 0-6 已落地，状态块原标记「待归档」 |
| [krea2_raw_migration_notes.md](../_archive/docs/proposal/krea2_raw_migration_notes.md) | 配套历史风险登记册 |
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
| [configuration/README.md](../_archive/docs/configuration/README.md) | 历史配置实施报告索引 |
| [implementation-report.md](../_archive/docs/configuration/implementation-report.md) | 2026-06-24 配置根外置的本机实施快照，已脱敏 |
