# Archived Documentation Proposals

这里放从 `docs/proposal/` 退下来的历史提案，或用于记录当前文档中提到但已经不再维护的旧设计线索。

本目录的材料默认不是当前实现说明。实现状态以 `docs/README.md` 指向的 `guidelines/`、`methods/`、`experimental/`、`structure/`、`configuration/`、`features/`、`findings/` 和 `optimizations/` 文档为准。

## 2026-11-09 归档

| 文档 | 原因 |
| --- | --- |
| [krea2_nf4_fp16_release_gates.md](krea2_nf4_fp16_release_gates.md) | 提案已被实测否决：fp16+loss-scaling 在 ≤512² 优于 bf16，但 k 自收敛误差随分辨率单调上升（256² 0.123% → 1024² 5.900%），受 fp16 标量上限限制 k≤15，窗口在 >512² 关闭。当前实现仍锁 `COMPUTE_DTYPE=bfloat16` |

## 2026-10-19 归档（根目录资料清理）

| 文档 | 原因 |
| --- | --- |
| [configs_external_task_completed_2026-07.md](configs_external_task_completed_2026-07.md) | 配置目录外置任务的完成报告，功能已落地；当前说明由 `docs/configuration/external-configs.md` 承接 |
| [upstream_core_merge_plan_2026-07-02.md](upstream_core_merge_plan_2026-07-02.md) | 一次性上游增量合并计划，执行状态已成历史，不再作为根目录活跃文档 |

## 2026-10-04 归档

| 文档 | 原因 |
| --- | --- |
| [legacy_ui_retirement_plan.md](legacy_ui_retirement_plan.md) | P0–P5 和 2026-09-30 生产只读验收已完成，保留退役过程及功能取舍 |
| [krea2_raw_gradient_checkpointing.md](krea2_raw_gradient_checkpointing.md) | 保留早期检查点调查与落地核对，当前能力见多模型说明 |
| [webui-instrument-panel-reskin/README.md](webui-instrument-panel-reskin/README.md) | 已完成的旧静态 UI 换皮设计、执行计划、迭代日志与最终评审 |

## 2026-07-07 归档（第二批）

| 文档 | 原因 |
| --- | --- |
| [anima-app-deglobalization.md](anima-app-deglobalization.md) | 阶段 0-3 已落地（85 轮，`globalThis` 从 1075 降到 270），剩余收尾工作由 `anima-app-legacy-bridge-cleanup.md` 接续 |

## 2026-07-07 归档

| 文档 | 原因 |
| --- | --- |
| [lora-network-decomposition.md](lora-network-decomposition.md) | `LoRANetwork` 分层拆分核心代码已落地，当前仅保留实施计划和历史上下文 |

## 2026-07-06 归档

| 文档 | 原因 |
| --- | --- |
| [upstream_high_value_merge_roadmap_2026-06-24.md](upstream_high_value_merge_roadmap_2026-06-24.md) | 上游合并路线图已执行 |
| [upstream_merge_completion_report_2026-06-24.md](upstream_merge_completion_report_2026-06-24.md) | 上游合并完成报告，已经不是活跃提案 |
| [upstream_merge_completion_report_2026-06-24_audit.md](upstream_merge_completion_report_2026-06-24_audit.md) | 上游合并完成报告审核记录 |
| [upstream_merge_completion_report_fixes_summary.md](upstream_merge_completion_report_fixes_summary.md) | 上游合并报告修复摘要 |
| [upstream_preprocess_robustness_analysis.md](upstream_preprocess_robustness_analysis.md) | 上游预处理健壮性分析，服务旧合并工作 |
| [configs_external_data_root_plan_2026-06-24.md](configs_external_data_root_plan_2026-06-24.md) | 配置外置已落地，当前说明由 `docs/configuration/` 承接 |
| [compile_safety_patches_analysis.md](compile_safety_patches_analysis.md) | Compile safety 分析已转为实现和测试事实 |
| [anima-app-runtime-migration.md](anima-app-runtime-migration.md) | WebUI `anima-app` runtime 迁移阶段计划已执行 |
| [dragon_frontend_rebuild_plan.md](dragon_frontend_rebuild_plan.md) | 历史重建计划；Next 已成为唯一入口 |
| [krea2_nf4_blockswap.md](krea2_nf4_blockswap.md) | NF4 × block swap 方向 A 已落地 |
| [krea2_raw_migration.md](krea2_raw_migration.md) | Anima → Krea-2-Raw 迁移计划；阶段 0-6 已落地 |
| [krea2_raw_migration_notes.md](krea2_raw_migration_notes.md) | 配套历史风险登记册 |
