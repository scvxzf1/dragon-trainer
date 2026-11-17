# Archived Documentation Proposals

状态：已归档

这里放从 `docs/proposal/`（及早期 `docs/`、`_archive/proposals/`）退下来的历史提案与已收口契约。

本目录的材料默认不是当前实现说明。实现状态以 [docs/README.md](../../../docs/README.md) 指向的 `guidelines/`、`methods/`、`experimental/`、`structure/`、`configuration/`、`features/`、`findings/` 和 `optimizations/` 文档为准。

## 2026-11-18 归档（提案治理与历史设计收口）

| 文档 | 原因 |
| --- | --- |
| [adaptive_precision_delivery_reset_20261001.md](adaptive_precision_delivery_reset_20261001.md) | 2026-10-01 自动精度交付收敛批次计划已执行，结论沉淀于 `docs/findings/` 与 `docs/experimental/adaptive-runtime.md` |
| [anima_pp_trainer_wiring.md](anima_pp_trainer_wiring.md) | Anima trainer 级 PP 接线 t1 冻结契约已落地并验收，现行入口见 `docs/experimental/anima_pp_trainer_wiring.md` |
| [convrot_90hx_development_validation_plan_20261002.md](convrot_90hx_development_validation_plan_20261002.md) | CMP 90HX 六阶段筛选计划已收尾（S5 条件跳过），结论见 `docs/findings/convrot_90hx_validation_20261002.md` |
| [convrot_w8a_training_plan.md](convrot_w8a_training_plan.md) | ConvRot W8A16/W8A8 核心训练路径已落地，现行说明见 `docs/experimental/convrot_int8_training.md` |
| [dev_push_audit_plan_20260927.md](dev_push_audit_plan_20260927.md) | 2026-09-27 一次性待推送变更审计计划，基线已成历史 |
| [dpdmd.md](dpdmd.md) | DP-DMD 少步蒸馏迁移提案已落地，现行说明见 `docs/experimental/dpdmd.md` |
| [dragon_config_progressive_disclosure.md](dragon_config_progressive_disclosure.md) | 旧静态 Dragon 配置渐进披露探索，原则已由 Next 训练配置计划吸收 |
| [krea2_pipeline_parallel.md](krea2_pipeline_parallel.md) | 早期双卡 PP 阶段规划；Anima / MOSS / Krea-2 三族 1F1B 专用会话现已上线 |
| [moss_image_pipeline_parallel_v0.md](moss_image_pipeline_parallel_v0.md) | MOSS-Image 库级 PP v0 需求冻结契约已落地并验收，现行说明见 `docs/experimental/moss_image_pipeline_parallel_v0.md` |
| [moss_image_pp_trainer_wiring.md](moss_image_pp_trainer_wiring.md) | MOSS-Image 主训练 PP 接线冻结契约（保留原始字节指纹），现行说明见 `docs/experimental/moss_image_pp_trainer_wiring.md` |
| [multi_gpu_training_plan.md](multi_gpu_training_plan.md) | `codex/multi-gpu-training` 分支已并入 `dev` 收口，逐点验收见 `docs/findings/multigpu_reference_validation_20261107.md` |
| [optimization-roadmap.md](optimization-roadmap.md) | 早期优化路线图已由 `docs/optimization-configs-current.md` 取代 |
| [qwen_image_2_1_training_block_stage3_20260927.md](qwen_image_2_1_training_block_stage3_20260927.md) | Qwen 2.1 阶段 3 冻结投影融合已落地并完成热测，现行说明见 `docs/experimental/qwen_image_2_1_fused_projections.md` |
| [separation_plan.md](separation_plan.md) | `anima_lora` 可安装包门面与 CLI 分离已落地 |
| [soft_tokens_contrastive.md](soft_tokens_contrastive.md) | Soft Tokens B=1 InfoNCE 早期提案已并入 `docs/experimental/soft_tokens.md` |
| [soft_tokens_softrank.md](soft_tokens_softrank.md) | Soft Tokens soft-rank 早期提案已落地并并入 `docs/experimental/soft_tokens.md` |
| [turbo_anima_dmd_lora.md](turbo_anima_dmd_lora.md) | CA-decoupled DMD2 旧蒸馏提案已被 DP-DMD 方案取代 |

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
| [anima-app-deglobalization.md](anima-app-deglobalization.md) | 阶段 0-3 已落地（85 轮，`globalThis` 从 1075 降到 270），剩余收尾工作由下行文档接续 |
| [anima-app-legacy-bridge-cleanup.md](anima-app-legacy-bridge-cleanup.md) | `anima-app` legacy bridge 收尾已完成，仅保留历史实施上下文 |

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
