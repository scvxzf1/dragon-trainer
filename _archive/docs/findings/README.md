# Archived Findings

状态：已归档

这里保存已经退出 live `bench/` / `docs/` 路径、但仍被当前方法文档引用作为结论来源的历史分析。
这些材料不是当前操作说明；可运行入口和默认值以 `docs/methods/`、`tasks.py --help` 和实时源码为准。

| 文档 | 说明 |
| --- | --- |
| [dcw-bias-findings-202605.md](dcw-bias-findings-202605.md) | DCW CFG=1 及早期 band-mask 实验结论；当前 DCW 说明已吸收后续 CFG × aspect 修订 |
| [smc-cfg-analysis-and-proposal-202605.md](smc-cfg-analysis-and-proposal-202605.md) | SMC-CFG 从固定增益改为 α-adaptive 控制器的历史分析 |
| [vr-loss-headroom/README.md](vr-loss-headroom/README.md) | Variance-reduction loss headroom 实验与可行性结论 |
| [vr-loss-headroom/proposal.md](vr-loss-headroom/proposal.md) | Variance-reduction loss 历史接入提案 |
| [anima_balanced_16g_blockswap_ablation_plan.md](anima_balanced_16g_blockswap_ablation_plan.md) | Anima Balanced 16G 预测式块交换阶段快照（2026-06-03）；不作为当前操作说明 |
| [anima_fp8_blockswap_transfer_ablation_plan.md](anima_fp8_blockswap_transfer_ablation_plan.md) | FP8 block swap transfer 消融计划；实验目标与验收口径已合并进 findings 的 transfer 报告 |
| [anima_lokr_16g_next_goal.md](anima_lokr_16g_next_goal.md) | LoKr 16G 下一步目标；活跃目标已由 allocator / blocks / FP8 追加消融收口 |
| [fasterdit_signal_densification_plan.md](fasterdit_signal_densification_plan.md) | FasterDiT-style 收敛信号加密方案与验证记录（2026-06-05）阶段快照 |
| [project_cleanup_long_running_goal_20260705.md](project_cleanup_long_running_goal_20260705.md) | 长程清理目标（2026-07-05）；状态块自述已完成归档 |
| [project_cleanup_next_stage_goal_20260705.md](project_cleanup_next_stage_goal_20260705.md) | 下一阶段清理目标（2026-07-05）；状态块自述已完成归档 |
| [project_cleanup_sustained_goal_20260705.md](project_cleanup_sustained_goal_20260705.md) | 持续清理目标（2026-07-05）；状态块自述已完成归档 |
| [krea2_3080_speed_stage1.md](krea2_3080_speed_stage1.md) | 阶段 1：瓶颈定位与两个候选消融；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage2.md](krea2_3080_speed_stage2.md) | 阶段 2：选择性梯度检查点；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage3.md](krea2_3080_speed_stage3.md) | 阶段 3：per-block compile；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage4.md](krea2_3080_speed_stage4.md) | 阶段 4：compile × selective 叠加；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage5.md](krea2_3080_speed_stage5.md) | 阶段 5：3080 20 步长窗口与热稳态；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage6.md](krea2_3080_speed_stage6.md) | 阶段 6：FP16 compute（REJECT）；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage7.md](krea2_3080_speed_stage7.md) | 阶段 7：Inductor mode（REJECT）；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage8.md](krea2_3080_speed_stage8.md) | 阶段 8：LoRA rank（NOT_WORTH）；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage9.md](krea2_3080_speed_stage9.md) | 阶段 9：multi-bucket compile 图复用；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage10.md](krea2_3080_speed_stage10.md) | 阶段 10：compile × 完整检查点；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage11.md](krea2_3080_speed_stage11.md) | 阶段 11：算子 profile 与天花板；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
| [krea2_3080_speed_stage12.md](krea2_3080_speed_stage12.md) | 阶段 12：FlashAttention varlen；汇总见 `docs/findings/krea2_3080_speed_stage_summary.md` |
