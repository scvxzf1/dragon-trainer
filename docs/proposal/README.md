# Proposal 索引

状态：索引
适用版本：当前文档树

这里只保留活跃、半活跃或刚完成待归档的提案。已经完成、过期或只服务旧合并工作的文档放到 [_archive/docs/proposal/](../../_archive/docs/proposal/)。

## 活跃或半活跃提案

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [block_swap_macro_pipeline_plan.md](block_swap_macro_pipeline_plan.md) | 活跃 / P0' 完成、P1 已实现待多分辨率验收 | Block Swap 宏观流水线 P0'–P3 开发计划；第 1 轮修正折返点前提、第 2 轮确认链路饱和、第 3 轮落地进程内水位调优 |
| [training_help_refactor_plan_20261005.md](training_help_refactor_plan_20261005.md) | 活跃 / 待实施 | 训练字段帮助的内容契约、建议优先展示、默认恢复与相关导航，P0–P5 分阶段迁移和验收 |
| [qwen_image_2_1_90hx_optimization_20260927.md](qwen_image_2_1_90hx_optimization_20260927.md) | 实验 / 阶段 3 开发中 | INT8 首轮热测、32-head 与 GPU wait 证据修正、PCIe 传输预算及结构性优化排序 |
| [qwen_image_2_1_training_block_stage3_20260927.md](qwen_image_2_1_training_block_stage3_20260927.md) | 开发中 | 冻结 MLP/QKV 融合、独立 LoRA 梯度与保存契约、激活预算及分段重算验收 |
| [dragon_next_training_config_plan.md](dragon_next_training_config_plan.md) | 活跃 / 待实施 | Next 训练配置的字段对账、选择规则、排列、补充入口与 T0–T5 分阶段验收计划 |
| [adaptive_training_roadmap.md](adaptive_training_roadmap.md) | 活跃 / P1、P2 推进中 | FP16/FP32 实际训练、数据游标、OOM 恢复、自动精度选择与双卡多模型分阶段验收计划 |
| [adaptive_precision_delivery_reset_20261001.md](adaptive_precision_delivery_reset_20261001.md) | 实验交付计划 / 批次状态见正文 | Anima 自动精度交付收敛、严格数值门槛与质量证据边界；不改变生产默认 |
| [dev_push_audit_plan_20260927.md](dev_push_audit_plan_20260927.md) | 待执行 / 2026-09-27 基线 | 待推送变更的范围隔离、审计、候选放行与 `dev` 推送计划；执行前重新核对基线 |
| [dragon_config_progressive_disclosure.md](dragon_config_progressive_disclosure.md) | 探索 / 未实现 | Dragon 训练配置由静态等级与双重分类迁移到模型族、Adapter、功能开关驱动的渐进披露方案 |
| [auto_vram_v1.md](auto_vram_v1.md) | 提案 / 未实现 | AutoVram v1 自动显存档位搜索与可审计协议 |
| [dcgen_anima_f32c32.md](dcgen_anima_f32c32.md) | POC / 阶段 1-2 | DC-Gen Anima f32c32 latent space、双缓存和 patch 对齐探针记录 |
| [krea2_nf4_fp16_layer_scaling.md](krea2_nf4_fp16_layer_scaling.md) | 活跃 / 原型已验证、生产未落地 | 把 fp16 缩放从全局 loss 标量下沉到每个 NF4 Linear 的 backward（frexp/ldexp 2 的整数次幂平移）。真实权重实测 LoRA 梯度与 bf16 持平（40.97% vs 40.61%）、单步 9.0s→3.1s（2.9×）、22 处契约 0 改动、免 GradScaler；**只提速不提精度，仅对 sm70/sm75（无原生 bf16 tensor core）有价值** |
| [krea2_pipeline_parallel.md](krea2_pipeline_parallel.md) | 实验 / 阶段 1 | Anima、Krea-2、Z-Image 双卡 PP 配置、family stage adapter 与 fail-closed 规划；1F1B 主训练 runtime 尚未接线 |
| [multi_gpu_training_plan.md](multi_gpu_training_plan.md) | 实验 / 研究分支 `codex/multi-gpu-training`（**2026-11-08 已归档**，提交并入 `dev` `7a6b13c4`） | 多卡（DDP/PP/TP）能力盘点、2×V100 NVLink 硬件约束与 S0–S5 阶段计划；不改生产默认。当前逐点状态以[多卡参考点集成验收](../findings/multigpu_reference_validation_20261107.md)为准 |
| [multigpu_reference_validation_contract.md](multigpu_reference_validation_contract.md) | 实验 / 验收契约（v1 冻结 `80dfdb02…`；**已被 [修订 v2](multigpu_reference_validation_contract_amendment_v2.md) 部分取代**——§13 S-d/S-g 两条数值门解除，其余条款继续有效；旧文一字未删，§19 为追加修订记录） | P0-0..5、P1-1..3、EXP-FSDP2-BASE/EXP-TP-COLROW 的验证目标/证据级别/采用否决条件/责任阶段；环境与资产核对、磁盘与 checkpoint 体积预算、50 次有效更新的预登记门槛和六个新测试接口。逐点裁定、取数顺序三层缺陷与口径更正见[集成验收报告](../findings/multigpu_reference_validation_20261107.md)（§12–§13） |
| [multigpu_reference_validation_contract_amendment_v2.md](multigpu_reference_validation_contract_amendment_v2.md) | **生效 / 修订 v2（2026-11-08，用户批准）** | 解除数学不可达的 S-d「10 步后权重 ≤1e-6」与 S-g「逐位」两条判据，换为 C1 数据对齐门 / C2 步级梯度 ULP 门 / C3 权重漂移 ≤ 实测噪声地板 / C4 恢复两段；含 §17 G-2 修订记录（改了什么/为什么/原判据实测结果/谁批准）与不可达论证 |
| [adapter-aware-checkpoint.md](adapter-aware-checkpoint.md) | 半活跃 | Adapter-aware activation checkpoint 可行性探索 |
| [turbo_anima_dmd_lora.md](turbo_anima_dmd_lora.md) | 半活跃 | Turbo Anima / DMD LoRA 蒸馏提案 |
| [prior_preservation_from_synth_pool.md](prior_preservation_from_synth_pool.md) | 半活跃 | synth pool prior preservation 提案 |
| [postfix_residual_for_directedit.md](postfix_residual_for_directedit.md) | 半活跃 | DirectEdit image-conditional postfix residual 提案 |
| [postfix_residual_per_image_inversion.md](postfix_residual_per_image_inversion.md) | 半活跃 | per-image postfix tail inversion probe 提案 |
| [soft_tokens_agsm.md](soft_tokens_agsm.md) | 半活跃 | Soft Tokens AGSM 提案 |
| [soft_tokens_contrastive.md](soft_tokens_contrastive.md) | 兼容入口 | Soft Tokens contrastive 方向记录 |
| [soft_tokens_softrank.md](soft_tokens_softrank.md) | 兼容入口 | Soft Tokens soft-rank 方向记录 |
| [personalization-region-curriculum.md](personalization-region-curriculum.md) | 半活跃 | 区域→整图课程、先验保持与 APT 风格自适应正则化的可行性及实施计划 |
| [convrot_w8a_training_plan.md](convrot_w8a_training_plan.md) | 提案 / 核心已实现（实验默认关闭） | ConvRot 战略 C：W8A16→W8A8 训练路径规格 + 落地状态 |
| [convrot_w8a_optimization_roadmap.md](convrot_w8a_optimization_roadmap.md) | 提案 / 半活跃（P0-A/A2/C/D 完成） | ConvRot 优化路线图；regular Hadamard opt-in；默认仍 sylvester；Triton 未达门槛 |
| [convrot_90hx_development_validation_plan_20261002.md](convrot_90hx_development_validation_plan_20261002.md) | 筛选完成 / S5条件跳过 | CMP 90HX 基线、Triton优化、反向精度与生产功能验证；无合格候选，500步未执行 |
| [convrot_90hx_followup_development_plan_20261002.md](convrot_90hx_followup_development_plan_20261002.md) | P0修复与短测完成 / 后续待执行 | 采样graph与offloader载荷生命周期修复已复测；完整18样本校准和性能复测待推进，500步按用户要求暂不进行 |

## 已实施与历史背景

| 文档 | 状态 | 说明 |
| --- | --- | --- |

## 已归档记录

| 文档 | 归档原因 |
| --- | --- |
| [legacy_ui_retirement_plan.md](../../_archive/docs/proposal/legacy_ui_retirement_plan.md) | P0–P5 和 2026-09-30 生产只读验收已完成，保留迁移及功能取舍证据 |
| [krea2_raw_gradient_checkpointing.md](../../_archive/docs/proposal/krea2_raw_gradient_checkpointing.md) | 保留 2026-08-08 调查与 2026-08-09 落地核对，原实现假设不再作为当前能力说明 |

| [dragon_frontend_rebuild_plan.md](../../_archive/docs/proposal/dragon_frontend_rebuild_plan.md) | 历史重建计划；Next 已是唯一入口，原阶段决策保留 |
| [krea2_nf4_blockswap.md](../../_archive/docs/proposal/krea2_nf4_blockswap.md) | 方向 A 端到端验证已落地；结论保留 |

| [krea2_raw_migration.md](../../_archive/docs/proposal/krea2_raw_migration.md) | Anima → Krea-2-Raw 历史迁移计划；阶段 0-6 已落地，状态块标记「待归档」 |
| [krea2_raw_migration_notes.md](../../_archive/docs/proposal/krea2_raw_migration_notes.md) | 配套历史风险登记册；核心迁移已落地 |

## 归档规则

- 已经落地并有正式说明的提案，移到 `_archive/docs/proposal/`。
- 服务一次性合并、审计或旧路线图的文档，移到 `_archive/docs/proposal/`。
- 算法仍可能继续推进、但尚未成为稳定方法的文档，继续留在本目录。
- 归档时同步更新 [../archive-index.md](../archive-index.md) 和 [_archive/docs/proposal/README.md](../../_archive/docs/proposal/README.md)。

已完成的 `anima-app` legacy bridge 收尾记录已归档到
[anima-app-legacy-bridge-cleanup.md](../../_archive/docs/proposal/anima-app-legacy-bridge-cleanup.md)。
