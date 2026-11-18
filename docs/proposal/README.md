# Proposal 索引

状态：索引
适用版本：当前文档树

这里只保留活跃或半活跃的探索与验收提案（共 18 篇）。已经落地收口、已被替代或服务历史阶段合并的提案统一迁入历史归档区，完整归档清单见 [../archive-index.md](../archive-index.md) 与 [_archive/docs/proposal/README.md](../../_archive/docs/proposal/README.md)。

## 活跃或半活跃提案

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [block_swap_macro_pipeline_plan.md](block_swap_macro_pipeline_plan.md) | 半活跃 / P0'–P2 四族已收口 | Block Swap 宏观流水线 P0'–P3 开发计划：`slab` 恢复、`--block_swap_auto` 四族水位调优与 1 Byte 编码地板已收口；仅剩 NF8 选择性接入与 P3 后续子项待定 |
| [training_help_refactor_plan_20261005.md](training_help_refactor_plan_20261005.md) | 活跃提案 / P0–P2、P4–P5 完成，P3 交互待实施 | Next 训练配置字段帮助的内容契约、建议优先展示、默认恢复与分阶段验收计划 |
| [qwen_image_2_1_90hx_optimization_20260927.md](qwen_image_2_1_90hx_optimization_20260927.md) | 半活跃（方案 1、3 已集成，方案 2、4 未实施） | Qwen 2.1 / CMP 90HX INT8 换层热测、32-head 与 GPU wait 证据修正、PCIe 传输预算及结构性优化排序 |
| [dragon_next_training_config_plan.md](dragon_next_training_config_plan.md) | 活跃提案 / T0–T3 首批切片已落地，T4–T5 待完成 | Next 训练配置的字段对账、选择与禁用规则、四阶段布局、补充入口及来源审查计划 |
| [adaptive_training_roadmap.md](adaptive_training_roadmap.md) | 半活跃提案 / 单进程受限中途恢复已落地，自动精度搜索待定 | DiT 自适应混合精度、数据游标与 OOM 恢复阶段计划（单进程受限恢复见 `adaptive-training-recovery.md`，双卡已分流至 DDP/PP） |
| [auto_vram_v1.md](auto_vram_v1.md) | 提案 / 未实现（与已落地 `--block_swap_auto` 独立） | AutoVram v1 高向低搜索锁定与近 OOM 紧急升档的可审计规格书 |
| [dcgen_anima_f32c32.md](dcgen_anima_f32c32.md) | 进行中（POC 阶段 0/1 已跑通） | DC-Gen Anima f32c32 高压缩潜空间迁移、双缓存与 patch 对齐探针方案 |
| [krea2_nf4_fp16_layer_scaling.md](krea2_nf4_fp16_layer_scaling.md) | 半活跃 / 核心已落地（`--nf4_dir1` 与 WebUI 预检已上线） | Krea-2 NF4 逐层 `frexp`/`ldexp` 内层 FP16 缩放（sm70/sm75 提速）；1024² 长训与整网编译待验收 |
| [multigpu_reference_validation_contract.md](multigpu_reference_validation_contract.md) | 实验 / 验收契约（冻结版 v1，§13 S-d/S-g 由 v2 修订） | 多卡（DDP 优先）P0/P1 与 FSDP2/TP 探索的资产核对、梯度主干、控制面与 50 步真实短训预登记契约 |
| [multigpu_reference_validation_contract_amendment_v2.md](multigpu_reference_validation_contract_amendment_v2.md) | 实验 / 正式修订（v2 生效） | 解除数学不可达的 S-d / S-g 两条旧数值门，替换为 C1 数据对齐门 / C2 步级梯度 ULP 门 / C3–C4 实测噪声地板判据 |
| [adapter-aware-checkpoint.md](adapter-aware-checkpoint.md) | 半活跃提案（P0 已接入 Anima，P1–P3 仍为提案） | PEFT-aware 选择性激活检查点探索；P0 size-based 策略已通过 `--selective_checkpoint adapter_aware` 等开关接入 Anima |
| [prior_preservation_from_synth_pool.md](prior_preservation_from_synth_pool.md) | 提案（Status: proposal） | 复用 `distill-mod` 教师生成潜变量池做低成本先验保持正则化的方案 |
| [postfix_residual_for_directedit.md](postfix_residual_for_directedit.md) | 半活跃提案 | 为 DirectEdit 学习图像条件 postfix 残差以提升干跑重建保真度的设计记录 |
| [postfix_residual_per_image_inversion.md](postfix_residual_per_image_inversion.md) | 半活跃提案（仅保留 `exp-invert-directedit` 探针） | 单图反演 postfix 尾部以探测 DirectEdit 残差流形可行性的配套提案 |
| [soft_tokens_agsm.md](soft_tokens_agsm.md) | 半活跃提案 / 探针后已由 `softrank` 取代 | Soft Tokens 有界双 bank 对齐引导（AGSM）推导与门控设计记录 |
| [personalization-region-curriculum.md](personalization-region-curriculum.md) | 半活跃提案（核心组件已落地，真实图像盲评待补） | 区域→整图两阶段课程、`foreground_mean` 面积归一化与 APT 风格动态增广/权重方案 |
| [convrot_w8a_optimization_roadmap.md](convrot_w8a_optimization_roadmap.md) | 半活跃（P0-A/A2/C/D 完成；默认仍 sylvester） | ConvRot W8A* 训练后续优化 ROI 排序（prequant、regular Hadamard 与 Triton 门槛） |
| [convrot_90hx_followup_development_plan_20261002.md](convrot_90hx_followup_development_plan_20261002.md) | 半活跃（P0 生命周期修复与短测完成，后续待执行） | CMP 90HX 采样图/换层载荷内存边界修复复测与后续确定性校准计划 |

## 归档说明与维护规则

- 已经落地并沉淀为正式/实验说明的提案（如 PP trainer 接线契约、多卡阶段计划、ConvRot 初版训练规格、Soft Tokens 早期对比/排序提案、旧前端与配置渐进披露计划等）均已移入 `_archive/docs/proposal/`。
- 查阅已归档提案请前往归档总索引 [../archive-index.md](../archive-index.md) 或分区索引 [_archive/docs/proposal/README.md](../../_archive/docs/proposal/README.md)。
- 算法仍可能继续推进、但尚未成为稳定方法的探索文档继续留在本目录；归档提案时必须同步更新上述两处归档索引。
