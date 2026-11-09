# Experimental 索引

一句话：这里放可运行但仍在实验、调参或验证阶段的方法说明。

状态：实验
适用版本：当前 main
相关入口：`docs/README.md`

## 当前文档

| 文档 | 说明 |
| --- | --- |
| [adaptive-runtime.md](adaptive-runtime.md) | DiT 自适应精度与 OOM 恢复，实现边界和双卡三模型验证进度 |
| [adaptive-training-recovery.md](adaptive-training-recovery.md) | 默认关闭的中途 OOM 恢复、sampler/预取与完整状态合同；Anima 90HX 短步验收 |
| [anima-native-fp16.md](anima-native-fp16.md) | Anima 原生 AMP 配置、自动路由与训练精度报告 |
| [anima_pipeline_parallel_v0.md](anima_pipeline_parallel_v0.md) | Anima 流水线并行 v0：2 阶段 stage-local 运行时、1F1B、fp16 事务、per-rank checkpoint；主训练入口仍具名 fail-closed |
| [krea2-native-fp16.md](krea2-native-fp16.md) | Krea-2 显式原生 FP16 窄配置、guard 与验证边界 |
| [anima-native-fp16-acceptance.md](anima-native-fp16-acceptance.md) | 原生 FP16 集中严格验收与独立训练质量协议 |
| [anima-mixed-nf4.md](anima-mixed-nf4.md) | Anima 固定 MLP 策略 BF16/NF4 保存与加载实验 |
| [multi-family-mixed-nf4.md](multi-family-mixed-nf4.md) | Krea-2、Z-Image、Qwen Image 2.1 主干 MLP 混合 NF4 格式、加载、转换与小模型训练验收 |
| [anima_tagger.md](anima_tagger.md) | Anima Tagger，多标签 tagger 与 DirectEdit 文本入口 |
| [byg.md](byg.md) | BYG unpaired instruction-editing：训练可用，专用推理仍是占位 |
| [chimera-hydra.md](chimera-hydra.md) | ChimeraHydra 双池 MoE，配合 [../structure/chimera-hydra.md](../structure/chimera-hydra.md) 阅读 |
| [convrot_int8_training.md](convrot_int8_training.md) | ConvRot int8 训练（W8A16/W8A8）；**可运行实验、默认关闭**；规格见 [../proposal/convrot_w8a_training_plan.md](../proposal/convrot_w8a_training_plan.md)；后续优化见 [../proposal/convrot_w8a_optimization_roadmap.md](../proposal/convrot_w8a_optimization_roadmap.md) |
| [nm-w4a16.md](nm-w4a16.md) | 2:4 uniform INT4/BF16 冻结基座实验；静态实现，首期以 dense BF16 为对照，默认不接生产训练 |
| [directedit_editing_v3.md](directedit_editing_v3.md) | DirectEdit v3，flow-inversion 图像编辑 |
| [dpdmd.md](dpdmd.md) | DP-DMD / Turbo Anima 少步蒸馏，配合 [../structure/dpdmd.md](../structure/dpdmd.md) 阅读 |
| [easycontrol.md](easycontrol.md) | EasyControl 图像条件控制 |
| [fera.md](fera.md) | FeRA / FEI 路由实验 |
| [ip-adapter.md](ip-adapter.md) | IP-Adapter 图像 cross-attention 条件 |
| [postfix.md](postfix.md) | Postfix 兼容入口，当前用户入口见训练参考 |
| [qwen_image_2_1_fused_projections.md](qwen_image_2_1_fused_projections.md) | Qwen Image 2.1 冻结投影融合与定额保留实验，默认关闭 |
| [qwen-nf4-compile.md](qwen-nf4-compile.md) | Qwen Image 2.1 Edit NF4 编译受限入口，默认关闭；真实两步 AdamW、预览与完整状态续训短测通过 |
| [soft_tokens.md](soft_tokens.md) | Soft Tokens / SoftREPA 风格 per-layer token bank |
| [spd.md](spd.md) | SPD：Spectral Progressive Diffusion 推理实验 |
| [vera_ablation.md](vera_ablation.md) | VeRA 短期消融计划 |
| [vr_loss.md](vr_loss.md) | Variance reduction loss 实验 |

## 维护规则

- 文档顶部或正文要说清楚实验边界、占位能力和当前可运行入口。
- 能力稳定并进入主路径后，迁到 `docs/methods/`，并同步更新两边索引。
- 原理、数学和架构细节放到 `docs/structure/`。
- 新增文档后，同步更新本索引和 [../README.md](../README.md)。
