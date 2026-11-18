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
| [anima_pipeline_parallel_v0.md](anima_pipeline_parallel_v0.md) | Anima 流水线并行 v0：2 阶段 stage-local 运行时、1F1B、纯 fp16 + GradScaler 事务、per-rank checkpoint；trainer 级接线已验收（见下行） |
| [anima_pp_trainer_wiring.md](anima_pp_trainer_wiring.md) | Anima PP 主训练接线：经族适配器复用库级 runtime，正式入口派发到专用 PP 会话；5D wire / max-padded crossattn / 按族 resume 格式戳；CPU/Gloo 与双 V100 CUDA tiny 热测均已独立验收（rc=0），真实大模型长训未验收 |
| [distributed-training-card.md](distributed-training-card.md) | WebUI「分布式训练」三态卡片使用说明：单卡/DDP/PP（已开放 `anima` / `moss_image` / `krea2_raw` 三族）、GPU 全卡可选＋解析级 fail-closed 守卫、🔴🟡🔵 三档诊断、Krea-2 `pipeline_parallel_blocks_to_swap` 与 `[distributed]` 设置落点 |
| [moss_image_pipeline_parallel_v0.md](moss_image_pipeline_parallel_v0.md) | MOSS-Image 流水线并行 v0：复用 PP v0 骨架的族专属双 stage、真实文本长度 conditioning 与分 rank checkpoint；库级 runtime 与主训练接线均已合入并验收 |
| [moss_image_pp_trainer_wiring.md](moss_image_pp_trainer_wiring.md) | MOSS-Image PP 主训练接线：正式 CLI/train 入口派发到专用 PP 会话、trainer 级保存/恢复与步进语义；CPU/Gloo 与双 V100 CUDA tiny 热测均已独立验收（rc=0），真实大模型长训未验收 |
| [pipeline_parallel_v0_prerequisites.md](pipeline_parallel_v0_prerequisites.md) | PP v0 上手条件与三族（Anima / MOSS / Krea-2）差异简表：默认是 DDP、最小配置、硬性约束（stages=2 / 仅限 `mixed_precision="fp16"` / batch 整除 microbatches / 普通 swap 互斥、Krea-2 支持 `pipeline_parallel_blocks_to_swap`）、双 V100 实测基线与通信路径 |
| [krea2-native-fp16.md](krea2-native-fp16.md) | Krea-2 单卡非量化显式原生 FP16 窄配置、guard，以及与双卡 PP / sm70-75 NF4 内层 FP16 的分流说明 |
| [anima-native-fp16-acceptance.md](anima-native-fp16-acceptance.md) | 原生 FP16 集中严格验收与独立训练质量协议 |
| [anima-mixed-nf4.md](anima-mixed-nf4.md) | Anima 固定 MLP 策略 BF16/NF4 保存与加载实验 |
| [multi-family-mixed-nf4.md](multi-family-mixed-nf4.md) | Krea-2、Z-Image、Qwen Image 2.1 主干 MLP 混合 NF4 格式、加载与转换（不含 `moss_image`，与 `--nf4_dir1` 边界独立） |
| [adapter-qkv-fuse-compat.md](adapter-qkv-fuse-compat.md) | Adapter QKV Fuse/Defuse 统一入口、旧 helper 兼容边界与审计结论 |
| [anima_tagger.md](anima_tagger.md) | Anima Tagger，多标签 tagger 与 DirectEdit 文本入口 |
| [byg.md](byg.md) | BYG unpaired instruction-editing：训练可用，专用推理仍是占位 |
| [chimera-hydra.md](chimera-hydra.md) | ChimeraHydra 双池 MoE，配合 [../structure/chimera-hydra.md](../structure/chimera-hydra.md) 阅读 |
| [convrot_int8_training.md](convrot_int8_training.md) | ConvRot int8 训练（W8A16/W8A8）；**可运行实验、默认关闭**；历史规格见 [../../_archive/docs/proposal/convrot_w8a_training_plan.md](../../_archive/docs/proposal/convrot_w8a_training_plan.md)；后续优化见 [../proposal/convrot_w8a_optimization_roadmap.md](../proposal/convrot_w8a_optimization_roadmap.md) |
| [nm-w4a16.md](nm-w4a16.md) | 2:4 uniform INT4/BF16 冻结基座实验；静态实现，首期以 dense BF16 为对照，默认不接生产训练 |
| [directedit_editing_v3.md](directedit_editing_v3.md) | DirectEdit v3，flow-inversion 图像编辑 |
| [dpdmd.md](dpdmd.md) | DP-DMD / Turbo Anima 少步蒸馏，配合 [../structure/dpdmd.md](../structure/dpdmd.md) 阅读 |
| [easycontrol.md](easycontrol.md) | EasyControl 图像条件控制；配套见 [EasyControl 适配器开发指南](../../easycontrol_adapters/ADAPTER_GUIDE.cn.md) 与 [上色适配器说明](../../easycontrol_adapters/colorization/README.md) |
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
