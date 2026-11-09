# 配置文档索引

这里放配置根目录、路径解析和配置外置相关说明。

## 当前文档

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [moss-image.md](moss-image.md) | 首版基础实现 | MOSS-Image 文生图 plain LoRA、NF4/BF16 文本精度与兼容边界 |
| [moss-image-training-preflight.md](moss-image-training-preflight.md) | 维护用清单 | MOSS 记录索引、训练前契约核对、本机现状与执行序列模板 |
| [external-configs.md](external-configs.md) | 当前实现说明 | 解释 `ANIMA_CONFIGS_ROOT`、WebUI 全局设置里的 `configs_root`、路径解析优先级和迁移建议 |
| [model-training-capabilities.md](model-training-capabilities.md) | 当前实现说明 | 模型训练能力标签、编辑数据集匹配、预检查、兼容预检策略（`--compat_unverified_policy`，默认 warn）与旧任务参数兼容 |
| [dragon-training-config-215.md](dragon-training-config-215.md) | 历史快照 | 记录已退役 Dragon 配置页当时显示的 215 项；当前配置入口为 `/next/training` |
| [dataset-cache-semantics.md](dataset-cache-semantics.md) | 当前实现说明 | Dragon 数据集的 `mask_mode`、family-aware 缓存、模型身份 stamp 与 runtime 预检契约 |
| [preprocess-auto-batch.md](preprocess-auto-batch.md) | 当前实现说明 | VAE/文本缓存从 1 起步的自动批大小、预测上拉、OOM 退避和固定预设兼容 |
| [qwen-edit-low-memory-preprocess.md](qwen-edit-low-memory-preprocess.md) | 当前实现说明 | Qwen Image 2.1 Edit 配对数据预处理，以及 Qwen3-VL TE 缓存的 BF16 CPU offload 选择 |
| [auto-block-swap.md](auto-block-swap.md) | 实验 | 独立进程启动校准、物理内存预算、完整训练步验证及手动配置兼容 |
| [sample-preview-memory.md](sample-preview-memory.md) | 当前实现说明 | 所有模型族训练预览 VAE 的分块、OOM 回退、CPU 策略与旧 Qwen 键兼容 |
| [qwen-preview-memory.md](qwen-preview-memory.md) | 当前实现说明 | Qwen 预览 DiT 块交换覆盖及通用 VAE 策略 |

## 维护规则

- 配置字段、路径环境变量、WebUI 全局设置变更时，优先更新本目录。
- 历史计划和已完成提案放到 `_archive/docs/proposal/`；本机实施快照放到
  `_archive/docs/configuration/`。本目录只保留当前使用说明。
- 示例路径不要写本机绝对路径，除非明确标注为示例。

历史实施快照已移到
[_archive/docs/configuration/](../../_archive/docs/configuration/README.md)。
