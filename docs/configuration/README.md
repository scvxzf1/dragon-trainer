# 配置文档索引

状态：稳定

这里放配置根目录、路径解析、多模型训练预检与缓存策略相关说明。

## 当前文档

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [moss-image.md](moss-image.md) | 稳定 | MOSS-Image 文生图 plain LoRA、NF4/BF16 文本精度、实验 PP=2 1F1B 与训练前契约核对 |
| [external-configs.md](external-configs.md) | 稳定 | 解释 `ANIMA_CONFIGS_ROOT`、WebUI 全局设置里的 `configs_root`、路径解析优先级和迁移建议 |
| [model-training-capabilities.md](model-training-capabilities.md) | 稳定 | 五模型族训练能力标签、编辑数据集匹配、独立推理边界与兼容预检策略（`--compat_unverified_policy`，默认 `warn`） |
| [dataset-cache-semantics.md](dataset-cache-semantics.md) | 稳定 | Next `/next/datasets` 数据集蓝图的 `mask_mode`、五族缓存后缀、模型身份 stamp 与 runtime 预检契约 |
| [preprocess-auto-batch.md](preprocess-auto-batch.md) | 当前实现 | 五模型族 VAE/文本缓存从 1 起步的自动批大小、预测上拉、OOM 退避和固定预设兼容 |
| [qwen-edit-low-memory-preprocess.md](qwen-edit-low-memory-preprocess.md) | 稳定 | Qwen Image 2.1 Edit 配对数据预处理，以及 Qwen3-VL TE 缓存的 BF16 CPU offload 选择 |
| [auto-block-swap.md](auto-block-swap.md) | 实验 | 独立进程启动校准、进程内水位调优（`--block_swap_auto`）、五族 `blocks_to_swap` 区间与预检策略说明 |
| [sample-preview-memory.md](sample-preview-memory.md) | 稳定 | 全部五个注册模型族的训练预览通用 VAE 解码策略，以及 Qwen 预览 DiT 块交换覆盖 |

## 维护规则

- 配置字段、路径环境变量、WebUI 全局设置变更时，优先更新本目录。
- 历史计划和已完成提案放到 `_archive/docs/proposal/`；本机实施快照与旧界面配置清单放到
  `_archive/docs/configuration/`。本目录只保留当前使用说明。
- 示例路径不要写本机绝对路径，除非明确标注为示例。

已归档的历史配置清单与实施快照见
[`../../_archive/docs/configuration/README.md`](../../_archive/docs/configuration/README.md)（参见 [`../archive-index.md`](../archive-index.md)）。
