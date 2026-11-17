# Dragon Trainer 文档导航

这里是 Dragon Trainer（原 Anima LoRA WebUI / ALW）的文档总入口。本文负责按任务引导至当前权威说明、分区索引或历史归档，不重复抄写各分区内部的文件清单。

## 按任务查阅

| 你要做什么 | 入口 |
| --- | --- |
| 安装并启动 WebUI | [根 README](../README.md)、[Linux 部署指南](guidelines/linux-deployment.zh.md) |
| 使用和维护当前 Next WebUI | [Next 前端 README](../web/frontend-next/README.md)、[WebUI 功能索引](features/README.md) |
| 准备数据并启动训练 | [训练参考](guidelines/training.md) |
| 查看五个模型族的支持边界与跨族隔离契约 | [多模型支持现状](multi_model_support.md) |
| 处理兼容预检拦截、编辑数据集配对与 `--compat_unverified_policy` | [模型训练能力与兼容预检策略](configuration/model-training-capabilities.md) |
| 配置单卡 / DDP / 双卡流水线并行（PP v0） | [分布式训练卡片](experimental/distributed-training-card.md)、[PP v0 前置契约](experimental/pipeline_parallel_v0_prerequisites.md) |
| 准备 MOSS-Image 训练与分辨率/文本档位 | [MOSS-Image 配置说明](configuration/moss-image.md) |
| 配置训练预览 VAE 分块解码与 Qwen 预览换块覆盖 | [训练预览显存策略](configuration/sample-preview-memory.md) |
| 配置实验性中途 OOM 恢复与混合精度数值边界 | [训练恢复](experimental/adaptive-training-recovery.md)、[混合精度数值论证](structure/adaptive-precision-numerics.md) |
| 运行独立推理并组合 DCW、Spectrum、SMC-CFG | [推理参考](guidelines/inference.md) |
| 为数据集打标或手绘训练蒙版 | [打标工作台](features/tagging-workbench.md)、[手动蒙版编辑器](features/manual-mask-editor.md) |
| 查阅历史日志、管理训练产物与队列 | [历史日志](features/history-logs.md)、[数据管理](features/data-management.md)、[训练队列](features/training-queue.md) |
| 判断哪些维护操作可直接执行 | [授权分级](guidelines/authorization_tiers.md) |
| 查阅稳定方法 / 可运行实验 / 架构原理 | [方法索引](methods/README.md)、[实验索引](experimental/README.md)、[结构索引](structure/README.md) |
| 查阅实验结论、失败路径与真机审计报告 | [研究结论索引](findings/README.md) |
| 评估活跃提案、贡献代码或认领待办 | [提案索引](proposal/README.md)、[PR 规范](../CONTRIBUTING.md)、[贡献优先事项](contribution-priorities.md) |
| 查阅历史规划、旧版 UI 记录与已归档材料 | [归档索引](archive-index.md) |

## 分区索引

| 分区 | 内容边界 |
| --- | --- |
| [guidelines/](guidelines/README.md) | 安装部署、训练、推理、Git 同步与维护分级指南 |
| [configuration/](configuration/README.md) | 配置合并、外置路径、模型任务能力、缓存与预览显存策略 |
| [methods/](methods/README.md) | 稳定或已接入方法的使用说明与配置入口 |
| [experimental/](experimental/README.md) | 可运行但仍需按硬件/范围验证的实验能力 |
| [structure/](structure/README.md) | 模型、路由、数值精度与蒸馏的数学及架构原理（配置统一以 `methods/` 为准） |
| [features/](features/README.md) | 当前 Next WebUI 各独立工作区说明与前端评审标准 |
| [findings/](findings/README.md) | 真机实验结果、失败路径、性能剖析与兼容审计报告 |
| [optimizations/](optimizations/README.md) | 编译、算子、显存与训练剖析专项说明 |
| [proposal/](proposal/README.md) | 尚未落地或半活跃的设计提案与冻结验收契约 |

每个分区的 `README.md` 负责列出该分区全部文档；本页只链接分区入口与仓库级参考。

## 仓库级参考

| 文档 | 用途 |
| --- | --- |
| [multi_model_support.md](multi_model_support.md) | 五个注册模型族的能力矩阵、PP 现状与跨族隔离单一事实源 |
| [optimization-configs-current.md](optimization-configs-current.md) | 与实时配置链同步的优化与资源配置速查 |
| [project-lineage.md](project-lineage.md) | 项目血缘、上游政策与不可破坏的兼容命名面 |
| [contribution-priorities.md](contribution-priorities.md) | 从 PR 契约分离出的开放贡献方向与验证设想 |
| [archive-index.md](archive-index.md) | `_archive/docs/` 全部历史提案、旧 UI 记录与归档材料总入口 |

## 文档生命周期与防漂移原则

- **代码事实优先**：源码、配置与测试是当前行为的最终事实源；文档冲突时先核实源码再修正文档。
- **单一事实源**：不在多篇文档重复抄写易漂移的参数表或模型支持表；细节归口到唯一专题页，其余位置薄引用。
- **分层归位**：
  - 稳定路径放入 `guidelines/`、`methods/`、`configuration/`、`features/`；
  - 可运行实验放入 `experimental/`，活跃/半活跃提案放入 `proposal/`，真机结论与审计保留在 `findings/`；这三类文档前 25 行必须标注 `状态：`；
  - `experimental/` 与 `proposal/` 严禁保留同名文档并存；已完成、已替代或阶段施工类文档一律迁入 `_archive/docs/` 并登记 [archive-index.md](archive-index.md)。
- **链接卫生**：所有文档须从本页可达；未受 Git 跟踪的运行产物目录（如 `output/...`、`/tmp/...`）与本机绝对路径一律用反引号纯文本表示，不得写成 Markdown 链接。

## 验证

日常文档修订做差异审阅并检查改动路径：

```bash
git diff --check -- <changed-doc-paths>
```

移动、归档、重排索引或涉及核心契约同步时，运行文档与契约完整性护卫：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
```
