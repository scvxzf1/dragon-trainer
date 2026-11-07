# Dragon Trainer 文档导航

这里是 Dragon Trainer（原 Anima LoRA WebUI / ALW）的文档总入口。本文只负责把读者带到正确的当前说明、分区索引或历史证据，不重复维护各分区的完整文件清单。

## 按任务查阅

| 你要做什么 | 入口 |
| --- | --- |
| 安装并启动 WebUI | [根 README](../README.md)、[Linux 部署指南](guidelines/linux-deployment.zh.md) |
| 使用和维护当前 WebUI | [Next 前端 README](../web/frontend-next/README.md) |
| 准备数据并训练 | [训练参考](guidelines/training.md) |
| 推理与组合 DCW、Spectrum | [推理参考](guidelines/inference.md) |
| 理解配置合并和外置路径 | [配置索引](configuration/README.md) |
| 使用 WebUI 独立功能 | [功能索引](features/README.md) |
| 查阅和清理训练产物 | [数据管理](features/data-management.md) |
| 阅读完整历史日志、跳转与搜索 | [历史日志](features/history-logs.md) |
| 使用外部模型 API 打标 | [打标工作台](features/tagging-workbench.md) |
| 查阅已退役 Dragon / classic 的历史记录 | [Dragon UI 历史指南](features/dragon-ui.md) |
| 了解旧 Dragon 八个界面的历史功能盘点 | [Dragon UI 功能图谱](features/dragon-ui-functional-map.md) |
| 查阅新前端实施记录 | [Dragon Next 实施记录](features/dragon-next-implementation.md) |
| 为数据集手动绘制训练蒙版 | [手动蒙版编辑器](features/manual-mask-editor.md) |
| 查阅 Next 视觉与工作流修复的阶段记录 | [Next UI/UX 实施记录](features/dragon-next-uiux-progress.md) |
| 审计并推进 WebUI 后端整改 | [后端审计后续计划（已归档）](../_archive/docs/superpowers/plans/2026-09-13-webui-backend-audit-followup.md) |
| 参考前端审阅方法和历史评分 | [前端健康度评分卡](features/frontend-health-scorecard.md) |
| 查稳定方法 | [方法索引](methods/README.md) |
| 查可运行实验 | [实验索引](experimental/README.md) |
| 按已验证配置试验 Qwen Edit NF4 编译 | [Qwen NF4 编译实验](experimental/qwen-nf4-compile.md) |
| 理解模型、路由和优化原理 | [结构索引](structure/README.md) |
| 理解 FP16/FP32 训练误差与静态测试边界 | [混合精度数值论证](structure/adaptive-precision-numerics.md) |
| 配置实验中途 OOM 恢复与独立预览回退 | [训练恢复](experimental/adaptive-training-recovery.md)、[预览内存策略](configuration/sample-preview-memory.md) |
| 查实验结论、失败路径和审计 | [研究结论索引](findings/README.md) |
| 复查 MOSS 热测试与作废尝试 | [CMP 50HX 热测试复盘（含尸体记录）](findings/moss_image_tier_hot_50hx_20261018.md) |
| 准备 MOSS-Image 真实训练 | [MOSS 训练前置清单](configuration/moss-image-training-preflight.md)、[MOSS 配置说明](configuration/moss-image.md) |
| 评估活跃提案或维护计划 | [提案索引](proposal/README.md)、[施工区归档索引](../_archive/docs/superpowers/README.md) |
| 准备贡献或认领待办 | [PR 规范](../CONTRIBUTING.md)、[贡献优先事项](contribution-priorities.md) |
| 查归档材料 | [归档索引](archive-index.md) |

## 分区索引

| 分区 | 内容边界 |
| --- | --- |
| [guidelines/](guidelines/README.md) | 安装、训练、推理和维护操作说明 |
| [configuration/](configuration/README.md) | 配置、路径和环境变量 |
| [methods/](methods/README.md) | 稳定或已接入能力的使用说明 |
| [experimental/](experimental/README.md) | 可运行但仍需验证或调参的能力 |
| [structure/](structure/README.md) | 原理、数学和架构 |
| [features/](features/README.md) | WebUI 独立功能 |
| [findings/](findings/README.md) | 实验结果、失败路径、审计和运行报告 |
| [optimizations/](optimizations/README.md) | compile、kernel、显存和性能优化 |
| [proposal/](proposal/README.md) | 活跃或半活跃提案 |
| [superpowers/（已归档）](../_archive/docs/superpowers/README.md) | 2026-07~09 维护期 spec、plan、执行日志和迭代协议；已整体归档 |

每个分区的 `README.md` 负责列出该分区文档；本页只链接分区入口。

## 仓库级参考

| 文档 | 用途 |
| --- | --- |
| [contribution-priorities.md](contribution-priorities.md) | 从 PR 规范中分离出的当前贡献方向和验收设想 |
| [optimization-configs-current.md](optimization-configs-current.md) | 与实时配置面同步的优化配置事实清单 |
| [optimization-roadmap.md](optimization-roadmap.md) | 优化候选方向与按风险选择的验收方式 |
| [multi_model_support.md](multi_model_support.md) | 多模型支持的架构耦合地图 |
| [project-lineage.md](project-lineage.md) | 项目血缘、上游政策与不能动的兼容面 |
| [separation_plan.md](separation_plan.md) | 推理拆分的历史规划快照；不表示搬迁已完成 |
| [archive-index.md](archive-index.md) | `_archive/docs/` 历史材料入口 |

## 文档生命周期

- 代码、配置和测试是当前行为的最终事实来源；文档冲突时先核实实现，再修正文档。
- 用户稳定路径放在 `guidelines/`、`methods/`、`configuration/` 和 `features/`。
- 可运行但边界未稳定的能力放在 `experimental/`，并在前 25 行标明状态和限制。
- 实验结果、失败结论和审计证据保留在 `findings/`；它们不因结论过时而删除，但必须与当前说明区分。
- 活跃设计放在 `proposal/`；完成、失效或只服务历史合并的材料通过 [归档索引](archive-index.md) 进入 `_archive/docs/`。
- `superpowers/` 的 2026-07~09 维护期材料已整体归档到 `_archive/docs/superpowers/`；新增维护施工材料如需复活该分区，应同时更新本页与 [归档索引](archive-index.md)。
- 新增、移动或归档 Markdown 时，必须更新本页或对应分区索引，保证从本页可达。
- 根 `README.md` 只保留项目介绍、部署快照和最高频入口；深入内容进入本树。

## 验证

日常文档修订按 [验证策略](../AGENTS.md#验证策略) 做差异审阅，并只检查本次修改的路径：

```bash
git diff --check -- <changed-doc-paths>
```

新增或修改链接时检查受影响的本地目标和锚点；移动、归档或重排索引时，还要确认相关文档
从本页可达。大规模整理使用本地链接检查，普通文字修订不默认运行测试套件。

需要集中验收文档结构、围栏、索引完整性、生命周期标记及当前配置声明时，可选用：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
```

外部链接只在任务需要时人工核验；不要为了整理导航改写实验结论或历史运行记录。
