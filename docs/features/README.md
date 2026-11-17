# 功能文档索引

状态：稳定

本目录存放当前 Next WebUI（`/next`）各独立业务工作区的用户与维护说明，不存放训练算法或模型数学正文。前端开发、构建与隔离验收入口见 [Next 前端 README](../../web/frontend-next/README.md)。

## 当前功能文档

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [config-workbench.md](config-workbench.md) | 稳定 | `/next/training` 配置工作台：四阶段表单、分布式与 PP 卡片、预检风险确认、启动与入队边界 |
| [preprocess-cache-reuse.md](preprocess-cache-reuse.md) | 稳定 | 预处理缓存复用：共享池、A/B/C 开关、指纹模式与历史删除隔离 |
| [dataset-editor.md](dataset-editor.md) | 稳定 | `/next/datasets` 数据集蓝图工作区：子集管理、字段帮助、Qwen Edit 配对与图片工作台 |
| [manual-mask-editor.md](manual-mask-editor.md) | 稳定 | 数据集子集图片工作台中的手绘蒙版编辑、原子保存与子集接入 |
| [tagging-workbench.md](tagging-workbench.md) | 稳定 | `/next/captioning` 打标工作台：外部 API 与本地 ONNX 打标、标签频次批量审阅与写回边界 |
| [training-queue.md](training-queue.md) | 稳定 | `/next/queue` 训练队列：排队、暂停、自动重试、失败策略与 revision 冲突保护 |
| [history-collections.md](history-collections.md) | 稳定 | `/next/history` 历史任务与集合：筛选、对比、合并查看、两档删除、原地续训与详情 |
| [history-logs.md](history-logs.md) | 稳定 | 历史详情完整日志：稀疏索引分页、虚拟滚动、行/页跳转与全局搜索 |
| [preview.md](preview.md) | 稳定 | `/next/preview` 预览工作区：训练样张、推理输出、自定义目录与权重管理 |
| [global-settings.md](global-settings.md) | 稳定 | `/next/settings` 全局设置：输出根、配置根、生图测试设置、显示偏好与分页面缩放 |
| [global-model-configs.md](global-model-configs.md) | 稳定 | `/next/models` 模型配置库：五模型族路径管理、能力标签筛选、路径状态检查与默认镜像 |
| [data-management.md](data-management.md) | 稳定 | `/next/data` 数据管理：目录容量扫描、租约保护清理与检查点保留策略抽屉 |
| [ui-scale.md](ui-scale.md) | 稳定 | UI 缩放：全局默认比例与分页面/历史详情子页独立比例 |
| [frontend-health-scorecard.md](frontend-health-scorecard.md) | 维护参考 | Next 前端集中审阅评分结构与模板（具体历史轮次审计见 `docs/findings/`） |

## 历史记录

已退役静态 Dragon / classic UI 盘点及 2026-09 早期实施流水账已归档至 [`_archive/docs/features/README.md`](../../_archive/docs/features/README.md)（总览见 [`../archive-index.md`](../archive-index.md)）。

## 维护规则

- 训练或推理算法方法放入 `docs/methods/` 或 `docs/experimental/`；本目录只承载 WebUI 工作区与设置说明。
- 新增或重命名本目录文档时，同步更新本索引与 [`../README.md`](../README.md)。
- 文档中引用的后端与前端测试路径必须对应仓库当前真实存在的测试文件。
