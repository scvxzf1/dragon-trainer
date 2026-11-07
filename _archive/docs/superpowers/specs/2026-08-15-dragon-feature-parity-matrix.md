# Dragon 单前端功能对照矩阵

状态：历史审计基线 / 冻结快照
日期：2026-08-15
说明：`已有` 只表示当前 Dragon 存在入口或实现，不等于新技术栈已经完成。

> 2026-10-04 整理：下文状态和证据保留 2026-08-15 至 2026-08-19 的迁移阶段记录，
> 不是当前功能目录或待办清单。现行入口见 [Next README](../../../../web/frontend-next/README.md)，
> 后续工程记录见 [Next 实施记录](../../../../docs/features/dragon-next-implementation.md)。
> 未核验项不因 Next 成为唯一入口而自动转为已验证；新增验收应单独记录日期和证据。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| 未核验 | 尚未逐行为审计 |
| 已有但旧依赖 | Dragon 有功能，但仍依赖 Classic/anima-app/bridge |
| 已有独立实现 | 当前 Dragon 有独立实现，仍需作为新前端行为参考 |
| 迁移中 | 新技术栈正在实现 |
| 已验证 | 新实现通过规格中的完整门禁 |

## P0：数据集蓝图

| 能力 | Classic 基线 | 当前 Dragon | 新前端状态 | 主要缺口/证据 |
| --- | --- | --- | --- | --- |
| 左侧预设库 + 右侧独立编辑栏 | 有 | 有 | 迁移中 | React/RHF 编辑表单已接入 defaults 和 subset field array；组件测试与真实浏览器只读验收已通过 |
| 新建、保存、另存、复制、重命名、删除 | 有 | 有 | 迁移中 | 已有 10 个组件事务测试；新建 `overwrite=false`、覆盖 `true`、重命名先另存后删除、只读限制均已固定；隔离临时数据的真实写入验收待补 |
| 导入、导出、搜索、刷新 | 有 | 有 | 迁移中 | 四项均已实现；导入使用 JSON/TOML 且默认 `overwrite=false`，导出使用 read 返回的原始 TOML；真实隔离文件验收待补 |
| 新建、重命名、删除分组 | 有 | 有独立 API 调用 | 迁移中 | typed mutation 和组件测试已覆盖三项；删除只移除分组元数据并保留 TOML，确认文案已经真实浏览器验收 |
| 分组排序 | 有 | 未证明等价 | 迁移中 | 新前端已接入 dnd-kit Pointer/Keyboard handle 及上下移按钮，提交 `{target:"group",scope:"dataset",index}`；搜索、锁定和未分组限制已有领域测试 |
| 组内预设排序 | 有 | 未证明等价 | 迁移中 | 拖动和箭头共用完整 `order` 事务；组件测试验证持久化 payload，真实浏览器验证键盘启动/Escape 取消 |
| 跨组拖动预设 | 有 | 有移动入口/API | 迁移中 | 新前端已有 Pointer/Keyboard 拖动目标、空组 dropzone 和显式分组选择器；未分组作为可写目标已补齐，隔离配置根真实写入验收待补 |
| subset 添加、删除、选择 | 有 | 有 | 迁移中 | RHF field array 已实现添加/删/稳定选中高亮，且至少保留一行普通训练数据 |
| subset 拖动排序 | 有 | 未证明完整等价 | 迁移中 | dnd-kit Pointer/Keyboard、箭头按钮和 `Alt+方向键` 已实现；组件测试验证顺序写入 payload，隔离 TOML 真实写入待补 |
| subset 批量 scope/高级字段 | 有 | 部分有 | 迁移中 | defaults 15 键、subset 7 键、行 settings 14 键及 NL/Tag/trigger clone 已全部对照；批量应用实验规则已有组件测试 |
| 图片与 caption 预览 | 有 | 有独立模块 | 已验证 | React 独立实现 saved/dirty guard、`source=source`、limit 120、Query 取消/刷新、lazy image、失败态、caption 展示/复制；37 个前端测试、59 个后端定向测试及真实 60 图浏览器验收通过 |
| 通用大图查看 | 有 | 有 | 已验证 | React 独立全屏查看器已覆盖图片详情、caption 复制、图片失败重试、Esc 分层关闭、Tab 焦点陷阱和焦点恢复；桌面与 390×844 移动端无横向溢出或详情覆盖 |
| 分阶段调度 | 有 | 有但旧依赖 | 已验证 | React 独立领域模型/dialog 已覆盖启用、模板、连续区间、subset 绑定、校验和保存 payload；无 Classic/anima-app/bridge import，组件与领域测试通过 |
| 未保存保护 | 有 | 有 | 已验证 | 切换、新建、重载、`beforeunload` 与 React Router `useBlocker` 已接入；测试覆盖 SPA 导航取消/确认和浏览器离开保护 |
| 应用到当前训练配置 | 有 | 有 | 已验证 | 新前端共享训练上下文选择器 + 二次确认 + typed apply API 已完成；dirty/未保存版本禁止应用，成功后刷新 merged config；隔离配置根验证 `dataset_config`、兼容路径、正则权重和阶段调度真实写入 |

数据集详细规格见
[`2026-08-15-dragon-dataset-editor-rebuild-spec.md`](2026-08-15-dragon-dataset-editor-rebuild-spec.md)。
字段级对照见
[`../../findings/dragon_dataset_subset_field_audit_20260815.md`](../../../../docs/findings/dragon_dataset_subset_field_audit_20260815.md)。

截图范围的逐项证据见
[`../../findings/dragon_classic_feature_parity_audit_20260815.md`](../../../../docs/findings/dragon_classic_feature_parity_audit_20260815.md)。

## P1：核心训练工作流

| 领域 | Classic 基线 | 当前 Dragon | 新前端状态 | 重点审计项 |
| --- | --- | --- | --- | --- |
| 训练配置 | 成熟 | 核心字段已有 | 迁移中 | `/next/training` 已覆盖共享上下文、16 个关键字段编辑、当前文件/继承来源、后端 patch preview、差异保存、另存防覆盖、dirty guard 和结构化 preflight；方法/模型族动态过滤、完整字段面与安全启动仍待实现 |
| 启动训练 | 成熟 | 已有入口 | 未核验 | preflight、危险确认、配置冻结、错误详情 |
| 实时训练 | 成熟 | 有 REST + WS | 迁移中 | 首版：状态/进度/指标/日志/GPU + 停止，REST 轮询 + WS 增量；待补日志搜索/下载与图表 |
| 队列 | 成熟 | 已有 | 迁移中 | 首版：统计筛选、策略表单、排序移动、重试/取消/停止、批量中止与清理；待补续训入队与运行目录清理删除 |
| 历史 | 成熟 | 已有 | 迁移中 | 首版：统计搜索筛选、批量归档/取消归档/删除、详情与配置快照；待补集合工作台、样张/权重、续训 |

## P1/P2：工具与系统

| 领域 | Classic 基线 | 当前 Dragon | 新前端状态 | 重点审计项 |
| --- | --- | --- | --- | --- |
| 生图测试 | 成熟 | 已有 | 未核验 | 权重解析、拖放、停止、多 sampler/dtype/attention |
| 预览工作区 | 成熟 | 已有 | 未核验 | 多来源、筛选、分组、删除安全和大图查看 |
| 权重分析 | 有 | 已有 | 未核验 | 上传、A/B、导出与错误恢复 |
| 全局模型配置 | 成熟 | 已有 | 未核验 | 创建、编辑、默认、删除、排序、引用冲突 |
| 全局设置 | 成熟 | 已有 | 未核验 | 路径校验、配置根覆盖、保存冲突 |
| 环境检测 | 有 | 已有 | 未核验 | 长请求、局部失败、复制诊断信息 |

## 完成纪律

- 每完成一个功能，必须把本矩阵对应项改为 `已验证` 并链接测试或浏览器证据。
- Dragon 当前存在的代码只能作为行为线索，不能直接计入新前端完成率。
- 无 Classic 显式入口的隐藏 dialog、快捷键、危险确认和错误路径仍属于功能范围。
- Classic 删除前，所有 P0/P1 项必须为 `已验证`；P2 项必须明确完成或经用户批准移除。

## 2026-08-15 本轮核验补充

- React 新前端的分阶段调度不是 Classic bridge 的包装：实现位于
  `web/frontend-next/src/features/dataset-editor/StageScheduleEditor.tsx` 与
  `stageSchedule.ts`，保存沿用 typed dataset preset API。
- 数据集工作区已补 React Router data-router blocker；浏览器刷新/关闭和 SPA 导航共享同一
  dirty 判定，不依赖 Classic router。
- 基础数据集 dialog 已统一 Escape、Tab 焦点陷阱、遮罩关闭、实例级标题 ID 和关闭后焦点恢复；
  分组、预设、subset 排序按钮统一使用 Lucide 图标。
- 当前前端门禁为 55 项测试，TypeScript 和 production build 继续作为每轮必跑项。
- 数据集应用已接入共享训练上下文：应用前明确目标训练 TOML、禁止 dirty 草稿和只读目标；隔离配置根验收覆盖新建防覆盖、subset/TOML 顺序、分组完整 order、阶段调度和训练配置实际落盘。
- 阶段 3 已推进到安全编辑：`/next/training` 只提交字段差异，preview 不写盘，另存强制不覆盖，preflight 将 `ok=false` 作为结构化检测结果而非请求失败；在模型族/方法过滤、配置冻结与危险确认完成前仍不暴露启动训练按钮。
- 隔离训练配置验收覆盖 preview 原文件不变、另存防覆盖、PATCH 持久化、空 `output_name` 原子拒绝和真实 preflight 结构。


## 2026-08-19 首版迁移补充

- 队列、实时训练、历史三块已在 `/next` 新前端落地首版，共用后端契约，无 Classic/anima-app/bridge import。
- 新增共享 `Topbar` 与 `useWebSocket`，`DatasetWorkspace`/`TrainingWorkspace` 已改为复用。
- 前端门禁升至 68 个测试，`typecheck` 与 production build 通过；详情见
  [`../../findings/dragon_next_stage1_queue_monitor_history_20260819.md`](../../../../docs/findings/dragon_next_stage1_queue_monitor_history_20260819.md)。
- 2026-08-19 完成一轮全功能审计与修复（真实后端契约 + Playwright 隔离环境浏览器验收），详见
  [`../../findings/dragon_next_audit_20260819.md`](../../../../docs/findings/dragon_next_audit_20260819.md)。
