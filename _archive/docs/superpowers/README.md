# Superpowers 索引

一句话：这里放维护用设计规格、执行计划和自动迭代协议；冻结的迁移快照单独列出，不是用户主路径。

状态：已归档
适用版本：2026-07 至 2026-09 维护期（不再对应当前 main）
相关入口：`docs/README.md`

## 生命周期

| 阶段 | 放哪里 |
| --- | --- |
| 进行中的设计 / 计划 | 本目录 `specs/`、`plans/` |
| 已沉淀的稳定结论 | `docs/findings/` 或正式功能/方法文档 |
| 冻结的设计 / 执行快照 | 可保留原位以维护引用，须在头部与索引标明历史范围和当前入口 |
| 已完成且不再推进的提案 | `_archive/docs/proposal/` |

## Specs

| 文档 | 说明 |
| --- | --- |
| [specs/2026-08-15-dragon-frontend-architecture-adr.md](specs/2026-08-15-dragon-frontend-architecture-adr.md) | 新前端技术架构 ADR |
| [specs/2026-08-15-dragon-dataset-editor-rebuild-spec.md](specs/2026-08-15-dragon-dataset-editor-rebuild-spec.md) | 数据集分组、拖动和图片预览重建规格 |
| [specs/2026-07-11-five-round-auto-iteration-protocol.md](specs/2026-07-11-five-round-auto-iteration-protocol.md) | 五轮自动迭代协议，前端强化版 |
| [specs/2026-07-11-frontend-config-optimization-design.md](specs/2026-07-11-frontend-config-optimization-design.md) | 前端配置优化设计 |
| [specs/2026-07-11-web-frontend-boulder-audit-design.md](specs/2026-07-11-web-frontend-boulder-audit-design.md) | Web 前端石山只读体检设计 |
| [specs/2026-07-11-backend-config-optimization-design.md](specs/2026-07-11-backend-config-optimization-design.md) | 后端配置优化设计 |
| [specs/2026-07-11-backend-next-optimization-design.md](specs/2026-07-11-backend-next-optimization-design.md) | 后端下一轮优化设计 |
| [specs/2026-07-11-backend-residual-optimization-design.md](specs/2026-07-11-backend-residual-optimization-design.md) | 后端残留优化设计 |
| [specs/2026-07-11-backend-round-c-product-decisions-design.md](specs/2026-07-11-backend-round-c-product-decisions-design.md) | 后端 C 轮产品决策设计 |
| [specs/2026-07-11-dataset-page-stage-schedule-ia-design.md](specs/2026-07-11-dataset-page-stage-schedule-ia-design.md) | 数据集页阶段/排期信息架构设计 |
| [specs/2026-07-11-preprocess-cache-reuse-design.md](specs/2026-07-11-preprocess-cache-reuse-design.md) | 预处理缓存复用设计 |
| [specs/2026-07-12-fp16-training-parity-design.md](specs/2026-07-12-fp16-training-parity-design.md) | FP16 训练一致性设计 |
| [specs/2026-07-12-webui-design-system-console-upgrade-design.md](specs/2026-07-12-webui-design-system-console-upgrade-design.md) | WebUI 设计系统控制台升级设计 |
| [specs/2026-07-27-history-manager-extra-filters-design.md](specs/2026-07-27-history-manager-extra-filters-design.md) | 历史任务全局搜索扩展筛选设计 |

## Plans

| 文档 | 说明 |
| --- | --- |
| [plans/2026-07-11-fullstack-auto-iteration-log.md](plans/2026-07-11-fullstack-auto-iteration-log.md) | 全栈自动迭代日志 |
| [plans/2026-07-11-frontend-config-optimization.md](plans/2026-07-11-frontend-config-optimization.md) | 前端配置优化执行计划 |
| [plans/2026-07-11-web-frontend-boulder-audit.md](plans/2026-07-11-web-frontend-boulder-audit.md) | Web 前端石山只读体检执行计划 |
| [plans/2026-07-11-backend-config-optimization.md](plans/2026-07-11-backend-config-optimization.md) | 后端配置优化执行计划 |
| [plans/2026-07-11-backend-next-optimization.md](plans/2026-07-11-backend-next-optimization.md) | 后端下一轮优化执行计划 |
| [plans/2026-07-11-backend-residual-optimization.md](plans/2026-07-11-backend-residual-optimization.md) | 后端残留优化执行计划 |
| [plans/2026-09-13-webui-backend-audit-followup.md](plans/2026-09-13-webui-backend-audit-followup.md) | WebUI 后端审计后续整改计划（5 小时窗口） |
| [plans/2026-07-11-backend-round-c-product-decisions.md](plans/2026-07-11-backend-round-c-product-decisions.md) | 后端 C 轮产品决策执行计划 |
| [plans/2026-07-11-dataset-page-stage-schedule-ia.md](plans/2026-07-11-dataset-page-stage-schedule-ia.md) | 数据集页阶段/排期信息架构执行计划 |
| [plans/2026-07-11-networks-cycle-break.md](plans/2026-07-11-networks-cycle-break.md) | networks 循环依赖打断计划 |
| [plans/2026-07-11-preprocess-cache-reuse.md](plans/2026-07-11-preprocess-cache-reuse.md) | 预处理缓存复用执行计划 |
| [plans/2026-07-12-fp16-training-parity.md](plans/2026-07-12-fp16-training-parity.md) | FP16 训练一致性执行计划 |
| [plans/2026-07-12-webui-design-system-console-upgrade.md](plans/2026-07-12-webui-design-system-console-upgrade.md) | WebUI 设计系统控制台升级执行计划 |
| [plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md](plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md) | WebUI 设计系统升级迭代日志 |
| [plans/2026-07-27-history-manager-extra-filters.md](plans/2026-07-27-history-manager-extra-filters.md) | 历史任务全局搜索扩展筛选执行计划 |

## 单前端迁移历史快照

当前运行与维护入口见 [Next README](../../../web/frontend-next/README.md)。这些材料保留各自阶段的设计、
勾选状态和验收证据；冻结快照不等于宣告全部迁移项已验证。

| 文档 | 说明 |
| --- | --- |
| [specs/2026-08-15-dragon-single-frontend-rebuild-spec.md](specs/2026-08-15-dragon-single-frontend-rebuild-spec.md) | 历史总规格，含当时的 Classic 回退与完成门槛 |
| [specs/2026-08-15-dragon-feature-parity-matrix.md](specs/2026-08-15-dragon-feature-parity-matrix.md) | 2026-08-15 至 2026-08-19 的功能审计基线 |
| [plans/2026-08-15-dragon-single-frontend-rebuild.md](plans/2026-08-15-dragon-single-frontend-rebuild.md) | 历史执行计划与阶段日志 |

## 已归档的实施材料

| 入口 | 说明 |
| --- | --- |
| [2026-07-12 仪表面板换皮记录](../proposal/webui-instrument-panel-reskin/README.md) | 已完成的旧静态 UI 设计、计划、迭代日志和最终评审；当前前端以 Next README 为准 |

## 维护规则

- 新增 plan / spec 后，同步更新本索引和 [../../../docs/README.md](../../../docs/README.md)。
- 用户功能说明不要放这里，放到 `docs/features/`。
- 算法方法说明不要放这里，放到 `docs/methods/` 或 `docs/experimental/`。
- 完成后及时迁出，避免施工区长期堆积。
