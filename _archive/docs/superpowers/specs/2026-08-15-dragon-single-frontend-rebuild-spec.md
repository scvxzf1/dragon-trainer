# Dragon 单前端重建总规格

状态：历史设计规格（2026-08 迁移阶段）<br>
适用版本：当时的 Dragon 单前端迁移工作树<br>
执行计划：[`../plans/2026-08-15-dragon-single-frontend-rebuild.md`](../plans/2026-08-15-dragon-single-frontend-rebuild.md)

> 2026-10-04 整理：本文保留迁移阶段的目标、Classic 回退决策和完成门槛。
> Next 已是唯一生产前端，见[当前前端说明](../../../../web/frontend-next/README.md)
> 和[退役记录](../../proposal/legacy_ui_retirement_plan.md)。
> 本文的历史门禁不替代当前 [验证策略](../../../../AGENTS.md#验证策略)，也不证明每项迁移验收均已完成。

## 1. 目标

以 Dragon 的视觉语言和产品信息架构为基础，使用新的前端技术栈重新实现
Classic 已有功能，最终只保留一个前端。

这里的“迁移”只迁移用户能力、业务规则、API 契约和异常行为，不迁移 Classic 的
DOM、CSS、chunks、bridge、全局状态或副作用模块图。

## 2. 已锁定决策

1. 最终产品只保留 Dragon 前端。
2. 当前 aiohttp 后端继续作为业务真相源，不在本项目中同步重写后端。
3. Classic 是功能清单和行为基线，不是代码依赖或组件库。
4. 当前 vanilla Dragon 是视觉与交互参考，不作为新实现的运行时基础。
5. 新前端按纵向功能切片迁移，每片必须包含 UI、API、状态、错误态和测试。
6. 在功能对照和行为验证完成前，不删除 Classic 回退入口。

## 3. 非目标

- 不把 Classic 页面嵌入新应用。
- 不通过 iframe、动态 import 或兼容 bridge 复用旧运行时。
- 不复制 `features/anima-app/chunks/*` 到新目录后改名。
- 不在迁移期间改变训练、配置合并、队列或历史任务的后端语义。
- 不把所有页面状态放进单一全局 store。
- 不以“导航里有入口”作为功能完成证据。

## 4. 目标结构

```text
Dragon visual language
        |
new typed frontend application
        |
feature/domain modules
        |
typed API + WebSocket clients
        |
existing aiohttp backend
```

依赖方向必须保持：

```text
app -> features -> domain/shared -> api

禁止：new frontend -> classic/anima-app/legacy DOM
```

技术选型见
[`2026-08-15-dragon-frontend-architecture-adr.md`](2026-08-15-dragon-frontend-architecture-adr.md)。

## 5. 功能范围

首批范围按风险和用户价值排序：

1. 数据集蓝图：独立栏位、分组、新建/复制/重命名/导入导出、拖动排序与跨组移动、图片与 caption 预览、分阶段调度。
2. 训练配置：配置文件、字段分组、动态联动、来源和值校验、保存和另存。
3. 训练运行时：启动、进度、日志、WebSocket、停止和异常恢复。
4. 队列与历史：队列动作、失败策略、搜索筛选、详情、续训和预览。
5. 生图测试、预览工作区、权重分析、模型配置、全局设置和环境检测。

当前功能矩阵见
[`2026-08-15-dragon-feature-parity-matrix.md`](2026-08-15-dragon-feature-parity-matrix.md)。

## 6. 完成定义

单个功能只有同时满足以下条件才算完成：

- Classic 的用户入口和隐蔽操作已经枚举。
- 新实现不 import Classic、`anima-app` 或 legacy bridge。
- API 请求、成功响应、业务错误和网络错误均有明确行为。
- 加载、空数据、dirty、只读、并发更新和危险操作状态均有覆盖。
- 单元、组件、API 契约和真实浏览器关键流程测试通过。
- 桌面与移动端完成视觉和交互验证。
- 对照矩阵状态从 `未核验` 或 `迁移中` 更新为 `已验证`，并链接证据。

整个目标只有在所有 P0/P1 功能达到上述门槛、Dragon 不再依赖旧前端运行时、
Classic 退役计划完成后才算完成。

## 7. 文档规则

- 总体决策写在本文件和 ADR。
- 每个 feature 使用独立规格，不把所有细节继续堆入总文档。
- 每项规格使用 `草案 -> 已确认 -> 实现中 -> 已实现 -> 已验证` 状态。
- 代码或后端事实变化时，以实时源码和测试为准，并同步更新矩阵。
- 不确定功能必须标记为 `未核验`，不能按“Dragon 看起来已有”推定完成。
