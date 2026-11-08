# Dragon 单前端重建执行计划

状态：历史执行快照（2026-08 阶段）<br>
规格：[`../specs/2026-08-15-dragon-single-frontend-rebuild-spec.md`](../specs/2026-08-15-dragon-single-frontend-rebuild-spec.md)

> 2026-10-04 整理：Next 已成为唯一生产前端，当前入口见
> [Next README](../../../../web/frontend-next/README.md)，退役过程见
> [退役记录](../../proposal/legacy_ui_retirement_plan.md)。
> 下文阶段勾选和门禁保留当时的执行状态，不用作当前待办或完成率；入口切换不代表所有原验收项已补齐。

## 阶段 0：规格与证据基线

- [x] 锁定“单 Dragon、保留后端、按功能重写、不复用旧前端代码”的方向。
- [x] 建立总体规格、技术 ADR、功能矩阵和数据集 P0 规格。
- [ ] 枚举全部 Classic 导航、隐藏 dialog、快捷操作和 API。
- [ ] 为功能矩阵每一项补充 `file:line` 和现有测试证据。

## 阶段 1：Toolchain Spike

- [x] 建立独立 TypeScript/Vite 前端目录和构建命令。
- [x] 接入 React、Router、Query、Form、Zod 和测试工具。
- [x] 验证 aiohttp 静态构建产物、开发代理和 SPA fallback。
- [x] 建立 Dragon tokens、基础布局、按钮、表单、dialog 和错误面。
- [x] 跑通真实 GET、mutation、表单和桌面/移动浏览器 smoke。

门禁：新应用不得加载 Classic stylesheet、`app.js` 或 anima-app runtime。

## 阶段 2：数据集蓝图 P0

- [x] 定义 typed dataset/file-group API client 和契约测试。
- [x] 实现预设库、独立编辑栏和 dirty guard。
- [x] 实现分组 CRUD、组内排序和跨组拖动。
- [x] 实现 subset 编辑、添加/删除和拖动排序。
- [x] 实现图片/caption 预览和大图查看。
- [x] 独立实现 stage schedule model/dialog。
- [ ] 完成隔离配置根的真实写入验收，并固化为可重复 Playwright 门禁。

门禁：数据集新实现对 Classic/anima-app/bridge 的 import 数为 0。

## 阶段 3：训练配置与启动

- [ ] 建立配置 schema、字段 catalog 和 typed merged-config contract。
- [ ] 实现配置文件管理、分类表单、动态兼容过滤和来源显示。
- [ ] 实现保存、另存、导入和 preflight。
- [ ] 实现训练启动、危险确认和错误详情。

## 阶段 4：训练运行时、队列与历史

- [ ] 实现独立 WebSocket client、REST fallback 和清理生命周期。
- [ ] 实现训练进度、日志和停止控制。
- [ ] 实现队列全部动作和失败策略。
- [ ] 实现历史筛选、详情、续训、预览和权重入口。

## 阶段 5：工具与系统

- [ ] 生图测试。
- [ ] 预览工作区。
- [ ] 权重分析。
- [ ] 全局模型配置。
- [ ] 全局设置和环境检测。

## 阶段 6：切换与 Classic 退役

- [ ] 所有 P0/P1 矩阵项达到 `已验证`。
- [ ] 完整 desktop/mobile Playwright、前端测试和后端 smoke 通过。
- [ ] 连续一个发布周期无需要 Classic 回退的高优先级问题。
- [ ] 默认入口切到新应用，保留可诊断错误页。
- [ ] 删除模式选择、Classic bootstrap、旧 DOM、CSS 和运行时。
- [ ] 创建退役前 Git tag，并更新用户与维护文档。

## 每轮工作记录

每轮实施结束必须记录：

1. 完成的矩阵项；
2. 新增或修改的 API 契约；
3. 自动化和真实浏览器证据；
4. 尚未迁移的 Classic 行为；
5. 对用户数据的保护措施；
6. 下一轮唯一优先事项。

### 2026-08-16：训练配置子页面 IA 收口

1. 完成的矩阵项：训练配置分类改为真实单子页面；基础模型页只展示底模、Qwen3
   文本编码器和 VAE 三个路径字段；左侧分组导航在桌面端固定、窄屏端转为横向滚动；
   当前子页通过 `data-config-subpage` 与 hash 路由保持一致。
2. API 契约：未新增后端 API；继续复用 merged config、raw patch 和训练上下文接口。
3. 自动化和浏览器证据：`tests/test_dragon_monitor_system_frontend.py` 定向门禁通过；
   桌面端确认 `#config/training-config/base-models` 仅渲染基础模型字段，切换到
   `output-save` 后基础模型字段消失；390px 视口确认无横向溢出。新增类别直链缺省
   子页的 hash 规范化，并刷新 Dragon UI cache-buster `20260816v46`。
4. 尚未迁移的 Classic 行为：配置保存、恢复默认、训练前检查和启动动作仍由现有
   Dragon 页面逻辑提供，尚未切换到 `web/frontend-next` 的独立实现。
5. 用户数据保护：本轮只读配置并进行浏览器导航验证，没有保存、启动训练、删除或
   覆盖用户文件。
6. 下一轮唯一优先事项：补齐配置字段 catalog 的动态方法/模型族过滤，并在不改变
   子页面 IA 的前提下接入安全的启动训练确认流程。
