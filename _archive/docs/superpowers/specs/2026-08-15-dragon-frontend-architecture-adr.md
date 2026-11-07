# ADR：Dragon 单前端技术架构

状态：方向已确认 / toolchain spike 待实施<br>
日期：2026-08-15<br>
关联总规格：[`2026-08-15-dragon-single-frontend-rebuild-spec.md`](2026-08-15-dragon-single-frontend-rebuild-spec.md)

## 决策

新前端采用 TypeScript 单页应用，建议基线为：

- React + TypeScript + Vite
- TanStack Query 管理服务端状态、缓存、轮询和请求失效
- React Hook Form + Zod 管理表单草稿、字段校验和提交边界
- Zustand 仅管理少量真正跨路由的客户端状态
- React Router 管理页面级路由；配置长页只保留分类级路由
- CSS Modules + Dragon design tokens 管理样式作用域
- Vitest + Testing Library + Playwright 建立行为测试层级
- `@dnd-kit` 实现可访问的排序、跨组移动和键盘拖动

版本在 toolchain spike 时根据仓库可安装依赖锁定，本 ADR 不凭文档写死未经验证的版本号。

## 理由

当前 Classic 已从巨型入口拆分，但业务仍依赖全局 DOM ID、runtime state、bridge 和
副作用 chunk 图。当前 Dragon 的页面 loader 边界更清晰，但仍使用字符串 renderer、
`innerHTML + onMount`，且数据集分阶段调度仍动态加载旧 bridge。

继续在两套 vanilla 运行时上补功能，会延长双维护期。新技术栈应提供：

- 可组合的表单和字段校验；
- 明确的服务端状态与客户端草稿边界；
- 页面卸载和异步请求取消能力；
- 可测试的拖动、对话框和 dirty-state 行为；
- 类型化 API，减少页面对响应结构的猜测。

## 状态所有权

| 状态 | 所有者 |
| --- | --- |
| API 列表、详情、轮询结果 | TanStack Query |
| 当前表单草稿、dirty、字段错误 | React Hook Form |
| 对话框开关、当前选中行 | feature 局部组件 |
| 当前训练配置身份、全局 UI 偏好 | 小型 Zustand store |
| URL 可恢复的页面状态 | Router |
| 训练实时流 | 独立 WebSocket client + query/store adapter |

禁止把 API 返回、表单草稿、WebSocket 状态和 UI 开关混入一个全局对象。

## 路由决策

- 页面级能力拥有路由，例如 `/datasets`、`/training`、`/history/:taskId`。
- 配置分类拥有路由，例如 `/config/memory-optimization`。
- 同一长页中的滚动章节不拥有伪页面路由。
- 筛选、搜索和选中项只有在刷新恢复确有价值时才进入 query string。
- aiohttp 需要提供 SPA fallback；在 fallback 完成前可使用 Hash Router 作为过渡。

## API 边界

- 页面不得直接调用裸 `fetch`。
- 每个领域提供 query/mutation functions 和显式 request/response 类型。
- HTTP 非 2xx、`ok=false` 业务 envelope、取消和网络错误必须区分。
- WebSocket 重连、订阅和卸载清理由独立 client 管理，不使用模块级页面 singleton。
- 后端契约缺失或不一致时先补契约测试，不在 UI 内静默兼容多个猜测形状。

## 样式边界

- 复用 Dragon 的颜色、排版、密度和控件语言，不复制现有大体量 CSS 文件。
- design tokens 是唯一跨 feature 样式依赖。
- feature 样式默认局部作用域，不允许 Classic 式 unscoped semantic selector。
- 运维工具保持高信息密度，避免把工作流拆成装饰性卡片或营销页面。

## 被否决方案

1. 继续扩展当前 Dragon vanilla 页面：无法彻底摆脱字符串契约和旧 bridge。
2. 将 Classic chunks 搬入 Dragon：只会转移技术债，不会形成新架构。
3. 同时重写前后端：扩大风险面，妨碍功能一致性验证。
4. 一次性 big-bang 替换：缺少逐功能证据和可回退节点。

## Toolchain Spike 验收

正式迁移功能前，最小应用壳必须证明：

- Vite 构建产物可由当前 aiohttp 静态服务加载。
- 开发环境可代理现有 `/api` 和 `/ws`。
- 一个示例 query、mutation、表单和路由能够运行。
- Playwright 能启动 WebUI 并执行真实浏览器 smoke。
- 构建产物不会无条件下载 Classic 资源。
