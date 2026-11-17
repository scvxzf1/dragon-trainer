# Dragon UI 与 classic UI 历史记录

状态：历史 / 已退役
适用范围：旧版静态前端迁移记录，不适用于当前 WebUI 操作

## 当前入口

当前 WebUI 使用 Next 工作台。访问服务根路径 `/` 会跳转到 `/next`；旧 `/?ui=dragon` 和
`/?ui=classic` 地址会兼容重定向到 Next，不再选择或加载旧界面。用户操作、开发和验证入口见
[Next 前端 README](../../web/frontend-next/README.md)。

当前兼容重定向不等于旧静态源码或历史资源已删除，也不证明任何既有部署端口已经切换。
本页只保留旧界面的背景事实，不提供启动、切换、回退或故障排查步骤。Next 自身故障时使用
Next 错误页提供的恢复入口；不要将旧 Dragon/classic 当作可用回退路径。

## 历史背景

旧 Dragon 与 classic 曾是共用 aiohttp 后端、配置、训练队列、历史任务、模型路径和输出目录的
两套前端壳。旧页面的 `Dragon trainer` 是界面品牌，并非模型族；训练配置中的
`model_family` 由模型族 registry 管理。旧 Dragon 曾提供外部 API 打标工作台，而 classic
没有对应专用页面。这些描述只记录迁移前的产品结构，不能据此推断当前旧前端仍可访问。

迁移期间，`/` 曾根据浏览器偏好在旧界面间选择，Dragon 初始化失败时也曾尝试加载 classic。
现行路由已改为将根路径和历史 `ui` 参数统一导向 `/next`。旧哈希资源或历史入口副本可能仍
作为静态发布回退材料留存；它们不是可选用户入口。部署和静态资源发布细节见
[Next 前端 README](../../web/frontend-next/README.md)。

## 相关历史材料

- [旧 Dragon 功能图谱](dragon-ui-functional-map.md)
- [Dragon Next 实施记录](dragon-next-implementation.md)
- [前端性能记录](../findings/dragon_frontend_performance_20260826.md)
- [Dragon 响应式硬编码审计](../findings/dragon_responsive_hardcoding_audit_20260824.md)
