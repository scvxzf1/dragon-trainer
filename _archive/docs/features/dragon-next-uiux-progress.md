# Dragon Next UI/UX 实施进度

状态：本计划工程实现、隔离验收及审计修复的历史快照。2026-09-30 退役计划已完成生产切换；本文测试数、截图和验收结论仍为当日历史快照，当前入口为 `/next`，真实生产签收以退役计划记录为准。

依据：2026-09-09 本线程《Dragon Next 前端审阅与优化方向》。保留其 F01-F09、任务概览专项、其他页面方向和完整验收矩阵；本文件记录当前事实，不把计划当完成。

## 范围与执行顺序

| 阶段 | 要求 | 状态 / 证据 |
| --- | --- | --- |
| A1 | F01 统一数值及列表/概览/对比口径 | 共享格式化、历史摘要；排除验证 CMMD，清除验证-only 旧 final_loss 回退但不写回历史文件。数值和完整任务状态矩阵已覆盖 |
| A2 | F02 返回筛选/加载深度/页码/展开/滚动上下文 | `from/depth/anchor`、组内页、展开、滚动恢复；深链接/前进后退/十批上限/删除目标/读取失败/隐藏选择范围 5/5 浏览器用例通过 |
| A3 | F03 搜索范围与服务端过滤/稳定分页 | 服务端 q、keyset cursor、total；10000 条压力用例。其他筛选明确只作用于已读取结果，显示有效条件和隐藏选择数，支持清除；不宣称实现完整服务端 facets |
| A4 | F04 共享样式路由无关 | 双主题同文档监控/打标/设置往返通过；训练配置和数据集改为 workspace 容器断点，保留两层缩放。1280 下全局150%/配置及概览200%路由和键盘用例通过 |
| A5 | F05 监控局部错误、任务身份/新鲜度、停止守卫 | HTTP task-scoped 快照为权威，WS 仅触发合并刷新。独立错误/重试/最近成功读取、停止绑定点击 ID、未知态禁用控制已实现。状态/指标/日志单独失败、旧事件、停止冲突、重连、四视口双主题通过；真实训练不在本轮操作范围 |
| B | 结果解释、预处理分流、配置指纹、来源、紧凑指标、空态、手机命令、长路径 | 成功/异常/中断/运行/预处理成功及失败/epoch/旧记录，8状态×4视口×2主题。缺失字段可展开，结果前置；来源监控身份变更有提示。样张/权重各请求一条元数据，显示总数和最近结果 |
| C1 | 产物总量/分页/排序、manifest、样张结构化详情 | 已实现固定分页、recent/name 权重排序、统一文件可用性、结构化图片详情；125 图片/505 权重的 8 组视口主题通过。配置视图下载入口也改为 manifest，不再凭有 TOML 推断所有文件存在 |
| C2 | 续训完整性/目标区分、错误重试、来源链接 | 实际 integrity.ok/state_complete 契约；目标总步数转为追加步数。确认绑定任务/检查点/目标/模式/快照，刷新或失败后重新确认，未知POST结果不盲重试。策略/组件及临时状态文件测试通过 |
| C3 | 打标跨页对象/草稿、写回范围、手机工作区 | 对象页码/定位、跨页选择、草稿计数、移动视图；保存候选与写TXT分离。跨页两草稿、任务切换取消/确认、精确写回ID、失败保持草稿通过；大图及缩略图失败态、日志局部重试已补 |
| C4 | 配置库/数据集搜索、设置读取失败、队列未知态与窄屏头部、配置工具区 | 库搜索/设置失败重试/队列未知及旧缓存态均覆盖。配置切换、失败保存阻断入队、另存保留草稿、预检失败、成功入队双尺寸通过。队列批量确认绑定revision，刷新竞争409拒绝并重新确认 |
| D | 增强现有对比、可选跨任务总览 | 2-4任务共同步数轴、重叠窗口、独立失败、参数差异、30行分页、加载/截断范围，4视口双主题及canvas像素通过。已选择服务端搜索及manifest，不新增可选跨任务首页 |
| 验收 | 单元/契约、浏览器隔离写入、四视口双主题、路由往返、失效/竞争、性能范围 | 见下方分范围证据；不能把局部通过合并表述为全站完成 |
| 审计 | 独立源码与验收审计、修复后回归、逐项完成核验 | 完成数值/续训/队列/配置/计划缺口的独立只读审计；主代理核验后修复CMMD、旧摘要、缩放、图片及日志反馈。最终回归见下表 |

## 已核验的接口边界

- `GET /api/training/history`：`q` 在服务端匹配任务/集合/配置名及路径；`total` 为搜索结果总量。游标为 `v1` 编码的 `(started_at, id)` keyset，兼容旧数字 offset。非法游标或 limit 返回 400。插入新任务、删除边界任务不重复或跳过旧页尾记录。
- 详情返回 URL 携带列表条件、目标任务和已加载批次数；自动恢复最多连续 10 批，更深位置需点击继续。不会因为 URL 声明很大 depth 就无限自动读取；出错保留重试入口。
- `GET /api/preview/images`、`weights`：任务产物接受 `offset/limit`，返回 `total/next_offset`。权重分页必须用 `recent` 或 `name`；legacy 排序非零 offset 和 group 模式非零 offset 明确拒绝。offset 不承诺文件增删期间的游标稳定性。
- `GET /api/training/history/{task_id}/artifacts`：仅列既有 artifact 白名单，复用输出目录安全解析器，返回 available/missing/blocked/unreadable；仅 stat，不读取大文件元数据，不返回服务器路径。
- 监控 WS 兼容旧后端无 `task_id` 广播，不把广播载荷写入任务状态。750ms 内合并刷新，先确认 status，再更新对应 ID 的指标和日志；轮询继续作为断线回退。
- `POST /api/training/stop`：Next 必须提交非空字符串 `task_id`，并在 launch lock 内重验。错误 JSON、非对象、无 ID 正文、query/body 冲突均返回 400；任务替换返回 409。真正无正文的旧客户端保留旧入口。
- 队列批量取消/中止/清理提交 `expected_revision`，包含队列项目、策略、当前任务及启动中任务身份；有正文却无有效revision返回400，过期返回409，不发生修改。真正无正文保留旧客户端兼容。
- `duration_overrides.max_train_steps` 是追加步数，不是总目标。step42、界面目标200提交158，后端形成目标200；完整性使用 `state_integrity.ok`，schedule-free 不强制要求scheduler。
- Loss窗口按读取到的原始事件限制，验证事件不进入Loss值或平滑；CMMD可独立绘图。没有kind/ev标记的旧事件不凭数值猜测类型；纯摘要且无事件文件的旧记录保持可读。

## 早期验证记录

- 历史/产物/HTTP 定向回归：123 passed，覆盖分页尾页、只解析当前页元数据、manifest 与已有路径安全、稳定游标等。
- 本轮停止请求/WS 既有契约/分页/manifest 组合：52 passed。含真实 aiohttp TestClient 的 8 个 stop 请求解析用例，仅使用 mock service。
- 本轮前端全量单测 27 文件、129/129 通过；相关 API 与刷新 hook 定向 6/6。TypeScript 检查通过。后续导航改动需纳入最终整套回归。
- `state-feedback.spec.ts` + `resilience.spec.ts`：13/13；另补模型库失败/重试 1/1。包括 50000 条日志/指标、20 次路由往返、监控/队列四视口双主题。
- `caption-context.spec.ts`：9/9；候选保存/写回边界既有组件测试 2/2。
- `library-style-contracts.spec.ts`：3/3，包含两种库的组名/空结果搜索以及双主题同文档样式往返。
- `history-return.spec.ts`：3/3；发现并修复连续分页读取旧 Router transition 参数的问题，分页同步提交。测试同时覆盖无 location state 的深链接返回与 10 批自动恢复上限。
- 浏览器测试均设置 `DRAGON_API_TARGET=http://127.0.0.1:29999` 并拦截 API/WS；写入由 fixture 响应，未连接真实训练服务。

## 最终验收对账

| 要求 | 当前实现与验证入口 |
| --- | --- |
| F01 数值可信 | `historySummary`、`metricSemantics`、`MonitorSummary`；`historySummary.test.ts`、`metricSeries.test.ts`、`MonitorSummary.test.tsx`、`test_history_metric_semantics.py` |
| F02 返回连续性 | `historyNavigation`、`useHistoryRestore`；`history-return.spec.ts` 的5个深链/删除/失败/上限场景 |
| F03 搜索和范围 | `training_history_list.py`、`historySelectionScope`；`test_history_cursor_contract.py`、`history-return.spec.ts`、`resilience.spec.ts` |
| F04 样式 | `sharedControls.css`、workspace容器；`library-style-contracts.spec.ts`、`scale-keyboard.spec.ts` |
| F05 监控与动作 | `useMonitorRefresh`、task-scoped API及停止guard；`state-feedback.spec.ts`、`test_live_monitor_task_guards.py` |
| F06 空/错/旧数据 | `QueryFeedback`及manifest状态；`state-feedback.spec.ts`、`overview-states.spec.ts`、`final-audit.spec.ts` |
| F07 窗口和时间 | `metricSeries`真实step/ts/显式索引轴、`MetricsChart`/`LogViewer`范围；`metricSeries.test.ts`、`resilience.spec.ts`，不以截断窗口推算全程峰值 |
| F08 产物与恢复 | `HistoryResultSummary`、`HistoryAssets`、`HistoryArtifacts`、`HistoryResume`；分页/路径安全/状态完整性/target→append测试及`history-overview-assets.spec.ts` |
| F09 打标对象/草稿 | `CaptionReviewContext`、保存与commit边界；`caption-context.spec.ts`、`final-audit.spec.ts` |
| 概览4.1-4.5 | 类型/结束原因/三项核心指标/实际配置/来源/结果/按需检查/折叠诊断；`overview-states.spec.ts`共64种组合及`final-audit.spec.ts` |
| 其他工作区第5节 | 配置保存/另存/预检/入队、库搜索、队列revision、设置错误；`training-draft-flow.spec.ts`、`library-style-contracts.spec.ts`、`queue-scope.spec.ts`、`state-feedback.spec.ts` |
| 视觉与响应式第6节 | 中性双主题、无新增装饰卡片、稳定图表/图片尺寸、缺失字段折叠；`workspace.spec.ts`五视口双主题、四视口概览/产物/打标/对比、缩放及焦点回归 |
| 可选4.6与D | 跨任务首页、详情抽屉、完整服务端facets没有强制引入；已实现可选搜索游标、manifest及增强对比，不复制任务实体 |

2026-09-10 当日最近已完成命令：后端14文件 **230 passed**（仅aiohttp AppKey建议）；前端 **32文件/146 passed**；TypeScript与临时目录构建通过。

- 浏览器整套：**105 passed / 2 skipped**，输出 `/tmp/dragon-next-final-verified`。两项skip沿用原测试标记，没有为了本轮通过而新增跳过。
- 最后产物摘要、独立CMMD及相关监控/概览/返回/缩放回归：44个用例中36项直接通过；8项因StrictMode重挂载导致“恰好两次请求”的断言失败，改为检查两个接口且每次limit=1后，8/8重新通过。输出分别为 `/tmp/dragon-next-final-delta` 和 `/tmp/dragon-next-final-assets`；不是把失败的那次执行表述为全绿。
- 最新源码的稀疏Loss与独立CMMD、概览未知字段展开、来源任务变化、图片404恢复及焦点恢复、日志503重试保持草稿、产物摘要单源失败/未知total，由 `final-audit.spec.ts` 的4项用例通过验证。
- 主代理复看最新1440浅色/390深色概览、390样张弹窗、1280全局150%/配置200%截图；结果数量与最近文件可读，未知字段按需展开，未见整页横向溢出或遮挡。
- 代表合成截图已复制到本线程 `implementation-evidence/final-20260910/`，包含 `overview-mobile-dark.png`、`overview-desktop-light.png`、`image-dialog-mobile-dark.png`、`comparison-mobile-light.png`、`training-scaled-dark.png`。
- 最终构建保留原有哈希资源和前一入口，不清空静态目录；`git diff --check`通过。没有提交或推送，也没有重启真实Python后端。

复现命令：

```bash
rtk proxy pnpm --dir web/frontend-next exec vitest run
rtk proxy pnpm --dir web/frontend-next run typecheck
rtk proxy pnpm --dir web/frontend-next run build
rtk proxy env DRAGON_E2E_PORT=20519 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test
```

只读预览已在 `http://127.0.0.1:20521/next/history/fixture-run` 启动。脚本为 `web/frontend-next/scripts/preview-isolated.mjs`，使用合成fixture而非用户任务，所有API写请求403，无后端代理。独立浏览器检查八个工作区和概览无pageerror、样张实际解码480px、写入拒绝通过。原始记录中的“真实20203仍未启动”已被后续生产切换取代；当前只读证据见退役计划的 `/tmp/dragon-next-20203-final/report.json`。

独立审计明确排除：设置保存仅更新queue策略字段，不会全量覆盖新入队任务；manifest不等于checkpoint完整性；允许“尚未检查”且不自动扫描；原生流式文件下载不改为将大权重读入JS内存。下载过程中外部文件被删除仍可能由浏览器/HTTP报错，应重新检查可用性，不保证检查与下载间文件不变化。

测试边界：真实GPU训练、外部计费打标、真实TXT写回、默认入口切换不在本次操作范围。浏览器可访问性证据是语义、焦点、键盘及几何检查，不声称完成实体读屏器或所有辅助设备人工认证。既有数据集折叠悬停的2项skip不属于本次改造，通过的跨组拖动与键盘用例不能替代它们。

## 安全边界

- 开始时工作树有大量已有修改；`origin/dev` 与 HEAD 相同，fetch 后比较为 `0 0`。未 reset/rebase/切分支/提交/推送。
- 不启动真实队列后端，不恢复/停止训练，不调用真实打标服务或下载模型。
- 前端验证使用 API/WS mocks；后端使用临时目录及 service fixture。启动的开发服务仅提供前端，不能隐式代理到真实数据执行写操作。
- 不删除旧入口，不迁移用户文件。新的列表和文件接口必须保持旧客户端兼容、路径白名单和输出根边界。
