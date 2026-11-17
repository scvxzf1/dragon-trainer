# Dragon Next 阶段性审计与功能验证计划（2026-09-24）

状态：进行中；S0 PASS、S1 进行中、S2 PASS、S3 PASS、S4 PASS、S5 BLOCKED；非发布验收

适用对象：`/next/` React 工作台；不评价 Classic/旧 Dragon 的完整功能

现场：`http://127.0.0.1:20203/next/`，`dev @ 5b60d111`，工作树含大量未提交修改

原则：本文的“当前”只指 2026-09-24 的源码和本机已发布静态包；二者尚未做逐资源同一性证明。

## 结论先行

Dragon Next 已形成八个主工作区、数据集图片子工作区和独立蒙版路由，复用现有 aiohttp API。当前适合继续做**隔离验收**，不能凭组件测试或只读浏览器巡检批准替换默认入口。辅助生图、权重分析、环境工具仍通过旧界面；流水线并行在主训练入口不可用。这些是明确的产品边界，不当作本轮新发现缺陷。

本轮没有确认产品 P0 缺陷。曾确认的**验证基础设施阻塞**（V-01）已在 S0 复验关闭：`scripts/verify-production.mjs` 先切换到训练计划阶段后退出码为 0，并生成完整八页 JSON 报告。八个主路由另经浏览器逐页只读打开，均出现预期 H1，1280px 默认视口下未见整页横向溢出；这只证明页面入口与初步渲染，不证明数据准确、写入成功或复杂交互可用。

| 本轮证据 | 结果 | 证据边界 |
| --- | --- | --- |
| 前端静态/单元检查（2026-09-27 最新复验） | TypeScript 通过；串行 Vitest **61 文件 / 256 用例通过**；文档完整性 8/8 | 组件、纯逻辑、mock 请求；不连接真实服务 |
| 完整 mock API/WS Playwright | **r115：303 项，301 passed、2 skipped、0 failed** | 当前树完整套件为不可达 API target + route mocks，不代表真实 API/WS 集成 |
| S1 状态矩阵登记 | **70 个 workspace/state 单元格，snapshot=r115，机器校验通过；overall_state=NOT_RUN** | [dragon_next_s1_state_matrix_r93.json](dragon_next_s1_state_matrix_r93.json) 文件名沿用，仅登记 mock/人工/未运行边界，不是 S1 签收 |
| 八主路由只读巡检 | `/training`、`/datasets`、`/queue`、`/monitor`、`/history`、`/models`、`/captioning`、`/settings` 均显示标题；默认桌面视口无 document 级横向溢出 | 未触发保存、启动、入队、取消、停止、删除、下载或 TXT 写回 |
| 手机抽查 | 390 x 844 的 `/settings` 无 document 级横向溢出 | 只覆盖一个页面、一个主题和首屏，不代表响应式矩阵通过 |
| 生产只读脚本 | **PASS**：S0 复验退出码 0，生成完整八页 JSON 报告 | `dragon_next_s0_baseline_20260924.md` 记录训练计划阶段就绪修复、四视口 32 次加载及完整报告路径 |
| 真实写入与训练 | **NOT RUN** | 用户配置、队列、历史、输出、模型和训练进程均保持原样 |

训练页现场曾显示“数据并行至少需要选择两张 GPU”提示；当前浏览器选择了数据并行但未选卡，属于可见的提交前校验状态，不能据此认定 GPU 选择功能故障。旧审计的配置库键盘排序、ESM token、帮助摘要等问题已有 [2026-09-13 审计收尾](uncommitted_audit_closure_20260913.md)记载修复；本轮不把修复前的 [旧问题清单](webui_frontend_audit_20260913.md)重新列为当前缺陷，后续仍需按现工作树复验。

## 范围与系统地图

`src/main.tsx` 装配 React；`src/app/DragonNextApp.tsx` 提供 QueryClient 与 Router；`src/app/router.tsx` 使用 `/next` basename、lazy 页面与加载态；`src/app/AppShell.tsx`/`Topbar.tsx` 管主导航、主题、移动导航和旧工具出口。服务端快照主要由 TanStack Query 管理，训练配置/硬件预设选择由 `trainingContextStore.ts` 持久化；表单局部状态与后端草稿分离。API 统一由 `src/api/client.ts::apiRequest` 发出，监控同时使用 HTTP 轮询和 `/ws/training` 刷新。生产路径由 Python 服务托管静态文件，不要求常驻 Node 服务。参见 [实施记录（已归档）](../../_archive/docs/features/dragon-next-implementation.md)和 [前端 README](../../web/frontend-next/README.md)。

| 工作区与路由 | 当前能力边界 | 审计重点 | 真实副作用 |
| --- | --- | --- | --- |
| 训练配置 `/training` | 配置库、四阶段字段、数据集选择、模型组合、估算、样张提示词、TOML、预检、启动、入队 | 来源/继承值、脏草稿、另存、未知键、兼容门禁、设备身份、保存后再执行 | 保存 TOML；启动/入队可触发预处理和训练 |
| 数据集蓝图 `/datasets` | 分组/预设 CRUD、子集规则、导入导出、图片预览及工作台入口 | 跨组排序、未保存离开、共享蓝图引用、图片计数口径、空目录 | 写数据集 TOML 与组元数据 |
| 图片工作台 `/datasets/workspace/{preview,masks,tagging}`；兼容 `/datasets/masks` | 同一 dataset/subset 上下文的预览、蒙版、打标 | 深链/返回、视图切换、脏蒙版保护、图片解码、权限与路径 | 蒙版 PNG、子集配置、候选标注/TXT 写回 |
| 训练队列 `/queue` | 状态筛选、排序、暂停/继续、重试、取消、批量控制、失败策略 | revision 竞争、目标范围、未知结果、重复提交、空队列 | 修改真实队列；部分操作影响运行进程 |
| 当前监控 `/monitor` | 当前任务、指标/日志/GPU、WS 刷新、停止 | 任务身份、新鲜度、局部失败、断线回退、图表轴与日志上限 | 停止真实训练 |
| 历史 `/history`、`/history/:taskId` | 搜索/集合/归档、详情/指标/产物/日志、对比、续训 | 游标与范围、返回位置、产物缺失、checkpoint 完整性、续训追加步数 | 归档/删除历史；续训可启动训练 |
| 模型配置 `/models` | 模型组合与分组编辑、路径、默认项、revision | 409 冲突、默认保护、空库、排序、路径展示 | 写模型配置 |
| 打标 `/captioning/*` | 图片/任务、provider、提示词、本地资源、候选审阅、翻译、写回 | 草稿与 commit 分离、跨页选择、密钥不回显、失败/重试、取消 | 可调用外部服务、下载资产、写 caption |
| 全局设置 `/settings` | 输出与配置根、保留数、生图选项、主题/缩放 | 根目录变化、查询失效、错误恢复、缩放与可访问性 | 写设置并重载运行时存储根 |

此表是**功能清单和待验范围**，不是逐项通过结论。源码锚点：`src/app/router.tsx:74-117`、`src/features/training-config/api.ts:51-153`、`src/features/dataset-editor/api.ts:28-132`、`src/features/training-queue/api.ts:66-120`、`src/features/training-history/api.ts:99-217`、`src/features/captioning/api.ts:4-174`，路径均相对 `web/frontend-next/`；后端路由见 `web/routes/{config,training,settings,tagging,mask_editor}.py`。

## 审计判断与风险登记

| ID | 判定 | 影响与下一证据 | 优先级 |
| --- | --- | --- | --- |
| V-01 | **已修复并关闭**：生产只读脚本已先切换训练计划阶段，再检查“输出名称”字段；S0 复验退出码 0 且生成完整八页 JSON。 | 证据见 [S0 只读基线](dragon_next_s0_baseline_20260924.md)。后续仍需把真实写入与训练验收保持为独立阶段，不得由只读脚本推断。 | G1 已关闭（S0 scoped） |
| V-02 | **S0 基线比对 PASS；本轮后当前工作树与现场静态包不再是同一版本**。本轮 AppShell/dirty guard/history CSS 只改源码；隔离构建产物留在 `/tmp`。 | 如需验收本轮改动，使用隔离服务/静态根建立新哈希证据；未获发布授权前不运行会替换 `web/static/dragon-next/` 的构建入口。S0 截图仍代表原基线包。 | G1 当前源码溯源 |
| V-03 | **未证：真实前后端 mutation 链路**。E2E 拦截 `/api/**` 和 WS；生产脚本拦截非 GET。 | 在临时配置根、历史根、队列根和输出根验证保存/冲突/回滚；绝不以用户当前目录做写入实验。 | P1 |
| V-04 | **设计边界：默认入口未切换**，辅助工具跳旧 UI；流水线并行被阻止。 | 验证链接、回退和禁用提示准确；不把缺失的原生工具或未接入执行模式算作已验收。 | 发布条件 |
| V-05 | **S2 已修复并隔离验收**：Next 的训练 TOML、数据集预设、样张 TXT、模型库与全局设置保存携带读取 revision；旧版本返回 409，保留草稿且不盲重试。 | 同一路径比较与原子写入在文件锁内；两标签冲突、断网结果未知和刷新恢复证据见 [S2 记录](dragon_next_s2_isolation_20260925.md)。不带 revision 的旧客户端仍保留兼容写入能力，不视为 Next 并发保护。 | P1 已关闭（Next 范围） |
| V-06 | **已关闭（S4 scoped）**：图片工作台 URL 上下文、返回滚动、蒙版脏态与打标 writeback 均有当前工作树的隔离测试；复核修正蒙版切子集后 URL 滞留及非法 apply JSON 写入风险。 | 临时 aiohttp + 当前 Vite 源码的浏览器探针已验证真实蒙版 HTTP 写入、重载、409、三视图与手机布局；caption 仍由本地 provider stub 后端测试和 mock 浏览器测试覆盖，未调用真实 provider。详见 [S4 报告](dragon_next_s4_image_mask_caption_20260925.md)。 | P1 已关闭（仅 S4） |
| V-07 | **已确认并修复：skip-link 原生 hash entry 可绕过 dirty POP guard**。隔离 fixture 中脏训练页按一次 Back 直接回到此前 `/queue`，未出现确认且丢失当前草稿。 | 改用 Router 管理 hash 导航；训练与数据集 dirty guard 仅放行 path/search 不变的 hash-only 导航。mock E2E 验证 hash 返回、离页阻止、草稿保留。尚未发布静态包。 | P1 已修复 |
| V-08 | **已确认并修复：360/390px history 搜索框分别只有 24/54px**，无法有效检索。 | 600px 以下搜索项独占工具栏一行；隔离 E2E 检查两种视口宽度均 >=240px 且页面无横向溢出。尚未发布静态包。 | P2 已修复 |

优先级约定：**P0** 为越界读写、错误目标停止/删除、泄密或无确认启动训练；**P1** 为主流程阻断、数据丢失或状态误导；**P2** 为可恢复的局部功能/响应式问题；**G1** 为验收证据无法产生。新发现需附复现步骤、输入/环境、请求与响应、截图或 trace、源码位置、影响范围，不从“测试失败”直接推断产品根因。

## 分阶段执行与退出标准

### S0 版本冻结与只读基线

1. 记录 `git status --short --branch`、HEAD、构建时间/资产清单、服务 URL、浏览器与视口；区分源码工作树、构建产物和真实服务数据。
2. 修复 V-01 验证脚本就绪断言；确保所有非 GET/HEAD/OPTIONS `/api/**` 请求继续被拦截，并记录被拦截路径。为每页保存状态码、标题、页面错误、静态资源错误、加载耗时和 document 溢出。
3. 在 1440 x 900、1280 x 720、390 x 844、360 x 800 至少覆盖八主路由；明暗主题、125%/150%/200% 缩放针对工作区代表页。截图只保存在隔离目录，不提交或外发包含真实路径/任务信息的截图。

**退出**：八页和关键深链能直接打开/刷新；无未解释 JS 错误、404 或整页溢出；生产只读脚本完整产出 JSON；源码与被验静态包关联可追溯。当前状态：**PASS（S0 scoped）**（见 [S0 只读基线](dragon_next_s0_baseline_20260924.md)）；下一步进入 S1。

### S1 壳、导航与状态语义

核对主导航、辅助旧入口、浏览器前进/后退、深链、刷新、主题、移动菜单、键盘焦点和跳过链接。每页至少检查加载、空、错误、重试、只读、dirty、busy、成功、409 冲突与断网后结果未知；加载骨架必须最终收敛到可解释状态。测试长配置名/路径、中文、窄屏、放大、`prefers-reduced-motion`、Tab/Esc 与弹窗焦点恢复。监控图表和历史图片需验证非空像素、图例/轴、失败占位及布局稳定性。

**退出**：无导航死路、不可关闭弹窗、不可达主要命令或状态误导；代表视口无文字/控件相互遮挡；自动化截图与人工核验各保留证据。现有 Playwright 视觉/交互用例可作为回归，不代替真实读屏器验收。

历史快照（2026-09-25，r15/r16）：**S1 NOT RUN**（S1 整体仍未达到签收条件）。壳层导航、dirty skip-link、reduced-motion、数据集保存 409 后保留草稿并要求显式 reload 获取新 revision、历史错误重试/空态与续训冲突恢复、移动 provider 弹窗焦点、lazy import 失败恢复，以及设置/模型冲突、caption 候选与 TXT commit 断网恢复、provider ping 502 错误可见/无自动重试/显式重试、训练入队未知结果与立即启动 500/断网状态、队列单项 move 与调度策略 mutation（pending/成功/500/断网/显式重试/继续确认取消）、单项取消遇 404 后保留失败时快照并在显式刷新成功后清除旧错误、监控停止与无任务/未知状态、GPU 503 错误和显式重试恢复、WebSocket hook 连续 close 的 1/2/4 秒退避/成功连接复位/卸载清理、监控图表与 canvas/控制项在 1440/1280/768/390 视口及 dark/light 的边界检查、蒙版 dirty 离页及切图/切子集保护、保存 409 重试、应用到子集 busy/409/500/断网后的显式重试、图片工作台缺 dataset 深链空态，以及数据集预设库 503 重试至确认空态已有 mock E2E/unit 证据。当时最新完整 mock API/WS Playwright suite 为 **175 项：173 passed、2 skipped、0 failed**；TypeScript 通过，串行 Vitest **50 文件 / 223 用例通过**。provider 失败恢复用例在 r15 完整套件之后新增，`interactions.spec.ts` 定向 **9/9 passed**，因此该用例不在 175 项全量计数内。390px dark/light 监控截图在 `/tmp/dragon-next-e2e-s1-20260925-r15/` 已人工抽查，未见图表、控件、日志与设备信息遮挡/裁切。训练启动后端 500 可能发生在子进程创建之后；当时 UI 阻止同弹窗重复提交，失败态提供当前监控核对链接，相关路径仅完成 mock 状态验证。后端 batch-start partial enqueue 有 `queued_count`/`failures` 响应，但 Next 无调用入口，属于当时前端审计范围外；蒙版 apply 以单次配置写入为成功边界，没有逐子集 partial-result 响应契约。真实 launcher 状态核对、完整状态矩阵、其余主题/缩放人工视觉核验和真实读屏器验收尚未完成；真实写入继续 `NOT RUN`，V-07/V-08 仍只修复在未发布源码；后续状态以本文件及 [S1 壳与导航审计进度](dragon_next_s1_shell_audit_20260925.md) 的更新为准。

本轮续审再覆盖 provider 测试取消确认零请求、502 超时错误/无自动重试/显式重试，以及模型默认项保护与切换、保存 busy/成功和 PUT 响应丢失后的服务器对账。模型结果未知时新增“核对服务器版本”并禁用盲重发；断网 mock 模拟服务端已提交后断开响应，再确认 GET 获取新 revision。另修正通用 dataset preview mock，补齐后端/TS 契约要求的 `row/settings`，有效预览与完整蒙版 spec **9/9 passed**。`interactions.spec.ts` 9/9、`state-feedback.spec.ts` 17/17、模型锁定复核 3/3、TypeScript 与串行 Vitest 50 文件/223 用例通过。全量 r16 收集 178 项后被外层 600 秒超时中止于 174/178，两个当时的等待失败各自 single-worker 重跑 1/1；因此最近一次完整通过仍是补测前 r15 的 175 项结果，r16 不能记作 PASS。S1 整体仍为 **NOT RUN**。

### S1 续审更新（2026-09-25）

以下结果覆盖本节前文的 r15 / 50 文件、223 用例快照；那些数字保留为当时的历史记录。当前最新完整 mock API/WS Playwright 为 r17：184 项中 182 passed、2 skipped、0 failed，单 worker 13.4 分钟，报告和 trace 在 `/tmp/dragon-next-e2e-s1-20260925-r17/`。两个 skip 是 `dataset-drag.spec.ts` 中已有的折叠分组拖动场景；没有连接真实 API/WS。

继续补测 caption rerun pending 与 409 后重读服务器 job 状态，修复旧终态快照仍暴露过期操作的问题。`caption-context.spec.ts` 15/15、`CaptionReview.test.tsx` 2/2 passed。队列单项 retry 确认/pending/成功、运行项 stop 的目标 ID 与 `delete_runtime:false`、已完成项 remove 仅移出队列列表均有 mock E2E 证据，`queue-monitor-mutations.spec.ts` 9/9 passed。最新前端门禁为 TypeScript 通过、串行 Vitest 50 文件/225 用例通过、`git diff --check` 通过。

S1 仍为 **NOT RUN**：数据集 CRUD/import-export、模型排序/长路径、历史 archive/delete 与 blocked/unreadable artifact、队列命令网络中断和 stale 竞争、真实读屏器顺序、剩余主题/视口人工视觉复核及真实 launcher/监控核验均未关闭。mock 成功不能替代真实服务写入或人工验收。细项见 [S1 壳与导航审计进度](dragon_next_s1_shell_audit_20260925.md)。

### S1 续审更新（2026-09-25，r18）

本更新覆盖前段 r17 最新完整结果；r17 留作历史记录。S1 仍为 **NOT RUN，未签收**。

新增队列 retry 响应丢失后的手动刷新对账、历史 archive/unarchive/delete 的目标与删除范围 mock E2E，以及数据集 TOML import→browser download export round-trip。历史 delete 已在服务端提交但响应丢失时，UI 原先会保留过期错误和失效选择；修复后显式刷新成功会清错误、移除不在已加载快照中的 ID，并提示当前列表已核对。训练/配置 launcher、队列服务和历史文件均未真实访问。

TypeScript 通过，串行 Vitest **50 文件 / 225 用例通过**，`rtk git diff --check` 通过。完整 mock API/WS Playwright r18 为 **189 项：186 passed、2 skipped、1 failed**，输出在 `/tmp/dragon-next-e2e-s1-20260925-r18/`。唯一失败为 `training-library-names.spec.ts` 的 1440 视口用例等待重命名按钮 45 秒；同 spec 单 worker 定向重跑 1440/390 为 **2/2 passed**，输出在 `/tmp/dragon-next-e2e-s1-library-names-r1/`。因此定向超时未复现，但 r18 全量不能记录为全绿，仍需保留为未确认波动。

新增 `history-actions.spec.ts` **2/2 passed**、`dataset-import-export.spec.ts` **1/1 passed**；队列 mutation spec 中新增 retry lost-response reconciliation 并在定向回归中通过。mock 通过不代表真实写入。

S1 未签收原因未变：读屏器顺序、剩余主题/缩放人工核验、数据集其他 CRUD/失败矩阵、队列 stop/remove 失败与结果未知、模型排序/长路径、损坏 checkpoint、真实 launcher/服务联通均未关闭。S2/S4 的隔离测试不代替这些 S1 人工和状态语义核验；真实写入、训练、provider 与发布静态包继续 **NOT RUN**。细项见 [S1 壳与导航审计进度](dragon_next_s1_shell_audit_20260925.md)。

### S1 续审更新（2026-09-25，r19）

本次完整运行使用当前工作树，分支 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`。工作树保留大量既存修改；运行期间新增的训练配置库详细管理改动也已纳入当前测试，没有清理、回滚或提交其他修改。

最新完整 mock API/WS Playwright 为 **190 项：188 passed、2 skipped、0 failed**，单 worker 13.1 分钟；输出在 `/tmp/dragon-next-e2e-s1-20260925-r19/`。2 项 skip 为既有折叠分组拖动场景。该运行包含新增 history、queue、dataset 用例和训练库详细管理测试。

当前训练库相关两份 E2E 定向 **6/6 passed**；TypeScript 通过；串行 Vitest **50 文件 / 226 用例通过**；`rtk git diff --check` 通过。r18 文档编辑后的 documentation integrity **8/8 passed**。r18 曾有一条训练库名称用例超时，单测及 r19 当前树全量均通过；保留 r18 记录，不把其结果改写成全绿。

S1 仍为 **NOT RUN，未签收**：真实读屏器朗读与 live-region 顺序、剩余主题/真实浏览器缩放人工核验、数据集其他 CRUD/失败矩阵、队列 stop/remove 失败与结果未知、模型排序/长路径、损坏 checkpoint、真实 launcher/服务联通均未关闭。上述均为 mock 或隔离证据；真实写入、训练、provider 与发布静态包继续 **NOT RUN**。细项见 [S1 壳与导航审计进度](dragon_next_s1_shell_audit_20260925.md)。

### S1 续审更新（2026-09-25，r20）

本轮基于当前工作树继续 mock-only 核验；分支 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`，保留原有大量未提交修改，没有清理、回滚或提交。

- `queue-monitor-mutations.spec.ts` 新增运行中 stop 的响应丢失场景：服务端先变为 canceled 再 abort DELETE；UI 保留旧 running 快照、提示结果待确认、不自动重试，显式刷新后收敛并清除错误。与 damaged checkpoint 浏览器链路一起，queue/history 两份定向 spec **14/14 passed**。
- `s3-queue-monitor-history.spec.ts` 新增不完整 optimizer checkpoint E2E：展示不可恢复原因、禁用确认续训，并确认无续训 POST。
- 新增 `dataset-crud-mutations.spec.ts`，覆盖新建草稿、保存、另存、重命名、删除；另存确认同步双击仅发一次请求，删除 409 时保留已有预设并允许显式重试。没有确认产品缺陷。
- 新增 `model-library-order-path.spec.ts`，验证长路径完整值、模型上移/下移边界 no-op，以及当前 DOM 顺序与 PUT `item_ids` 一致。
- 四份相关 E2E 合并定向 **16/16 passed**；完整 mock API/WS Playwright r20 **195 项：193 passed、2 skipped、0 failed**，单 worker 13.1 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r20/`。2 项 skip 为既有 dataset-drag 折叠组拖动用例；未连接真实 API/WS。预期 lazy import 故障注入产生 Vite console error，由路由错误边界处理。
- TypeScript 通过；串行 Vitest **50 文件 / 226 用例通过**；`rtk git diff --check` 通过；本次计划/进度文档编辑后文档完整性 **8/8 passed**。

S1 仍为 **NOT RUN / 未签收**：真实读屏器与 live-region 朗读顺序、剩余主题及真实浏览器缩放人工核验、逐页完整状态矩阵、队列其他 stop/remove/retry 失败及未知结果、数据集另存/重命名 offline/409/unknown 矩阵、caption job cancel/rerun 失败矩阵、真实 launcher/服务联通仍未关闭。真实写入、训练、provider 调用和发布静态包继续 `NOT RUN`；mock 不替代人工或隔离 HTTP 验收。

### S1 续审更新（2026-09-25，r21）

`queue-monitor-mutations.spec.ts` 新增 completed remove 响应丢失后的对账路径：服务端先移除再 abort DELETE，UI 保留旧快照并提示结果未知；显式刷新后列表收敛、错误清除且无重复请求，定向 **1/1 passed**。新增项纳入当前树完整 mock API/WS Playwright r21：**196 项，194 passed、2 skipped、0 failed**，单 worker 13.1 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r21/`。2 项 skip 是既有 dataset-drag 折叠组拖动场景。TypeScript 通过；串行 Vitest **50 文件 / 226 用例**沿用 r20 结果，本轮只增加 E2E 测试；`rtk git diff --check` 及文档完整性 **8/8 passed**；实际 API/WS、训练 launcher、写盘、provider 与发布仍未接入或授权。

S1 仍为 **NOT RUN / 未签收**：队列 retry/stop 的 404/5xx/断网失败矩阵、数据集另存/重命名失败矩阵、caption job cancel/rerun 失败矩阵、逐页状态矩阵、真实读屏器/live-region 与浏览器缩放人工核验、launcher/服务联通均待完成。mock 不能替代人工或隔离 HTTP 证据。

### S1 续审更新（2026-09-25，r22）

- `caption-context.spec.ts` 新增运行中打标任务取消：要求确认；请求 pending 时锁定取消、重跑和项目选择；成功响应后页面收敛到 `canceled` 并恢复终态控件。请求方法、目标路径、请求次数均有断言；未调用 provider 或写入图片/TXT。整份 spec **16/16 passed**（`/tmp/dragon-next-e2e-s1-caption-context-r22/`），新增单项 **1/1 passed**（`/tmp/dragon-next-e2e-s1-caption-cancel-r1/`）。
- 当前树完整 mock API/WS Playwright r22：**197 项，195 passed、2 skipped、0 failed**，单 worker 13.2 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r22/`。2 项 skip 仍是 dataset-drag 中既有折叠分组拖动；未连接真实 API/WS。
- TypeScript 通过；本轮只新增 E2E，Vitest 沿用 r20 的 **50 文件 / 226 用例**结果；文档完整性 **8/8 passed**，`rtk git diff --check` 通过。

S1 仍为 **NOT RUN / 未签收**：caption cancel 409/结果未知、rerun 500/断网核对、队列 retry/stop 其他失败结果、数据集另存/重命名冲突与离线矩阵、逐页完整状态矩阵、真实读屏器和浏览器缩放人工核验、launcher/服务联通仍开放。真实写入、训练、provider 调用和发布静态包继续 `NOT RUN`；mock 通过不替代人工或隔离 HTTP 证据。

### S1 续审更新（2026-09-25，r23）

- `dataset-crud-mutations.spec.ts` 增加 save-as 409 竞态：mock 服务端在请求到达时被另一窗口占用目标名；页面保留当前预设与未保存表单值，错误可见，用户重新打开另存并成功保存同一草稿。`queue-monitor-mutations.spec.ts` 增加 retry 503：pending 锁定、保留 error 快照、显示服务端错误、解除锁且无自动重试。两份相关 E2E 定向 **14/14 passed**（`/tmp/dragon-next-e2e-s1-failure-matrix-r1/`）；未发现需改产品逻辑的问题。
- 当前树完整 mock API/WS Playwright r23：**198 项，196 passed、2 skipped、0 failed**，单 worker 13.1 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r23/`。2 项 skip 为既有折叠分组拖动；未连接真实 API/WS。TypeScript 通过；Vitest 沿用 r20 的 **50 文件 / 226 用例**结果，本轮只改 E2E 与审计记录。
- 文档完整性 **8/8 passed**，`rtk git diff --check` 通过。真实数据集/队列服务、训练、provider 和静态包发布均未接入或授权。

S1 仍为 **NOT RUN / 未签收**：队列 retry 404、running stop 的 404/5xx 服务错误，数据集 rename 的 save-as 冲突/断网结果未知、caption cancel 409/未知结果与 rerun 失败，逐页完整状态矩阵、真实读屏器/live-region 与浏览器缩放人工核验，以及 launcher/服务联通仍未关闭。

### S1 续审更新（2026-09-25，r24）

- 数据集重命名增加旧文件删除 409 部分成功覆盖：新文件成为当前选择、旧/新文件都保留，表单从已保存的新预设 hydrate，并清楚报告部分失败。caption 取消遇 409 后重取权威 job 快照，页面从 running 收敛到 done、取消禁用、重跑可用。相关 caption/dataset/queue E2E **31/31 passed**（`/tmp/dragon-next-e2e-s1-mutation-matrix-r1/`）。没有复现需修产品逻辑的问题。
- 当前树完整 mock API/WS Playwright r24：**199 项，197 passed、2 skipped、0 failed**，单 worker 13.2 分钟，报告 `/tmp/dragon-next-e2e-s1-20260925-r24/`；2 项 skip 为既有折叠分组拖动。TypeScript 通过；Vitest 沿用 r20 的 **50 文件 / 226 用例**结果，本轮只变更 E2E 和审计文档。
- 文档完整性与 `rtk git diff --check` 在本次更新后复验结果见 S1 进度报告；真实服务、写入、provider、训练和静态包发布未接入或授权。

S1 仍为 **NOT RUN / 未签收**：队列、数据集和 caption 的上述 mock 失败路径已补测；八页完整状态矩阵、真实读屏器/live-region 朗读顺序、剩余视口/主题/真实浏览器缩放人工检查及 launcher/服务核对仍未关闭。

### S1 续审更新（2026-09-25，r25）

本轮仍只使用 Playwright mock API/WS。队列 retry 404 与 running stop 404/503 均验证旧快照、错误反馈、pending 释放、无自动重试和显式刷新后的快照收敛；数据集覆盖重命名首个 save-as 409、已写入但响应丢失、另存离线后保留 dirty 草稿并由用户重试。caption 覆盖 rerun 500 与 cancel/rerun 未知结果；修复 rerun 响应丢失后可重复提交的风险：任务状态未知期间锁住取消、重跑、候选选择、编辑和写回，提供显式状态核对，成功读取服务器快照后才解锁。

当前完整 mock API/WS Playwright 为 **208 项，206 passed、2 skipped、0 failed**，单 worker 13.5 分钟，输出 `/tmp/dragon-next-e2e-s1-20260925-r25/`；两项 skip 是既有折叠分组拖动。caption 未知结果/取消定向 3/3 passed。`web-next-check` 通过（TypeScript、Vitest 50 文件/226 用例）；文档完整性 8/8 passed，`rtk git diff --check` 通过。工作树仍在 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`，ahead 13，存在大量既存修改；未清理、回滚、暂存或提交，未替换静态包。

S1 仍为 **NOT RUN / 未签收**：完整八页状态矩阵与人工读屏器、live-region、主题/视口及真实浏览器缩放核验尚未完成；mock 不证明 launcher、真实服务或隔离持久态。真实写入、provider、训练与静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-25，r26）

训练启动断网会由 `apiRequest` 转为 `ApiError(status=0)`；此前启动弹窗没有把 status 0 纳入 unknown-result 判定。现修复为将网络断开与 HTTP 5xx 一并标记为启动结果可能未知，保持提交锁定并引导核对监控，避免盲目重发。`training-devices.spec.ts` 定向 **7/7 passed**；测试模拟服务端已接受启动、响应丢失，再从 mock 监控快照核对运行任务。未连接真实 launcher。

完整 mock API/WS Playwright r26：**209 项，207 passed、2 skipped、0 failed**，2 worker、8.0 分钟，输出 `/tmp/dragon-next-e2e-s1-20260925-r26/`；两项 skip 是既有折叠分组拖动。`web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。测试使用隔离 Vite 端口和不可达 API target，不连接真实 API/WS。

S1 仍为 **NOT RUN / 未签收**：八页完整状态矩阵、真实读屏器/live-region 朗读顺序、剩余主题/视口与真实浏览器缩放人工核验尚未完成。真实 launcher、服务写入和隔离持久态未由本轮 mock 证明；真实写入、provider、训练与静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-25，r27）

`HistoryDetailPage` 主详情读取新增 loading 与失败恢复 mock E2E：挂起读取期间呈现 status，HTTP 503 呈现 alert 和“重新读取”，用户显式重试后详情恢复且错误消失。测试容忍 React StrictMode 重挂载与 query 的一次自动 GET 重试；未连接真实 history 根。

`history-overview-assets.spec.ts` 全量 **10/10 passed**。当前完整 mock API/WS Playwright r27：**211 项，209 passed、2 skipped、0 failed**，2 worker、8.0 分钟，输出 `/tmp/dragon-next-e2e-s1-20260925-r27/`；2 项 skip 是既有折叠分组拖动。`web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。文档完整性及 `rtk git diff --check` 在本更新后复验。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器/live-region 朗读顺序、剩余主题/视口与真实浏览器缩放人工核验未完成。真实 launcher、服务写入与隔离持久态未由 mock 证明；真实写入、provider、训练和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-25，r28）

历史详情新增预处理任务空态/只读 mock E2E：无训练 Loss 时显示明确的“不适用”状态，无配置快照时说明该状态，且非训练任务不提供检查点续训命令。新用例定向 **1/1 passed**。

当前完整 mock API/WS Playwright r28：**212 项，210 passed、2 skipped、0 failed**，2 worker、8.0 分钟，输出 `/tmp/dragon-next-e2e-s1-20260925-r28/`；两项 skip 是既有折叠分组拖动。`web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。文档完整性与 `rtk git diff --check` 在本更新后复验。未连接真实 API/WS 或 history 根。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵、真实读屏器/live-region 朗读顺序、剩余主题/视口及真实浏览器缩放人工核验尚未完成。真实服务写入、launcher、provider、训练和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r29-r31）

r29 完整 mock API/WS Playwright 为 **213 项：211 passed、2 skipped、0 failed**；r30 为 **214 项：212 passed、2 skipped、0 failed**。r29 截图抽查了 390px dark 训练/设置/监控与 1440px light 历史/监控，未见明显遮挡或溢出。并发图表曾出现 ECharts 零尺寸警告，监控布局单 worker 定向 8/8 未复现，作为观察项保留。

本轮修复和补测：

- 图片预览 pending 文案此前只有 `aria-busy`，没有可播报 status；现增 `role="status" aria-live="polite"`。蒙版 dirty 时从嵌入工作台返回父页面的浏览器级守卫用例确认：取消仍留在原页并保留草稿，确认后离开，不发写请求。
- 训练配置 context/raw 初始读取失败此前没有恢复命令；现提供“重试读取”，重新请求失败的查询，并在实际请求 pending 时锁按钮。mock 持续返回 503 后，显式重试成功，训练命令恢复可用。
- 训练启动 pending 按 Escape 不关弹窗/不重复请求、模型库空库 409 无危险编辑控件、历史详情 loading/503 恢复与预处理任务空指标状态均有 mock 覆盖。

定向验证：`mask-workspace.spec.ts` **11/11 passed**；`training-draft-flow.spec.ts` **4/4 passed**。最终 `web-next-check` 通过，TypeScript 与 Vitest **50 文件 / 226 用例通过**。当前工作树完整 mock Playwright r31 **217 项：215 passed、2 skipped、0 failed**，2 worker、8.1 分钟，报告 `/tmp/dragon-next-e2e-s1-20260926-r31/`；skip 为既有折叠分组拖动。使用隔离 Vite 端口 `20702`、不可达 API target `127.0.0.1:29999`；没有连接真实服务或 launcher。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器朗读顺序、剩余视口/主题/真实浏览器缩放人工复核、训练 capabilities 错误恢复、历史产物目录缺失与单路失败状态、监控状态 live announcement、数据集 dirty import 及已保存预设工作台往返仍有缺口。真实服务写入、launcher/训练/provider、隔离持久态和静态包发布不由 mock 证明，继续保持 `NOT RUN`。

### S1 续审更新（2026-09-26，r32）

- 当前 `dev` / HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27` 工作树含大量既存修改，本轮未清理、暂存或提交。完整 mock API/WS Playwright：**221 项，219 passed、2 skipped、0 failed**，2 workers、约 8.3 分钟；运行产物在 `/tmp/dragon-next-e2e-s1-20260926-r32/`。命令为 `rtk proxy env DRAGON_E2E_PORT=20703 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --output=/tmp/dragon-next-e2e-s1-20260926-r32`。API/WS 由 mock fixture 接管；2 项 skip 仍是 `dataset-drag.spec.ts` 的折叠分组拖动。
- `rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check` 通过：TypeScript 与串行 Vitest **50 文件 / 226 用例通过**。本轮新增/补齐的浏览器证据包括训练 capabilities 目录 503 时锁住启动/入队并在显式重试后恢复、历史样张目录缺失与单路失败时保留权重结果、监控任务/连接状态的 polite status、数据集 dirty 草稿导入保护及已保存预设进入图片工作台再返回。
- 套件中仍出现 ECharts 容器零尺寸控制台警告；本轮监控布局/图表边界断言通过，保留为观察项。`web-next-check` 和 E2E 均未连接真实 API、WebSocket、launcher、provider 或用户数据根，也未写入持久数据、启动训练或替换静态包。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵还未逐项对账；真实读屏器朗读/live-region 顺序、剩余主题/视口及真实浏览器缩放仍需人工核验；训练 launcher、真实服务状态和隔离持久态未由 mock 证明。真实服务写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r33）

- 对照当前数据集 mutation E2E 与 `useDatasetPresetEditor` 的两步 rename 行为，补测旧文件 DELETE 已由服务端执行、但响应丢失的结果未知分支。请求 pending 时表单/命令锁定；mock server 随后删除旧文件并 abort 响应；错误提示“操作结果尚未确认”，界面选中新预设并刷新列表，列表与服务端快照一致，未重复 DELETE。
- 定向 `dataset-crud-mutations.spec.ts` **5/5 passed**（含新增场景 **1/1**），`web-next-check` 通过（TypeScript、Vitest **50 文件 / 226 用例**）。命令：`rtk proxy env DRAGON_E2E_PORT=20705 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r2`；测试状态仅保存在内存 fixture。
- r32 的 221 项完整 mock suite 在新增用例之前运行；本轮不将 r32 计数表述为当前树完整 suite 结果。未连接真实服务、launcher、provider 或用户数据根，未执行真实写入/训练，也未构建或发布静态包。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵未逐项对账；数据集 rename 创建阶段断网和另存慢请求 pending 锁仍待补，真实读屏器朗读顺序、剩余主题/视口/真实缩放需人工核验；真实服务、launcher 与持久态未由 mock 证明。

### S1 续审更新（2026-09-26，r34）

- 新增数据集 rename 创建阶段断网恢复用例：第一次 save-as 中断后保留旧预设和 dirty 草稿、不发送 DELETE；显式重试后新预设写入、旧预设删除且列表/表单同步。完整 `dataset-crud-mutations.spec.ts` **6/6 passed**，产物在 `/tmp/dragon-next-e2e-s1-dataset-crud-r3/`。
- 命令：`rtk proxy env DRAGON_E2E_PORT=20706 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r3`。只改 mock E2E，无产品源码改动；fixture 不落真实数据。
- 最终 `rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例通过**。
- r32 完整 mock suite 早于 r33/r34 新增用例；本轮按 mutation spec 定向验证，不把 r32 计数视为当前树完整全量结果。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵未逐项对账；另存慢请求 pending 锁、长名称/路径、真实读屏器朗读顺序及剩余主题/视口/真实缩放人工核验尚未完成；真实服务、launcher 和持久态仍未由 mock 证明。

### S1 续审更新（2026-09-26，r35）

- 新增另存慢请求 pending 状态用例：POST 挂起期间保存/另存/重命名和表单输入均锁定；成功响应后新预设被选中、内容同步，旧预设未删除。`dataset-crud-mutations.spec.ts` **7/7 passed**。
- 命令：`rtk proxy env DRAGON_E2E_PORT=20707 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r4`；`web-next-check` 通过，TypeScript 与 Vitest **50 文件 / 226 用例通过**。
- r32 完整 mock suite 早于 r33-r35 新增 E2E；当前以新增完整 spec 定向结果验证增量，未把 r32 计数外推。测试只用内存 fixture，不访问真实服务或持久数据。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账，数据集长名称/路径仍待验证；真实读屏器朗读顺序、剩余主题/视口/真实缩放人工检查、launcher/真实服务与持久态尚未完成。

### S1 续审更新（2026-09-26，r36）

- 新增长数据集名称/路径 E2E：约 120 字符的名称与完整预设路径在 **390×844、1440×900** 下保持详情文本可读、列表路径以省略号显示，页面无横向溢出；`dataset-crud-mutations.spec.ts` **8/8 passed**。
- 命令：`rtk proxy env DRAGON_E2E_PORT=20709 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/dataset-crud-mutations.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-dataset-crud-r5`。最终 `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例通过**。
- r32 完整 mock suite 早于 r33-r36 新增用例；本轮以完整 dataset mutation spec 的 8/8 定向回归证明这些增量，不把 r32 计数外推。未连接真实 API/WS、launcher、provider 或持久数据。

S1 仍为 **NOT RUN / 未签收**：数据集专属 dirty/import、CRUD mutation、结果未知、pending 和长路径证据已补齐矩阵指定项；八页其余状态仍未逐项对账。真实读屏器朗读顺序、剩余主题/视口/真实缩放人工核验及 launcher/服务/持久态验证未完成。

### S1 续审更新（2026-09-26，r37）

- caption 取消 HTTP 500 mock 用例确认错误可见、任务仍显示 `running`、取消操作可由用户再次确认，且 `rerun` 继续禁用；等待 1.2 秒没有自动重试，显式第二次确认后 fixture 才转为 `canceled`。`caption-context.spec.ts` **21/21 passed**，输出 `/tmp/dragon-next-e2e-s1-caption-cancel-r37/`。
- 历史检查点续训新增队列已接受但 POST 响应丢失场景：确认期间锁住控件，随后显示结果未确认和历史/队列核对入口；队列 mock 快照显示已排队项目，且只提交一次。`s3-queue-monitor-history.spec.ts` **4/4 passed**，输出 `/tmp/dragon-next-e2e-s1-history-queue-r3/`。该结果仅证明 mock 状态语义，不证明真实服务接受或持久化。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。两个浏览器 spec 使用隔离 Vite 端口 `20710`、`20711` 和不可达 API target `127.0.0.1:29999`；API/WS 由 mock fixture 接管。r32 **221 项、219 passed、2 skipped** 仍是最近一次完整 Playwright；本轮只以 caption/history 定向结果证明增量，没有把旧全量计数外推。真实 API/WS、launcher、provider、用户数据与持久态均未接触。

S1 仍为 **NOT RUN / 未签收**：八页完整状态矩阵仍未逐项对账；真实读屏器朗读顺序、剩余主题/视口/真实浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r38）

- 历史详情“产物”页新增样张/权重独立 503 与显式重试 mock E2E：两路错误同时可见，单独恢复样张时权重错误仍保留，随后两路分别收敛到明确空态；每次显式重试只新增一次 GET，未产生写入。`history-overview-assets.spec.ts` **12/12 passed**，输出 `/tmp/dragon-next-e2e-s1-history-assets-r38/`。现有组件错误提示与重试行为符合预期，本轮只补证据，无产品逻辑改动。
- 本用例使用隔离 Vite 端口 `20712` 和不可达 API target `127.0.0.1:29999`；mock fixture 接管 API/WS。r32 **221 项、219 passed、2 skipped** 仍为最近一次完整 Playwright；r33-r38 增量通过各完整定向 spec 验证，不将旧计数外推。真实 API/WS、history/output 持久目录、launcher、provider 均未接触。

S1 仍为 **NOT RUN / 未签收**：八页完整状态矩阵仍未逐项对账；真实读屏器朗读顺序、剩余主题/视口/真实浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r39）

- 当前工作树完整 mock API/WS Playwright：**228 项，226 passed、2 skipped、0 failed**，2 workers、8.4 分钟；运行产物 `/tmp/dragon-next-e2e-s1-20260926-r39/`。两个 skip 是 `dataset-drag.spec.ts` 中折叠分组 hover 与折叠 header drop 场景。
- 本次全量包含 r33-r38 的数据集 mutation、caption cancel 500、历史续训响应丢失及历史产物独立错误恢复用例。测试使用隔离 Vite 端口 `20713` 和不可达 API target `127.0.0.1:29999`，API/WS 全由 mock fixture 接管；不构成真实服务或持久态验收。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。本节更新后文档完整性 **8/8 passed**、`rtk git diff --check` 通过。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器/live-region 顺序、剩余主题/视口与真实浏览器缩放人工核验、真实 launcher/服务/隔离持久态验证仍未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r40）

- caption provider ping 新增网络中断 mock E2E：断网后的 status 0 提示可见，pending 按钮锁定，错误后不自动重试；用户显式再次确认后成功。`interactions.spec.ts` **10/10 passed**，输出 `/tmp/dragon-next-e2e-s1-caption-ping-r40/`。
- 使用隔离 Vite 端口 `20714` 和不可达 API target `127.0.0.1:29999`，未调用真实 provider。`web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。
- r39 全量 **228 项、226 passed、2 skipped** 在 r40 用例新增前完成；本轮只记录 interactions 定向回归，不声称 r39 是含 r40 新用例的全量结果。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项对账；真实读屏器/live-region 顺序、剩余主题/视口与真实浏览器缩放人工核验、真实 launcher/服务/隔离持久态验证仍未完成。

### S1 续审更新（2026-09-26，r41-r43）

- 设置根切换后新根数据集读取遇 503 的 mock E2E 通过：旧缓存不泄漏，错误可见，显式重试后展示新根数据；设置仅提交一次。`s2-isolation.spec.ts` **4/4 passed**，输出 `/tmp/dragon-next-e2e-s1-root-switch-r41/`。
- 历史 resume-options GET 503 状态通过 mock E2E：确认与提交保持禁用，未自动重试或 POST；显式重读成功后才恢复确认。`s3-queue-monitor-history.spec.ts` **5/5 passed**，输出 `/tmp/dragon-next-e2e-s1-resume-options-r42/`。provider ping status 0 网络中断、pending/no-auto-retry/显式恢复在 `interactions.spec.ts` **10/10 passed**。
- 当前完整 mock API/WS Playwright：**231 项，229 passed、2 skipped、0 failed**，2 workers、8.6 分钟；输出 `/tmp/dragon-next-e2e-s1-20260926-r43/`。2 个 skip 为折叠分组拖动场景。使用隔离 Vite `20717` 与不可达 API target `127.0.0.1:29999`，不构成真实服务或持久态验收。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**；文档完整性 **8/8**、`rtk git diff --check` 通过。全程未连接真实 API/WS、launcher/provider，未执行真实写入、训练或静态包发布。

S1 仍为 **NOT RUN / 未签收**：八页状态矩阵与其他 workspace query 的失效覆盖范围未全部逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。

### S1 续审更新（2026-09-26，r44）

- 扩展配置根切换 mock E2E：同时切换 `configs_root`、`history_root`、`queue_root`，确认训练配置 groups/presets/merged、数据集、模型、历史和队列代表性 query 均重新读取各自新根的数据；既有新根 dataset GET 503 错误显示、无旧缓存泄漏和显式重试恢复仍通过。完整 `s2-isolation.spec.ts` **4/4 passed**，输出 `/tmp/dragon-next-e2e-s1-root-invalidation-r44d/`。
- 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20721 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/s2-isolation.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-root-invalidation-r44d`。仅扩展 mock E2E，无产品逻辑改动；未连接真实 API/WS、launcher/provider 或持久数据。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**；文档完整性 **8/8 passed**、`rtk git diff --check` 通过。r44 只执行定向 `s2-isolation.spec.ts`，不把 r43 的 231 项完整套件结果外推到新增用例。
- 覆盖限于上述代表性根相关 query 家族，不等于逐项穷尽所有 workspace query key，也不替代人工读屏、主题/缩放、真实服务和隔离持久态验收。

S1 仍为 **NOT RUN / 未签收**：完整 query key 与八页状态矩阵仍待逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r45）

- 历史续训补充混合检查点 mock E2E：一个 optimizer 不完整 checkpoint 与一个可用 checkpoint 并存；选择不可用项时显示原因并禁用提交，切回可用项后要求重新确认，最终只提交重新确认的有效路径。`s3-queue-monitor-history.spec.ts` **6/6 passed**，输出 `/tmp/dragon-next-e2e-s1-history-mixed-checkpoint-r45/`。
- 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20722 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/s3-queue-monitor-history.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-history-mixed-checkpoint-r45`。仅 mock API/WS，无真实 history/output、launcher、provider 或持久数据访问。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**。新增场景先经完整定向 spec 验证；当前树完整 Playwright 结果见 r46。
- 这是 Next UI 状态与目标绑定证据，不是后端 checkpoint 所有完整性组合或真实持久态验收。

S1 仍为 **NOT RUN / 未签收**：其他完整性字段组合、完整 query key 与八页状态矩阵仍待逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r46）

- 当前工作树完整 mock API/WS Playwright **232 项，230 passed、2 skipped、0 failed**，2 workers、8.6 分钟，输出 `/tmp/dragon-next-e2e-s1-20260926-r46/`。两个 skip 仍是 `dataset-drag.spec.ts` 折叠分组 hover/drop；全量包含 r44 根切换 query 覆盖与 r45 混合 checkpoint 场景。
- 命令：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20723 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260926-r46`。隔离 Vite、mock API/WS；未连接真实 API/WS、launcher/provider 或持久数据。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**；本节更新后的文档完整性 **8/8 passed**、`rtk git diff --check` 通过。自动化 mock 结果不替代 S1 人工与真实服务验收。

S1 仍为 **NOT RUN / 未签收**：其他完整性字段组合、完整 query key 与八页状态矩阵仍待逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r47）

- 蒙版工作区新增两个 mock-only 读取恢复 E2E：列表 GET 503 错误态下画布不误呈现为可编辑，显式重试后列表与画布恢复；预设 GET 持续 503 时保留蒙版列表，用户显式重试后错误清除。preset query 遵循当前全局一次自动重试；fixture 持续返回 503 直至测试显式放行，以覆盖稳定错误态。`mask-workspace.spec.ts` **13/13 passed**，产物 `/tmp/dragon-next-e2e-s1-mask-retry-r48/`。
- 开发模式 StrictMode 会取消首个 GET；首轮“只让首请求失败”的夹具因此未呈现错误。改为维持失败直至人工放行后，完整定向 spec 通过。本轮未改产品逻辑。
- 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20725 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/mask-workspace.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-mask-retry-r48`。仅隔离 Vite + mock API/WS，无真实服务、launcher、provider、用户目录或持久数据访问。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。r46 **232 项，230 passed、2 skipped** 仍为最近一次完整 Playwright；本轮只验证完整蒙版 spec，不将旧全量结果外推到新增用例。

S1 仍为 **NOT RUN / 未签收**：其他完整性字段组合、完整 query key 与八页状态矩阵仍待逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r49-r50）

- 队列单项 move 新增 HTTP 409/503 mock E2E，验证目标 ID 与 `{ direction: "down" }` 请求体、旧快照排序保留、pending 后解锁、错误可见及无自动重试；完整 `queue-monitor-mutations.spec.ts` **18/18 passed**，产物 `/tmp/dragon-next-e2e-s1-queue-move-http-r49/`。
- 历史 archive 新增 HTTP 409/503 mock E2E，验证确认范围、pending 锁、失败后任务和选择保留、操作解锁、无自动重试，以及显式刷新核对后清错；完整 `history-actions.spec.ts` **4/4 passed**，产物 `/tmp/dragon-next-e2e-s1-history-archive-http-r50/`。
- 两个 spec 均在隔离 Vite + route mock 下运行，API target 为不可达的 `127.0.0.1:29999`，端口分别为 `20726`、`20727`；未访问真实队列、history/output、launcher 或持久数据。本轮只补测试，没有产品逻辑修改。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。r46 **232 项，230 passed、2 skipped** 仍为最近一次完整 Playwright；本轮仅运行两个完整定向 spec，不将 r46 计数外推。

S1 仍为 **NOT RUN / 未签收**：其他完整性字段组合、完整 query key 与八页状态矩阵仍待逐项对账；真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r51-r53）

- 全局设置普通 PUT 成功路径新增 mock E2E：提交 `{ ui_scale: 125 }`，mock 返回规范化 `ui_scale: 130`，确认响应值进入表单、“全局设置已保存”提示与“已同步”状态出现、保存按钮禁用且 PUT 仅一次。完整 `state-feedback.spec.ts` **19/19 passed**，输出 `/tmp/dragon-next-e2e-s1-settings-success-r51/`。
- 训练立即启动成功用例现核对完整 POST body：配置路径、preset、variant、methods_subdir、两项确认标志和 GPU 白名单均绑定当前选择。完整 `training-devices.spec.ts` **7/7 passed**，输出 `/tmp/dragon-next-e2e-s1-training-target-r52/`。
- 随后完整 mock API/WS Playwright r53 **239 项，237 passed、2 skipped、0 failed**，2 workers、8.8 分钟，输出 `/tmp/dragon-next-e2e-s1-20260926-r53/`；两个 skip 为 `dataset-drag.spec.ts` 的折叠分组 hover/drop。命令：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20730 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260926-r53`。完整运行包含 r51/r52 新增场景。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例通过**。定向 E2E 使用隔离 Vite 端口 `20728`/`20729`，全量使用 `20730`；API target 均为不可达 `127.0.0.1:29999`，API/WS 由 route mocks 接管。没有连接真实 API/WS、launcher/provider 或用户持久数据。
- 工作树为 `dev` / HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`、ahead 13，含大量既存修改。本轮只修改 `state-feedback.spec.ts`、`training-devices.spec.ts` 和两份 S1 findings；静态包未构建或替换，没有清理、暂存或提交。
- Query key 对账确认 r44 仅验证根切换后若干代表性 query。training raw/estimate/preflight/picker-preview、dataset preset/image/cover/mask、history collections/detail/assets/summary、captioning 各查询尚未与根切换组合核验；这是证据缺口，不是已确认的缓存泄漏。

S1 仍为 **NOT RUN / 未签收**：完整八页状态矩阵和 query-key 根切换矩阵尚未逐项关闭；真实读屏器/live-region 朗读顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证仍未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r54-r59）

- 蒙版配置根切换用例改为同一 dataset file、同一源图路径 `images/studio/photo.png`，仅响应中的 `mask_dir`/revision 随根变化；不使用整页刷新回到数据集，而经设置页 SPA 导航返回。新根 mask-list GET pending 时旧 `masks/configs` 消失并显示“正在加载图片”；放行列表后 gate 同一图片参数的 mask-image GET，验证编辑区域 busy、canvas 未残留，释放后显示 `masks/external-configs` 并完成 canvas 加载。该证据覆盖 dataset preset、`['dataset-masks', file, index, 0]` 及相同图片参数的再次 GET；图片 GET 是组件 effect，不是 React Query key。
- 迭代中的失败均为测试断言问题：r54 把列表 pending 误断言为编辑区 `aria-busy`；该属性只绑定图片编辑加载。r55 改查 status 的可访问名称，但快照中的 status 未命名；两次快照均确认旧蒙版目录已消失、界面显示加载文案。改按精确可见文案查询后，r56 完整 `s2-isolation.spec.ts` **5/5 passed**，输出 `/tmp/dragon-next-e2e-s1-mask-root-r56/`。
- r56 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20733 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/s2-isolation.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-mask-root-r56`。仅隔离 Vite 与 mock API/WS；未连接真实服务、launcher、provider 或持久数据。
- 完整 mock API/WS Playwright r57 **240 项：238 passed、2 skipped、0 failed**，2 workers、8.8 分钟，输出 `/tmp/dragon-next-e2e-s1-20260926-r57/`。两个 skip 是 `dataset-drag.spec.ts` 的折叠分组 hover 与折叠 header drop。命令：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20734 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260926-r57`。该全量包含 r56 蒙版根切换场景；API target 不可达，API/WS 由 mock 接管。
- r58 为根切换补充 dataset cover/preview mock E2E：同一 dataset file 先加载旧根封面与预览；切换配置根后 gate cover GET，确认旧封面图已撤下，再放行并确认新封面；重新打开同一 preview query，pending 时旧目录/摘要不可见，放行后显示新根目录/摘要。覆盖 `['datasets','cover', file]`（60 秒 staleTime）和 `['datasets','preview', file, index]`，请求参数保持不变。完整 `s2-isolation.spec.ts` **6/6 passed**，输出 `/tmp/dragon-next-e2e-s1-root-cover-preview-r58/`。命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20735 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/s2-isolation.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-root-cover-preview-r58`。仅隔离 Vite 与 mock API/WS。
- 当前完整 mock API/WS Playwright r59 **241 项：239 passed、2 skipped、0 failed**，2 workers、8.8 分钟，输出 `/tmp/dragon-next-e2e-s1-20260926-r59/`。两个 skip 是 `dataset-drag.spec.ts` 的折叠分组 hover 与折叠 header drop。命令：`rtk proxy timeout 1200 env DRAGON_E2E_PORT=20736 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test --workers=2 --reporter=line --output=/tmp/dragon-next-e2e-s1-20260926-r59`。该全量包含 r56/r58 根切换场景；API/WS 由 route mocks 接管。
- `web-next-check` 通过：TypeScript 与 Vitest **50 文件 / 226 用例**；文档完整性 **8/8 passed**，`rtk git diff --check` 通过。没有连接真实 API/WS、launcher/provider 或用户持久数据。
- 本轮只修改 `web/frontend-next/e2e/s2-isolation.spec.ts` 和两份 S1 findings；没有改产品逻辑、发布静态包、暂存、提交或清理工作树。

S1 仍为 **NOT RUN / 未签收**：非零 offset 蒙版页及 training raw/estimate/preflight/picker-preview、history 与 captioning 等 workspace query 的根切换失效仍待核对；完整八页状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r60-r64）

- r60 在 `s2-isolation.spec.ts` 增加训练 raw TOML 根切换 mock E2E：新旧根返回不同 raw 内容，但始终使用 `configs/imported/studio-portrait.toml`；切换后经 SPA 导航重新选择同一文件，断言 raw GET 仍带相同 `file`、显示新根内容，设置 PUT 仅一次且无其他写入。该用例所在 spec 当时 **7/7 passed**，输出 `/tmp/dragon-next-e2e-s1-training-raw-r60/`。
- r61-r62 扩展蒙版 fixture 分页，并验证 offset 48 的页面跨根重新读取。两次初始尝试暴露的是测试 locator 问题：工作台列表已默认打开、切换按钮不在可访问树；随后全页查询对旧名称的匹配不稳定。快照中的当前图片列表已显示新根内容。修正为直接使用可见分页按钮，并只检查 `.mask-images .mask-image-item` 内当前图片名后，定向用例 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-mask-offset-r63/`；无产品逻辑改动。
- 最终 `s2-isolation.spec.ts` **8/8 passed**，输出 `/tmp/dragon-next-e2e-s1-root-training-mask-r64/`，隔离 Vite 端口 `20741`、不可达 API target `127.0.0.1:29999`、mock API/WS。包含 raw、offset 48、默认 offset 0、cover/preview 和设置根切换错误恢复场景。`web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。文档完整性 **8/8 passed**；普通 `rtk git diff --check` 无输出，但本次目标文件为未跟踪文件，另以 `--no-index --check` 检查它们。
- r59 的全量 mock Playwright **241 项，239 passed、2 skipped** 仍是最近一次完整 suite；本轮没有重跑全量，不能把新用例计入 r59。未连接真实 API/WS、launcher、provider 或用户持久数据；未构建/替换静态包、暂存、提交或清理工作树。

S1 仍为 **NOT RUN / 未签收**：training estimate/preflight/picker-preview、dataset preset read/preview image 与 mask-image 查询、history 和 captioning query 的根切换尚未逐项核验；完整八页状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/浏览器缩放人工验收，以及真实 launcher、服务和隔离持久态验证仍未完成。真实写入、训练、provider 调用和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r65-r70）

- r65-r68 将 history 根切换补测扩展到同一 task ID 的列表、详情、artifact manifest、样张/权重摘要读取。切换后重新打开同一历史任务，核对新根的任务标题、配置 artifact、最新样张和权重文件，且记录同一 query 的新根 GET；首轮测试曾因历史分组默认折叠及 StrictMode 重复 GET 的断言假设失败，改用实际可见分组交互并核对最终根请求后通过，未发现产品缺陷。`s2-isolation.spec.ts` **9/9 passed**，输出 `/tmp/dragon-next-e2e-s1-root-query-r68/`。
- r69 新增 `captioning-root-isolation.spec.ts`：配置根切换后，经打标来源选择器对同一 dataset file 重读 preset 与图片列表；gate 新根 GET 时确认旧图片/目录不显示，放行后展示新根路径与图片名。preset、图片请求参数保持一致，图片查询覆盖 `dataset_index=0`、`source=source`、`limit=60`、`offset=0`。第一次运行只因 mock PUT 断言漏了 `revision: settings-configs` 失败；补齐请求契约后 **1/1 passed**，产物 `/tmp/dragon-next-e2e-s1-caption-root-r69b/`，无产品逻辑改动。
- r70 新增 `training-root-isolation.spec.ts`：配置根切换后，同一训练 config key 的 estimate 与 dataset picker preview 分别重读新根数据；gate 响应期间旧估算路径/预览缩略图不显示。另以 mock preflight POST 核对 config file、variant、preset 和 methods_subdir 仍绑定所选训练文件，未启动或入队。前两次测试失败来自未打开默认收起的估算弹窗、以及把 StrictMode 下 preset GET 次数写死为两次；调整为实际交互与根/参数契约断言后 **1/1 passed**，产物 `/tmp/dragon-next-e2e-s1-training-root-r70/`。
- r70 命令：`rtk proxy timeout 180 env DRAGON_E2E_PORT=20838 DRAGON_API_TARGET=http://127.0.0.1:29999 pnpm --dir web/frontend-next exec playwright test e2e/training-root-isolation.spec.ts --workers=1 --reporter=line --output=/tmp/dragon-next-e2e-s1-training-root-r70`。r68/r69 为前一条记录中的定向根隔离测试。
- `web-next-check` 通过：TypeScript、Vitest **50 文件 / 226 用例**。r68-r70 均为隔离 Vite 与 route mocks；API target 为不可达 `127.0.0.1:29999`，未连接真实 API/WS、launcher、provider 或用户数据。
- 最近一次完整 mock API/WS Playwright 仍为 r59：**241 项，239 passed、2 skipped、0 failed**。没有重跑全量，r68-r70 增量不得计入 r59。工作树仍是 `dev`、HEAD `5b60d1113a5cb1211c9a5e8cc33e889f60e74a27`、ahead 13；本轮新增两份隔离 E2E 和本审计记录，没有构建/替换静态包、暂存、提交或清理工作树。

当前根切换证据覆盖范围：训练 raw TOML、estimate、dataset picker preview 与 preflight POST；数据集 library/cover/preview、蒙版 offset 0/48 与图片请求；history list/detail/artifact manifest、样张/权重摘要、resume-options 与日志 metadata/offset=400 分页；captioning 来源使用的数据集 library/preset/image offset=0/60。尚未逐项核验数据集其他预览参数与 mask-image 组合、history collections、log search 与其他 offset/limit 组合，以及 captioning profiles/jobs/detail/logs/prompts/assets/dictionary。它们仍是证据缺口，不等同已确认缓存泄漏。

### S1 续审更新（2026-09-26，r71、r74、r76-r77）

- 在 `s2-isolation.spec.ts` 的 history 根切换场景加入同一 task 的 resume-options 对照：旧、新根返回不同 checkpoint；新根读取挂起时旧 checkpoint 不再被选中，放行后选择新根路径。没有调用续训 POST。
- 完整 `s2-isolation.spec.ts` **9/9 passed**，输出 `/tmp/dragon-next-e2e-s1-history-resume-r74/`；使用隔离 Vite、API route mocks 与不可达 target `127.0.0.1:29999`。最近一次完整 mock Playwright 仍是 r59（241 项，239 passed、2 skipped），本次定向结果不计入全量。
- 新增 `history-root-log-isolation.spec.ts`：同一 task 在两根分别请求 `limit=1` metadata 与 `offset=400&limit=400` 页面；新根读取挂起时旧 800 行快照/日志行不显示，响应后显示新根 500 行与第 500 行日志。定向 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-history-logs-r76/`；无日志写入。
- 扩展 `captioning-root-isolation.spec.ts` 至 61 张 mock 图片：根切换后读取第一页，再请求 `offset=60&limit=60`；pending 期间旧页撤下，成功后显示新根第 61 张图片。定向 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-caption-offset-r77/`；未创建 job 或调用 provider。
- TypeScript 通过；单 worker Vitest **53 文件 / 230 用例 passed**。一次并行 `web-next-check` 的 Vitest 阶段为 53 文件 / 226 passed、4 failed；随后单 worker 全量重跑通过，不把失败直接归因于本轮改动。文档完整性 **8/8 passed**。本次没有发现产品逻辑缺陷。history collections/log search 与其他分页组合、其他 captioning query 仍是未核验覆盖，不等同缓存泄漏；八页状态矩阵、读屏器/视觉人工验收和真实服务/持久态继续未完成。

### S1 续审更新（2026-09-26，r78-r80）

- 数据集导入 POST 的 409 与响应丢失增加 mock E2E。409 时原选择和导入弹窗保留，pending 锁住提交、名称编辑、取消与 Escape，错误可见、无自动重试，用户显式重试成功后才切换；响应丢失时修复可直接盲重发的问题，改为禁用提交并提供关闭弹窗、刷新预设库的核对入口。mock 模拟已提交但响应丢失，刷新后新预设出现在列表，POST 仅一次。列表核对并不自动证明同名内容一致。
- 完整 `dataset-import-export.spec.ts` **5/5 passed**，最终输出 `/tmp/dragon-next-e2e-s1-dataset-import-r80/`。TypeScript 通过；单 worker Vitest **53 文件 / 231 用例通过**。当前树完整 Playwright 尚待本轮复跑，此前完整结果为 r59 的 **241 项、239 passed、2 skipped**。仅隔离 Vite、route mocks 与不可达 API target；没有真实配置写入或 provider/launcher 调用。
- 根边界只读复核：history collections 后端使用动态 `history_root/collections.json`，前端固定 query key；r82 mock E2E 确认设置切根后重新 GET，并由旧集合 `history-only` 切换至新集合 `external-only`。完整 `s2-isolation.spec.ts` **9/9 passed**，输出 `/tmp/dragon-next-e2e-s1-history-collections-r82/`，无集合 PUT。captioning profiles/settings/prompts 随启动时 `ANIMA_HOME`，assets/dictionary 随专用环境变量或 `ANIMA_HOME`，jobs/logs 为进程内状态；不能以 `configs_root` 切换验证它们的隔离。job 引用的 dataset/image 另按配置 resolver 验证。见 [S1 进度](dragon_next_s1_shell_audit_20260925.md)。

S1 仍为 **NOT RUN / 未签收**：history log search 与其他分页组合、八页完整状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实 launcher、服务和隔离持久态验证尚未完成。真实写入、训练、provider 和静态包发布继续 `NOT RUN`。

### S1 续审更新（2026-09-26，r84-r85）

- `history-root-log-isolation.spec.ts` 增加根切换后的全局日志搜索：旧根响应 `1 / 2`，新根请求使用 `query=root needle&cursor=0&direction=forward`；新根搜索挂起时显示“搜索中…”且没有旧匹配，放行后日志跳到新根第 500 行并显示 `2 / 2`。完整定向 **1/1 passed**，输出 `/tmp/dragon-next-e2e-s1-history-log-search-r84/`；无写入、未连接真实 history 服务。
- 修复蒙版缩放状态的可访问性：`MaskTools` 的百分比输出现在带 `aria-live="polite"` 与 `画布缩放` 标签；新增 `MaskTools.test.tsx` 验证 Enter/点击缩放命令与 100%→125%→100% 状态同步，定向 Vitest **1 文件 / 1 用例 passed**，TypeScript 通过。该修复不改变 canvas 缩放算法。
- 上轮完整 mock Playwright r83 在 265 项中运行到 258 项后因会话中断且未生成报告目录，不能记作通过；本轮完整套件需重新取得终态。S1 仍为 **NOT RUN / 未签收**，真实读屏器/人工视觉、真实服务和持久态边界不变。

### S1 续审更新（2026-09-27，r86-r90）

- r86 完整 mock suite 收集 265 项但为 **262 passed、2 skipped、1 failed**：监控状态错误后自动轮询在服务恢复时卸载“重试任务状态”按钮，`monitor-layout.spec.ts:129` 点击超时。单 worker r87 与完整 `monitor-layout.spec.ts` r89 均通过，确认问题与并行/轮询时序相关。
- 修复 `LiveMonitorPage`：状态 query 出错时停止 `refetchInterval`，只允许用户显式重试；E2E 断言错误窗口内 5.5 秒没有自动状态请求，点击后请求和指标/日志刷新恢复。`monitor-layout.spec.ts` **11/11 passed**。
- r90 完整 mock API/WS Playwright **265 项，263 passed、2 skipped、0 failed**，输出 `/tmp/dragon-next-e2e-s1-20260927-r90/`，耗时约 9.5 分钟。全程隔离 Vite、不可达 `127.0.0.1:29999` target、API/WS mocks；未访问真实 launcher、provider、用户目录或持久态。
- TypeScript 通过；单 worker Vitest **54 文件 / 234 用例通过**；文档完整性 **8/8**；`rtk git diff --check` 通过。S1 仍为 **NOT RUN / 未签收**，真实读屏器、人工视觉/缩放、服务和持久态验收继续保留为未完成边界。

### S1 续审更新（2026-09-27，r91）

- 新增 `dataset-media-root-isolation.spec.ts` 两项 mock-only E2E：第二子集深链 preview/mask 使用 `dataset_index=1`；preview 保持 `source=source&limit=120`，mask 列表保持 `offset=0`，蒙版图片请求携带第二子集的完整 image 路径。切换 `configs_root` 后，新根请求 pending 期间旧目录、caption、mask 目录和 canvas 均撤下，放行后只显示新根数据。
- 新增 `captioning-assets-states.spec.ts`：`/api/captioning/model-assets` 与 `/api/captioning/tag-dictionary` 连续失败后保持错误，默认查询重试窗口结束后不再增加请求；只有显式点击“刷新状态”才恢复。下载、provider 和其他写请求均为零。
- r91 定向命令使用隔离 Vite 端口 `20975`、不可达 `127.0.0.1:29999` target，结果 **3/3 passed**，产物 `/tmp/dragon-next-e2e-s1-r91d/`；TypeScript 与 `rtk git diff --check` 通过，未修改后端逻辑或用户数据。
- S1 仍为 **NOT RUN / 未签收**：source/offset/limit 其余组合、八页完整状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实服务、launcher 和隔离持久态继续未完成。

### S1 续审更新（2026-09-27，r92）

- 完整 mock API/WS Playwright **268 项，266 passed、2 skipped、0 failed**，输出 `/tmp/dragon-next-e2e-s1-r92/`，耗时约 16.7 分钟；2 个 skip 仍是 `dataset-drag.spec.ts` 的折叠分组 hover/drop 场景。r91 的第二子集媒体隔离和 Captioning 资源/词典恢复用例已纳入全量。
- 单 worker Vitest **56 文件 / 241 用例通过**；TypeScript、文档完整性 **8/8**、`rtk git diff --check` 通过。完整运行仍只使用隔离 Vite、API/WS route mocks 和不可达 target，未触碰真实 launcher、provider、用户目录或持久态。
- S1 仍为 **NOT RUN / 未签收**：source/offset/limit 其余组合、八页完整状态矩阵、真实读屏器/live-region 顺序、剩余视口/主题/真实缩放人工核验，以及真实服务、launcher 和隔离持久态继续未完成。

### S1 续审更新（2026-09-27，r93）

- 修复并核验壳层可访问性：`LogViewer` 的可滚动日志区新增 `aria-label="日志内容"` 与 `tabIndex=0`；`CommandDialog` 关闭按钮与 Escape 统一尊重 busy 状态。新增 `LogViewer.test.tsx`、`CommandDialog.test.tsx`，组件行为定向通过。
- 修复 Captioning prompt-presets 失败后的局部恢复：提示词预设页与图片来源页均显示“重试提示词”，连续 503 的默认查询重试窗口结束后不再请求，显式重试只增加一次 GET 并恢复 fixture 预设。`captioning-prompts-states.spec.ts` **2/2 passed**，无 fixture writes/未处理请求。
- 新增训练量估算失败态 mock E2E：`training-estimate.spec.ts` **6/6 passed**，覆盖 503 alert、空态、请求停止和显式“重新估算”恢复；不改变现有自动重试策略。
- 新增机器可核验状态登记：[dragon_next_s1_state_matrix_r93.json](dragon_next_s1_state_matrix_r93.json) 共 70 个 workspace/state 单元格；`node web/frontend-next/scripts/validate-s1-state-matrix.mjs` 校验测试文件、标题和未运行/人工边界通过，矩阵总体仍保持 `NOT_RUN`。
- r93 完整 mock API/WS Playwright **271 项，269 passed、2 skipped、0 failed**，输出 `/tmp/dragon-next-e2e-s1-r93/`；单 worker Vitest **58 文件 / 244 用例通过**，TypeScript 通过。所有浏览器证据仍只使用隔离 Vite、route mocks 和不可达 API target。
- S1 仍为 **NOT RUN / 未签收**：八页状态矩阵尚未逐项机器化对账，真实读屏器/live-region 顺序、剩余主题/视口/真实缩放人工核验，以及真实服务、launcher、provider、持久态和发布仍未执行。

### S1 续审更新（2026-09-27，r94）

- 训练配置专属 409 mock E2E 已补齐：首次保存冲突保留本地草稿并锁定重复保存；显式 reload 后采用服务器 `server-v2` revision，之后用户再次编辑才发送第二次 PATCH。`training-draft-flow.spec.ts` 完整定向 **6/6 passed**，产物 `/tmp/dragon-next-e2e-s1-training-config-409-r94/`。
- 本轮只使用隔离 Vite、route mocks 和不可达 API target `127.0.0.1:29999`，不证明真实配置服务、磁盘写入或持久化；r93 的 **271 项、269 passed、2 skipped** 仍是最近一次完整 Playwright 基线，不将 r94 定向结果外推到全量。
- 机器状态矩阵仍为 70 行、`overall_state=NOT_RUN`；`training.conflict_409` 已从 `NOT_RUN` 登记为 `PASS_EVIDENCE`，真实服务、人工读屏器/视觉、launcher/provider 和发布门禁仍未执行。

S1 仍为 **NOT RUN / 未签收**。

### S1 续审更新（2026-09-27，r95）

- 图片工作台预览读取失败新增 mock-only E2E：持续 503 时错误可见、图片卡片为空且默认重试停止，显式“重试”后恢复 3 张图片；`mask-workspace.spec.ts` **14/14 passed**，产物 `/tmp/dragon-next-e2e-s1-dataset-image-error-r95/`。
- 只读图片 GET 不具备提交后结果未知/服务端对账契约，因此 `dataset-image.offline_unknown` 已登记为 `N/A`；本轮不把网络 abort 误标成未知提交结果。r93 完整 271 项仍是全量基线，r94/r95 为定向增量。
- 矩阵仍为 70 行、`overall_state=NOT_RUN`；真实服务、人工读屏器/视觉、launcher/provider、持久态和发布门禁仍未执行。

S1 仍为 **NOT RUN / 未签收**。

### S1 续审更新（2026-09-27，r96）

- Captioning confirmed-empty mock E2E 已补齐：任务列表返回空数组时显示“暂无打标任务”，图片来源和“新任务”入口仍可用；`captioning-prompts-states.spec.ts` **3/3 passed**，无 provider 调用或写请求，产物 `/tmp/dragon-next-e2e-s1-captioning-empty-r96/`。
- 本轮只使用隔离 Vite、route mocks 和不可达 API target `127.0.0.1:29999`；r93 完整 271 项仍是全量基线，r94-r96 仅为定向增量，不外推全量通过数。
- 矩阵仍为 70 行、`overall_state=NOT_RUN`；`captioning.empty` 已登记为 `PASS_EVIDENCE`，剩余人工核验和真实服务边界保持未签收。

S1 仍为 **NOT RUN / 未签收**。

### S1 续审更新（2026-09-27，r97-r98）

- r97 的 `state-feedback.spec.ts` **23/23 passed**，覆盖设置读取/保存 pending、模型读取 pending 与成功默认项；产物 `/tmp/dragon-next-e2e-s1-settings-model-loading-r97/`。该结果只证明 mock API/WS 下的前端锁定和同步语义。
- r98 的数据集、蒙版和监控初始读取 gate 合计 **33/33 passed**：`training-dataset-picker.spec.ts`、`mask-workspace.spec.ts`、`monitor-layout.spec.ts`。命令使用隔离 Vite 端口 `5192`、不可达 API target `127.0.0.1:29999`，产物 `/tmp/dragon-next-e2e-s1-loading-r98/`；没有 fixture writes/未处理请求。
- 矩阵快照更新为 `r98`：59 个 `PASS_EVIDENCE`、6 个 `MANUAL_REQUIRED`、5 个 `N/A`，`overall_state` 仍为 `NOT_RUN`。mock 增量不外推为真实 API/WS、读屏器、视觉、持久态、launcher、provider 或发布通过。

S1 仍为 **NOT RUN / 未签收**：剩余人工核验、契约 `N/A` 以及真实服务和发布边界不变。

### S1 续审更新（2026-09-27，r99-r102）

- r99 的训练/历史 mock E2E 合计 **22/22 passed**，覆盖训练上下文 loading、空配置库、正常命令面，以及历史完成态的摘要、产物分页、日志同一任务身份。产物 `/tmp/dragon-next-e2e-s1-training-history-r99/`。
- Captioning success 用例在 r100 通过；loading 首次因桌面隐藏任务切换按钮的 locator 假设失败，修正后 r102 定向 **1/1 passed**，产物 `/tmp/dragon-next-e2e-s1-captioning-loading-r102/`。无 provider、真实 TXT、持久化或非 GET fixture 写入。
- r104 补强 dataset-image busy 语义：Next 大图 viewer 现在在真实 `<img>` `load` 前显示 `aria-busy="true"`/“正在读取图片”，`error` 先解除 pending 再给出重试；组件级 mock 生命周期测试覆盖 pending、error、retry。该证据不外推为真实图片服务可达或真实文件完整性。
- r104 图片/蒙版 mock E2E **15/15 passed**；训练、设置/模型、图片/蒙版、监控、历史、打标提示词六组重点状态 spec 合计 **76/76 passed**。命令使用隔离 Vite 端口 `21994`、不可达 API target `127.0.0.1:29999`，产物 `/tmp/dragon-next-e2e-s1-r104-focused/`；没有 fixture writes/未处理请求。
- r105 扩展 `history-overview-assets.spec.ts` 的成功态：完成历史任务现在在同一 mock 流程中读取 checkpoint、显式确认并提交 `{task_id, checkpoint}`，成功后显示“查看监控”；历史详情与既有续训冲突 spec 合计 **22/22 passed**。该流程仍只证明前端/API contract，不证明真实 checkpoint 或队列持久化。
- r106 补齐训练设备 loading 语义：`TrainingDevices` 的快捷设备区域在 GPU 查询 pending/fetching 时暴露 `aria-busy`；`training-devices.spec.ts` **8/8 passed**，验证首读挂起、执行命令禁用和释放后的设备恢复。
- r103 重跑 Captioning 两个完整定向 spec **26/26 passed**，确认 loading、success 以及既有错误/取消/响应丢失路径在修正后稳定；产物 `/tmp/dragon-next-e2e-s1-captioning-r103/`。
- r107 修复 Captioning profiles/jobs 读取失败后的恢复语义：失败停止自动重试/轮询并提供显式 QueryFeedback 重试；详情读取沿用既有重试控件并新增浏览器证据。`captioning-prompts-states.spec.ts` **7/7 passed**，未调用 provider 或执行写请求。
- r107 补充 history 完成态的非完整产物组合：可用配置快照与不可读日志、缺失运行配置并存时保持状态可解释；样张/权重目录不存在时摘要明确显示目录缺失，且不可用日志不生成下载链接。`history-overview-assets.spec.ts` **14/14 passed**。
- r107 增加 dataset preview API 的非默认 `limit` 契约测试，保持当前 `source=source` 语义，不扩大为 `source=training` 或 offset 功能。相关 API 单测 **10/10 passed**；修正既有图片 viewer fixture 类型后 TypeScript 通过。
- 矩阵更新为 `snapshot=r107`：65 个 `PASS_EVIDENCE`、0 个 `MANUAL_REQUIRED`、5 个 `N/A`，`overall_state=NOT_RUN`。本轮重点 spec 合计 **21/21 passed**，单 worker Vitest **59 文件 / 247 用例通过**；mock 状态矩阵完成不改变 S1 未签收、真实服务和人工读屏/视觉边界。
- r108 在当前工作树完成完整 mock API/WS Playwright 回归：**292 项，290 passed、2 skipped、0 failed**，耗时约 10.1 分钟，产物 `/tmp/dragon-next-e2e-s1-r108/`；两个 skip 仍是 `dataset-drag.spec.ts` 的折叠分组 hover/drop 场景。使用隔离 Vite、不可达 API target 和 2 workers，未连接真实 API/WS、launcher、provider 或持久态。
- 矩阵更新为 `snapshot=r108`：65 个 `PASS_EVIDENCE`、0 个 `MANUAL_REQUIRED`、5 个 `N/A`，`overall_state=NOT_RUN`。r108 全量通过不改变 S1 未签收、真实服务和人工读屏/视觉边界。

### S1 续审更新（2026-09-27，r109-r110）

- r109 新增 Captioning profile mock E2E **2/2 passed**：服务端异常中的 `api_key`、token、password 等敏感字段不回显，也不会被无意带回 PUT；保存 503 后名称与新密钥草稿保留，等待期间不自动重试，显式重试后成功。产物 `/tmp/dragon-next-e2e-s1-caption-profile-r109/`；未调用 provider 或真实配置写入。
- r109 补充蒙版非零分页 mock E2E：完整 `mask-workspace.spec.ts` **16/16 passed**，验证 `offset=48`、第二页图片身份以及返回第一页的缓存；产物 `/tmp/dragon-next-e2e-s1-mask-pagination-r109-r3/`。未触碰真实图片、蒙版或用户目录。
- r110 完成当前树完整 mock API/WS Playwright 回归：**295 项，293 passed、2 skipped、0 failed**，耗时约 10.2 分钟，产物 `/tmp/dragon-next-e2e-s1-r110/`；2 个 skip 仍是 `dataset-drag.spec.ts` 的折叠分组 hover/drop 场景。使用隔离 Vite 端口 `5191`、不可达 API target `127.0.0.1:29999`、2 workers。
- `pnpm run typecheck`、`pnpm run test` 通过，Vitest **59 文件 / 249 用例**；矩阵仍为 **70 行、65 个 `PASS_EVIDENCE`、0 个 `MANUAL_REQUIRED`、5 个 `N/A`**，`overall_state=NOT_RUN`。新增 mock 证据不改变 S1 未签收，也不替代真实 API/WS、持久态、launcher/provider 或人工读屏/视觉验收。

### S1 续审更新（2026-09-27，r111-r115）

- r111 修复 Captioning 初始建 job 的协议边界：缺少 `job.id` 的 202 响应现在 fail-closed 显示错误，不导航到不存在的任务；新增 503 保留选择后显式重试成功、请求 payload 与缺少任务 ID 两项 mock E2E，**2/2 passed**。产物 `/tmp/dragon-next-e2e-s1-caption-create-r111-r2/`。
- r112 新增历史续训边界 **2/2 passed**：空 checkpoint 保持确认与 POST 禁用；响应丢失显示“核对历史任务/核对队列”链接并禁止盲重试。产物 `/tmp/dragon-next-e2e-s1-history-resume-edge-r112/`。
- r113 首次并发全量为 **300 项，296 passed、2 skipped、2 failed**；失败仅为既有 `dataset-scroll`、`history-grouping` fixture 等待超时。r114 将两份完整定向 spec 串行重跑为 **9/9 passed**，不将 r113 作为最终快照。
- r115 在空闲环境完成当前树完整 mock API/WS Playwright 回归：**303 项，301 passed、2 skipped、0 failed**，耗时约 12.5 分钟，产物 `/tmp/dragon-next-e2e-s1-r115/`；使用隔离 Vite 端口 `5199`、不可达 API target `127.0.0.1:29999`、2 workers。
- `pnpm run typecheck`、串行 `pnpm run test` 通过，Vitest **61 文件 / 256 用例**；矩阵仍为 **70 行、65 个 `PASS_EVIDENCE`、0 个 `MANUAL_REQUIRED`、5 个 `N/A`**，`overall_state=NOT_RUN`。新增 mock 证据不改变 S1 未签收，也不替代真实 API/WS、持久态、launcher/provider 或人工读屏/视觉验收。

S1 仍为 **NOT RUN / 未签收**：5 个契约 `N/A` 与真实服务、读屏器、视觉、持久态、launcher/provider 和发布边界继续保留。

### S2 配置、数据集、模型与设置的隔离写入

用临时 `configs_root`、settings、dataset 与 sample-prompts fixture 启动隔离 aiohttp 服务；先备份 fixture，再对照磁盘结果。覆盖训练配置“编辑→预览 diff→保存→重载”、只读变体另存、未知键保留、raw TOML 解析/重命名/导入、样张格式保留；数据集分组/排序/子集/应用引用及回滚；模型库 revision 409；设置根切换的查询失效与不迁移文件。针对两标签冲突、网络断在提交前/后、重复点击、非法路径和服务端 400/409/500 建用例。

**退出**：请求 payload、响应、磁盘文件与刷新后 UI 一致；失败不丢草稿、不越界写入、不静默重复提交。不得对真实 `configs/imported/`、`configs/web-ui-settings.toml` 或数据集目录做此阶段实验。

当前进度：**PASS（仅 S2，非发布验收）**。训练 TOML、数据集预设、样张 TXT、模型库及全局设置的旧 revision 写入均以 409 拒绝；隔离磁盘结果、草稿保留、显式重载、根切换与结果未知状态已验证。旧客户端不带 revision 的兼容写入及直接文件系统写入不在 Next 的并发保证内。证据与门禁波动见 [S2 隔离写入验收记录](dragon_next_s2_isolation_20260925.md)。

### S3 队列、监控、历史与续训控制面

先在 mock E2E 注入等待/运行/异常/完成/取消、revision 竞争、WS 旧任务事件、HTTP 局部 5xx、慢请求和重连，再用隔离服务/假 launcher 验证队列和历史持久化。重点核对批量操作的选择范围与 `expected_revision`、停止请求绑定 `task_id`、未知状态禁用危险按钮、监控指标/日志任务身份、历史搜索游标与返回位置、产物 missing/blocked/unreadable、检查点完整性及“总目标→追加步数”换算。

**退出**：控制命令只作用于确认的任务/范围；并发冲突不误操作；历史文件删除与恢复只在临时根；不真实停止用户训练。真实 GPU 启动/续训不属于本阶段。

**当前状态：PASS（仅 S3，非发布验收）**。证据见 [S3 队列、监控、历史与续训控制面验证报告](dragon_next_s3_queue_monitor_history_20260925.md)。本轮新增五状态队列/刷新与确认范围 E2E、产物四态与续训追加步数 E2E，以及临时 queue/history 重启恢复和陈旧 task ID 停止拒绝测试；本次修复进一步通过真实 `set_queue_settings()` 写盘后的 revision 重启复原、active fake process 停止清理、产物下载链接契约和 3 项 Findings 索引门禁。Mock E2E 50/50（修复 spec 2/2）、隔离控制面 Python 120/120、文档完整性 8/8、Vitest 49 文件/218 用例和 TypeScript typecheck 均通过。真实服务写入、真实训练停止/GPU 续训保持 `NOT RUN`，S1/S2/S4/S5 状态不因本轮结果改变。

### S4 图片、蒙版与打标闭环

使用临时图片、mask、caption 和 provider stub。检查图片工作台三视图共享 dataset/subset、深链缺参数回退、预览 404、蒙版画笔/缩放/撤销/保存/批量应用、离开保护、配置 sidecar 变化；打标检查跨页对象选择、候选草稿与保存/commit 两道边界、TXT 写回精确范围、翻译只影响草稿、日志错误恢复、密钥不回显。对 PNG/TOML/TXT 比较提交前后内容。

**退出**：每次写回均由明确命令触发且目标精确；失败保留可重试草稿；不会触发外部计费 provider、模型下载或真实用户 caption 修改。

当前进度：**PASS（仅 S4，非发布验收）**。临时目录后端测试覆盖蒙版读写、revision 冲突、批量应用目标校验及配置持久化；captioning 通过本地 provider stub 完成候选生成、人工编辑和对选中 TXT sidecar 的显式 commit，未选文件保持不变。复核修复非法蒙版 JSON 误写、打标本地 profile 错收密钥、返回位置校验和蒙版切子集 URL 不同步，并补回归。前端测试覆盖图片上下文、返回滚动恢复、蒙版脏态/冲突/重试、批量应用、caption 草稿/commit 恢复、翻译候选显式应用和日志重试。临时 aiohttp + 当前 Vite 源码的真实浏览器蒙版探针已通过；caption 浏览器测试仍 mock API。真实 provider、训练、模型下载、用户数据写回和发布静态包均保持 `NOT RUN`。完整命令、边界和遗留覆盖项见 [S4 报告](dragon_next_s4_image_mask_caption_20260925.md)。

### S5 发布候选与受控热验收

先完成 S0-S4 和后端 HTTP/路径安全测试，再在独立静态根检查构建的哈希资产、深链刷新、旧 chunks 保留、`previous-index.html`、原子替换及回退。真实 GPU 短训、续训、训练中停止、外部 provider 调用、下载及真实 TXT 写回分别列为**需用户明确授权的独立验收**，预先列出设备、测试数据、输出目录、时长、成本与中止办法。未授权时这些项保持 `NOT RUN`，不得通过 mock 成功替代。

**退出**：P0/P1 无开放项，G1 门禁通过，P2 有负责人和回归计划；发布候选与证据快照一一对应；默认入口切换需另行批准，保留旧 UI 回退路径。

**当前状态：BLOCKED（仅完成 S5 候选门禁准备）**。独立临时静态根构建、69 个资产哈希清单、四个 `/next` 深链、实际 aiohttp 静态 handler 的入口/资产读取与越界拒绝、旧 chunk 保留、`previous-index.html`、原子替换和临时根回退均通过；证据见 [S5 发布候选记录](dragon_next_s5_release_candidate_20260925.md)。静态服务测试 27/27、typecheck、最小 aiohttp probe 和当前 Vitest 86 suites/221 tests 通过；静态服务 + HTTP 契约合计仍为 48/1（HTTP 子集 21/1），S1 仍为 `NOT RUN`，S4 虽已完成 scoped 隔离报告但部署态联通仍 `NOT RUN`，因此不满足 S5 完整退出条件；真实 GPU/provider/下载/TXT 写回继续保持 `NOT RUN`。

## 功能验证矩阵

| 领域 | 最小成功场景 | 必测失败/竞争场景 | 推荐层级 |
| --- | --- | --- | --- |
| 配置/设备 | 保存重载一致；单卡与双卡白名单；预检确认后执行 | 只读原位保存、无 GPU/设备消失、兼容拒绝、保存失败阻止启动、请求结果未知 | Vitest + mock E2E + 临时根 HTTP |
| 数据集/图片 | 子集编辑、搜索/排序、图片上下文、应用引用 | 跨组失败回滚、脏态切换、图片缺失、非法路径 | 组件 + mock E2E + 临时目录 |
| 蒙版 | 绘制后 PNG 保存、应用到指定子集 | revision/If-Match 409、未保存离开、图片变更、目标列表校验/拒绝、响应丢失后的持久态核对 | canvas 组件 + 临时目录 HTTP |
| 队列 | 入队确认、策略保存、重排、暂停/恢复 | revision 409、重复提交、批量范围不符、启动中任务、未知结果 | mock E2E + 假 launcher |
| 监控 | 同任务 HTTP/WS 合并刷新、日志/图表/设备信息 | 旧 task WS、断线、局部 5xx、停止时任务切换 | mock E2E + HTTP 契约 |
| 历史 | 搜索分页、返回位置、详情产物、对比、检查点选项 | 游标失效、已删任务、缺失/阻止产物、续训目标换算、归档/删除冲突 | mock E2E + 临时 history/output |
| 模型/全局设置 | revision 保存、根路径展示、缩放与主题持久化 | 409、读取失败、路径拒绝、根切换后旧 query 残留 | 组件 + 临时 settings HTTP |
| 打标 | 跨页选择、审阅草稿、候选保存、指定 TXT commit | provider 失败/取消、密钥回显、重复 commit、写回冲突 | mock E2E + stub provider/临时 TXT |

统一断言：用户可见结果、请求目标与方法、请求次数、服务端持久态、刷新后的展示必须互相印证；不能只断言按钮可点击或 toast 出现。对于启动、入队、停止、删除和 writeback，另断言确认弹窗中的**目标身份与范围**，以及操作期间按钮锁定与失败后恢复。

## 现有工具、执行顺序与安全边界

推荐从便宜、无副作用的门禁逐层扩大：

```bash
# 当前已通过；会运行 typecheck 和 Vitest，不连接真实后端
rtk proxy timeout 180 .venv/bin/python tasks.py web-next-check

# mock API/WS 的浏览器测试；使用空闲端口及独立输出目录
rtk proxy env DRAGON_E2E_PORT=20624 DRAGON_API_TARGET=http://127.0.0.1:29999 \
  pnpm --dir web/frontend-next exec playwright test \
  --output=/tmp/dragon-next-e2e-20260924

# 生产只读巡检；须先修复 V-01，且截图可能含用户信息
rtk proxy env DRAGON_VERIFY_URL=http://127.0.0.1:20203 \
  DRAGON_VERIFY_OUTPUT=/tmp/dragon-next-production-20260924 \
  node web/frontend-next/scripts/verify-production.mjs

# 文档与 Web HTTP 契约，后者用临时目录/服务 mock，不能指向当前用户根
rtk proxy timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
rtk proxy timeout 180 .venv/bin/python -m pytest \
  tests/test_web_static_server.py tests/test_web_http_contracts.py \
  tests/test_web_mask_editor.py -q
```

`web-next-e2e` 的 `fixtures.ts::mockWorkspace` 拦截 API 与 WS，非 GET 写入只进内存记录，不是后端集成证据。`verify-production.mjs` 只允许 loopback，拦截非只读 `/api/**`，并把真实页面截图写到 `/tmp`；S0 已在训练计划阶段就绪后生成完整 `report.json`（路径见 S0 报告）。`web-next-build` 会发布到 `web/static/dragon-next/`，不应作为“只读测试”运行；需先在临时目录构建并验资产，再经发布批准。上述命令中的 E2E 与 HTTP 契约测试是**下一阶段计划**，不是本轮已执行结果。

用户运行数据保护线：不清理或覆盖 `.venv/`、`models/`、`output/`、`post_image_dataset/`、`logs/`、`configs/imported/`、`configs/web-training-history/`、`configs/web-training-queue/`；不启动长训练、不取消现有任务、不下载模型、不调用计费 API。读生产页面不等于有权进行生产写操作。

## 每阶段交付模板

每轮在新记录中写明：日期、分支/HEAD、脏文件摘要、静态包身份、服务地址、fixture 根、浏览器/视口/主题、命令与完整结果、截图/trace 的本地位置、已确认问题及复现、未执行原因、P0/P1/P2/G1 清单、回退方式和签收人。状态只使用 `PASS` / `FAIL` / `BLOCKED` / `NOT RUN` / `N/A`；历史文档的通过数不得移植成当前工作树分数。

发布判断以 [前端健康度评分卡](../features/frontend-health-scorecard.md) 的结构、测试、交互、配置体验四域为参考，但本轮证据不足以重新打总分。完成 S0-S4 后再评分，并和 [Dragon Next 实施记录（已归档）](../../_archive/docs/features/dragon-next-implementation.md)、[UI/UX 历史进度（已归档）](../../_archive/docs/features/dragon-next-uiux-progress.md)逐项对账。
