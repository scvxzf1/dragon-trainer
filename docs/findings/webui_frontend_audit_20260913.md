# WebUI 前端审计（2026-09-13）

> 历史快照：下文保留修正前的审计状态、行号和失败记录，不代表当前提交状态。
> 后续修复、复验及尚未关闭的风险以[审计收尾记录](uncommitted_audit_closure_20260913.md)为准。

- 状态：当前工作树审计；未修改前端实现和测试，未提交。
- 分支/基线：`dev`，`HEAD=2e96bd0c`。工作树含大量未提交前端、后端和文档改动；以下结论针对磁盘当前状态，不代表该 commit 或发布版本。
- 复核：2026-09-13；本次核对前端关键路径与审计文档，重跑 Next Vitest 与文档完整性测试。表中完整 Classic/Dragon 门禁、Next Playwright 和 Next typecheck/build 数字沿用本轮较早的执行记录；本次未重跑完整门禁、完整 Playwright、typecheck 或 build。
- 覆盖：Classic / Dragon 静态前端、Dragon Next React 前端、API 路由契约、前端测试与门禁、CSS/DOM/模块加载。
- 安全边界：Next E2E 仅使用隔离 Vite 端口和 mock API；没有连接真实训练服务、执行训练或写用户配置；未构建/发布静态前端，没有清理截图、队列、历史或输出。

## 结论

没有确认的 P0/P1 级问题。在本轮抽查的 API 路由和 URL 生成范围内，未发现可确认的前后端错配或可利用的前端 XSS；这不是系统性安全保证，仍需完整安全测试。Next 类型检查和组件/领域测试通过，但当前工作树的 Classic 静态测试及 Next 浏览器 E2E 均有失败，不能据此宣称前端全绿。

主要产品风险是训练配置库使用键盘 ArrowDown 命中下一项时，当前源码路径可能计算出原位置并跳过排序 mutation。单项 E2E 复跑在错误提示断言处提前失败，没有直接记录 POST 数量；修正用例后仍需重新验证该交互。另有帮助摘要/详细说明不一致、多个 ESM 模块使用不同 query token 导致重复实例，以及一批绑定旧文案/旧 token/旧字段数量的测试快照。E2E 失败中，部分是测试坐标或 mock/对话框处理问题，须先修测试后才能有效验收相应产品路径。

## 验证结果

| 面 | 命令/证据 | 结果 |
|---|---|---|
| Classic / Dragon 静态门禁（此前运行记录） | `timeout 360 .venv/bin/python -m pytest -q tests/*frontend*.py tests/test_training_frontend_*.py tests/test_dragon_ui_bootstrap_runtime.py tests/test_dragon_route_styles_runtime.py tests/test_webui_design_system.py` | **357 passed / 11 failed**，266.47 秒；本次未重跑完整组 |
| Classic 模块/DOM/启动定向组（此前运行记录） | `timeout 180 .venv/bin/python -m pytest -q tests/test_training_frontend_modules.py tests/test_training_frontend_dom.py tests/test_dragon_ui_bootstrap_runtime.py tests/test_dragon_route_styles_runtime.py` | **30 passed / 1 failed**；此前运行记录，本次未重跑 |
| Dragon Next TypeScript（此前运行记录） | `pnpm --dir web/frontend-next run typecheck` | 此前通过；本次未重跑 |
| Dragon Next Vitest（本次复核） | `pnpm --dir web/frontend-next exec vitest run` | **46 个文件 / 208 项通过** |
| Dragon Next production bundle（此前运行记录） | `pnpm --dir web/frontend-next exec vite build --outDir /tmp/dragon-next-frontend-audit-build --emptyOutDir` | 此前通过；构建输出只写入 `/tmp`，未调用发布脚本或修改 `web/static/dragon-next`；本次未重跑 |
| Dragon Next Playwright（此前运行记录） | 隔离 Vite `20619`、`DRAGON_API_TARGET=http://127.0.0.1:29999`、输出 `/tmp/dragon-next-frontend-audit-20260913` | **122 passed / 6 failed / 2 skipped**，6.8 分钟；本次未重跑完整套件 |
| 文档完整性 | `timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q` | **7 passed / 1 failed**；失败仅报告 `auto_block_swap_families_20260909.md`、`anima_dual_gpu_parallel_probe_20260904.md` 缺生命周期状态标签；本轮审计文档有状态标签且索引可达 |
| 配置库 E2E 复验（此前运行记录） | `training-library.spec.ts`、`training-library-names.spec.ts`，单 worker/隔离端口 `20620` | 此前记录 4 项失败；本次未重跑 |
| 键盘排序 E2E 单项复验（此前运行记录） | `training-library.spec.ts --grep "keyboard sorting"`，单 worker/隔离端口 `20622` | 此前运行记录 1 项失败：命中下一条后没有预期的错误提示；本次复核了代码路径，但该复跑在 requests 计数断言前已失败 |
| 打标切换 E2E 复验（此前运行记录） | `caption-context.spec.ts --grep "switching caption jobs"`，隔离端口 `20621` | 此前记录失败；trace 只记录一次 confirm，却对同一 Dialog 先调用 dismiss 再调用 accept；本次未重跑 |

E2E 跳过的两项是现有 `dataset-drag.spec.ts` skip，不是本轮新增。输出都在 `/tmp/dragon-next-frontend-audit-*`，不在仓库测试产物目录。Classic / Dragon 本轮没有进行完整人工浏览器视口巡检；Next 的 Playwright 断言不替代实体读屏器或人工辅助技术验收。没有执行 `web-next-build`，因为该入口会发布静态产物；部署目录构建/回退原子性仍需单独隔离验证。

## 已确认风险

### P2：训练配置库键盘向下排序可能无动作

`TrainingLibraryDrag.tsx:49` 使用 `y > rect 中线` 决定 before/after；键盘传入的 `collisionRect` 在 ArrowDown 命中下一项时以目标项中心参与计算，中心落在严格大于条件之外。`trainingLibraryDrag.ts:40-43` 会把同组同顺序位置变为 `null`，而 `TrainingLibraryDrag.tsx:134-138` 只在 placement 非空时发 mutation。E2E 单项复跑在预期错误提示断言处失败；该次执行没有到达 requests 计数断言。当前无排序写入结论来自测试失败现象与上述源码路径推导，未由该复跑直接观测 POST 数量为零。

指针排序失败则不同：用例 `training-library.spec.ts:62-67` 将指针放在目标手柄的垂直中心，同时期望得到 `position=after`；现实现以中线划分，中心应属于 before。因此当前失败更像测试几何与期望不一致，不能据此认定指针排序普遍失效。应改用明确位于目标行下半部的落点，再验证请求体与失败回滚。

### P2：紧凑帮助摘要与完整帮助有 7 项漂移

`test_config_help_summary_matches_full_catalog` 把 `FIELD_HELP_SUMMARY_ZH` 与 `FIELD_HELP_ZH` 的 summary/purpose 对照，发现以下键描述不一致：

- `dim_from_weights`
- `max_train_epochs`
- `max_train_steps`
- `save_last_n_epochs`
- `checkpointing_last_n_epochs`
- `sample_ratio`
- `auto_block_swap_swap_io_limit_mb`

例如 `max_train_epochs` 完整帮助说明“有值会重新计算并覆盖 max_train_steps”，紧凑摘要却没有完整表达这个互斥/覆盖行为。需确认哪一侧为单一事实源并生成或同步另一侧，避免 compact/help 两种 UI 解释不同。

### P2：cache token 造成同一 ESM 模块重复实例

Classic 入口解析到 294 个模块、没有缺 token 的 import，但出现 6 种 query token；`tests/test_training_frontend_modules.py:311-329` 仍要求整张图与入口 token 完全一致。当前实际有 4 个目标模块被不同 token 导入：`dataset-editor/preview.js`、`config-form/form-fields.js`、`form-fields-adapters.js` 和 `config-value-collector.js`。这会让浏览器按不同 URL 建立不同 ESM 实例。

Dragon 全目录 token 扫描也发现 4 个重复目标：`icons.js`、`dataset-editor-fields.js`、`tagging-api.js` 和 `shared/dialog.js`。其中 `shared/dialog.js` 分别以 `module-bootstrap-20260901-dialog-v1` 和 `dragon-ui-20260901v2` 导入。该模块在 `shared/dialog.js:8-10,98,132-133` 持有 `dialogQueue`、`activeDialogRequest` 等模块级 singleton 状态；重复加载会拆成两套队列/host 状态。相关引用包括 `dragon-ui/pages/tagging-results-controller.js` 与 `sample-prompts-dialog.js`。这是当前最需要统一的 token，不应仅把静态断言改成放过重复单例。

### P2：配置帮助文案较旧基线缺少关键限定

当前 `field-help-training.js:450-453` 说明 `base_compute` 选择压缩路径，NF4“目前只给 Krea-2 使用”；但没有包含旧测试期待的“冻结 DiT 底模 Linear 的计算路径”以及“已验证可与 block swap 组合”。`pretrained_model_name_or_path` 帮助在 `:795-800` 已说明 Krea-2 NF4 v2 路径。现有代码仍有模型族过滤逻辑；风险是字段提示不再完整表达底模冻结边界和已验证组合，不等同于选项过滤缺陷。

### P2：配置管理 E2E 的部分入口/事件契约已过期或不稳定

- `training-library-names.spec.ts:56` 寻找当前界面不存在的“配置库管理”文本；现实现由 `TrainingLibraryActions.tsx:85` 直接提供“新建分组”。两个视口测试都因此超时，属于测试过期，不是确认的界面不可达。
- `history-return.spec.ts:60-74` 切换到日志视图后缺少 `/api/training/history/run-15/logs` mock，最终把该 GET 记为 unhandled。生产端 `HistoryLogs.tsx` 会请求此接口，后端 route 已注册；这是测试 fixture 缺项。
- `caption-context.spec.ts:123-129` 的 confirm 处理在单测复跑中竞态：trace 显示单个 confirm event 被 dismiss 与 accept 两次处理，导致浏览器会话中断、job 未切换。产品接受离开路径尚未通过该用例有效验收；先让测试等待并处理唯一 Dialog，再判断是否有产品问题。

- 键盘排序单项复跑在 `getByRole("alert")` 断言失败，发生在 requests 计数断言之前。失败快照显示已命中下一配置；当前 `TrainingLibraryDrag.tsx` 按目标中线判断 `after=false`，`trainingLibraryDrag.ts` 计算后得到与原顺序相同的位置并返回 `null`，`onDragEnd` 只对非空 placement 调用 mutation。源码路径支持“该交互可能不发排序请求”的判断，但本次复跑没有直接测得 POST 数量。

## 静态测试失败分类

11 个 Classic / Dragon 失败不是同一根因：

| 类型 | 失败/现象 | 判定 |
|---|---|---|
| 真实数据漂移 | `test_config_help_summary_matches_full_catalog` | 上述 7 个帮助摘要确实不一致 |
| 真实 cache 一致性风险 | `test_dragon_module_cache_tokens_do_not_duplicate_module_instances` | 4 个 Dragon 文件以多个 query token 导入；包括模块级 singleton `shared/dialog.js` |
| 过期数量快照 | `test_stage_catalog_covers_every_layout_candidate` 期望 217、现值 218 | 实际 218 个唯一 layout key 全部在 catalog 中；缺失列表为空，隐藏 timestep 字段仍在 orthogonal cluster |
| 过期结构/函数快照 | `test_training_config_recomputes_scoped_fields_and_supports_search` 在旧 `config-page.js` 寻找 filter 函数 | 实现已拆至 `dragon-ui/pages/config-field-filter.js`；测试应跟随模块边界 |
| 过期文案断言 | `test_config_form_uses_navigation_search_and_progressive_disclosure`、`test_base_compute_nf4_option_scoped_to_krea2_family` | 选项过滤代码存在；help 文案的旧精确短语已变化，需恢复用户所需信息或更新到语义断言 |
| 过期 release token 快照 | dataset、tagging、Classic entry、V100 catalog 相关 5 个断言 | 测试固定了旧日期 token；更新快照时仍应保留“同一目标模块 token 唯一”和真实资源可达护栏 |

CSS 样式入口检查中，Classic `style.css` 的 26 个 import 和 Dragon `dragon-style.css` 的 4 个 import 目标均存在；CSS token 顺序当前一致。维护风险包括 `90-responsive.css` 在 1000/720px 有重复媒体块，历史工作台局部仍有 `min-width:680px/780px`，以及 32px Queue/History 按钮规则可能覆盖窄屏 44px 通用基线；当前没有 Classic/Dragon 浏览器计算样式证据证明发生溢出或触控失败，列为待测而非已确认 bug。

## API、DOM 与安全边界

- 抽查 Next 训练配置保存/预检/启动/排队、数据集蓝图、历史游标/产物、monitor HTTP/WS、全局/模型设置及 captioning 的主要 method/path/schema，与工作树后端路由整体匹配；在该抽查范围内未发现可确认的 High/Medium API 错配。单项队列操作未用批量命令的 `expected_revision`，但服务仍按 ID 和状态检查，本轮未确认用户可见竞态。
- Classic/Dragon 与 Next 是并存两套 UI；`index.html` 的 DOM id 是共享契约，Classic/Dragon 静态测试覆盖不等于浏览器视觉验收。Dragon/classic 入口切换、Dragon 失败回退已由 bootstrap runtime 测试覆盖；真实 CSS load/error/10 秒超时场景未覆盖。
- 检查的 Next 源码未发现 `dangerouslySetInnerHTML` 或 `innerHTML`，也未在 Dragon 预览 URL 生成链中发现可控 scheme；后端 `web/services/preview/images.py` 与 `weights.py` 生成固定 `/api/preview/...` 相对 URL，并对文件路径做编码/allowlist 解析。因此本轮未发现明显前端 XSS sink 或利用路径，但未完成系统性安全验证。权重历史 `output_dir` 的枚举边界属于相邻后端风险，未纳入本次前端结论。
- 可访问性正面证据包括 Next 的 focus-visible/reduced-motion、跳过链接、共享 dialog focus trap、键盘/焦点单测和多视口双主题 E2E。未完成实体读屏器/手工键盘认证；`trapDialogFocus` 过滤 CSS 隐藏控件、Queue/History 移动端 hit target 等仍需补测。

## 后续顺序

1. 修复键盘 DnD 的“ArrowDown 到下一项”位置语义，并补一个成功移动用例和一个服务端失败后顺序回滚用例。
2. 统一同一模块的 cache token，优先保证 Dragon `shared/dialog.js` 为全图单一实例；同步调整测试以检查每个解析目标唯一，而不是只依赖整图统一字符串。
3. 统一完整/紧凑帮助的事实源，恢复 `base_compute` 的冻结范围、NF4 family 及 block swap 组合边界。
4. 更新历史 token、模块拆分和字段数量快照；为 history logs 增补 E2E mock；重写 caption confirm 用例等待逻辑。
5. 对 Classic/Dragon 做真实浏览器窄屏及 stylesheet load failure 验收；补 Queue/History hit target 和 dialog 隐藏控件的可访问性断言。
6. 在临时静态根验证 Next build、哈希资源保留、previous-index 与原子替换/回退，再重新计算评分卡分数。

## 已落地文档

- [前端健康度评分卡](../features/frontend-health-scorecard.md)：区分 2026-07-11 历史基线与 `dev @ 2e96bd0c` 当前工作树快照，保留 69/D 评分和证据入口。
- [Dragon Next 实施与验收记录（已归档）](../../_archive/docs/features/dragon-next-implementation.md)：明确 P1-P8 是八个主工作区，补充 `/next/datasets/masks` 子路由，并修正历史日志分页/虚拟窗口表述。
- [Dragon Next UI/UX 实施进度（已归档）](../../_archive/docs/features/dragon-next-uiux-progress.md)：将 2026-09-10 的测试数字和截图标为历史快照，指向本轮当前工作树复核。
- [手动蒙版编辑器](../features/manual-mask-editor.md)：补充“应用到子集”会设置 `alpha_mask=true` 的配置副作用。
- [历史任务日志](../features/history-logs.md)：补充搜索关键词空值/长度边界、非对象 JSON 占位和任务目录路径保护。
- [配置工作台](../features/config-workbench.md)：区分 Classic Dragon 配置库与 Next React 当前能力，修正预检阶段、GPU 选择、另存、配置文件/分组管理、TOML 导入导出和续训入口描述，并补充 Next 定向测试入口。
- [打标工作台](../features/tagging-workbench.md)：区分 Classic 结果页与 Next 审阅页，修正图片/任务分页和结构化标注旁路 `.txt` 写回语义。

上述文档均从 `docs/features/README.md` 或本 finding 可达；本轮没有修改前端实现、测试实现或用户数据。
