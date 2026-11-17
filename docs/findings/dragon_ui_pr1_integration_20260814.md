# Dragon UI PR #1 集成记录（2026-08-14）

状态：完成  
适用版本：`integration/frontend-pr1`  
线上 PR：<https://github.com/scvxzf1/krea2-webui/pull/1>

本文记录 Dragon UI 前端合并的阶段、门禁、调试结论和最终发布信息。它是维护审计，用户操作说明以完成后的 `docs/features/dragon-ui.md` 为准。

## 集成基线

- `origin/main`：`cd07cd3c7feac714a8e06642c542b56605b9ddff`
- 本地已提交基线：`1211e7b17adbdd19d6f42a8e1fbce48ddd2d01f6`
- PR head：`817d68cf8807bb5b2381d83c2b57f424ade1aac2`
- PR 状态：Open、非 Draft、`MERGEABLE/CLEAN`
- PR 规模：75 个文件，约 `+16,300/-50`，无 CI checks、无 review

主工作区含未提交的用户改动，因此集成在独立 worktree 和 `integration/frontend-pr1` 分支完成。主工作区文件不被 stash、reset 或覆盖；各阶段只把已验证提交推送到同名线上集成分支，最终门禁通过后再更新 `origin/main`。

## 已知合并门禁

1. Dragon 动态入口失败时必须能显式启动 classic UI，不能依赖重复 import 已缓存的 `app.js`。
2. Dragon 与 classic 模式不应无条件下载另一套完整 CSS/JS 入口。
3. 预览删除必须保持输出根/任务目录边界，并覆盖成功、缺失、越界、混合请求。
4. 图片扫描应容忍扫描期间文件被删除或损坏，500 项上限不能在截断前产生无界 metadata 开销。
5. 必须完成项目 `.venv` 下的后端 smoke、前端契约/Node 测试和真实浏览器 smoke。

## 阶段记录

| 阶段 | 状态 | 内容 | 提交/验证 |
| --- | --- | --- | --- |
| 0. 线上核验与隔离 | 完成 | 核验 PR、保护脏工作区、建立本地/线上集成分支 | `6e11da77` |
| 1. 前端入口集成 | 完成 | 导入 PR、修复启动回退与资源加载、补模式测试 | `cd1bf78`；106 项前端/静态契约测试通过 |
| 2. 后端安全加固 | 完成 | 预览删除、图片扫描并发/性能、专项 pytest | `6b0b402`；102 项专项测试通过 |
| 3. 文档与使用引导 | 完成 | 默认模式、classic 回退、部署、故障排查 | `d61f1b2`；8 项文档完整性测试通过 |
| 4. 完整验证与 debug | 完成 | 全量测试、类型检查、真实浏览器 smoke、安全复核 | `20bc35b6`；变更后 2438 项 pytest 通过 |
| 5. 发布 | 完成 | 最终审计、更新 `origin/main`、处理 PR、记录残留 | 功能发布基线 `3b97ea7a`；PR #1 已关闭 |

## 调试与复核入口

```bash
# 线上元数据（GH_TOKEN 只在进程环境中注入，不写入仓库）
GH_TOKEN="$GITHUB_PAT_TOKEN" gh pr view 1 --repo scvxzf1/krea2-webui

# 后端与 WebUI 定向门禁
.venv/bin/python tasks.py test-backend-smoke
.venv/bin/python tasks.py test-focused -- tests/test_web_static_server.py tests/test_preview_service.py

# 文档完整性
timeout 60 .venv/bin/python -m pytest tests/test_documentation_integrity.py -q
```

每一阶段都应在本表补充提交 hash、测试数量、失败原因和修复结论，避免把“GitHub 可合并”误写成“功能已经通过验证”。

## 阶段 1：前端入口集成

本阶段没有直接合入 PR 的三个历史提交，而是在当前 `main` 基线上移植最终前端树，并修复审查中确认的启动缺陷：

- `index.html` 只加载统一的 `ui-bootstrap.js`；classic 与 Dragon 的 JS entry 由 bootstrap 按模式动态导入。
- classic 和 Dragon 不再同时下载完整样式入口。Dragon 使用独立 CSS entry，并通过小型、作用域受限的共享 dialog bridge 复用数据集预览与分阶段调度界面。
- Dragon 初始化失败时会清理 router、导航、主题、动画监听器、DOM、缩放变量和移动菜单状态，再切换 classic stylesheet 并显式调用 `startClassicUI()`。
- `?ui=classic` / `?ui=dragon` 优先于 `localStorage.anima_ui_mode`；无有效显式参数时，存储为 classic 才进入 classic，否则默认 Dragon。
- 新增 Node 运行时测试，实际模拟“Dragon 已部分初始化后抛错”，不再只靠静态字符串断言证明 fallback 存在。

定向验证：

```bash
node --check web/static/js/ui-bootstrap.js
node --check web/static/js/dragon-ui/{index,nav,router,theme,animations}.js

.venv/bin/python -m pytest -q \
  tests/test_dragon_dataset_editor_frontend.py \
  tests/test_dragon_monitor_system_frontend.py \
  tests/test_dragon_shell_performance_frontend.py \
  tests/test_dragon_ui_bootstrap_runtime.py \
  tests/test_web_static_server.py \
  tests/test_training_frontend_history.py \
  tests/test_global_settings_runtime.py \
  tests/test_training_frontend_modules.py \
  tests/test_webui_design_system.py
```

结果：`106 passed`。调试过程中先发现 4 个旧测试仍假设 `app.js` / `style.css` 是 HTML 静态入口，已改为验证新的 bootstrap 常量和 classic 模块图，不通过降低断言强度掩盖入口变化。

## 阶段 2：后端安全加固

本阶段补齐 Dragon 预览工作区依赖的删除 API，并处理大图片目录下的扫描开销和并发变化：

- 新增 `DELETE /api/preview/images`，保留 `deleted`、`missing`、`blocked` 的部分成功 envelope；混合请求返回 HTTP 200 和 `ok=false`，顶层非法请求返回 400，目录不存在返回 404。
- 保留已支持的绝对 `inference_dir` / `custom_dir`，但删除仅允许当前来源目录的直接普通图片文件。嵌套路径、目录、非图片、`..` 和 symlink 都不会删除。
- 单次删除最多 500 个去重后的目标；501 个目标在任何 `unlink` 前整体拒绝。
- Unix 删除路径使用目录 fd、`O_DIRECTORY`、`O_NOFOLLOW`、目录 inode 复核和 `unlink(..., dir_fd=...)`，避免校验后通过替换最终目录或 symlink 改变删除目标；不支持目录 fd 的平台走重复边界检查的保守 fallback。
- 新增共享 bounded Top-K 扫描器。列表仍统计全部可用候选，但只保留最近 K 项并对它们读取尺寸/PNG metadata，复杂度为 `O(N log K)`、额外空间 `O(K)`。
- 训练预览、配置分组合并预览和生图测试图库统一使用 bounded scan；生图测试图库上限从 12 扩展为 500，返回 URL、尺寸和文件大小。
- 扫描期间消失、无法 stat、损坏或变成非普通文件的候选会被跳过；损坏图片仍可列出，尺寸返回 `null`。列表扫描不跟随 symlink，避免读取外部目标的尺寸或 PNG metadata。
- 配置分组的 `days` 过滤已从路由完整传入服务，重复任务目录只扫描一次，最终 metadata 工作量不超过全局 limit。

定向验证：

```bash
.venv/bin/python -m pytest -q \
  tests/test_image_listing.py \
  tests/test_preview_service.py \
  tests/test_cross_domain_delete_boundaries.py \
  tests/test_web_http_contracts.py \
  tests/test_image_test_service.py \
  tests/test_web_route_registry.py

.venv/bin/python -m ruff check \
  web/services/image_listing.py \
  web/services/preview/images.py \
  web/services/image_test_service.py \
  web/routes/preview.py \
  tests/test_image_listing.py \
  tests/test_preview_service.py \
  tests/test_image_test_service.py \
  tests/test_web_http_contracts.py \
  tests/test_web_route_registry.py
```

结果：`102 passed`，Ruff `All checks passed`。独立安全复核发现目录校验和 `os.open` 之间仍可替换最终目录，随后加入 `O_NOFOLLOW` 与 inode 复核；复核还发现列表会跟随图片 symlink，随后统一改为 `lstat()` 并增加 symlink、嵌套目录、混合删除和消失文件测试。

阶段提交已推送到 `origin/integration/frontend-pr1`，GitHub API 校验远端对象为 `6b0b402382c21fbfd5f0632907f0ec72ecd89b37`。随后从该提交建立干净 detached worktree 重跑，结果为 `102 passed, 1 warning`，Ruff 通过。

## 阶段 3：文档与使用引导

本阶段新增 [Dragon UI 与 classic 兼容界面（已归档）](../../_archive/docs/features/dragon-ui.md)，并同步根 README、文档总索引、功能索引、预览、全局设置、全局模型配置和 Linux 部署指南：

- 明确没有有效选择时默认 Dragon，`?ui=classic` / `?ui=dragon` 优先于 `localStorage.anima_ui_mode`。
- 记录 Dragon 左上角界面菜单和 classic **新版界面**按钮的双向切换入口。
- 明确两套 UI 共用后端、配置、队列、历史、模型和输出；`Dragon trainer` 是界面品牌，不是模型族。
- 记录 macOS `preview-dragon-ui.command` 的 `20102`–`20120` 端口选择、约 45 秒等待和 `Ctrl+C` 停止语义。
- 统一常规 `tasks.py web` 示例端口为 20102，同时保留并解释 `launch_webui_app.sh` 默认 20103、不会自动打开浏览器且只检查监听端口的事实。
- 修正文档中非 loopback 可直接启动的旧说法：外部绑定必须使用 `ANIMA_WEBUI_TOKEN` 或 `--token`。
- 增加首次加载约 10–40 秒、`?ui=classic` 排障、静态资源 404、`[dragon-ui] failed to start` 和 `[ui-bootstrap] failed to start any UI` 的 debug 路径。
- 更新预览文档，写明绝对 inference/custom 目录、500 项列表/删除上限、Top-K metadata、部分成功 envelope、直接子文件和 symlink 边界。

验证：

```bash
timeout 120 .venv/bin/python -m pytest -q tests/test_documentation_integrity.py
```

结果：`8 passed`。文档索引可达性、分区索引覆盖、章节锚点和代码围栏均通过。

## 阶段 4：完整验证与 debug

### 自动化门禁

- 全部 `web/static/js/**/*.js` 逐文件执行 `node --check`，无语法错误。
- Dragon/classic 扩展前端套件：`208 passed, 6 warnings in 267.03s`。首次使用 240 秒门限时运行到约 79% 且无失败，随后把门限调整为 600 秒完成全套，不把超时误报为用例失败。
- 后端 WebUI smoke：`225 passed, 2 warnings in 57.86s`。
- Pyright 默认检查面与本次改动文件均为 `0 errors`。
- 阶段 2 提交在干净 detached worktree 复验：`102 passed, 1 warning`。
- 初始全量 pytest：`2437 passed, 6 skipped, 38 warnings in 493.49s`。

独立安全复核确认 Unix 删除路径已经使用 `O_NOFOLLOW`、目录 inode 比对和 `dir_fd` unlink，Top-K 列表也已用 `lstat()` 拒绝 symlink。复核同时发现图片列表虽不展示 symlink，但 `/api/preview/image` 仍可直读允许目录内的 symlink。提交 `20bc35b6` 在既有 allowlist 后对用户实际提交路径执行 `lstat()`，要求请求项本身是普通文件，并增加同目录 symlink 回归测试。

安全补丁后的验证：

```bash
.venv/bin/python -m pytest -q \
  tests/test_preview_service.py \
  tests/test_image_listing.py \
  tests/test_web_http_contracts.py

npx pyright web/services/preview/images.py tests/test_preview_service.py
.venv/bin/python -m pytest -q
```

结果：定向测试 `64 passed in 6.65s`，Pyright `0 errors`，最终全量测试 `2438 passed, 6 skipped, 39 warnings in 510.71s`。新增 warning 来自 `albumentations` 在线版本检查的 TLS 握手超时，不是应用测试失败。

### 真实浏览器验证

浏览器服务使用独立配置根和临时预览目录启动在 `127.0.0.1:20118`，没有修改环境检测中显示的真实训练输出根。验证结束后已停止服务并删除临时目录。

- 桌面路由：训练中心、训练配置、数据集蓝图、训练历史、训练队列、全局设置、全局模型配置、生图测试、预览工作区和环境检测均可加载。
- 模式切换：从 Dragon 左上角菜单实际进入 `?ui=classic`，无参数 `/` 继续加载 classic；再点击 classic 的 **新版界面**进入 `?ui=dragon`，无参数 `/` 继续加载 Dragon。两条持久化链均通过黑盒验证，没有读取 `localStorage` 内部值。
- 预览删除：自定义来源最初显示 3 张临时 PNG；浏览器勾选并确认删除 `browser-2.png` 后，页面显示 2/2 张，GET API 返回 `count=2,total=2`，磁盘仅保留 `browser-0.png`、`browser-1.png`。
- 移动端：在 390 x 844 视口验证导航抽屉、训练中心、训练配置、数据集蓝图、训练历史和自定义预览画廊；页面没有水平滚动，关键标题、按钮、表单和图片卡片没有越过视口。
- 控制台：没有应用 JavaScript error 或静态资源 404；仅出现 Codex Electron 开发环境的 CSP warning。

浏览器验证中的 JavaScript confirm 由真实页面按钮触发并接受，删除结果再通过新标签页、HTTP API 和磁盘清单交叉确认。临时移动端 viewport override 已在结束前恢复，浏览器验证标签页已清理。

## 阶段 5：发布

发布前执行 `git fetch origin --prune`，确认 `origin/main` 仍为集成基线 `cd07cd3c7feac714a8e06642c542b56605b9ddff`，当前集成分支不落后于远端，merge-base 也是该提交，因此可使用普通 fast-forward push，不需要 force push 或历史改写。

功能与阶段 4 审计首先以 `3b97ea7a70ea3cd729aee62bd8792af1ce207c6a` 发布到 `origin/main`，GitHub commits API 返回相同对象。主工作区的本地 `main` 未切换、未 reset、未 stash，原有未提交文件没有纳入本次发布；发布始终从独立 worktree 的 `integration/frontend-pr1` 执行。

PR #1 没有直接 merge。收口评论列出替代提交、测试门禁和文档入口：<https://github.com/scvxzf1/krea2-webui/pull/1#issuecomment-5292379133>。随后 PR 状态更新为 Closed，原 head `817d68cf8807bb5b2381d83c2b57f424ade1aac2` 仅保留为历史参考。

本发布记录随当前审计提交再次 fast-forward 到 `origin/integration/frontend-pr1` 和 `origin/main`；远端 `main` 的当前对象是最终发布 hash。临时 WebUI 服务、浏览器标签页、测试预览目录和 viewport override 均已清理，不遗留后台服务。
