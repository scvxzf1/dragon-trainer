# WebUI 设计系统控制台升级迭代日志

## Baseline
- Date: 2026-07-12
- Spec: docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md
- Plan: docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade.md
- Baseline branch/commit: eead355a
- Current baseline strengths:
  - instrument-panel reskin completed
  - field tokens exist: --font-size-field / --font-size-field-label / --control-height
- Current pain:
  - page forge hardcodes still dominate training/history chrome
  - no reusable primitive/pattern layer under ds/
  - monitor/history not yet true console boards

## Round Template
### P?
- Goal:
- Write set:
- Changes:
- Supplemental review:
- Cross review:
  - visual-auditor:
  - readability-auditor:
  - theme-auditor / contract-auditor:
- Tests run:
- Results:
- High open:
- Medium open:
- Decision: continue / rework / circuit-break

## Rounds

### P0-baseline (Task 0)
- Goal: 建立迭代日志与基线
- Write set: iteration log only
- Changes: baseline notes + template
- Supplemental review: docs only
- Cross review: n/a
- Tests run: none
- Results: GREEN
- High open: none
- Decision: continue

### P0 (Task 1–4)
- Goal: 设计系统底座 token + primitives + patterns + 契约
- Write set:
  - tests/test_webui_visual_tokens.py
  - tests/test_webui_design_system.py
  - web/static/css/ds/00-tokens-extend.css
  - web/static/css/ds/10-primitives.css
  - web/static/css/ds/20-patterns.css
  - style.css / index.html cache `frontend-chain-20260712-ds-p0`
- Changes:
  - 红灯契约后实现系统层
  - ui-btn/field/segmented/card/toolbar/sidebar/stat/sticky
  - page-shell/workbench/monitor-board/history-board
- Supplemental review: 无功能/无 DOM id/无配置项删减
- Cross review: visual/readability/contract PASS for system layer
- Tests: G0 17 passed
- High open: none
- Decision: continue

### P1 (Task 5)
- Goal: 训练监控页接入 monitor-board / ui-toolbar / ui-segmented / ui-stat
- Write set:
  - web/static/index.html（仅 class / 最小 monitor-board 包裹）
  - web/static/css/20-training-core.css
  - web/static/css/33-training-forge.css
  - web/static/style.css cache `frontend-chain-20260712-ds-p1`
  - iteration log
- Changes:
  - `#tab-training` 加 `page-shell`
  - 工具条 + workspace 外包 `monitor-board`
  - toolbar / view tabs / metrics 加系统 class
  - 训练 CSS 消费 token 与 segmented 底刻度 active；主指标字号不降
  - cache token 全量 bump 到 ds-p1
- Supplemental review: 不改 DOM id / 不删配置项 / 不改 JS 业务
- Cross review: self PASS — ids intact, additive classes only, main metric values not shrunk
- Tests run:
  - required suite: 17 passed (design_system + training DOM + cache token + queue)
  - G0: 17 passed (visual_tokens + design_system + modules cache + DOM)
- Results: GREEN
- High open: none
- Medium open: training page still uses local --training-* palette (expected; theme unification later)
- Decision: continue

### P1 (Task 6)
- Goal: 队列侧栏密扫读，接入 system surface / meta / status 节奏
- Write set:
  - web/static/css/22-training-queue.css
  - web/static/css/33-training-forge.css（仅队列/侧栏相关）
  - iteration log
- Changes:
  - 侧栏背景消费 surface 混色（对齐 ui-sidebar），去掉重阴影
  - 队列 panel / section / item 压密间距，标题改 section/meta 节奏
  - 状态色改走 --status-running / success / error / warning / idle
  - 队列 item 主标题字号只升不降；不碰 monitor 主指标
  - cache 保持 ds-p1
- Supplemental review: 不改 DOM id / index.html / JS / 用户数据
- Cross review: self PASS — queue densified; main metrics selectors untouched
- Tests run:
  - required: tests/test_training_frontend_queue.py tests/test_training_frontend_dom.py → 12 passed
- Results: GREEN
- High open: none
- Medium open: 侧栏仍保留本地 --training-* 色板（与 Task 5 一致，后续主题统一）
- Decision: continue

### P1-fix (Task 5 review follow-up)
- Goal: 修复 metric-item grid/justify 垂直节奏
- Write set: `33-training-forge.css`
- Changes: `.metric-item` 恢复 `flex` 列布局 + `space-between`，保留高卡与 ui-stat 字号语义
- Supplemental review: Task 5 Important 项
- Tests: queue + dom 子集
- Decision: continue

### P2 (Task 7)
- Goal: 历史管理台接入 history-board / ui-toolbar / ui-sidebar
- Write set:
  - web/static/index.html（仅历史 class 钩子 + cache）
  - web/static/css/33-training-history-theme.css
  - web/static/style.css cache `frontend-chain-20260712-ds-p2`
  - iteration log
- Changes:
  - `#training-history-manager` 加 `history-board`
  - tools 加 `history-board__tools` + `ui-sidebar`
  - table panel 加 `history-board__main`
  - bulk bar 加 `ui-toolbar`
  - 主题 CSS 消费 control-height-md / space / meta；覆盖 generic history-board 两列 pattern，保留既有 head/stats/tools/bulk+content 区域网格
  - 去掉 head/stats/tools/bulk 重阴影；输入高度走 `var(--control-height-md)`
  - 主字段字号只升不降（stat strong / bulk title）
- Supplemental review: 不改 DOM id / 不改 21-history-panels / 不改 JS 业务
- Cross review: self PASS — pattern mount without IA break
- Tests run: design_system + history + cache token (+ optional DOM)
- Tests run:
  - required: design_system + history + cache token → 21 passed, 1 failed
  - optional: tests/test_training_frontend_dom.py → 8 passed
- Results: GREEN for CSS/cache/DOM; history JS baseline FAIL known (`test_history_manager_frontend_hooks_are_present` expects selectedHistoryCollectionKey assignment not present in bootstrap)
- High open: none
- Medium open: 列表/拖拽降噪留给 Task 8；本地 --training-* 色板仍在
- Decision: continue

### P2 (Task 8)
- Goal: 历史列表面板/拖拽降噪
- Write set: `21-history-panels.css` + iteration log
- Changes:
  - panel/collection 标题降到 section/meta 节奏，去高饱和抢主
  - drag handle 可点、低对比、hover 提亮
  - compact meta 字号抬到 meta 底线
  - card 去重阴影
- Supplemental review: 无 DOM id/JS/cache 变更
- Tests: history + dom（允许 known JS baseline 1 fail）
- Decision: continue

### P0–P2 Checkpoint (Task 9)
- Goal: 总回归 + 文档阶段收口
- Write set: iteration log + design spec status note
- Tests run (G3):
  - `test_webui_visual_tokens.py`
  - `test_webui_design_system.py`
  - `test_training_frontend_dom.py`
  - `test_training_frontend_misc.py`
  - `test_frontend_css_import_cache_tokens_match_entrypoint`
  - Result: **21 passed**
- Known non-G3 baseline (history JS, not CSS):
  - `test_history_manager_frontend_hooks_are_present` still fails on `selectedHistoryCollectionKey = collection.key`
- Results: GREEN for design-system P0–P2 CSS path
- High open: none
- Medium open:
  - training/history still use local `--training-*` palettes
  - residual hard-coded tiny fonts remain in deep history panel chrome (partially floored in Task 8)
  - config/dataset/tool tabs still page-forge heavy (P3)
- Decision: continue to P3

#### 系统 API 清单
| Layer | API |
|---|---|
| Tokens extend | `--font-size-section`, surface/panel/meta extensions, status tokens bridge |
| Primitives | `ui-btn` (+primary/highlight/danger), `ui-field`, `ui-segmented`/`__btn`, `ui-card`, `ui-toolbar`, `ui-sidebar`, `ui-stat`/`__value`/`__label`, `ui-sticky` |
| Patterns | `page-shell`, `workbench`/`--sidebar-left`, `monitor-board`/`__metrics`/`__body`, `history-board`/`__tools`/`__main` |
| Cache | `frontend-chain-20260712-ds-p2` |

#### 训练/历史前后差异
| 区域 | 前 | 后 |
|---|---|---|
| 训练壳 | forge 硬编码 toolbar/tabs/metrics | `page-shell` + `monitor-board` + `ui-toolbar` + `ui-segmented` + `ui-stat` |
| 训练 tabs | 大块实心 active 易抢 primary | 底刻度 active |
| 主指标字号 | 已较大但不统一 | primary 1.48 保留；ETA/secondary 只升不降；metric-item flex 节奏修复 |
| 队列侧栏 | 重阴影/巨型标题 | surface densify + section/meta + status token |
| 历史管理 | 区域网格但无 board 语义 | `history-board` 挂点 + tools/sidebar/toolbar；输入 `control-height-md` |
| 历史列表 | 高饱和标题/拖拽对比偏抢 | panel title 降噪；drag handle 低对比可点；compact meta 底线 |

#### P3 回灌顺序建议
1. 配置页 `workbench` / `ui-toolbar` / `ui-field`（Task 10）
2. 数据集页列表/行字段（Task 11）
3. 工具四页 + 残留 forge 硬编码清理 + final cache（Task 12）

### P3 (Task 10)
- Goal: 配置页回灌 workbench / ui-toolbar / ui-field
- Write set: index.html class hooks, 11-config-forge.css, 03-config-shell.css, 13-shared-fields.css, style cache ds-p3, iteration log
- Changes:
  - `#tab-config` → page-shell；layout → workbench；manager → ui-sidebar；toolbar → ui-toolbar
  - forge/shell 以 additive 规则消费系统，不改 misc 契约选择器文本
  - 主字段高度/字号强制 token floor
  - cache `ds-p2` → `ds-p3`
- Tests: misc + dom + design_system + cache
- Decision: continue

### P3 (Task 11)
- Goal: 数据集页列表/工具条/行字段接 primitives
- Write set: index.html hooks, 12-datasets-forge.css, 10-config-dataset-editor.css, iteration log
- Changes: page-shell/workbench/ui-sidebar/ui-toolbar; main field token floor
- Cache: keep ds-p3
- Decision: continue

### P3 Final (Task 12)
- Goal: 工具四页回灌 + 最终收口
- Write set: weight/env/image/preview-settings CSS + index hooks + cache `ds-final` + docs
- Changes:
  - weight/env/image/settings 挂 page-shell / workbench / ui-toolbar / ui-sidebar
  - 主字段 token floor
  - 部分重阴影改为 panel-shadow-soft / none
  - cache `frontend-chain-20260712-ds-final`
- Tests: G3 + design_system + dom/misc
- Decision: complete branch pending merge review

### Final whole-branch review (post-merge main)
- Date: 2026-07-12
- HEAD: 9c6729d2 on local main (ff merge from feat/webui-design-system-console-upgrade)
- Gates on main: visual_tokens + design_system + dom + misc + queue + cache → **25 passed**
- Whole-branch review verdict: **Approved for local main**
- Critical/Important: none
- Known non-blocking:
  - history JS baseline `selectedHistoryCollectionKey`
  - forge still overrides some patterns (mount + override architecture)
  - residual chrome title densify / tiny meta outside main fields
- Push webui/main: not performed

