# WebUI 设计系统控制台升级 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可复用设计系统（token + primitives + patterns），先接入训练监控与历史，在不改 IA、不减配置项、不加功能的前提下把控制台密度/组件一致性/骨架/质感再升一级。

**Architecture:** 在现有 instrument-panel reskin 之上新增 `web/static/css/ds/` 系统层；页面 CSS 改为消费系统 class/token，而不是继续页级硬编码。严格 A 路径：P0 系统底座 → P1 训练监控 → P2 历史 → P3 回灌其余页。HTML 仅允许新增 class 与最小包裹层，禁止改 DOM id。

**Tech Stack:** 现有静态 WebUI（`web/static` CSS + 最小 HTML class）、pytest 前端契约测试、人工 16:9 浅/深走查。

**Spec:** `docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md`  
**Baseline:** [仪表面板换皮设计](../../proposal/webui-instrument-panel-reskin/specs/2026-07-12-webui-instrument-panel-reskin-design.md)（已完成并归档）\
**Iteration log:** `docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md`

## Global Constraints

- 不加新功能；不删除 / 隐藏 / 合并配置项。
- 不做信息架构重构；只做视觉/布局骨架/组件系统升级。
- 中等边界：不改 DOM id；允许新增 class 与最小 HTML 包裹层。
- 浅色优先，深色单独精修；禁止“浅色反相即深色”。
- 主字段字号只升不降；密度靠壳变薄、间距节奏、信息分层。
- `btn-highlight` / `ui-btn--highlight` 永远弱于 primary。
- 改 CSS import 时必须同步 cache token：`web/static/style.css` 与 `web/static/index.html` 的 `?v=` 保持一致。
- 测试一律：`timeout 60 .venv/bin/python -m pytest ...`（长套件可到 120/180，但优先拆分）。
- 不碰用户数据目录：`configs/web-training-history/`、`configs/web-training-queue/`、`output/`、`models/`。
- 每阶段必须写迭代日志；High 未清零不得进入下一阶段。
- 每阶段至少并行 3 个只读审核角色：`visual-auditor` / `readability-auditor` / `theme-auditor` 或 `contract-auditor`。
- 默认不改 JS 业务；若必须接线 class，只做最小补丁。

---

## File Map

| 区域 | 文件 | 责任 |
|---|---|---|
| 设计 / 计划 / 日志 | `docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md`, `docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade.md`, `docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md` | 需求、任务、每阶段证据 |
| 样式入口 | `web/static/style.css`, `web/static/index.html` | import 顺序与 cache token |
| 既有 token / 按钮 / 字段 | `web/static/css/00-tokens.css`, `02-buttons.css`, `13-shared-fields.css` | 兼容桥；逐步对齐系统 token |
| 设计系统层 | `web/static/css/ds/00-tokens-extend.css`, `ds/10-primitives.css`, `ds/20-patterns.css` | 系统唯一新增主写集 |
| 训练消费方 | `20-training-core.css`, `22-training-queue.css`, `33-training-forge.css` | monitor-board 接入 |
| 历史消费方 | `21-history-panels.css`, `33-training-history-theme.css` | history-board 接入 |
| 契约测试 | `tests/test_webui_visual_tokens.py`, 新增 `tests/test_webui_design_system.py`, 既有 training/history/dom/modules 测试 | 回归门禁 |
| DOM 锚点 | `web/static/index.html` | 仅新增 class / 最小包裹，不改 id |

---

## Gate Library

### G0 每阶段最少门禁

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_webui_design_system.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_dom.py \
  -q
```

Expected: all passed

### G1 训练壳

```bash
timeout 120 .venv/bin/python -m pytest \
  tests/test_training_frontend_queue.py \
  tests/test_training_frontend_live.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

Note: live 若遇已知 JS bridge 基线失败，记录为 non-CSS baseline，不得用业务 JS 大改去“刷绿”，除非单独开任务。

### G2 历史壳

```bash
timeout 120 .venv/bin/python -m pytest \
  tests/test_training_frontend_history.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

### G3 收口

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_webui_design_system.py \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_misc.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

### Visual Gate

浅色 16:9 + 深色同分辨率，检查：

1. 训练监控首屏：工具条/分段/指标/主图/侧栏层级
2. 历史首屏：侧栏工具条化、主列表扫读
3. 主字段字号未回退
4. primary > highlight
5. sticky/侧栏不严重抢首屏

---

## Cache Token 约定

- P0 完成后：`frontend-chain-20260712-ds-p0`
- P1 完成后：`frontend-chain-20260712-ds-p1`
- P2 完成后：`frontend-chain-20260712-ds-p2`
- P3 每页或收口再 bump：`...-ds-p3` / `...-ds-final`

每次 bump 必须同时改：

- `web/static/style.css` 全部 `?v=`
- `web/static/index.html` 的 stylesheet `?v=`

---

### Task 0: 建立迭代日志基线

**Files:**
- Create: `docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md`
- Test: docs only

**Interfaces:**
- Consumes: approved design spec
- Produces: round template + baseline notes

- [ ] **Step 1: 写 iteration log 骨架**

```markdown
# WebUI 设计系统控制台升级迭代日志

## Baseline
- Date: 2026-07-12
- Spec: docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md
- Plan: docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade.md
- Baseline branch/commit: <fill git rev-parse --short HEAD>
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
```

- [ ] **Step 2: 填 baseline commit hash**

```bash
git rev-parse --short HEAD
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md
git commit -m "docs: add WebUI design system console upgrade iteration log"
```

---

### Task 1: 红灯契约 — 设计系统 token/入口

**Files:**
- Modify: `tests/test_webui_visual_tokens.py`
- Create: `tests/test_webui_design_system.py`
- Test: same

**Interfaces:**
- Consumes: current `00-tokens.css`, `style.css`
- Produces: failing contracts that define P0 API

- [ ] **Step 1: 扩展 visual token 测试，要求系统扩展 token 存在**

在 `tests/test_webui_visual_tokens.py` 增加：

```python
def test_design_system_extended_tokens_are_defined() -> None:
    required = [
        "--font-size-title",
        "--font-size-section",
        "--font-size-mono",
        "--control-height-sm",
        "--control-height-md",
        "--control-height-lg",
        "--space-5",
        "--surface-page",
        "--surface-panel",
        "--surface-raised",
        "--surface-input",
        "--surface-sticky",
        "--status-success",
        "--panel-shadow-soft",
    ]
    # tokens may live in 00-tokens.css and/or ds/00-tokens-extend.css
    root = Path(__file__).resolve().parents[1]
    blob = (root / "web/static/css/00-tokens.css").read_text(encoding="utf-8")
    extend = root / "web/static/css/ds/00-tokens-extend.css"
    if extend.exists():
        blob += "\n" + extend.read_text(encoding="utf-8")
    missing = [name for name in required if name not in blob]
    assert not missing, f"missing design-system tokens: {missing}"
```

- [ ] **Step 2: 新建 design system 入口/组件契约测试（先红灯）**

创建 `tests/test_webui_design_system.py`：

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE = (ROOT / "web/static/style.css").read_text(encoding="utf-8")
DS_DIR = ROOT / "web/static/css/ds"


def test_style_entry_imports_design_system_layer() -> None:
    for rel in [
        "./css/ds/00-tokens-extend.css",
        "./css/ds/10-primitives.css",
        "./css/ds/20-patterns.css",
    ]:
        assert rel in STYLE, f"style.css missing import {rel}"


def test_design_system_files_exist() -> None:
    for name in ["00-tokens-extend.css", "10-primitives.css", "20-patterns.css"]:
        assert (DS_DIR / name).is_file(), f"missing {name}"


def test_primitives_define_core_classes() -> None:
    prim = (DS_DIR / "10-primitives.css").read_text(encoding="utf-8")
    for selector in [
        ".ui-btn",
        ".ui-btn--primary",
        ".ui-btn--highlight",
        ".ui-btn--danger",
        ".ui-field",
        ".ui-field__label",
        ".ui-field__control",
        ".ui-segmented",
        ".ui-segmented__btn",
        ".ui-card",
        ".ui-toolbar",
        ".ui-sidebar",
        ".ui-stat",
        ".ui-stat__value",
        ".ui-sticky",
    ]:
        assert selector in prim, f"missing primitive {selector}"


def test_patterns_define_console_boards() -> None:
    patterns = (DS_DIR / "20-patterns.css").read_text(encoding="utf-8")
    for selector in [
        ".page-shell",
        ".workbench",
        ".monitor-board",
        ".history-board",
    ]:
        assert selector in patterns, f"missing pattern {selector}"
```

- [ ] **Step 3: 跑测试确认红灯**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py::test_design_system_extended_tokens_are_defined \
  tests/test_webui_design_system.py \
  -q
```

Expected: FAIL（缺 ds 文件/import/token）

- [ ] **Step 4: Commit 红灯测试**

```bash
git add tests/test_webui_visual_tokens.py tests/test_webui_design_system.py
git commit -m "test: add failing contracts for WebUI design system layer"
```

---

### Task 2: P0 Token 扩展 + 入口接线

**Files:**
- Create: `web/static/css/ds/00-tokens-extend.css`
- Modify: `web/static/css/00-tokens.css`（仅兼容别名，必要时）
- Modify: `web/static/style.css`
- Modify: `web/static/index.html`
- Test: token + design_system import/existence 子集

**Interfaces:**
- Consumes: existing `--font-size-field*`, `--control-height`, `--space-1..4`
- Produces: extended token set used by primitives

- [ ] **Step 1: 创建 `ds/00-tokens-extend.css`**

最小内容（可按主题微调数值，但名字必须齐）：

```css
/* Design-system token extension on top of instrument-panel tokens. */
:root {
    --font-size-title: 1.15rem;
    --font-size-section: 0.95rem;
    --font-size-mono: 0.8rem;

    --control-height-sm: calc(var(--control-height) - 0.25rem);
    --control-height-md: var(--control-height);
    --control-height-lg: calc(var(--control-height) + 0.25rem);

    --space-5: 1.25rem;

    --radius-sm: 4px;
    --radius-md: var(--radius);

    --surface-page: var(--bg);
    --surface-panel: var(--bg-card);
    /* NOTE: existing codebase already defines --surface-raised; do not reassign to itself.
       Expose design-system alias and keep primitives reading var(--surface-raised). */
    --surface-input: var(--bg-input);
    --surface-sticky: color-mix(in srgb, var(--bg-card) 92%, transparent);

    --status-success: var(--success);
    --panel-shadow-soft: 0 6px 14px rgba(2, 8, 20, 0.16);
}

:root[data-theme="light"] {
    --surface-page: var(--bg);
    --surface-panel: var(--bg-card);
    --surface-input: var(--bg-input);
    --surface-sticky: color-mix(in srgb, var(--bg-card) 94%, transparent);
    --panel-shadow-soft: 0 6px 14px rgba(15, 23, 42, 0.05);
}
```

注意：若 `--surface-raised: var(--surface-raised)` 自引用有问题，改为显式色值或改名映射到既有 raised 表面变量（实施时以可计算 CSS 为准）。

- [ ] **Step 2: 在 `style.css` 于 `00-tokens.css` 后插入 ds imports**

顺序：

```css
@import url("./css/00-tokens.css?v=frontend-chain-20260712-ds-p0");
@import url("./css/ds/00-tokens-extend.css?v=frontend-chain-20260712-ds-p0");
@import url("./css/ds/10-primitives.css?v=frontend-chain-20260712-ds-p0");
@import url("./css/ds/20-patterns.css?v=frontend-chain-20260712-ds-p0");
@import url("./css/01-base.css?v=frontend-chain-20260712-ds-p0");
/* ...其余现有 imports 同步 bump 到 ds-p0 ... */
```

- [ ] **Step 3: 同步 `index.html` cache token 到 `frontend-chain-20260712-ds-p0`**

- [ ] **Step 4: 先放空的 primitives/patterns 文件占位，避免 import 404**

```css
/* web/static/css/ds/10-primitives.css */
/* populated in Task 3 */

/* web/static/css/ds/20-patterns.css */
/* populated in Task 4 */
```

- [ ] **Step 5: 跑 token 相关测试**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

Expected: token 测试转绿；design_system 的 primitive/pattern 内容断言仍可能红

- [ ] **Step 6: Commit**

```bash
git add web/static/css/ds/00-tokens-extend.css web/static/css/ds/10-primitives.css web/static/css/ds/20-patterns.css web/static/style.css web/static/index.html
git commit -m "feat(webui): add design-system token extension and entry imports"
```

---

### Task 3: P0 Primitives 组件层

**Files:**
- Modify: `web/static/css/ds/10-primitives.css`
- Optional bridge: `web/static/css/02-buttons.css`, `13-shared-fields.css`（只做兼容映射，不删旧 class）
- Test: `tests/test_webui_design_system.py`

**Interfaces:**
- Consumes: extended tokens
- Produces: `.ui-btn*` `.ui-field*` `.ui-segmented*` `.ui-card` `.ui-toolbar` `.ui-sidebar` `.ui-stat*` `.ui-sticky`

- [ ] **Step 1: 实现 primitives 最小可用样式**

`10-primitives.css` 必须包含（示意，实施可补全细节）：

```css
.ui-btn {
    min-height: var(--control-height-md);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface-input);
    color: var(--text-dim);
    font-size: var(--font-size-field);
    font-weight: 650;
    padding: 0.2rem 0.7rem;
    cursor: pointer;
}
.ui-btn--primary {
    border-color: var(--accent-strong);
    background: var(--accent-strong);
    color: var(--on-accent);
    font-weight: 750;
}
.ui-btn--highlight {
    border-color: color-mix(in srgb, var(--accent) 42%, var(--border));
    background: color-mix(in srgb, var(--accent) 12%, var(--surface-input));
    color: var(--text);
}
.ui-btn--danger {
    border-color: color-mix(in srgb, var(--danger) 50%, var(--border));
    background: color-mix(in srgb, var(--danger) 10%, transparent);
    color: var(--danger);
}

.ui-field { display: grid; gap: var(--space-1); min-width: 0; }
.ui-field__label {
    color: var(--text);
    font-size: var(--font-size-field-label);
    font-weight: 700;
    min-height: var(--control-height-md);
    display: inline-flex;
    align-items: center;
}
.ui-field__control,
.ui-field input,
.ui-field select,
.ui-field textarea {
    min-height: var(--control-height-md);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface-input);
    color: var(--text);
    font-size: var(--font-size-field);
}
.ui-field__help {
    color: var(--text-muted);
    font-size: var(--font-size-meta);
    line-height: 1.35;
}

.ui-segmented {
    display: inline-flex;
    gap: 0.12rem;
    padding: 0.12rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface-input);
}
.ui-segmented__btn {
    min-height: var(--control-height-sm);
    border: 0;
    background: transparent;
    color: var(--text-dim);
    font-size: 0.8rem;
    padding: 0.16rem 0.55rem;
    box-shadow: none;
}
.ui-segmented__btn.is-active,
.ui-segmented__btn.active {
    color: var(--text);
    font-weight: 750;
    box-shadow: inset 0 -2px 0 var(--accent);
}

.ui-card {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface-panel);
    box-shadow: var(--panel-shadow-soft);
}
.ui-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--space-2);
    min-height: calc(var(--control-height-md) + 0.85rem);
    padding: 0.42rem 0.85rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface-panel);
}
.ui-sidebar {
    background: color-mix(in srgb, var(--surface-panel) 94%, var(--surface-page));
    border-color: var(--border);
}
.ui-stat { display: grid; gap: 0.12rem; min-width: 0; }
.ui-stat__value {
    color: var(--text);
    font-size: 1.28rem;
    font-weight: 800;
    line-height: 1.05;
}
.ui-stat__label {
    color: var(--text-dim);
    font-size: var(--font-size-meta);
    font-weight: 700;
}
.ui-sticky {
    position: sticky;
    z-index: 40;
    background: var(--surface-sticky);
    backdrop-filter: blur(8px);
    box-shadow: none;
}
```

- [ ] **Step 2: 兼容桥（可选但推荐）**

在 `02-buttons.css` 末尾增加：

```css
/* Compatibility: existing .btn* remain source of truth for old markup;
   ui-btn is the forward API. Keep highlight weaker than primary. */
```

不要删除 `.btn`；P1/P2 可逐步双挂 class。

- [ ] **Step 3: 跑 design system primitive 测试**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_webui_design_system.py -q
```

Expected: files/import/primitive 断言通过；pattern 若仍空则仅 pattern 失败

- [ ] **Step 4: Commit**

```bash
git add web/static/css/ds/10-primitives.css web/static/css/02-buttons.css web/static/css/13-shared-fields.css
git commit -m "feat(webui): add design-system primitives for console controls"
```

---

### Task 4: P0 Patterns 骨架

**Files:**
- Modify: `web/static/css/ds/20-patterns.css`
- Test: `tests/test_webui_design_system.py` 全绿 + G0

**Interfaces:**
- Consumes: primitives/tokens
- Produces: `.page-shell` `.workbench` `.monitor-board` `.history-board`

- [ ] **Step 1: 实现 patterns**

```css
.page-shell {
    min-height: calc(100vh - var(--header-height));
    background: var(--surface-page);
    color: var(--text);
}

.workbench {
    display: grid;
    grid-template-columns: minmax(0, 1fr) clamp(240px, 22vw, 320px);
    gap: 0;
    align-items: stretch;
    min-height: calc(100vh - var(--header-height));
}
.workbench--sidebar-left {
    grid-template-columns: clamp(240px, 22vw, 320px) minmax(0, 1fr);
}

.monitor-board {
    display: grid;
    grid-template-rows: auto auto auto minmax(0, 1fr);
    gap: var(--space-2);
    min-width: 0;
}
.monitor-board__metrics {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: var(--space-2);
}
.monitor-board__body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(220px, 300px);
    gap: var(--space-2);
    min-height: 0;
}

.history-board {
    display: grid;
    grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
    gap: 0;
    min-height: calc(100vh - var(--header-height));
}
.history-board__tools {
    display: grid;
    gap: var(--space-2);
    padding: var(--space-3);
    border-right: 1px solid var(--border);
    background: var(--surface-panel);
}
.history-board__main {
    min-width: 0;
    padding: var(--space-3);
}
```

- [ ] **Step 2: 跑 G0**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_webui_design_system.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_dom.py \
  -q
```

Expected: PASS

- [ ] **Step 3: 写 P0 iteration log + Commit**

```bash
git add web/static/css/ds/20-patterns.css docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md
git commit -m "feat(webui): add design-system console layout patterns"
```

---

### Task 5: P1 训练监控接入 monitor-board

**Files:**
- Modify: `web/static/index.html`（`#tab-training` 相关容器加 class，不改 id）
- Modify: `web/static/css/33-training-forge.css`
- Modify: `web/static/css/20-training-core.css`
- Modify: cache token → `ds-p1`
- Test: G0 + queue；live 尽力

**Interfaces:**
- Consumes: `.monitor-board`, `.ui-toolbar`, `.ui-segmented`, `.ui-stat`
- Produces: training page as first consumer

- [ ] **Step 1: HTML 最小 class 钩子**

在不改 id 的前提下，给训练根/工具条/分段/指标容器加 class，例如：

- `#tab-training` 增加 `page-shell`
- 训练主布局容器增加 `monitor-board`
- `.training-toolbar` 增加 `ui-toolbar`
- `.training-view-tabs` 增加 `ui-segmented`
- `.training-view-tab` 增加 `ui-segmented__btn`
- 指标卡 value/label 增加 `ui-stat__value` / `ui-stat__label`

先用 `rg` 定位准确 DOM，再改。

- [ ] **Step 2: 训练 CSS 改为消费系统**

重点：

- toolbar 高度/间距用 token
- view tab active 用 segmented 底刻度，不再大块实心抢 primary
- metric value/label 用 `ui-stat` 节奏
- 删除与系统冲突的重复硬编码主路径

- [ ] **Step 3: bump cache 到 `frontend-chain-20260712-ds-p1`**

- [ ] **Step 4: 测试**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_design_system.py \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_queue.py \
  -q
```

- [ ] **Step 5: 审核 + Commit**

```bash
git add web/static/index.html web/static/css/33-training-forge.css web/static/css/20-training-core.css web/static/style.css docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md
git commit -m "feat(webui): mount training monitor on design-system board"
```

---

### Task 6: P1 队列侧栏密扫读

**Files:**
- Modify: `web/static/css/22-training-queue.css`
- Modify: `web/static/css/33-training-forge.css`（队列相关）
- Test: queue + dom

**Interfaces:**
- Consumes: `ui-sidebar`, `ui-card`, meta tokens
- Produces: denser queue without stealing main metrics

- [ ] **Step 1: 队列标题/item/section 改系统节奏**

- 标题用 section/meta，不再巨型 accent 标题
- item 更密，状态色走 status token
- 侧栏背景用 surface，不自造重阴影

- [ ] **Step 2: 测试 + Commit**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_queue.py tests/test_training_frontend_dom.py -q
git add web/static/css/22-training-queue.css web/static/css/33-training-forge.css docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md
git commit -m "feat(webui): densify training queue sidebar with system surfaces"
```

---

### Task 7: P2 历史接入 history-board

**Files:**
- Modify: `web/static/index.html`（历史容器 class）
- Modify: `web/static/css/33-training-history-theme.css`
- Modify: cache token → `ds-p2`
- Test: G0 + history

**Interfaces:**
- Consumes: `.history-board`, `.ui-toolbar`, `.ui-field`, `.ui-sidebar`
- Produces: history as second consumer

- [ ] **Step 1: HTML class 钩子**

- 历史 manager 根增加 `history-board`
- 侧栏工具区增加 `history-board__tools` / `ui-sidebar`
- 主区增加 `history-board__main`
- bulk bar 增加 `ui-toolbar`

- [ ] **Step 2: 历史 theme CSS 消费系统**

- head/stats/tools/bulk 进一步工具化
- 输入高度 `var(--control-height-md)`
- eyebrow/meta 统一
- 去掉与系统冲突的厚 padding/重阴影

- [ ] **Step 3: bump `ds-p2` + 测试**

```bash
timeout 120 .venv/bin/python -m pytest \
  tests/test_webui_design_system.py \
  tests/test_training_frontend_history.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

- [ ] **Step 4: Commit**

```bash
git add web/static/index.html web/static/css/33-training-history-theme.css web/static/style.css docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md
git commit -m "feat(webui): mount training history on design-system board"
```

---

### Task 8: P2 历史列表面板与拖拽降噪

**Files:**
- Modify: `web/static/css/21-history-panels.css`
- Test: history + dom

**Interfaces:**
- Consumes: card/meta/status tokens
- Produces: quieter collection/config cards

- [ ] **Step 1: 卡片头/拖拽手柄/面板标题对齐系统**

- panel title 不用高饱和 accent 大字抢主区
- drag handle 可点、低对比
- compact 模式继续保留，但字号不低于 meta 可读底线

- [ ] **Step 2: 测试 + Commit**

```bash
timeout 120 .venv/bin/python -m pytest tests/test_training_frontend_history.py tests/test_training_frontend_dom.py -q
git add web/static/css/21-history-panels.css docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md
git commit -m "feat(webui): quiet history collection panels via design system"
```

---

### Task 9: P0–P2 总回归与文档阶段收口

**Files:**
- Modify: iteration log
- Optional: `docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md` 状态注记“P0–P2 已实施，P3 待回灌”
- Test: G3

- [ ] **Step 1: 跑 G3**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_webui_visual_tokens.py \
  tests/test_webui_design_system.py \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_misc.py \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  -q
```

- [ ] **Step 2: 写 P0–P2 复盘到 iteration log**

必须包含：

- 系统 API 清单
- 训练/历史前后差异
- High/Medium 残留
- P3 回灌顺序建议

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-12-webui-design-system-console-upgrade-iteration-log.md docs/superpowers/specs/2026-07-12-webui-design-system-console-upgrade-design.md
git commit -m "docs: close design-system P0-P2 console upgrade checkpoint"
```

---

### Task 10: P3 配置页回灌（系统消费者）

**Files:**
- Modify: `web/static/css/11-config-forge.css`, `03-config-shell.css`, `13-shared-fields.css`
- Modify: `web/static/index.html`（config 容器 class）
- Cache bump
- Test: misc + config_ui 子集 + G0

- [ ] **Step 1: config 主壳接 `workbench` / `ui-toolbar` / `ui-field`**
- [ ] **Step 2: 删除与系统重复的主路径硬编码**
- [ ] **Step 3: 更新/守护 `tests/test_training_frontend_misc.py` 布局契约**
- [ ] **Step 4: 测试 + Commit**

```bash
timeout 90 .venv/bin/python -m pytest \
  tests/test_training_frontend_misc.py \
  tests/test_training_frontend_dom.py \
  tests/test_webui_design_system.py \
  -q
git commit -m "feat(webui): migrate config chrome onto design system"
```

---

### Task 11: P3 数据集页回灌

**Files:**
- Modify: `web/static/css/12-datasets-forge.css`, `10-config-dataset-editor.css`
- Test: dom + modules css cache + 必要 config_ui 子集

- [ ] **Step 1: 列表/工具条/行字段接 primitives**
- [ ] **Step 2: 共享 editor 主字段强制 token，避免旧字号回流**
- [ ] **Step 3: 测试 + Commit**

```bash
git commit -m "feat(webui): migrate dataset chrome onto design system"
```

---

### Task 12: P3 工具四页回灌 + 最终收口

**Files:**
- Modify: `40-weight-analysis.css`, `30-preview-settings-dialogs.css`, `41-environment-check.css`, `42-image-test.css`
- Modify: 残留 forge 硬编码清理
- Cache final token
- Test: G3 + weight/image service 冒烟

- [ ] **Step 1: 四页 hero/toolbar/field/stat 接系统**
- [ ] **Step 2: 扫残留 tiny font / heavy shadow，能改 token 的改 token**
- [ ] **Step 3: 最终 G0/G3 + 16:9 浅/深七 Tab 走查**
- [ ] **Step 4: spec 状态改为已完成 + iteration final retro + Commit**

```bash
git commit -m "feat(webui): finish design-system migration across tool tabs"
```

---

## Spec Coverage Self-Check

| Spec 要求 | 对应任务 |
|---|---|
| 设计系统 tokens 扩展 | Task 1–2 |
| primitives | Task 3 |
| patterns（page-shell/workbench/monitor/history） | Task 4 |
| A 路径先系统后页面 | Task 2–4 then 5–8 |
| 训练监控首批消费 | Task 5–6 |
| 历史首批消费 | Task 7–8 |
| 中等边界（class/包裹，不改 id） | Task 5/7 HTML steps |
| 字只升不降 / highlight < primary | Task 3 primitives + 审核清单 |
| 浅色优先深色精修 | tokens-extend light/dark + Visual Gate |
| P3 回灌其余页 | Task 10–12 |
| 审核/门禁/熔断 | Gate Library + 每任务审核步 + Global Constraints |
| 迭代日志 | Task 0/4/5/7/9/12 |

Placeholder scan: none  
Task dependency order: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12

---

## Execution Notes

- 推荐 worktree：`feat/webui-design-system-console-upgrade`
- 推荐执行方式：subagent-driven-development（每任务实现 + 审核）
- 若 live/history/modules 出现已知 JS 基线失败：记录、deselect/说明，不在本视觉任务中扩 scope 修业务桥，除非阻断 DOM 契约且与本次 class 钩子直接相关
