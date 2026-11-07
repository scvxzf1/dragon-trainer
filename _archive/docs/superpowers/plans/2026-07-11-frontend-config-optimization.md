# 前端健康治理与配置优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改训练语义的前提下，把 WebUI 前端从健康分 61(D) 推到 >=78，并同时还清关键工程债与配置体验债。

**Architecture:** 三轨并行、按轮串行收口。工程轨治理 chunks/bridge/token；配置体验轨治理来源/命名/门禁；交互壳轨治理 CSS/DOM/文档。所有任务走 TDD 与固定门禁包。

**Tech Stack:** 现有 WebUI ES modules、catalog、pytest 源码契约测试、可选 Node fixture。

**Spec:** `docs/superpowers/specs/2026-07-11-frontend-config-optimization-design.md`
**Scorecard:** `docs/features/frontend-health-scorecard.md`
**Protocol:** `docs/superpowers/specs/2026-07-11-five-round-auto-iteration-protocol.md`
**Log:** `docs/superpowers/plans/2026-07-11-fullstack-auto-iteration-log.md`

## Global Constraints

- 用户可见文案简体中文；代码标识英文。
- 测试一律 `timeout 60` + `.venv/bin/python`。
- 不重写框架，不一次删光 chunks。
- 改 JS/CSS 必须同步 cache token（完成 token 单源前继续手改）。
- 不新增 globalThis 业务导出。
- 不删除/移动用户 history、queue、output、models。
- 每个 Task：失败测试 → 红灯 → 最小实现 → 绿灯 → 域回归 → 记录。

## File Map

| 区域 | 主要文件 | 责任 |
|---|---|---|
| 评分/协议/日志 | `docs/features/frontend-health-scorecard.md`, `docs/superpowers/**` | 可持久推进入口 |
| 启动与装配 | `web/static/app.js`, `web/static/js/features/anima-app/index.js` | bootstrap / import 顺序 |
| 过渡层 | `anima-app/chunks/*`, `anima-app/helpers/*-bridge.js` | 退役与 fail-fast |
| 配置表面 | `web/static/js/config/catalog/*`, `features/config-form/*` | 命名/默认值/快捷按钮/来源 |
| 样式壳 | `web/static/style.css`, `web/static/css/*`, `index.html` | 层叠与 DOM 契约 |
| 测试护栏 | `tests/frontend_test_support.py`, `tests/test_training_frontend_*.py` | 红绿门禁 |

## Auto Iteration Queue

| Round | 任务 | 目标分 |
|---|---|---:|
| R1 | S0, T0, U0, C2a | 66 |
| R2 | C2b, C3, C4, T1 | 70 |
| R3 | E1, E2, C1, C6 | 74 |
| R4 | C5, E3, U1 | 76 |
| R5 | E4, U2, Freeze | 78+ |

## Debug Gate Library

### G0 快速红灯

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py::test_frontend_module_graph_follows_production_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_module_cache_tokens_match_entrypoint \
  tests/test_training_frontend_modules.py::test_frontend_css_import_cache_tokens_match_entrypoint \
  tests/test_training_frontend_modules.py::test_anima_app_global_this_writes_do_not_grow \
  tests/test_training_frontend_modules.py::test_split_frontend_features_do_not_write_global_this \
  tests/test_training_frontend_dom.py \
  -q
```

### G1 架构

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py \
  tests/test_training_frontend_dom.py \
  tests/test_training_frontend_state.py \
  -q
```

### G2 队列/live

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_live.py \
  tests/test_training_frontend_queue.py \
  tests/test_training_queue.py \
  tests/test_training_queue_resume.py \
  -q
```

### G3 历史/预览

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_history.py \
  tests/test_preview_service.py \
  tests/test_training_history_list.py \
  -q
```

### G4 配置

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_config_ui.py \
  tests/test_web_config_service.py \
  tests/test_web_preflight_compat_matrix.py \
  tests/test_config_provenance.py \
  -q
```

### G5 收口

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_*.py \
  tests/test_weight_analysis_service.py \
  tests/test_image_test_service.py \
  -q
```

---

### Task S0: 冻结评分入口与结构预算清单

**Round:** R1
**Files:**
- Modify: `docs/features/README.md`
- Modify: `docs/features/frontend-health-scorecard.md`
- Modify: 迭代日志
- Test: 文档链接可达

**Interfaces:**
- Produces: 评分结构、规模快照、High 基线

- [ ] **Step 1: 确认评分卡与 features 索引互链**
- [ ] **Step 2: 生成结构预算快照**

```bash
python3 - <<'PY'
from pathlib import Path
chunks=list(Path('web/static/js/features/anima-app/chunks').glob('*.js'))
bridges=list(Path('web/static/js/features/anima-app/helpers').glob('*-bridge.js'))
print('chunks', len(chunks), 'bridges', len(bridges))
PY
```

- [ ] **Step 3: 写入预算规则**
  - `ALLOW_NEW_LOGIC_IN_CHUNKS=false`
  - `ALLOW_FEATURE_IMPORT_CHUNKS=false`（分域灰度）
  - `ALLOW_LEGACYROOT_FALLBACK=false`（分域灰度）
- [ ] **Step 4: 记录基线 61/D**
- [ ] **Step 5: docs commit

---

### Task T0: 同步 globalThis baseline 与快速红灯

**Round:** R1
**Files:**
- Modify: `tests/frontend_test_support.py`
- Modify: `tests/test_training_frontend_state.py`（若重复常量）
- Test: `tests/test_training_frontend_modules.py`

- [ ] **Step 1: 跑 baseline 相关测试，记录现状**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_modules.py::test_anima_app_global_this_writes_do_not_grow -q
```

- [ ] **Step 2: 把缺失 chunk 补进 `ANIMA_APP_GLOBAL_THIS_BASELINE`**
  - `26a-global-settings.js`
  - `26b-preview-view.js`
  - `26c-queue-view.js`
  - `26d-history-list.js`
- [ ] **Step 3: 收敛 state 测试重复常量，只留 support 单源**
- [ ] **Step 4: 跑 G0 + G1，期望 PASS**
- [ ] **Step 5: Commit**

```bash
git add tests/frontend_test_support.py tests/test_training_frontend_state.py
git commit -m "test(web): sync frontend globalThis baseline with live chunks"
```

---

### Task U0: CSS 入口顺序与 shared-fields 断头止血

**Round:** R1
**Files:**
- Modify: `web/static/style.css`
- Modify: `web/static/css/13-shared-fields.css`
- Modify: CSS cache token（`style.css` / `index.html`）
- Test: modules CSS token / misc

- [ ] **Step 1: 增加顺序契约测试**

```python
def test_style_import_order_puts_responsive_last():
    text = STYLE_CSS_PATH.read_text(encoding="utf-8")
    assert text.index('90-responsive.css') > text.index('42-image-test.css')
```

- [ ] **Step 2: 跑测试确认红灯**
- [ ] **Step 3: 把 `90-responsive.css` 移到 feature CSS 之后，并 bump token**
- [ ] **Step 4: 修复 `13-shared-fields.css` 文件头孤立声明，补完整选择器**
- [ ] **Step 5: 回归 modules + misc**
- [ ] **Step 6: Commit**

```bash
git add web/static/style.css web/static/css/13-shared-fields.css web/static/index.html tests
git commit -m "fix(web): restore CSS cascade order and shared field header"
```

---

### Task C2: guide/variant 与 gui-methods 同步

**Round:** R1 清单 / R2 落地
**Files:**
- Modify: `web/static/js/config/catalog/guides.js`
- Test: `tests/test_training_frontend_config_ui.py`

- [ ] **Step 1: 写失败测试**

```python
def test_variant_guides_match_gui_methods_or_legacy_aliases():
    from pathlib import Path
    import re
    gui = {p.stem for p in Path('configs/gui-methods').glob('*.toml')}
    guides = Path('web/static/js/config/catalog/guides.js').read_text(encoding='utf-8')
    keys = set(re.findall(r'^\s{4}([A-Za-z0-9_]+):\s*choiceHelp\(', guides, re.M))
    legacy = {'lora_longer', 'tlora_ortho', 'hydralora_sigma', 'fera'}
    unknown = sorted(keys - gui - legacy)
    assert unknown == [], unknown
```

- [ ] **Step 2: 跑红**
- [ ] **Step 3: 现网变体补齐；ghost 移入 legacy 别名并标注**
- [ ] **Step 4: 跑 G4**
- [ ] **Step 5: Commit**

---

### Task C3: 资源命名三层分层

**Round:** R2
**Files:**
- Modify: `web/static/js/features/anima-app/helpers/app-constants.js`
- Modify: `web/static/js/config/catalog/guides.js`
- Test: config_ui + `tests/test_config.py -k balanced_16g`

分层：
1. 硬件 preset
2. 方法变体
3. 资源快捷按钮

- [ ] **Step 1: 文案契约测试（保留稳定 id，只改 label/note）**
- [ ] **Step 2: 跑红 → 改文案 → 跑绿**
- [ ] **Step 3: Commit**

---

### Task C4: 快捷按钮 diff 预览 + 方法门禁

**Round:** R2
**Files:**
- Modify: `web/static/js/features/config-form/stage-resolution-presets.js`
- Modify: `app-constants.js` / 按钮渲染点
- Test: config_ui

**Interfaces:**
- Produces: `previewQuickPresetDiff(preset, currentValues)`
- Produces: `isQuickPresetApplicable(preset, methodFamily)`

- [ ] **Step 1: 钩子存在性失败测试**
- [ ] **Step 2: 跑红**
- [ ] **Step 3: 实现 diff 列表 + 方法不适用禁用/强警告**
- [ ] **Step 4: G4 回归**
- [ ] **Step 5: Commit**

---

### Task T1: DOM contract 扩展 + Node harness 起步

**Round:** R2
**Files:**
- Modify: `tests/frontend_test_support.py`
- Modify: `tests/test_training_frontend_dom.py`
- Optional: queue/live node fixture

- [ ] **Step 1: 定义 critical DOM ids（保存/入队/启动/历史/预览/全局设置）**
- [ ] **Step 2: 测试 index.html 必含这些 id**
- [ ] **Step 3: 若有 node，补 1 个最小 harness smoke**
- [ ] **Step 4: G1 + G2**
- [ ] **Step 5: Commit**

---

### Task E1: 统一路径 formatter

**Round:** R3
**Files:**
- Modify: `web/static/js/shared/format.js`
- Modify: history/queue/dataset 路径展示点
- Test: history/queue/modules

**Interfaces:**
- Produces: `formatPathLabel(path, {mode, maxLength})`
- mode: `length | basename | parent-basename`

- [ ] **Step 1: 契约测试要求导出 formatPathLabel**
- [ ] **Step 2: 实现并用 compactPathLabel 兼容包装**
- [ ] **Step 3: 截断节点补 title=full path**
- [ ] **Step 4: 回归 history/queue/modules**
- [ ] **Step 5: Commit**

---

### Task E2: 高频 bridge fail-fast 收敛

**Round:** R3
**Files:**
- Modify: history/config/toml 相关 `*-bridge.js`
- Modify: configure 调用点
- Test: modules + history + config_ui

- [ ] **Step 1: 选 history-task-actions 或 config-form 做模板**
- [ ] **Step 2: 未 configure 时不得静默成功**
- [ ] **Step 3: 去掉该域 legacyRoot 默认成功路径**
- [ ] **Step 4: 校验装配顺序**
- [ ] **Step 5: 回归**
- [ ] **Step 6: Commit**

---

### Task C1: 字段来源徽标 + 保存前 diff

**Round:** R3
**Files:**
- Modify: `web/static/js/features/config-form/index.js`
- Optional backend: provenance attach on load
- Test: provenance + config_ui

**Interfaces:**
- Produces: FieldPresentation `{key,value,source,isUiDefault,isDirty,conflicts}`

- [ ] **Step 1: 后端 provenance 单测保底**
- [ ] **Step 2: 前端来源徽标钩子失败测试**
- [ ] **Step 3: 最小实现来源显示 + 保存前 dirty diff**
- [ ] **Step 4: G4**
- [ ] **Step 5: Commit**

---

### Task C6: FORM_UI_DEFAULTS 校准

**Round:** R3-R4
**Files:**
- Modify: `web/static/js/config/catalog/defaults.js` 与相关 help
- Test: config_ui

- [ ] **Step 1: snapshot `lr_scheduler` 等关键键**
- [ ] **Step 2: 禁止 help/defaults 与 base 关键事实冲突**
- [ ] **Step 3: 修文案或标记 ui_only_default**
- [ ] **Step 4: G4**
- [ ] **Step 5: Commit**

---

### Task C5: 表单 live 兼容提示前移

**Round:** R4
**Files:**
- Modify: config-form change handlers
- Test: preflight compat matrix + config_ui

- [ ] **Step 1: 选定互斥组合（selective+gc、soft tokens+block swap 等）**
- [ ] **Step 2: 测试可产出 conflict code**
- [ ] **Step 3: 实现 live 警告，不替代 preflight**
- [ ] **Step 4: G4**
- [ ] **Step 5: Commit**

---

### Task E3: 启动 import 分组并行

**Round:** R4
**Files:**
- Modify: `web/static/js/features/anima-app/index.js`
- Test: modules

- [ ] **Step 1: 标注无依赖分组**
- [ ] **Step 2: Promise.all 并行，且保持 bridge configure 顺序**
- [ ] **Step 3: G1 + smoke G2**
- [ ] **Step 4: Commit**

---

### Task U1: 关键 DOM id 注册表

**Round:** R4
**Files:**
- Modify: `tests/test_training_frontend_dom.py`
- Optional: scorecard 引用

- [ ] **Step 1: 固化 critical id 列表测试**
- [ ] **Step 2: 给 queue/history/preview/settings 补 required/optional（不做大爆炸重命名）**
- [ ] **Step 3: G1/G3**
- [ ] **Step 4: Commit**

---

### Task E4: history 列表渲染性能

**Round:** R5
**Files:**
- Modify: `web/static/js/features/history-list/*` 与相关 chunk shim
- Test: history

- [ ] **Step 1: 建立大批量渲染可测契约（优先分片，不必一次上虚拟列表）**
- [ ] **Step 2: 实现分片/增量渲染**
- [ ] **Step 3: G3**
- [ ] **Step 4: Commit**

---

### Task U2: docs/features 对齐真实 UI

**Round:** R5
**Files:**
- Modify: `docs/features/README.md`
- Create: config-workbench / dataset-editor / training-queue / history-collections / preview / global-settings 最小文档
- Update: `ui-scale.md` 改为用户向

- [ ] **Step 1: 索引补齐**
- [ ] **Step 2: 每篇含入口、关键配置项、危险项、相关测试**
- [ ] **Step 3: `git diff --check -- docs`**
- [ ] **Step 4: Commit**

---

### Task Freeze: 五轮收口

**Round:** R5
**Files:**
- Modify: 迭代日志 + scorecard 基线

- [ ] **Step 1: 跑 G5**
- [ ] **Step 2: 复评健康分，目标 >=78 或记录残留 High**
- [ ] **Step 3: 冻结未完成项**
- [ ] **Step 4: docs commit**

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|---|---|
| 规范评分结构 | S0 + scorecard |
| 快速审核流程 | S0 + protocol |
| 工程债治理 | E1 E2 E3 E4 T0 |
| 配置优化项 | C1 C2 C3 C4 C5 C6 |
| CSS/DOM 止血 | U0 U1 |
| 严格 debug 门禁 | G0-G5 |
| 五轮自动迭代 | Round 映射 + Freeze |
| 文档入口 | U2 |

## Self-Review

- 每项都有文件、命令、验收，无空 TODO
- 配置轨与工程轨都覆盖
- 写集按轮拆分，降低 index/bridge 冲突
- 所有门禁带 timeout 60 与 `.venv/bin/python`
