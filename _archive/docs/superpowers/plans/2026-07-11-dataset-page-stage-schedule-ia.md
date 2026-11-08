# 数据集页重构 + 可变阶段课表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把数据集页收成「精简主卡 + 顶栏双弹窗」，阶段数 N 可变，课表仍写入当前训练配置的 `stage_schedule*`。

**Architecture:** 不改训练引擎语义。数据集行仍在数据集预设；`stage_schedule_enabled` / `stage_schedule[]` 仍在训练配置 draft/TOML。UI 主编辑迁到数据集页顶栏：实验性弹窗只改当前选中 subset；分阶段弹窗复用现有 `stage-resolution.js` 逻辑。训练配置页只保留只读摘要。

**Tech Stack:** 现有 WebUI ES modules、`dataset-state` / `config-state`、aiohttp 静态前端、pytest 源码契约测试、`library/training/stage_schedule.py`。

**Spec:** `docs/superpowers/specs/2026-07-11-dataset-page-stage-schedule-ia-design.md`

## Global Constraints

- 所有用户可见文案用简体中文；代码标识保持现有英文键名。
- N 可变：禁止写死 3 段 / 3 数据集槽位；默认 2 段。
- 课表存储归属：训练配置 `stage_schedule*`，不迁入数据集预设文件。
- 实验性弹窗：只编辑当前选中数据集。
- 双入口：数据集页顶栏同一行。
- 主卡精简：路径、分辨率、分桶开关、重复、标注来源、验证比例常驻。
- 阶段时间：全局 %（`start_pct`/`end_pct` fraction）。
- 预处理完成后按 % 热切；本计划不改训练切换算法。
- 改前端模块后同步 cache token（当前多为 `?v=module-bootstrap-20260707-93`，改动文件时按仓库惯例 bump 相关 import）。
- 热点大文件只做小范围接入，新逻辑优先放到 `web/static/js/features/dataset-editor/` 或 `config-form/` 新文件。
- 后台测试命令加 `timeout 60`，优先 `.venv/bin/python`。
- 工作区可能已有无关/并行改动：只提交本计划相关文件，不 revert 他人改动。

---

## File Map

| 文件 | 职责 |
|---|---|
| `web/static/js/features/anima-app/state/dataset-state.js` | 增加 `selectedDatasetIndex` |
| `web/static/js/features/dataset-editor/toolbar.js` **(new)** | 顶栏双入口按钮 + 实验性弹窗打开 |
| `web/static/js/features/dataset-editor/experimental-dialog.js` **(new)** | 实验性 dialog 渲染（当前选中集） |
| `web/static/js/features/dataset-editor/row.js` | 主卡精简字段；选中高亮；可选去掉卡内实验性折叠条 |
| `web/static/js/features/anima-app/chunks/09-setup-config-group-drop-target.js` | `renderDatasetEditor` 顶栏接入 toolbar |
| `web/static/js/features/anima-app/chunks/10-create-dataset-config-input.js` | `createDatasetEditorItem` 不再内嵌实验性折叠（或改为仅主卡） |
| `web/static/js/features/config-form/stage-resolution.js` | 导出/增强摘要；配置页只读模式入口 |
| `web/static/js/features/anima-app/chunks/04-create-config-group-entry.js` | 配置页去掉可写主按钮，改只读摘要 |
| `web/static/index.html` | 增加实验性 dialog 锚点（若需要） |
| `web/static/css/12-datasets-forge.css` | 顶栏按钮、选中态、精简主卡样式 |
| `tests/test_training_frontend_config_ui.py` | 入口挂载与精简字段契约 |
| `tests/test_stage_schedule.py` | 多段 N 边界（若缺） |

---

### Task 1: 选中数据集状态 + 契约测试骨架

**Files:**
- Modify: `web/static/js/features/anima-app/state/dataset-state.js`
- Modify: `tests/test_training_frontend_config_ui.py`
- Test: `tests/test_training_frontend_config_ui.py`

**Interfaces:**
- Produces: `datasetState.selectedDatasetIndex: number`（默认 `0`）
- Consumes: 无

- [ ] **Step 1: 写失败测试 — 状态字段与顶栏入口契约**

在 `tests/test_training_frontend_config_ui.py` 追加：

```python
def test_dataset_page_toolbar_hosts_experimental_and_stage_entries() -> None:
    """数据集页顶栏承载实验性 + 分阶段入口；配置页不再当主编辑面。"""
    dataset_state = _frontend_module_text("js/features/anima-app/state/dataset-state.js")
    # 新模块落地后改为读真实文件；Task1 先断言状态字段存在。
    assert "selectedDatasetIndex" in dataset_state

    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="stage-resolution-dialog"' in html


def test_stage_schedule_primary_entry_moves_to_dataset_page() -> None:
    """分阶段主入口迁到数据集页后，配置分组不再要求 createOpenStageResolutionDialogButton。"""
    group_entry = _frontend_module_text("js/features/anima-app/chunks/04-create-config-group-entry.js")
    # Task3/5 完成后改为：
    # assert "createOpenStageResolutionDialogButton" not in group_entry
    # assert "createStageScheduleInlineSummary" in group_entry
    assert "createStageScheduleInlineSummary" in group_entry or "createOpenStageResolutionDialogButton" in group_entry
```

先只加强制断言 `selectedDatasetIndex`（其它保持宽松，避免一次红一片）。

- [ ] **Step 2: 跑测试确认失败**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_dataset_page_toolbar_hosts_experimental_and_stage_entries -v
```

Expected: FAIL，提示 `selectedDatasetIndex` 不在 `dataset-state.js`。

- [ ] **Step 3: 最小实现 — 状态字段**

修改 `createDatasetState()`：

```javascript
export function createDatasetState() {
    return {
        // ...existing fields...
        selectedDatasetIndex: 0,
        datasetEditorState: {
            loading: false,
            loaded: false,
            dirty: false,
            dataset_config: '',
            datasets: [],
            defaults: {},
            error: '',
        },
        // ...
    };
}
```

- [ ] **Step 4: 再跑测试**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_dataset_page_toolbar_hosts_experimental_and_stage_entries -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/static/js/features/anima-app/state/dataset-state.js tests/test_training_frontend_config_ui.py
git commit -m "feat(web): track selected dataset index for toolbar dialogs"
```

---

### Task 2: 数据集顶栏 Toolbar 模块（双入口）

**Files:**
- Create: `web/static/js/features/dataset-editor/toolbar.js`
- Modify: `web/static/js/features/anima-app/chunks/09-setup-config-group-drop-target.js`（`renderDatasetEditor` 的 header actions）
- Modify: `web/static/css/12-datasets-forge.css`
- Modify: `tests/test_training_frontend_config_ui.py`
- Test: `tests/test_training_frontend_config_ui.py`

**Interfaces:**
- Consumes: `getDatasetState().selectedDatasetIndex`；`openStageResolutionDialog` from stage-resolution
- Produces:
  - `createDatasetEditorToolbarActions(): HTMLElement`
  - 按钮 id：`btn-dataset-open-experimental`、`btn-dataset-open-stage-schedule`

- [ ] **Step 1: 写失败测试**

替换/收紧 Task1 测试：

```python
def test_dataset_page_toolbar_hosts_experimental_and_stage_entries() -> None:
    toolbar = _frontend_module_text("js/features/dataset-editor/toolbar.js")
    editor_chunk = _frontend_module_text("js/features/anima-app/chunks/09-setup-config-group-drop-target.js")
    assert "btn-dataset-open-experimental" in toolbar
    assert "btn-dataset-open-stage-schedule" in toolbar
    assert "createDatasetEditorToolbarActions" in toolbar
    assert "createDatasetEditorToolbarActions" in editor_chunk
    assert "btn-dataset-open-stage-schedule" in editor_chunk or "createDatasetEditorToolbarActions" in editor_chunk
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_dataset_page_toolbar_hosts_experimental_and_stage_entries -v
```

Expected: FAIL（找不到 `toolbar.js` 或符号）

- [ ] **Step 3: 实现 `toolbar.js`**

```javascript
/**
 * Dataset editor top toolbar: experimental + stage-schedule entries.
 */
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { openStageResolutionDialog } from '../config-form/stage-resolution.js?v=module-bootstrap-20260707-93';
import { openDatasetExperimentalDialog } from './experimental-dialog.js?v=module-bootstrap-20260707-93';

const datasetState = getDatasetState();

export function createDatasetEditorToolbarActions() {
    const actions = document.createElement('div');
    actions.className = 'dataset-editor-actions dataset-editor-toolbar-actions';

    const experimentalBtn = document.createElement('button');
    experimentalBtn.id = 'btn-dataset-open-experimental';
    experimentalBtn.type = 'button';
    experimentalBtn.className = 'btn btn-small';
    experimentalBtn.textContent = '实验性/高级';
    experimentalBtn.title = '编辑当前选中数据集的低频/旧功能字段';
    experimentalBtn.addEventListener('click', () => {
        const rows = datasetState.datasetPresetState?.datasets
            || datasetState.datasetEditorState?.datasets
            || [];
        if (!rows.length) {
            experimentalBtn.title = '请先添加数据集';
            return;
        }
        const idx = Math.max(0, Math.min(Number(datasetState.selectedDatasetIndex) || 0, rows.length - 1));
        datasetState.selectedDatasetIndex = idx;
        openDatasetExperimentalDialog(idx);
    });

    const stageBtn = document.createElement('button');
    stageBtn.id = 'btn-dataset-open-stage-schedule';
    stageBtn.type = 'button';
    stageBtn.className = 'btn btn-small';
    stageBtn.textContent = '分阶段调度';
    stageBtn.title = '按总训练步数百分比切换数据集子集（写入当前训练配置草稿）';
    stageBtn.addEventListener('click', () => openStageResolutionDialog());

    // 调用方还会追加「添加数据集」按钮；这里只返回双入口容器片段。
    actions.append(experimentalBtn, stageBtn);
    return actions;
}
```

若 `experimental-dialog.js` 尚未实现，先在同 Task 建 stub：

```javascript
// experimental-dialog.js stub for Task2 compile graph; Task3 fills body
export function openDatasetExperimentalDialog(index) {
    console.warn('openDatasetExperimentalDialog not implemented', index);
}
```

- [ ] **Step 4: 接入 `renderDatasetEditor` header**

在 `09-setup-config-group-drop-target.js` 的 header actions 中：

```javascript
import { createDatasetEditorToolbarActions } from '../../dataset-editor/toolbar.js?v=module-bootstrap-20260707-93';

// inside renderDatasetEditor:
const actions = document.createElement('div');
actions.className = 'dataset-editor-actions';
const toolbar = createDatasetEditorToolbarActions();
const addBtn = document.createElement('button');
addBtn.type = 'button';
addBtn.className = 'btn btn-small';
addBtn.textContent = '添加数据集';
// ...existing add handler...
actions.append(toolbar, addBtn);
```

CSS（`12-datasets-forge.css`）补：

```css
#tab-datasets .dataset-editor-toolbar-actions {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}
```

- [ ] **Step 5: 跑测试**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_dataset_page_toolbar_hosts_experimental_and_stage_entries -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add web/static/js/features/dataset-editor/toolbar.js \
  web/static/js/features/dataset-editor/experimental-dialog.js \
  web/static/js/features/anima-app/chunks/09-setup-config-group-drop-target.js \
  web/static/css/12-datasets-forge.css \
  tests/test_training_frontend_config_ui.py
git commit -m "feat(web): add dataset toolbar entries for experimental and stage schedule"
```

---

### Task 3: 实验性弹窗（仅当前选中集）

**Files:**
- Modify: `web/static/js/features/dataset-editor/experimental-dialog.js`
- Modify: `web/static/index.html`（dialog 锚点）
- Modify: `web/static/js/features/dataset-editor/row.js`（点击选中）
- Modify: `web/static/js/features/anima-app/chunks/10-create-dataset-config-input.js`（item 不再强制内嵌折叠实验性）
- Modify: `web/static/css/12-datasets-forge.css`
- Modify: `tests/test_training_frontend_config_ui.py`
- Test: `tests/test_training_frontend_config_ui.py`

**Interfaces:**
- Consumes: `createDatasetExperimentalFeaturesEditor(row, index)`（可复用现有 builder，放进 dialog body）
- Produces: `openDatasetExperimentalDialog(index: number): void`
- DOM: `#dataset-experimental-dialog`, `#dataset-experimental-dialog-body`, `#dataset-experimental-dialog-title`

- [ ] **Step 1: 写失败测试**

```python
def test_dataset_experimental_dialog_edits_selected_subset_only() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    dialog = _frontend_module_text("js/features/dataset-editor/experimental-dialog.js")
    row = _frontend_module_text("js/features/dataset-editor/row.js")
    item = _frontend_module_text("js/features/anima-app/chunks/10-create-dataset-config-input.js")

    assert 'id="dataset-experimental-dialog"' in html
    assert "openDatasetExperimentalDialog" in dialog
    assert "selectedDatasetIndex" in row
    assert "createDatasetExperimentalFeaturesEditor(row, index)" in dialog
    # 主列表不再默认拼接卡内实验性折叠
    assert "createDatasetExperimentalFeaturesEditor(row, index)" not in item
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_dataset_experimental_dialog_edits_selected_subset_only -v
```

Expected: FAIL

- [ ] **Step 3: `index.html` 增加 dialog 锚点**

放在 `stage-resolution-dialog` 附近：

```html
<dialog id="dataset-experimental-dialog" class="preview-dialog dataset-experimental-dialog" aria-labelledby="dataset-experimental-dialog-title">
    <div class="preview-dialog-head">
        <div>
            <h2 id="dataset-experimental-dialog-title">实验性/高级</h2>
            <p>仅编辑当前选中的数据集子集。</p>
        </div>
        <button type="button" class="btn btn-small" data-close-dialog>关闭</button>
    </div>
    <div id="dataset-experimental-dialog-body" class="dataset-experimental-dialog-body"></div>
</dialog>
```

关闭按钮复用项目现有 dialog close 绑定方式（搜 `data-close-dialog` / `stage-resolution-dialog` 的 close 模式并保持一致）。

- [ ] **Step 4: 实现 `openDatasetExperimentalDialog`**

```javascript
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { createDatasetExperimentalFeaturesEditor } from './row.js?v=module-bootstrap-20260707-93';
import { normalizeDatasetDefaults } from '../anima-app/helpers/dataset-values.js?v=module-bootstrap-20260707-93';

const datasetState = getDatasetState();

export function openDatasetExperimentalDialog(index) {
    const dialog = document.getElementById('dataset-experimental-dialog');
    const body = document.getElementById('dataset-experimental-dialog-body');
    const title = document.getElementById('dataset-experimental-dialog-title');
    if (!dialog || !body) return;

    const rows = datasetState.datasetPresetState?.datasets
        || datasetState.datasetEditorState?.datasets
        || [];
    if (!rows.length) return;
    const idx = Math.max(0, Math.min(Number(index) || 0, rows.length - 1));
    datasetState.selectedDatasetIndex = idx;
    const row = rows[idx];
    const settings = normalizeDatasetDefaults(row.settings || {});
    if (title) {
        title.textContent = `实验性 · SUBSET ${idx + 1} · ${settings.resolution || '?'}px`;
    }
    body.innerHTML = '';
    // Reuse existing advanced editor; force open state for dialog context.
    const editor = createDatasetExperimentalFeaturesEditor(row, idx);
    editor.classList.add('is-dialog-embedded');
    body.appendChild(editor);

    if (dialog.showModal && !dialog.open) dialog.showModal();
    else if (!dialog.open) dialog.setAttribute('open', 'open');
}
```

注意：若 `row.js` 与 `experimental-dialog.js` 互相 import 成环，把 `createDatasetExperimentalFeaturesEditor` 抽到 `dataset-editor/experimental-panel.js`，或从现有 chunk 再导出一层。**优先避免环依赖。**

- [ ] **Step 5: 卡片选中**

在 `createDatasetEditorRow` / `createDatasetEditorItem` 根节点：

```javascript
wrap.classList.toggle('is-selected', index === datasetState.selectedDatasetIndex);
wrap.addEventListener('click', (event) => {
    // 忽略按钮/输入内点击冒泡误伤时可用 closest
    if (event.target.closest('button, input, select, textarea, a, label')) return;
    datasetState.selectedDatasetIndex = index;
    renderDatasetEditor();
});
```

- [ ] **Step 6: 主列表去掉内嵌实验性**

`createDatasetEditorItem` 改为只 append 主卡：

```javascript
item.append(createDatasetEditorRow(row, index, item));
```

- [ ] **Step 7: 跑测试**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_dataset_experimental_dialog_edits_selected_subset_only -v
```

并跑可能受影响的旧实验性断言：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py -k "experimental or dataset_experimental or stage_schedule" -v
```

Expected: 新测试 PASS；旧测试若仍断言卡内拼接，按本任务语义更新断言到 dialog 路径。

- [ ] **Step 8: Commit**

```bash
git add web/static/index.html \
  web/static/js/features/dataset-editor/experimental-dialog.js \
  web/static/js/features/dataset-editor/row.js \
  web/static/js/features/anima-app/chunks/10-create-dataset-config-input.js \
  web/static/css/12-datasets-forge.css \
  tests/test_training_frontend_config_ui.py
git commit -m "feat(web): move dataset experimental controls into selected-row dialog"
```

---

### Task 4: 分阶段主编辑迁到数据集页；配置页只读摘要

**Files:**
- Modify: `web/static/js/features/anima-app/chunks/04-create-config-group-entry.js`
- Modify: `web/static/js/features/config-form/stage-resolution.js`
- Modify: `tests/test_training_frontend_config_ui.py`
- Test: `tests/test_training_frontend_config_ui.py`

**Interfaces:**
- Consumes: `createStageScheduleInlineSummary()`, `openStageResolutionDialog()`, `stageSchedulePayload()`
- Produces: 配置页只读摘要；主打开按钮仅数据集顶栏

- [ ] **Step 1: 写失败测试**

```python
def test_stage_schedule_primary_entry_moves_to_dataset_page() -> None:
    group_entry = _frontend_module_text("js/features/anima-app/chunks/04-create-config-group-entry.js")
    toolbar = _frontend_module_text("js/features/dataset-editor/toolbar.js")
    stage_ui = _frontend_module_text("js/features/config-form/stage-resolution.js")

    assert "createOpenStageResolutionDialogButton" not in group_entry
    assert "createStageScheduleInlineSummary" in group_entry
    assert "btn-dataset-open-stage-schedule" in toolbar
    assert "Always re-sync from draft || currentConfig" in stage_ui
    assert "key === 'stage_schedule' || key === 'stage_schedule_enabled'" in _frontend_module_text(
        "js/features/config-form/index.js"
    )
```

更新旧测试 `test_stage_schedule_dialog_is_wired_from_dataset_group`：

```python
def test_stage_schedule_dialog_is_wired_from_dataset_group() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    toolbar = _frontend_module_text("js/features/dataset-editor/toolbar.js")
    stage_ui = _frontend_module_text("js/features/config-form/stage-resolution.js")
    assert 'id="stage-resolution-dialog"' in html
    assert "btn-dataset-open-stage-schedule" in toolbar
    assert "stage_schedule_enabled" in stage_ui
    assert "subset_index" in stage_ui
    assert "start_pct" in stage_ui
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_stage_schedule_primary_entry_moves_to_dataset_page -v
```

Expected: FAIL（配置页仍挂 `createOpenStageResolutionDialogButton`）

- [ ] **Step 3: 改配置分组入口**

在 `04-create-config-group-entry.js`：

- 删除/停止调用 `createOpenStageResolutionDialogButton()`
- 保留 `createStageScheduleInlineSummary()` 作为只读摘要
- 摘要内已有「编辑课表」按钮（`createStageScheduleInlineSummary` 内部 `openStageResolutionDialog`）可保留；若要强制「去数据集页」，把按钮文案改成「编辑课表」并仍打开同一 dialog（符合 spec：同一实现，不必真 tab switch）

最小 diff 原则：配置页不再额外放标题级主按钮。

- [ ] **Step 4: 增强摘要文案（轻量）**

在 `createStageScheduleInlineSummary()` 中，当 enabled 且有 stages 时：

```javascript
const resChain = stages.map((s) => {
    // resolution from listSubsetOptions if available
    return s.subset_index;
}).join('→');
detail.textContent = `${stages.length} 段 · subset ${resChain}${dirty ? ' · 未保存' : ''}`;
```

P0 至少显示段数；分辨率链可在同函数用 `listSubsetOptions()` 补齐。

- [ ] **Step 5: 跑测试**

Run:

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_config_ui.py::test_stage_schedule_primary_entry_moves_to_dataset_page \
  tests/test_training_frontend_config_ui.py::test_stage_schedule_dialog_is_wired_from_dataset_group \
  tests/test_stage_schedule.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add web/static/js/features/anima-app/chunks/04-create-config-group-entry.js \
  web/static/js/features/config-form/stage-resolution.js \
  tests/test_training_frontend_config_ui.py
git commit -m "feat(web): make dataset page the primary stage-schedule editor"
```

---

### Task 5: 主卡精简（P1 字段分层）

**Files:**
- Modify: `web/static/js/features/dataset-editor/row.js`（`createDatasetRowSettingsEditor`）
- Modify: `web/static/js/features/dataset-editor/experimental-dialog.js` 或 experimental panel builder（把挪走字段渲染进弹窗）
- Modify: `tests/test_training_frontend_config_ui.py`
- Test: `tests/test_training_frontend_config_ui.py`

**Interfaces:**
- Produces: 主卡 settings 仅含
  - `resolution`
  - `enable_bucket`
  - `validation_split`
  - + 既有 `num_repeats` / 标注来源
- 弹窗额外展示：
  - `min_bucket_reso`, `max_bucket_reso`, `bucket_reso_steps`, `bucket_no_upscale`
  - `validation_split_num`, `validation_seed`

- [ ] **Step 1: 写失败测试**

```python
def test_dataset_main_card_keeps_only_high_frequency_settings() -> None:
    row = _frontend_module_text("js/features/dataset-editor/row.js")
    # 定位 createDatasetRowSettingsEditor 的 fields 列表
    assert "['resolution', 'number']" in row
    assert "['enable_bucket', 'select']" in row
    assert "['validation_split', 'number']" in row
    assert "['min_bucket_reso', 'number']" not in row.split("function createDatasetRowSettingsEditor")[1].split("function ")[0]
    assert "['validation_seed', 'number']" not in row.split("function createDatasetRowSettingsEditor")[1].split("function ")[0]
```

（若字符串切片脆弱，可改为断言主卡 fields 数组字面量整段。）

- [ ] **Step 2: 跑测试确认失败**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_config_ui.py::test_dataset_main_card_keeps_only_high_frequency_settings -v
```

Expected: FAIL

- [ ] **Step 3: 精简主卡 fields**

```javascript
const fields = [
    ['resolution', 'number'],
    ['enable_bucket', 'select'],
    ['validation_split', 'number'],
];
```

- [ ] **Step 4: 弹窗补低频 settings**

在实验性 dialog body 中，于 advanced 控件前插入：

```javascript
function createDatasetAdvancedSettingsEditor(row, index) {
    // same input wiring as createDatasetRowSettingsEditor but only:
    // min_bucket_reso, max_bucket_reso, bucket_reso_steps, bucket_no_upscale,
    // validation_split_num, validation_seed
}
```

键名不变，继续走 `updateDatasetEditorRowsSettingValue` / `createDatasetRowSettingInput`。

- [ ] **Step 5: 跑测试**

Run:

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_config_ui.py::test_dataset_main_card_keeps_only_high_frequency_settings \
  tests/test_training_frontend_config_ui.py -k "dataset_row or experimental or caption" -v
```

Expected: PASS（更新任何仍假设主卡含桶细节的旧断言）

- [ ] **Step 6: Commit**

```bash
git add web/static/js/features/dataset-editor/row.js \
  web/static/js/features/dataset-editor/experimental-dialog.js \
  tests/test_training_frontend_config_ui.py
git commit -m "feat(web): slim dataset main card and move rare settings to dialog"
```

---

### Task 6: 可变 N 课表 hardening（模板/校验/多段测试）

**Files:**
- Modify: `web/static/js/features/config-form/stage-resolution.js`
- Modify: `tests/test_stage_schedule.py`
- Modify: `tests/test_training_frontend_config_ui.py`
- Test: `tests/test_stage_schedule.py`, `tests/test_training_frontend_config_ui.py`

**Interfaces:**
- Consumes: 现有 `applyStageTemplate(n)`, `validate_stage_specs`, `normalize_stage_dicts`
- Produces: 前端模板含「均分当前 N」；后端测试覆盖 N=5

- [ ] **Step 1: 写失败测试（后端多段）**

```python
def test_validate_five_stage_cover_and_resolve():
    specs = parse_stage_specs([
        {"subset_index": 0, "start_pct": 0.0, "end_pct": 0.2},
        {"subset_index": 1, "start_pct": 0.2, "end_pct": 0.4},
        {"subset_index": 0, "start_pct": 0.4, "end_pct": 0.6},
        {"subset_index": 2, "start_pct": 0.6, "end_pct": 0.8},
        {"subset_index": 1, "start_pct": 0.8, "end_pct": 1.0},
    ])
    assert validate_stage_specs(specs, subset_count=3) == []
    assert resolve_stage_index(specs, 0.0) == 0
    assert resolve_stage_index(specs, 0.2) == 1
    assert resolve_stage_index(specs, 0.599) == 2
    assert resolve_stage_index(specs, 1.0) == 4
```

前端：

```python
def test_stage_schedule_ui_is_variable_n_not_hardcoded_three() -> None:
    stage_ui = _frontend_module_text("js/features/config-form/stage-resolution.js")
    assert "applyStageTemplate(2)" in stage_ui
    assert "Math.min(12" in stage_ui or "Math.min(12," in stage_ui
    assert "defaultStageScheduleStages" in stage_ui
    assert "阶段3" not in stage_ui.split("defaultStageScheduleStages")[1].split("export function")[0]
```

- [ ] **Step 2: 跑测试确认失败/通过**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_stage_schedule.py::test_validate_five_stage_cover_and_resolve -v
```

若仅缺测试则 FAIL；实现已支持时应在加测试后 PASS。

- [ ] **Step 3: 前端补「均分」按钮（若缺失）**

在 chart panel actions：

```javascript
const equalBtn = document.createElement('button');
equalBtn.type = 'button';
equalBtn.className = 'btn btn-small';
equalBtn.textContent = '均分当前段';
equalBtn.addEventListener('click', () => {
    applyStageTemplate(Math.max(1, stageResolutionState.stages.length || 2));
});
```

保持软上限 12；默认模板仍 2 段。

- [ ] **Step 4: 跑全量相关测试**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_stage_schedule.py tests/test_training_frontend_config_ui.py -k "stage_schedule or dataset_page or dataset_main_card or dataset_experimental" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/static/js/features/config-form/stage-resolution.js \
  tests/test_stage_schedule.py \
  tests/test_training_frontend_config_ui.py
git commit -m "test(web): harden variable-N stage schedule coverage"
```

---

### Task 7: 收尾回归与文档指针

**Files:**
- Modify: `docs/superpowers/specs/2026-07-11-dataset-page-stage-schedule-ia-design.md`（状态改为「实现中/已落地 P0-P1」）
- Optional: `docs/features/` 若已有 WebUI 数据集说明则补一句入口；没有则不要新建长文

- [ ] **Step 1: 跑回归包**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_stage_schedule.py \
  tests/test_training_frontend_config_ui.py \
  tests/test_web_config_service.py \
  -q
```

Expected: 全绿。若 `test_web_config_service` 与本改无关失败，记录但不扩大修复面。

- [ ] **Step 2: 手工检查清单（执行者勾选）**

1. 数据集页顶栏同一行可见：实验性 / 分阶段 / 添加数据集  
2. 点选 SUBSET 2 → 开实验性 → 标题含 SUBSET 2  
3. 无选中/无数据集时实验性有保护  
4. 分阶段默认 2 段，可加到 5 段，应用到草稿  
5. 训练配置页只有摘要，无第二套大编辑器  
6. 主卡看不到 min/max bucket 字段，弹窗看得到  
7. 保存训练 TOML 后 reload，课表还在  

- [ ] **Step 3: Commit 文档状态**

```bash
git add docs/superpowers/specs/2026-07-11-dataset-page-stage-schedule-ia-design.md
git commit -m "docs: mark dataset stage-schedule IA P0-P1 implementation status"
```

---

## Out of Scope（本计划不写任务）

- P2：缓存文件探测 ✓/✗、监控条 `阶段 1/2`、窄段 hover 全名、分辨率递减 warning UI
- P3：名义 epoch 区间糖、课表存进数据集预设、每段本地 epoch 引擎
- 训练 loop / preprocess 算法改动

---

## Spec Coverage Self-Review

| Spec 项 | Task |
|---|---|
| 顶栏双入口同一行 | Task 2 |
| 实验性=当前选中集 | Task 1 + 3 |
| 分阶段主编辑在数据集页 | Task 2 + 4 |
| 配置页只读摘要 | Task 4 |
| 主卡精简 | Task 5 |
| N 可变 / 默认 2 / 非写死 3 | Task 6（+ 现有 stage-resolution） |
| 存储仍在训练配置 | Task 4（不改 raw_files 语义） |
| 预处理后按 % 切 | 无代码任务（已有引擎；文档约束） |
| 测试/回归 | Task 6 + 7 |

## Placeholder / Consistency Check

- 无 TBD 实现步骤；stub 仅允许在 Task2 短暂存在，Task3 必须填满。
- 按钮 id 统一：`btn-dataset-open-experimental` / `btn-dataset-open-stage-schedule`。
- 状态字段统一：`selectedDatasetIndex`。
- 课表字段统一：`stage_schedule` / `stage_schedule_enabled` / `subset_index` / `start_pct` / `end_pct`。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-11-dataset-page-stage-schedule-ia.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — 每个 Task 新开子代理，Task 间审查  
2. **Inline Execution** — 本会话按 executing-plans 连续推进并设检查点

Which approach?
