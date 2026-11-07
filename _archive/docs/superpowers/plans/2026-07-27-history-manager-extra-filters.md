# History Manager Extra Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在「训练 → 历史任务」全局搜索下新增「训练变体 / 预处理精度 / 块交换精度」三个下拉筛选，语义与历史详情概览实时指标 chip 一致。

**Architecture:** 方案 A——`_history_summary` 只读解析 `config.snapshot.toml`，向列表 task 注入 `training_variant` / `preprocess_precision` / `block_swap_precision`；前端用与现有 `kind/state/source` 同构的 filter 状态与 `<select>` 消费这些字段。变体/精度解析规则与详情 chip 对齐：Python helper 供列表用，JS 抽 shared 模块供详情与（若需要）对拍测试用；**列表过滤只读 API 字段**。

**Tech Stack:** Python 3.13 + `tomllib`，aiohttp WebUI，vanilla ES modules，pytest 字符串/集成断言。

**Spec:** `docs/superpowers/specs/2026-07-27-history-manager-extra-filters-design.md`

## Global Constraints

- 沟通与 UI 文案用简体中文；代码键名保持现有英文风格。
- 「训练变体」= 方法族（`lokr` 等），**不是** `task.variant` 配置 stem。
- 不写回用户 `meta.json` / 历史目录；只读 snapshot。
- 不做 mtime 缓存、不做动态精度选项 append、不新增「训练精度」筛选。
- 热点大文件只做小范围接入；新逻辑放小模块。
- 验证：`timeout 60 .venv/bin/python -m pytest …`；改完 `git diff --check` 干净。
- 前端 import 继续使用现有 cache token：`?v=module-bootstrap-20260714-stage-dataset5`（本任务不强制换 token，除非项目其它改动已换）。

## File map

| 文件 | 职责 |
|---|---|
| **Create** `web/services/training/history_config_chips.py` | 从 snapshot 文本/路径推导三字段 |
| **Modify** `web/services/training/history_store.py` | `_history_summary` 写入三字段 |
| **Create** `web/static/js/features/history-detail/config-chips.js` | 与详情同源的 JS 解析/格式化（export） |
| **Modify** `web/static/js/features/history-detail/overview.js` | 改用 shared export，删掉内联重复实现 |
| **Modify** `web/static/index.html` | 三个 `<select>` |
| **Modify** `web/static/js/features/anima-app/state/history-state.js` | filter 默认键 |
| **Modify** `web/static/js/features/history-list/task-collections.js` | `syncHistoryFilterControls` map |
| **Modify** `web/static/js/features/history-list/collections-workbench.js` | 过滤、stat reset/active、search text |
| **Modify** `web/static/js/features/app-shell/event-listeners-setup.js` | `historyFilterMap` |
| **Modify** `web/static/js/features/app-shell/event-listeners-contract.js` | id 列表 |
| **Modify** `web/static/js/features/app-shell/beginner-tooltips.js` | 三条 tooltip |
| **Modify** `tests/test_training_history_list.py` | summary 三字段断言 |
| **Modify** `tests/test_training_frontend_history.py` | DOM/filter/state/overview import 断言 |

---

### Task 1: Backend snapshot chip helper + history summary fields

**Files:**
- Create: `web/services/training/history_config_chips.py`
- Modify: `web/services/training/history_store.py`（`_history_summary` 末尾写入字段，约在 `out["metric_count"] = …` 之后、`return out` 之前）
- Test: `tests/test_training_history_list.py`（追加用例；可继续用文件顶部对 `training_resume_test_support` 的 globals 注入）

**Interfaces:**
- Produces:
  - `history_config_chips_from_snapshot_text(text: str, *, variant: str = "") -> dict[str, str]`  
    keys: `training_variant`, `preprocess_precision`, `block_swap_precision`（值均为 `str`，缺省 `""`）
  - `history_config_chips_for_task_dir(task_dir: Path, *, variant: str = "") -> dict[str, str]`  
    读 `task_dir / "config.snapshot.toml"`；不存在或读失败 → 三键 `""`
  - `_history_summary(...)` 保证返回上述三键

- [ ] **Step 1: 写失败单测（无 helper / 无字段时会红）**

在 `tests/test_training_history_list.py` 末尾追加：

```python
def test_history_summary_includes_config_chip_fields_from_snapshot(tmp_path, monkeypatch):
    from web.services.training import history_store as history_store_impl

    history_dir = tmp_path / "history"
    task_id = "20260727-chip-training-imported-demo"
    snapshot = "\n".join(
        [
            "network_module = \"networks.lora_anima\"",
            "use_lokr = true",
            'preprocess_precision_preference = "bf16"',
            'block_swap_transfer_dtype = "fp8_e4m3"',
            'mixed_precision = "bf16"',
        ]
    ) + "\n"
    task_dir = _write_group_task(
        history_dir,
        task_id,
        job="training",
        variant="okkotsu_goddess_demo",
        started_at=2000.0,
        config_text=snapshot,
    )
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    summary = history_store_impl._history_summary(meta, task_dir)

    assert summary["training_variant"] == "lokr"
    assert summary["preprocess_precision"] == "bf16"
    assert summary["block_swap_precision"] == "fp8_e4m3"
    # 配置 stem 仍是 variant，不能被 chip 覆盖
    assert summary["variant"] == "okkotsu_goddess_demo"


def test_history_summary_config_chips_empty_without_snapshot(tmp_path, monkeypatch):
    from web.services.training import history_store as history_store_impl

    history_dir = tmp_path / "history"
    task_id = "20260727-chip-nosnap"
    task_dir = _write_group_task(history_dir, task_id, job="training", started_at=2100.0)
    (task_dir / "config.snapshot.toml").unlink()
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    summary = history_store_impl._history_summary(meta, task_dir)

    assert summary["training_variant"] == ""
    assert summary["preprocess_precision"] == ""
    assert summary["block_swap_precision"] == ""


def test_history_config_chips_hydralora_and_tlora_from_text():
    from web.services.training.history_config_chips import history_config_chips_from_snapshot_text

    hydra = history_config_chips_from_snapshot_text(
        'use_moe_style = "shared_A"\nnetwork_module = "networks.lora_anima"\n',
        variant="whatever",
    )
    assert hydra["training_variant"] == "hydralora"

    tlora = history_config_chips_from_snapshot_text(
        "use_timestep_mask = true\nnetwork_module = \"networks.lora_anima\"\n",
        variant="tlora-8gb",
    )
    assert tlora["training_variant"] == "tlora"

    chimera = history_config_chips_from_snapshot_text(
        "use_chimera_hydra = true\nuse_moe_style = \"shared_A\"\n",
        variant="x",
    )
    assert chimera["training_variant"] == "chimera"
```

- [ ] **Step 2: 跑测确认失败**

Run:

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_history_list.py::test_history_summary_includes_config_chip_fields_from_snapshot tests/test_training_history_list.py::test_history_summary_config_chips_empty_without_snapshot tests/test_training_history_list.py::test_history_config_chips_hydralora_and_tlora_from_text -q
```

Expected: FAIL（ImportError 或 KeyError / assert 缺字段）。

- [ ] **Step 3: 实现 `history_config_chips.py`**

创建 `web/services/training/history_config_chips.py`：

```python
"""Derive history-list chip fields from config.snapshot.toml.

Aligned with web/static/js/features/history-detail overview chips:
training variant family, preprocess precision, block-swap transfer dtype.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_KNOWN_VARIANTS = frozenset(
    {
        "lora",
        "lokr",
        "loha",
        "vera",
        "glora",
        "dora",
        "hydralora",
        "reft",
        "tlora",
        "ortholora",
        "chimera",
        "chimera_hydra",
        "soft_tokens",
        "ip_adapter",
        "easycontrol",
    }
)

_EMPTY = {
    "training_variant": "",
    "preprocess_precision": "",
    "block_swap_precision": "",
}


def history_config_chips_for_task_dir(task_dir: Path, *, variant: str = "") -> dict[str, str]:
    snapshot = Path(task_dir) / "config.snapshot.toml"
    try:
        if not snapshot.is_file():
            return dict(_EMPTY)
        text = snapshot.read_text(encoding="utf-8")
    except OSError:
        return dict(_EMPTY)
    return history_config_chips_from_snapshot_text(text, variant=variant)


def history_config_chips_from_snapshot_text(text: str, *, variant: str = "") -> dict[str, str]:
    raw = str(text or "")
    if not raw.strip():
        return dict(_EMPTY)
    # 与详情「无配置快照 / 无法生成」占位一致：不当作有效配置
    if raw.lstrip().startswith("#") and (
        "无配置快照" in raw or "无法生成配置快照" in raw
    ):
        # 仍可能后面有真内容；仅当几乎全是占位时才空。简单策略：tomllib 失败再空。
        pass
    try:
        data = tomllib.loads(raw)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    flat = _flatten_toml_dict(data)
    return {
        "training_variant": _infer_training_variant(flat, raw, variant=variant),
        "preprocess_precision": _norm_precision(flat.get("preprocess_precision_preference")),
        "block_swap_precision": _norm_precision(flat.get("block_swap_transfer_dtype")),
    }


def _flatten_toml_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Root keys win; also merge one-level tables so sectioned snapshots still work."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for nested_key, nested_val in value.items():
                out.setdefault(str(nested_key), nested_val)
        else:
            out[str(key)] = value
    return out


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _norm_precision(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text


def _infer_training_variant(flat: dict[str, Any], raw_text: str, *, variant: str) -> str:
    module_name = str(flat.get("network_module") or "").lower()
    moe_style = str(flat.get("use_moe_style") or "").strip().lower()

    if _truthy(flat.get("use_chimera_hydra")) or "chimera" in module_name:
        return "chimera"
    if _truthy(flat.get("use_ip_adapter")) or "ip_adapter" in module_name:
        return "ip_adapter"
    if _truthy(flat.get("use_easycontrol")) or "easycontrol" in module_name:
        return "easycontrol"
    if "soft_tokens" in module_name:
        return "soft_tokens"
    if _truthy(flat.get("use_loha")):
        return "loha"
    if _truthy(flat.get("use_lokr")):
        return "lokr"
    if _truthy(flat.get("use_vera")):
        return "vera"
    if _truthy(flat.get("use_glora")):
        return "glora"
    if _truthy(flat.get("dora_wd")) or _truthy(flat.get("use_dora")):
        return "dora"
    if _truthy(flat.get("add_reft")):
        return "reft"
    if moe_style and moe_style not in {"", "false", "none", "0", "off"}:
        return "hydralora"
    if _truthy(flat.get("use_timestep_mask")):
        return "tlora"
    if _truthy(flat.get("use_ortho")):
        return "ortholora"
    if "lora_anima" in module_name:
        return "lora"

    has_snapshot = bool(raw_text.strip()) and not (
        raw_text.lstrip().startswith("#")
        and ("无配置快照" in raw_text or "无法生成配置快照" in raw_text)
        and "network_module" not in raw_text
        and "use_" not in raw_text
    )
    if has_snapshot and flat and not module_name:
        # 有效 dict 但无 module：与前端 hasSnapshot && !moduleName → lora 对齐需谨慎。
        # 前端：hasSnapshot 且无 module → lora。有任意训练键时同理。
        if any(k for k in flat if k not in {"output_dir", "output_name"}):
            return "lora"
        if "network_module" not in raw_text and any(
            token in raw_text
            for token in ("mixed_precision", "preprocess_precision", "network_args", "rank")
        ):
            return "lora"

    variant_key = str(variant or "").strip().lower()
    if variant_key in _KNOWN_VARIANTS:
        return "chimera" if variant_key == "chimera_hydra" else variant_key
    compact = variant_key.replace("-8gb", "") if variant_key.endswith("-8gb") else variant_key
    # 前端是 replace(/-8gb$/, '')
    if variant_key.endswith("-8gb"):
        compact = variant_key[: -len("-8gb")]
    if compact in _KNOWN_VARIANTS:
        return "chimera" if compact == "chimera_hydra" else compact
    return ""
```

实现时以**与 `formatHistoryTrainingVariant` 同序同分支**为准；若上面 `has_snapshot` 分支过复杂，可简化为：

```python
    if flat.get("network_module") is None and raw_text.strip() and "lora_anima" not in module_name:
        # 仅当 tomllib 成功且存在非空配置时，前端在 !moduleName && hasSnapshot 返回 lora
        if module_name == "" and raw_text.strip() and not raw_text.lstrip().startswith("# 无配置快照"):
            # 与 JS：hasSnapshot && !moduleName return 'lora'
            if "无法生成配置快照" not in raw_text:
                # 若 flat 只有 output_* 测试夹具，不要误判为 lora
                meaningful = {
                    k for k in flat
                    if k not in {"output_dir", "output_name"} and flat.get(k) not in (None, "")
                }
                if meaningful:
                    return "lora"
```

单测夹具都带明确 flag，**优先保证有 flag 的路径正确**；`hasSnapshot → lora` 用额外小单测钉死即可。

- [ ] **Step 4: 接入 `_history_summary`**

在 `web/services/training/history_store.py`：

1. 顶部增加：

```python
from web.services.training.history_config_chips import history_config_chips_for_task_dir
```

2. 在 `_history_summary` 的 `out["metric_count"] = ...` 之后、`return out` 之前：

```python
    chips = history_config_chips_for_task_dir(
        task_dir,
        variant=str(out.get("variant") or ""),
    )
    out["training_variant"] = chips["training_variant"]
    out["preprocess_precision"] = chips["preprocess_precision"]
    out["block_swap_precision"] = chips["block_swap_precision"]
```

- [ ] **Step 5: 跑测确认通过**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_history_list.py::test_history_summary_includes_config_chip_fields_from_snapshot tests/test_training_history_list.py::test_history_summary_config_chips_empty_without_snapshot tests/test_training_history_list.py::test_history_config_chips_hydralora_and_tlora_from_text tests/test_training_history_list.py::test_history_store_keeps_direct_history_meta_helpers -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add web/services/training/history_config_chips.py web/services/training/history_store.py tests/test_training_history_list.py
git commit -m "feat(history): expose training variant and precision chips on list summary"
```

---

### Task 2: Frontend shared config-chips module + overview 改用 import

**Files:**
- Create: `web/static/js/features/history-detail/config-chips.js`
- Modify: `web/static/js/features/history-detail/overview.js`
- Test: `tests/test_training_frontend_history.py`（overview 相关断言改为 shared 模块 + overview import）

**Interfaces:**
- Produces (ESM exports):
  - `readConfigString(configText, key) -> string|undefined`
  - `readConfigBool(configText, key) -> boolean`
  - `formatHistoryTrainingPrecision(configText) -> string`
  - `formatHistoryPreprocessPrecision(configText) -> string`
  - `formatHistoryBlockSwapPrecision(configText) -> string`
  - `formatHistoryTrainingVariant(task, configText) -> string`  // 显示用，空时 `'-'`
  - `inferHistoryTrainingVariant(task, configText) -> string`  // 过滤/API 对齐用，空时 `''`（可选；若 format 已返回 `'-'`，过滤侧不要用 format）

说明：详情 chip 继续显示 `'-'`；列表 API 用 `""`。JS shared 保持现有 format 行为（`'-'`），列表过滤**不**依赖 JS infer。

- [ ] **Step 1: 更新前端测试断言（先红）**

找到 `tests/test_training_frontend_history.py` 中断言 overview 内联 `function formatHistoryTrainingVariant` 的测试（约 1386–1407 行），改为：

```python
    chips_source = _frontend_module_text("js/features/history-detail/config-chips.js")
    assert "export function formatHistoryTrainingVariant" in chips_source
    assert "export function formatHistoryPreprocessPrecision" in chips_source
    assert "export function formatHistoryBlockSwapPrecision" in chips_source
    assert "export function formatHistoryTrainingPrecision" in chips_source
    assert "use_lokr" in chips_source
    assert "preprocess_precision_preference" in chips_source
    assert "block_swap_transfer_dtype" in chips_source
    assert "from './config-chips.js?v=module-bootstrap-20260714-stage-dataset5'" in overview_source
    assert "formatHistoryTrainingVariant(task, payload.config_toml)" in overview
    # 内联定义应消失
    assert "function formatHistoryTrainingVariant(task, configText)" not in overview_source
```

保留 chip 行调用断言。若原测试函数名是整段 overview 结构测，只改 helper 相关行，勿删 progress/path 断言。

- [ ] **Step 2: 跑相关测试确认失败**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_history.py -k "overview or chip or TrainingPrecision or detail_overview" -q
```

若 `-k` 匹配不到，直接跑包含该断言的测试函数全名（先 `rg -n "formatHistoryTrainingVariant" tests/test_training_frontend_history.py` 定位 `def test_...`）。

Expected: FAIL（缺 config-chips.js 或 overview 仍内联）。

- [ ] **Step 3: 创建 `config-chips.js`**

把 `overview.js` 里现有 `readConfigString` / `readConfigBool` / `formatHistoryTrainingPrecision` / `formatHistoryPreprocessPrecision` / `formatHistoryBlockSwapPrecision` / `formatHistoryTrainingVariant` **原样搬出**为 top-level `export function`（逻辑不要改，避免 chip 回归）。

文件头：

```javascript
/**
 * History config chip helpers (training variant / precisions).
 * Shared by history detail overview; keep in sync with
 * web/services/training/history_config_chips.py for list filters.
 */
```

`formatHistoryTrainingVariant` 全文从 overview 剪切，含 known Set 与 `-8gb` 处理；返回 `'-'` 的行为不变。

- [ ] **Step 4: 改 `overview.js`**

在 import 区增加：

```javascript
import {
    formatHistoryTrainingPrecision,
    formatHistoryTrainingVariant,
    formatHistoryPreprocessPrecision,
    formatHistoryBlockSwapPrecision,
} from './config-chips.js?v=module-bootstrap-20260714-stage-dataset5';
```

删除 factory 内部上述四个 format + `readConfigString` + `readConfigBool` 的 function 定义（若其它内部函数仍用 `readConfigString`，改为也从 config-chips import）。

调用处保持：

```javascript
['训练精度', formatHistoryTrainingPrecision(payload.config_toml), 'chip'],
['训练变体', formatHistoryTrainingVariant(task, payload.config_toml), 'chip'],
['预处理精度', formatHistoryPreprocessPrecision(payload.config_toml), 'chip'],
['块交换精度', formatHistoryBlockSwapPrecision(payload.config_toml), 'chip'],
```

- [ ] **Step 5: 跑测通过**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_history.py -q --tb=line 2>&1 | tail -30
```

Expected: 与本任务相关的断言 PASS；若文件内其它无关失败为预存问题，记录但不在本任务「修复全世界」。本任务改动引起的失败必须修。

- [ ] **Step 6: Commit**

```bash
git add web/static/js/features/history-detail/config-chips.js web/static/js/features/history-detail/overview.js tests/test_training_frontend_history.py
git commit -m "refactor(history): extract overview config chip helpers to shared module"
```

---

### Task 3: DOM + state + event wiring for three filters

**Files:**
- Modify: `web/static/index.html`（`#history-filter-source` 与 `#history-sort-mode` 之间）
- Modify: `web/static/js/features/anima-app/state/history-state.js`
- Modify: `web/static/js/features/history-list/task-collections.js`（`syncHistoryFilterControls`）
- Modify: `web/static/js/features/app-shell/event-listeners-setup.js`
- Modify: `web/static/js/features/app-shell/event-listeners-contract.js`
- Modify: `web/static/js/features/app-shell/beginner-tooltips.js`
- Test: `tests/test_training_frontend_history.py`（及若 contract 有独立测试则同步）

**Interfaces:**
- Produces filter keys on `historyManagerFilters`:
  - `trainingVariant: 'all'`
  - `preprocessPrecision: 'all'`
  - `blockSwapPrecision: 'all'`
- DOM ids:
  - `history-filter-training-variant`
  - `history-filter-preprocess-precision`
  - `history-filter-block-swap-precision`

- [ ] **Step 1: 写/更新失败断言**

在 `tests/test_training_frontend_history.py` 增加或扩展：

```python
def test_history_manager_extra_filter_controls_are_wired():
    html = (ROOT / "web/static/index.html").read_text(encoding="utf-8")  # 若测试里已有 INDEX_HTML/ROOT 常量则复用
    assert 'id="history-filter-training-variant"' in html
    assert 'id="history-filter-preprocess-precision"' in html
    assert 'id="history-filter-block-swap-precision"' in html
    assert "<span>训练变体</span>" in html
    assert "<span>预处理精度</span>" in html
    assert "<span>块交换精度</span>" in html

    state_src = _frontend_module_text("js/features/anima-app/state/history-state.js")
    assert "trainingVariant: 'all'" in state_src
    assert "preprocessPrecision: 'all'" in state_src
    assert "blockSwapPrecision: 'all'" in state_src

    setup = _frontend_module_text("js/features/app-shell/event-listeners-setup.js")
    assert "'history-filter-training-variant': 'trainingVariant'" in setup
    assert "'history-filter-preprocess-precision': 'preprocessPrecision'" in setup
    assert "'history-filter-block-swap-precision': 'blockSwapPrecision'" in setup

    coll = _frontend_module_text("js/features/history-list/task-collections.js")
    assert "'history-filter-training-variant': 'trainingVariant'" in coll

    contract = _frontend_module_text("js/features/app-shell/event-listeners-contract.js")
    assert "'history-filter-training-variant'" in contract
    assert "'history-filter-preprocess-precision'" in contract
    assert "'history-filter-block-swap-precision'" in contract

    tips = _frontend_module_text("js/features/app-shell/beginner-tooltips.js")
    assert "'history-filter-training-variant'" in tips
    assert "'history-filter-preprocess-precision'" in tips
    assert "'history-filter-block-swap-precision'" in tips
```

（`ROOT` / `_frontend_module_text` 以该测试文件现有 helper 为准，不要新造路径工具。）

- [ ] **Step 2: 跑测确认失败**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_history.py::test_history_manager_extra_filter_controls_are_wired -q
```

Expected: FAIL

- [ ] **Step 3: 改 `index.html`**

在 `history-filter-source` 的 `</label>` 之后、`history-sort-mode` 的 `<label>` 之前插入：

```html
                            <label>
                                <span>训练变体</span>
                                <select id="history-filter-training-variant">
                                    <option value="all">全部</option>
                                    <option value="lora">lora</option>
                                    <option value="lokr">lokr</option>
                                    <option value="loha">loha</option>
                                    <option value="vera">vera</option>
                                    <option value="glora">glora</option>
                                    <option value="dora">dora</option>
                                    <option value="hydralora">hydralora</option>
                                    <option value="reft">reft</option>
                                    <option value="tlora">tlora</option>
                                    <option value="ortholora">ortholora</option>
                                    <option value="chimera">chimera</option>
                                    <option value="soft_tokens">soft_tokens</option>
                                    <option value="ip_adapter">ip_adapter</option>
                                    <option value="easycontrol">easycontrol</option>
                                </select>
                            </label>
                            <label>
                                <span>预处理精度</span>
                                <select id="history-filter-preprocess-precision">
                                    <option value="all">全部</option>
                                    <option value="bf16">bf16</option>
                                    <option value="fp16">fp16</option>
                                    <option value="fp32">fp32</option>
                                </select>
                            </label>
                            <label>
                                <span>块交换精度</span>
                                <select id="history-filter-block-swap-precision">
                                    <option value="all">全部</option>
                                    <option value="bf16">bf16</option>
                                    <option value="fp8_e4m3">fp8_e4m3</option>
                                </select>
                            </label>
```

缩进与相邻 label 对齐（空格风格跟文件一致）。

- [ ] **Step 4: 改 state**

`history-state.js` 的 `historyManagerFilters`：

```javascript
        historyManagerFilters: {
            search: '',
            kind: 'all',
            state: 'all',
            archived: 'active',
            source: 'all',
            trainingVariant: 'all',
            preprocessPrecision: 'all',
            blockSwapPrecision: 'all',
            sort: 'newest',
        },
```

- [ ] **Step 5: 改 sync / listeners / contract / tooltips**

`task-collections.js` → `syncHistoryFilterControls` 的 `controls` 对象增加三对 id→key。

`event-listeners-setup.js` → `historyFilterMap` 同样增加三对（search 仍用 `input`，这三只走 `change`，因 id !== search）。

`event-listeners-contract.js` 在 `'history-filter-source'` 后插入三个 id 字符串。

`beginner-tooltips.js`：

```javascript
        'history-filter-training-variant': '按历史详情概览中的训练变体（方法族，如 lokr / lora）筛选。',
        'history-filter-preprocess-precision': '按预处理精度偏好（preprocess_precision_preference）筛选。',
        'history-filter-block-swap-precision': '按块交换传输精度（block_swap_transfer_dtype）筛选。',
```

`historyManagerFilterDefault`：三键未单独分支时已 `return 'all'`，**无需改**（确认 `key === 'search'|'archived'|'sort'` 之外默认 `all`）。

- [ ] **Step 6: 跑测通过并 commit**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_history.py::test_history_manager_extra_filter_controls_are_wired -q
```

```bash
git add web/static/index.html \
  web/static/js/features/anima-app/state/history-state.js \
  web/static/js/features/history-list/task-collections.js \
  web/static/js/features/app-shell/event-listeners-setup.js \
  web/static/js/features/app-shell/event-listeners-contract.js \
  web/static/js/features/app-shell/beginner-tooltips.js \
  tests/test_training_frontend_history.py
git commit -m "feat(history-ui): add variant and precision filter controls under global search"
```

---

### Task 4: Filter logic + stat reset/active + search text

**Files:**
- Modify: `web/static/js/features/history-list/collections-workbench.js`
- Test: `tests/test_training_frontend_history.py`

**Interfaces:**
- Consumes: `task.training_variant`, `task.preprocess_precision`, `task.block_swap_precision`；filters 三键
- Produces: `historyManagerBaseFilteredTasks` 尊重三筛选；`applyHistoryStatFilter` 重置三键为 `all`；`historyStatFilterIsActive` 要求三键均为 `all`；`historyTaskSearchText` 含三字段

- [ ] **Step 1: 失败断言**

```python
def test_history_manager_base_filter_includes_config_chip_fields():
    src = _frontend_module_text("js/features/history-list/collections-workbench.js")
    base = _section(src, "export function historyManagerBaseFilteredTasks", "export function historyManagerVisibleTasks")
    assert "trainingVariant" in base
    assert "preprocessPrecision" in base
    assert "blockSwapPrecision" in base
    assert "training_variant" in base
    assert "preprocess_precision" in base
    assert "block_swap_precision" in base

    apply = _section(src, "export function applyHistoryStatFilter", "export function historyStatFilterIsActive")
    assert "trainingVariant: 'all'" in apply
    assert "preprocessPrecision: 'all'" in apply
    assert "blockSwapPrecision: 'all'" in apply

    active = _section(src, "export function historyStatFilterIsActive", "export function historyManagerFilteredTasks")
    assert "trainingVariant" in active
    assert "preprocessPrecision" in active
    assert "blockSwapPrecision" in active

    search = _section(src, "export function historyTaskSearchText", "export function historyTaskMatchesCollectionSearch")
    assert "task.training_variant" in search
    assert "task.preprocess_precision" in search
    assert "task.block_swap_precision" in search
```

（`_section` helper 若签名不同，按文件现有用法调整起止锚点。）

- [ ] **Step 2: 跑测确认失败**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_frontend_history.py::test_history_manager_base_filter_includes_config_chip_fields -q
```

- [ ] **Step 3: 实现过滤**

在 `historyManagerBaseFilteredTasks` 内，`historyTaskMatchesSourceFilter(...)` 之后、`search` 之前插入：

```javascript
        if (!historyTaskMatchesChipFilters(task, historyState.historyManagerFilters)) return false;
```

在同文件 `historyTaskMatchesSourceFilter` 附近新增：

```javascript
export function historyTaskMatchesChipFilters(task, filters = {}) {
    const trainingVariant = filters.trainingVariant || 'all';
    if (trainingVariant !== 'all') {
        const value = String(task?.training_variant || '').trim().toLowerCase();
        if (value !== trainingVariant) return false;
    }
    const preprocessPrecision = filters.preprocessPrecision || 'all';
    if (preprocessPrecision !== 'all') {
        const value = String(task?.preprocess_precision || '').trim().toLowerCase();
        if (value !== preprocessPrecision) return false;
    }
    const blockSwapPrecision = filters.blockSwapPrecision || 'all';
    if (blockSwapPrecision !== 'all') {
        const value = String(task?.block_swap_precision || '').trim().toLowerCase();
        if (value !== blockSwapPrecision) return false;
    }
    return true;
}
```

- [ ] **Step 4: stat filter**

`applyHistoryStatFilter` 的 `next` 对象加入：

```javascript
        trainingVariant: 'all',
        preprocessPrecision: 'all',
        blockSwapPrecision: 'all',
```

`historyStatFilterIsActive` 的 `base` 计算在 `searchEmpty && …` 上追加：

```javascript
        (historyState.historyManagerFilters.trainingVariant || 'all') === 'all' &&
        (historyState.historyManagerFilters.preprocessPrecision || 'all') === 'all' &&
        (historyState.historyManagerFilters.blockSwapPrecision || 'all') === 'all' &&
```

（保持与现有 `archived` 条件同一布尔表达式风格。）

- [ ] **Step 5: search text**

`historyTaskSearchText` 数组在 `task.variant` 附近加入：

```javascript
        task.training_variant,
        task.preprocess_precision,
        task.block_swap_precision,
```

- [ ] **Step 6: 跑测 + 回归 + commit**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_frontend_history.py::test_history_manager_base_filter_includes_config_chip_fields \
  tests/test_training_frontend_history.py::test_history_manager_extra_filter_controls_are_wired \
  tests/test_training_history_list.py::test_history_summary_includes_config_chip_fields_from_snapshot \
  -q
```

```bash
git add web/static/js/features/history-list/collections-workbench.js tests/test_training_frontend_history.py
git commit -m "feat(history-ui): filter history list by training variant and precisions"
```

---

### Task 5: 端到端定向验证与收尾

**Files:** 无新功能文件；只验证与必要时修小问题。

- [ ] **Step 1: 跑后端 history list 全文件**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_history_list.py -q
```

Expected: PASS

- [ ] **Step 2: 跑前端 history 测试**

```bash
timeout 90 .venv/bin/python -m pytest tests/test_training_frontend_history.py -q --tb=line
```

Expected: 本功能相关全部 PASS。若出现与本 diff 无关的预存失败，在最终回复列出，不扩大修复范围。

- [ ] **Step 3: `git diff --check` 与状态**

```bash
git diff --check
git status --short --branch
git log --oneline -6
```

- [ ] **Step 4: 对照 spec 完成标准清单**

- [ ] 工具栏三个下拉，默认全部，文案正确  
- [ ] 列表 API 带三字段；lokr 快照 → `training_variant=lokr`  
- [ ] 过滤键与 stat 重置/active 行为符合 spec  
- [ ] 详情 overview 仍显示四 chip，逻辑在 shared 模块  
- [ ] 未写用户历史目录  

- [ ] **Step 5: 若有未提交修修补补则 commit**

```bash
git add -u
git commit -m "test(history): tighten extra filter wiring assertions"
```

（无变更则跳过。）

---

## Spec coverage checklist（plan self-review）

| Spec 要求 | Task |
|---|---|
| 三下拉 UI 与现有风格一致 | T3 |
| 默认全部、change 重渲染、sync 双向 | T3 |
| 训练变体=方法族非 stem | T1 + T4 |
| 精度键 preprocess / block_swap_transfer | T1 |
| 列表读 snapshot，失败为空 | T1 |
| 变体推断顺序对齐详情 | T1 + T2 |
| 固定选项表、无动态 append | T3 |
| API 字段名 training_variant 等 | T1 |
| 过滤只读 API 字段 | T4 |
| stat 重置三键 + active 要求 all | T4 |
| search text 并入三字段 | T4 |
| shared JS helper | T2 |
| 后端单测 + 前端断言 | T1–T4 |
| 无 meta 写回、无缓存 | T1（不实现写回/缓存） |
| 不筛训练精度 | 全 plan 未做 |

**Placeholder scan:** 无 TBD；实现片段完整。  
**命名一致性:** `trainingVariant` / `training_variant`、`preprocessPrecision` / `preprocess_precision`、`blockSwapPrecision` / `block_swap_precision` 前后端对应关系在 T1/T3/T4 固定。
