# 后端 Round C 产品决策 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `feat/backend-config-optimization` 上落地 Round C：item 级 retry override（T-R2）、`image_test_save_root`（S-R6）、路径便利冻结测（S-R3/S-R4 不收紧）。

**Architecture:** 不新开配置系统。retry 在 queue item 上可选三键，effective 解析为 item > queue 运行态；save_root 进 global settings，空回退 `output/tests`；路径 allow 行为零改只加回归锁。

**Tech Stack:** Python 3.13、aiohttp WebUI、pytest、现有 `web/services/training/*`、`settings_service`、`image_test_service`。

**Spec:** `docs/superpowers/specs/2026-07-11-backend-round-c-product-decisions-design.md`

## Global Constraints

- 工作目录：`/home/scv/nvme0n1p1/训练器相关/anima_lora/.worktrees/backend-config-optimization`
- Python：`/home/scv/nvme0n1p1/训练器相关/anima_lora/.venv/bin/python`
- 不改主仓 main 脏区；不 push 除非用户要求
- 不真训 / 不下载大模型 / 不删用户 history/queue/runtime
- T-R1 策略层级不变：queue 缺键 seed policy；有键不覆盖
- S-R3/S-R4 **禁止**收紧整仓相对路径
- 新测独立文件，禁止堆 2000+ 行大文件
- 每 Task：`timeout 60` 定向；域包/跨域/smoke 可到 180

### 固定命令

```bash
WT=/home/scv/nvme0n1p1/训练器相关/anima_lora/.worktrees/backend-config-optimization
PY=/home/scv/nvme0n1p1/训练器相关/anima_lora/.venv/bin/python
cd "$WT"
```

---

## File Map

| 文件 | 职责 |
|---|---|
| `web/services/training/service_state.py` | `resolve_item_retry_policy` + 复用 normalize |
| `web/services/training/queue_enqueue.py` | enqueue 透传；`_maybe_auto_retry` 用 effective |
| `web/routes/training.py` | 单条/批量入队 body 可选三键 |
| `web/services/settings_service.py` | `image_test_save_root` |
| `web/services/image_test_service.py` | 默认 save 用 effective save_root |
| `tests/test_queue_item_retry_override.py` | T-R2 |
| `tests/test_settings_image_test_save_root.py` | S-R6 |
| `tests/test_path_allowlist_freeze.py` | S-R3/S-R4 冻结 |
| `scripts/tasks/utilities.py` | smoke 扩包 |
| residual / Round C docs | 台账 |

---

### Task 1: resolve_item_retry_policy + `_maybe_auto_retry`（T-R2 核心）

**Files:**
- Modify: `web/services/training/service_state.py`
- Modify: `web/services/training/queue_enqueue.py`
- Create: `tests/test_queue_item_retry_override.py`

**Interfaces:**
- Produces:

```python
def resolve_item_retry_policy(
    item: dict[str, Any] | None,
    *,
    queue_auto_retry: bool,
    queue_max_attempts: int,
    queue_retry_backoff_sec: float,
) -> dict[str, Any]:
    # {"auto_retry": bool, "max_attempts": int, "retry_backoff_sec": float}
```

- `_maybe_auto_retry` 必须用该 helper，不再只读 `self._queue_*`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_queue_item_retry_override.py
from web.services.training.service_state import resolve_item_retry_policy


def test_resolve_item_retry_policy_prefers_item_keys():
    out = resolve_item_retry_policy(
        {"auto_retry": False, "max_attempts": 5, "retry_backoff_sec": 12},
        queue_auto_retry=True,
        queue_max_attempts=2,
        queue_retry_backoff_sec=1.0,
    )
    assert out["auto_retry"] is False
    assert out["max_attempts"] == 5
    assert out["retry_backoff_sec"] == 12.0


def test_resolve_item_retry_policy_partial_override():
    out = resolve_item_retry_policy(
        {"max_attempts": 4},
        queue_auto_retry=True,
        queue_max_attempts=2,
        queue_retry_backoff_sec=3.0,
    )
    assert out["auto_retry"] is True
    assert out["max_attempts"] == 4
    assert out["retry_backoff_sec"] == 3.0


def test_resolve_item_retry_policy_missing_item_uses_queue():
    out = resolve_item_retry_policy(
        {},
        queue_auto_retry=True,
        queue_max_attempts=3,
        queue_retry_backoff_sec=9.0,
    )
    assert out == {
        "auto_retry": True,
        "max_attempts": 3,
        "retry_backoff_sec": 9.0,
    }
```

再补集成向：构造 fake service 或直接调 `_maybe_auto_retry`（沿用 `tests/test_training_retry_integration.py` monkeypatch 模式）：

- item `auto_retry=False` + queue `auto_retry=True` → 不 clone
- item `max_attempts=3` + attempt=2 + queue max=1 → 仍 clone
- clone 后新 item 仍带原 override 键

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_queue_item_retry_override.py -q
```

Expected: FAIL（`resolve_item_retry_policy` 不存在或 `_maybe_auto_retry` 忽略 item）

- [ ] **Step 3: 最小实现**

在 `service_state.py` 增加：

```python
def resolve_item_retry_policy(
    item: dict[str, Any] | None,
    *,
    queue_auto_retry: bool,
    queue_max_attempts: int,
    queue_retry_backoff_sec: float,
) -> dict[str, Any]:
    src = item if isinstance(item, dict) else {}
    auto_retry = (
        _normalize_queue_auto_retry(src["auto_retry"])
        if "auto_retry" in src and src.get("auto_retry") is not None
        else _normalize_queue_auto_retry(queue_auto_retry)
    )
    max_attempts = (
        _normalize_queue_max_attempts(src["max_attempts"])
        if "max_attempts" in src and src.get("max_attempts") is not None
        else _normalize_queue_max_attempts(queue_max_attempts)
    )
    retry_backoff_sec = (
        _normalize_queue_retry_backoff(src["retry_backoff_sec"])
        if "retry_backoff_sec" in src and src.get("retry_backoff_sec") is not None
        else _normalize_queue_retry_backoff(queue_retry_backoff_sec)
    )
    return {
        "auto_retry": auto_retry,
        "max_attempts": max_attempts,
        "retry_backoff_sec": float(retry_backoff_sec),
    }
```

在 `_maybe_auto_retry` 开头：

```python
policy = resolve_item_retry_policy(
    item,
    queue_auto_retry=bool(getattr(self, "_queue_auto_retry", False)),
    queue_max_attempts=int(getattr(self, "_queue_max_attempts", 1) or 1),
    queue_retry_backoff_sec=float(getattr(self, "_queue_retry_backoff_sec", 0.0) or 0.0),
)
if not policy["auto_retry"]:
    return None
attempt = int(item.get("attempt") or 1)
if attempt >= int(policy["max_attempts"]):
    return None
# ... classify ...
backoff = float(policy["retry_backoff_sec"])
```

- [ ] **Step 4: 绿测 + 域包**

```bash
timeout 90 $PY -m pytest -q \
  tests/test_queue_item_retry_override.py \
  tests/test_queue_policy_layers.py \
  tests/test_training_retry_integration.py \
  tests/test_training_queue_retry_wake.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add web/services/training/service_state.py web/services/training/queue_enqueue.py tests/test_queue_item_retry_override.py
git commit -m "$(cat <<'EOF'
feat: resolve queue retry policy with per-item overrides

EOF
)"
```

---

### Task 2: enqueue / HTTP 透传 item 三键（T-R2 入口）

**Files:**
- Modify: `web/services/training/queue_enqueue.py`（`enqueue_training`、`enqueue_training_batch`）
- Modify: `web/routes/training.py`（单条 start + batch）
- Extend: `tests/test_queue_item_retry_override.py`

**Interfaces:**
- `enqueue_training(..., auto_retry=None, max_attempts=None, retry_backoff_sec=None)`
- 仅非 None 时写入 item（normalize 后）
- batch raw 同名字段透传

- [ ] **Step 1: 失败测试**

```python
def test_enqueue_training_persists_item_retry_fields(monkeypatch, tmp_path):
    # 用现有 TrainingService 测试夹具模式：
    # enqueue 后 queue item 含 auto_retry/max_attempts/retry_backoff_sec
    ...
```

HTTP 可选：若已有 aiohttp contract 夹具，断言 body 三键进入 snapshot item；否则 service 级即可。

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_queue_item_retry_override.py -q -k enqueue
```

- [ ] **Step 3: 最小实现**

`enqueue_training` 增加可选参数；构建 item 后：

```python
if auto_retry is not None:
    item["auto_retry"] = _normalize_queue_auto_retry(auto_retry)
if max_attempts is not None:
    item["max_attempts"] = _normalize_queue_max_attempts(max_attempts)
if retry_backoff_sec is not None:
    item["retry_backoff_sec"] = _normalize_queue_retry_backoff(retry_backoff_sec)
```

route 单条：

```python
auto_retry = data.get("auto_retry") if "auto_retry" in data else None
max_attempts = data.get("max_attempts") if "max_attempts" in data else None
retry_backoff_sec = data.get("retry_backoff_sec") if "retry_backoff_sec" in data else None
# pass into enqueue_training
```

batch：从 `raw` 读同名键传入。

- [ ] **Step 4: 绿测**

```bash
timeout 90 $PY -m pytest -q tests/test_queue_item_retry_override.py tests/test_training_queue.py
```

- [ ] **Step 5: Commit**

```bash
git add web/services/training/queue_enqueue.py web/routes/training.py tests/test_queue_item_retry_override.py
git commit -m "$(cat <<'EOF'
feat: accept per-item retry fields on queue enqueue

EOF
)"
```

---

### Task 3: `image_test_save_root`（S-R6）

**Files:**
- Modify: `web/services/settings_service.py`
- Modify: `web/services/image_test_service.py`
- Create: `tests/test_settings_image_test_save_root.py`

**Interfaces:**
- settings 键 `image_test_save_root: str`，默认 `""`
- `get_image_test_save_root() -> str` 返回 display 相对/绝对串；空 → `output/tests`（或 `DEFAULT_INFERENCE_DIR` 常量）
- 请求空 `save_path` 时用该默认；显式非空优先

- [ ] **Step 1: 失败测试**

```python
def test_empty_save_root_falls_back_to_output_tests(tmp_path, monkeypatch):
    # save settings image_test_save_root=""
    # effective default path ends with output/tests


def test_custom_relative_save_root_persists(tmp_path, monkeypatch):
    # save "output/my-tests" → get_global_settings 读回
    # image_test 默认 save_path 使用它


def test_save_root_rejects_dotdot():
    # save_global_settings({"image_test_save_root": "foo/../bar"}) → ValueError/ok false


def test_explicit_request_save_path_wins(monkeypatch):
    # settings root = output/my-tests
    # request save_path = output/other → 用 other
```

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_settings_image_test_save_root.py -q
```

- [ ] **Step 3: 最小实现**

`settings_service`:

- `_default_global_settings` 加 `"image_test_save_root": ""`
- `GLOBAL_IMAGE_TEST_KEYS` 纳入该键
- normalize：类似 output_root 但允许 empty；禁 `..`
- helper `resolve_image_test_save_root() -> str` 返回非空路径串

`image_test_service` 默认：

```python
default_save = settings_service.resolve_image_test_save_root()
# normalize request: if not save_path: save_path = default_save
```

`_resolve_save_dir` 保持；注意绝对路径策略与现网一致（当前实现把路径钉在 ROOT 下——不要偷偷改成任意绝对写穿，除非 settings normalize 已允许绝对且 `_resolve_save_dir` 需小幅对齐；**若绝对 save_root 与现 `_resolve_save_dir` 冲突，优先：相对路径完整支持；绝对路径若现函数不支持，测试只锁相对路径 + 文档说明**）。

- [ ] **Step 4: 绿测 + 域包**

```bash
timeout 90 $PY -m pytest -q \
  tests/test_settings_image_test_save_root.py \
  tests/test_settings_image_test_flags.py \
  tests/test_image_test_service.py
```

- [ ] **Step 5: Commit**

```bash
git add web/services/settings_service.py web/services/image_test_service.py tests/test_settings_image_test_save_root.py
git commit -m "$(cat <<'EOF'
feat: add configurable image_test_save_root with output/tests fallback

EOF
)"
```

---

### Task 4: S-R3/S-R4 路径冻结测 + smoke + 台账

**Files:**
- Create: `tests/test_path_allowlist_freeze.py`
- Modify: `scripts/tasks/utilities.py`
- Modify: residual plan/spec 台账 + Round C plan checkboxes 回写

- [ ] **Step 1: 冻结测试**

```python
def test_image_test_allowlist_still_includes_project_root(monkeypatch):
    # call _image_test_weight_allowlist → ROOT in list


def test_preview_relative_project_path_still_resolves(tmp_path, monkeypatch):
    # 相对路径在项目内可 resolve（沿用 preview common 测法）
```

- [ ] **Step 2: 红/绿**（若实现未改路径，应直接绿；若红说明误改）

```bash
timeout 60 $PY -m pytest tests/test_path_allowlist_freeze.py -q
```

- [ ] **Step 3: smoke 扩包**

`cmd_test_backend_smoke` targets 增加：

- `tests/test_queue_item_retry_override.py`
- `tests/test_settings_image_test_save_root.py`
- `tests/test_path_allowlist_freeze.py`

```bash
timeout 180 $PY tasks.py test-backend-smoke
```

Expected: 全部 passed（约 148+）

- [ ] **Step 4: 文档台账**

- Round C plan 勾选完成项
- residual 计划增加 Round C 台账
- spec DoD 勾选

- [ ] **Step 5: Commit**

```bash
git add tests/test_path_allowlist_freeze.py scripts/tasks/utilities.py docs/superpowers/
git commit -m "$(cat <<'EOF'
test: freeze path allow convenience and close round C ledger

EOF
)"
```

---

## Self-Review

1. **Spec coverage:** T-R2 helper/enqueue/HTTP/`_maybe_auto_retry` → Task 1–2；S-R6 → Task 3；S-R3/S-R4 冻结 → Task 4；smoke/台账 → Task 4。
2. **Placeholder scan:** 集成夹具写“沿用现有 retry integration monkeypatch”，实施时必须抄真实夹具，不得留 `...` 提交。
3. **Type consistency:** 三键名全局统一 `auto_retry` / `max_attempts` / `retry_backoff_sec`；settings 键 `image_test_save_root`。

## 健康度目标

| 节点 | 总分 |
|---|---:|
| 实施前 @ 61ac90b0 | ≈ 89 |
| Round C 后 | ≥ 90 (A-) |

## 执行

本会话默认 **Inline Execution**（用户已 spec 确认并要求推进）。

---

## 执行台账

| Task | 状态 | 提交 |
|---|---|---|
| T-R2 helper + maybe_auto_retry + enqueue | ✅ | 08fc1a07 及后续 |
| S-R6 image_test_save_root | ✅ | 本轮 |
| S-R3/S-R4 冻结 + smoke | ✅ | 162 passed |

