# 后端残留债优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `feat/backend-config-optimization` @ `ff257245` 基线上，清掉残留 High/关键 Med 债：methods 文件真相、策略层级与 clamp、home_search 一等设置、preview/analysis 统一 path resolve、retry 集成测、跨域换 root 删除矩阵；并把契约门禁补到可持久回归。

**Architecture:** 不新开第三套配置系统。全局偏好进 `settings_service` / `web-ui-settings.toml`；队列运行态留 `queue.json`；路径统一 `path_safety.resolve_allowed_file`；每个 Task 强制 TDD：红测 → 最小实现 → 域包 → 跨域最小回归 → 提交。

**Tech Stack:** Python 3.13、aiohttp WebUI、pytest、现有 `web/services/*`、`library/env.py`。

**Spec:** `docs/superpowers/specs/2026-07-11-backend-residual-optimization-design.md`

## Global Constraints

- 工作目录固定：`/home/scv/nvme0n1p1/训练器相关/anima_lora/.worktrees/backend-config-optimization`
- Python：`/home/scv/nvme0n1p1/训练器相关/anima_lora/.venv/bin/python`
- 不改主仓 `main` 脏工作区；不 push 除非用户明确要求
- 不启动真实长训练 / 不下载大模型 / 不删用户 history/queue/runtime 数据
- 维持 schema：unknown=warning，invalid choice=error
- 不改 `load_method_preset` 合并顺序
- 不主动大拆 `_legacy` / file_groups / datasets 编辑器
- Round A 不收紧整仓相对路径便利（S-R3/S-R4 需用户另批）
- 新测试禁止继续堆进 2000+ 行大文件；优先新建 `tests/test_*`
- 每 Task 后台测试默认 `timeout 60`（域包/跨域可到 180）

### 固定命令变量

```bash
WT=/home/scv/nvme0n1p1/训练器相关/anima_lora/.worktrees/backend-config-optimization
PY=/home/scv/nvme0n1p1/训练器相关/anima_lora/.venv/bin/python
cd "$WT"
```

### 跨域最小回归（每个 Task 收尾）

```bash
timeout 180 $PY -m pytest -q \
  tests/test_training_queue.py \
  tests/test_training_queue_resume.py \
  tests/test_training_queue_retry_wake.py \
  tests/test_training_retry_classification.py \
  tests/test_training_history_delete.py \
  tests/test_training_runtime_config_core.py \
  tests/test_preview_service.py \
  tests/test_image_test_service.py \
  tests/test_env_config_paths.py \
  tests/test_global_settings_runtime.py \
  tests/test_web_config_preflight.py \
  tests/test_web_config_raw_files.py \
  tests/test_web_http_contracts.py \
  tests/test_stage_schedule.py \
  tests/test_path_safety.py \
  tests/test_cross_domain_delete_boundaries.py
```

### 失败诊断速查

| 现象 | 先看 |
|---|---|
| methods 幽灵项 | `merge.list_methods` 的 known / `or True` |
| policy 改了队列不变 | queue.json 是否已写死 auto_retry 三键 |
| clamp 不一致 | `settings_service._normalize_training_policy` vs `service_state` normalize |
| home_search 假开 | `get_global_settings` 是否透出并保存该键 |
| 路径分叉 | preview/analysis 是否仍手写 resolve |
| 换 root 仍能删旧路径 | cross-domain 删除测是否覆盖 |

---

## File Map（本计划主要落点）

| 文件 | 职责 |
|---|---|
| `web/services/config/merge.py` | list_methods 文件真相 |
| `web/services/config_service.py` | set_configs_root 广播（Round B） |
| `web/services/config/raw_files.py` | save/patch warnings（Round B） |
| `web/services/training/service_state.py` | queue clamp |
| `web/services/training/queue_state.py` / `queue_control.py` | 策略层级 seed/覆盖语义 |
| `web/services/training/anomalies.py` | classify 加固 |
| `web/services/training/queue_enqueue.py` | failure_class / stop_requested 防御 |
| `web/services/settings_service.py` | home_search 一等字段 + policy |
| `web/services/image_test_service.py` | 读真实 settings 开关 |
| `web/services/preview/common.py` | 权重 resolve 统一 |
| `web/services/weight_analysis/paths.py` | 权重 resolve 统一 |
| `web/services/path_safety.py` | 共享 resolve |
| `scripts/tasks/utilities.py` | smoke 扩包（Round B） |
| `tests/test_method_discovery.py` | methods 红绿 |
| `tests/test_queue_policy_layers.py` | 策略层级 |
| `tests/test_settings_image_test_flags.py` | home_search |
| `tests/test_path_resolve_unified.py` | 三入口一致 |
| `tests/test_training_retry_integration.py` | stop/ckpt/pause 集成 |
| `tests/test_cross_domain_delete_boundaries.py` | 换 root 删除矩阵 |
| `tests/test_training_websocket.py` | WS 契约（Round B） |
| `tests/test_web_http_contracts.py` | HTTP 扩（Round B） |

---

# Round A（先做，可独立收口）

### Task 1: list_methods 文件真相（C-R1）

**Files:**
- Modify: `web/services/config/merge.py`
- Create: `tests/test_method_discovery.py`

**Interfaces:**
- Consumes: `CONFIGS_DIR / "methods" / "*.toml"`
- Produces: `list_methods() -> list[str]` 仅包含磁盘存在的方法；known 只决定排序，不决定“是否存在”

- [ ] **Step 1: 写失败测试**

```python
# tests/test_method_discovery.py
from pathlib import Path
import web.services.config.merge as merge


def test_list_methods_omits_missing_known_files(tmp_path, monkeypatch):
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "lora.toml").write_text("[network]\n", encoding="utf-8")
    monkeypatch.setattr(merge, "CONFIGS_DIR", tmp_path)
    names = merge.list_methods()
    assert "lora" in names
    assert "lokr" not in names
    assert "tlora" not in names
    assert "hydralora" not in names


def test_list_methods_includes_disk_extras(tmp_path, monkeypatch):
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "turbo.toml").write_text("[network]\n", encoding="utf-8")
    monkeypatch.setattr(merge, "CONFIGS_DIR", tmp_path)
    names = merge.list_methods()
    assert "turbo" in names
```

- [ ] **Step 2: 跑红测**

```bash
timeout 60 $PY -m pytest tests/test_method_discovery.py -q
```

Expected: FAIL（`lokr` 等仍出现）

- [ ] **Step 3: 最小实现**

把 `merge.list_methods` 改成：

```python
def list_methods() -> list[str]:
    known = [
        "lora", "lokr", "ortholora", "tlora", "hydralora",
        "reft", "chimera", "soft_tokens", "ip_adapter", "easycontrol",
        "spd",
    ]
    found: list[str] = []
    try:
        methods_dir = Path(CONFIGS_DIR) / "methods"
        if methods_dir.is_dir():
            for path in sorted(methods_dir.glob("*.toml")):
                name = path.stem
                if name and name not in found:
                    found.append(name)
    except OSError:
        pass
    ordered = [name for name in known if name in found]
    extras = [name for name in found if name not in known]
    return ordered + extras
```

- [ ] **Step 4: 绿测 + 域包**

```bash
timeout 60 $PY -m pytest tests/test_method_discovery.py tests/test_web_config_preflight.py -q
```

Expected: PASS

- [ ] **Step 5: 跨域最小回归 + Commit**

```bash
# 跑跨域最小回归
git add web/services/config/merge.py tests/test_method_discovery.py
git commit -m "$(cat <<'EOF'
fix: list methods only when config files exist

EOF
)"
```

---

### Task 2: 队列策略层级语义锁定（T-R1）

**Files:**
- Modify: `web/services/training/queue_state.py` 和/或 training 初始化读队列处（以代码实查为准）
- Modify: `web/services/training/queue_control.py`（若 set_queue_settings 需注释/返回层信息）
- Create: `tests/test_queue_policy_layers.py`

**Interfaces:**
- Consumes: `settings_service.get_training_policy()`；`queue.json` 运行态键
- Produces:
  - 缺键时 seed policy
  - 已有键时不因 policy 变更覆盖
  - 测试锁住这两条

- [ ] **Step 1: 写失败测试**

```python
# tests/test_queue_policy_layers.py
from pathlib import Path
import json
import toml


def test_queue_seed_from_policy_only_when_keys_missing(tmp_path, monkeypatch):
    # 构造空 queue.json（无 auto_retry 键）
    # save_training_policy(auto_retry=True, max_attempts=3, retry_backoff_sec=5)
    # 初始化 TrainingService / 读队列
    # assert 运行态被 seed 为 3/5/True
    ...


def test_queue_runtime_keys_not_overwritten_by_policy_change(tmp_path, monkeypatch):
    # queue.json 已有 auto_retry=False,max_attempts=2,backoff=1
    # policy 改为 auto_retry=True,max_attempts=9,backoff=99
    # 重新加载服务
    # assert 仍是 2/1/False
    ...
```

（实现时填完整 fixture：用现有 `tests/test_global_settings_runtime.py` / queue 测试的 monkeypatch 模式，不要新造第三套 root。）

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_queue_policy_layers.py -q
```

Expected: 至少一条 FAIL 或明确暴露当前语义；若已满足则改测试断言对齐并补文档字符串说明锁定语义。

- [ ] **Step 3: 最小实现/文档化锁**
  - 在读队列处保持：仅 missing key seed
  - 在 `set_queue_settings` docstring 写清“只改运行态”
  - 若缺 seed 逻辑则补最小 seed
  - 不要隐式把 policy 变更刷进已有 queue.json

- [ ] **Step 4: 域包**

```bash
timeout 120 $PY -m pytest tests/test_queue_policy_layers.py tests/test_training_queue.py tests/test_global_settings_runtime.py -q
```

- [ ] **Step 5: 跨域 + Commit**

```bash
git add web/services/training tests/test_queue_policy_layers.py
git commit -m "$(cat <<'EOF'
docs+test: lock training policy vs queue runtime layers

EOF
)"
```

---

### Task 3: queue clamp 对齐 policy（T-R3）

**Files:**
- Modify: `web/services/training/service_state.py`（normalize max_attempts / backoff）
- Test: `tests/test_queue_policy_layers.py` 或 `tests/test_training_queue.py` 新测函数

**Interfaces:**
- Produces: `max_attempts` clamp 1..10；`retry_backoff_sec` clamp 0..3600

- [ ] **Step 1: 写失败测试**

```python
def test_queue_settings_clamp_match_policy_bounds(tmp_path, monkeypatch):
    # set_queue_settings(max_attempts=999, retry_backoff_sec=99999)
    # snapshot = get_queue_snapshot()
    # assert snapshot["max_attempts"] == 10
    # assert snapshot["retry_backoff_sec"] == 3600.0
    ...
```

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_queue_policy_layers.py -k clamp -q
```

Expected: FAIL（当前无上界）

- [ ] **Step 3: 最小实现**

在 queue normalize 中：

```python
def _normalize_queue_max_attempts(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(10, n))

def _normalize_queue_retry_backoff(value) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        n = 0.0
    return max(0.0, min(3600.0, n))
```

（函数名以文件现有为准，只改逻辑。）

- [ ] **Step 4: 绿测 + 域包**

```bash
timeout 120 $PY -m pytest tests/test_queue_policy_layers.py tests/test_training_queue.py -q
```

- [ ] **Step 5: Commit**

```bash
git add web/services/training/service_state.py tests/test_queue_policy_layers.py
git commit -m "$(cat <<'EOF'
fix: clamp queue retry settings to policy bounds

EOF
)"
```

---

### Task 4: image_test_allow_home_search 一等设置（S-R2）

**Files:**
- Modify: `web/services/settings_service.py`
- Modify: `web/services/image_test_service.py`（若需改读取路径）
- Create: `tests/test_settings_image_test_flags.py`

**Interfaces:**
- Produces: global settings 读写键 `image_test_allow_home_search: bool`，默认 `False`

- [ ] **Step 1: 写失败测试**

```python
def test_image_test_allow_home_search_roundtrip(tmp_path, monkeypatch):
    # 指向临时 settings 文件
    # save_global_settings({"image_test_allow_home_search": True, ...必要字段})
    # got = get_global_settings()
    # assert got["image_test_allow_home_search"] is True
    # 再读 image_test_service._image_test_allow_home_search() is True
    ...


def test_image_test_allow_home_search_default_false(tmp_path, monkeypatch):
    # 空 settings
    # assert get_global_settings()["image_test_allow_home_search"] is False
    ...
```

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_settings_image_test_flags.py -q
```

Expected: FAIL（键丢失）

- [ ] **Step 3: 最小实现**
  - 在 defaults / load / save 中纳入 `image_test_allow_home_search`
  - bool 归一化：`true/1/yes/on`
  - `image_test_service` 继续从 `get_global_settings()` 读，不再靠 monkeypatch 才能生效

- [ ] **Step 4: 域包**

```bash
timeout 120 $PY -m pytest tests/test_settings_image_test_flags.py tests/test_image_test_service.py tests/test_global_settings_runtime.py -q
```

- [ ] **Step 5: Commit**

```bash
git add web/services/settings_service.py web/services/image_test_service.py tests/test_settings_image_test_flags.py
git commit -m "$(cat <<'EOF'
feat: persist image_test_allow_home_search in global settings

EOF
)"
```

---

### Task 5: preview/analysis 统一 resolve_allowed_file（S-R1）

**Files:**
- Modify: `web/services/preview/common.py`
- Modify: `web/services/weight_analysis/paths.py`
- Create: `tests/test_path_resolve_unified.py`
- Optionally extend: `tests/test_path_safety.py`

**Interfaces:**
- Consumes: `path_safety.resolve_allowed_file` / `allowed_weight_dirs`
- Produces: image_test / continue / preview / analysis 对 outside/`..`/合法相对路径一致

- [ ] **Step 1: 写失败测试**

```python
def test_preview_and_analysis_reject_outside_like_resolve_allowed(tmp_path, monkeypatch):
    # root/output/secret 布局
    # preview._resolve_weight_file(secret) raises or returns None/error 与 path_safety 同语义
    # weight_analysis.resolve_analysis_weight(secret) 同样拒绝
    ...


def test_preview_and_analysis_accept_under_output_root(tmp_path, monkeypatch):
    # output/run/a.safetensors 可解析
    ...
```

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_path_resolve_unified.py -q
```

- [ ] **Step 3: 最小实现**
  - 删除手写 absolute/relative 分叉中与 allowlist 重复的逻辑
  - 调用 `resolve_allowed_file(..., allowed_dirs=allowed_weight_dirs(...))`
  - 保持既有错误文案尽量兼容（可包一层 ValueError 消息）

- [ ] **Step 4: 域包**

```bash
timeout 120 $PY -m pytest tests/test_path_resolve_unified.py tests/test_path_safety.py tests/test_preview_service.py tests/test_image_test_service.py -q
```

- [ ] **Step 5: Commit**

```bash
git add web/services/preview/common.py web/services/weight_analysis/paths.py tests/test_path_resolve_unified.py
git commit -m "$(cat <<'EOF'
refactor: route preview and analysis weights through resolve_allowed_file

EOF
)"
```

---

### Task 6: retry 集成测（T-R4 + T-R6 + T-R7）

**Files:**
- Modify: `web/services/training/anomalies.py`（仅当英文 checkpoint 分类红测失败）
- Create: `tests/test_training_retry_integration.py`

**Interfaces:**
- Produces: stop 不 clone；checkpoint_missing 不 clone；pause 策略下 clone 但 paused 且不启动

- [ ] **Step 1: 写失败测试**

```python
def test_user_stop_does_not_auto_retry_clone(...):
    # 开 auto_retry，跑到 running，请求 stop
    # assert 无 retry_of 新项 / 或 canceled 且 items 不增
    ...


def test_checkpoint_missing_launch_failure_does_not_retry(...):
    # resume 指向缺失 train_state.json
    # assert classify checkpoint_missing 且不 clone
    ...


def test_pause_failure_policy_clones_but_stays_paused(...):
    # failure_policy=pause, auto_retry=True
    # 触发 process/launch 失败
    # assert 有 clone item，paused True，且未 start 下一个
    ...


def test_classify_english_checkpoint_missing():
    kind = classify_training_failure(reason="error", message="checkpoint missing for resume")
    assert kind == "checkpoint_missing"
    assert should_auto_retry_failure(kind) is False
```

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_training_retry_integration.py -q
```

- [ ] **Step 3: 最小修复**
  - 优先只补测试锁住已正确行为
  - 若英文 classify 失败，修 `anomalies.py` 运算符优先级：

```python
if (
    "train_state.json" in text
    or "检查点" in f"{reason}\n{message}"
    or ("checkpoint" in text and "missing" in text)
):
    return "checkpoint_missing"
```

- [ ] **Step 4: 域包**

```bash
timeout 120 $PY -m pytest tests/test_training_retry_integration.py tests/test_training_retry_classification.py tests/test_training_queue_retry_wake.py -q
```

- [ ] **Step 5: Commit**

```bash
git add web/services/training/anomalies.py tests/test_training_retry_integration.py
git commit -m "$(cat <<'EOF'
test: lock auto-retry integration for stop, checkpoint, pause

EOF
)"
```

---

### Task 7: 换 output_root 跨域删除/下载矩阵（S-R5 + Q-R3）

**Files:**
- Modify: `tests/test_cross_domain_delete_boundaries.py`
- 若发现实现 bug，最小修 preview 删除边界（禁止大重构）

**Interfaces:**
- Produces: 新 root 内可操作；旧 root / secret 拒

- [ ] **Step 1: 写失败测试**

```python
def test_switch_output_root_blocks_old_root_weight_and_delete(tmp_path, monkeypatch):
    # old_output/run/a.safetensors 与 new_output/run/b.safetensors
    # save_global_settings output_root=new
    # resolve/download/delete old 路径应失败
    # new 路径应成功（按现有 preview API/service 入口）
    ...
```

- [ ] **Step 2: 红测**

```bash
timeout 60 $PY -m pytest tests/test_cross_domain_delete_boundaries.py -q
```

- [ ] **Step 3: 若失败，最小修实现**（仅当测试证明 bug）

- [ ] **Step 4: 域包 + 跨域**

```bash
timeout 120 $PY -m pytest tests/test_cross_domain_delete_boundaries.py tests/test_path_safety.py tests/test_preview_service.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_cross_domain_delete_boundaries.py web/services/preview
git commit -m "$(cat <<'EOF'
test: lock delete and weight boundaries after output_root switch

EOF
)"
```

---

### Task 8: Round A 收口 smoke

**Files:**
- Modify: `scripts/tasks/utilities.py`（可选：把 Round A 新测加入 smoke）

- [ ] **Step 1: 扩 smoke targets（若未包含）**

```python
targets = [
    "tests/test_web_http_contracts.py",
    "tests/test_training_queue_retry_wake.py",
    "tests/test_training_retry_classification.py",
    "tests/test_training_retry_integration.py",
    "tests/test_path_safety.py",
    "tests/test_path_resolve_unified.py",
    "tests/test_method_discovery.py",
    "tests/test_queue_policy_layers.py",
    "tests/test_settings_image_test_flags.py",
    "tests/test_cross_domain_delete_boundaries.py",
    "tests/test_stage_schedule.py",
    "tests/test_env_config_paths.py",
    "tests/test_global_settings_runtime.py",
    "tests/test_web_config_raw_files.py",
    "tests/test_image_test_service.py",
]
```

- [ ] **Step 2: 跑 smoke**

```bash
timeout 180 $PY tasks.py test-backend-smoke
```

Expected: PASS

- [ ] **Step 3: 全跨域最小回归**

```bash
# 使用本文顶部跨域包
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/tasks/utilities.py
git commit -m "$(cat <<'EOF'
test: expand backend smoke with residual round A gates

EOF
)"
```

---

# Round B（契约硬化，Round A 合并后再做）

### Task 9: save_raw/patch 结构化 warnings（C-R3/C-R9）

**Files:**
- Modify: `web/services/config/raw_files.py`
- Modify: `web/routes/config.py`
- Create: `tests/test_raw_file_warnings_contract.py`

- [ ] 红测：unknown key 写成功且 `warnings` 为 list  
- [ ] 实现：`save_raw_file` / patch 返回 warnings；route JSON 带 `warnings`  
- [ ] 兼容：旧 `message` 字段可保留  
- [ ] 域包 + commit

### Task 10: config 写接口 envelope 统一（C-R6）+ HTTP 写删契约（Q-R1）

**Files:**
- Modify: `web/routes/config.py`（mutation 路径）
- Extend: `tests/test_web_http_contracts.py` 或新建 `tests/test_web_http_write_contracts.py`

- [ ] 红测：raw put/patch/delete 成功 `ok:true`，失败 `ok:false`+`error`  
- [ ] 实现：只改 mutation envelope，list 端点可渐进  
- [ ] commit

### Task 11: 真 WS 契约（Q-R2 + T-R5）

**Files:**
- Create: `tests/test_training_websocket.py`
- 可能小改：`web/routes/training.py` 仅当测试需要可注入 fake service

- [ ] 用 aiohttp test client 连 `/ws/training`  
- [ ] 触发 queue broadcast / status  
- [ ] 断言 `type` 与 snapshot 必填字段  
- [ ] 加入 smoke  
- [ ] commit

### Task 12: configs_root 广播补全 + 矩阵测（C-R2 + C-R11）

**Files:**
- Modify: `web/services/config_service.py`
- Extend: `tests/test_global_settings_runtime.py` 或新建 `tests/test_configs_root_broadcast_matrix.py`

- [ ] 红测：切换后 `file_groups/datasets/output_runs` 的 CONFIGS_DIR 等于新根  
- [ ] 实现：扩大 `_CONFIGS_ROOT_SYNC_MODULES` 或统一 accessor  
- [ ] 域包 + 跨域 + commit

---

# Round C（可选，需产品确认）

### Task 13: item 级 retry override（T-R2）

- item 字段：`auto_retry` / `max_attempts` / `retry_backoff_sec` 可选  
- 解析优先级：item > queue runtime > policy default  
- 完整 TDD + snapshot 契约

### Task 14: 整仓相对路径是否收紧（S-R3/S-R4）

- **先问用户**，不要默认收紧  
- 若收紧：image_test 去掉默认整 ROOT；preview 相对图限 preview dirs

### Task 15: image_test save_root（S-R6）

- settings 字段 `image_test_save_root` 或跟随 `output_root`  
- 默认保持 `output/tests` 兼容，除非用户要求跟随

---

## Self-Review（计划自检）

1. **Spec coverage:** Round A 覆盖 C-R1, T-R1, T-R3, S-R2, S-R1, T-R4, T-R6, T-R7, S-R5, Q-R3；Round B 覆盖 C-R3/C-R9/C-R6/C-R2/C-R11, Q-R1/Q-R2/T-R5；Round C 覆盖 T-R2/S-R3/S-R4/S-R6。  
2. **Placeholder scan:** 测试里个别 fixture 细节用“沿用现有 monkeypatch 模式”标注，实现时必须补全，不得留 TBD 提交。  
3. **Type consistency:** `resolve_allowed_file`、`get_training_policy`、`list_methods` 名称与现网一致。  

## 健康度目标

| 节点 | 目标总分 |
|---|---:|
| 当前 @ ff257245 | 75 (C+) |
| Round A 完成 | ≥ 82 (B) |
| Round B 完成 | ≥ 88 (B+) |
| Round C（含产品决策） | ≥ 90 (A-) 可选 |

## 执行交接

Plan complete and saved to `docs/superpowers/plans/2026-07-11-backend-residual-optimization.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** - 每 Task 新子代理 + 两阶段审查  
2. **Inline Execution** - 本会话按 executing-plans 批量推进并设检查点  

Which approach?


---

## Round A' 执行台账（@ 17d6a52e）

| 项 | 状态 | 说明 |
|---|---|---|
| C-R1 methods | ✅ 先前完成 | 6736e43b |
| T-R1 policy layers | ✅ | load 不再伪造 missing keys；测 lock seed/不覆盖 |
| T-R3 clamp | ✅ | attempts 1–10，backoff 0–3600 |
| S-R2 home_search | ✅ | 全局 settings 一等字段 |
| S-R1 resolve 统一 | ✅ | preview/analysis 走 resolve_allowed_file |
| T-R4/T-R6/T-R7 | ✅ | 集成测 + 英文 checkpoint classify |
| smoke 扩包 | ✅ | 140 passed |


---

## Round B 执行台账（部分）

| 项 | 状态 | 说明 |
|---|---|---|
| C-R3/C-R9 structured warnings | ✅ | save/patch 返回 warnings；HTTP 带 `warnings[]` |
| C-R2/C-R11 configs_root broadcast | ✅ | 广播覆盖 file_groups/datasets/output_runs 等 |
| C-R6 mutation envelope | 部分 | raw mutation 已带 ok/error/warnings；list 仍裸数组（兼容） |
| smoke | ✅ | 144 passed |


---

## Round B' 执行台账（C-R4/C-R5 + T-R5 queue）

| 项 | 状态 | 说明 |
|---|---|---|
| C-R5 schema 加载失败可观测 | ✅ | `get_schema_load_status` + load 失败 warning，禁止静默 no-op |
| C-R4 nested/dict 边界 | ✅ | top-level scalar only；nested table/list 跳过；文档化 + 测 |
| T-R5 WS `type=queue` 契约 | ✅ | `test_ws_training_queue_broadcast_contract` |
| smoke | ✅ | 148 passed（含 `test_schema_gate_observability.py`） |

---

## Round C 执行台账

| 项 | 状态 | 说明 |
|---|---|---|
| S-R3/S-R4 路径 | ✅ 冻结不收紧 | `test_path_allowlist_freeze.py` |
| T-R2 item retry | ✅ | `resolve_item_retry_policy` + enqueue 透传 + `_maybe_auto_retry` |
| S-R6 save_root | ✅ | `image_test_save_root` 空回退 `output/tests` |
| smoke | ✅ | 162 passed |

---

## C-R6 渐进 list envelope（Round C 后补刀）

| 项 | 状态 | 说明 |
|---|---|---|
| methods/variants/presets | ✅ | 成功体 `{ok:true, items:[...]}` |
| 前端双读 | ✅ | `startup.js` / chunk22 兼容裸数组与 items |
| merged 读接口 | 仍裸 dict | 兼容故意保留；失败仍 `ok:false` |
| 契约测 | ✅ | `test_web_http_contracts` methods/variants/presets |

---

## Q-R5 route registry

| 项 | 状态 | 说明 |
|---|---|---|
| REQUIRED_ROUTES 清单 | ✅ | bootstrap/training/config/settings/preview 关键面 |
| `setup_routes` 完整性测 | ✅ | `tests/test_web_route_registry.py` |
| batch-start 双别名 | ✅ | `/batch/start` + `/batch-start` |
| smoke | ✅ | 纳入 `test-backend-smoke` |

---

## 纯后端补刀（C-R2 强制 import / S-R8 / T-R10 light）

| 项 | 状态 | 说明 |
|---|---|---|
| configs_root broadcast 强制 import | ✅ | 未加载模块也会 import 后 setattr |
| broadcast 矩阵测 | ✅ | `test_configs_root_broadcast_matrix.py` |
| environment effective roots | ✅ | platform.effective_roots + web_runtime checks |
| manual retry 标记 | ✅ | `manual_retry`/`retry_source=manual`，attempt 重置为 1 |

