# 后端 Round C 产品决策设计

> 状态：Round C 已实施（worktree）  
> 分支：`feat/backend-config-optimization`  
> 工作区：`.worktrees/backend-config-optimization`  
> 基线：`61ac90b0`（Round B' 后）  
> 关联：`docs/superpowers/specs/2026-07-11-backend-residual-optimization-design.md`  
> 计划：待 `docs/superpowers/plans/2026-07-11-backend-round-c-product-decisions.md`

## 1. 一句话目标

把 residual Round C 里需要产品拍板的三项落成可测行为：**路径不收紧**、**item 级 retry override**、**image_test 可配 save_root（默认兼容）**。

## 2. 已锁定决策

| ID | 决策 | 用户选择 |
|---|---|---|
| S-R3 / S-R4 | 整仓相对路径便利 **不收紧**；只文档 + 回归锁现状 | A |
| T-R2 | 完整 item 级 retry override；优先级 **item > queue 运行态 > policy 默认** | A |
| S-R6 | 新增 `image_test_save_root`；空值回退 `output/tests` | D |

### 明确不做

- 不默认收紧 image_test allowlist 中的整仓 `ROOT`
- 不收紧 preview 项目内相对图路径
- 不把 image_test 默认保存目录改成跟随 `output_root`
- 不做 T-R10（手动 retry 破 attempt 上限）
- 不做 C-R6 list 读接口完全 envelope
- 不做 C-R10 DynamicPath 大收敛

## 3. 现状基线（代码事实）

| 域 | 现状 |
|---|---|
| 队列 retry | `_maybe_auto_retry` 只读 `self._queue_auto_retry/max_attempts/retry_backoff_sec` |
| 入队 item | `enqueue_training` 不写 item 级 retry 三键 |
| clone retry | `_clone_queue_item_for_retry` 会复制 item 上未知键（除时间/id 等剔除集） |
| policy 层级 | T-R1 已锁定：queue 缺键 seed policy；有键不覆盖；`set_queue_settings` 只写运行态 |
| image_test 权重 allowlist | preferred/search + **默认追加 `ROOT`** |
| preview 相对图 | 项目内相对路径可解析到 ROOT 下 |
| image_test 默认保存 | `DEFAULT_INFERENCE_DIR`（`output/tests`）；请求可带 `save_path` |
| settings | 已有 `image_test_allow_home_search`；尚无 `image_test_save_root` |

## 4. 设计

### 4.1 T-R2 item 级 retry override

#### 字段

Queue item 可选字段（均可缺省）：

| 字段 | 类型 | clamp / 归一 |
|---|---|---|
| `auto_retry` | bool | 同 queue：字符串 `1/true/yes/on` → true |
| `max_attempts` | int | 1–10 |
| `retry_backoff_sec` | float | 0–3600 |

缺省语义：字段不存在或为 `null` → 不算 override，回退上层。

#### 优先级

```text
effective(key) =
  item 含该键（且非 null） ? normalize(item[key])
  : queue 运行态 normalize(self._queue_*)
```

queue 运行态本身仍服从 T-R1：

1. 读 `queue.json`
2. 仅缺键时用 `training_policy` seed
3. `set_queue_settings` 只改运行态，不静默回写 policy

**不**引入第四层“隐式同步”。

#### 解析 helper（建议）

新增纯函数（位置优先 `web/services/training/service_state.py` 或 `queue_enqueue.py` 旁）：

```python
def resolve_item_retry_policy(
    item: dict[str, Any] | None,
    *,
    queue_auto_retry: bool,
    queue_max_attempts: int,
    queue_retry_backoff_sec: float,
) -> dict[str, Any]:
    """Return effective {auto_retry, max_attempts, retry_backoff_sec}."""
```

规则：

- 仅当 item 中 **显式出现键** 时覆盖该键（可单键 override，不必三键齐）
- 各键独立覆盖
- 返回值始终经过与 queue 相同的 normalize/clamp

#### 写入路径

| 入口 | 行为 |
|---|---|
| `enqueue_training(...)` | 增加可选 kwargs 或从未来 HTTP body 透传；若传入则写入 item |
| `enqueue_training_batch` | 每条 raw 可带三键，透传到 `enqueue_training` |
| resume / 其他入队 | 无显式字段则不写；行为与现网一致 |
| `_clone_queue_item_for_retry` | 保留 item 级三键（已在 copy-all-except 中自然保留；测试锁住） |
| `_maybe_auto_retry` | 用 `resolve_item_retry_policy(item, queue_*)` 替代直接读 `self._queue_*` |

#### HTTP / snapshot

- `POST` 入队（单条/批量）请求体可选 `auto_retry` / `max_attempts` / `retry_backoff_sec`
- 非法值：400 + `ok:false`（clamp 可在 service 内完成；类型完全非法则 400）
- `get_queue_snapshot` / WS `type=queue`：item 上 **有 override 才带字段**；无则可不带（前端靠缺省回退队列级展示也可）
- **不**要求前端本轮改 UI；后端契约先可测

#### 失败分类

仍走 `classify_training_failure` + `should_auto_retry_failure`；item override 只影响“是否允许重试 / 次数 / 退避”，不改 failure class 语义。

### 4.2 S-R3 / S-R4 路径冻结

#### 行为

**零行为变更。**

#### 文档与测试

- residual / Round C 文档写明：产品选择维持便利
- 回归测试（可放现有或小新文件）断言：
  - image_test allowlist 仍包含 `ROOT`（或等价“项目根在允许目录中”）
  - preview 相对项目内路径仍可 resolve（非 `..`、非越权绝对路径）
- 禁止本轮 PR 删除 `dirs.append(ROOT.resolve())` 或收紧 `_resolve_preview_file` 相对路径逻辑

### 4.3 S-R6 `image_test_save_root`

#### 配置键

| 层 | 键 | 默认 |
|---|---|---|
| global settings / `web-ui-settings.toml` `[global]` | `image_test_save_root` | `""`（空） |

空字符串含义：使用兼容默认 `output/tests`（即现有 `DEFAULT_INFERENCE_DIR`）。

#### 规范化

复用/对齐 output_root 风格：

- 禁止 `..`
- 相对路径：相对 `anima_home()`，规范化为 posix 相对串
- 绝对路径：resolve 后保存/使用
- 空白 → `""`（表示默认回退）

实现位置：

- `settings_service._default_global_settings` 增加键
- `GLOBAL_IMAGE_TEST_KEYS`（或等价列表）纳入 `image_test_save_root`
- `get_global_settings` / `save_global_settings` 读写
- helper：`get_image_test_save_root() -> Path` 或返回 display + resolved

#### 运行时行为

| 场景 | 行为 |
|---|---|
| 请求 **未** 提供 `save_path`（或空） | 默认 `save_path` = effective save_root（空设置 → `output/tests`） |
| 请求 **显式** 提供非空 `save_path` | **请求优先**（保持现网可控） |
| 服务初始 `output_dir` | 可用 effective save_root，避免硬编码漂移 |

路径解析继续走 `_resolve_save_dir`；不在本轮扩大“任意绝对路径写穿”的能力，与现 `_resolve_save_dir` 行为对齐。

#### 与 output_root 关系

- **不**自动跟随 `output_root`
- 用户若想统一目录，可手动把 `image_test_save_root` 设到期望位置

## 5. 文件落点（实施预期）

| 文件 | 改动 |
|---|---|
| `web/services/training/queue_enqueue.py` | resolve effective policy；enqueue 透传；`_maybe_auto_retry` |
| `web/services/training/service_state.py` | 可选：共享 normalize；`resolve_item_retry_policy` |
| `web/routes/training.py` | 入队 body 读可选三键 |
| `web/services/settings_service.py` | `image_test_save_root` |
| `web/services/image_test_service.py` | 默认 save 使用 effective save_root |
| `tests/test_queue_item_retry_override.py` | T-R2 新建 |
| `tests/test_settings_image_test_save_root.py` | S-R6 新建 |
| `tests/test_image_test_service.py` 或小新文件 | S-R3 冻结断言 |
| `tests/test_preview_service.py` 或小新文件 | S-R4 冻结断言 |
| `scripts/tasks/utilities.py` | smoke 纳入新测 |
| `docs/superpowers/plans/2026-07-11-backend-round-c-product-decisions.md` | 实施计划 |
| residual 计划/设计台账 | Round C 进度回写 |

## 6. 严格 debug / 测试流程

每 Task：

```text
红测 → 确认 FAIL → 最小实现 → 绿测 → 域包 → 跨域最小 →（触及门禁则）smoke → commit
```

固定命令：

```bash
WT=/home/scv/nvme0n1p1/训练器相关/anima_lora/.worktrees/backend-config-optimization
PY=/home/scv/nvme0n1p1/训练器相关/anima_lora/.venv/bin/python
cd "$WT"
```

域包建议：

```bash
# T-R2
timeout 90 $PY -m pytest -q tests/test_queue_item_retry_override.py   tests/test_queue_policy_layers.py   tests/test_training_retry_integration.py   tests/test_training_queue_retry_wake.py

# S-R6 + 路径冻结
timeout 90 $PY -m pytest -q tests/test_settings_image_test_save_root.py   tests/test_settings_image_test_flags.py   tests/test_image_test_service.py   tests/test_preview_service.py   tests/test_path_safety.py
```

跨域最小：沿用 residual 计划中的跨域包；收尾：

```bash
timeout 180 $PY tasks.py test-backend-smoke
```

### 失败诊断

| 现象 | 先看 |
|---|---|
| item 设了 auto_retry 仍不重试 | item 键是否写入；failure class 是否允许；attempt 是否已满 |
| item 想回退队列级却仍用旧值 | 是否写入了显式键；应用删键而非写 false/1 |
| save 仍进 output/tests | settings 是否持久化；请求是否显式带了 save_path |
| 路径测试红 | 是否误改 S-R3/S-R4 冻结行为 |

## 7. 风险与回滚

| 风险 | 缓解 | 回滚 |
|---|---|---|
| 入队 API 多字段导致旧客户端异常 | 字段全可选；缺省行为不变 | 忽略 body 三键 |
| item 单键 override 误解为“必须三键” | 文档 + 测：可单键 | — |
| save_root 配错导致写到意外目录 | 禁 `..`；相对路径钉在 anima_home | 清空键回退 output/tests |
| 误收紧路径 | S-R3/S-R4 冻结测 | revert 路径 diff |

## 8. 完成定义（Round C）

- [x] T-R2：effective policy helper + enqueue 透传 + `_maybe_auto_retry` 使用 item 优先
- [x] T-R2 测试：关重试 / 提高 attempts / 缺键回退 / clone 保留
- [x] S-R6：settings 键 + 空回退 + 默认 save 路径
- [x] S-R6 测试：空回退 / 自定义 / 禁 `..` / 显式 save_path 优先
- [x] S-R3/S-R4：冻结测 + 文档声明不收紧
- [x] smoke 绿（含新测）
- [x] residual / Round C 台账更新
- [x] 健康度自评 ≥ 90（A-）目标可陈述

## 9. 健康度预期

| 节点 | 总分 |
|---|---:|
| Round B' @ 61ac90b0 | ≈ 89 (B+) |
| Round C 完成 | ≥ 90 (A-) |

主要加分：D1 调度（item retry）、D2 配置可观测（save_root）、D7 门禁扩包；路径分不因“不收紧”扣到不及格——产品已接受便利。

## 10. 决策记录

| 时间 | 决策 | 来源 |
|---|---|---|
| 2026-07-11 | S-R3/S-R4 = 不收紧 | 用户选 A |
| 2026-07-11 | T-R2 = 完整 item override | 用户选 A |
| 2026-07-11 | S-R6 = image_test_save_root + 空回退 output/tests | 用户选 D |
| 2026-07-11 | 本设计确认 | 用户回复「确认」 |
