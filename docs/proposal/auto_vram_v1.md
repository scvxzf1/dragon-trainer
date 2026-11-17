# AutoVram v1 正式规格书

状态：提案 / 未实现（与已落地 `--block_swap_auto` 独立，见下方边界说明）
适用版本：当前 `dev`（设计冻结用）
入口命令：无（本规格书描述的高向低搜索锁定 + 近 OOM 紧急升档闭环尚未实现；当前已上线的进程内单向水位晋升调优器请使用 `--block_swap_auto` / `--block_swap_auto_stop_ratio`，见 [block_swap_macro_pipeline_plan.md](block_swap_macro_pipeline_plan.md) 与 `library/runtime/block_swap_auto.py`）
相关代码：

- `library/anima/models.py`（`enable_block_swap` / pause-resume）
- `library/runtime/offloading.py`（`ModelOffloader`）
- `library/training/model_loading.py`
- `library/training/loop.py`
- `library/training/memory_probe.py`
- `library/training/compat_matrix.py`
- 旧 catalog 的 `form-layout.js`（Classic/Dragon 静态界面的历史抓手；当前配置工作台入口为 `web/frontend-next/`）

---

## 0. 一句话目标

用户只打开一个**自动显存加速**开关；系统从较高的 `blocks_to_swap` 向下搜索，锁定“刚好不炸、尽量不换”的驻留点；锁定后默认冻结，仅在近 OOM 时允许紧急升档；全过程决策可解释、可复现校验、可跨机审计回放。

---

## 1. 非目标（v1 明确不做）

1. 不做全程自由升降的黑盒控制器。
2. 不做 RL / 学习型策略。
3. 不偷偷改 `selective_checkpoint`、`torch_compile`、`block_swap_transfer_dtype`、sample 频率等其它旋钮。
4. 不承诺“换 GPU 仍得到同一个 `locked_s`”。
5. 不在每个 step 微调 `S`。
6. 不对 Soft Tokens / BYG 等已知互斥路径强行启用。

---

## 2. 术语

| 术语 | 含义 |
|---|---|
| `S` | 有效 `blocks_to_swap` |
| `S_max` | 搜索上限；v1 直接取用户配置的 `blocks_to_swap` |
| `S_legal_max` | 模型允许的最大交换块数，通常为 `num_blocks - 2` |
| `effective_s` | 协议当前真正生效的 S |
| `tier` | 离散候选档位 |
| `margin_gb` | 峰值安全垫目标（GB） |
| `peak_margin_gb` | `cuda_total_gb - cuda_max_allocated_gb`（主安全信号） |
| `phase` | `search` / `locked` / `emergency` |
| `audit` | 跨机可读、可校验的决策日志 |

补充：

- “跨机可审计”= 另一台机器只凭日志即可验证“当时决策是否符合协议”。
- “跨机结果相同”= 非目标；只有硬件 profile + 数据序指纹 + 协议版本等价时，才期望 `locked_s` 一致。

---

## 3. 用户面配置

### 3.1 配置项

| Key | 类型 | 默认 | 必显 | 说明 |
|---|---|---|---|---|
| `auto_vram_accel` | bool | `false` | 是 | 总开关。默认关闭。 |
| `blocks_to_swap` | int | 既有默认 | 是 | 开启自动后作为 `S_max`（上限），不是最终锁定值。 |
| `auto_vram_margin_gb` | float | `1.5` | 否（高级） | 峰值安全垫目标。 |
| `auto_vram_tier_step` | int | `4` | 否（高级） | 档位步长。 |
| `auto_vram_search_steps_per_tier` | int | `3` | 否（高级） | 每档最少观察步数。 |
| `auto_vram_require_high_risk_bucket` | bool | `true` | 否（高级） | 搜索期是否要求覆盖高风险 bucket 后才允许降档接受。 |

### 3.2 WebUI 位置

- 分类：配置 → **优化**
- 分组：`显存与速度优化`
- 建议顺序：放在 `blocks_to_swap` 之后

建议控件：

1. 开关：`自动显存加速`
2. 只读状态条（训练中 / 历史回看）：
   - 阶段：搜索中 / 已锁定 / 紧急升档
   - 有效交换块：`effective_s`
   - 一句话原因
3. 高级折叠：安全垫、档位步长、每档步数

### 3.3 UI 文案（中文）

#### 开关标题
自动显存加速

#### 开关帮助
打开后，系统把当前“CPU/GPU 交换块数”当作上限，先从较高交换档位向下试，找到刚好不炸、尽量少换的档位并锁定。锁定后默认不再下调；只有接近显存爆掉时才允许紧急升高交换块数。默认关闭。

#### 状态文案模板
- 搜索中：`正在从 {s_max} 下探，当前试 {effective_s}`
- 已锁定：`已锁定 {effective_s}（从上限 {s_max} 下探）`
- 紧急升档：`近 OOM，已从 {s_before} 升到 {effective_s} 并重锁`
- 拒绝更低档：`{candidate} 不安全：峰值余量 {peak_margin_gb}GB < 目标 {margin_gb}GB`

#### 跨机说明（历史页脚注）
决策日志可跨机器审计；不同显卡得到不同锁定值是正常的。审计看的是“当时为什么这样选”，不是要求所有机器选同一个数。

### 3.4 标签与默认目录登记（实现时）

- `labels-options.js`：`auto_vram_accel: '自动显存加速'`
- `defaults.js`：`auto_vram_accel: false`
- `form-layout.js`：放入 `显存与速度优化`
- `field-help-training.js`：使用上面帮助文案

---

## 4. 协议总览：`auto_vram_v1`

### 4.1 宪法级规则（不可破）

1. **Downward search**：搜索期只允许尝试更低或相等的 S；接受更低档前必须先证明当前档安全。
2. **Hard lock**：锁定后禁止自动降 S。
3. **Emergency only up**：危险时只允许升 S 或保持，然后重锁；本 run 内不再自动下探。
4. **Single knob**：v1 只调节有效 `blocks_to_swap`。
5. **Audit first**：任何 S 变化必须先/同时落审计事件；无事件的静默改 S 视为协议违规。
6. **Deterministic policy**：同样的输入证据 + 同样协议版本，校验器必须得到同样 accept/reject 结论。

### 4.2 状态机

```text
disabled
  -> (auto_vram_accel=true 且 preflight 通过) policy_start
search
  -> tier_enter(S)
  -> tier_observe * N
  -> tier_accept 后尝试更低档
  -> tier_reject 后回退到 last_feasible，lock
locked
  -> 仅监控
  -> 触发风险：emergency_trigger -> tier_upshift -> relock -> locked
emergency
  -> 只做升档与重锁，不做降档搜索
```

状态字段（运行时内存）：

```text
phase: disabled|search|locked|emergency
s_max: int
effective_s: int
last_feasible_s: int|null
locked_s: int|null
candidates_tried: list[int]
rejected: dict[str, reason]
emergency_count: int
search_complete: bool
```

---

## 5. 候选档位生成

### 5.1 合法上界

```text
S_legal_max = max(0, num_blocks - 2)
S_max_eff = min(S_max, S_legal_max)
```

若用户 `blocks_to_swap <= 0` 且打开自动：

- preflight **警告**并建议给一个较大上限；
- v1 行为：直接 `policy_skip_reason=s_max_non_positive`，保持 `effective_s=0`，不进入搜索。

### 5.2 档位序列

从高到低生成：

```text
tiers = [S_max_eff, S_max_eff - step, ..., 0]  # 去重、夹紧、降序
```

默认 `step=4`。
若 `S_max_eff` 不被 step 整除，仍必须包含 `S_max_eff` 与 `0`。

示例：

- `S_max_eff=20, step=4` → `[20,16,12,8,4,0]`
- `S_max_eff=23, step=4` → `[23,19,15,11,7,3,0]`

### 5.3 搜索起点

始终从最高档 `tiers[0] = S_max_eff` 开始。
这是用户已拍板的“高 S 往下探”。

---

## 6. 观测信号

### 6.1 主信号（安全）

| 信号 | 定义 | 用途 |
|---|---|---|
| `peak_margin_gb` | `cuda_total_gb - cuda_max_allocated_gb` | 主安全垫 |
| `reserved_margin_gb` | `cuda_total_gb - cuda_max_reserved_gb` | 辅助，防 reserved 顶满 |
| `alloc_fail` | 本步是否出现 OOM/分配失败 | 硬危险 |
| `near_oom` | 见 6.3 | 紧急触发 |

### 6.2 辅信号（速度，不单独决定安全）

| 信号 | 定义 | 用途 |
|---|---|---|
| `swap_wait_ms` | 来自 block swap profile 的 wait | 解释“为何不继续降也可能不够快” |
| `step_time_ms` | 步耗时 | 摘要展示 |
| `bucket_key` | 当前分辨率 / token family | 高风险覆盖判断 |

说明：v1 **不因为 wait 高就自动升/降 S**。wait 只进审计与解释，避免把速度噪声写成安全策略。

### 6.3 近 OOM 定义（v1）

满足任一即 `near_oom=true`：

1. `alloc_fail == true`
2. `peak_margin_gb < 0.5 * margin_gb`
3. `reserved_margin_gb < 0.25 * margin_gb`
4. 实现层捕获到 CUDA OOM 并进入恢复路径前

### 6.4 档位安全定义（search 期）

一个档位在观察窗口内判定 `safe`，当且仅当：

1. 无 `alloc_fail`
2. 窗口内最小 `peak_margin_gb >= margin_gb`
3. 若 `auto_vram_require_high_risk_bucket=true`，则窗口内至少见过一次“当前 run 已知高风险 bucket”，或已达该档最大等待步数上限（见 7.3）仍未见时，**不得 accept 更低档**，应保守 lock 当前档

> v1 采取保守语义：高风险形状没见过，就不允许宣称“可以更少换”。

---

## 7. 搜索期算法

### 7.1 伪代码

```text
policy_start(s_max, margin_gb, step, steps_per_tier)
effective_s = s_max_eff
last_feasible_s = null
for S in tiers:  # 降序
    apply_s(S)
    observations = []
    for i in 1..steps_per_tier (可延长，见 7.3):
        obs = observe_step()
        write tier_observe
        if near_oom or alloc_fail:
            write tier_reject(S, reason)
            if last_feasible_s is null:
                emergency/fail path: upshift toward s_max_eff or abort run
            else:
                lock(last_feasible_s)
            return
        observations.append(obs)
    if safe(observations):
        last_feasible_s = S
        write tier_accept(S)
        continue  # 尝试更低档
    else:
        write tier_reject(S, reason)
        break
lock(last_feasible_s or s_max_eff)
```

### 7.2 接受更低档的条件

仅当 `current S` 被 `tier_accept` 后，才进入下一更低档。
若下一更低档 reject，则 lock 上一个 `last_feasible_s`。

### 7.3 高风险 bucket 覆盖

若开启 `auto_vram_require_high_risk_bucket`：

- 每档基础观察 `steps_per_tier`
- 若还未见到高风险 bucket，可延长到 `max(steps_per_tier, 8)` 步
- 仍未见：
  - **不降档**
  - `reason_code = high_risk_bucket_unseen`
  - 以当前可行档 lock（保守）

高风险 bucket 的 v1 定义：

- 当前 epoch 计划中 token 数最大的 bucket family；若未知，则取已采样中 `tokens` / 面积最大者。

### 7.4 搜索期改 S 的工程约束

v1 允许改 S 的时机：

1. 训练正式 loop 早期 step 边界；
2. 必须在 forward 外、完成必要 synchronize 后；
3. 必须走受控 reconfigure API（实现任务），禁止直接瞎改字段。

若某环境无法安全热改 S：

- 允许退化实现：`startup_probe` 模式（正式训练前短探针）
- 但审计 schema 不变，只是 `apply_mode=startup_probe|step_boundary_reconfigure`

---

## 8. 锁定期与紧急升档

### 8.1 锁定期

`phase=locked` 后：

- `effective_s` 固定
- 持续写轻量监控（可降采样，如每 N step 一条 `tier_observe` 或 `monitor`）
- **禁止**自动降 S
- **禁止**重新开启下行搜索

### 8.2 紧急升档触发

锁定期出现 `near_oom=true`：

1. 写 `emergency_trigger`
2. `phase=emergency`
3. 计算目标：

```text
# v1 推荐默认：直接回到 last known safe 或 S_max_eff
S_target = max(effective_s, last_feasible_s or effective_s)
if S_target == effective_s:
    S_target = min(S_max_eff, next_higher_tier(effective_s))
if still risky preference:
    S_target = S_max_eff
```

v1 规范默认策略（写死，减少分歧）：

- **第一次紧急**：升到 `min(S_max_eff, effective_s + tier_step 对齐后的更高档)`；若更高档不存在，则 `S_max_eff`
- 若升档后仍 `near_oom`：**直接拉到 `S_max_eff`**
- 若已在 `S_max_eff` 仍 OOM：写 `emergency_exhausted`，交还训练框架既有 OOM 失败路径

### 8.3 紧急后重锁

升档成功后：

1. 写 `tier_upshift`
2. 写 `relock`
3. `phase=locked`
4. `locked_s = effective_s`
5. `emergency_count += 1`
6. **本 run 永久关闭下行搜索**

这是用户拍板的“锁死但允许近 OOM 紧急升档”。

---

## 9. 兼容性与 preflight

### 9.1 硬拒绝（error）

当 `auto_vram_accel=true` 且存在以下情况，preflight 失败：

| code | 条件 |
|---|---|
| `auto_vram_requires_block_swap_surface` | 方法明确禁止 block swap（如 Soft Tokens、BYG 既有约束） |
| `auto_vram_block_swap_cpu_offload` | `cpu_offload_checkpointing=true` |
| `auto_vram_block_swap_unsloth_offload` | `unsloth_offload_checkpointing=true` |
| `auto_vram_negative_s_max` | `blocks_to_swap < 0` |

> 与现有 `check_training_compat` 对齐：自动模式不能绕过 block swap 既有互斥。

### 9.2 软警告（warning）

| code | 条件 | 建议 |
|---|---|---|
| `auto_vram_s_max_zero` | `blocks_to_swap==0` | 提示设置较大上限 |
| `auto_vram_with_cudagraphs` | compile cudagraph 模式 | 提示可能降级/互斥 |
| `auto_vram_margin_too_small` | `margin_gb < 0.75` | 可能频繁紧急升档 |

### 9.3 与 profile/probe 的关系

推荐（非强制）：

- 开启自动时，若 `memory_probe_jsonl=off`，可自动建议 `auto`
- 若 `block_swap_profile_jsonl=off`，搜索期可临时开启内部计时；不一定改用户配置值

审计日志必须自带所需 evidence，不能要求审计者再去拼其它文件才能做主判定。
其它 jsonl 仅作补充证据。

---

## 10. 审计日志规格（跨机可审计）

### 10.1 产物路径

写到当前任务目录（与 WebUI run 目录边界一致）：

```text
<run_dir>/auto_vram_decision.jsonl
<run_dir>/auto_vram_summary.json
```

若 CLI 无 Web run 目录，则：

```text
<output_dir>/../logs/<output_name>.auto_vram_decision.jsonl
<output_dir>/../logs/<output_name>.auto_vram_summary.json
```

路径解析风格对齐 `MemoryProbe.resolve_path` 的 auto 语义。

### 10.2 通用事件信封

每一行 JSONL 事件必须包含：

```json
{
  "schema": "anima.auto_vram.event.v1",
  "policy_id": "auto_vram_v1",
  "policy_version": "1.0.0",
  "event_id": "evt_000017",
  "seq": 17,
  "ts_unix": 1720700000.123,
  "ts_utc": "2026-07-11T12:34:56.123Z",
  "run_id": "20260711-123456-abcd",
  "event": "tier_accept",
  "phase": "search",
  "step": 12,
  "global_step": 12,
  "s_before": 16,
  "s_after": 16,
  "reason_code": "tier_safe",
  "reason_zh": "该档峰值余量满足安全垫",
  "evidence": {},
  "config_fingerprint": "sha256:...",
  "data_order_fingerprint": "sha256:...",
  "hardware_profile_fingerprint": "sha256:...",
  "prev_event_hash": "sha256:...",
  "event_hash": "sha256:..."
}
```

### 10.3 Hash 规则

1. `hardware_profile_fingerprint`
   对规范化后的 hardware_profile 对象做 canonical JSON SHA256。
2. `config_fingerprint`
   对“影响训练显存形态”的配置子集做 canonical JSON SHA256（至少包含 method、precision、batch/bucket 相关、compile/checkpoint、S_max、margin、tier_step、steps_per_tier）。
3. `data_order_fingerprint`
   对已确定的 sample/bucket 顺序摘要做 SHA256；若尚未确定，事件里写 `null`，summary 在可算时补齐。
4. `event_hash`
   对“除 event_hash 外的事件字段”做 canonical JSON SHA256。
5. `prev_event_hash`
   第一条为 `null` 或 `sha256:genesis`；其后形成链。

Canonical JSON 要求：

- UTF-8
- 对象键排序
- 无 NaN/Infinity
- 浮点用有限小数（建议保留 6 位）
- 不含本机绝对路径作为身份字段（路径只可出现在可选 debug 段）

### 10.4 `hardware_profile` 最小字段

```json
{
  "gpu_name": "NVIDIA GeForce RTX 4060 Ti",
  "gpu_index": 0,
  "vram_total_bytes": 17179869184,
  "cuda_available": true,
  "cuda_version": "12.8",
  "torch_version": "2.7.0+cu128",
  "driver_version": "570.x",
  "platform": "linux"
}
```

### 10.5 事件类型与必填 evidence

| event | 必填 evidence 关键字段 | 说明 |
|---|---|---|
| `policy_start` | `inputs`, `tiers`, `apply_mode` | 协议启动 |
| `policy_skip` | `skip_reason` | 未进入搜索 |
| `tier_enter` | `candidate_s`, `tier_index` | 进入候选档 |
| `tier_observe` | `peak_margin_gb`, `reserved_margin_gb`, `cuda_total_gb`, `cuda_max_allocated_gb`, `bucket_key?`, `step_time_ms?`, `swap_wait_ms?` | 观测 |
| `tier_accept` | `candidate_s`, `min_peak_margin_gb`, `target_margin_gb`, `observed_steps` | 接受 |
| `tier_reject` | `candidate_s`, `target_margin_gb`, `min_peak_margin_gb?`, `reject_metric` | 拒绝 |
| `lock` | `locked_s`, `last_feasible_s`, `candidates_tried` | 锁定 |
| `monitor` | 同 observe 子集 | 锁定期降采样监控 |
| `emergency_trigger` | `near_oom_parts`, `peak_margin_gb`, `reserved_margin_gb` | 触发紧急 |
| `tier_upshift` | `s_target`, `upshift_policy` | 升档 |
| `relock` | `locked_s`, `emergency_count` | 重锁 |
| `emergency_exhausted` | `effective_s`, `peak_margin_gb` | 已到顶仍危险 |
| `policy_end` | `final_s`, `locked_s`, `emergency_count`, `status` | 结束 |

### 10.6 `reason_code` 表

| reason_code | 中文 | 出现场景 |
|---|---|---|
| `policy_enabled` | 已启用自动显存加速 | start |
| `s_max_non_positive` | 上限 S<=0，跳过搜索 | skip |
| `compat_blocked` | 兼容性阻止 | skip/error |
| `tier_safe` | 档位满足安全垫 | accept |
| `peak_margin_below_target` | 峰值余量低于目标 | reject/emergency |
| `reserved_margin_too_low` | reserved 余量过低 | reject/emergency |
| `alloc_fail` | 分配失败/OOM | reject/emergency |
| `high_risk_bucket_unseen` | 未覆盖高风险 bucket，停止下探 | lock conservative |
| `no_lower_tier` | 已到最低可行档 | lock |
| `lower_tier_rejected` | 更低档失败，回退锁定 | lock |
| `near_oom` | 近 OOM | emergency |
| `upshift_next_tier` | 升到下一更高档 | upshift |
| `upshift_to_s_max` | 直接升到上限 | upshift |
| `emergency_exhausted` | 已在上限仍不安全 | exhausted |
| `run_finished` | 正常结束 | end |
| `run_failed` | 训练失败结束 | end |

### 10.7 `summary.json` schema

```json
{
  "schema": "anima.auto_vram.summary.v1",
  "policy_id": "auto_vram_v1",
  "policy_version": "1.0.0",
  "created_at_utc": "2026-07-11T12:34:56Z",
  "run_id": "20260711-123456-abcd",
  "status": "locked|emergency_relocked|skipped|failed",
  "config_fingerprint": "sha256:...",
  "data_order_fingerprint": "sha256:...",
  "hardware_profile": {},
  "hardware_profile_fingerprint": "sha256:...",
  "inputs": {
    "auto_vram_accel": true,
    "s_max": 20,
    "s_max_eff": 20,
    "margin_gb": 1.5,
    "tier_step": 4,
    "search_steps_per_tier": 3,
    "require_high_risk_bucket": true,
    "apply_mode": "step_boundary_reconfigure"
  },
  "tiers": [20, 16, 12, 8, 4, 0],
  "result": {
    "locked_s": 12,
    "final_s": 16,
    "lock_step": 14,
    "emergency_count": 1,
    "candidates_tried": [20, 16, 12, 8],
    "last_feasible_s": 12,
    "rejected": {
      "8": {
        "reason_code": "peak_margin_below_target",
        "min_peak_margin_gb": 0.41,
        "target_margin_gb": 1.5
      }
    }
  },
  "explain_zh": "从 20 下探，12 可安全运行；8 因峰值余量不足被拒绝并锁定 12。第 128 步近 OOM，紧急升到 16 后重锁。",
  "decision_jsonl": "auto_vram_decision.jsonl",
  "decision_events": 42,
  "chain_head_hash": "sha256:...",
  "chain_tail_hash": "sha256:..."
}
```

### 10.8 独立审计校验器（实现必须提供）

纯函数接口（建议）：

```text
validate_auto_vram_audit(summary: dict, events: list[dict]) -> AuditResult
```

校验项：

1. schema / policy_version 支持
2. 事件 seq 严格递增
3. hash chain 连续正确
4. 搜索期不出现“锁定后降 S”
5. 锁定期不出现自动降 S 事件
6. 每次 S 变化都有对应 reason_code 与 evidence
7. `tier_accept` / `tier_reject` 的判定可用 evidence 重算，并与 reason 一致
8. summary.result 与事件终态一致
9. fingerprint 字段存在且格式合法

`AuditResult` 最小字段：

```json
{
  "ok": true,
  "errors": [],
  "warnings": [],
  "recomputed_locked_s": 12,
  "recomputed_final_s": 16
}
```

---

## 11. 可解释性要求

### 11.1 用户层（必须）

一句话，不堆术语。示例：

- `从 20 下探到 12 后锁定；8 会顶太满。`
- `训练中差点爆显存，已从 12 升到 16 并重新锁定。`

### 11.2 工程层（必须可回放）

- 完整 events + evidence
- 可被校验器重算

### 11.3 禁止的解释

- “系统觉得这样更好”（无 evidence）
- 只给最终 S，不给拒绝过的更低档原因

---

## 12. 复现语义

### 12.1 同机复现（目标）

在以下等价时，应得到相同 `locked_s` 与相同决策轨迹（允许时间戳不同）：

- 同一 `policy_version`
- 同一 `config_fingerprint`
- 同一 `hardware_profile_fingerprint`
- 同一 `data_order_fingerprint`
- 同一随机种子与可决定的 step 顺序

### 12.2 跨机审计（目标）

任意机器：

- 读取 summary + jsonl
- 运行校验器
- 能得到 pass/fail，以及决策故事

### 12.3 跨机同 S（非目标）

不同 VRAM / 不同 allocator 状态导致不同锁定值，是预期行为。

---

## 13. 模块边界（落地时）

建议新增（不要塞进热点大文件）：

```text
library/training/auto_vram/
  __init__.py
  policy.py          # 状态机
  tiers.py           # 档位生成
  signals.py         # margin/near_oom
  audit.py           # jsonl/summary/hash
  validate.py        # 独立校验器
  explain.py         # reason_zh / explain_zh
```

与现有系统的窄接口：

| 时机 | 调用 |
|---|---|
| preflight | compat 检查 `auto_vram_accel` |
| train session start | `policy_start` |
| 每 step 后 | `on_step(metrics)` |
| OOM/near-OOM 路径 | `on_danger(...)` |
| 需要改 S | `apply_effective_blocks_to_swap(S)`（单独受控实现） |
| run end | `policy_end` + 写 summary |

热点文件 `train.py` / `loop.py` 只保留编排调用，不塞策略细节。

---

## 14. 测试计划（实现时最低集）

### 14.1 纯单元

1. 档位生成：含 23/20/0 边界
2. safe/reject 判定
3. 锁定后降 S 被拒绝
4. 紧急升档只升不降
5. hash chain 校验
6. summary 与 events 一致性
7. 跨机：刻意改 evidence 后校验器应 fail

### 14.2 集成（可用 stub offloader）

1. 搜索 20→16→12，8 reject，lock 12
2. lock 后注入 near_oom，升到 16 并 relock
3. compat：unsloth + auto 应 error
4. `S_max=0` 跳过

### 14.3 不默认跑

- 真卡长训练
- 大模型下载

---

## 15. 分阶段实现建议

### Phase A：协议与审计（可先做）
- schema、校验器、explain
- 用假 metrics 跑状态机

### Phase B：启动/早期搜索接入
- 接 memory 信号
- 受控应用 S
- WebUI 开关默认 false

### Phase C：紧急升档接入
- loop/OOM 路径挂钩
- 历史页回放摘要

### Phase D：缓存同指纹结果（可选）
- 同 `hardware+config` 指纹可复用上次 `locked_s` 作起点，但仍写完整审计

---

## 16. 验收清单（DoD）

1. 默认关闭，不影响现有用户路径。
2. 开启后从高 S 下探，锁定最小必要可行档。
3. 锁定后无自动降 S。
4. 近 OOM 可升档并重锁。
5. 产出 `decision.jsonl` + `summary.json`。
6. 独立校验器可通过“合法日志”、拒绝“篡改/违规日志”。
7. UI 能显示有效 S + 一句话原因。
8. 与既有 block swap 互斥项在 preflight 被拦住。
9. 文档说明：跨机可审计 ≠ 跨卡同 S。

---

## 17. 开放问题（刻意留到实现前再定）

1. `apply_effective_blocks_to_swap` 最终走 step-boundary 热重配，还是 startup probe？
2. 紧急升档默认“升一档”还是“直接 S_max”在真实 16GB LoKr 上哪个更稳？（规范已给默认，可被实验结果修正为 v1.1）
3. 历史页是内嵌摘要，还是链到独立审计查看器？

---

## 18. 变更控制

- 任何改变 accept/reject 判定、紧急触发、锁后禁止降档语义的改动，必须升 `policy_version`。
- 仅文案/UI 不升版本。
- 日志字段只增不改义；废弃字段标记 `deprecated` 而不是直接改含义。

---

## 19. 结论

`auto_vram_v1` 不是“自动乱调块数”，而是：

> **可审计的下行搜索式块驻留协议：高 S 起探，安全则继续少换，不安全则回退锁定；锁定后只许近 OOM 升档保命。**

这与当前训练器已有的 `blocks_to_swap`、memory probe、block swap profile、compat matrix 和 WebUI 优化分组自然对齐，且把“解释 / 复现校验 / 回放”放进了第一性需求，而不是事后补丁。
