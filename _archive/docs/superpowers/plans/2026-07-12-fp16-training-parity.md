# FP16 训练防护链对齐 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 MonadForge 的训练侧 fp16 完整防护链合进 `anima_lora`：pre-Ampere 自动 `bf16→fp16`、VAE 半精度保护、DiT residual 防溢出，并保留现有推理 `runtime_dtype` 与 V100 `lora_fp32_compute`。

**Architecture:** 精度策略抽到可单测纯函数模块 `library/training/precision_policy.py`；启动接线放在 `train_session.py` / `bootstrap.py`；数值护栏放在 `library/anima/models.py` 的 `fp32_residual` 路径。严格 TDD：先红测，再最小实现，再定向回归。

**Tech Stack:** Python 3.13、PyTorch、HuggingFace Accelerate、pytest、现有 `library/training/*` 与 `library/anima/models.py`。

**Spec:** `docs/superpowers/specs/2026-07-12-fp16-training-parity-design.md`

## Global Constraints

- 工作目录：`/home/scv/nvme0n1p1/训练器相关/anima_lora`
- Python：`.venv/bin/python`；后台测试命令加 `timeout 60`
- 所有沟通用简体中文；代码标识保持英文
- 默认配置仍是 `mixed_precision = "bf16"`，只在 `sm < 8` 时自动切 `fp16`
- 不改推理 `runtime_dtype`；不 hardcode 回 bf16
- 不启动真实长训练、不下载大模型、不删除 history/queue/runtime 用户数据
- 热点文件只做小范围接入；策略逻辑进新模块
- 工作区可能已有无关改动：只提交本计划相关文件，不 revert 他人改动
- 每个 Task 结束必须留下：失败测试命令、通过测试命令、变更文件列表

---

## File Map

| 文件 | 职责 |
|---|---|
| `library/training/precision_policy.py` | 新建：`resolve_mixed_precision` / `resolve_vae_dtype` |
| `library/training/train_session.py` | 启动前切精度；用 VAE dtype resolver 替换粗暴 `no_half_vae` 分支 |
| `library/training/cli_args.py` | 增加 `--half_vae`；收紧 `--no_half_vae` 文案 |
| `library/anima/models.py` | `fp32_residual` 护栏、`enable_fp32_residual`、FinalLayer fp32 投影 |
| `library/training/bootstrap.py` | compile 前按 `mixed_precision=="fp16"` 打开 residual |
| `tests/test_mixed_precision_resolver.py` | mixed precision 自动切换契约 |
| `tests/test_vae_dtype_resolver.py` | VAE dtype 优先级契约 |
| `tests/test_fp16_residual_safe.py` | residual / FinalLayer / enable 传播契约 |
| `tests/test_training_bootstrap.py` | 已有 `lora_fp32_compute` 回归 |
| `configs/gui-methods/lora-v100-stable.toml` | 可选注释补充，不强制改行为 |

## 与现有代码的关键关系

- `anima_lora` 的 `Anima.forward` 已有 `use_fp32 = (x.dtype == torch.float16)`，会把 block 输入抬到 fp32，并给 FinalLayer 传 `use_fp32`。
- 这**不能替代** MonadForge 的 `fp32_residual`：
  - residual 的 `gate * branch` 仍可能在 autocast 下先以 fp16 溢出成 `inf`，再和 fp32 residual 相加。
  - FinalLayer 现有 `use_fp32` 只抬了 modulation 输入，**没有** MonadForge 的 `_fp32_project` 最终投影保护。
- 本计划新增 `fp32_residual` 作为训练 fp16 正式护栏；保留现有 `use_fp32` 参数，不在本轮删除。

## Debug Gate（每个 Task 强制）

1. 先写/改失败测试（红）
2. 跑测试确认红因正确
3. 最小实现
4. 跑域测试包（绿）
5. 需要时跑跨域最小回归
6. 只提交本 Task 相关文件

跨域最小回归（实现后至少跑一次）：

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_mixed_precision_resolver.py \
  tests/test_vae_dtype_resolver.py \
  tests/test_fp16_residual_safe.py \
  tests/test_training_bootstrap.py -q
```

---

### Task 1: precision_policy 纯函数 + mixed/VAE 单测

**Files:**
- Create: `library/training/precision_policy.py`
- Create: `tests/test_mixed_precision_resolver.py`
- Create: `tests/test_vae_dtype_resolver.py`

**Interfaces:**
- Consumes: `args.mixed_precision` / `args.no_half_vae` / `args.half_vae`；`torch.cuda.is_available` / `get_device_capability`（可注入）
- Produces:
  - `resolve_mixed_precision(args, *, get_capability=None) -> None`
  - `resolve_vae_dtype(args, weight_dtype, *, get_capability=None) -> torch.dtype`

- [ ] **Step 1: 写 mixed precision 失败测试**

创建 `tests/test_mixed_precision_resolver.py`：

```python
from __future__ import annotations

import types

import pytest

from library.training.precision_policy import resolve_mixed_precision


def _fake_args(mp="bf16"):
    return types.SimpleNamespace(mixed_precision=mp)


def test_no_switch_on_ampere(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 0))
    args = _fake_args()
    resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_switch_on_v100_back_writes_args(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args("bf16")
    with caplog.at_level("WARNING"):
        resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"
    assert any("fp16" in r.getMessage() for r in caplog.records)


def test_switch_on_t4(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))
    args = _fake_args("bf16")
    resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"


def test_explicit_fp16_left_alone(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args("fp16")
    resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"


def test_capability_probe_failure_is_safe(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def _raise(*a, **k):
        raise RuntimeError("cuda init failed")

    monkeypatch.setattr("torch.cuda.get_device_capability", _raise)
    args = _fake_args("bf16")
    with caplog.at_level("WARNING"):
        resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_no_cuda_is_noop(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    args = _fake_args("bf16")
    resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_missing_mixed_precision_attr_is_safe(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = types.SimpleNamespace()
    resolve_mixed_precision(args)
    assert not hasattr(args, "mixed_precision")
```

- [ ] **Step 2: 写 VAE dtype 失败测试**

创建 `tests/test_vae_dtype_resolver.py`：

```python
from __future__ import annotations

import types

import torch

from library.training.precision_policy import resolve_vae_dtype


def _fake_args(mp="fp16", no_half_vae=False, half_vae=False):
    return types.SimpleNamespace(
        mixed_precision=mp,
        no_half_vae=no_half_vae,
        half_vae=half_vae,
    )


def test_no_half_vae_forces_fp32(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 0))
    args = _fake_args(mp="bf16", no_half_vae=True)
    assert resolve_vae_dtype(args, torch.bfloat16) == torch.float32


def test_no_half_vae_beats_half_vae(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="fp16", no_half_vae=True, half_vae=True)
    assert resolve_vae_dtype(args, torch.float16) == torch.float32


def test_half_vae_overrides_auto_fp32_on_v100(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="fp16", half_vae=True)
    assert resolve_vae_dtype(args, torch.float16) == torch.float16


def test_auto_fp32_on_v100_fp16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="fp16")
    assert resolve_vae_dtype(args, torch.float16) == torch.float32


def test_no_force_on_ampere_fp16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 0))
    args = _fake_args(mp="fp16")
    assert resolve_vae_dtype(args, torch.float16) == torch.float16


def test_no_force_on_bf16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="bf16")
    assert resolve_vae_dtype(args, torch.bfloat16) == torch.bfloat16


def test_capability_probe_failure_keeps_weight_dtype(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def _raise(*a, **k):
        raise RuntimeError("cuda init failed")

    monkeypatch.setattr("torch.cuda.get_device_capability", _raise)
    args = _fake_args(mp="fp16")
    with caplog.at_level("WARNING"):
        dtype = resolve_vae_dtype(args, torch.float16)
    assert dtype == torch.float16
```

- [ ] **Step 3: 跑红测**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_mixed_precision_resolver.py \
  tests/test_vae_dtype_resolver.py -q
```

Expected: FAIL，原因是 `library.training.precision_policy` 不存在或函数未定义。

- [ ] **Step 4: 最小实现 `precision_policy.py`**

创建 `library/training/precision_policy.py`：

```python
"""Training mixed-precision / VAE dtype policy helpers."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

CapabilityFn = Callable[[], tuple[int, int]]


def _default_get_capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


def resolve_mixed_precision(
    args,
    *,
    get_capability: Optional[CapabilityFn] = None,
) -> None:
    """Back-write ``args.mixed_precision`` for pre-Ampere GPUs in place.

    Only acts when current value is ``bf16`` and GPU major < 8.
    Capability probe failure is fail-closed: keep bf16 and warn.
    """
    if getattr(args, "mixed_precision", None) != "bf16":
        return
    if not torch.cuda.is_available():
        return

    probe = get_capability or _default_get_capability
    try:
        major, _minor = probe()
    except Exception:
        logger.warning(
            "could not read GPU compute capability; keeping --mixed_precision bf16."
        )
        return

    if major < 8:
        args.mixed_precision = "fp16"
        logger.warning(
            "GPU sm_%d0 has no native bf16 (bf16 autocast runs the slower "
            "fp32 emulation) — auto-switching --mixed_precision from bf16 to "
            "fp16. Pass --mixed_precision bf16 explicitly to keep bf16.",
            major,
        )


def resolve_vae_dtype(
    args,
    weight_dtype: torch.dtype,
    *,
    get_capability: Optional[CapabilityFn] = None,
) -> torch.dtype:
    """Derive VAE dtype, forcing fp32 where fp16 decode is unsafe."""
    if getattr(args, "no_half_vae", False):
        return torch.float32
    if getattr(args, "half_vae", False):
        return weight_dtype
    if getattr(args, "mixed_precision", None) != "fp16":
        return weight_dtype
    if not torch.cuda.is_available():
        return weight_dtype

    probe = get_capability or _default_get_capability
    try:
        major, minor = probe()
    except Exception:
        logger.warning(
            "could not read GPU compute capability; keeping VAE dtype at "
            f"{weight_dtype} (fp16 decode artifacts possible on pre-Ampere)."
        )
        return weight_dtype

    if major < 8:
        logger.info(
            "pre-Ampere GPU (sm_%d%d) under fp16: forcing VAE to fp32 to avoid "
            "decode artifacts (花图/糊图). Pass --half_vae to allow half-precision "
            "VAE (not recommended).",
            major,
            minor,
        )
        return torch.float32
    return weight_dtype
```

- [ ] **Step 5: 跑绿测**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_mixed_precision_resolver.py \
  tests/test_vae_dtype_resolver.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add library/training/precision_policy.py \
  tests/test_mixed_precision_resolver.py \
  tests/test_vae_dtype_resolver.py
git commit -m "$(cat <<'EOF'
feat(training): add pre-Ampere fp16 precision policy helpers

Extract mixed-precision auto-switch and VAE dtype protection into a
testable pure module before wiring it into train startup.
EOF
)"
```

---

### Task 2: 接线 train_session + CLI `--half_vae`

**Files:**
- Modify: `library/training/train_session.py`
- Modify: `library/training/cli_args.py`
- Optional test add: 在 `tests/test_vae_dtype_resolver.py` 或现有 CLI 测试中确认 `--half_vae` 注册

**Interfaces:**
- Consumes: `resolve_mixed_precision` / `resolve_vae_dtype`
- Produces: 启动时 `args.mixed_precision` 已定稿；`vae_dtype` 按策略解析

- [ ] **Step 1: 改 `train_session.py` import 与启动顺序**

在文件顶部 import 区加入：

```python
from library.training.precision_policy import (
    resolve_mixed_precision,
    resolve_vae_dtype,
)
```

把当前：

```python
    # Prepare accelerator
    logger.info("preparing accelerator")
    accelerator = prepare_accelerator(args)
    ...
    # mixed precision dtype
    weight_dtype, save_dtype = prepare_dtype(args)
    vae_dtype = (
        (torch.float32 if args.no_half_vae else weight_dtype)
        if trainer.cast_vae(args)
        else None
    )
```

改成：

```python
    # Resolve mixed precision BEFORE prepare_accelerator: Accelerator() bakes
    # the autocast dtype at construction time.
    resolve_mixed_precision(args)

    # Prepare accelerator
    logger.info("preparing accelerator")
    accelerator = prepare_accelerator(args)
    ...
    # mixed precision dtype
    weight_dtype, save_dtype = prepare_dtype(args)
    vae_dtype = (
        resolve_vae_dtype(args, weight_dtype)
        if trainer.cast_vae(args)
        else None
    )
```

- [ ] **Step 2: 改 CLI 参数**

在 `library/training/cli_args.py` 找到 `--no_half_vae`，替换/扩展为：

```python
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help=(
            "Run the VAE in fp32 (never half). Forces fp32 unconditionally on "
            "every GPU and precision."
        ),
    )
    parser.add_argument(
        "--half_vae",
        action="store_true",
        help=(
            "Explicitly allow the VAE to run in half precision, overriding the "
            "automatic fp32 protection that kicks in on pre-Ampere GPUs (sm<8, "
            "e.g. V100/T4) under fp16 training. NOT recommended there: fp16 VAE "
            "decode can produce artifacts (花图/糊图). No-op on Ampere+ or under "
            "bf16/fp32."
        ),
    )
```

- [ ] **Step 3: 补一个轻量 CLI 存在性断言（可选但推荐）**

若仓库已有解析 CLI 的测试入口，加：

```python
def test_half_vae_cli_flag_exists():
    from library.training.cli_args import setup_parser_common  # 或实际 parser 工厂名
    # 若工厂名不同，用仓库现有测试同款入口
```

如果 CLI 工厂难复用，可跳过，改为手动：

```bash
timeout 60 .venv/bin/python - <<'PY'
from library.training import cli_args
import inspect
src = inspect.getsource(cli_args)
assert "--half_vae" in src
print("ok")
PY
```

- [ ] **Step 4: 跑域测试**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_mixed_precision_resolver.py \
  tests/test_vae_dtype_resolver.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add library/training/train_session.py library/training/cli_args.py
git commit -m "$(cat <<'EOF'
feat(training): wire pre-Ampere precision policy into train startup

Resolve mixed precision before Accelerator construction and protect VAE
dtype on fp16/pre-Ampere paths, with an explicit --half_vae override.
EOF
)"
```

---

### Task 3: DiT residual / FinalLayer fp16 护栏

**Files:**
- Modify: `library/anima/models.py`
- Create: `tests/test_fp16_residual_safe.py`

**Interfaces:**
- Consumes: 现有 `Block` / `FinalLayer` / `Anima`
- Produces:
  - `Block.fp32_residual: bool`
  - `FinalLayer.fp32_residual: bool`
  - `Block._residual_add(a, b) -> Tensor`
  - `Block._gated_residual_add(residual, gate, branch) -> Tensor`
  - `FinalLayer._fp32_project(x_modulated) -> Tensor`
  - `Anima.enable_fp32_residual() -> None`

- [ ] **Step 1: 写 residual 失败测试**

创建 `tests/test_fp16_residual_safe.py`（先覆盖核心契约，不必一次抄完 MonadForge 全部 bench）：

```python
from __future__ import annotations

import torch

from library.anima.models import Anima, Block, FinalLayer

_FP16_MAX = torch.finfo(torch.float16).max


def test_residual_add_unit_overflow_guard():
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    a = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)
    b = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)

    naive = (a + b).to(torch.float16)
    assert torch.isinf(naive).any()

    block.fp32_residual = False
    assert torch.equal(block._residual_add(a, b), naive)

    block.fp32_residual = True
    guarded = block._residual_add(a, b)
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all()


def test_gated_residual_add_overflow_guard():
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    residual = torch.full((2, 4), 1000.0, dtype=torch.float16)
    gate = torch.full((2, 4), 8.0, dtype=torch.float16)
    branch = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)

    block.fp32_residual = False
    naive = residual + gate * branch
    assert torch.isinf(naive).any()

    block.fp32_residual = True
    guarded = block._gated_residual_add(residual, gate, branch)
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all()


def test_enable_fp32_residual_propagates_to_all_modules():
    anima = Anima(
        max_img_h=16,
        max_img_w=16,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=128,
        num_blocks=3,
        num_heads=4,
        crossattn_emb_channels=128,
        pos_emb_learnable=False,
        rope_enable_fps_modulation=False,
        use_llm_adapter=False,
        use_adaln_lora=False,
        attn_mode="torch",
    )
    assert all(not b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is False

    anima.enable_fp32_residual()

    assert all(b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is True
```

如果 `Anima(...)` 构造参数与当前签名不完全一致，以仓库现有测试（如 `tests/test_native_flatten.py`）里的最小构造为准改写，但断言契约不变。

- [ ] **Step 2: 跑红测**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_fp16_residual_safe.py -q
```

Expected: FAIL（缺少 `fp32_residual` / helper / `enable_fp32_residual`）

- [ ] **Step 3: 改 `FinalLayer`**

在 `FinalLayer.__init__` 的 `init_weights()` 前加入：

```python
        # Runtime flag, not weight state. Must not live in init_weights().
        self.fp32_residual = False
```

新增：

```python
    def _fp32_project(self, x_modulated: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x_modulated.device.type, enabled=False):
            return torch.nn.functional.linear(x_modulated, self.linear.weight.float())
```

在 `FinalLayer.forward` 中，保留现有 `use_fp32` 兼容路径，但优先/并行支持 `fp32_residual`：

```python
        shift_B_T_1_1_D = shift_B_T_D[:, :, None, None, :]
        scale_B_T_1_1_D = scale_B_T_D[:, :, None, None, :]

        if self.fp32_residual:
            normed = self.layer_norm(x_B_T_H_W_D)
            x_modulated = (
                normed.float() * (1.0 + scale_B_T_1_1_D.float())
                + shift_B_T_1_1_D.float()
            )
            return self._fp32_project(x_modulated)

        # existing use_fp32 / default path continues below
```

注意：不要删除现有 `use_fp32` 参数，避免误伤当前调用点。

- [ ] **Step 4: 改 `Block`**

在 `Block.__init__` 末尾加入：

```python
        self.fp32_residual = False
```

在 `Block` 中新增：

```python
    def _residual_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual:
            return a + b
        return a.float() + b.float()

    def _gated_residual_add(
        self,
        residual: torch.Tensor,
        gate: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        if not self.fp32_residual:
            return residual + gate * branch
        return residual.float() + gate.float() * branch.float()
```

把 `Block._forward` 三处 residual 更新：

```python
x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result
...
x_B_T_H_W_D = result * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D
...
x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result
```

改成：

```python
x_B_T_H_W_D = self._gated_residual_add(
    x_B_T_H_W_D, gate_self_attn_B_T_1_1_D, result
)
...
x_B_T_H_W_D = self._gated_residual_add(
    x_B_T_H_W_D, gate_cross_attn_B_T_1_1_D, result
)
...
x_B_T_H_W_D = self._gated_residual_add(
    x_B_T_H_W_D, gate_mlp_B_T_1_1_D, result
)
```

- [ ] **Step 5: 改 `Anima`**

在 `Anima` 中靠近 gradient checkpointing helpers 的位置新增：

```python
    def enable_fp32_residual(self) -> None:
        """Promote residual stream / final projection to fp32-safe path for fp16."""
        for block in self.blocks:
            block.fp32_residual = True
        self.final_layer.fp32_residual = True
```

- [ ] **Step 6: 跑绿测**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_fp16_residual_safe.py -q
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add library/anima/models.py tests/test_fp16_residual_safe.py
git commit -m "$(cat <<'EOF'
feat(anima): add fp16-safe residual accumulation guards

Keep DiT residual adds and final projection finite under fp16 autocast
while leaving the default bf16 path inert.
EOF
)"
```

---

### Task 4: bootstrap 在 compile 前启用 residual

**Files:**
- Modify: `library/training/bootstrap.py`
- Modify/Create test: 优先扩 `tests/test_training_bootstrap.py`，或新建轻量测试验证调用顺序

**Interfaces:**
- Consumes: `args.mixed_precision`、`unet.enable_fp32_residual()`
- Produces: fp16 路径在 `compile_blocks_for_training` 前打开 residual

- [ ] **Step 1: 写接线测试（推荐 monkeypatch 风格）**

在 `tests/test_training_bootstrap.py` 增加：

```python
def test_bootstrap_enables_fp32_residual_before_compile(monkeypatch):
    # 构造最小化伪对象：args.mixed_precision="fp16", args.torch_compile=True
    # unet 带 enable_fp32_residual 记录调用顺序
    # monkeypatch compile_blocks_for_training，记录其被调用时 residual 是否已开
    # 断言：enable 在 compile 前发生
    ...
```

如果完整 `prepare_network` 链路太重，允许改成更窄的单元：

```python
def test_fp16_residual_enable_order_contract():
    calls = []

    class DummyUNet:
        def enable_fp32_residual(self):
            calls.append("enable")

    def fake_compile(unet, *a, **k):
        calls.append("compile")

    # 复制 bootstrap 中目标代码片段的顺序约束到 helper，或直接测真实代码路径
```

更稳妥做法：在 `bootstrap.py` 抽一个极小 helper，便于测：

```python
def maybe_enable_fp32_residual(args, unet, *, anima_cls) -> bool:
    if getattr(args, "mixed_precision", None) != "fp16":
        return False
    if not isinstance(unet, anima_cls):
        return False
    unet.enable_fp32_residual()
    return True
```

然后单测这个 helper + 确认 bootstrap 在 `if args.torch_compile:` 前调用它。

- [ ] **Step 2: 跑红测**

```bash
timeout 60 .venv/bin/python -m pytest tests/test_training_bootstrap.py -k "fp32_residual or lora_fp32" -q
```

Expected: 新断言 FAIL

- [ ] **Step 3: 改 `bootstrap.py`**

在 gradient checkpointing 段之后、`if args.torch_compile:` 之前插入：

```python
        # fp16 overflow guard — MUST run before compile_blocks below: dynamo
        # specializes block._forward on the per-module fp32_residual bool.
        if args.mixed_precision == "fp16":
            from library.anima import models as anima_models

            if isinstance(unet, anima_models.Anima):
                unet.enable_fp32_residual()
                logger.info(
                    "fp16 mixed precision: enabled fp32 residual accumulation "
                    "(DiT residual stream exceeds fp16 range; prevents NaN). "
                    "Sublayer matmuls still run fp16 under autocast; bf16/fp32 "
                    "runs are unaffected."
                )
```

若采用 helper，则：

```python
        if maybe_enable_fp32_residual(args, unet, anima_cls=anima_models.Anima):
            logger.info(...)
```

- [ ] **Step 4: 跑绿测 + 回归**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_training_bootstrap.py \
  tests/test_fp16_residual_safe.py \
  tests/test_mixed_precision_resolver.py \
  tests/test_vae_dtype_resolver.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add library/training/bootstrap.py tests/test_training_bootstrap.py
git commit -m "$(cat <<'EOF'
feat(training): enable fp32 residual before compile on fp16 runs

Flip Anima residual guards after adapter apply/load and before
compile_blocks so fp16 training stays finite without recompile storms.
EOF
)"
```

---

### Task 5: 文档注释与最终验收

**Files:**
- Modify: `configs/gui-methods/lora-v100-stable.toml`（注释）
- Optional: `docs/configuration/` 下最短说明；若无现成入口，可只改 V100 预设注释
- Keep: `docs/superpowers/specs/2026-07-12-fp16-training-parity-design.md`

- [ ] **Step 1: 更新 V100 预设注释**

在 `configs/gui-methods/lora-v100-stable.toml` 顶部 Notes 增加：

```toml
#   - Default base.toml still uses mixed_precision=bf16; train startup now
#     auto-switches bf16 -> fp16 on pre-Ampere GPUs (sm<8).
#   - fp16 path auto-enables DiT fp32 residual guards before compile.
#   - pre-Ampere + fp16 forces VAE to fp32 unless --half_vae is set.
```

- [ ] **Step 2: 最终回归**

```bash
timeout 60 .venv/bin/python -m pytest \
  tests/test_mixed_precision_resolver.py \
  tests/test_vae_dtype_resolver.py \
  tests/test_fp16_residual_safe.py \
  tests/test_training_bootstrap.py -q
```

Expected: PASS

- [ ] **Step 3: 手工核对清单**

1. `train_session` 中 `resolve_mixed_precision` 在 `prepare_accelerator` 前
2. `bootstrap` 中 residual enable 在 `compile_blocks_for_training` 前
3. 默认 `fp32_residual=False`
4. 推理 `library/inference/precision.py` 未改
5. 无真实训练/下载被触发

- [ ] **Step 4: Commit**

```bash
git add configs/gui-methods/lora-v100-stable.toml
# 若有 docs 说明一并 add
git commit -m "$(cat <<'EOF'
docs: note pre-Ampere fp16 auto guards for V100 preset

Document the train-startup precision switch, residual guard, and VAE
fp32 protection so operators know the defaults on sm<8 GPUs.
EOF
)"
```

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|---|---|
| 自动 `bf16→fp16`（sm<8） | Task 1 + Task 2 |
| VAE 自动 fp32 + `--half_vae` / `--no_half_vae` | Task 1 + Task 2 |
| DiT residual / FinalLayer 护栏 | Task 3 |
| residual 在 compile 前启用 | Task 4 |
| 保留 `lora_fp32_compute` | Task 4 回归 |
| 不改推理 `runtime_dtype` | 全任务约束 + Task 5 核对 |
| 单测覆盖 | Task 1/3/4/5 |
| 文档最小更新 | Task 5 |

## Placeholder / Consistency Self-Review

- 无 TBD/TODO 占位
- 函数名统一为 `resolve_mixed_precision` / `resolve_vae_dtype` / `enable_fp32_residual`
- residual 开启条件统一为 `mixed_precision == "fp16"` 且 `isinstance(unet, Anima)`
- capability 失败统一 fail-closed
- 与现有 `use_fp32` 的关系已写明：并存，不删除

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-12-fp16-training-parity.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — 每个 Task 派一个新子代理，Task 间人工/父代理复查
2. **Inline Execution** — 本会话按 `executing-plans` 连续执行，设检查点

Which approach?
