"""fp16 保护与 NaN 后处理的静态等效判定（CPU-only，真 `torch.amp.GradScaler("cpu")`）。

本模块是 A 项目（dragon-trainer，分支 `codex/fp16-nan-guards`）fp16 保护的**静态**判定套件：
不加载模型、不碰 GPU、不改生产代码。它把「等效」拆成五条可判定的不变量：

* **E1 位级** —— 无 scaler 时接缝对 loss / 梯度一字节不动（`TestE1BitwiseInvariance`）。
* **E2 位级 + 语义** —— 溢出步 step() 被调用但权重一字节不动（executed ≠ applied，
  `TestE2SkipSemanticSplit`）。
* **E3 计数级** —— 非有限 loss 丢弃整个累积窗口、计数与记账不推进；A 侧未实现，
  只有 fail-fast，记在 `TestE3WindowAccountingGap`（xfail 规格 + 缺口钉子）。
* **E4 位级** —— 真 GradScaler 的 growth / backoff / state_dict 状态机
  （`TestE4RealGradScalerStateMachine`）。
* **E5 语义** —— 有活 scaler 时「梯度非有限」归 scaler 所有、保护层放弃该读数。A 侧未实现，
  且**当前实现违反 E5（缺陷 D1）**：debug 梯度检查（loop.py:861）早于 unscale / clip（:898），
  会把 fp16 例行上溢判成训练失败。证据在 `TestE5ScalerOwnershipGap`（缺口钉子 + D1 用例 + xfail 规格）。

本文件的实测数字与 t9（math-verifier）的 CPU 探针一致：P1→E4（growth_interval=2 时
8→16→16→32、溢出 32→16）、P2→E2（溢出步内层 optimizer.step 未被调用、权重逐位不变、
scale 1024→512）、P3→E5/D1（有限 loss + scale=65536 下梯度上溢被判失败）、P5→E1（无 scaler 路径
loss 对象 id 不变、梯度与参数逐位相等）。P4 那个「clip 先于 unscale 也能过 norm<=max_norm」的假绿
反例在本文件**没有**对应断言顺序：E1 全部用逐位比对，不用阈值断言。

反「假绿」纪律（本文件的硬约束）：

1. scaler 必须是**真的** `torch.amp.GradScaler("cpu")`；本文件不定义任何 scaler 替身，
   并在 `test_scaler_under_test_is_real_torch_object` 里显式断言实例类型与真实默认值
   （init_scale=65536、state_dict 五键），替身过不了这两条。
2. 每条位级断言都配**负控制**：让被测机制缺席（不挂 scaler / native_amp=False / 无溢出）
   后同一条断言必须不成立，用来证明它不是恒真。
3. 未实现的能力一律写成 xfail 规格（strict=True），绝不写成「已支持」。

生产替代点（诚实声明，不是造假）：

* 生产上 native_amp=True 由 accelerate 在 mixed_precision="fp16" + CUDA 时置位，而
  Accelerator.unscale_gradients 的门就是它（accelerate 1.13.0 accelerator.py:2921）；生产调用点见
  library/training/loop.py:825（accelerator.backward）、:898（accelerator.clip_grad_norm_）、
  :906（state.optimizer.step()）。CPU 上 accelerate 不会置位，因此夹具在**真实 Accelerator 实例**
  上显式置 native_amp=True 并挂**真实** GradScaler("cpu")；被测的 backward / clip_grad_norm_ /
  AcceleratedOptimizer.step 全部是 accelerate 原码。与生产的差异只有「设备是 CPU」与「这条标志位
  需要手工置位」。
* 模型**不**进 Accelerator.prepare（只 prepare optimizer）：CPU 上 fp16 会让 accelerate 给 forward
  套 fp16 autocast，那会引入与 scaler 无关的舍入，把接缝算术掩盖掉。
* 进程级 AcceleratorState / PartialState / GradientState 单例在夹具里做了保存-恢复，避免把 fp16 档
  泄漏给同进程其它测试模块（accelerate 拒绝同进程出现两种 mixed_precision）。
* **已验证 accelerate 1.13.0**：本文件依赖三个内部面（Accelerator.native_amp、
  AcceleratorState._mixed_precision、AcceleratedOptimizer.step_was_skipped）。它们不是公开 API，
  升级 accelerate 时可能静默削弱覆盖，因此 TestAccelerateInternalContract 用 canary 显式失败
  （版本固定 + 属性存在 + unscale 门双向语义），不允许退化成跳过。

本文件**不**覆盖：真实 CUDA 上的 fp16 autocast、unscale_ 的 kernel 级数值、累积窗口的真实循环
记账、process_batch_inner 的 latent nan_to_num（需要完整 trainer 上下文）。这些都进「待热测 /
待实现」清单，不在静态判定的证据范围内。
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.state import AcceleratorState, GradientState, PartialState

from library.runtime.accelerator import patch_accelerator_for_fp16_training
from library.training.finite_checks import (
    check_loss_finite,
    check_trainable_grads_finite,
    debug_finite_enabled,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_ROOT = REPO_ROOT / "library"
LOOP_PATH = LIBRARY_ROOT / "training" / "loop.py"

# 2^3：2 的幂，fp32 下「乘 scale / 乘 1/scale」可逆，才能把 E1 断言到逐位相等。
INIT_SCALE = 8.0

# E3 契约：非有限窗口 → 重放 1 次 → 仍失败则整窗丢弃。每行 = (观测点, 语义, 今日实现位置)。
# 2026-11-06（E3 用户裁定后）：8 个观测点全部已实现，第三列标出真实实现位置。
_WINDOW_ACCOUNTING_POINTS: tuple[tuple[str, str, str], ...] = (
    ("optimizer", "optimizer.step 调用计数", "loop.py::_run_microbatch + _commit_step_boundary（只在提交窗口调用）"),
    ("scheduler", "scheduler.step 调用计数", "loop.py::_commit_step_boundary → _step_lr_scheduler（只在提交窗口调用）"),
    ("global_step", "state.global_step 自增次数", "loop.py::_run_epoch_steps 的提交分支（sync_gradients 门控内）"),
    ("ema", "模型 EMA 更新计数（本仓无 EMA：显式占位槽）", "state.ema_observation 由 window_accounting.commit_window 写"),
    ("ckpt", "checkpoint 保存计数", "loop.py::_run_epoch_steps 提交分支的 saver.maybe_save_step"),
    ("total_loss", "loss 账本 total_loss（窗口级）", "state.window_total_loss，由 window_accounting.commit_window 写"),
    ("num_steps", "loss 账本 num_steps（窗口级微批数）", "state.window_num_steps，由 window_accounting.commit_window 写"),
    ("last_loss", "loss 账本 last_loss（窗口级）", "state.window_last_loss，由 window_accounting.commit_window 写"),
)

# E3 实现清单（2026-11-06 已落地，保留作为接缝清单）：每一条对应上面的观测点。
_WINDOW_SEAM_CHECKLIST: tuple[str, ...] = (
    "library/training/window_accounting.py：窗口账本（WindowBuffer）+ 重放策略（WINDOW_MAX_REPLAYS=1）",
    "loop.py::_run_epoch_steps：窗口边界三岔（commit / replay / discard），失败尝试零副作用",
    "loop.py::_run_microbatch：无条件检查微批 loss 有限性；窗口已坏时 force_skip 阻止该次更新",
    "loop.py::_commit_step_boundary：LR scheduler / adaptive / precision / block-swap 只在提交路径执行",
    "drop 路径：window_discards += 1、清梯度、不推进任何计数或账本，记 warning",
    "已知限制：本仓无 EMA（ema_observation 是占位槽）；窗口丢弃目前只在 LoopState + warning 里可见，"
    "尚未写进 precision_report.json",
)


def _seeded_model() -> torch.nn.Linear:
    torch.manual_seed(7)
    return torch.nn.Linear(4, 3)


def _seeded_batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    return torch.randn(5, 4), torch.randn(5, 3)


def _bits(tensor: torch.Tensor) -> bytes:
    """返回张量的逐字节视图：-0.0 / +0.0 与 NaN payload 都能区分（t2 比较器前提）。"""
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return _bits(actual) == _bits(expected)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, *, label: str) -> None:
    assert _bitwise_equal(actual, expected), f"{label} 逐位不等（uint8 视图对比）"


def _nan_with_payload(payload: int) -> torch.Tensor:
    """构造指定 payload 的 float32 NaN，用于比较器元测试。"""
    return torch.tensor([payload], dtype=torch.int32).view(torch.float32)


@pytest.fixture
def make_accelerator():
    """构造真实 Accelerator（CPU / mixed_precision="fp16"）+ 真实 CPU GradScaler。

    负控制入口：init_scale=None 表示「不挂 scaler」（生产 bf16/fp32 档，此时 accelerate 的
    native_amp 必须为假），native_amp=False 表示「scaler 在、但 unscale 门关着」。
    """
    saved = (dict(AcceleratorState._shared_state), dict(PartialState._shared_state))

    def _make(
        *,
        native_amp: bool = True,
        init_scale: float | None = INIT_SCALE,
        growth_interval: int = 2000,
    ) -> Accelerator:
        if init_scale is None and native_amp:
            raise ValueError(
                "无 scaler 档在加速器侧必须 native_amp=False：生产 bf16/fp32 档不是 fp16，"
                "否则 unscale_gradients 会对 None 调 unscale_ 而崩溃"
            )
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._shared_state.clear()
        accelerator = Accelerator(cpu=True, mixed_precision="fp16")
        accelerator.native_amp = native_amp
        accelerator.scaler = (
            None
            if init_scale is None
            else torch.amp.GradScaler(
                "cpu", init_scale=init_scale, growth_interval=growth_interval
            )
        )
        assert GradientState().sync_gradients is True
        return accelerator

    yield _make
    AcceleratorState._shared_state.clear()
    AcceleratorState._shared_state.update(saved[0])
    PartialState._shared_state.clear()
    PartialState._shared_state.update(saved[1])
    GradientState._shared_state.clear()


def _counting_optimizer(model: torch.nn.Module, lr: float = 0.1):
    """返回 (optimizer, calls)：calls["n"] 数的是**底层** optimizer.step() 的真实调用次数。

    accelerate 在 scaler.step() 期间会用自己的探针替换 optimizer.step 再复原
    （accelerate/optimizer.py:66、:169-178）；被跳过的步根本走不到底层 step，所以这个计数器
    就是「权重到底有没有被更新」的位级证据，而不是靠 step_was_skipped 的自述。
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    calls = {"n": 0}
    raw_step = optimizer.step

    def counting_step(*args, **kwargs):
        calls["n"] += 1
        return raw_step(*args, **kwargs)

    # 必须在 prepare 之前替换，AcceleratedOptimizer 捕获到的才是这个包装
    optimizer.step = counting_step
    return optimizer, calls


def _run_scaled_step(accelerator, model, optimizer, *, inject_inf: bool = False) -> dict:
    """跑一次语义完整的 A 侧步（backward → clip → step），返回可观测量。"""
    x, y = _seeded_batch()
    optimizer.zero_grad(set_to_none=True)
    accelerator.backward(torch.nn.functional.mse_loss(model(x), y))
    if inject_inf:
        for parameter in model.parameters():
            parameter.grad.fill_(float("inf"))
    clip_norm = accelerator.clip_grad_norm_(list(model.parameters()), 10.0)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    after = [parameter.detach().clone() for parameter in model.parameters()]
    return {
        "clip_norm": clip_norm,
        "skipped": accelerator.optimizer_step_was_skipped,
        "weights_moved": any(not _bitwise_equal(b, a) for b, a in zip(before, after)),
        "weights_before": before,
        "weights_after": after,
        "scale": accelerator.scaler.get_scale(),
    }


def _plain_scaled_step(
    model, optimizer, scaler, *, inject_inf: bool = False
) -> tuple[float, bool]:
    """不经 accelerate 的最小缩放步，只用于 E4（scaler 自身状态机）。"""
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones(1, 4)).sum()).backward()
    if inject_inf:
        for parameter in model.parameters():
            parameter.grad.fill_(float("inf"))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    scaler.step(optimizer)
    scaler.update()
    after = [parameter.detach().clone() for parameter in model.parameters()]
    return scaler.get_scale(), any(not _bitwise_equal(b, a) for b, a in zip(before, after))


def _unit_model() -> torch.nn.Linear:
    """w=1、x=1、y=0 的解析模型：MSE 对 w 的梯度恒为 2，总范数恒为 2.0。"""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


class TestE4RealGradScalerStateMachine:
    """E4（位级）：被测 scaler 必须是真 torch 状态机，而不是替身。"""

    def test_scaler_under_test_is_real_torch_object(self):
        """E4 前置：真 GradScaler 的默认值与 state_dict 形状，任何替身都过不了。

        负控制：本用例是全局的**反替身守卫**——历史 P0（假绿）正是被类替身藏住的。
        """
        scaler = torch.amp.GradScaler("cpu")
        assert type(scaler) is torch.amp.GradScaler
        assert scaler.get_scale() == 65536.0
        assert scaler.state_dict() == {
            "scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 0,
        }

    def test_growth_backoff_and_weight_invariance(self):
        """E4 正例：干净步按 growth_interval 翻倍，溢出步退档且权重逐位不变。

        观测（init_scale=8、growth_interval=2）：8 → 16 → 16 → 32，溢出后回到 16。
        负控制：同一次构造里干净步必须真的移动权重（weights_moved is True），否则
        「溢出步权重不变」可能只是模型根本没训练起来。
        """
        model = _seeded_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)

        trace = [_plain_scaled_step(model, optimizer, scaler) for _ in range(4)]
        assert [scale for scale, _ in trace] == [8.0, 16.0, 16.0, 32.0]
        assert all(moved for _, moved in trace), "负控制失败：干净步必须真的更新权重"

        scale_after_overflow, moved = _plain_scaled_step(
            model, optimizer, scaler, inject_inf=True
        )
        assert scale_after_overflow == 16.0
        assert moved is False
        assert scaler.state_dict()["_growth_tracker"] == 0

    def test_state_dict_roundtrip_resumes_exact_scale_sequence(self):
        """E4 正例：state_dict 往返后后续 scale 轨迹与不中断的原对象逐位一致。

        轨迹取自一次溢出之后（scale 已退到 4）：连续四个干净步给出 [4, 8, 8, 16]。
        负控制：不加载状态的同构新 scaler 给出 [8, 16, 16, 32]，与恢复后的轨迹不同——
        从而证明「轨迹一致」不是因为两个 scaler 都从默认值出发。
        """
        model = _seeded_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        live = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
        _plain_scaled_step(model, optimizer, live, inject_inf=True)
        snapshot = live.state_dict()
        assert snapshot["scale"] == 4.0
        live_trace = [_plain_scaled_step(model, optimizer, live)[0] for _ in range(4)]

        restored = torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=99)
        restored.load_state_dict(snapshot)
        restored_model = _seeded_model()
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
        restored_trace = [
            _plain_scaled_step(restored_model, restored_optimizer, restored)[0]
            for _ in range(4)
        ]

        fresh = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
        fresh_model = _seeded_model()
        fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1)
        fresh_trace = [
            _plain_scaled_step(fresh_model, fresh_optimizer, fresh)[0] for _ in range(4)
        ]

        assert live_trace == restored_trace == [4.0, 8.0, 8.0, 16.0]
        assert fresh_trace == [8.0, 16.0, 16.0, 32.0]
        assert restored_trace != fresh_trace


class TestE1ComparatorMeta:
    """E1 元测试：先证明「逐位比较器」有牙齿，再用它下结论（t2 判据表的硬前提）。

    为什么必须有这一节：torch.equal 是**值相等**比较器，它把 -0.0 与 +0.0 判成相等，
    又对两个逐位完全相同的 NaN 判成不等（NaN != NaN）；torch.allclose(equal_nan=True)
    则把 payload 不同的 NaN 判成相等。三种情况都会让「逐位相等」的结论变成可绕过的假绿，
    所以 E1 的断言改用 uint8 视图比较，并在这里先把比较器本身的判别力钉死。
    """

    def test_value_equality_cannot_replace_bitwise_comparison(self):
        """E1 元测试（-0.0 / +0.0）：torch.equal 漏判，uint8 视图抓得住。

        负控制：同一个逐位比较器必须对真正的差异报不等，否则「一律报不等」也能通过本用例的
        前半段——下面用 AssertionError 明确要求它必须失败。
        """
        negative_zero = torch.tensor([-0.0], dtype=torch.float32)
        positive_zero = torch.tensor([0.0], dtype=torch.float32)
        assert torch.equal(negative_zero, positive_zero) is True
        assert _bits(negative_zero) != _bits(positive_zero)
        with pytest.raises(AssertionError, match="逐位不等"):
            _assert_bitwise_equal(negative_zero, positive_zero, label="-0.0 vs +0.0")

    def test_nan_payload_is_distinguished_and_identical_nan_is_not_a_false_alarm(self):
        """E1 元测试（NaN payload）：相同 payload 判等、不同 payload 判不等，且值比较器两头都错。

        负控制：torch.equal 对同 payload 的 NaN 报 False、allclose(equal_nan=True) 对不同 payload
        的 NaN 报 True——两种值比较器在这里各错一边，只有逐位比较器两边都对。
        """
        nan_a = _nan_with_payload(0x7FC00001)
        nan_a_twin = _nan_with_payload(0x7FC00001)
        nan_b = _nan_with_payload(0x7FC00002)

        assert torch.equal(nan_a, nan_a_twin) is False
        assert torch.allclose(nan_a, nan_b, equal_nan=True) is True
        assert _bits(nan_a) == _bits(nan_a_twin)
        assert _bits(nan_a) != _bits(nan_b)
        _assert_bitwise_equal(nan_a, nan_a_twin, label="same NaN payload")
        with pytest.raises(AssertionError, match="逐位不等"):
            _assert_bitwise_equal(nan_a, nan_b, label="different NaN payload")


class TestE1BitwiseInvariance:
    """E1（位级）：没有 scaler 时，接缝对数值一字节不动。

    接缝取生产调用点：library/training/loop.py:825 accelerator.backward(loss)、
    :898 accelerator.clip_grad_norm_(...)、:906 state.optimizer.step()。
    A 侧没有 B 项目那种「返回 loss 的 scale_loss_for_backward」形状，所以「同一对象」在 A 侧
    落到两条可判定事实：loss 张量未被替换/未被原地放大，且梯度与无 scaler 参照逐位相同。
    """

    def test_no_scaler_backward_and_clip_are_bitwise_passthrough(self, make_accelerator):
        """E1 正例：不挂 scaler（生产 bf16/fp32 档）→ backward + clip 与原生 torch 逐位一致。

        负控制：test_scaled_backward_multiplies_gradients_by_scale 用同一个比较方式，在挂上
        scaler 后必须不成立。
        """
        accelerator = make_accelerator(native_amp=False, init_scale=None)
        assert accelerator.scaler is None

        seam_model, reference_model = _seeded_model(), _seeded_model()
        seam_optimizer = accelerator.prepare(torch.optim.SGD(seam_model.parameters(), lr=0.1))
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        x, y = _seeded_batch()

        seam_optimizer.zero_grad(set_to_none=True)
        seam_loss = torch.nn.functional.mse_loss(seam_model(x), y)
        loss_object = seam_loss
        loss_pointer = seam_loss.data_ptr()
        loss_snapshot = seam_loss.detach().clone()
        accelerator.backward(seam_loss)
        seam_norm = accelerator.clip_grad_norm_(list(seam_model.parameters()), 0.75)
        seam_grads = [parameter.grad.detach().clone() for parameter in seam_model.parameters()]

        reference_optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(reference_model(x), y).backward()
        reference_norm = torch.nn.utils.clip_grad_norm_(
            list(reference_model.parameters()), 0.75
        )
        reference_grads = [
            parameter.grad.detach().clone() for parameter in reference_model.parameters()
        ]

        seam_optimizer.step()
        reference_optimizer.step()
        seam_parameters = [parameter.detach().clone() for parameter in seam_model.parameters()]
        reference_parameters = [
            parameter.detach().clone() for parameter in reference_model.parameters()
        ]

        _assert_bitwise_equal(seam_loss.detach(), loss_snapshot, label="loss")
        assert seam_loss is loss_object, "接缝不得替换调用方手里的 loss 对象（t9 P5）"
        assert seam_loss.data_ptr() == loss_pointer, "接缝不得换掉 loss 的存储"
        _assert_bitwise_equal(seam_norm, reference_norm, label="clip 范数")
        assert 0.0 < float(seam_norm) < 100.0, "范数必须是真实测量值，不能是 0 或 inf"
        for grads_a, grads_b in zip(seam_grads, reference_grads):
            _assert_bitwise_equal(grads_a, grads_b, label="梯度")
        for params_a, params_b in zip(seam_parameters, reference_parameters):
            _assert_bitwise_equal(params_a, params_b, label="参数")

    def test_scaled_backward_multiplies_gradients_by_scale(self, make_accelerator):
        """E1 负控制：挂上真 scaler 后，backward 接缝必须把梯度精确放大 INIT_SCALE 倍。

        这是上一条「逐位相同」的非恒真证据：同一条 backward 接缝在挂 scaler 时产生了
        精确 8（2^3）倍的梯度，说明接缝确实改写了数学，而不是什么都没做。
        """
        accelerator = make_accelerator(native_amp=True)
        reference_model = _seeded_model()
        scaled_model = _seeded_model()
        scaled_optimizer = accelerator.prepare(
            torch.optim.SGD(scaled_model.parameters(), lr=0.1)
        )
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        x, y = _seeded_batch()

        scaled_optimizer.zero_grad(set_to_none=True)
        accelerator.backward(torch.nn.functional.mse_loss(scaled_model(x), y))
        scaled_grads = [
            parameter.grad.detach().clone() for parameter in scaled_model.parameters()
        ]

        reference_optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(reference_model(x), y).backward()
        reference_grads = [
            parameter.grad.detach().clone() for parameter in reference_model.parameters()
        ]

        assert all(
            torch.equal(s, r * INIT_SCALE) for s, r in zip(scaled_grads, reference_grads)
        )
        assert not all(torch.equal(s, r) for s, r in zip(scaled_grads, reference_grads))

    def test_unscaled_seam_is_bitwise_equal_to_unscaled_reference(self, make_accelerator):
        """E1 正例（核心）：native_amp=True 时 clip 接缝先 unscale 再裁剪。

        scale=2^3 时乘/除可逆，因此裁剪后的范数与梯度必须与「不缩放」参照**逐位**相同：
        这就是 B 报告「裁剪前必须 unscale，否则拿放大 65536 倍的范数比未放大的阈值」的
        可判定形式。负控制：test_known_norm_proves_unscale_before_clip 里关掉 unscale 门，
        已知范数立刻从 2.0 变成 16.0。
        """
        accelerator = make_accelerator(native_amp=True)
        seam_model, reference_model = _seeded_model(), _seeded_model()
        seam_optimizer = accelerator.prepare(torch.optim.SGD(seam_model.parameters(), lr=0.1))
        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
        x, y = _seeded_batch()

        seam_optimizer.zero_grad(set_to_none=True)
        accelerator.backward(torch.nn.functional.mse_loss(seam_model(x), y))
        assert seam_model.weight.grad.abs().max() > 1.0, "scaled 梯度必须还带着 scale"
        seam_norm = accelerator.clip_grad_norm_(list(seam_model.parameters()), 0.75)
        seam_grads = [parameter.grad.detach().clone() for parameter in seam_model.parameters()]

        reference_optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(reference_model(x), y).backward()
        reference_norm = torch.nn.utils.clip_grad_norm_(
            list(reference_model.parameters()), 0.75
        )
        reference_grads = [
            parameter.grad.detach().clone() for parameter in reference_model.parameters()
        ]

        _assert_bitwise_equal(seam_norm, reference_norm, label="clip 范数")
        for grads_a, grads_b in zip(seam_grads, reference_grads):
            _assert_bitwise_equal(grads_a, grads_b, label="梯度")
        assert accelerator.scaler.get_scale() == INIT_SCALE, "unscale_ 不得改动 scale"

    def test_known_norm_proves_unscale_before_clip(self, make_accelerator):
        """E1 正例（解析值）：w=1、x=1、y=0 的 MSE 梯度恒为 2.0，裁剪接缝必须报 2.0。

        正例：native_amp=True + 真 scaler（init_scale=8）→ 梯度先被放大到 16，unscale 后
        范数精确等于 2.0。负控制：同一个真 scaler 但 native_amp=False → 范数精确等于 16.0，
        说明这条断言依赖 unscale 门真的生效，不是恒真。
        """
        def _norm(native_amp: bool) -> float:
            accelerator = make_accelerator(native_amp=native_amp)
            model = _unit_model()
            optimizer = accelerator.prepare(torch.optim.SGD(model.parameters(), lr=0.1))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                model(torch.ones(1, 1)), torch.zeros(1, 1)
            )
            accelerator.backward(loss)
            return float(accelerator.clip_grad_norm_(list(model.parameters()), 10.0))

        assert _norm(native_amp=True) == 2.0
        assert _norm(native_amp=False) == 16.0


class TestE2SkipSemanticSplit:
    """E2（位级 + 语义）：溢出步必须能分开「step 跑了」与「权重动了」。

    A 侧可观测面是 accelerate 的 AcceleratedOptimizer：step() 一定被调用（executed），溢出时
    内部 scaler.step/update 丢弃更新并把 step_was_skipped 置真（accelerate/optimizer.py:169-189）。
    但 A 的**主循环 loop.py 0 处消费**它，precision_report.py:206 只把 skipped / effective 各 +1，
    全仓也没有 optimizer_step_executed / optimizer_update_applied 两个命名：A 侧当前只有**计数级**
    语义（见 test_e2_a_side_keeps_only_counting_semantics）。所以下面两条用例断言的是
    「accelerate 原语可被观测」，不是「A 已实现双字段」。
    """

    def test_overflow_step_runs_but_does_not_move_weights(self, make_accelerator):
        """E2 正例：溢出步 executed=True、applied=False，权重逐位不变、scale 退档。

        用底层 optimizer.step() 的真实调用次数作为 applied 的位级证据，不依赖自述字段。
        负控制：同夹具的干净步必须 calls=1、skipped=False、权重确实移动。
        """
        accelerator = make_accelerator(growth_interval=2)
        model = _seeded_model()
        raw_optimizer, calls = _counting_optimizer(model)
        optimizer = accelerator.prepare(raw_optimizer)
        assert isinstance(optimizer, AcceleratedOptimizer)

        clean = _run_scaled_step(accelerator, model, optimizer)
        assert calls["n"] == 1
        assert clean["skipped"] is False
        assert clean["weights_moved"] is True
        assert clean["scale"] == INIT_SCALE

        overflow = _run_scaled_step(accelerator, model, optimizer, inject_inf=True)
        assert calls["n"] == 1, "溢出步不得真正走到底层 optimizer.step"
        assert overflow["skipped"] is True, "executed=True 的事实由 accelerate 的跳步标记承载"
        assert overflow["weights_moved"] is False
        for weights_b, weights_a in zip(
            overflow["weights_before"], overflow["weights_after"]
        ):
            _assert_bitwise_equal(weights_b, weights_a, label="溢出步权重")
        assert overflow["scale"] == INIT_SCALE * 0.5

        recovered = _run_scaled_step(accelerator, model, optimizer)
        assert calls["n"] == 2
        assert recovered["skipped"] is False
        assert recovered["weights_moved"] is True

    def test_overflow_step_hands_back_non_finite_clip_norm(self, make_accelerator):
        """E2 边界（位级事实）：溢出步的 clip 接缝返回 inf 范数。

        这就是 B 报告 §3.8「不能把 inf 当读数发布、必须回退到上一步 grad_norm」的成因。
        A 侧现状：library/training/loop.py:898 丢弃返回值，所以当前没有发布面，但也**没有任何
        防线**——任何未来的消费者都会拿到 inf。注意：B 报告那条「grad_norm 账本不被 inf 污染」
        子判据在 A 侧**不适用**（A 没有该账本，loop.py:1194 明确写 mean_grad_norm not tracked here），
        本用例只断言接缝确实交回 inf，不得记成「账本不被污染已通过」。
        负控制：干净步的范数必须有限且为正。
        """
        accelerator = make_accelerator()
        model = _seeded_model()
        optimizer = accelerator.prepare(torch.optim.SGD(model.parameters(), lr=0.1))

        clean = _run_scaled_step(accelerator, model, optimizer)
        assert torch.isfinite(clean["clip_norm"]).all()
        assert float(clean["clip_norm"]) > 0.0

        overflow = _run_scaled_step(accelerator, model, optimizer, inject_inf=True)
        assert overflow["skipped"] is True
        assert not torch.isfinite(overflow["clip_norm"]).all()
        assert float(overflow["clip_norm"]) == float("inf")

    def test_e2_a_side_keeps_only_counting_semantics(self):
        """E2 缺口钉子：A 侧当前只有计数级语义，没有 executed / applied 双字段。

        证据（t9 P6/P7 已复核）：library/training/loop.py 0 处出现 optimizer_step_was_skipped；
        precision_report.py:206 只做 skipped / effective 计数；全仓 0 处 optimizer_step_executed /
        optimizer_update_applied。缺口级别是「语义未命名、未在步内发布」，不是「能力不存在」。
        负控制：precision_report 的计数消费点必须仍然存在，否则这条钉子会退化成「全仓什么都没有」。
        """
        assert _path_symbol_occurrences(LOOP_PATH, "optimizer_step_was_skipped") == []
        assert _path_symbol_occurrences(
            LIBRARY_ROOT / "training" / "precision_report.py", "optimizer_step_was_skipped"
        ), "precision_report.py:206 的计数消费点必须仍在，否则口径证据消失"
        assert _symbol_occurrences("optimizer_step_executed", "optimizer_update_applied") == []

    def test_e2_a_side_counts_skipped_and_effective_updates(self):
        """E2 口径（计数级）：A 唯一消费跳步事实的地方是 precision_report 的 attempted/effective/skipped。

        最小 mock 面：只用 SimpleNamespace 提供 sync_gradients 与 optimizer_step_was_skipped 两个字段
        （卡片要求的「只暴露必要面」）；PrecisionReport 是真类（library/training/precision_report.py:200-209），
        本用例不涉及 scaler，也不构造假 scaler。
        负控制：sync_gradients=False（非累积边界）时三个计数器都必须不动，证明计数受 sync_gradients 门控，
        不是无条件累加。
        """
        from types import SimpleNamespace

        from library.training.precision_report import PrecisionReport

        report = PrecisionReport(SimpleNamespace())
        updates = report.data["updates"]

        def _state(sync: bool, skipped: bool):
            return SimpleNamespace(
                accelerator=SimpleNamespace(
                    sync_gradients=sync, optimizer_step_was_skipped=skipped
                )
            )

        report.optimizer(_state(True, False))
        report.optimizer(_state(True, False), after=True)
        report.optimizer(_state(True, True))
        report.optimizer(_state(True, True), after=True)
        assert (updates["attempted"], updates["effective"], updates["skipped"]) == (2, 1, 1)

        before = dict(updates)
        report.optimizer(_state(False, True))
        report.optimizer(_state(False, True), after=True)
        assert dict(updates) == before, "非累积边界不得推进任何计数"


class TestE5ScalerOwnershipGap:
    """E5（语义）：有活 scaler 时「梯度非有限」归 scaler 所有，保护层必须放弃该读数。

    2026-11-06：所有权判定已实现（library/training/scaler_ownership.py），本类中原「缺口钉」
    与 strict xfail 规格均已转正；缺陷 D1 的用例改名为
    test_e5_d1_scaler_owned_overflow_is_no_longer_fatal。

    A 侧当前**没有**任何所有权判定：全仓搜不到 scaled_gradient_overflow_is_scaler_owned 或等价
    命名（B 报告 §2.2 的 grad_scaler_runtime.py:97-143）。A 的 NaN 侧只有 fail-fast
    （library/training/finite_checks.py:16-33 与 :35-49），且唯一调用点被 debug 开关门控
    （library/training/loop.py:804-809 与 :860-861）。

    **结论：当前实现违反 E5（缺陷 D1）**——不是「E5 还没做」这么轻：debug 梯度有限性检查在
    loop.py:861，早于 :898 的 clip（clip 内部才 unscale），所以把 fp16 的例行上溢当发散判死。
    该缺陷用 test_e5_defect_d1_debug_grad_check_runs_before_unscale 钉住，本轮只记录不修复。
    """

    def test_e5_scaler_ownership_judgment_exists(self, make_accelerator):
        """E5 能力断言（2026-11-06 由缺口钉转正）：所有权判定已实现且成对成立。

        有活 scaler + 溢出梯度 → True（归 scaler 所有，保护层放弃读数）；
        无 scaler 同输入 → False（无人接管，保护层必须自己处置）。
        """
        from library.training.scaler_ownership import (
            scaled_gradient_overflow_is_scaler_owned,
        )

        assert _symbol_occurrences(
            "scaled_gradient_overflow_is_scaler_owned", "is_scaler_owned", "scaler_owned"
        ), "所有权判定符号必须存在于 library/"

        accelerator = make_accelerator(native_amp=True)
        model = _seeded_model()
        optimizer = accelerator.prepare(torch.optim.SGD(model.parameters(), lr=0.1))
        _run_scaled_step(accelerator, model, optimizer, inject_inf=True)
        overflow_grads = [parameter.grad for parameter in model.parameters()]
        assert scaled_gradient_overflow_is_scaler_owned(accelerator, overflow_grads) is True

        plain_accelerator = make_accelerator(native_amp=False, init_scale=None)
        assert (
            scaled_gradient_overflow_is_scaler_owned(plain_accelerator, overflow_grads)
            is False
        )

    def test_e5_overflow_ownership_pair_live_scaler_vs_no_scaler(self, make_accelerator):
        """E5 成对断言（正例 + 负控制，真 CPU scaler，无 mock）：同一组非有限梯度在两个世界里结局不同。

        正例（有活 scaler）：backward → clip → step 全链路走完，step 被跳过（skipped=True）、
        权重逐位不变、scale 8→4 —— 这就是「梯度非有限归 scaler 所有」的行为层事实。
        负控制（无 scaler，生产 bf16/fp32 档）：同构造同样注入 inf → clip 把 inf 梯度乘成 NaN
        （inf*0=NaN），optimizer.step 照常执行、权重变成非有限 —— 没有任何机制接管该读数。
        两半必须同时成立：只断言「有 scaler 时权重不变」无法区分「所有权判定」与「碰巧没更新」。
        """
        accelerator = make_accelerator(native_amp=True)
        model = _seeded_model()
        optimizer = accelerator.prepare(torch.optim.SGD(model.parameters(), lr=0.1))
        overflow = _run_scaled_step(accelerator, model, optimizer, inject_inf=True)
        assert overflow["skipped"] is True
        assert overflow["weights_moved"] is False
        for weights_b, weights_a in zip(
            overflow["weights_before"], overflow["weights_after"]
        ):
            _assert_bitwise_equal(weights_b, weights_a, label="有 scaler 的溢出步权重")
        assert overflow["scale"] == INIT_SCALE * 0.5

        plain_accelerator = make_accelerator(native_amp=False, init_scale=None)
        plain_model = _seeded_model()
        plain_optimizer = plain_accelerator.prepare(
            torch.optim.SGD(plain_model.parameters(), lr=0.1)
        )
        x, y = _seeded_batch()
        plain_optimizer.zero_grad(set_to_none=True)
        plain_accelerator.backward(torch.nn.functional.mse_loss(plain_model(x), y))
        for parameter in plain_model.parameters():
            parameter.grad.fill_(float("inf"))
        plain_norm = plain_accelerator.clip_grad_norm_(list(plain_model.parameters()), 10.0)
        plain_optimizer.step()
        assert plain_accelerator.optimizer_step_was_skipped is False
        assert not torch.isfinite(plain_norm).all(), "无 scaler 时裁剪读数必须是 inf（无人接管）"
        assert not all(
            torch.isfinite(parameter).all() for parameter in plain_model.parameters()
        ), "无 scaler 时非有限梯度会直接写进权重（负控制必须与正例结局不同）"

    def test_e5_spec_scaler_owned_overflow_is_not_a_training_failure(self, make_accelerator):
        """E5 规格（2026-11-06 已实现，strict xfail 已转正）：有活 scaler 时非有限梯度不得被判失败。

        目标行为（B 报告 grad_scaler_runtime.py:97-143 的只读契约）：有 scaler 且该步溢出 →
        返回 True，保护层放弃读数、只记一条 scaler_overflow_ignored；无 scaler 同输入 →
        返回 False，原有 NaN 判定保持。负控制就写在同一条用例里。
        """
        from library.training.scaler_ownership import (
            scaled_gradient_overflow_is_scaler_owned,
        )

        accelerator = make_accelerator(native_amp=True)
        model = _seeded_model()
        optimizer = accelerator.prepare(torch.optim.SGD(model.parameters(), lr=0.1))
        _run_scaled_step(accelerator, model, optimizer, inject_inf=True)
        overflow_grads = [parameter.grad for parameter in model.parameters()]

        assert scaled_gradient_overflow_is_scaler_owned(accelerator, overflow_grads) is True

        plain_accelerator = make_accelerator(native_amp=False, init_scale=None)
        assert plain_accelerator.scaler is None
        assert (
            scaled_gradient_overflow_is_scaler_owned(plain_accelerator, overflow_grads) is False
        )

    def test_e5_d1_scaler_owned_overflow_is_no_longer_fatal(self, make_accelerator):
        """E5/D1（2026-11-06 已修复）：scaler 所有的溢出不得再被保护层判成训练失败。

        构造（t9 P3 同款）：有限 loss=1.0、真梯度=2.0、scale=65536 → fp16 梯度上溢成 inf。
        修复前 loop.py 的 debug 检查读到仍乘 scale 的梯度并抛 FloatingPointError；
        修复后同一读数被判定为 scaler 所有：`check_trainable_grads_finite(model,
        scaler_owned=True)` 只返回 offenders 不抛错，真 scaler 随后 skip 且权重逐位不变。
        负控制 1：不挂 scaler 时同一组 inf 梯度仍必须 fail-closed（默认 `scaler_owned=False`）。
        负控制 2：不挂 scaler 时梯度是有限的 2.0，检查必须放行。
        """
        plain_model = _unit_model().half()
        plain_loss = torch.nn.functional.mse_loss(
            plain_model(torch.ones(1, 1).half()).float(), torch.zeros(1, 1)
        )
        plain_loss.backward()
        assert [float(parameter.grad) for parameter in plain_model.parameters()] == [2.0]
        check_trainable_grads_finite(plain_model)

        accelerator = make_accelerator(native_amp=True, init_scale=65536.0)
        model = _unit_model().half()
        optimizer = accelerator.prepare(torch.optim.SGD(model.parameters(), lr=0.1))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            model(torch.ones(1, 1).half()).float(), torch.zeros(1, 1)
        )
        assert float(loss.detach()) == 1.0 and torch.isfinite(loss).all()
        accelerator.backward(loss)
        assert all(torch.isinf(parameter.grad).all() for parameter in model.parameters())
        # 负控制 1：无 scaler 语义（默认参数）仍 fail-closed
        with pytest.raises(FloatingPointError, match="non-finite trainable gradients"):
            check_trainable_grads_finite(model)

        # 修复后行为：所有权判定为 scaler 所有 → 只报告 offenders，不抛错
        from library.training.scaler_ownership import (
            scaled_gradient_overflow_is_scaler_owned,
        )

        assert (
            scaled_gradient_overflow_is_scaler_owned(
                accelerator, [parameter.grad for parameter in model.parameters()]
            )
            is True
        )
        reported = check_trainable_grads_finite(model, scaler_owned=True)
        assert reported, "scaler 所有的溢出必须回传 offenders 供记录"

        # 对照：真 scaler 对同一组梯度只会跳步（patch 复刻 full_fp16 生产路径 bootstrap.py:1094）
        patch_accelerator_for_fp16_training(accelerator)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.step()
        after = [parameter.detach().clone() for parameter in model.parameters()]
        assert accelerator.optimizer_step_was_skipped is True
        for weights_b, weights_a in zip(before, after):
            _assert_bitwise_equal(weights_b, weights_a, label="D1 溢出步权重")
        assert accelerator.scaler.get_scale() == 65536.0 * 0.5


class TestE3WindowAccountingGap:
    """E3（计数级）：非有限窗口整窗作废——重放 1 次，仍失败则丢弃且不推进任何计数。

    **状态（2026-11-06，E3 用户裁定后）：已实现。** 窗口账本与重放策略在
    ``library/training/window_accounting.py``（``WINDOW_MAX_REPLAYS = 1``），``loop.py`` 只接线：
    窗口边界要么 commit（账本 → scheduler → adaptive → precision → global_step），要么 replay，
    要么 discard。因此本类原来的「缺口钉」与 strict xfail 规格都已转正：

    * ``_window_readings_after_nonfinite_batch`` 改为真实驱动一段最小 loop（首批 NaN + 重放成功）；
    * ``test_e3_window_accounting_seams_absent`` → ``test_e3_window_accounting_capabilities_exist``；
    * ``test_e3_spec_nonfinite_batch_discards_whole_accumulation_window`` 去掉 xfail，改为正式断言。

    已知限制：本仓**没有模型 EMA**，``state.ema_observation`` 是显式占位槽（只在提交路径写快照），
    本类用它证明「丢弃窗口时不触碰该槽」，不代表真实 EMA 更新。
    """

    def test_e3_nonfinite_loss_helper_still_fails_fast_when_asked(self):
        """debug 钩子仍是 fail-fast：窗口重放不改、也不依赖 finite_checks 的抛错语义。

        生产默认路径（debug 关闭）现在由窗口层无条件判定并重放/丢弃（见下面两条用例）；
        显式打开 ``--debug_finite_checks`` 时仍旧立刻中止，便于定位首个坏 batch。
        """
        with pytest.raises(FloatingPointError, match="non-finite training loss"):
            check_loss_finite(torch.tensor(float("nan")), mixed_precision="fp16")

        network = torch.nn.Linear(2, 2)
        network.weight.grad = torch.full_like(network.weight, float("inf"))
        with pytest.raises(FloatingPointError, match="non-finite trainable gradients"):
            check_trainable_grads_finite(network)

        # 负控制：有限输入的同一入口必须放行，否则上面两条 cannot-fail。
        check_loss_finite(torch.tensor(1.5), mixed_precision="fp16")

    def test_e3_loss_finiteness_is_checked_unconditionally_in_the_window(self):
        """E3 静态不变量：窗口层的 loss 有限性检查**不再**被 debug 开关门控。

        2026-11-06 起 ``_run_microbatch`` 用 ``window_accounting.loss_value`` 无条件判定，
        使「非有限 loss → 整窗作废」在默认配置下也成立（D 缺口 D1/E3 的接线）。
        debug 钩子（``check_loss_finite``）本身仍只在 debug 打开时调用——两条断言必须同时成立，
        否则要么窗口保护缺席，要么默认训练被 fail-fast 打死。
        """
        assert debug_finite_enabled(_args_stub()) is False
        assert debug_finite_enabled(_args_stub(debug_finite_checks=True)) is True

        guarded, bare = _check_loss_finite_call_sites(LOOP_PATH)
        assert guarded, "loop.py 里必须仍能定位到 check_loss_finite 调用点"
        assert bare == [], f"check_loss_finite 出现无 debug 门控的调用点：{bare}"

        unconditional = _path_symbol_occurrences(
            LOOP_PATH, "loss_value", "losses_are_finite"
        )
        assert unconditional, "loop.py 必须保留无条件的窗口级 loss 有限性判定"

    def test_e3_window_accounting_capabilities_exist(self):
        """E3 能力断言（2026-11-06 由缺口钉转正）：窗口账本与重放策略已实现。

        负控制：``window_accounting`` 的三个累计计数与三个窗口级账本必须全部存在，
        且 ``LoopState`` 上同样有这些字段——只加模块字段而没接进 state 同样不合格。
        """
        from library.training import window_accounting
        from library.training.loop import LoopState

        assert window_accounting.WINDOW_MAX_REPLAYS == 1, "用户裁定：同一窗口最多重放 1 次"
        state = LoopState.__dataclass_fields__
        for field_name in (
            "window_attempts",
            "window_replays",
            "window_discards",
            "window_total_loss",
            "window_num_steps",
            "window_last_loss",
            "ema_observation",
        ):
            assert field_name in state, f"LoopState 缺少 E3 字段 {field_name}"

        buffer = window_accounting.WindowBuffer()
        buffer.add("batch", torch.tensor(2.0))
        assert buffer.failure_reason(update_skipped=False) is None
        assert buffer.failure_reason(update_skipped=True) == window_accounting.REASON_UPDATE_SKIPPED
        nan_buffer = window_accounting.WindowBuffer()
        nan_buffer.add("batch", torch.tensor(float("nan")))
        assert (
            nan_buffer.failure_reason(update_skipped=False)
            == window_accounting.REASON_NONFINITE_LOSS
        )

    def test_e3_boundary_window_discard_is_not_single_step_skip(self):
        """E3/E2 边界：单步跳步只冻结权重、计数照走；窗口丢弃要求连计数一起冻结。

        可判定事实：``global_step += 1`` 必须仍在 ``sync_gradients`` 门控内，且不得带跳步门控。
        E3 落地后跳步本身会在窗口层被重放/丢弃（见 ``tests/test_window_replay.py``），
        但「单步跳过」与「窗口丢弃」仍是两种语义，不能互相替代。
        """
        guarded, bare = _global_step_increments_in_sync_gate(LOOP_PATH)
        assert guarded, "loop.py 必须仍能在 sync_gradients 门控内定位到 global_step 自增"
        assert bare == [], f"global_step 出现累积边界外的自增：{bare}"

    def test_e3_spec_nonfinite_batch_discards_whole_accumulation_window(self):
        """E3 规格（2026-11-06 转正）：失败尝试对五个计数器与账本三元组的副作用必须为零。

        观测口径：``at_window_start`` = 上一个窗口提交完成时的快照；``after`` = 本窗口失败尝试
        已经跑完、重放尚未开始的快照。两者逐位相等 ⇒ 失败尝试零副作用。
        浮点容差：abs=0（逐位相等）——失败尝试的账本不得留下任何求和残渣。
        补充证据（``_evidence``）证明重放确实成功提交了一次，排除「什么都没发生」的假绿。
        """
        readings = _window_readings_after_nonfinite_batch()
        evidence = readings.pop("_evidence")
        missing = [
            f"{name}（{description}）"
            for name, description, _ in _WINDOW_ACCOUNTING_POINTS
            if readings.get(name) is None
        ]
        assert not missing, "窗口账本观测点缺失：" + "、".join(missing)

        for name, description, _ in _WINDOW_ACCOUNTING_POINTS:
            assert readings[name]["after"] == readings[name]["at_window_start"], (
                f"{name}（{description}）必须与窗口起点逐位相等：窗口丢弃 ≠ 单步跳过"
            )
        for ledger in ("total_loss", "num_steps", "last_loss"):
            assert readings[ledger]["after"] == pytest.approx(
                readings[ledger]["at_window_start"], rel=0, abs=0
            )

        # 防假绿：重放确实发生了，并且只提交了一次（不是「全都没动」）。
        assert evidence["replays"] == 1, "首批 NaN 必须触发一次重放"
        assert evidence["discards"] == 0, "重放成功不得计入丢窗"
        assert evidence["global_step"] == 2, "第一个窗口 + 重放成功后只推进 2 步"
        assert evidence["total_loss"] == pytest.approx(2.0, abs=1e-12), "账本只记成功的重放"
        assert evidence["ema_step"] == 2, "EMA 占位槽只在提交路径更新"


class TestAccelerateInternalContract:
    """accelerate 内部契约 canary（本文件已验证 accelerate 1.13.0）。

    接缝用例依赖三个内部面：Accelerator.native_amp（unscale 门）、AcceleratorState._mixed_precision
    （门控条件的另一半）、accelerate.optimizer.AcceleratedOptimizer.step_was_skipped。
    它们不是公开 API；升级 accelerate 后若消失或语义变化，本类必须**显式失败**，不得退化成跳过。
    """

    def test_internals_exist_and_unscale_gate_semantics_unchanged(self, make_accelerator):
        """canary：版本固定 + 三个内部面存在 + unscale 门双向语义（True 除以 scale / False 是 no-op）。

        负控制：native_amp=False 的同一调用必须是 no-op（梯度逐位不变），否则「门」的语义已经变了，
        上层的 E1/E2 结论全部失效。
        """
        import accelerate

        assert accelerate.__version__ == "1.13.0", (
            "accelerate 版本变化：本文件基于 1.13.0 内部面验证，升级后必须重新核对 "
            "native_amp 门控 / AcceleratedOptimizer.step_was_skipped / AcceleratorState._mixed_precision"
        )

        accelerator = make_accelerator(native_amp=True)
        assert hasattr(accelerator, "native_amp"), "Accelerator.native_amp 消失"
        assert accelerator.native_amp is True
        assert accelerator.mixed_precision == "fp16"
        state = AcceleratorState()
        assert hasattr(state, "_mixed_precision"), "AcceleratorState._mixed_precision 消失"
        assert state._mixed_precision == "fp16"

        model = _seeded_model()
        optimizer = accelerator.prepare(torch.optim.SGD(model.parameters(), lr=0.1))
        assert isinstance(optimizer, AcceleratedOptimizer)
        assert hasattr(optimizer, "step_was_skipped"), "AcceleratedOptimizer.step_was_skipped 消失"
        assert optimizer.step_was_skipped is False
        x, y = _seeded_batch()
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(torch.nn.functional.mse_loss(model(x), y))
        scaled = [parameter.grad.detach().clone() for parameter in model.parameters()]
        accelerator.unscale_gradients(optimizer=optimizer)
        unscaled = [parameter.grad.detach().clone() for parameter in model.parameters()]
        for unscaled_grad, scaled_grad in zip(unscaled, scaled):
            _assert_bitwise_equal(
                unscaled_grad * INIT_SCALE, scaled_grad, label="native_amp=True 时 unscale 必须除以 scale"
            )

        plain_accelerator = make_accelerator(native_amp=False, init_scale=None)
        plain_model = _seeded_model()
        plain_optimizer = plain_accelerator.prepare(
            torch.optim.SGD(plain_model.parameters(), lr=0.1)
        )
        plain_optimizer.zero_grad(set_to_none=True)
        plain_accelerator.backward(torch.nn.functional.mse_loss(plain_model(x), y))
        plain_before = [parameter.grad.detach().clone() for parameter in plain_model.parameters()]
        plain_accelerator.unscale_gradients(optimizer=plain_optimizer)
        plain_after = [parameter.grad.detach().clone() for parameter in plain_model.parameters()]
        for after_grad, before_grad in zip(plain_after, plain_before):
            _assert_bitwise_equal(after_grad, before_grad, label="native_amp=False 时 unscale 必须是 no-op")


def test_patch_accelerator_for_fp16_training_allows_fp16_unscale(make_accelerator):
    """A 侧 fp16 参数档（位级）：patch 只放开 allow_fp16，未 patch 时必须拒绝。

    生产路径：library/training/bootstrap.py:1093-1094 在 args.full_fp16 时调用
    library/runtime/accelerator.py:201-217 的 patch_accelerator_for_fp16_training。
    负控制：同一个真 scaler + 同样的 fp16 梯度，patch 之前 unscale_ 必须抛
    ValueError("Attempting to unscale FP16 gradients.")，patch 之后必须成功且梯度保持有限。
    """

    def _fp16_setup():
        torch.manual_seed(3)
        layer = torch.nn.Linear(3, 2).half()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        return layer, optimizer

    layer, optimizer = _fp16_setup()
    unpatched_scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    unpatched_scaler.scale(layer(torch.ones(1, 3).half()).float().sum()).backward()
    assert {parameter.grad.dtype for parameter in layer.parameters()} == {torch.float16}
    with pytest.raises(ValueError, match="unscale FP16 gradients"):
        unpatched_scaler.unscale_(optimizer)

    accelerator = make_accelerator(native_amp=True)
    patched_scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
    accelerator.scaler = patched_scaler
    patch_accelerator_for_fp16_training(accelerator)
    layer, optimizer = _fp16_setup()
    patched_scaler.scale(layer(torch.ones(1, 3).half()).float().sum()).backward()
    patched_scaler.unscale_(optimizer)
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def _args_stub(**overrides) -> object:
    from types import SimpleNamespace

    base = {"debug_finite_checks": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def _parse(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8"))


def _symbol_occurrences(*symbols: str) -> list[str]:
    """在 library/ 源码里搜符号（AST 级，避免 grep 的注释/字符串误报）。"""
    return _path_symbol_occurrences(LIBRARY_ROOT, *symbols)


def _path_symbol_occurrences(root: Path, *symbols: str) -> list[str]:
    """在 root（文件或目录）里搜符号；目录会递归 *.py。"""
    wanted = set(symbols)
    hits: list[str] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for path in paths:
        try:
            tree = _parse(path)
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - 防御性
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
            else:
                continue
            if name in wanted:
                hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return hits


def _check_loss_finite_call_sites(path: Path) -> tuple[list[int], list[int]]:
    """返回 (被 debug 门控的 check_loss_finite 调用点行号, 裸调用点行号)。"""
    tree = _parse(path)
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not any(
            isinstance(inner, ast.Name) and inner.id == "debug_finite_enabled"
            for inner in ast.walk(node.test)
        ):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                if inner.func.id == "check_loss_finite":
                    guarded.add(inner.lineno)
    every = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check_loss_finite"
    ]
    return sorted(guarded), sorted(set(every) - guarded)


def _global_step_increments_in_sync_gate(path: Path) -> tuple[list[int], list[int]]:
    """返回 (sync_gradients 门控内的 global_step += 1 行号, 门控外的行号)。

    E3/E2 边界证据：A 的 global_step 自增必须在累积边界内，且不带跳步门控——
    否则「单步跳过」与「窗口丢弃」就分不开了。
    """
    tree = _parse(path)
    guarded: list[int] = []
    bare: list[int] = []

    def _visit(node: ast.AST, in_sync_gate: bool) -> None:
        if isinstance(node, ast.If):
            mentions_sync = any(
                isinstance(inner, ast.Attribute) and inner.attr == "sync_gradients"
                for inner in ast.walk(node.test)
            )
            for child in node.body:
                _visit(child, in_sync_gate or mentions_sync)
            for child in node.orelse:
                _visit(child, in_sync_gate)
            return
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Attribute)
            and node.target.attr == "global_step"
        ):
            (guarded if in_sync_gate else bare).append(node.lineno)
        for child in ast.iter_child_nodes(node):
            _visit(child, in_sync_gate)

    _visit(tree, False)
    return sorted(guarded), sorted(bare)


def _window_readings_after_nonfinite_batch() -> dict:
    """E3 规格的观测入口：真实驱动一段最小 loop（首批 NaN + 重放成功）。

    场景：窗口 1 干净提交（1.0）→ 窗口 2 首批 loss 为 NaN → 整窗重放（2.0）→ 提交。
    观测点：``at_window_start`` 取窗口 1 提交完成、窗口 2 尚未开始时的快照；
    ``after`` 取窗口 2 的失败尝试跑完、重放尚未开始时（``loop._replay_window`` 钩子）的快照。
    返回 ``{观测点: {"at_window_start": X, "after": Y}}``，外加 ``_evidence`` 的补充读数。
    """
    try:  # pytest 把 tests/ 放进 sys.path；显式回退便于直接导入
        from window_replay_harness import Harness
    except ImportError:  # pragma: no cover
        from tests.window_replay_harness import Harness

    snapshots: dict[str, dict] = {}
    with pytest.MonkeyPatch.context() as patch:
        harness = Harness(
            patch,
            num_steps=1,
            loss_plan=[1.0, float("nan"), 2.0],
            batches=2,
            on_before_forward=lambda item: (
                snapshots.setdefault("at_window_start", item.readings())
                if item.forward_calls == 1  # 第二次 forward = 第二个窗口的第一个微批
                else None
            ),
            on_before_replay=lambda item: snapshots.setdefault("after", item.readings()),
        )
        harness.run()
        final = harness.readings()

    assert "at_window_start" in snapshots and "after" in snapshots, "观测钩子未触发"
    readings = {
        name: {
            "at_window_start": snapshots["at_window_start"][name],
            "after": snapshots["after"][name],
        }
        for name, _, _ in _WINDOW_ACCOUNTING_POINTS
    }
    readings["_evidence"] = {
        "replays": harness.state.window_replays,
        "discards": harness.state.window_discards,
        "global_step": harness.state.global_step,
        "total_loss": harness.state.window_total_loss,
        "ema_step": (harness.state.ema_observation or {}).get("step"),
        "final_scheduler_steps": final["scheduler"],
        "final_ckpt": final["ckpt"],
    }
    return readings
