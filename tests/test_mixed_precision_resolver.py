from __future__ import annotations

from pathlib import Path
import types

import pytest

from library.training.precision_policy import resolve_mixed_precision

ROOT = Path(__file__).resolve().parents[1]


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


def test_train_session_resolves_mixed_precision_before_first_prepare_dtype():
    source = (ROOT / "library" / "training" / "train_session.py").read_text(
        encoding="utf-8"
    )
    resolve_idx = source.find("resolve_mixed_precision(args)")
    first_prepare_idx = source.find("prepare_dtype(args)")
    accelerator_idx = source.find("prepare_accelerator(args)")
    assert resolve_idx != -1
    assert first_prepare_idx != -1
    assert accelerator_idx != -1
    assert resolve_idx < first_prepare_idx
    assert resolve_idx < accelerator_idx



def test_recheck_after_resolve_raises_with_rewrite_context(monkeypatch):
    """P0-1 回归：sm<8 改写引入的 fail-closed 冲突必须报错，且归因于自动降级。

    改写前 ``check_training_compat`` 看到的是 bf16（放行），改写后才变成 fp16
    （本应拒绝）。如果不复检，Z-Image 会以未经 islands 的 fp16 结构启动、Qwen 的
    flash+fp16 硬 code 从不生效。报错必须点明是「自动降级」引入的，否则用户会
    以为自己写错了配置。
    """
    from library.training import extra_args as ea
    from library.training.compat_matrix import TrainingCompatIssue, TrainingCompatResult

    seen: list[object] = []

    def _fake_check(config):
        seen.append(config)
        return TrainingCompatResult(
            errors=(
                TrainingCompatIssue(
                    code="z_image_bf16_only",
                    key="mixed_precision",
                    message="Z-Image 只支持 bf16 / fp32；fp16 需要 adaptive islands",
                ),
            ),
            warnings=(),
            mutations=(),
        )

    monkeypatch.setattr(ea, "check_training_compat", _fake_check)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))

    args = _fake_args("bf16")
    resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16", "sm<8 必须把 bf16 改写为 fp16"

    with pytest.raises(ValueError) as exc:
        ea.recheck_compat_after_precision_resolve(args)

    message = str(exc.value)
    assert "自动" in message, "报错必须归因于自动改写，而不是让用户以为自己配错了"
    assert "sm80+" in message, "报错要指出出路（换支持原生 bf16 的硬件）"
    assert "z_image_bf16_only" in message, "必须带上被绕过的 code，便于排障"
    assert len(seen) == 1, "复检只调用一次 compat"


def test_recheck_after_resolve_is_silent_when_rewrite_is_clean(monkeypatch):
    """改写没有引入冲突时不得干扰启动。"""
    from library.training import extra_args as ea
    from library.training.compat_matrix import TrainingCompatResult

    monkeypatch.setattr(
        ea,
        "check_training_compat",
        lambda config: TrainingCompatResult(errors=(), warnings=(), mutations=()),
    )

    ea.recheck_compat_after_precision_resolve(_fake_args("fp16"))


def test_train_session_rechecks_compat_after_precision_rewrite():
    """P0-1 接线：复检必须在改写之后、首次 prepare_dtype/model load 之前。

    ``resolve_mixed_precision`` 自己不能内联这个复检——它跑在大量精简 args 上
    （其他单测只造 ``mixed_precision``），完整的 ``check_training_compat`` 会缺键
    报错。所以复检放在 ``train_session`` 的编排点，这里用源码顺序守住。
    """
    source = (ROOT / "library" / "training" / "train_session.py").read_text(encoding="utf-8")
    resolve_idx = source.find("resolve_mixed_precision(args)")
    recheck_idx = source.find("recheck_compat_after_precision_resolve(args)")
    first_prepare_idx = source.find("prepare_dtype(args)")
    assert resolve_idx != -1
    assert recheck_idx != -1, (
        "缺少改写后的兼容复检：sm<8 的 bf16→fp16 会绕过 z_image_bf16_only 与 "
        "qwen_image_2_1_mixed_precision_flash 两个硬 code"
    )
    assert resolve_idx < recheck_idx, "复检必须在 dtype 改写之后"
    assert recheck_idx < first_prepare_idx, "复检必须早于首次 prepare_dtype（模型加载前）"
