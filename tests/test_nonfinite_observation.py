"""R6 / R7 CPU contract: no-scaler warning and non-finite observation (read path untouched).

Covers M12 (warning-only when fp16 was requested but no scaler exists), M21 (source
non-finite vs cast overflow split, identity, counters, throttled logs, tensor behaviour
unchanged), and M26③④⑤ (the scaler bucket is empty without a live scaler; a real scaler
skip is attributable; the three attribution layers are all published).

The policy under test is *observation only*: the NaN-gated scrub and ``nan_to_num``, the
dtypes, the returned tensors and the cache bytes must stay exactly as they were. Any test
here that changed behaviour would be a defect, so each case asserts the unchanged value
alongside the new counter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training import batch_step, numerical_observation, window_coordination
from library.training.precision_report import PrecisionReport


@pytest.fixture(autouse=True)
def _clean_observation():
    numerical_observation.reset()
    yield
    numerical_observation.reset()


def _accelerator(device="cpu"):
    return SimpleNamespace(device=torch.device(device), print=print)


def _latent_args():
    return SimpleNamespace(vae_batch_size=None)


def _trainer():
    return SimpleNamespace(
        encode_images_to_latents=lambda *_a, **_k: None,
        shift_scale_latents=lambda _args, latents: latents,
    )


# ---------------------------------------------------------------------------
# M21 a/b — the read path keeps its behaviour; only the reporting is new
# ---------------------------------------------------------------------------
def test_nan_latents_are_still_zeroed_and_reported_with_the_original_message(capsys):
    tensor = torch.tensor([[1.0, float("nan"), 3.0]])
    out = batch_step._prepare_latents(
        _latent_args(), _trainer(), None, {"latents": tensor}, _accelerator(), torch.float32
    )

    captured = capsys.readouterr()
    assert "NaN found in cached latents, replacing with zeros" in captured.out, (
        "既有用户可见信息保持原文"
    )
    assert torch.equal(out, torch.tensor([[1.0, 0.0, 3.0]])), "清洗策略未改"
    assert out.dtype == torch.float32, "dtype 未改"
    snap = numerical_observation.snapshot()
    assert snap["sites"][numerical_observation.SITE_LATENTS][
        numerical_observation.KIND_SOURCE_NAN
    ] == 1, "计数 = 非有限元素个数"
    assert snap["events"][numerical_observation.SITE_LATENTS][
        numerical_observation.KIND_SOURCE_NAN
    ] == 1, "events = 携带非有限的张量次数"
    assert snap["last"][numerical_observation.SITE_LATENTS][
        numerical_observation.KIND_SOURCE_NAN
    ]["elements"] == 3


def test_inf_only_latents_are_reported_but_never_modified(capsys):
    tensor = torch.tensor([[1.0, float("inf"), 3.0]])
    out = batch_step._prepare_latents(
        _latent_args(), _trainer(), None, {"latents": tensor}, _accelerator(), torch.float32
    )

    captured = capsys.readouterr()
    assert "NaN found" not in captured.out, "Inf-only 分支不得伪造 NaN 消息"
    assert torch.isinf(out[0, 1]), "未改变的既有行为：Inf 不被清洗"
    assert torch.equal(out, tensor)
    snap = numerical_observation.snapshot()
    latents = snap["sites"][numerical_observation.SITE_LATENTS]
    assert latents[numerical_observation.KIND_SOURCE_INF] == 1
    assert numerical_observation.KIND_SOURCE_NAN not in latents


def test_read_path_adds_no_collective_and_no_device_transfer(monkeypatch):
    """M21：读取层不得新增 collective（也即不得引入新的失败点）。"""
    import torch.distributed as dist

    def boom(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("read path must not reduce")

    monkeypatch.setattr(dist, "all_reduce", boom, raising=False)
    monkeypatch.setattr(dist, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(dist, "is_initialized", lambda: True, raising=False)

    tensor = torch.full((2, 3), float("nan"))
    out = batch_step._prepare_latents(
        _latent_args(), _trainer(), None, {"latents": tensor}, _accelerator(), torch.float32
    )
    assert torch.count_nonzero(out) == 0


# ---------------------------------------------------------------------------
# M21 c — source non-finite vs cast overflow are distinct kinds
# ---------------------------------------------------------------------------
def test_cast_overflow_is_classified_apart_from_source_nonfinite():
    from library.preprocess.text import _report_cast_numeric

    # 源在 float32 里有限、但放不进 float16（上限 65504）⇒ 转换溢出
    source = torch.tensor([[1.0, 1e5]], dtype=torch.float32)
    stored = source.to(torch.float16)
    _report_cast_numeric("prompt_embeds", source, stored, torch.float16)

    snap = numerical_observation.snapshot()
    site = "preprocess.text.prompt_embeds"
    assert snap["sites"][site] == {numerical_observation.KIND_CAST_OVERFLOW: 1}
    entry = snap["last"][site][numerical_observation.KIND_CAST_OVERFLOW]
    assert entry["before_dtype"] == "torch.float32" and entry["after_dtype"] == "torch.float16"
    assert entry["shape"] == [1, 2], "来源身份/形状只记录有界元数据"


def test_source_nonfinite_is_not_reported_as_cast_overflow():
    from library.preprocess.text import _report_cast_numeric

    source = torch.tensor([[float("nan"), float("inf")]], dtype=torch.float32)
    _report_cast_numeric("prompt_embeds", source, source.to(torch.float16), torch.float16)

    snap = numerical_observation.snapshot()
    site = "preprocess.text.prompt_embeds"
    assert snap["sites"][site][numerical_observation.KIND_SOURCE_NAN] == 1
    assert snap["sites"][site][numerical_observation.KIND_SOURCE_INF] == 1
    assert numerical_observation.KIND_CAST_OVERFLOW not in snap["sites"][site]


def test_sidecar_identity_is_recorded_without_reading_the_file(tmp_path: Path):
    from library.preprocess.text import _note_sidecar_numeric

    cache_path = tmp_path / "0001_anima_te.safetensors"
    payload = {
        "prompt_embeds": torch.tensor([[float("nan"), 1.0]], dtype=torch.float16),
        "pooled_prompt_embeds": torch.zeros(1, 2, dtype=torch.float16),
    }
    _note_sidecar_numeric(cache_path, payload)

    snap = numerical_observation.snapshot()
    identities = {
        entry["identity"]
        for per_kind in snap["last"].values()
        for entry in per_kind.values()
        if entry.get("identity")
    }
    assert str(cache_path) in identities, "来源身份必须指向具体 sidecar"
    assert "prompt_embeds" in snap["sites"] or any(
        "prompt_embeds" in site for site in snap["sites"]
    )


# ---------------------------------------------------------------------------
# R7.5 — counters are cheap, logs are throttled, no tensors are stored
# ---------------------------------------------------------------------------
def test_repeated_observations_count_but_warn_once(caplog):
    caplog.set_level(logging.WARNING, logger="library.training.numerical_observation")
    for _ in range(3):
        numerical_observation.record(
            "unit.site",
            numerical_observation.KIND_SOURCE_NAN,
            nonfinite=2,
            elements=8,
            before_dtype=None,
            after_dtype=None,
            shape=(2, 4),
            identity="unit",
        )

    site_counts = numerical_observation.snapshot()["sites"]["unit.site"]
    assert site_counts[numerical_observation.KIND_SOURCE_NAN] == 6, "计数照常累加"
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1, "R7.5：同一 (site, kind) 只告警一次"


def test_snapshot_carries_only_scalars_and_bounded_identity():
    tensor = torch.full((4, 4), float("nan"))
    numerical_observation.record(
        "unit.site",
        numerical_observation.KIND_SOURCE_NAN,
        nonfinite=16,
        elements=tensor.numel(),
        before_dtype=tensor.dtype,
        after_dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        identity="unit",
    )

    import json

    snap = numerical_observation.snapshot()
    entry = snap["last"]["unit.site"][numerical_observation.KIND_SOURCE_NAN]
    assert all(
        isinstance(value, (int, float, str, list, type(None))) for value in entry.values()
    ), f"不得存放张量：{entry}"
    json.dumps(snap), "可安全写入精度报告产物"


# ---------------------------------------------------------------------------
# M12 — no scaler, warning only
# ---------------------------------------------------------------------------
def test_fp16_without_scaler_warns_once_and_does_not_raise(caplog):
    """M12：请求 fp16 但没有 scaler ⇒ 只 warning（一次性），不 raise、不改可达性。"""
    from library.runtime import accelerator as accelerator_module

    caplog.set_level(logging.WARNING, logger=accelerator_module.logger.name)
    fake = SimpleNamespace(
        mixed_precision="no",
        scaler=None,
        device=SimpleNamespace(type="cuda"),
        process_index=0,
        num_processes=2,
        distributed_type="MULTI_GPU",
    )
    args = SimpleNamespace(mixed_precision="fp16")

    assert accelerator_module._warn_fp16_without_scaler(args, fake) is True
    assert accelerator_module._warn_fp16_without_scaler(args, fake) is False, "只 warning 一次"

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for fragment in (
        "no GradScaler",
        "requested mixed_precision=fp16",
        "resolved mixed_precision=no",
        "device.type=cuda",
        "scaler is None",
        "world_size=2",
        "Warning only",
    ):
        assert fragment in message, fragment

    # 有 scaler 或未请求 fp16 时完全静默
    caplog.clear()
    with_scaler = SimpleNamespace(
        mixed_precision="fp16", scaler=object(), device=SimpleNamespace(type="cuda"),
        process_index=0, num_processes=1, distributed_type="NO",
    )
    assert accelerator_module._warn_fp16_without_scaler(args, with_scaler) is False
    assert accelerator_module._warn_fp16_without_scaler(
        SimpleNamespace(mixed_precision="bf16"), fake
    ) is False
    assert caplog.records == []


def test_no_scaler_guard_path_is_unchanged_and_has_no_scaler_overflow_bucket():
    state = SimpleNamespace(
        accelerator=SimpleNamespace(scaler=None, optimizer_step_was_skipped=False),
        update_skipped=True,
    )
    window_coordination.ensure(state)

    cause = window_coordination.note_attempt(
        state,
        skip_update=True,          # 无 scaler 的 D2 梯度守卫丢弃
        force_skip=False,
        loss_is_finite=True,
        step_attempted=True,
    )
    snap = window_coordination.snapshot(state)

    assert cause == window_coordination.CAUSE_GRAD_GUARD
    assert snap["originating_causes"] == {window_coordination.CAUSE_GRAD_GUARD: 1}
    assert window_coordination.CAUSE_SCALER_OVERFLOW not in snap["originating_causes"], (
        "M26③：无 scaler 时该桶必须为空"
    )
    assert snap["updates_skipped_by_scaler"] == 0


def test_scaler_overflow_requires_a_live_scaler_skip():
    scaler_state = SimpleNamespace(
        accelerator=SimpleNamespace(scaler=object(), optimizer_step_was_skipped=True),
        update_skipped=True,
    )
    window_coordination.ensure(scaler_state)
    cause = window_coordination.note_attempt(
        scaler_state,
        skip_update=False,
        force_skip=False,
        loss_is_finite=True,
        step_attempted=True,
    )
    snap = window_coordination.snapshot(scaler_state)
    assert cause == window_coordination.CAUSE_SCALER_OVERFLOW
    assert snap["updates_skipped_by_scaler"] == 1
    assert snap["updates_dropped_with_scaler"] == 1

    # 同一 flag 但没有 scaler ⇒ 不得归因 scaler_overflow
    no_scaler = SimpleNamespace(
        accelerator=SimpleNamespace(scaler=None, optimizer_step_was_skipped=True),
        update_skipped=True,
    )
    window_coordination.ensure(no_scaler)
    assert (
        window_coordination.note_attempt(
            no_scaler,
            skip_update=False,
            force_skip=False,
            loss_is_finite=True,
            step_attempted=True,
        )
        is None
    )
    assert window_coordination.snapshot(no_scaler)["updates_dropped_without_scaler"] == 1


# ---------------------------------------------------------------------------
# M26⑤ — the report publishes all three layers at once
# ---------------------------------------------------------------------------
def test_precision_report_publishes_three_attribution_layers(tmp_path: Path):
    state = SimpleNamespace(
        accelerator=SimpleNamespace(scaler=None, optimizer_step_was_skipped=False),
        update_skipped=False,
        global_step=1,
    )
    window_coordination.ensure(state)
    window_coordination.note_attempt(
        state, skip_update=False, force_skip=False, loss_is_finite=True
    )
    state.update_skipped = True
    window_coordination.note_attempt(
        state, skip_update=False, force_skip=False, loss_is_finite=False
    )
    window_coordination.note_window_decision(state, "discard")
    numerical_observation.record(
        "unit.site", numerical_observation.KIND_SOURCE_INF, nonfinite=1, elements=4,
        before_dtype=None, after_dtype=None, shape=(2, 2), identity="unit",
    )

    report = PrecisionReport(SimpleNamespace(adaptive_precision="off", resume=None,
                                             auto_resume=False))
    report.transaction_accounting(state)

    updates = report.data["updates"]
    assert updates["step_attempts"] == 2
    assert updates["attribution"]["L1_originating_causes"][
        window_coordination.CAUSE_NONFINITE_LOSS
    ] == 1
    assert updates["attribution"]["L3_boundary_decisions"] == {"apply": 1, "drop": 1}
    assert updates["attribution"]["L3_window_decisions"] == {"discard": 1}
    assert "L2_propagated_drops" in updates["attribution"]
    assert updates["discard_causes"] == {}
    assert updates["legacy_keys_note"] and "commit path" in updates["scope"]
    assert report.data["nonfinite_observations"]["sites"]["unit.site"][
        numerical_observation.KIND_SOURCE_INF
    ] == 1


def test_krea2_cache_cast_records_source_nonfinite_and_keeps_the_cast_unchanged():
    """Krea-2 的 Qwen3-VL cache cast：只记源非有限，转换与字节保持不变。

    注意（实测边界）：``KREA2_CACHE_DTYPE`` 是 bfloat16，与 float32 同指数范围，因此
    "源有限但转换溢出" 在 Krea-2 这条路径上**结构性不可达**；转换溢出分类只在更窄的
    存储 dtype（如 Anima TE cache 的 float16）上出现，见 ``_report_cast_numeric`` 的用例。
    """
    from library.models.krea2_raw import strategy as krea2_strategy

    source = torch.tensor([[float("nan"), float("inf")]], dtype=torch.float32)
    stored = source.to(krea2_strategy.KREA2_CACHE_DTYPE)
    krea2_strategy._observe_cache_cast("hiddens", source, stored)

    snap = numerical_observation.snapshot()
    site = "krea2_raw.strategy.hiddens"
    assert snap["sites"][site] == {
        numerical_observation.KIND_SOURCE_NAN: 1,
        numerical_observation.KIND_SOURCE_INF: 1,
    }
    entry = snap["last"][site][numerical_observation.KIND_SOURCE_NAN]
    assert entry["identity"] == "krea2_te.hiddens"
    assert entry["after_dtype"] == str(krea2_strategy.KREA2_CACHE_DTYPE)
    assert stored.dtype == krea2_strategy.KREA2_CACHE_DTYPE, "转换本身未改"
    assert torch.isnan(stored[0, 0]) and torch.isinf(stored[0, 1]), "缓存内容未被清洗"

    # 有限源在 bfloat16 上不会溢出 ⇒ 不得凭空产生 cast_overflow
    numerical_observation.reset()
    finite = torch.tensor([[1.0, 1e30]], dtype=torch.float32)
    krea2_strategy._observe_cache_cast("hiddens", finite, finite.to(krea2_strategy.KREA2_CACHE_DTYPE))
    assert numerical_observation.snapshot()["sites"] == {}
