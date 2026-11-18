"""Anima WebUI/config PP wiring tests (t5, CPU service layer only).

Contract §5 (_archive/docs/proposal/anima_pp_trainer_wiring.md):

* the WebUI preflight shows the wired Anima PP trainer session (no launch
  block), and surfaces the trainer-admission knob view (e.g. a preview field
  under PP) as an error carrying the exact ``pp_*`` code;
* ``pipeline_parallel`` and ``model_family`` must travel **as a pair** into a
  queue/runtime config — the launcher and the resume gate both refuse a
  config that enables PP without naming the family; non-PP runs are untouched;
* the stale family doc / compat wording no longer claims Anima PP is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.web_config_test_support import _write_selected_checkpoint_preflight_config
from web.services import config_service
from web.services.training.resume_pipeline_gate import (
    ensure_pipeline_parallel_family_pairing,
    ensure_resume_pipeline_compatible,
)


def _preflight(*, world_size: int | None = None) -> dict:
    return config_service.preflight_training_config(
        "lora", "default", "imported",
        config_file="configs/imported/selected.toml", world_size=world_size,
    )


def _messages(result: dict, key: str) -> list[str]:
    return [item["message"] for item in result["errors"] if item["key"] == key]


def _warnings(result: dict, key: str) -> list[str]:
    return [item["message"] for item in result["warnings"] if item["key"] == key]


_PP_LINES = [
    "pipeline_parallel = true",
    'mixed_precision = "fp16"',
    "pipeline_parallel_stages = 2",
    "pipeline_parallel_microbatches = 4",
    'pipeline_parallel_schedule = "1f1b"',
    'pipeline_parallel_split = "balanced"',
    "network_train_unet_only = true",
    "torch_compile = false",
    "blocks_to_swap = 0",
    'selective_checkpoint = "off"',
]


# ---------------------------------------------------------------------------
# WebUI preflight: wired session view + trainer-admission knob surfacing
# ---------------------------------------------------------------------------


def test_preflight_surfaces_the_wired_anima_pp_session(tmp_path: Path, monkeypatch) -> None:
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, _PP_LINES)
    result = _preflight()
    assert result["ok"] is True
    assert _messages(result, "pipeline_parallel") == []
    warnings = _warnings(result, "pipeline_parallel")
    assert any("专用 PP 训练会话" in message for message in warnings)


def test_preflight_surfaces_an_anima_admission_refusal_before_launch(tmp_path: Path, monkeypatch) -> None:
    """A preview field under Anima PP is shown as a named launch refusal."""

    _write_selected_checkpoint_preflight_config(
        tmp_path, monkeypatch, [*_PP_LINES, 'sample_prompts = "prompts.txt"']
    )
    result = _preflight()
    assert result["ok"] is False
    errors = _messages(result, "pipeline_parallel")
    assert any("pp_config_invalid" in message for message in errors), errors
    assert any("sample_prompts" in message or "preview" in message.lower() for message in errors)


# ---------------------------------------------------------------------------
# pipeline_parallel / model_family pairing guard
# ---------------------------------------------------------------------------


def test_pairing_guard_requires_model_family_when_pp_enabled():
    with pytest.raises(ValueError) as excinfo:
        ensure_pipeline_parallel_family_pairing({"pipeline_parallel": True})
    assert "model_family" in str(excinfo.value)


def test_pairing_guard_accepts_a_paired_pp_config():
    ensure_pipeline_parallel_family_pairing(
        {"pipeline_parallel": True, "model_family": "anima"}
    )
    # PP off (or absent): neither key is required.
    ensure_pipeline_parallel_family_pairing({})
    ensure_pipeline_parallel_family_pairing({"pipeline_parallel": False})


def test_pairing_guard_keeps_a_broken_pp_config_a_named_error():
    with pytest.raises(ValueError) as excinfo:
        ensure_pipeline_parallel_family_pairing({"pipeline_parallel": "maybe"})
    assert "流水线配置无效" in str(excinfo.value)


def test_resume_gate_refuses_an_unpaired_pp_config(tmp_path: Path) -> None:
    config_file = tmp_path / "pp-resume.toml"
    config_file.write_text("pipeline_parallel = true\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        ensure_resume_pipeline_compatible(str(config_file), gpu_whitelist=None)
    assert "model_family" in str(excinfo.value)


def test_resume_gate_passes_a_paired_non_pipeline_config(tmp_path: Path) -> None:
    config_file = tmp_path / "plain-resume.toml"
    config_file.write_text("model_family = \"anima\"\n", encoding="utf-8")
    # Non-PP resume: the gate returns without any PP evaluation.
    ensure_resume_pipeline_compatible(str(config_file), gpu_whitelist=None)


# ---------------------------------------------------------------------------
# launcher preconditions: the training start path runs the same guard
# ---------------------------------------------------------------------------


def test_launcher_start_module_imports_the_pairing_guard():
    import inspect

    from web.services.training import launcher_start

    source = inspect.getsource(launcher_start)
    assert "ensure_pipeline_parallel_family_pairing" in source
    assert launcher_start.ensure_pipeline_parallel_family_pairing is (
        ensure_pipeline_parallel_family_pairing
    )


# ---------------------------------------------------------------------------
# t9: UUID-level gpu_whitelist guard through the resume gate
# ---------------------------------------------------------------------------


def test_resume_gate_allows_a_whitelist_that_resolves_to_the_gtx_1050(
    tmp_path: Path, monkeypatch
) -> None:
    # A/B: t9 时代该用例断言 1050 白名单被拒绝；全卡开放后 1050 可选且为
    # 建议级警告语义（诊断端点承载），续训门不再 raise。
    from web.services.training import gpu as gpu_helpers

    rows = "\n".join(
        [
            "0, GPU-410a51d5-c1e2-6616-b56d-ba2e3ee2aa1b, NVIDIA GeForce GTX 1050",
            "1, GPU-b4b014b2-953b-28e8-b8c1-425c4a401165, Tesla V100-SXM2-16GB",
        ]
    )
    monkeypatch.setattr(gpu_helpers, "_query_nvidia_smi", lambda *a, **k: rows)
    config_file = tmp_path / "plain-resume.toml"
    config_file.write_text("model_family = \"anima\"\n", encoding="utf-8")
    ensure_resume_pipeline_compatible(str(config_file), gpu_whitelist=[0])
