from __future__ import annotations

from pathlib import Path

import pytest

from tests.web_config_test_support import _write_selected_checkpoint_preflight_config
from web.services import config_service


def _preflight(*, world_size: int | None = None) -> dict:
    return config_service.preflight_training_config(
        "lora", "default", "imported",
        config_file="configs/imported/selected.toml", world_size=world_size,
    )


def _messages(result: dict, key: str) -> list[str]:
    return [item["message"] for item in result["errors"] if item["key"] == key]


def test_preflight_releases_anima_pipeline_and_warns_about_the_trainer_gate(tmp_path: Path, monkeypatch) -> None:
    """Anima PP v0 has a runtime, so the compat layer stops rejecting it.

    The trainer entry still refuses the run (``pp_trainer_path_unavailable``),
    which the preflight surfaces as a warning so the decision is visible before
    the launch instead of only in the worker log.
    """

    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, ["pipeline_parallel = true"])
    result = _preflight()
    assert result["ok"] is True
    assert _messages(result, "pipeline_parallel") == []
    warnings = [
        item["message"] for item in result["warnings"] if item["key"] == "pipeline_parallel"
    ]
    assert any("pp_trainer_path_unavailable" in message for message in warnings)
    assert not any("流水线配置无效" in message for message in warnings)


def test_preflight_keeps_valid_krea_pipeline_launch_blocked(tmp_path: Path, monkeypatch) -> None:
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, [
        'model_family = "krea2_raw"', "pipeline_parallel = true",
        "pipeline_parallel_stages = 2", "pipeline_parallel_microbatches = 4",
        'pipeline_parallel_schedule = "1f1b"', 'pipeline_parallel_split = "balanced"',
        "torch_compile = false", "blocks_to_swap = 0", 'selective_checkpoint = "off"',
    ])
    result = _preflight()
    messages = _messages(result, "pipeline_parallel")
    assert result["ok"] is False
    assert any("主训练 loop 的 1F1B 调度尚未接入" in message for message in messages)
    assert not any("流水线配置无效" in message for message in messages)


@pytest.mark.parametrize("world_size", [1, 3])
def test_preflight_rejects_pipeline_when_selected_gpu_count_is_not_two(tmp_path: Path, monkeypatch, world_size: int) -> None:
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, [
        'model_family = "krea2_raw"', "pipeline_parallel = true",
        "pipeline_parallel_stages = 2", "torch_compile = false", "blocks_to_swap = 0",
        'selective_checkpoint = "off"',
    ])
    result = _preflight(world_size=world_size)
    messages = _messages(result, "pipeline_parallel")
    assert result["ok"] is False
    assert any("流水线配置无效" in message for message in messages)
    assert not any("主训练 loop 的 1F1B 调度尚未接入" in message for message in messages)


def test_preflight_reports_invalid_pipeline_config_before_runtime_gate(tmp_path: Path, monkeypatch) -> None:
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, [
        'model_family = "krea2_raw"', "pipeline_parallel = true", "torch_compile = true",
        "blocks_to_swap = 0", 'selective_checkpoint = "off"',
    ])
    result = _preflight()
    messages = _messages(result, "pipeline_parallel")
    assert result["ok"] is False
    assert any("流水线配置无效" in message for message in messages)
    assert not any("主训练 loop 的 1F1B 调度尚未接入" in message for message in messages)
