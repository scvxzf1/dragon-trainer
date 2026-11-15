"""Pipeline-parallel compatibility gate shared by resume entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import toml

from library.env import anima_home
from library.models.pipeline_parallel import PipelineParallelConfig
from library.runtime.launch import resolve_training_world_size_for_gpu_selection
from library.training.compat_matrix import check_training_compat
from web.services.training.gpu import (
    ensure_gpu_whitelist_authorized,
    normalize_gpu_whitelist,
)


def ensure_pipeline_parallel_family_pairing(config: dict[str, Any]) -> None:
    """``pipeline_parallel`` and ``model_family`` must travel as a pair.

    A queue/runtime config that turns on ``pipeline_parallel`` without naming
    ``model_family`` would let the launch silently resolve a different family
    (default ``anima``) than the one the WebUI preflight inspected.  Normal
    non-PP runs are untouched: neither key is required when PP is off.
    """

    try:
        pipeline_config = PipelineParallelConfig.from_config(config)
    except ValueError as exc:
        raise ValueError(f"流水线配置无效: {exc}") from exc
    if not pipeline_config.enabled:
        return
    family = str(config.get("model_family") or "").strip()
    if not family:
        raise ValueError(
            "流水线并行训练配置必须同时携带 model_family 与 pipeline_parallel "
            "（不允许只传其一）: pipeline_parallel=true 缺少 model_family，"
            "启动将无法保证与预检相同的模型族"
        )


def ensure_resume_pipeline_compatible(
    config_file: str,
    gpu_whitelist: list[Any] | None,
) -> None:
    """Reject PP resume requests before they launch or enter the queue."""

    expanded = Path(os.path.expandvars(config_file)).expanduser()
    config_path = expanded if expanded.is_absolute() else anima_home() / expanded
    try:
        config = toml.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError) as exc:
        raise ValueError(
            f"续训流水线配置无法读取或解析: {config_file}: {exc}"
        ) from exc

    ensure_pipeline_parallel_family_pairing(config)

    gpu_selection = normalize_gpu_whitelist(gpu_whitelist)
    # UUID-level guard runs for every resume (PP or not): the whitelist is
    # index-based under PCI_BUS_ID order, so a stale index can resolve to the
    # GTX 1050 on either path.
    ensure_gpu_whitelist_authorized(gpu_selection)

    try:
        pipeline_config = PipelineParallelConfig.from_config(config)
    except ValueError as exc:
        raise ValueError(f"续训流水线预检测失败: {exc}") from exc
    if not pipeline_config.enabled:
        return

    world_size = resolve_training_world_size_for_gpu_selection(gpu_selection)
    compatibility = check_training_compat(config, world_size=world_size)
    errors = [
        item.message
        for item in compatibility.errors
        if item.key in {"model_family", "pipeline_parallel"}
    ]
    if errors:
        raise ValueError("续训流水线预检测失败: " + "; ".join(errors))
