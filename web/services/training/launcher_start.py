"""Training start/preprocess orchestration helpers for WebUI launcher."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from library.runtime.launch import (
    ACCELERATE_MIXED_PRECISION_ENV,
    accelerate_training_command_prefix,
    configure_accelerate_for_gpu_selection,
    resolve_training_world_size_for_gpu_selection,
)
from web.services.data_cleanup_gate import guard_data_launch
from web.services.training.risk_acknowledgement import pending_preflight, without_risk_flag
from web.services.training.common import _format_ts
from web.services.training.launch_support import (
    _normalize_continue_lora_info,
)
from web.services.training.launcher_runtime import (
    _accelerate_mixed_precision_for_training,
    _apply_gpu_whitelist,
    _apply_runtime_env,
    _ensure_training_data_dirs,
    _load_config_file_config,
    _normalize_gpu_whitelist,
    _prepare_web_runtime_config,
    _resolve_training_runtime_info,
    _root,
    _runtime_from_config_file,
    _runtime_meta,
    _sample_config_from_cfg,
    preflight_training_config,
)


async def start(
    self,
    variant: str,
    preset: str,
    extra_args: list[str] | None = None,
    methods_subdir: str = "gui-methods",
    *,
    reset_logs: bool = True,
    config_file: str | None = None,
    start_message: str | None = None,
    command_label: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[Any] | None = None,
    source_config_file: str | None = None,
    use_runtime_dir: bool = True,
    queue_item_id: str = "",
):
    async with self._launch_lock:
        await self._start_unlocked(
            variant,
            preset,
            extra_args,
            methods_subdir,
            reset_logs=reset_logs,
            config_file=config_file,
            start_message=start_message,
            command_label=command_label,
            resume_info=resume_info,
            continue_info=continue_info,
            gpu_whitelist=gpu_whitelist,
            source_config_file=source_config_file,
            use_runtime_dir=use_runtime_dir,
            queue_item_id=queue_item_id,
        )

@guard_data_launch
async def _start_unlocked(
    self,
    variant: str,
    preset: str,
    extra_args: list[str] | None = None,
    methods_subdir: str = "gui-methods",
    *,
    reset_logs: bool = True,
    config_file: str | None = None,
    start_message: str | None = None,
    command_label: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[Any] | None = None,
    source_config_file: str | None = None,
    use_runtime_dir: bool = True,
    queue_item_id: str = "",
):
    self._ensure_launch_allowed(queue_item_id)

    runtime = None
    if source_config_file and use_runtime_dir:
        runtime = _prepare_web_runtime_config(
            variant,
            preset,
            methods_subdir,
            source_config_file=source_config_file,
        )
        config_file = runtime["runtime_config_file"]
    elif source_config_file and not config_file:
        config_file = source_config_file

    venv_python = str(_root() / ".venv" / "bin" / "python")
    if not Path(venv_python).exists():
        venv_python = sys.executable

    env = os.environ.copy()
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    _apply_gpu_whitelist(env, gpu_selection)
    configure_accelerate_for_gpu_selection(env, gpu_selection)
    mixed_precision = _accelerate_mixed_precision_for_training(config_file, extra_args)
    if mixed_precision:
        env[ACCELERATE_MIXED_PRECISION_ENV] = mixed_precision
    cmd = [
        *accelerate_training_command_prefix(venv_python, _root() / "train.py", env),
        "--method", variant,
        "--preset", preset,
        "--methods_subdir", methods_subdir,
    ]
    if config_file:
        cmd.extend(["--config_file", config_file])
    if extra_args:
        cmd.extend(extra_args)
    continue_payload = _normalize_continue_lora_info(
        continue_info,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    if continue_payload:
        cmd.extend([
            "--network_weights",
            continue_payload["continue_from_weight_abs_path"],
            "--dim_from_weights",
        ])

    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = str(_root() / ".venv" / "bin") + ":" + env.get("PATH", "")
    active_runtime = runtime or _runtime_from_config_file(
        config_file,
        source_config_file=source_config_file,
    )
    _apply_runtime_env(env, active_runtime)
    runtime_info = _runtime_meta(active_runtime)

    if active_runtime:
        output_dir = active_runtime["output_dir"]
        sample_dir = active_runtime["sample_dir"]
        sample_config = _sample_config_from_cfg(
            _load_config_file_config(active_runtime["runtime_config_file"]),
            extra_args or [],
        )
        data_dirs = active_runtime["data_dirs"]
    else:
        output_dir, sample_dir, sample_config = _resolve_training_runtime_info(
            variant,
            preset,
            methods_subdir,
            extra_args or [],
            config_file=config_file,
        )
        data_dirs = _ensure_training_data_dirs(
            variant,
            preset,
            methods_subdir,
            config_file=config_file,
        )
    await self._launch_job(
        cmd,
        env,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        output_dir=output_dir,
        sample_dir=sample_dir,
        data_dirs=data_dirs,
        sample_config=sample_config,
        job="training",
        start_message=start_message or f"训练启动: {methods_subdir}/{variant} / {preset}",
        command_label=command_label or "训练命令",
        reset_logs=reset_logs,
        config_file=config_file,
        resume_info=resume_info,
        continue_info=continue_payload,
        gpu_whitelist=gpu_selection,
        runtime_info=runtime_info,
        queue_item_id=queue_item_id,
    )

async def start_preprocess(
    self,
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    extra_args: list[str] | None = None,
    train_after: bool = False,
    gpu_whitelist: list[Any] | None = None,
    config_file: str | None = None,
    continue_info: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    queue_item_id: str = "",
):
    async with self._launch_lock:
        await self._start_preprocess_unlocked(
            variant,
            preset,
            methods_subdir,
            extra_args,
            train_after,
            gpu_whitelist,
            config_file,
            continue_info,
            runtime,
            queue_item_id,
        )

@guard_data_launch
async def _start_preprocess_unlocked(
    self,
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    extra_args: list[str] | None = None,
    train_after: bool = False,
    gpu_whitelist: list[Any] | None = None,
    config_file: str | None = None,
    continue_info: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    queue_item_id: str = "",
):
    self._ensure_launch_allowed(queue_item_id)

    continue_payload = _normalize_continue_lora_info(
        continue_info,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    venv_python = str(_root() / ".venv" / "bin" / "python")
    if not Path(venv_python).exists():
        venv_python = sys.executable

    runtime = runtime or _prepare_web_runtime_config(
        variant,
        preset,
        methods_subdir,
        source_config_file=config_file,
    )

    cmd = [venv_python, "tasks.py", "preprocess"]
    cmd.extend(without_risk_flag(extra_args))

    env = os.environ.copy()
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    _apply_gpu_whitelist(env, gpu_selection)
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = str(_root() / ".venv" / "bin") + ":" + env.get("PATH", "")
    env["METHOD"] = variant
    env["METHODS_SUBDIR"] = methods_subdir
    _apply_runtime_env(env, runtime)
    env["PRESET"] = preset

    output_dir = runtime["output_dir"]
    sample_dir = runtime["sample_dir"]
    sample_config = runtime["sample_config"]
    data_dirs = runtime["data_dirs"]
    runtime_info = _runtime_meta(runtime)
    self._pending_train_after_preprocess = {
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "extra_args": list(extra_args or []),
        "config_file": runtime["runtime_config_file"],
        "source_config_file": runtime.get("history_source_config_file") or config_file,
        "gpu_whitelist": gpu_selection,
        "continue_info": continue_payload,
        "queue_item_id": queue_item_id,
    } if train_after else None
    await self._launch_job(
        cmd,
        env,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        output_dir=output_dir,
        sample_dir=sample_dir,
        data_dirs=data_dirs,
        sample_config=sample_config,
        job="preprocess",
        start_message=f"预处理启动: {methods_subdir}/{variant} / {preset}",
        command_label="预处理命令",
        gpu_whitelist=gpu_selection,
        config_file=runtime["runtime_config_file"],
        runtime_info=runtime_info,
        queue_item_id=queue_item_id,
    )

async def _start_pending_training(self, pending: dict[str, Any]) -> None:
    self._remember_log("status", "预处理完成，自动开始训练")
    await self._broadcast({
        "type": "log",
        **self._remember_log("log", "[状态] 预处理完成，自动开始训练"),
    })
    try:
        preflight = preflight_training_config(
            pending["variant"],
            pending["preset"],
            pending["methods_subdir"],
            config_file=pending.get("config_file"),
            world_size=resolve_training_world_size_for_gpu_selection(
                pending.get("gpu_whitelist") or []
            ),
            # 预处理完成后的复检必须按任务实际可见的 GPU 判定架构，不能退回
            # Web 进程 device-0：异构机上两者可能不是同一张卡。
            gpu_whitelist=pending.get("gpu_whitelist"),
        )
        preflight = pending_preflight(preflight, pending.get("extra_args"))
        if not preflight.get("ok", False):
            errors = preflight.get("summary", {}).get("errors", 0)
            details = []
            for item in preflight.get("errors") or []:
                if not isinstance(item, dict):
                    continue
                msg = str(item.get("message") or "").strip()
                if not msg:
                    continue
                path = str(item.get("path") or "").strip()
                details.append(f"{msg} ({path})" if path else msg)
            detail_text = "；".join(details[:3])
            if detail_text:
                raise RuntimeError(f"预处理后仍有 {errors} 个预检测错误：{detail_text}")
            raise RuntimeError(f"预处理后仍有 {errors} 个预检测错误")
        await self.start(
            pending["variant"],
            pending["preset"],
            pending.get("extra_args") or [],
            pending["methods_subdir"],
            reset_logs=False,
            config_file=pending.get("config_file"),
            source_config_file=pending.get("source_config_file"),
            gpu_whitelist=pending.get("gpu_whitelist"),
            continue_info=pending.get("continue_info"),
            use_runtime_dir=False,
            queue_item_id=pending.get("queue_item_id") or "",
        )
    except Exception as e:
        msg = f"自动开始训练失败: {e}"
        queue_item_id = str(pending.get("queue_item_id") or "")
        if queue_item_id:
            self._pause_queue_after_failure()
            self._update_queue_item(queue_item_id, {
                "state": "error",
                "message": msg,
                "finished_at": time.time(),
                "finished_at_text": _format_ts(time.time()),
            })
            self._save_queue()
            await self._broadcast_queue()
        self._remember_log("status", msg)
        await self._broadcast({
            "type": "status",
            "state": "error",
            "job": "training",
            "message": msg,
            "output_dir": self.current_output_dir,
            "sample_dir": self.current_sample_dir,
            "sample_config": self.current_sample_config,
            **self.current_runtime_info,
            "task_id": self.current_task_id,
            "queue_item_id": queue_item_id,
        })
        self._schedule_queue_dispatch()
