"""Training job lifecycle helpers for WebUI launcher."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

import psutil

from web.services.training.common import _format_ts
from web.services.training.launch_support import (
    _command_has_option,
    _command_option_value,
    _resolve_block_swap_profile_auto_arg,
    _resolve_block_swap_profile_auto_config,
    _resolve_memory_probe_auto_arg,
    _resolve_memory_probe_auto_config,
    _resolve_peak_probe_auto_arg,
    _resolve_peak_probe_auto_config,
)
from web.services.training.launcher_runtime import (
    _resolve_display_path,
    _root,
    _runtime_meta,
)
from web.services.training.task_lifecycle import cancel_run_tasks, track_run_task

STOP_OUTPUT_TIMEOUT_SECONDS = 10.0


async def _launch_job(
    self,
    cmd: list[str],
    env: dict[str, str],
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    output_dir: str,
    sample_dir: str,
    data_dirs: dict[str, str],
    sample_config: dict[str, Any],
    job: str,
    start_message: str,
    command_label: str,
    reset_logs: bool = True,
    config_file: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[int] | None = None,
    runtime_info: dict[str, str] | None = None,
    queue_item_id: str = "",
    distributed_mode: str | None = None,
):
    self._run_generation += 1
    generation = self._run_generation
    self._stopping = False
    self.status = "running"
    self._current_queue_item_id = str(queue_item_id or "")
    self.current_job = job
    self.current_variant = variant
    self.current_preset = preset
    self.current_methods_subdir = methods_subdir
    self.current_output_dir = output_dir
    self.current_sample_dir = sample_dir
    self.current_sample_config = sample_config
    self.current_runtime_info = _runtime_meta(runtime_info)
    self.current_gpu_whitelist = list(gpu_whitelist or [])
    self._anchor = None
    self._reset_progress_rate_state()
    self._reset_metric_runtime_state()
    self._latest_progress = None
    self._latest_system_stats = None
    self._progress_jsonl_path = None
    self._progress_jsonl_offset = 0
    self._progress_jsonl_seen = set()
    self._progress_jsonl_lock = asyncio.Lock()
    self._progress_total_steps = None
    self._detected_error_hint = ""
    self._stop_requested = False
    self.current_task_id = ""
    self.current_task_dir = None
    self._current_history_log_count = 0
    if job != "preprocess":
        self._pending_train_after_preprocess = None
    self._last_output_at = time.time()
    self._last_log_line = ""
    if reset_logs:
        self._log_records.clear()
        self._next_log_id = 1

    task_dir = self._reserve_history_task_dir(job, methods_subdir, variant)
    if job == "training" and not _command_has_option(cmd, "--progress_jsonl"):
        cmd = [*cmd, "--progress_jsonl", str(task_dir / "progress.jsonl")]
    if job == "training":
        block_swap_profile_path = task_dir / "block_swap_profile.jsonl"
        config_wants_block_swap_profile = _resolve_block_swap_profile_auto_config(
            config_file, block_swap_profile_path
        )
        cmd = _resolve_block_swap_profile_auto_arg(cmd, block_swap_profile_path)
        if (
            config_wants_block_swap_profile
            and not _command_has_option(cmd, "--block_swap_profile_jsonl")
        ):
            cmd = [*cmd, "--block_swap_profile_jsonl", str(block_swap_profile_path)]
        memory_probe_path = task_dir / "memory_probe.jsonl"
        config_wants_memory_probe = _resolve_memory_probe_auto_config(
            config_file, memory_probe_path
        )
        cmd = _resolve_memory_probe_auto_arg(cmd, memory_probe_path)
        if (
            config_wants_memory_probe
            and not _command_has_option(cmd, "--memory_probe_jsonl")
        ):
            cmd = [*cmd, "--memory_probe_jsonl", str(memory_probe_path)]
        peak_probe_path = task_dir / "peak_probe.jsonl"
        config_wants_peak_probe = _resolve_peak_probe_auto_config(
            config_file, peak_probe_path
        )
        cmd = _resolve_peak_probe_auto_arg(cmd, peak_probe_path)
        if (
            config_wants_peak_probe
            and not _command_has_option(cmd, "--peak_probe_jsonl")
        ):
            cmd = [*cmd, "--peak_probe_jsonl", str(peak_probe_path)]
    if job == "training":
        progress_jsonl = _command_option_value(cmd, "--progress_jsonl")
        self._progress_jsonl_path = _resolve_display_path(progress_jsonl or str(task_dir / "progress.jsonl"))
    self.current_command = cmd
    self._start_history_task(
        job=job,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        output_dir=output_dir,
        sample_dir=sample_dir,
        data_dirs=data_dirs,
        sample_config=sample_config,
        command=cmd,
        config_file=config_file,
        resume_info=resume_info,
        continue_info=continue_info,
        gpu_whitelist=gpu_whitelist,
        runtime_info=self.current_runtime_info,
        queue_info=self._queue_history_meta(self._current_queue_item_id),
        distributed_mode=distributed_mode,
    )
    try:
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(_root()),
            start_new_session=True,
        )
    except Exception as e:
        self.status = "idle"
        self._finish_history_task(state="error", message=f"任务启动失败: {e}", returncode=-1)
        raise
    self._remember_log("status", f"{command_label}: {' '.join(cmd)}")
    if gpu_whitelist:
        self._remember_log("status", f"GPU 白名单: {','.join(str(item) for item in gpu_whitelist)}")

    await self._broadcast({
        "type": "status",
        "state": "running",
        "job": job,
        "message": start_message,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "output_dir": self.current_output_dir,
        "sample_dir": self.current_sample_dir,
        "sample_config": self.current_sample_config,
        **self.current_runtime_info,
        "task_id": self.current_task_id,
        "queue_item_id": self._current_queue_item_id,
    })
    if self._current_queue_item_id:
        self._attach_history_task_to_queue_item(self._current_queue_item_id, self.current_task_id)
        await self._broadcast_queue()
    process = self.process
    track_run_task(
        self,
        self._read_output(process=process, generation=generation),
        generation=generation,
        output=True,
    )
    track_run_task(
        self,
        self._monitor_system(
            generation=generation,
            gpu_whitelist=tuple(self.current_gpu_whitelist),
        ),
        generation=generation,
    )
    if self._progress_jsonl_path:
        track_run_task(
            self,
            self._tail_progress_jsonl(generation=generation),
            generation=generation,
        )

async def stop(self, expected_task_id: str | None = None):
    async with self._launch_lock:
        current_task_id = getattr(self, "current_task_id", None)
        if expected_task_id and current_task_id != expected_task_id:
            raise RuntimeError("当前训练任务已发生变化，请刷新后重试")
        await _stop_unlocked(self)


async def _stop_unlocked(self):
    process = self.process
    generation = self._run_generation
    active_run_tasks = any(
        not task.done()
        for task in self._job_tasks.get(generation, ())
    )
    output_task = (
        self._output_task
        if self._output_task_generation == generation
        else None
    )
    if (
        (process is None or process.returncode is not None)
        and (output_task is None or output_task.done())
        and not active_run_tasks
    ):
        self.status = "idle"
        self._stopping = False
        self._stop_requested = False
        self._pending_train_after_preprocess = None
        return

    self._stopping = True
    self._stop_requested = True
    self._pending_train_after_preprocess = None
    queue_item_id = self._current_queue_item_id
    job = self.current_job
    if queue_item_id:
        self._queue_paused = True
        self._queue["paused"] = True
        self._update_queue_item(queue_item_id, {
            "state": "canceled",
            "message": "用户停止了队列任务，队列已自动暂停",
            "finished_at": time.time(),
            "finished_at_text": _format_ts(time.time()),
        })
        self._save_queue()
    if process is not None and process.returncode is None:
        try:
            pid = process.pid
            parent = psutil.Process(pid)
            family = [parent] + parent.children(recursive=True)
            for p in family:
                try:
                    p.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            _, alive = psutil.wait_procs(family, timeout=3.0)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if output_task is not None and not output_task.done():
        try:
            await asyncio.wait_for(
                asyncio.shield(output_task),
                timeout=STOP_OUTPUT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self._remember_log("error", "等待训练输出任务退出超时，已强制清理后台任务")
            await cancel_run_tasks(self, generation)
        except Exception as exc:
            self._remember_log("error", f"等待训练输出任务退出失败: {exc}")

    if generation != self._run_generation:
        return

    await cancel_run_tasks(self, generation)
    self.status = "idle"
    self._stopping = False
    message = "预处理已停止" if job == "preprocess" else "训练已停止"
    await self._broadcast({
        "type": "status",
        "state": "idle",
        "job": job,
        "message": message,
        "output_dir": self.current_output_dir,
        "sample_dir": self.current_sample_dir,
        "sample_config": self.current_sample_config,
        **self.current_runtime_info,
        "task_id": self.current_task_id,
        "queue_item_id": queue_item_id,
    })
    if queue_item_id:
        await self._broadcast_queue()


async def shutdown(self) -> None:
    self._shutting_down = True
    wake_handle = self._queue_dispatch_wake_handle
    if wake_handle is not None:
        wake_handle.cancel()
        self._queue_dispatch_wake_handle = None
    dispatch_task = self._queue_dispatch_task
    if dispatch_task is not None and dispatch_task is not asyncio.current_task():
        if not dispatch_task.done():
            dispatch_task.cancel()
        await asyncio.gather(dispatch_task, return_exceptions=True)
        if self._queue_dispatch_task is dispatch_task:
            self._queue_dispatch_task = None
    self._queue_launching_item_id = ""
    async with self._launch_lock:
        await _stop_unlocked(self)
        await cancel_run_tasks(self)

def _ensure_launch_allowed(self, queue_item_id: str = "") -> None:
    from web.services.data_cleanup_gate import is_data_cleanup_active

    launching = self._queue_launching_item_id
    same_queue_item = launching and str(queue_item_id or "") == launching
    if (
        self.status == "running"
        or self._stopping
        or self._shutting_down
        or is_data_cleanup_active()
        or (launching and not same_queue_item)
    ):
        raise RuntimeError("已有任务在运行中" if not is_data_cleanup_active() else "数据清理正在进行中")

def _write_terminal(self, text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        pass
