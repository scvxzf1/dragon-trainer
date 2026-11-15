"""Queue enqueue helpers delegated from ``TrainingService``."""

from __future__ import annotations

import time
from typing import Any

from web.services.training.anomalies import classify_training_failure, should_auto_retry_failure
from web.services.training.service_state import (
    resolve_item_retry_policy,
    _normalize_queue_auto_retry,
    _normalize_queue_max_attempts,
    _normalize_queue_retry_backoff,
)
from web.services.training.common import _format_ts, _int_or_none
from web.services.training.distributed import normalize_distributed_mode
from web.services.training.gpu import (
    ensure_gpu_whitelist_authorized,
    normalize_gpu_whitelist as _normalize_gpu_whitelist,
)
from web.services.training.launch_support import _normalize_continue_lora_info
from web.services.training.resume_pipeline_gate import (
    ensure_resume_pipeline_compatible,
)
from web.services.training.service_state import _new_queue_item_id


def _prepare_web_runtime_config(*args, **kwargs):
    from web.services import training_service as facade

    return facade._prepare_web_runtime_config(*args, **kwargs)


def _clone_frozen_runtime_config(*args, **kwargs):
    from web.services import training_service as facade

    return facade._clone_frozen_runtime_config(*args, **kwargs)


def _runtime_meta(*args, **kwargs):
    from web.services import training_service as facade

    return facade._runtime_meta(*args, **kwargs)


def _display_project_path(*args, **kwargs):
    from web.services import training_service as facade

    return facade._display_project_path(*args, **kwargs)

async def enqueue_training(
    self,
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    extra_args: list[str] | None = None,
    config_file: str | None = None,
    gpu_whitelist: list[Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    requires_preprocess: bool = True,
    start_paused: bool = False,
    auto_retry: bool | None = None,
    max_attempts: int | None = None,
    retry_backoff_sec: float | None = None,
    distributed_mode: Any = None,
) -> dict[str, Any]:
    extra = list(extra_args or [])
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    ensure_gpu_whitelist_authorized(gpu_selection)
    runtime = None
    runtime_config_file = str(config_file or "").strip()
    source_config_file = str(config_file or "").strip()
    if requires_preprocess:
        runtime = _prepare_web_runtime_config(
            variant,
            preset,
            methods_subdir,
            source_config_file=config_file,
        )
        runtime_config_file = runtime["runtime_config_file"]
        source_config_file = runtime.get("history_source_config_file") or source_config_file

    continue_payload = _normalize_continue_lora_info(
        continue_info,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=runtime_config_file or config_file,
    )
    now = time.time()
    item = {
        "id": _new_queue_item_id("training", methods_subdir, variant),
        "state": "queued",
        "kind": "training",
        "requires_preprocess": bool(requires_preprocess),
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "runtime_config_file": runtime_config_file,
        "source_config_file": source_config_file,
        "extra_args": extra,
        "gpu_whitelist": gpu_selection,
        # 分布式模式三态卡片：旧调用面/旧队列缺省 DDP 兼容。
        "distributed_mode": normalize_distributed_mode(distributed_mode),
        "continue_info": continue_payload or {},
        "resume_info": {},
        "retry_of": "",
        "attempt": 1,
        "history_task_ids": [],
        "message": "等待队列调度",
        "created_at": now,
        "created_at_text": _format_ts(now),
        "started_at": None,
        "started_at_text": "",
        "finished_at": None,
        "finished_at_text": "",
        "runtime_info": _runtime_meta(runtime) if runtime else {},
    }
    if auto_retry is not None:
        item["auto_retry"] = _normalize_queue_auto_retry(auto_retry)
    if max_attempts is not None:
        item["max_attempts"] = _normalize_queue_max_attempts(max_attempts)
    if retry_backoff_sec is not None:
        item["retry_backoff_sec"] = _normalize_queue_retry_backoff(retry_backoff_sec)
    if start_paused:
        self._queue_paused = True
        self._queue["paused"] = True
    self._queue_items().append(item)
    self._compact_queue()
    self._save_queue()
    await self._broadcast_queue()
    if not self._queue_paused:
        self._schedule_queue_dispatch()
    message = "已加入训练队列，队列已暂停" if start_paused else "已加入训练队列"
    return {"ok": True, "message": message, "item": dict(item), **self.get_queue_snapshot()}


async def enqueue_training_batch(
    self,
    entries: list[dict[str, Any]],
    *,
    default_preset: str = "default",
    gpu_whitelist: list[Any] | None = None,
    start_paused: bool = True,
) -> dict[str, Any]:
    if not isinstance(entries, list) or not entries:
        raise ValueError("items 必须是非空数组")

    queued: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    ensure_gpu_whitelist_authorized(gpu_selection)
    if start_paused:
        self._queue_paused = True
        self._queue["paused"] = True

    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            failures.append({
                "index": index,
                "error": "队列项格式不合法",
            })
            break
        variant = str(raw.get("variant") or "").strip()
        preset = str(raw.get("preset") or default_preset or "default").strip() or "default"
        methods_subdir = str(raw.get("methods_subdir") or "gui-methods").strip() or "gui-methods"
        config_file = str(raw.get("config_file") or "").strip() or None
        label = str(raw.get("label") or raw.get("filename") or raw.get("display_name") or "").strip()
        requires_preprocess = bool(raw.get("requires_preprocess", True))
        if not variant:
            failure = {
                "index": index,
                "config_file": config_file or "",
                "error": "缺少 variant",
            }
            if label:
                failure["label"] = label
            failures.append(failure)
            break
        try:
            payload = await self.enqueue_training(
                variant,
                preset,
                methods_subdir,
                extra_args=list(raw.get("extra_args") or []),
                config_file=config_file,
                gpu_whitelist=gpu_selection,
                continue_info=raw.get("continue_info") if isinstance(raw.get("continue_info"), dict) else None,
                requires_preprocess=requires_preprocess,
                start_paused=start_paused,
                auto_retry=raw.get("auto_retry") if "auto_retry" in raw else None,
                max_attempts=raw.get("max_attempts") if "max_attempts" in raw else None,
                retry_backoff_sec=raw.get("retry_backoff_sec") if "retry_backoff_sec" in raw else None,
                distributed_mode=raw.get("distributed_mode"),
            )
        except Exception as exc:
            failure = {
                "index": index,
                "variant": variant,
                "preset": preset,
                "methods_subdir": methods_subdir,
                "config_file": config_file or "",
                "error": str(exc),
            }
            if label:
                failure["label"] = label
            failures.append(failure)
            break
        queued.append(dict(payload.get("item") or {}))

    snapshot = self.get_queue_snapshot()
    queued_count = len(queued)
    total = len(entries)
    ok = not failures
    if ok:
        message = f"已将 {queued_count} 个配置加入训练队列"
    elif queued_count:
        message = f"已加入 {queued_count} 个配置，批量加入在第 {failures[0]['index'] + 1} 项停止"
    else:
        message = "批量加入队列失败"
    return {
        **snapshot,
        "ok": ok,
        "message": message,
        "queued_count": queued_count,
        "requested_count": total,
        "queued_items": queued,
        "failures": failures,
    }


async def enqueue_resume_from_history_task(
    self,
    task_id: str,
    checkpoint: str | None = None,
    *,
    duration_overrides: dict[str, Any] | None = None,
    gpu_whitelist: list[Any] | None = None,
) -> dict[str, Any]:
    task, selected, snapshot_path, resume_info = self._build_resume_payload(
        task_id,
        checkpoint,
        duration_overrides=duration_overrides,
    )
    snapshot_config_file = _display_project_path(str(snapshot_path))
    ensure_resume_pipeline_compatible(snapshot_config_file, gpu_whitelist)
    runtime = _clone_frozen_runtime_config(
        snapshot_config_file,
        source_config_file=str(task.get("history_source_config_file") or ""),
        reset_data_dirs=False,
        resume_step=_int_or_none(selected.get("step")),
        duration_overrides=duration_overrides,
    )
    if isinstance(runtime.get("resume_duration"), dict) and runtime["resume_duration"]:
        resume_info["duration_overrides"] = runtime["resume_duration"]
        resume_info["target_total_steps"] = runtime["resume_duration"].get("target_total_steps")
        resume_info["remaining_steps"] = runtime["resume_duration"].get("append_steps")
        for key in ("stage_before", "stage_after", "warning"):
            if key in runtime["resume_duration"] and runtime["resume_duration"].get(key) is not None:
                resume_info[key] = runtime["resume_duration"][key]
    config_file = str(
        runtime.get("runtime_config_file")
        or snapshot_config_file
    )
    now = time.time()
    item = {
        "id": _new_queue_item_id(
            "resume",
            str(task.get("methods_subdir") or "gui-methods"),
            str(task.get("variant") or "training"),
        ),
        "state": "queued",
        "kind": "resume",
        "requires_preprocess": False,
        "variant": str(task.get("variant") or ""),
        "preset": str(task.get("preset") or "default"),
        "methods_subdir": str(task.get("methods_subdir") or "gui-methods"),
        "runtime_config_file": config_file,
        "source_config_file": str(runtime.get("history_source_config_file") or task.get("history_source_config_file") or ""),
        "extra_args": ["--resume", selected["path"], "--skip_until_initial_step"],
        "gpu_whitelist": _normalize_gpu_whitelist(gpu_whitelist),
        # 旧队列 resume 无模式字段缺省 DDP；PP 快照仍由 pipeline_parallel 键决定。
        "distributed_mode": normalize_distributed_mode(
            "pp" if bool(runtime.get("pipeline_parallel")) else task.get("distributed_mode"),
        ),
        "continue_info": {},
        "resume_info": resume_info,
        "retry_of": "",
        "attempt": 1,
        "history_task_ids": [],
        "message": "等待续训队列调度",
        "created_at": now,
        "created_at_text": _format_ts(now),
        "started_at": None,
        "started_at_text": "",
        "finished_at": None,
        "finished_at_text": "",
        "runtime_info": _runtime_meta(runtime),
    }
    self._queue_items().append(item)
    self._compact_queue()
    self._save_queue()
    await self._broadcast_queue()
    self._schedule_queue_dispatch()
    return {
        "ok": True,
        "message": "续训任务已加入队列",
        "item": dict(item),
        "checkpoint": selected,
        **self.get_queue_snapshot(),
    }


def _clone_queue_item_for_retry(self, item: dict[str, Any]) -> dict[str, Any]:
    runtime = _clone_frozen_runtime_config(
        str(item.get("runtime_config_file") or ""),
        source_config_file=str(item.get("source_config_file") or ""),
        reset_data_dirs=bool(item.get("requires_preprocess")),
    )
    now = time.time()
    retry_of = str(item.get("retry_of") or item.get("id") or "")
    attempt = int(item.get("attempt") or 1) + 1
    retry = {
        key: value for key, value in item.items()
        if key not in {
            "id", "state", "message", "created_at", "created_at_text",
            "started_at", "started_at_text", "finished_at", "finished_at_text",
            "history_task_ids", "runtime_config_file", "runtime_info", "next_run_at",
        }
    }
    retry.update({
        "id": _new_queue_item_id(
            str(item.get("kind") or "retry"),
            str(item.get("methods_subdir") or "gui-methods"),
            str(item.get("variant") or "training"),
        ),
        "state": "queued",
        "runtime_config_file": runtime["runtime_config_file"],
        "source_config_file": str(item.get("source_config_file") or runtime.get("history_source_config_file") or ""),
        "retry_of": retry_of,
        "attempt": attempt,
        "history_task_ids": [],
        "message": f"第 {attempt} 次尝试，等待队列调度",
        "created_at": now,
        "created_at_text": _format_ts(now),
        "started_at": None,
        "started_at_text": "",
        "finished_at": None,
        "finished_at_text": "",
        "runtime_info": _runtime_meta(runtime),
    })
    return retry




def _maybe_auto_retry(
    self,
    item: dict[str, Any] | None,
    *,
    reason: str = "",
    message: str = "",
    stop_requested: bool = False,
) -> dict[str, Any] | None:
    """Clone a failed queue item when auto_retry policy allows.

    Shared by process-exit failures and launch failures so attempt counting,
    max_attempts, and next_run_at backoff stay consistent.
    """
    if item is None:
        return None
    policy = resolve_item_retry_policy(
        item,
        queue_auto_retry=bool(getattr(self, "_queue_auto_retry", False)),
        queue_max_attempts=int(getattr(self, "_queue_max_attempts", 1) or 1),
        queue_retry_backoff_sec=float(getattr(self, "_queue_retry_backoff_sec", 0.0) or 0.0),
    )
    if not bool(policy.get("auto_retry")):
        return None
    attempt = int(item.get("attempt") or 1)
    max_attempts = int(policy.get("max_attempts") or 1)
    if attempt >= max_attempts:
        return None
    kind = classify_training_failure(
        reason=reason,
        message=str(message or item.get("message") or ""),
        stop_requested=bool(stop_requested),
    )
    if not should_auto_retry_failure(kind):
        return None
    retry = self._clone_queue_item_for_retry(item)
    retry["failure_class"] = kind
    backoff = float(policy.get("retry_backoff_sec") or 0.0)
    if backoff > 0:
        retry["next_run_at"] = time.time() + backoff
        retry["message"] = (
            f"第 {retry.get('attempt')} 次尝试，"
            f"{int(backoff)}s 后自动重试"
        )
    if reason:
        retry.setdefault("retry_reason", str(reason))
    self._queue_items().append(retry)
    self._compact_queue()
    return retry
