"""History list/update/delete/lifecycle helpers for WebUI training."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from web.services.training.common import _format_ts, _safe_task_id
from web.services.training.history_meta import (
    _default_preprocess_history_name,
    _history_artifact_path,
    _history_group_meta,
    _history_log_path,
    _history_snapshot_path,
    _history_task_ids_for_delete,
    _load_history_task,
    _load_history_task_summary,
    _update_history_task,
)
from web.services.training.history_runtime import (
    _batch_archive_history_tasks,
    _batch_set_history_group,
    _build_config_group_timeline,
    _default_history_archived,
    _display_project_path,
    _history_collections_file,
    _history_delete_run_key,
    _history_delete_task_preview,
    _history_dir,
    _history_runtime_delete_dirs_for_tasks,
    _json_safe_training_payload,
    _list_history_tasks,
    _delete_history_tasks,
    _path_exists,
    _queue_runtime_delete_blockers,
    _resolve_display_path,
    _runtime_meta,
)
from web.services.training.launch_support import (
    _continue_lora_history_meta,
    _write_config_snapshot,
)
from web.services.training.service_state import (
    _load_history_collection_settings,
    _normalize_history_collection_settings,
    _normalize_history_task_ids,
)
from web.services.training.storage import _count_jsonl, _read_json, _write_json_atomic

def list_history_tasks(self, *, include_archived: bool = False, limit: int | None = None, search: str = "", cursor: int = 0) -> list[dict[str, Any]]:
    return _list_history_tasks(include_archived=include_archived, limit=limit, search=search, cursor=cursor)

def get_history_task(self, task_id: str) -> dict[str, Any]:
    return _load_history_task(task_id)

def get_history_task_summary(self, task_id: str) -> dict[str, Any]:
    return _load_history_task_summary(task_id)

def get_history_log_path(self, task_id: str) -> Path:
    return _history_log_path(task_id)

def get_history_artifact_path(self, task_id: str, artifact_key: str) -> Path:
    return _history_artifact_path(task_id, artifact_key)

def get_config_group_timeline(
    self,
    methods_subdir: str,
    variant: str,
    preset: str,
    *,
    group_key: str = "",
    include_archived: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    return _build_config_group_timeline(
        methods_subdir,
        variant,
        preset,
        group_key=group_key,
        include_archived=include_archived,
        task_ids=task_ids,
    )

def get_history_collection_settings(self) -> dict[str, Any]:
    return _load_history_collection_settings()

def save_history_collection_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
    settings = _normalize_history_collection_settings(payload)
    settings["updated_at"] = datetime.now().timestamp()
    settings["updated_at_text"] = _format_ts(settings["updated_at"])
    _write_json_atomic(_history_collections_file(), settings)
    return {"ok": True, **settings}

def update_history_task(self, task_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    return _update_history_task(task_id, patch)

def batch_update_history_tasks(self, payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "").strip().lower()
    task_ids = _normalize_history_task_ids(payload.get("task_ids"))
    if not task_ids:
        raise ValueError("请先选择历史任务")
    if action in {"archive", "unarchive"}:
        return _batch_archive_history_tasks(task_ids, archived=(action == "archive"))
    if action == "set_group":
        return _batch_set_history_group(task_ids, payload.get("group"))
    if action == "delete":
        return self._batch_delete_history_tasks(payload, task_ids)
    raise ValueError("不支持的批量操作")

def delete_history_task(self, task_id: str) -> dict[str, Any]:
    delete_task_ids = _history_task_ids_for_delete(task_id)
    if self.status == "running" and self.current_task_id in delete_task_ids:
        raise RuntimeError("当前运行中的任务不能删除")
    return _delete_history_tasks(delete_task_ids)

def _batch_delete_history_tasks(self, payload: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    delete_runtime_dirs = bool(payload.get("delete_runtime_dirs"))
    plan = self._plan_history_delete(task_ids, delete_runtime_dirs=delete_runtime_dirs)
    if payload.get("dry_run", False):
        return {"ok": True, "dry_run": True, **plan}
    expected_fields = ("expected_task_ids", "expected_runtime_dirs")
    if any(field in payload for field in expected_fields):
        expected_tasks = payload.get("expected_task_ids")
        expected_dirs = payload.get("expected_runtime_dirs")
        if not isinstance(expected_tasks, list) or not isinstance(expected_dirs, list):
            raise RuntimeError("删除预览已变化，请重新预览后确认")
        if not all(isinstance(value, str) for value in expected_tasks + expected_dirs):
            raise RuntimeError("删除预览已变化，请重新预览后确认")
        current_tasks = [str(item.get("id") or "") for item in plan["tasks"]]
        current_dirs = [str(item.get("path") or "") for item in plan["runtime_dirs"]]
        if Counter(expected_tasks) != Counter(current_tasks) or Counter(expected_dirs) != Counter(current_dirs):
            raise RuntimeError("删除预览已变化，请重新预览后确认")
    if plan["blocked"]:
        raise RuntimeError("存在不能删除的任务或运行目录，请先处理阻止项")
    if delete_runtime_dirs and payload.get("confirmed") is not True:
        raise ValueError("彻底删除需要完成二次按钮确认")
    result = _delete_history_tasks([item["id"] for item in plan["tasks"]])
    deleted_runtime_dirs: list[str] = []
    runtime_cleanup_errors: dict[str, str] = {}
    if delete_runtime_dirs:
        for item in plan["runtime_dirs"]:
            path = _resolve_display_path(str(item.get("path") or ""))
            if path is None or not _path_exists(path):
                continue
            try:
                from library.cache_pool.gc import safe_rmtree_run_dir
                from library.cache_pool.refs import release_ref
                from library.cache_pool.store import pool_entry_dir, default_pool_root
                import json as _json
                meta_path = path / "run.meta.json"
                if meta_path.is_file():
                    try:
                        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                    bindings = meta.get("dataset_cache_bindings") if isinstance(meta, dict) else None
                    run_id = path.name
                    if isinstance(bindings, list):
                        for binding in bindings:
                            if not isinstance(binding, dict):
                                continue
                            pool_path = str(binding.get("pool_path") or "").strip()
                            if not pool_path:
                                continue
                            entry = Path(pool_path)
                            if not entry.is_absolute():
                                resolved = _resolve_display_path(pool_path)
                                if resolved is not None:
                                    entry = resolved
                            if entry.is_dir():
                                release_ref(entry, run_id)
                safe_rmtree_run_dir(path)
                deleted_runtime_dirs.append(str(item.get("path") or ""))
            except OSError as exc:
                runtime_cleanup_errors[str(item.get("path") or "")] = str(exc)
    result.update({
        "dry_run": False,
        "deleted_runtime_dirs": deleted_runtime_dirs,
        "runtime_cleanup_errors": runtime_cleanup_errors,
        "preview": plan,
    })
    if delete_runtime_dirs:
        result["message"] = f"已彻底删除 {len(result.get('deleted_task_ids') or [])} 个历史记录和 {len(deleted_runtime_dirs)} 个运行目录"
    return result

def _plan_history_delete(self, task_ids: list[str], *, delete_runtime_dirs: bool) -> dict[str, Any]:
    tasks_by_id = {
        str(task.get("id") or ""): task
        for task in _list_history_tasks(include_archived=True, limit=0)
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    blocked: list[dict[str, str]] = []
    for task_id in task_ids:
        task = tasks_by_id.get(task_id)
        if not task:
            blocked.append({"id": task_id, "reason": "任务不存在"})
            continue
        for linked_id in _history_task_ids_for_delete(task_id):
            if linked_id in seen:
                continue
            linked = tasks_by_id.get(linked_id)
            if linked:
                selected.append(linked)
                seen.add(linked_id)
    if self.status == "running" and self.current_task_id in seen:
        blocked.append({"id": self.current_task_id, "reason": "当前运行中的任务不能删除"})

    runtime_dirs: list[dict[str, str]] = []
    if delete_runtime_dirs:
        run_keys = {
            _history_delete_run_key(task)
            for task in selected
            if _history_delete_run_key(task)
        }
        for candidate in tasks_by_id.values():
            candidate_id = str(candidate.get("id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            if _history_delete_run_key(candidate) not in run_keys:
                continue
            selected.append(candidate)
            seen.add(candidate_id)
        if self.status == "running" and self.current_task_id in seen and not any(
            item.get("id") == self.current_task_id for item in blocked
        ):
            blocked.append({"id": self.current_task_id, "reason": "当前运行中的任务不能删除"})
        runtime_dirs, runtime_blocked = _history_runtime_delete_dirs_for_tasks(selected)
        blocked.extend(runtime_blocked)
        queue_refs = _queue_runtime_delete_blockers(self._queue_items(), runtime_dirs)
        blocked.extend(queue_refs)

    return {
        "tasks": [_history_delete_task_preview(task) for task in selected],
        "runtime_dirs": runtime_dirs,
        "blocked": blocked,
        "delete_runtime_dirs": delete_runtime_dirs,
        "task_count": len(selected),
        "runtime_dir_count": len(runtime_dirs),
    }

def _reserve_history_task_dir(self, job: str, methods_subdir: str, variant: str) -> Path:
    _history_dir().mkdir(parents=True, exist_ok=True)
    task_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{job}-{methods_subdir}-{variant}"
    task_id = _safe_task_id(task_id)
    task_dir = _history_dir() / task_id
    suffix = 1
    while task_dir.exists():
        suffix += 1
        task_dir = _history_dir() / f"{task_id}-{suffix}"
    task_dir.mkdir(parents=True, exist_ok=True)
    self.current_task_id = task_dir.name
    self.current_task_dir = task_dir
    return task_dir

def _start_history_task(
    self,
    *,
    job: str,
    variant: str,
    preset: str,
    methods_subdir: str,
    output_dir: str,
    sample_dir: str,
    data_dirs: dict[str, str],
    sample_config: dict[str, Any],
    command: list[str],
    config_file: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[int] | None = None,
    runtime_info: dict[str, str] | None = None,
    queue_info: dict[str, Any] | None = None,
    distributed_mode: str | None = None,
) -> None:
    task_dir = self.current_task_dir or self._reserve_history_task_dir(job, methods_subdir, variant)
    now = time.time()
    runtime_meta = _runtime_meta(runtime_info)
    history_meta = _history_group_meta(
        methods_subdir,
        variant,
        preset,
        output_dir=output_dir,
        runtime_info=runtime_meta,
        resume_info=resume_info,
        task_id=task_dir.name,
    )
    default_name = _default_preprocess_history_name({
        "id": task_dir.name,
        "job": job,
        "output_dir": output_dir,
        **_continue_lora_history_meta(continue_info),
        **runtime_meta,
        **history_meta,
    })
    continue_meta = _continue_lora_history_meta(continue_info)
    queue_meta = queue_info if isinstance(queue_info, dict) else {}
    meta = {
        "id": task_dir.name,
        "name": default_name,
        "group": "",
        "archived": _default_history_archived(job),
        "job": job,
        "state": "running",
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "output_dir": output_dir,
        "sample_dir": sample_dir,
        "source_image_dir": data_dirs.get("source_image_dir", ""),
        "resized_image_dir": data_dirs.get("resized_image_dir", ""),
        "lora_cache_dir": data_dirs.get("lora_cache_dir", ""),
        "data_dirs": data_dirs,
        "sample_config": sample_config,
        "command": command,
        "resume_from": resume_info or {},
        **continue_meta,
        "gpu_whitelist": gpu_whitelist or [],
        **runtime_meta,
        **history_meta,
        **queue_meta,
        "started_at": now,
        "started_at_text": _format_ts(now),
        "finished_at": None,
        "finished_at_text": "",
        "message": "",
        "returncode": None,
        "log_count": 0,
        "metric_count": 0,
    }
    if distributed_mode is not None:
        meta["distributed_mode"] = distributed_mode
    _write_json_atomic(task_dir / "meta.json", meta)
    _write_config_snapshot(
        task_dir / "config.snapshot.toml",
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
        continue_info=continue_info,
    )

def _finish_history_task(self, *, state: str, message: str, returncode: int) -> None:
    if not self.current_task_dir:
        return
    meta = _read_json(self.current_task_dir / "meta.json")
    now = time.time()
    meta.update({
        "state": state,
        "finished_at": now,
        "finished_at_text": _format_ts(now),
        "message": message,
        "returncode": returncode,
        "log_count": _count_jsonl(self.current_task_dir / "logs.jsonl"),
        "metric_count": _count_jsonl(self.current_task_dir / "metrics.jsonl"),
    })
    _write_json_atomic(self.current_task_dir / "meta.json", meta)

def _append_history_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
    if not self.current_task_dir:
        return
    try:
        from web.services.training.live_utils import _json_safe_training_payload

        with (self.current_task_dir / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe_training_payload(payload), ensure_ascii=False) + "\n")
    except Exception:
        pass
