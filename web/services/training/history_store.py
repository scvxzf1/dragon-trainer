"""History storage and delete-planning helpers.

This module is a mechanical extraction from ``web.services.training_service``.
It keeps legacy helper names so the service facade can stay stable while the
implementation is split into smaller modules.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from web.services.training.common import (
    _clean_history_text,
    _float_or_none,
    _format_ts,
    _int_or_none,
)
from web.services.training.context import training_facade as _training_facade
from web.services.training.history_config_chips import history_config_chips_for_task_dir
from web.services.training.history_meta import (
    _default_preprocess_history_name,
    _fill_history_group_meta,
    _fill_history_runtime_meta,
    _history_task_archived,
    _history_task_dir,
    _is_legacy_auto_preprocess_name,
)
from web.services.training.storage import (
    _count_jsonl as _storage_count_jsonl,
    _read_json as _storage_read_json,
    _read_jsonl as _storage_read_jsonl,
    _write_json_atomic as _storage_write_json_atomic,
)

if TYPE_CHECKING:
    from web.services.training_service import (
        HISTORY_DIR,
        MAX_HISTORY_ITEMS,
        ROOT,
        _absolute_display_path,
        _clean_history_text,
        _default_preprocess_history_name,
        _display_settings_path,
        _display_project_path,
        _fill_history_group_meta,
        _fill_history_runtime_meta,
        _format_ts,
        _history_task_archived,
        _history_task_dir,
        _int_or_none,
        _is_deleting_history_dir,
        _is_legacy_auto_preprocess_name,
        _path_is_relative_to,
        _path_exists,
        _queue_item_runtime_delete_dir,
        _resolve_display_path,
        resolve_output_root,
    )


_LOCAL_IMPL_NAMES = {
    "_list_history_tasks",
    "_history_meta_paths",
    "_history_meta_records",
    "_history_meta_record",
    "_sync_bound_history_collection_groups",
    "_preferred_bound_history_collection_group",
    "_history_summary",
    "_history_jsonl_count",
    "_safe_history_summary",
    "_repair_history_meta",
    "_linked_preprocess_task_for_training",
    "_linked_preprocess_tasks_for_training",
    "_history_linked_task_brief",
    "_history_delete_run_key",
    "_history_delete_task_preview",
    "_history_runtime_delete_dirs_for_tasks",
    "_queue_runtime_delete_blockers",
    "_delete_history_tasks",
    "_delete_history_task",
    "_is_deleting_history_dir",
    "_reserve_deleting_history_dir",
    "_default_preprocess_history_name",
    "_fill_history_group_meta",
    "_fill_history_runtime_meta",
    "_history_task_archived",
    "_history_task_dir",
    "_int_or_none",
    "_is_legacy_auto_preprocess_name",
}



def _project_root() -> Path:
    return _training_facade().ROOT


def _history_dir() -> Path:
    return _training_facade().HISTORY_DIR


def _max_history_items() -> int:
    return _training_facade().MAX_HISTORY_ITEMS


def _absolute_display_path(*args, **kwargs):
    return _training_facade()._absolute_display_path(*args, **kwargs)


def _display_settings_path(*args, **kwargs):
    return _training_facade()._display_settings_path(*args, **kwargs)


def _display_project_path(*args, **kwargs):
    return _training_facade()._display_project_path(*args, **kwargs)


def _path_is_relative_to(*args, **kwargs):
    return _training_facade()._path_is_relative_to(*args, **kwargs)


def _path_exists(*args, **kwargs):
    return _training_facade()._path_exists(*args, **kwargs)


def _queue_item_runtime_delete_dir(*args, **kwargs):
    return _training_facade()._queue_item_runtime_delete_dir(*args, **kwargs)


def _resolve_display_path(*args, **kwargs):
    return _training_facade()._resolve_display_path(*args, **kwargs)


def resolve_output_root() -> Path:
    return _training_facade().resolve_output_root()


def _read_json(*args, **kwargs):
    reader = getattr(_training_facade(), "_read_json", _storage_read_json)
    return reader(*args, **kwargs)


def _read_jsonl(*args, **kwargs):
    reader = getattr(_training_facade(), "_read_jsonl", _storage_read_jsonl)
    return reader(*args, **kwargs)


def _count_jsonl(*args, **kwargs):
    counter = getattr(_training_facade(), "_count_jsonl", _storage_count_jsonl)
    return counter(*args, **kwargs)


def _write_json_atomic(*args, **kwargs):
    writer = getattr(_training_facade(), "_write_json_atomic", _storage_write_json_atomic)
    return writer(*args, **kwargs)


def _history_search_query(search: str) -> tuple[str, str]:
    raw = str(search or "").strip()
    separator = min((index for mark in (":", "：") if (index := raw.find(mark)) >= 0), default=-1)
    if separator >= 0:
        prefix = raw[:separator].strip().casefold()
        query = raw[separator + 1 :].strip().casefold()
        if query:
            if prefix in {"组", "集合", "group", "collection"}:
                return "group", query
            if prefix in {"配置", "配置组", "config"}:
                return "config", query
    return "global", raw.casefold()


def _history_task_matches_search(task: dict[str, Any], search: str) -> bool:
    mode, needle = _history_search_query(search)
    return _history_task_matches_query(task, mode, needle)


def _history_task_matches_query(task: dict[str, Any], mode: str, needle: str) -> bool:
    if not needle:
        return True
    fields = {
        "group": ("group", "collection", "history_collection"),
        "config": ("history_group_label", "history_source_config_file", "config_group", "variant", "preset"),
        "global": (
            "id", "name", "output_name", "task_name", "history_run_label", "group",
            "history_group_label", "history_source_config_file", "source_task_name",
            "model_family", "variant", "training_variant", "preset", "methods_subdir",
            "run_dir", "output_dir", "message",
        ),
    }[mode]
    if any(needle in str(task.get(key) or "").casefold() for key in fields):
        return True
    if mode == "global":
        state = str(task.get("state") or task.get("status") or "unknown").casefold()
        if needle in state:
            return True
        state_text = {
            "idle": "已完成", "completed": "已完成", "done": "已完成",
            "running": "训练中", "training": "训练中", "compiling": "编译中",
            "queued": "排队中", "interrupted": "已中断", "stopped": "已停止",
            "canceled": "已取消", "cancelled": "已取消", "error": "失败",
            "failed": "失败", "unknown": "未知",
        }.get(state, state)
        return needle in state_text.casefold()
    return False


def _list_history_tasks(*, include_archived: bool = False, limit: int | None = None, search: str = "", cursor: int = 0) -> list[dict[str, Any]]:
    meta_paths = _history_meta_paths()
    records = _history_meta_records(meta_paths, repair=True)
    _sync_bound_history_collection_groups(records=records)
    search_mode, search_needle = _history_search_query(search)

    tasks = []
    for record in records:
        meta_path = record["path"]
        task = _safe_history_summary(record["meta"], meta_path.parent)
        if task is None:
            continue
        if include_archived or not task.get("archived"):
            if not _history_task_matches_query(task, search_mode, search_needle):
                continue
            tasks.append(task)
    tasks.sort(key=lambda item: item.get("started_at") or 0, reverse=True)
    cursor = max(0, int(cursor or 0))
    if limit is None:
        limit = _max_history_items()
    if limit and limit > 0:
        return tasks[cursor : cursor + limit]
    return tasks


def _history_meta_paths() -> list[Path]:
    if not _path_exists(_history_dir()):
        return []
    try:
        candidates = list(_history_dir().iterdir())
    except OSError:
        return []
    out: list[Path] = []
    for task_dir in candidates:
        if _is_deleting_history_dir(task_dir):
            continue
        meta_path = task_dir / "meta.json"
        if _path_exists(meta_path):
            out.append(meta_path)
    return out


def _history_meta_records(
    meta_paths: list[Path] | None = None,
    *,
    repair: bool = False,
) -> list[dict[str, Any]]:
    paths = meta_paths if meta_paths is not None else _history_meta_paths()
    records: list[dict[str, Any]] = []
    for meta_path in paths:
        if _is_deleting_history_dir(meta_path.parent):
            continue
        meta = _read_json(meta_path)
        if not meta:
            continue
        if repair:
            _repair_history_meta(meta_path, meta)
        records.append(_history_meta_record(meta_path, meta))
    return records


def _history_meta_record(meta_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    task_id = str(meta.get("id") or meta_path.parent.name).strip()
    work = dict(meta)
    _fill_history_runtime_meta(work)
    _fill_history_group_meta(work)
    return {
        "id": task_id,
        "path": meta_path,
        "meta": meta,
        "group_key": str(work.get("history_group_key") or "").strip(),
        "job": str(meta.get("job") or "").strip(),
        "group": _clean_history_text(meta.get("group"), max_len=48),
        "updated_at": float(meta.get("updated_at") or 0),
        "started_at": float(meta.get("started_at") or 0),
    }


def _sync_bound_history_collection_groups(
    meta_paths: list[Path] | None = None,
    *,
    records: list[dict[str, Any]] | None = None,
) -> int:
    records = records if records is not None else _history_meta_records(meta_paths)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("group_key") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(record)

    changed = 0
    for records_in_group in grouped.values():
        target_group = _preferred_bound_history_collection_group(records_in_group)
        if target_group is None:
            continue
        for record in records_in_group:
            if str(record.get("group") or "") == target_group:
                continue
            meta = dict(record["meta"])
            meta["group"] = target_group
            now = time.time()
            meta["updated_at"] = now
            meta["updated_at_text"] = _format_ts(now)
            try:
                _write_json_atomic(record["path"], meta)
                record["meta"] = meta
                record["group"] = target_group
                record["updated_at"] = now
                changed += 1
            except OSError:
                continue
    return changed


def _preferred_bound_history_collection_group(records: list[dict[str, Any]]) -> str | None:
    candidates = [record for record in records if str(record.get("group") or "").strip()]
    if not candidates:
        return None
    candidates.sort(
        key=lambda record: (
            1 if record.get("job") == "training" else 0,
            float(record.get("updated_at") or 0),
            float(record.get("started_at") or 0),
        ),
        reverse=True,
    )
    return str(candidates[0].get("group") or "").strip()


def _history_summary(meta: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    out = dict(meta)
    out["id"] = task_dir.name
    out["name"] = str(out.get("name") or "")
    out["group"] = str(out.get("group") or "")
    if not str(out.get("training_mode") or "").strip():
        out["training_mode"] = "continue_lora" if out.get("continue_from_weight_abs_path") else "fresh"
    for key in (
        "continue_from_weight_abs_path",
        "continue_from_weight_name",
        "continue_from_weight_kind",
    ):
        out[key] = str(out.get(key) or "")
    out["archived"] = _history_task_archived(out)
    out["project_root_abs"] = str(_project_root().resolve())
    out["history_dir"] = _display_project_path(str(task_dir))
    out["history_dir_abs"] = str(task_dir)
    out["config_snapshot"] = _display_project_path(str(task_dir / "config.snapshot.toml"))
    out["logs_path"] = _display_project_path(str(task_dir / "logs.jsonl"))
    out["metrics_path"] = _display_project_path(str(task_dir / "metrics.jsonl"))
    out["system_path"] = _display_project_path(str(task_dir / "system.jsonl"))
    data_dirs = out.get("data_dirs") if isinstance(out.get("data_dirs"), dict) else {}
    for key in ("source_image_dir", "resized_image_dir", "lora_cache_dir"):
        out[key] = str(out.get(key) or data_dirs.get(key) or "")
    _fill_history_runtime_meta(out)
    out["run_dir_abs"] = _absolute_display_path(out.get("run_dir"))
    out["absolute_paths"] = _history_absolute_paths(out, task_dir)
    _fill_history_group_meta(out)
    if not out["name"]:
        out["name"] = _default_preprocess_history_name(out)
    out["log_count"] = _history_jsonl_count(out, "log_count", task_dir / "logs.jsonl")
    out["metric_count"] = _history_metric_count(out, task_dir)
    out.update(_history_metric_summary(task_dir, out["metric_count"]))
    command = out.get("command")
    legacy_training = out.get("job") == "training" and not out.get("distributed_mode")
    chips = history_config_chips_for_task_dir(
        task_dir,
        variant=str(out.get("variant") or ""),
        legacy_command=(command if isinstance(command, list) else []) if legacy_training else None,
    )
    if chips["distributed_mode"]:
        out["distributed_mode"] = chips["distributed_mode"]
    out["model_family"] = chips["model_family"] or str(out.get("model_family") or "")
    out["training_variant"] = chips["training_variant"]
    out["preprocess_precision"] = chips["preprocess_precision"]
    out["block_swap_precision"] = chips["block_swap_precision"]
    out["base_compute"] = chips["base_compute"]
    out["precision_preference"] = chips["precision_preference"]
    return out


_HISTORY_PATH_FIELDS = (
    "run_dir_abs",
    "history_dir_abs",
    "runtime_config_file",
    "original_config_file",
    "dataset_config_file",
    "model_cache_dir",
    "dataset_cache_dir",
    "training_output_dir",
    "sample_dir",
    "logs_dir",
    "logs_path",
    "metrics_path",
    "system_path",
    "config_snapshot",
)


def _history_absolute_paths(task: dict[str, Any], task_dir: Path) -> dict[str, str]:
    """Resolve detail-copy paths on the server using configured path roots."""
    run_dir = _resolve_display_path(str(task.get("run_dir") or ""))
    absolute: dict[str, str] = {}
    for key in _HISTORY_PATH_FIELDS:
        raw = str(task.get(key) or "").strip()
        if key == "run_dir_abs":
            raw = str(task.get("run_dir") or "")
        elif key == "history_dir_abs":
            raw = str(task_dir)
        elif key in {"logs_path", "metrics_path", "system_path", "config_snapshot"}:
            raw = str(task_dir / {
                "logs_path": "logs.jsonl",
                "metrics_path": "metrics.jsonl",
                "system_path": "system.jsonl",
                "config_snapshot": "config.snapshot.toml",
            }[key])
        if not raw:
            continue
        is_run_file = key in {
            "runtime_config_file", "original_config_file", "dataset_config_file",
        } and Path(raw).name == raw
        if is_run_file:
            if run_dir is None:
                continue
            resolved = (run_dir / raw).resolve()
        else:
            resolved = _resolve_display_path(raw)
        if resolved is not None:
            absolute[key] = str(resolved)
    return absolute


def _linked_preprocess_task_for_training(task: dict[str, Any]) -> dict[str, Any] | None:
    tasks = _linked_preprocess_tasks_for_training(task)
    if not tasks:
        return None
    return _history_linked_task_brief(tasks[0])


def _linked_preprocess_tasks_for_training(task: dict[str, Any]) -> list[dict[str, Any]]:
    if str(task.get("job") or "").strip() != "training":
        return []
    run_key = _history_delete_run_key(task)
    if not run_key:
        return []

    current_id = str(task.get("id") or "").strip()
    out: list[dict[str, Any]] = []
    seen = {current_id} if current_id else set()
    for candidate in _list_history_tasks(include_archived=True, limit=0):
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        if str(candidate.get("job") or "").strip() != "preprocess":
            continue
        if _history_delete_run_key(candidate) != run_key:
            continue
        out.append(candidate)
        seen.add(candidate_id)
    return out


def _history_linked_task_brief(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "name",
        "job",
        "state",
        "archived",
        "started_at",
        "started_at_text",
        "finished_at",
        "finished_at_text",
        "history_run_label",
        "history_source_config_file",
        "history_group_key",
        "run_dir",
        "output_dir",
        "training_output_dir",
        "log_count",
        "metric_count",
    )
    return {key: task.get(key) for key in keys if key in task}


def _history_delete_run_key(task: dict[str, Any]) -> str:
    for key in ("run_dir", "training_output_dir", "output_dir"):
        path = _resolve_display_path(str(task.get(key) or ""))
        if path is None:
            continue
        if path.name == "training_output":
            path = path.parent
        return str(path)
    return ""


def _history_delete_task_preview(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(task.get("id") or ""),
        "name": str(task.get("name") or task.get("history_run_label") or ""),
        "job": str(task.get("job") or ""),
        "state": str(task.get("state") or ""),
        "started_at_text": str(task.get("started_at_text") or ""),
        "run_dir": str(task.get("run_dir") or ""),
        "output_dir": str(task.get("training_output_dir") or task.get("output_dir") or ""),
    }


def _history_runtime_delete_dirs_for_tasks(
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    out: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    seen: set[str] = set()
    for task in tasks:
        raw = _history_delete_run_key(task)
        if not raw or raw in seen:
            continue
        seen.add(raw)
        path = _resolve_display_path(raw)
        label = _display_settings_path(path) if path is not None else raw
        if path is None:
            blocked.append({"path": raw, "reason": "运行目录路径无效"})
            continue
        if not _path_exists(path):
            out.append({"path": label, "status": "missing"})
            continue
        if not path.is_dir():
            blocked.append({"path": label, "reason": "运行目录不是文件夹"})
            continue
        try:
            output_root = resolve_output_root()
        except Exception as exc:
            blocked.append({"path": label, "reason": f"无法解析输出根目录: {exc}"})
            continue
        if path == output_root or not _path_is_relative_to(path, output_root):
            blocked.append({"path": label, "reason": "运行目录不在 WebUI 输出根目录内"})
            continue
        if not _is_web_runtime_dir(path):
            blocked.append({"path": label, "reason": "缺少 WebUI runtime 标记"})
            continue
        out.append({"path": label, "status": "ready"})
    return out, blocked


def _is_web_runtime_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl

    return _impl._is_web_runtime_dir(*args, **kwargs)


def _queue_runtime_delete_blockers(
    queue_items: list[dict[str, Any]],
    runtime_dirs: list[dict[str, str]],
) -> list[dict[str, str]]:
    protected = {
        str(item.get("path") or "")
        for item in runtime_dirs
        if str(item.get("status") or "") == "ready"
    }
    if not protected:
        return []
    blocked: list[dict[str, str]] = []
    for item in queue_items:
        if item.get("state") not in {"queued", "running"}:
            continue
        run_dir = _queue_item_runtime_delete_dir(item)
        if run_dir is None:
            continue
        label = _display_settings_path(run_dir)
        if label in protected:
            blocked.append({
                "id": str(item.get("id") or ""),
                "path": label,
                "reason": "运行目录仍被等待或运行中的队列项引用",
            })
    return blocked


def _delete_history_tasks(task_ids: list[str]) -> dict[str, Any]:
    cleanup_errors: dict[str, str] = {}
    deleted_task_ids: list[str] = []
    for task_id in task_ids:
        result = _delete_history_task(task_id)
        deleted_task_ids.append(task_id)
        if result.get("cleanup_error"):
            cleanup_errors[task_id] = str(result.get("cleanup_error"))

    linked_count = max(0, len(deleted_task_ids) - 1)
    message = "任务已删除"
    if linked_count:
        message = f"任务已删除，并一并删除 {linked_count} 个对应预处理任务"
    if cleanup_errors:
        message = "任务已从列表移除，部分磁盘残留稍后可手动清理"
    payload: dict[str, Any] = {
        "ok": True,
        "message": message,
        "deleted_task_ids": deleted_task_ids,
        "linked_preprocess_deleted": linked_count,
    }
    if cleanup_errors:
        payload["cleanup_errors"] = cleanup_errors
        payload["cleanup_error"] = "; ".join(
            f"{key}: {value}" for key, value in cleanup_errors.items()
        )
    return payload


def _delete_history_task(task_id: str) -> dict[str, Any]:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    deleting_dir = _reserve_deleting_history_dir(task_dir)
    try:
        task_dir.rename(deleting_dir)
    except OSError as exc:
        raise ValueError(f"删除任务失败: {exc}") from exc

    try:
        shutil.rmtree(deleting_dir)
    except OSError as exc:
        return {
            "ok": True,
            "message": "任务已从列表移除，部分磁盘残留稍后可手动清理",
            "cleanup_error": str(exc),
        }
    return {"ok": True, "message": "任务已删除"}


def _is_deleting_history_dir(task_dir: Path) -> bool:
    return ".deleting-" in task_dir.name


def _reserve_deleting_history_dir(task_dir: Path) -> Path:
    base = f".{task_dir.name}.deleting-{int(time.time() * 1000)}"
    candidate = task_dir.with_name(base)
    suffix = 1
    while _path_exists(candidate):
        suffix += 1
        candidate = task_dir.with_name(f"{base}-{suffix}")
    return candidate


def _history_jsonl_count(meta: dict[str, Any], key: str, path: Path) -> int:
    if str(meta.get("state") or "") in {"running", "compiling"}:
        return _count_jsonl(path)
    if key in meta:
        count = _int_or_none(meta.get(key))
        if count is not None and count >= 0:
            return count
    return _count_jsonl(path)


def _history_metric_count(meta: dict[str, Any], task_dir: Path) -> int:
    """Prefer metrics.jsonl, fall back to progress.jsonl step events.

    CLI / debug runs often only write ``progress.jsonl``; without this fallback
    the history list shows ``0 loss`` even though detail charts can render.
    """
    metrics_path = task_dir / "metrics.jsonl"
    count = _history_jsonl_count(meta, "metric_count", metrics_path)
    if count > 0:
        return count
    # Ignore a stale explicit 0 when progress has real step metrics.
    progress_path = task_dir / "progress.jsonl"
    progress_count = _count_progress_metric_events(progress_path)
    if progress_count > 0:
        return progress_count
    return count


def _history_metric_summary(task_dir: Path, metric_count: int) -> dict[str, Any]:
    """Return bounded list-row metrics without loading the full detail payload."""
    metrics_path = task_dir / "metrics.jsonl"
    progress_path = task_dir / "progress.jsonl"
    summary = _read_history_metric_summary(metrics_path, metric_count, progress=False)
    if summary.get("final_loss") is not None or summary.get("last_step") is not None:
        return summary
    fallback = _read_history_metric_summary(progress_path, metric_count, progress=True)
    return fallback or summary


def _read_history_metric_summary(
    path: Path,
    metric_count: int,
    *,
    progress: bool,
) -> dict[str, Any]:
    if not _path_exists(path) or not path.is_file():
        return {}

    stride = max(1, (max(1, metric_count) + 23) // 24)
    preview: list[float] = []
    final_loss: float | None = None
    last_step: int | None = None
    valid_index = 0
    saw_validation = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(event, dict):
                    continue
                if progress and str(event.get("ev") or "") not in {"step", "val"}:
                    continue
                # Legacy validation events duplicate CMMD into the loss field.
                if event.get("kind") == "val" or event.get("ev") == "val":
                    saw_validation = True
                    continue
                loss = next(
                    (
                        value
                        for key in ("loss", "loss/average", "loss/current")
                        if (value := _float_or_none(event.get(key))) is not None
                    ),
                    None,
                )
                step = next(
                    (
                        value
                        for key in ("step", "global_step", "current_step")
                        if (value := _int_or_none(event.get(key))) is not None
                    ),
                    None,
                )
                if step is not None:
                    last_step = step
                if loss is None:
                    continue
                final_loss = loss
                if valid_index % stride == 0:
                    preview.append(loss)
                valid_index += 1
                if len(preview) > 48:
                    preview = preview[::2]
                    stride *= 2
    except OSError:
        return {}

    if final_loss is None:
        summary = {"last_step": last_step} if last_step is not None else {}
        if saw_validation:
            summary.update(final_loss=None, loss_preview=[])
        return summary
    if not preview or preview[-1] != final_loss:
        preview.append(final_loss)
    if len(preview) > 24:
        preview = preview[:23] + [preview[-1]]
    return {
        "final_loss": final_loss,
        "last_step": last_step,
        "loss_preview": preview,
    }


def _count_progress_metric_events(progress_path: Path) -> int:
    if not _path_exists(progress_path) or not progress_path.is_file():
        return 0
    count = 0
    try:
        with progress_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if str(event.get("ev") or "") not in {"step", "val"}:
                    continue
                if any(
                    key in event
                    for key in ("loss", "loss/average", "loss/current", "lr", "lr/unet", "cmmd")
                ):
                    count += 1
    except OSError:
        return 0
    return count


def _safe_history_summary(meta: dict[str, Any], task_dir: Path) -> dict[str, Any] | None:
    try:
        return _history_summary(meta, task_dir)
    except (OSError, TypeError, ValueError):
        return None


def _repair_history_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    before = dict(meta)
    _fill_history_runtime_meta(meta)
    _fill_history_group_meta(meta)
    if str(meta.get("job") or "").strip() == "preprocess":
        # 旧版本写入 archived=false；没有 updated_at 表示用户没有手动取消归档。
        if "updated_at" not in meta and meta.get("archived") is not True:
            meta["archived"] = True
        name = _default_preprocess_history_name(meta)
        if name and _is_legacy_auto_preprocess_name(meta.get("name"), name):
            meta["name"] = name
    if meta != before:
        try:
            _write_json_atomic(meta_path, meta)
        except OSError:
            pass
