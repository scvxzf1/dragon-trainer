"""Queue state helpers delegated from ``TrainingService``."""

from __future__ import annotations

import time
from typing import Any

from web.services.training.common import _format_ts, _positive_int_or_none
from web.services.training.constants import max_queue_items, queue_dir
from web.services.training.distributed import item_distributed_mode
from web.services.training.queue_revision import queue_revision
from web.services.training.service_state import (
    _normalize_queue_auto_retry,
    _normalize_queue_failure_policy,
    _normalize_queue_max_attempts,
    _normalize_queue_retry_backoff,
    _write_training_queue_state,
)

_LOCAL_IMPL_NAMES = {
    "start_queue_on_startup",
    "get_queue_snapshot",
    "_repair_queue_on_startup",
    "_normalize_queue",
    "_queue_items",
    "_find_queue_item",
    "_update_queue_item",
    "_attach_history_task_to_queue_item",
    "_queue_history_meta",
    "_compact_queue",
    "_save_queue",
    "_broadcast_queue",
}


async def start_queue_on_startup(self) -> None:
    self._schedule_queue_dispatch()


def get_queue_snapshot(self) -> dict[str, Any]:
    self._normalize_queue()
    summary = {
        "total": 0,
        "queued": 0,
        "running": 0,
        "done": 0,
        "error": 0,
        "canceled": 0,
    }
    items = []
    for item in self._queue_items():
        summary["total"] += 1
        state = str(item.get("state") or "")
        if state in summary:
            summary[state] += 1
        snapshot_item = dict(item)
        mode = item_distributed_mode(item)
        if mode == "pp" or "distributed_mode" in item:
            snapshot_item["distributed_mode"] = mode
        items.append(snapshot_item)
    snapshot = {
        "ok": True,
        "paused": self._queue_paused,
        "failure_policy": self._queue_failure_policy,
        "auto_retry": bool(getattr(self, "_queue_auto_retry", False)),
        "max_attempts": int(getattr(self, "_queue_max_attempts", 1) or 1),
        "retry_backoff_sec": float(getattr(self, "_queue_retry_backoff_sec", 0.0) or 0.0),
        "status": self.status,
        "current_item_id": self._current_queue_item_id,
        "current_task_id": self.current_task_id,
        "launching_item_id": self._queue_launching_item_id,
        "summary": summary,
        "items": items,
    }
    snapshot["revision"] = queue_revision({
        **snapshot,
        "process_active": bool(self.process and getattr(self.process, "returncode", None) is None),
    })
    return snapshot


def _repair_queue_on_startup(self) -> None:
    changed = False
    now = time.time()
    self._normalize_queue()
    for item in self._queue_items():
        if item.get("state") == "running":
            item.update({
                "state": "error",
                "message": "WebUI 重启时发现旧运行中队列项，已标记为异常",
                "finished_at": now,
                "finished_at_text": _format_ts(now),
            })
            changed = True
    if changed:
        self._queue_paused = True
        self._queue["paused"] = True
        self._save_queue()


def _normalize_queue(self) -> None:
    if not isinstance(self._queue, dict):
        self._queue = {}
    items = self._queue.get("items")
    if not isinstance(items, list):
        self._queue["items"] = []
    self._queue_paused = bool(self._queue.get("paused", self._queue_paused))
    self._queue["paused"] = self._queue_paused
    self._queue_failure_policy = _normalize_queue_failure_policy(
        self._queue.get("failure_policy", self._queue_failure_policy)
    )
    self._queue["failure_policy"] = self._queue_failure_policy
    self._queue_auto_retry = _normalize_queue_auto_retry(
        self._queue.get("auto_retry", getattr(self, "_queue_auto_retry", False))
    )
    self._queue["auto_retry"] = self._queue_auto_retry
    self._queue_max_attempts = _normalize_queue_max_attempts(
        self._queue.get("max_attempts", getattr(self, "_queue_max_attempts", 1))
    )
    self._queue["max_attempts"] = self._queue_max_attempts
    self._queue_retry_backoff_sec = _normalize_queue_retry_backoff(
        self._queue.get("retry_backoff_sec", getattr(self, "_queue_retry_backoff_sec", 0.0))
    )
    self._queue["retry_backoff_sec"] = self._queue_retry_backoff_sec
    for item in self._queue["items"]:
        if not isinstance(item, dict):
            continue
        item.setdefault("retry_of", "")
        item["attempt"] = max(1, _positive_int_or_none(item.get("attempt")) or 1)
        try:
            next_run_raw = item.get("next_run_at")
            item["next_run_at"] = (
                float(next_run_raw)
                if next_run_raw not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            item["next_run_at"] = None


def _queue_items(self) -> list[dict[str, Any]]:
    self._normalize_queue()
    items = self._queue["items"]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    self._queue["items"] = out
    return out


def _find_queue_item(self, item_id: str) -> dict[str, Any] | None:
    needle = str(item_id or "").strip()
    for item in self._queue_items():
        if str(item.get("id") or "") == needle:
            return item
    return None


def _update_queue_item(self, item_id: str, patch: dict[str, Any]) -> None:
    item = self._find_queue_item(item_id)
    if item is not None:
        item.update(patch)


def _attach_history_task_to_queue_item(self, item_id: str, task_id: str) -> None:
    item = self._find_queue_item(item_id)
    if item is None or not task_id:
        return
    history_ids = item.get("history_task_ids")
    if not isinstance(history_ids, list):
        history_ids = []
        item["history_task_ids"] = history_ids
    if task_id not in history_ids:
        history_ids.append(task_id)
    item["message"] = "正在运行"
    self._save_queue()


def _queue_history_meta(self, item_id: str) -> dict[str, Any]:
    item = self._find_queue_item(item_id)
    if item is None:
        return {}
    return {
        "from_queue": True,
        "queue_item_id": str(item.get("id") or item_id or ""),
        "queue_kind": str(item.get("kind") or ""),
        "queue_retry_of": str(item.get("retry_of") or ""),
        "queue_attempt": max(1, _positive_int_or_none(item.get("attempt")) or 1),
        "queue_created_at": item.get("created_at"),
        "queue_created_at_text": str(item.get("created_at_text") or ""),
    }


def _compact_queue(self) -> None:
    items = self._queue_items()
    if len(items) <= max_queue_items():
        return
    protected = [item for item in items if item.get("state") in {"queued", "running"}]
    finished = [item for item in items if item.get("state") not in {"queued", "running"}]
    keep_finished = max(0, max_queue_items() - len(protected))
    self._queue["items"] = [*protected, *(finished[-keep_finished:] if keep_finished else [])]


def _save_queue(self) -> None:
    self._normalize_queue()
    queue_dir().mkdir(parents=True, exist_ok=True)
    self._queue["paused"] = self._queue_paused
    self._queue["failure_policy"] = self._queue_failure_policy
    self._queue["auto_retry"] = bool(getattr(self, "_queue_auto_retry", False))
    self._queue["max_attempts"] = int(getattr(self, "_queue_max_attempts", 1) or 1)
    self._queue["retry_backoff_sec"] = float(getattr(self, "_queue_retry_backoff_sec", 0.0) or 0.0)
    self._queue["updated_at"] = time.time()
    self._queue["updated_at_text"] = _format_ts(self._queue["updated_at"])
    _write_training_queue_state(self._queue)


async def _broadcast_queue(self) -> None:
    await self._broadcast({"type": "queue", **self.get_queue_snapshot()})
