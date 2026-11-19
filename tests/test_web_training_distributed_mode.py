from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.training_resume_test_support import (
    _patch_queue_storage,
    _patch_resume_runtime_output_root,
    _patch_runtime_service_paths,
    _write_group_task,
    _write_resume_history,
    _write_runtime_config_tree,
)
from web.routes import training as routes
from web.services import training_service
from web.services.training import gpu


@pytest.fixture
def app_and_storage(tmp_path, monkeypatch):
    queue_dir = _patch_queue_storage(monkeypatch, tmp_path)
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(gpu, "query_gpu_identities", lambda selection, **kwargs: dict.fromkeys(selection))
    monkeypatch.delenv("ANIMA_ACCELERATE_LAUNCH", raising=False)
    monkeypatch.delenv("ANIMA_ACCELERATE_NUM_PROCESSES", raising=False)
    app = web.Application()
    service = training_service.TrainingService(app)
    app["training_service"] = service
    routes.setup_training_routes(app)
    monkeypatch.setattr(service, "_schedule_queue_dispatch", lambda: None)
    return app, service, queue_dir / "queue.json", history_dir


def test_next_queue_persists_mode_from_materialized_config(tmp_path, monkeypatch, app_and_storage):
    app, _, queue_file, _ = app_and_storage
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    config = tmp_path / "configs/imported/522.toml"
    config.write_text(
        config.read_text() + '\nmodel_family = "krea2_raw"\npipeline_parallel = true\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})

    async def request():
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/training/queue", json={
                "variant": "522", "preset": "default", "methods_subdir": "imported",
                "config_file": "configs/imported/522.toml", "gpu_whitelist": [0, 1],
                "start_paused": True, "confirmed": True, "confirm_preprocess": True,
            })
            payload = await response.json()
            assert response.status == 200, payload
            queue_response = await client.get("/api/training/queue")
            assert (await queue_response.json())["items"][0]["distributed_mode"] == "pp"
            return payload["item"]

    item = asyncio.run(request())
    assert item["requires_preprocess"] is True
    assert item["distributed_mode"] == "pp"
    assert tomllib.loads((tmp_path / item["runtime_config_file"]).read_text())["pipeline_parallel"] is True
    assert json.loads(queue_file.read_text())["items"][0]["distributed_mode"] == "pp"


def test_batch_modes_apply_overrides_without_inferring_ddp_world_size(tmp_path, monkeypatch, app_and_storage):
    app, _, queue_file, _ = app_and_storage
    enabled = tmp_path / "enabled.toml"
    disabled = tmp_path / "disabled.toml"
    empty = tmp_path / "empty.toml"
    enabled.write_text('model_family = "anima"\npipeline_parallel = true\n')
    disabled.write_text('model_family = "anima"\npipeline_parallel = "false"\n')
    empty.touch()
    monkeypatch.setattr(routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(routes, "is_web_runtime_config", lambda *_: True)

    async def request():
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/training/queue/batch/start", json={
                "gpu_whitelist": [0, 1],
                "items": [
                    {"variant": "lora", "config_file": str(enabled), "extra_args": ["--no-pipeline_parallel"]},
                    {"variant": "lora", "config_file": str(disabled), "extra_args": ["--pipeline_parallel"]},
                    {"variant": "lora", "config_file": str(disabled)},
                    {"variant": "lora", "config_file": str(empty),
                     "extra_args": ["--pipeline_parallel", "--model_family", "anima"]},
                ],
            })
            payload = await response.json()
            assert response.status == 200 and payload["ok"], payload
            return payload

    payload = asyncio.run(request())
    expected = ["ddp", "pp", "ddp", "pp"]
    assert [item["distributed_mode"] for item in payload["queued_items"]] == expected
    assert [item["distributed_mode"] for item in json.loads(queue_file.read_text())["items"]] == expected


def test_legacy_queue_and_history_views_leave_saved_modes_intact(tmp_path, app_and_storage):
    _, service, queue_file, history_dir = app_and_storage
    config_text = 'model_family = "moss_image"\npipeline_parallel = true\n'
    runtime_config = tmp_path / "config.runtime.toml"
    runtime_config.write_text(config_text)
    queue_file.parent.mkdir()
    queue_file.write_text(json.dumps({"paused": True, "items": [
        {"id": "old-pp", "distributed_mode": "ddp", "runtime_config_file": str(runtime_config)},
        {"id": "old-off", "distributed_mode": "single", "runtime_config_file": str(runtime_config),
         "extra_args": ["--no-pipeline_parallel"]},
        {"id": "old-unlabeled-pp", "runtime_config_file": str(runtime_config)},
        {"id": "old-missing", "runtime_config_file": str(tmp_path / "removed.toml")},
    ]}))
    original_queue = queue_file.read_bytes()
    service._queue = json.loads(original_queue)
    modes = [item.get("distributed_mode") for item in service.get_queue_snapshot()["items"]]
    assert modes == ["pp", "single", "pp", None]
    assert service._queue["items"][0]["distributed_mode"] == "ddp"
    assert queue_file.read_bytes() == original_queue
    runtime_config.write_text(config_text.replace("true", "false"))
    assert service.get_queue_snapshot()["items"][0]["distributed_mode"] == "ddp"

    training = _write_group_task(history_dir, "queued-pp", config_text=config_text)
    _write_group_task(history_dir, "queued-preprocess", job="preprocess", config_text=config_text)
    _write_group_task(history_dir, "queued-off", config_text=config_text, history_meta={
        "command": ["python", "train.py", "--config_file", str(runtime_config), "--no-pipeline_parallel"],
    })
    summaries = {task["id"]: task for task in service.list_history_tasks(include_archived=True)}
    assert summaries["queued-pp"]["distributed_mode"] == "pp"
    assert "distributed_mode" not in summaries["queued-preprocess"]
    assert "distributed_mode" not in summaries["queued-off"]
    assert "distributed_mode" not in json.loads((training / "meta.json").read_text())


def test_resume_and_retry_recompute_the_frozen_pp_mode(tmp_path, monkeypatch, app_and_storage):
    _, service, queue_file, _ = app_and_storage
    history, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot = history / task_id / "config.snapshot.toml"
    snapshot.write_text(snapshot.read_text() + (
        '\nmodel_family = "anima"\npipeline_parallel = true\nmixed_precision = "fp16"\n'
        'network_train_unet_only = true\n'
    ))
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)

    result = asyncio.run(service.enqueue_resume_from_history_task(task_id, str(state_dir), gpu_whitelist=[0, 1]))
    item = result["item"]
    assert item["distributed_mode"] == "pp"
    assert json.loads(queue_file.read_text())["items"][0]["distributed_mode"] == "pp"
    stale = {**item, "distributed_mode": "ddp"}
    retry = service._clone_queue_item_for_retry(stale)
    assert retry["distributed_mode"] == "pp"
    assert stale["distributed_mode"] == "ddp"
    assert Path(retry["runtime_config_file"]).is_file()


@pytest.mark.parametrize("extra_args, expected", [([], "pp"), (["--no-pipeline_parallel"], None)])
def test_training_launch_writes_pp_history_from_the_launch_config(
    tmp_path, monkeypatch, app_and_storage, extra_args, expected,
):
    app, service, _, _ = app_and_storage
    config = tmp_path / "config.runtime.toml"
    config.write_text(
        'model_family = "anima"\npipeline_parallel = true\nmixed_precision = "fp16"\n'
        'network_train_unet_only = true\ngradient_checkpointing = true\n'
        'use_vae_cache = true\nuse_text_cache = true\n',
    )
    fake_subprocess = AsyncMock(return_value=object())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    for name in ("_read_output", "_monitor_system", "_tail_progress_jsonl"):
        monkeypatch.setattr(service, name, AsyncMock())

    async def request():
        await service.start("lora", "default", extra_args, config_file=str(config), gpu_whitelist=[0, 1])
        async with TestClient(TestServer(app)) as client:
            response = await client.get(f"/api/training/history/{service.current_task_id}")
            assert response.status == 200
            return (await response.json())["task"]

    task = asyncio.run(request())
    fake_subprocess.assert_awaited_once()
    assert task.get("distributed_mode") == expected
    assert json.loads((service.current_task_dir / "meta.json").read_text()).get("distributed_mode") == expected


def test_malformed_pipeline_toggle_does_not_get_queued_as_ddp(tmp_path, app_and_storage):
    _, service, queue_file, _ = app_and_storage
    config = tmp_path / "config.runtime.toml"
    config.write_text('pipeline_parallel = "maybe"\n')

    with pytest.raises(ValueError, match="pipeline_parallel"):
        asyncio.run(service.enqueue_training(
            "lora", "default", config_file=str(config), requires_preprocess=False, start_paused=True,
        ))

    assert service.get_queue_snapshot()["items"] == []
    assert not queue_file.exists()
