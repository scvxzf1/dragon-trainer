"""Queue resume behavior tests split from test_training_resume.py."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

from web.services.training import resume_pipeline_gate as _resume_pipeline_gate

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: queue_resume

def test_queue_resume_clones_runtime_when_enqueued(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None

    result = asyncio.run(svc.enqueue_resume_from_history_task(task_id, str(state_dir)))

    assert result["ok"] is True
    item = result["item"]
    runtime_config = Path(item["runtime_config_file"])
    assert runtime_config.name == "config.runtime.toml"
    assert output_root in runtime_config.parents
    assert runtime_config.exists()
    assert not item["runtime_config_file"].endswith(f"{task_id}/config.snapshot.toml")
    assert item["runtime_info"]["runtime_config_file"] == item["runtime_config_file"]
    assert item["runtime_info"]["training_output_dir"].startswith(str(output_root))
    assert item["extra_args"] == ["--resume", str(state_dir), "--skip_until_initial_step"]
    assert item["resume_info"]["checkpoint"] == str(state_dir)


def test_queue_resume_rejects_pipeline_world_size_mismatch(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8")
        + '\nmodel_family = "krea2_raw"\n'
        + "pipeline_parallel = true\n"
        + "pipeline_parallel_stages = 2\n"
        + "pipeline_parallel_microbatches = 4\n"
        + 'pipeline_parallel_schedule = "1f1b"\n'
        + 'pipeline_parallel_split = "balanced"\n'
        + "network_train_unet_only = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)
    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None
    # t9 UUID guard is orthogonal here; stub it so this test keeps exercising
    # the world-size refusal (index 0 is the GTX 1050 on this host).
    monkeypatch.setattr(
        _resume_pipeline_gate, "ensure_gpu_whitelist_authorized", lambda *_a, **_k: None
    )

    with pytest.raises(ValueError, match="distributed world size"):
        asyncio.run(
            svc.enqueue_resume_from_history_task(
                task_id,
                str(state_dir),
                gpu_whitelist=[0, 1, 2],
            )
        )

    assert svc._queue_items() == []
    assert not output_root.exists()

def test_queue_resume_duration_step_override_appends_steps(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None

    result = asyncio.run(
        svc.enqueue_resume_from_history_task(
            task_id,
            str(state_dir),
            duration_overrides={"max_train_steps": 7},
        )
    )

    assert result["ok"] is True
    item = result["item"]
    runtime_cfg = toml.loads(Path(item["runtime_config_file"]).read_text(encoding="utf-8"))
    assert "max_train_epochs" not in runtime_cfg
    assert runtime_cfg["max_train_steps"] == 49
    assert item["resume_info"]["duration_overrides"] == {
        "mode": "steps",
        "max_train_steps": 7,
        "steps_per_epoch": None,
        "resume_step": 42,
        "append_steps": 7,
        "target_total_steps": 49,
    }
    assert item["resume_info"]["remaining_steps"] == 7

def test_queue_resume_missing_train_state_marks_item_error(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None
    asyncio.run(svc.enqueue_resume_from_history_task(task_id, str(state_dir)))
    item = svc._queue_items()[0]
    (state_dir / "train_state.json").unlink()

    called = False

    async def fake_start_unlocked(*_args, **_kwargs):
        nonlocal called
        called = True

    svc._start_unlocked = fake_start_unlocked

    asyncio.run(svc._dispatch_queue())

    assert called is False
    assert item["state"] == "error"
    assert "续训检查点状态已不存在" in item["message"]
    assert svc._queue_paused is True

def test_queue_resume_incomplete_state_marks_item_error(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None
    asyncio.run(svc.enqueue_resume_from_history_task(task_id, str(state_dir)))
    item = svc._queue_items()[0]
    (state_dir / "scheduler.bin").unlink()

    called = False

    async def fake_start_unlocked(*_args, **_kwargs):
        nonlocal called
        called = True

    svc._start_unlocked = fake_start_unlocked

    asyncio.run(svc._dispatch_queue())

    assert called is False
    assert item["state"] == "error"
    assert "缺少 scheduler.bin" in item["message"]
    assert svc._queue_paused is True


def test_queue_resume_schedule_free_state_without_scheduler_starts(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8")
        + '\noptimizer_type = "ProdigyPlusScheduleFree"\n',
        encoding="utf-8",
    )
    (state_dir / "scheduler.bin").unlink()
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None
    asyncio.run(svc.enqueue_resume_from_history_task(task_id, str(state_dir)))
    item = svc._queue_items()[0]

    called = False

    async def fake_start_unlocked(*_args, **_kwargs):
        nonlocal called
        called = True

    svc._start_unlocked = fake_start_unlocked
    asyncio.run(svc._dispatch_queue())

    assert called is True
    assert item["state"] == "running"
    assert svc._queue_paused is False


def test_queue_resume_duration_override_reports_stage_shift(tmp_path, monkeypatch):
    """追加步数后按新总步重算阶段，并返回 before/after 诊断。"""
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8")
        + "\n".join(
            [
                "stage_schedule_enabled = true",
                "[[stage_schedule]]",
                'name = "low"',
                "subset_index = 0",
                "start_pct = 0.0",
                "end_pct = 0.5",
                "[[stage_schedule]]",
                'name = "high"',
                "subset_index = 1",
                "start_pct = 0.5",
                "end_pct = 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None

    result = asyncio.run(
        svc.enqueue_resume_from_history_task(
            task_id,
            str(state_dir),
            duration_overrides={"max_train_steps": 7},
        )
    )

    assert result["ok"] is True
    resume_info = result["item"]["resume_info"]
    # original 100 steps @42 -> 0.42 (low); new total 49 @42 -> ~0.857 (high)
    assert resume_info["stage_before"]["index"] == 0
    assert resume_info["stage_before"]["name"] == "low"
    assert abs(resume_info["stage_before"]["progress"] - 0.42) < 1e-9
    assert resume_info["stage_after"]["index"] == 1
    assert resume_info["stage_after"]["name"] == "high"
    assert abs(resume_info["stage_after"]["progress"] - (42 / 49)) < 1e-9
    assert "追加步数后阶段边界已按新总步数重算" in str(resume_info.get("warning") or "")
    duration = resume_info["duration_overrides"]
    assert duration["stage_before"] == resume_info["stage_before"]
    assert duration["stage_after"] == resume_info["stage_after"]


def test_queue_resume_without_stage_schedule_skips_stage_diagnosis(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    svc._schedule_queue_dispatch = lambda: None

    result = asyncio.run(
        svc.enqueue_resume_from_history_task(
            task_id,
            str(state_dir),
            duration_overrides={"max_train_steps": 7},
        )
    )

    resume_info = result["item"]["resume_info"]
    assert "stage_before" not in resume_info or resume_info.get("stage_before") is None
    assert "stage_after" not in resume_info or resume_info.get("stage_after") is None
    assert not resume_info.get("warning")
