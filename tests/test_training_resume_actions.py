"""Resume-from-history and continue-from-weight action tests."""

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

# Split: resume_actions


def test_handle_resume_passes_duration_overrides():
    class FakeService:
        async def resume_from_history_task(
            self,
            task_id,
            checkpoint=None,
            *,
            duration_overrides=None,
            gpu_whitelist=None,
        ):
            return {
                "ok": True,
                "task_id": task_id,
                "checkpoint": checkpoint,
                "duration_overrides": duration_overrides,
                "gpu_whitelist": gpu_whitelist,
            }

    req = _FakeJsonRequest(
        {
            "task_id": "task-a",
            "checkpoint": "state-dir",
            "duration_overrides": {"max_train_epochs": 1},
            "gpu_whitelist": [0],
        },
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_resume(req))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["task_id"] == "task-a"
    assert payload["checkpoint"] == "state-dir"
    assert payload["duration_overrides"] == {"max_train_epochs": 1}
    assert payload["gpu_whitelist"] == [0]


def test_resume_from_history_allows_remaining_steps_and_clones_runtime(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8")
        + 'network_weights = "weights/old-hotstart.safetensors"\n'
        + "dim_from_weights = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        config_service,
        "estimate_training_steps",
        lambda *_args, **_kwargs: {"total_steps": 100},
    )

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update({
            "variant": variant,
            "preset": preset,
            "extra_args": extra_args,
            "methods_subdir": methods_subdir,
            **kwargs,
        })

    svc.start = fake_start

    result = asyncio.run(svc.resume_from_history_task(task_id))

    assert result["ok"] is True
    runtime_config = Path(captured["config_file"])
    assert runtime_config.name == "config.runtime.toml"
    assert output_root in runtime_config.parents
    assert runtime_config.exists()
    assert captured["extra_args"][:2] == ["--resume", str(state_dir)]
    assert captured["resume_info"]["target_total_steps"] == 100
    assert captured["resume_info"]["remaining_steps"] == 58
    runtime_cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    assert runtime_cfg["output_dir"].startswith(str(output_root))
    assert runtime_cfg["output_dir"].endswith("/training_output")
    assert runtime_cfg["output_dir"] != str(state_dir.parent)
    assert "network_weights" not in runtime_cfg
    assert "dim_from_weights" not in runtime_cfg


def test_resume_from_history_rejects_pipeline_world_size_mismatch(
    tmp_path, monkeypatch
):
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
    # t9 UUID guard is orthogonal here; stub it so this test keeps
    # exercising the pipeline refusal (index 0 is the GTX 1050 here).
    monkeypatch.setattr(
        _resume_pipeline_gate, "ensure_gpu_whitelist_authorized", lambda *_a, **_k: None
    )
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    svc = TrainingService(web.Application())
    started = False

    async def fake_start(*_args, **_kwargs):
        nonlocal started
        started = True

    svc.start = fake_start

    with pytest.raises(ValueError, match="distributed world size"):
        asyncio.run(
            svc.resume_from_history_task(
                task_id,
                str(state_dir),
                gpu_whitelist=[0],
            )
        )

    assert started is False
    assert not output_root.exists()


def test_resume_from_history_rejects_pipeline_with_unknown_model_family(
    tmp_path, monkeypatch
):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8")
        + '\nmodel_family = "unknown-family"\n'
        + "pipeline_parallel = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    # t9 UUID guard is orthogonal here; stub it so this test keeps
    # exercising the pipeline refusal (index 0 is the GTX 1050 here).
    monkeypatch.setattr(
        _resume_pipeline_gate, "ensure_gpu_whitelist_authorized", lambda *_a, **_k: None
    )
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    svc = TrainingService(web.Application())
    started = False

    async def fake_start(*_args, **_kwargs):
        nonlocal started
        started = True

    svc.start = fake_start

    with pytest.raises(ValueError, match="model_family must be one of"):
        asyncio.run(
            svc.resume_from_history_task(
                task_id,
                str(state_dir),
                gpu_whitelist=[0, 1],
            )
        )

    assert started is False
    assert not output_root.exists()


def test_resume_from_history_duration_epoch_override_appends_steps(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    resized_dir = tmp_path / "post_image_dataset" / "resized"
    for index in range(3):
        (resized_dir / f"image-{index}.png").write_bytes(b"png")
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8") + "max_train_epochs = 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        config_service,
        "estimate_training_steps",
        lambda *_args, **_kwargs: {"total_steps": 42},
    )

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update({
            "variant": variant,
            "preset": preset,
            "extra_args": extra_args,
            "methods_subdir": methods_subdir,
            **kwargs,
        })

    svc.start = fake_start

    result = asyncio.run(
        svc.resume_from_history_task(
            task_id,
            str(state_dir),
            duration_overrides={"max_train_epochs": 1},
        )
    )

    assert result["ok"] is True
    runtime_config = Path(captured["config_file"])
    assert output_root in runtime_config.parents
    runtime_cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    assert "max_train_epochs" not in runtime_cfg
    assert runtime_cfg["max_train_steps"] == 45
    assert captured["resume_info"]["duration_overrides"] == {
        "mode": "epochs",
        "max_train_epochs": 1,
        "steps_per_epoch": 3,
        "resume_step": 42,
        "append_steps": 3,
        "target_total_steps": 45,
    }
    assert captured["resume_info"]["remaining_steps"] == 3


def test_resume_from_history_rejects_other_directory_state(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    other_state = state_dir.parent / "other-checkpoint-state"
    other_state.mkdir()
    (other_state / "train_state.json").write_text(
        json.dumps({"current_epoch": 9, "current_step": 999}),
        encoding="utf-8",
    )
    os.utime(other_state / "train_state.json", (3000.0, 3000.0))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())

    try:
        asyncio.run(svc.resume_from_history_task(task_id, str(other_state)))
    except ValueError as e:
        assert "未找到指定的检查点" in str(e)
    else:
        raise AssertionError("不应允许从同目录其他训练状态续训")


def test_resume_from_history_uses_snapshot_and_resume_args(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update({
            "variant": variant,
            "preset": preset,
            "extra_args": extra_args,
            "methods_subdir": methods_subdir,
            **kwargs,
        })

    svc.start = fake_start

    result = asyncio.run(svc.resume_from_history_task(task_id, str(state_dir)))

    assert result["ok"] is True
    assert captured["variant"] == "demo"
    assert captured["methods_subdir"] == "imported"
    assert captured["extra_args"] == [
        "--resume",
        str(state_dir),
        "--skip_until_initial_step",
    ]
    runtime_config = Path(captured["config_file"])
    assert runtime_config.name == "config.runtime.toml"
    assert output_root in runtime_config.parents
    assert runtime_config.exists()
    assert not captured["config_file"].endswith(f"{task_id}/config.snapshot.toml")
    runtime_cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    assert runtime_cfg["output_dir"].endswith("/training_output")
    assert runtime_cfg["output_dir"] != str(state_dir.parent)
    assert captured["use_runtime_dir"] is False
    assert captured["resume_info"]["checkpoint"] == str(state_dir)
    assert captured["resume_info"]["history_group_key"] == "legacy:imported\u0001demo\u0001default"
    assert captured["resume_info"]["history_group_label"] == "imported / demo / default"


def test_resume_from_history_forwards_gpu_whitelist(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update(kwargs)

    svc.start = fake_start

    result = asyncio.run(
        svc.resume_from_history_task(task_id, str(state_dir), gpu_whitelist=["1", "bad", 2, 2])
    )

    assert result["ok"] is True
    assert captured["gpu_whitelist"] == ["1", "bad", 2, 2]


def test_resume_from_history_requires_config_snapshot(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    (history_dir / task_id / "config.snapshot.toml").unlink()

    svc = TrainingService(web.Application())

    try:
        asyncio.run(svc.resume_from_history_task(task_id, str(state_dir)))
    except ValueError as e:
        assert "配置快照" in str(e)
    else:
        raise AssertionError("缺少配置快照时不应允许续训")


def test_resume_history_meta_inherits_source_group(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    svc._start_history_task(
        job="training",
        variant="demo",
        preset="default",
        methods_subdir="imported",
        output_dir="output/runs/demo-resume-20260523-130000/training_output",
        sample_dir="output/runs/demo-resume-20260523-130000/training_output/sample",
        data_dirs={},
        sample_config={},
        command=["python", "train.py"],
        resume_info={
            "source_task_id": "source-task",
            "history_group_key": "source:configs/imported/demo.toml",
            "history_group_label": "configs/imported/demo.toml",
            "history_source_config_file": "configs/imported/demo.toml",
            "checkpoint_name": "demo-checkpoint-state",
        },
    )

    task = svc.list_history_tasks()[0]

    assert task["history_group_key"] == "source:configs/imported/demo.toml"
    assert task["history_group_label"] == "configs/imported/demo.toml"
    assert task["history_source_config_file"] == "configs/imported/demo.toml"
    assert task["history_run_label"] == "demo-resume-20260523-130000"


def test_resume_from_history_duration_override_reports_stage_shift(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8")
        + "\n".join(
            [
                "stage_schedule_enabled = true",
                "[[stage_schedule]]",
                'name = "mid"',
                "subset_index = 0",
                "start_pct = 0.0",
                "end_pct = 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        config_service,
        "estimate_training_steps",
        lambda *_args, **_kwargs: {"total_steps": 100},
    )

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update(kwargs)
        svc.current_task_id = "new-task"

    svc.start = fake_start

    result = asyncio.run(
        svc.resume_from_history_task(
            task_id,
            str(state_dir),
            duration_overrides={"max_train_steps": 7},
        )
    )

    assert result["ok"] is True
    assert result["stage_before"]["name"] == "mid"
    assert abs(result["stage_before"]["progress"] - 0.42) < 1e-9
    assert abs(result["stage_after"]["progress"] - (42 / 49)) < 1e-9
    assert "追加步数后阶段边界已按新总步数重算" in str(result.get("warning") or "")
    resume_info = captured["resume_info"]
    assert resume_info["stage_before"]["progress"] == result["stage_before"]["progress"]
    assert resume_info["stage_after"]["progress"] == result["stage_after"]["progress"]
    assert resume_info["warning"] == result["warning"]


def test_inspect_continue_lora_weight_detects_supported_variants(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    lora_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo.safetensors",
        kind="LoRA",
    )
    dora_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_dora.safetensors",
        kind="DoRA",
    )
    loha_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_loha.safetensors",
        kind="LoHa",
    )
    lokr_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_lokr.safetensors",
        kind="LoKr",
    )
    glora_path = _write_continue_lora_weight(
        tmp_path / "weights" / "demo_glora.safetensors",
        kind="GLoRA",
    )

    lora_payload = training_service.inspect_continue_lora_weight(
        str(lora_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    dora_payload = training_service.inspect_continue_lora_weight(
        str(dora_path),
        variant="dora",
        preset="default",
        methods_subdir="gui-methods",
    )
    dora_blocked = training_service.inspect_continue_lora_weight(
        str(dora_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    loha_payload = training_service.inspect_continue_lora_weight(
        str(loha_path),
        variant="loha",
        preset="default",
        methods_subdir="gui-methods",
    )
    loha_blocked = training_service.inspect_continue_lora_weight(
        str(loha_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    lokr_payload = training_service.inspect_continue_lora_weight(
        str(lokr_path),
        variant="lokr",
        preset="default",
        methods_subdir="gui-methods",
    )
    lokr_blocked = training_service.inspect_continue_lora_weight(
        str(lokr_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    glora_payload = training_service.inspect_continue_lora_weight(
        str(glora_path),
        variant="glora",
        preset="default",
        methods_subdir="gui-methods",
    )
    glora_blocked = training_service.inspect_continue_lora_weight(
        str(glora_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )

    assert lora_payload["kind"] == "LoRA"
    assert lora_payload["compatible"] is True
    assert dora_payload["kind"] == "DoRA"
    assert dora_payload["compatible"] is True
    assert dora_payload["metadata"]["ss_adapter_variant"] == "dora"
    assert dora_blocked["compatible"] is False
    assert "dora" in dora_blocked["message"].lower()
    assert loha_payload["kind"] == "LoHa"
    assert loha_payload["compatible"] is True
    assert loha_blocked["compatible"] is False
    assert "loha" in loha_blocked["message"].lower()
    assert lokr_payload["kind"] == "LoKr"
    assert lokr_payload["compatible"] is True
    assert lokr_blocked["compatible"] is False
    assert "lokr" in lokr_blocked["message"].lower()
    assert glora_payload["kind"] == "GLoRA"
    assert glora_payload["compatible"] is True
    assert glora_blocked["compatible"] is False
    assert "glora" in glora_blocked["message"].lower()


def test_inspect_continue_lora_weight_rejects_complex_lora_like_weights(
    tmp_path,
    monkeypatch,
):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    plain_lora_tensors = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(4, 8),
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.randn(12, 4),
        "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
    }
    cases = [
        (
            "hydra_keys",
            {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(
                    4, 8
                ),
                "lora_unet_blocks_0_self_attn_q_proj.lora_ups.0.weight": torch.randn(
                    12, 4
                ),
                "lora_unet_blocks_0_self_attn_q_proj.router.weight": torch.randn(2, 4),
            },
            None,
        ),
        (
            "stacked_keys",
            {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down_weight": torch.randn(
                    2, 4, 8
                ),
                "lora_unet_blocks_0_self_attn_q_proj.lora_up_weight": torch.randn(
                    2, 12, 4
                ),
            },
            None,
        ),
        ("hydra_spec", plain_lora_tensors, {"ss_network_spec": "hydra"}),
        (
            "stacked_spec",
            plain_lora_tensors,
            {"ss_network_spec": "stacked_experts_global_fei"},
        ),
        ("chimera_spec", plain_lora_tensors, {"ss_network_spec": "chimera_hydra"}),
        (
            "reft_key",
            {"reft_unet_blocks_0.rotate_layer.weight": torch.randn(4, 4)},
            {"ss_network_spec": "reft"},
        ),
    ]

    for name, tensors, metadata in cases:
        path = _write_continue_lora_weight(
            tmp_path / "weights" / f"{name}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )
        with pytest.raises(
            ValueError, match="未识别为 LoRA、DoRA、LoHa、LoKr 或 GLoRA"
        ):
            training_service.inspect_continue_lora_weight(
                str(path),
                variant="lora",
                preset="default",
                methods_subdir="gui-methods",
            )


def test_inspect_continue_lora_weight_reports_path_errors(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="权重文件不存在"):
        training_service.inspect_continue_lora_weight(
            str(tmp_path / "weights" / "missing.safetensors"),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    txt_path = tmp_path / "weights" / "demo.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("not a safetensors file", encoding="utf-8")
    with pytest.raises(ValueError, match="只支持 .safetensors"):
        training_service.inspect_continue_lora_weight(
            str(txt_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    directory_path = tmp_path / "weights" / "directory.safetensors"
    directory_path.mkdir()
    with pytest.raises(ValueError, match="权重路径不是文件"):
        training_service.inspect_continue_lora_weight(
            str(directory_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    unreadable_path = _write_continue_lora_weight(
        tmp_path / "weights" / "unreadable.safetensors"
    )
    real_access = os.access

    def fake_access(path, mode):
        if Path(path) == unreadable_path and mode == os.R_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(training_service.os, "access", fake_access)
    with pytest.raises(ValueError, match="权重文件不可读取"):
        training_service.inspect_continue_lora_weight(
            str(unreadable_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )
