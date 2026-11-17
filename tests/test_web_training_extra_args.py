"""Web preflight and workers must consume the same explicit training argv."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from library.training.pp.contract import PipelineRuntimeConfig
from library.training.pp.errors import PipelineParallelError
from library.training.pp.trainer_admission import require_family_trainer_admission
from tests.web_config_test_support import _write_selected_checkpoint_preflight_config
from web.routes import training as routes
from web.services import config_service
from web.services.training import launcher_runtime
from web.services.training.extra_args import merge_training_extra_args


def test_overrides_keep_explicit_defaults_and_last_occurrence_without_mutation():
    cfg = {"blocks_to_swap": 12, "pipeline_parallel": True, "mixed_precision": "bf16"}
    result = merge_training_extra_args(cfg, [
        "--blocks_to_swap", "0", "--no-pipeline_parallel",
        "--mixed_precision", "no", "--mixed_precision=fp16",
    ])
    assert result == {"blocks_to_swap": 0, "pipeline_parallel": False, "mixed_precision": "fp16"}
    assert cfg["blocks_to_swap"] == 12 and cfg["pipeline_parallel"] is True


@pytest.mark.parametrize("argv", [
    "--mixed_precision fp16", {"mixed_precision": "fp16"}, [1],
    ["--mixed_precision"], ["--mixed_precision", "invalid"],
    ["--unknown-training-option"], ["--mixed_prec", "fp16"],
    ["--config_file", "other.toml"], ["--config_file=other.toml"],
    ["--config_f", "other.toml"], ["--method", "lora"], ["--preset=default"],
    ["--methods_subdir", "imported"], ["--help"], ["-h"],
    ["--print_config"], ["--output_config"], ["--config_snapshot"],
    ["--config_strict"],
    ["--print-config"], ["--config-snapshot"], ["--no-config-snapshot"],
    ["--config-strict"],
])
def test_invalid_or_web_owned_overrides_raise_without_system_exit(argv):
    with pytest.raises(ValueError):
        merge_training_extra_args({}, argv)


def test_accelerate_uses_the_same_last_precision_as_worker(monkeypatch):
    monkeypatch.setattr(
        launcher_runtime, "_load_config_file_config", lambda _: {"mixed_precision": "bf16"}
    )
    argv = ["--mixed_precision", "no", "--mixed_precision=fp16"]
    assert launcher_runtime._accelerate_mixed_precision_for_training("fixture.toml", argv) == "fp16"


_PP = [
    'model_family = "krea2_raw"', "pipeline_parallel = true",
    "pipeline_parallel_stages = 2", "pipeline_parallel_microbatches = 4",
    "pipeline_parallel_blocks_to_swap = 4", "train_batch_size = 4",
    'mixed_precision = "fp16"', "torch_compile = false", "blocks_to_swap = 0",
    "network_train_unet_only = true", 'selective_checkpoint = "off"',
]


@pytest.mark.parametrize("extra", [
    ["--mixed_precision", "bf16"],
    ["--save_every_n_epochs", "1"],
    ["--pipeline_parallel_microbatches", "8"],
])
def test_real_preflight_rejects_overrides_the_trainer_refuses(tmp_path, monkeypatch, extra):
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, _PP)
    result = config_service.preflight_training_config(
        "lora", "default", "imported", config_file="configs/imported/selected.toml",
        world_size=2, extra_args=extra,
    )
    assert result["ok"] is False
    assert any(item["key"].startswith("pipeline_parallel") for item in result["errors"])


def test_override_can_fix_a_saved_pp_precision_before_preflight(tmp_path, monkeypatch):
    lines = [line for line in _PP if not line.startswith("mixed_precision")]
    _write_selected_checkpoint_preflight_config(
        tmp_path, monkeypatch, [*lines, 'mixed_precision = "bf16"']
    )
    result = config_service.preflight_training_config(
        "lora", "default", "imported", config_file="configs/imported/selected.toml",
        world_size=2, extra_args=["--mixed_precision=fp16"],
    )
    assert result["ok"] is True, result["errors"]


class _Request:
    def __init__(self, data):
        self.data = data
        self.app = {"training_service": SimpleNamespace()}

    async def json(self):
        return self.data


@pytest.mark.parametrize("handler", [
    routes.handle_preflight, routes.handle_start, routes.handle_queue_start,
    routes.handle_queue_batch_start,
])
def test_all_request_entries_preflight_the_actual_argv(tmp_path, monkeypatch, handler):
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, _PP)
    monkeypatch.setattr(routes, "preflight_training_config", config_service.preflight_training_config)
    monkeypatch.setattr(routes, "_preflight_world_size", lambda _: 2)
    item = {
        "variant": "lora", "preset": "default", "methods_subdir": "imported",
        "config_file": "configs/imported/selected.toml",
        "extra_args": ["--mixed_precision", "bf16"],
        "confirmed": True, "confirm_preprocess": True,
    }
    body = {"items": [item]} if handler is routes.handle_queue_batch_start else item
    response = asyncio.run(handler(_Request(body)))
    payload = json.loads(response.text)
    assert payload["ok"] is False
    assert response.status == (200 if handler is routes.handle_preflight else 400)
    if handler is routes.handle_queue_batch_start:
        assert payload["failed_index"] == 0


def test_runtime_materializes_training_overrides_before_dataset_generation(tmp_path, monkeypatch):
    from tests.training_resume_test_support import (
        _patch_runtime_service_paths, _write_runtime_config_tree, training_service,
    )

    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    runtime = training_service._prepare_web_runtime_config(
        "522", "default", "imported", source_config_file="configs/imported/522.toml",
        extra_args=["--train_batch_size", "4", "--mixed_precision=fp16"],
    )
    cfg = toml.loads((tmp_path / runtime["runtime_config_file"]).read_text())
    datasets = toml.loads((tmp_path / runtime["dataset_config_file"]).read_text())
    assert cfg["train_batch_size"] == 4 and cfg["mixed_precision"] == "fp16"
    assert all(dataset["batch_size"] == 4 for dataset in datasets["datasets"])


def test_invalid_runtime_override_creates_no_run_directory(tmp_path, monkeypatch):
    from tests.training_resume_test_support import (
        _patch_runtime_service_paths, _write_runtime_config_tree, training_service,
    )

    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    before = set((tmp_path / "output").rglob("*")) if (tmp_path / "output").exists() else set()
    with pytest.raises(ValueError):
        training_service._prepare_web_runtime_config(
            "522", "default", "imported", source_config_file="configs/imported/522.toml",
            extra_args=["--config_file", "other.toml"],
        )
    after = set((tmp_path / "output").rglob("*")) if (tmp_path / "output").exists() else set()
    assert after == before


def test_krea_cli_admission_checks_batch_before_model_loading():
    cfg = {
        "model_family": "krea2_raw", "pipeline_parallel": True,
        "mixed_precision": "fp16", "train_batch_size": 2,
        "pipeline_parallel_microbatches": 4, "network_train_unet_only": True,
    }
    with pytest.raises(PipelineParallelError, match="pp_krea2_batch_not_divisible"):
        require_family_trainer_admission(cfg, world_size=2)


def test_runtime_config_copy_keeps_swap_and_accepts_a_swap_override():
    cfg = PipelineRuntimeConfig(blocks_to_swap=4)
    assert cfg.with_overrides(mixed_precision="fp16").blocks_to_swap == 4
    assert cfg.with_overrides(blocks_to_swap=0).blocks_to_swap == 0


@pytest.mark.parametrize("equal_form", [False, True])
def test_preprocess_and_pending_training_use_the_materialized_dataset(
    tmp_path, monkeypatch, equal_form,
):
    from aiohttp import web
    from tests.training_resume_test_support import (
        _patch_runtime_service_paths, _write_runtime_config_tree, training_service,
    )

    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    source = "configs/datasets/522.toml"
    argv = [f"--dataset_config={source}"] if equal_form else ["--dataset_config", source]
    argv += ["--mixed_precision=fp16"]
    svc = training_service.TrainingService(web.Application())
    captured = {}

    async def launch(cmd, env, **kwargs):
        captured["cmd"] = cmd
        captured["config_file"] = kwargs["config_file"]

    svc._launch_job = launch
    asyncio.run(svc.start_preprocess(
        "522", "default", "imported", argv, train_after=True,
        config_file="configs/imported/522.toml",
    ))
    pending = svc._pending_train_after_preprocess
    assert pending["extra_args"] == ["--mixed_precision=fp16"]
    assert not any(arg.startswith("--dataset_config") for arg in captured["cmd"])
    cfg = toml.loads((tmp_path / captured["config_file"]).read_text())
    assert cfg["dataset_config"].endswith("/dataset.runtime.toml")
    assert cfg["dataset_config"] != source
    assert cfg["mixed_precision"] == "fp16"
    rechecked = {}

    def preflight(*args, **kwargs):
        rechecked.update(kwargs)
        return {"ok": True}

    async def start(*args, **kwargs):
        captured["training_args"] = args

    monkeypatch.setattr(training_service, "preflight_training_config", preflight)
    svc.start = start
    asyncio.run(svc._start_pending_training(pending))
    assert rechecked["extra_args"] == ["--mixed_precision=fp16"]
    assert rechecked["config_file"] == captured["config_file"]
    assert captured["training_args"][2] == ["--mixed_precision=fp16"]


def test_existing_runtime_direct_start_keeps_an_explicit_dataset_override(tmp_path, monkeypatch):
    from aiohttp import web
    from tests.training_resume_test_support import (
        _patch_runtime_service_paths, _write_runtime_config_tree, training_service,
    )

    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("ANIMA_ACCELERATE_LAUNCH", raising=False)
    monkeypatch.delenv("ANIMA_ACCELERATE_NUM_PROCESSES", raising=False)
    svc = training_service.TrainingService(web.Application())
    captured = {}

    async def launch(cmd, env, **kwargs):
        captured["cmd"] = cmd

    svc._launch_job = launch
    argv = ["--dataset_config=configs/datasets/522.toml", "--mixed_precision=fp16"]
    asyncio.run(svc.start(
        "522", "default", argv, "imported",
        config_file="configs/imported/522.toml", use_runtime_dir=False,
    ))
    assert "--dataset_config=configs/datasets/522.toml" in captured["cmd"]


def test_queue_keeps_the_generated_dataset_and_other_explicit_argv(tmp_path, monkeypatch):
    from tests.training_resume_test_support import (
        _patch_runtime_service_paths, _write_runtime_config_tree,
    )
    from web.services.training import queue_enqueue

    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    items = []

    async def broadcast():
        pass

    svc = SimpleNamespace(
        _queue_paused=True, _queue={"paused": True}, _queue_items=lambda: items,
        _compact_queue=lambda: None, _save_queue=lambda: None,
        _broadcast_queue=broadcast, get_queue_snapshot=lambda: {},
    )
    response = asyncio.run(queue_enqueue.enqueue_training(
        svc, "522", "default", "imported", config_file="configs/imported/522.toml",
        extra_args=[
            "--dataset_config", "ignored-first.toml",
            "--dataset_config=configs/datasets/522.toml", "--mixed_precision=fp16",
        ],
        start_paused=True,
    ))
    item = response["item"]
    assert item["extra_args"] == ["--mixed_precision=fp16"]
    cfg = toml.loads((tmp_path / item["runtime_config_file"]).read_text())
    assert cfg["dataset_config"].endswith("/dataset.runtime.toml")
    assert cfg["mixed_precision"] == "fp16"
