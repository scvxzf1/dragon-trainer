from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web
import pytest
import toml

from web.routes import training as training_routes
from web.services import config_service, training_service
from web.services.training_service import TrainingService


class _FakeJsonRequest:
    def __init__(self, payload: dict, app: dict | None = None, match_info: dict | None = None):
        self._payload = payload
        self.app = app or {}
        self.match_info = match_info or {}
        self.query = {}

    async def json(self):
        return self._payload


def _patch_queue_paths(tmp_path: Path, monkeypatch):
    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(training_service, "QUEUE_DIR", queue_dir)
    monkeypatch.setattr(training_service, "QUEUE_FILE", queue_dir / "queue.json")
    monkeypatch.setattr(training_service, "HISTORY_DIR", tmp_path / "history")
    return queue_dir


def _runtime_payload(tmp_path: Path, name: str = "demo") -> dict:
    run_dir = tmp_path / "runs" / name
    runtime_config = run_dir / "config.runtime.toml"
    for path in (
        run_dir / "model_cache",
        run_dir / "dataset_cache",
        run_dir / "training_output",
        run_dir / "training_output" / "sample",
        run_dir / "model_cache" / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        "\n".join([
            f'output_dir = "{(run_dir / "training_output").as_posix()}"',
            f'logging_dir = "{(run_dir / "model_cache" / "logs").as_posix()}"',
            'source_image_dir = "image_dataset/a"',
            'resized_image_dir = "resized/a"',
            'lora_cache_dir = "cache/a"',
        ]),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "runtime_config_file": str(runtime_config),
        "original_config_file": str(run_dir / "config.original.toml"),
        "dataset_config_file": str(run_dir / "dataset.runtime.toml"),
        "output_dir": str(run_dir / "training_output"),
        "sample_dir": str(run_dir / "training_output" / "sample"),
        "model_cache_dir": str(run_dir / "model_cache"),
        "dataset_cache_dir": str(run_dir / "dataset_cache"),
        "training_output_dir": str(run_dir / "training_output"),
        "logs_dir": str(run_dir / "model_cache" / "logs"),
        "history_source_config_file": "configs/imported/source.toml",
        "sample_config": {},
        "data_dirs": {},
    }


def _patch_runtime_config_paths(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    output_root = tmp_path / "output" / "runs"
    monkeypatch.setattr(config_service, "ROOT", tmp_path)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(config_service, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(training_service, "ROOT", tmp_path)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(
        training_service,
        "_display_settings_path",
        lambda path: _display_under_root(Path(path), tmp_path),
    )
    return configs, output_root


def _patch_external_runtime_config_paths(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    configs = tmp_path / "external-configs"
    output_root = root / "output" / "runs"
    root.mkdir()
    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(config_service, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(training_service, "ROOT", root)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(
        training_service,
        "_display_settings_path",
        lambda path: _display_under_root(Path(path), root),
    )
    return root, configs, output_root


def _display_under_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def test_prepare_web_runtime_config_freezes_frontend_editable_parameters(tmp_path, monkeypatch):
    configs, output_root = _patch_runtime_config_paths(tmp_path, monkeypatch)
    for rel in ("gui-methods", "imported", "datasets"):
        (configs / rel).mkdir(parents=True)
    (configs / "base.toml").write_text(
        "\n".join([
            'source_image_dir = "image_dataset/default"',
            'resized_image_dir = "post_image_dataset/default_resized"',
            'lora_cache_dir = "post_image_dataset/default_lora"',
            'pretrained_model_name_or_path = "models/anima.safetensors"',
            'qwen3 = "models/qwen.safetensors"',
            'vae = "models/vae.safetensors"',
        ]),
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text(
        "[default]\nblocks_to_swap = 4\nblock_swap_transfer_dtype = \"bf16\"\nblock_swap_restore_mode = \"foreach\"\n",
        encoding="utf-8",
    )
    (configs / "gui-methods" / "lora.toml").write_text(
        "\n".join([
            'output_name = "base_lora"',
            "learning_rate = 0.0001",
            "train_batch_size = 1",
            'network_args = ["lokr_factor_group_size=8"]',
        ]),
        encoding="utf-8",
    )
    (configs / "datasets" / "ui-selected.toml").write_text(
        "\n".join([
            "[[datasets]]",
            "resolution = 768",
            "bucket_reso_steps = 32",
            "",
            "[[datasets.subsets]]",
            'image_dir = "post_image_dataset/selected_resized"',
            'cache_dir = "post_image_dataset/selected_lora"',
            "num_repeats = 3",
            'custom_attributes = { source_dir = "image_dataset/selected" }',
        ]),
        encoding="utf-8",
    )
    source_config = configs / "imported" / "ui-selected.toml"
    source_config.write_text(
        "\n".join([
            'output_name = "ui_selected"',
            'dataset_config = "configs/datasets/ui-selected.toml"',
            "train_batch_size = 3",
            "gradient_accumulation_steps = 2",
            "sample_every_n_epochs = 2",
            'sample_prompts = "configs/sample-prompts/imported/ui-selected.txt"',
            "use_lokr = true",
            "lokr_factor = 16",
            'network_args = ["lokr_factor_group_size=12", "lokr_project_chunk_bytes=1048576"]',
            "blocks_to_swap = 23",
            'block_swap_transfer_dtype = "fp8_e4m3"',
            'block_swap_restore_mode = "slab"',
            'memory_probe_jsonl = "auto"',
            'block_swap_profile_jsonl = "auto"',
        ]),
        encoding="utf-8",
    )

    runtime = training_service._prepare_web_runtime_config(
        "lora",
        "default",
        "gui-methods",
        source_config_file="configs/imported/ui-selected.toml",
    )

    runtime_cfg = toml.loads((tmp_path / runtime["runtime_config_file"]).read_text(encoding="utf-8"))
    dataset_cfg = toml.loads((tmp_path / runtime["dataset_config_file"]).read_text(encoding="utf-8"))
    assert Path(runtime["runtime_config_file"]).name == "config.runtime.toml"
    assert Path(runtime["original_config_file"]).name == "config.original.toml"
    assert runtime["run_dir"].startswith(output_root.relative_to(tmp_path).as_posix())

    assert runtime_cfg["output_name"] == "ui_selected"
    assert runtime_cfg["train_batch_size"] == 3
    assert runtime_cfg["gradient_accumulation_steps"] == 2
    assert runtime_cfg["sample_every_n_epochs"] == 2
    assert runtime_cfg["sample_prompts"] == "configs/sample-prompts/imported/ui-selected.txt"
    assert runtime_cfg["use_lokr"] is True
    assert runtime_cfg["lokr_factor"] == 16
    assert runtime_cfg["network_args"] == [
        "lokr_factor_group_size=12",
        "lokr_project_chunk_bytes=1048576",
    ]
    assert runtime_cfg["blocks_to_swap"] == 23
    assert runtime_cfg["block_swap_transfer_dtype"] == "fp8_e4m3"
    assert runtime_cfg["block_swap_restore_mode"] == "slab"
    assert runtime_cfg["memory_probe_jsonl"] == "auto"
    assert runtime_cfg["block_swap_profile_jsonl"] == "auto"
    assert runtime_cfg["dataset_config"] == runtime["dataset_config_file"]
    assert runtime_cfg["output_dir"] == runtime["training_output_dir"]
    assert runtime_cfg["logging_dir"] == runtime["logs_dir"]

    dataset = dataset_cfg["datasets"][0]
    subset = dataset["subsets"][0]
    assert dataset["batch_size"] == 3
    assert dataset["validation_split"] == 0
    assert "validation_split_num" not in dataset
    assert subset["num_repeats"] == 3
    assert subset["image_dir"].endswith("/dataset_cache/dataset-01/resized")
    assert subset["cache_dir"].endswith("/dataset_cache/dataset-01/lora")
    assert subset["custom_attributes"]["source_dir"] == "image_dataset/selected"
    assert subset["custom_attributes"]["preprocess"]["resolution"] == 768
    assert subset["custom_attributes"]["preprocess"]["bucket_reso_steps"] == 32


def test_prepare_web_runtime_config_resolves_external_configs_root_source(tmp_path, monkeypatch):
    root, configs, output_root = _patch_external_runtime_config_paths(tmp_path, monkeypatch)
    for rel in ("gui-methods", "imported", "datasets"):
        (configs / rel).mkdir(parents=True)
    (configs / "base.toml").write_text('source_image_dir = "image_dataset/default"\n', encoding="utf-8")
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "gui-methods" / "lora.toml").write_text('output_name = "base_lora"\n', encoding="utf-8")
    (configs / "imported" / "rokkotsu_goddess_528_tag.toml").write_text(
        'output_name = "rokkotsu_goddess_528_tag"\n',
        encoding="utf-8",
    )

    runtime = training_service._prepare_web_runtime_config(
        "lora",
        "default",
        "gui-methods",
        source_config_file="configs/imported/rokkotsu_goddess_528_tag.toml",
    )

    runtime_cfg_path = root / runtime["runtime_config_file"]
    runtime_cfg = toml.loads(runtime_cfg_path.read_text(encoding="utf-8"))
    assert runtime["run_dir"].startswith(output_root.relative_to(root).as_posix())
    assert runtime["history_source_config_file"] == "configs/imported/rokkotsu_goddess_528_tag.toml"
    assert runtime_cfg["output_name"] == "rokkotsu_goddess_528_tag"


def test_prepare_web_runtime_config_preserves_validation_split_controls(tmp_path, monkeypatch):
    configs, _output_root = _patch_runtime_config_paths(tmp_path, monkeypatch)
    for rel in ("gui-methods", "imported", "datasets"):
        (configs / rel).mkdir(parents=True)
    (configs / "base.toml").write_text(
        "\n".join([
            'source_image_dir = "image_dataset/default"',
            'resized_image_dir = "post_image_dataset/default_resized"',
            'lora_cache_dir = "post_image_dataset/default_lora"',
            'pretrained_model_name_or_path = "models/anima.safetensors"',
            'qwen3 = "models/qwen.safetensors"',
            'vae = "models/vae.safetensors"',
        ]),
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "gui-methods" / "lora.toml").write_text(
        'output_name = "base_lora"\ntrain_batch_size = 1\n',
        encoding="utf-8",
    )
    (configs / "datasets" / "validation-controls.toml").write_text(
        "\n".join([
            "[[datasets]]",
            "validation_split = 0.125",
            "validation_seed = 11",
            "",
            "[[datasets.subsets]]",
            'image_dir = "post_image_dataset/ratio_resized"',
            'cache_dir = "post_image_dataset/ratio_lora"',
            "num_repeats = 1",
            'custom_attributes = { source_dir = "image_dataset/ratio" }',
            "",
            "[[datasets]]",
            "validation_split = 0.5",
            "validation_split_num = 3",
            "validation_seed = 22",
            "",
            "[[datasets.subsets]]",
            'image_dir = "post_image_dataset/fixed_resized"',
            'cache_dir = "post_image_dataset/fixed_lora"',
            "num_repeats = 1",
            'custom_attributes = { source_dir = "image_dataset/fixed" }',
        ]),
        encoding="utf-8",
    )
    (configs / "imported" / "validation-controls.toml").write_text(
        "\n".join([
            'output_name = "validation_controls"',
            'dataset_config = "configs/datasets/validation-controls.toml"',
        ]),
        encoding="utf-8",
    )

    runtime = training_service._prepare_web_runtime_config(
        "lora",
        "default",
        "gui-methods",
        source_config_file="configs/imported/validation-controls.toml",
    )

    dataset_cfg = toml.loads((tmp_path / runtime["dataset_config_file"]).read_text(encoding="utf-8"))
    ratio_dataset = dataset_cfg["datasets"][0]
    fixed_dataset = dataset_cfg["datasets"][1]
    assert ratio_dataset["validation_split"] == 0.125
    assert "validation_split_num" not in ratio_dataset
    assert ratio_dataset["validation_seed"] == 11
    assert fixed_dataset["validation_split"] == 0.5
    assert fixed_dataset["validation_split_num"] == 3
    assert fixed_dataset["validation_seed"] == 22


def test_prepare_web_runtime_config_fills_blank_model_paths_from_global_settings(tmp_path, monkeypatch):
    configs, _output_root = _patch_runtime_config_paths(tmp_path, monkeypatch)
    for rel in ("gui-methods", "imported"):
        (configs / rel).mkdir(parents=True)
    (configs / "web-ui-settings.toml").write_text(
        "\n".join([
            "[global]",
            'pretrained_model_name_or_path = "models/global-anima.safetensors"',
            'qwen3 = "models/global-qwen.safetensors"',
            'vae = "models/global-vae.safetensors"',
        ]),
        encoding="utf-8",
    )
    (configs / "base.toml").write_text(
        "\n".join([
            'source_image_dir = "image_dataset/default"',
            'resized_image_dir = "post_image_dataset/default_resized"',
            'lora_cache_dir = "post_image_dataset/default_lora"',
            'pretrained_model_name_or_path = "models/base-anima.safetensors"',
            'qwen3 = "models/base-qwen.safetensors"',
            'vae = "models/base-vae.safetensors"',
        ]),
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "gui-methods" / "lora.toml").write_text('output_name = "base_lora"\n', encoding="utf-8")
    source_config = configs / "imported" / "blank-model-paths.toml"
    source_config.write_text(
        "\n".join([
            'source_image_dir = "image_dataset/selected"',
            'pretrained_model_name_or_path = ""',
            'qwen3 = ""',
            'vae = ""',
        ]),
        encoding="utf-8",
    )

    runtime = training_service._prepare_web_runtime_config(
        "lora",
        "default",
        "gui-methods",
        source_config_file="configs/imported/blank-model-paths.toml",
    )

    runtime_cfg = toml.loads((tmp_path / runtime["runtime_config_file"]).read_text(encoding="utf-8"))
    original_cfg = toml.loads((tmp_path / runtime["original_config_file"]).read_text(encoding="utf-8"))
    assert runtime_cfg["pretrained_model_name_or_path"] == "models/global-anima.safetensors"
    assert runtime_cfg["qwen3"] == "models/global-qwen.safetensors"
    assert runtime_cfg["vae"] == "models/global-vae.safetensors"
    assert original_cfg["pretrained_model_name_or_path"] == ""
    assert original_cfg["qwen3"] == ""
    assert original_cfg["vae"] == ""


def test_enqueue_training_freezes_runtime_config_while_running(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "_prepare_web_runtime_config", lambda *args, **kwargs: runtime)
    svc = TrainingService(web.Application())
    svc.status = "running"

    payload = asyncio.run(svc.enqueue_training(
        "demo",
        "default",
        "imported",
        config_file="configs/imported/source.toml",
        requires_preprocess=True,
        gpu_whitelist=["1", "bad", 1],
    ))

    assert payload["ok"] is True
    item = payload["item"]
    assert item["state"] == "queued"
    assert item["runtime_config_file"] == runtime["runtime_config_file"]
    assert item["source_config_file"] == "configs/imported/source.toml"
    assert item["gpu_whitelist"] == [1]


def test_enqueue_training_can_pause_queue_for_manual_start(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "_prepare_web_runtime_config", lambda *args, **kwargs: runtime)
    svc = TrainingService(web.Application())
    called = {"dispatch": False}

    monkeypatch.setattr(svc, "_schedule_queue_dispatch", lambda: called.update(dispatch=True))

    payload = asyncio.run(svc.enqueue_training(
        "demo",
        "default",
        "imported",
        config_file="configs/imported/source.toml",
        requires_preprocess=True,
        start_paused=True,
    ))

    assert payload["ok"] is True
    assert payload["paused"] is True
    assert payload["message"] == "已加入训练队列，队列已暂停"
    assert payload["item"]["state"] == "queued"
    assert called["dispatch"] is False


def test_set_queue_settings_unpauses_and_dispatches_waiting_item(tmp_path, monkeypatch):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": True,
        "failure_policy": "pause",
        "items": [{"id": "q1", "state": "queued"}],
    }
    svc._queue_paused = True
    called = {"broadcast": 0, "dispatch": 0}

    async def fake_broadcast_queue():
        called["broadcast"] += 1

    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast_queue)
    monkeypatch.setattr(
        svc,
        "_schedule_queue_dispatch",
        lambda: called.update(dispatch=called["dispatch"] + 1),
    )

    snapshot = asyncio.run(svc.set_queue_settings(paused=False, failure_policy="continue"))

    assert snapshot["paused"] is False
    assert snapshot["failure_policy"] == "continue"
    assert snapshot["summary"]["queued"] == 1
    assert called == {"broadcast": 1, "dispatch": 1}
    saved = json.loads((queue_dir / "queue.json").read_text(encoding="utf-8"))
    assert saved["paused"] is False
    assert saved["failure_policy"] == "continue"


def test_enqueue_training_batch_stops_on_first_failed_runtime_freeze(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    first_runtime = _runtime_payload(tmp_path, "first")
    calls = []

    def fake_prepare(variant, preset, methods_subdir, *, source_config_file):
        calls.append((variant, preset, methods_subdir, source_config_file))
        if source_config_file == "configs/imported/broken.toml":
            raise FileNotFoundError("训练配置不存在: configs/imported/broken.toml")
        return first_runtime

    monkeypatch.setattr(training_service, "_prepare_web_runtime_config", fake_prepare)
    svc = TrainingService(web.Application())

    payload = asyncio.run(svc.enqueue_training_batch(
        [
            {
                "variant": "first",
                "preset": "default",
                "methods_subdir": "imported",
                "config_file": "configs/imported/first.toml",
                "requires_preprocess": True,
            },
            {
                "variant": "broken",
                "preset": "default",
                "methods_subdir": "imported",
                "config_file": "configs/imported/broken.toml",
                "label": "损坏配置",
                "requires_preprocess": True,
            },
        ],
        start_paused=True,
    ))

    assert payload["ok"] is False
    assert payload["queued_count"] == 1
    assert payload["requested_count"] == 2
    assert payload["failures"][0]["index"] == 1
    assert payload["failures"][0]["label"] == "损坏配置"
    assert "broken.toml" in payload["failures"][0]["error"]
    assert payload["paused"] is True
    assert len(payload["items"]) == 1
    assert payload["items"][0]["runtime_config_file"] == first_runtime["runtime_config_file"]
    assert calls == [
        ("first", "default", "imported", "configs/imported/first.toml"),
        ("broken", "default", "imported", "configs/imported/broken.toml"),
    ]


def test_queue_move_and_cancel_waiting_items(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": True,
        "items": [
            {"id": "a", "state": "queued"},
            {"id": "b", "state": "queued"},
            {"id": "c", "state": "done"},
        ],
    }
    svc._queue_paused = True

    asyncio.run(svc.move_queue_item("b", "up"))
    assert [item["id"] for item in svc.get_queue_snapshot()["items"][:2]] == ["b", "a"]

    asyncio.run(svc.cancel_queue_item("a"))
    item = next(item for item in svc.get_queue_snapshot()["items"] if item["id"] == "a")
    assert item["state"] == "canceled"


def test_queue_startup_repairs_stale_running_item(tmp_path, monkeypatch):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    queue_dir.mkdir(parents=True)
    (queue_dir / "queue.json").write_text(
        json.dumps({
            "paused": False,
            "items": [
                {"id": "old", "state": "running"},
                {"id": "next", "state": "queued"},
            ],
        }),
        encoding="utf-8",
    )

    svc = TrainingService(web.Application())
    items = svc.get_queue_snapshot()["items"]
    assert items[0]["state"] == "error"
    assert items[1]["state"] == "queued"
    assert svc.get_queue_snapshot()["paused"] is True


def test_get_queue_snapshot_normalizes_dirty_queue_state(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue_failure_policy = "continue"
    svc._queue = {
        "paused": "",
        "failure_policy": "invalid",
        "items": [
            {"id": "q1", "state": "queued", "attempt": 0},
            "not-a-dict",
            {"id": "done", "state": "done", "attempt": "3"},
        ],
    }

    snapshot = svc.get_queue_snapshot()

    assert snapshot["paused"] is False
    assert snapshot["failure_policy"] == "pause"
    assert snapshot["summary"] == {
        "total": 2,
        "queued": 1,
        "running": 0,
        "done": 1,
        "error": 0,
        "canceled": 0,
    }
    assert [(item["id"], item["attempt"], item["retry_of"]) for item in snapshot["items"]] == [
        ("q1", 1, ""),
        ("done", 3, ""),
    ]
    assert svc._queue["items"] == snapshot["items"]


def test_compact_queue_preserves_waiting_and_running_over_limit(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(training_service, "MAX_QUEUE_ITEMS", 2)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "items": [
            {"id": "done-old", "state": "done"},
            {"id": "q1", "state": "queued"},
            {"id": "error-old", "state": "error"},
            {"id": "running", "state": "running"},
            {"id": "q2", "state": "queued"},
            {"id": "done-new", "state": "done"},
        ],
    }

    svc._compact_queue()

    assert [(item["id"], item["state"]) for item in svc._queue["items"]] == [
        ("q1", "queued"),
        ("running", "running"),
        ("q2", "queued"),
    ]


def test_attach_history_task_to_queue_item_deduplicates_and_normalizes_ids(
    tmp_path,
    monkeypatch,
):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "items": [
            {"id": "q1", "state": "running", "history_task_ids": "bad"},
            {"id": "q2", "state": "queued", "history_task_ids": ["hist-a"]},
        ],
    }

    svc._attach_history_task_to_queue_item("q1", "hist-a")
    svc._attach_history_task_to_queue_item("q1", "hist-a")
    svc._attach_history_task_to_queue_item("q2", "hist-a")
    svc._attach_history_task_to_queue_item("missing", "hist-b")
    svc._attach_history_task_to_queue_item("q2", "")

    items = {item["id"]: item for item in svc._queue["items"]}
    assert items["q1"]["history_task_ids"] == ["hist-a"]
    assert items["q1"]["message"] == "正在运行"
    assert items["q2"]["history_task_ids"] == ["hist-a"]
    assert items["q2"]["message"] == "正在运行"


def test_queue_startup_dispatches_when_unpaused_and_clean(tmp_path, monkeypatch):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    queue_dir.mkdir(parents=True)
    (queue_dir / "queue.json").write_text(
        json.dumps({"paused": False, "items": [{"id": "next", "state": "queued"}]}),
        encoding="utf-8",
    )
    svc = TrainingService(web.Application())
    called = {"dispatch": False}

    async def fake_dispatch():
        called["dispatch"] = True

    monkeypatch.setattr(svc, "_dispatch_queue", fake_dispatch)

    async def run():
        await svc.start_queue_on_startup()
        await asyncio.sleep(0)

    asyncio.run(run())

    assert called["dispatch"] is True


def test_queue_launch_guard_blocks_manual_start_during_startup_window(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "queued"}]}
    svc._queue_paused = False
    checked = {"manual_start_rejected": False}
    launched = {"item_id": ""}

    async def fake_broadcast_queue():
        if checked["manual_start_rejected"]:
            return
        with pytest.raises(RuntimeError, match="已有任务在运行中"):
            await svc.start("manual", "default")
        checked["manual_start_rejected"] = True

    async def fake_start_queue_item(item):
        launched["item_id"] = item["id"]

    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast_queue)
    monkeypatch.setattr(svc, "_start_queue_item", fake_start_queue_item)

    asyncio.run(svc._dispatch_queue())

    assert checked["manual_start_rejected"] is True
    assert launched["item_id"] == "q1"
    assert svc._queue_launching_item_id == ""


def test_queue_state_recovers_from_backup_when_main_file_is_corrupt(tmp_path, monkeypatch):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "queued"}]}

    svc._save_queue()

    backup_file = queue_dir / "queue.json.bak"
    assert backup_file.is_file()
    (queue_dir / "queue.json").write_text("{broken", encoding="utf-8")

    recovered = training_service._load_training_queue_state()

    assert recovered["items"][0]["id"] == "q1"
    restored = json.loads((queue_dir / "queue.json").read_text(encoding="utf-8"))
    assert restored["items"][0]["id"] == "q1"


def test_queue_history_metadata_is_written_on_launch(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": True,
        "items": [{
            "id": "q1",
            "state": "running",
            "kind": "training",
            "retry_of": "old-q",
            "attempt": 3,
            "created_at": 123.0,
            "created_at_text": "2026-05-27 10:00:00",
        }],
    }

    async def fake_create_subprocess_exec(*args, **kwargs):
        return object()

    async def fake_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(training_service.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(svc, "_read_output", fake_background_task)
    monkeypatch.setattr(svc, "_monitor_system", fake_background_task)

    asyncio.run(svc._launch_job(
        ["python", "-c", "pass"],
        {},
        variant="demo",
        preset="default",
        methods_subdir="imported",
        output_dir=str(tmp_path / "out"),
        sample_dir=str(tmp_path / "out" / "sample"),
        data_dirs={},
        sample_config={},
        job="preprocess",
        start_message="queued",
        command_label="queued",
        queue_item_id="q1",
    ))

    meta = json.loads((Path(svc.current_task_dir) / "meta.json").read_text(encoding="utf-8"))
    assert meta["from_queue"] is True
    assert meta["queue_item_id"] == "q1"
    assert meta["queue_retry_of"] == "old-q"
    assert meta["queue_attempt"] == 3


def test_start_queue_item_applies_multi_gpu_launch_to_child_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ANIMA_ACCELERATE_LAUNCH", raising=False)
    monkeypatch.delenv("ANIMA_ACCELERATE_NUM_PROCESSES", raising=False)
    runtime = _runtime_payload(tmp_path)
    svc = TrainingService(web.Application())
    captured = {}

    async def fake_launch(cmd, env, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["kwargs"] = kwargs

    svc._launch_job = fake_launch
    asyncio.run(svc._start_queue_item({
        "id": "q-gpu",
        "state": "queued",
        "kind": "training",
        "requires_preprocess": False,
        "variant": "demo",
        "preset": "default",
        "methods_subdir": "imported",
        "runtime_config_file": runtime["runtime_config_file"],
        "source_config_file": "configs/imported/source.toml",
        "extra_args": [],
        "gpu_whitelist": [2, 1],
        "continue_info": {},
        "resume_info": {},
    }))

    assert captured["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2,1"
    assert captured["env"]["ANIMA_ACCELERATE_LAUNCH"] == "1"
    assert captured["env"]["ANIMA_ACCELERATE_NUM_PROCESSES"] == "2"
    assert "accelerate.commands.accelerate_cli" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--num_processes") + 1] == "2"
    assert captured["kwargs"]["gpu_whitelist"] == [2, 1]
    assert captured["kwargs"]["queue_item_id"] == "q-gpu"


def test_stop_running_queue_item_cancels_and_pauses(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "running"}]}
    svc._queue_paused = False
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"

    class FakeProcess:
        pid = 123
        returncode = None

    class FakePsutilProcess:
        def children(self, recursive=True):
            return []

        def terminate(self):
            return None

    monkeypatch.setattr(training_service.psutil, "Process", lambda pid: FakePsutilProcess())
    monkeypatch.setattr(training_service.psutil, "wait_procs", lambda family, timeout: (family, []))
    svc.process = FakeProcess()

    asyncio.run(svc.stop())

    item = svc.get_queue_snapshot()["items"][0]
    assert svc.get_queue_snapshot()["paused"] is True
    assert item["state"] == "canceled"


def test_queue_process_error_pauses_and_keeps_next_waiting(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "pause",
        "items": [
            {"id": "q1", "state": "running"},
            {"id": "q2", "state": "queued"},
        ],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "pause"
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"

    class FakeStdout:
        async def read(self, _size):
            return b""

    class FakeProcess:
        stdout = FakeStdout()

        async def wait(self):
            return 7

    svc.process = FakeProcess()

    asyncio.run(svc._read_output())

    snapshot = svc.get_queue_snapshot()
    assert snapshot["paused"] is True
    items = {item["id"]: item for item in snapshot["items"]}
    assert items["q1"]["state"] == "error"
    assert items["q2"]["state"] == "queued"


def test_queue_process_error_continues_when_failure_policy_continue(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "continue",
        "items": [
            {"id": "q1", "state": "running"},
            {"id": "q2", "state": "queued"},
        ],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "continue"
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"
    dispatch_calls: list[bool] = []
    monkeypatch.setattr(svc, "_schedule_queue_dispatch", lambda: dispatch_calls.append(True))

    class FakeStdout:
        async def read(self, _size):
            return b""

    class FakeProcess:
        stdout = FakeStdout()

        async def wait(self):
            return 7

    svc.process = FakeProcess()

    asyncio.run(svc._read_output())

    snapshot = svc.get_queue_snapshot()
    assert svc.status == "idle"
    assert snapshot["paused"] is False
    assert dispatch_calls == [True]
    items = {item["id"]: item for item in snapshot["items"]}
    assert items["q1"]["state"] == "error"
    assert items["q2"]["state"] == "queued"


def test_queue_retry_clones_frozen_runtime_config(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    retry_root = tmp_path / "retry-runs"
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: retry_root)
    runtime = _runtime_payload(tmp_path, "old-run")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [{
            "id": "q1",
            "state": "error",
            "kind": "training",
            "requires_preprocess": True,
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "runtime_config_file": runtime["runtime_config_file"],
            "source_config_file": "configs/imported/source.toml",
            "extra_args": [],
            "gpu_whitelist": [1],
            "continue_info": {},
            "resume_info": {},
            "history_task_ids": ["old-history"],
            "attempt": 1,
        }],
    }

    payload = asyncio.run(svc.retry_queue_item("q1"))

    retry = payload["item"]
    assert retry["state"] == "queued"
    assert retry["retry_of"] == "q1"
    assert retry["attempt"] == 1
    assert retry["manual_retry"] is True
    assert retry["retry_source"] == "manual"
    assert retry["history_task_ids"] == []
    assert retry["runtime_config_file"] != runtime["runtime_config_file"]
    retry_cfg = toml.loads(Path(retry["runtime_config_file"]).read_text(encoding="utf-8"))
    old_cfg = toml.loads(Path(runtime["runtime_config_file"]).read_text(encoding="utf-8"))
    assert retry_cfg["output_dir"] != old_cfg["output_dir"]
    assert retry_cfg["source_image_dir"] == old_cfg["source_image_dir"]


def test_queue_retry_training_clones_dataset_cache_before_old_runtime_delete(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    output_root = tmp_path / "runs"
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root)
    runtime = _runtime_payload(tmp_path, "old-run")
    old_run_dir = Path(runtime["run_dir"])
    old_resized = old_run_dir / "dataset_cache" / "dataset-01" / "resized"
    old_lora = old_run_dir / "dataset_cache" / "dataset-01" / "lora"
    old_resized.mkdir(parents=True, exist_ok=True)
    old_lora.mkdir(parents=True, exist_ok=True)
    (old_resized / "sample.png").write_text("image", encoding="utf-8")
    (old_lora / "sample.npz").write_text("cache", encoding="utf-8")
    dataset_config = old_run_dir / "dataset.runtime.toml"
    dataset_config.write_text(
        toml.dumps({
            "general": {"caption_extension": ".txt", "keep_tokens": 3},
            "datasets": [{
                "batch_size": 1,
                "subsets": [{
                    "image_dir": str(old_resized),
                    "cache_dir": str(old_lora),
                    "num_repeats": 1,
                    "custom_attributes": {"source_dir": "image_dataset/a"},
                }],
            }],
        }),
        encoding="utf-8",
    )
    runtime_config = Path(runtime["runtime_config_file"])
    cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    cfg.update({
        "dataset_config": str(dataset_config),
        "source_image_dir": "image_dataset/a",
        "resized_image_dir": str(old_resized),
        "lora_cache_dir": str(old_lora),
    })
    runtime_config.write_text(toml.dumps(cfg), encoding="utf-8")
    runtime["dataset_config_file"] = str(dataset_config)
    runtime["data_dirs"] = {
        "source_image_dir": "image_dataset/a",
        "resized_image_dir": str(old_resized),
        "lora_cache_dir": str(old_lora),
    }

    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [{
            "id": "q1",
            "state": "error",
            "kind": "training",
            "requires_preprocess": False,
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "runtime_config_file": runtime["runtime_config_file"],
            "source_config_file": "configs/imported/source.toml",
            "extra_args": [],
            "gpu_whitelist": [1],
            "continue_info": {},
            "resume_info": {},
            "history_task_ids": ["old-history"],
            "runtime_info": runtime,
            "attempt": 1,
        }],
    }

    payload = asyncio.run(svc.retry_queue_item("q1"))
    retry = payload["item"]
    retry_cfg = toml.loads(Path(retry["runtime_config_file"]).read_text(encoding="utf-8"))
    retry_dataset = toml.loads(Path(retry_cfg["dataset_config"]).read_text(encoding="utf-8"))
    retry_subset = retry_dataset["datasets"][0]["subsets"][0]

    assert retry_cfg["resized_image_dir"] != str(old_resized)
    assert retry_cfg["lora_cache_dir"] != str(old_lora)
    assert retry_subset["image_dir"] == retry_cfg["resized_image_dir"]
    assert retry_subset["cache_dir"] == retry_cfg["lora_cache_dir"]
    assert Path(retry_subset["image_dir"], "sample.png").read_text(encoding="utf-8") == "image"
    assert Path(retry_subset["cache_dir"], "sample.npz").read_text(encoding="utf-8") == "cache"

    deleted = asyncio.run(svc.cancel_queue_item("q1", delete_runtime=True))

    assert deleted["deleted_runtime"] is True
    assert not old_run_dir.exists()
    assert Path(retry_subset["image_dir"], "sample.png").exists()
    assert Path(retry_subset["cache_dir"], "sample.npz").exists()


def test_queue_top_bottom_cancel_waiting_and_clear_terminal_states_separately(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {"id": "a", "state": "queued"},
            {"id": "b", "state": "queued"},
            {"id": "c", "state": "queued"},
            {"id": "d", "state": "done"},
            {"id": "e", "state": "error"},
        ],
    }

    asyncio.run(svc.move_queue_item("c", "top"))
    assert [item["id"] for item in svc.get_queue_snapshot()["items"][:3]] == ["c", "a", "b"]
    asyncio.run(svc.move_queue_item("c", "bottom"))
    assert [item["id"] for item in svc.get_queue_snapshot()["items"][:3]] == ["a", "b", "c"]

    canceled = asyncio.run(svc.cancel_waiting_queue_items())
    assert canceled["canceled"] == 3
    assert all(item["state"] != "queued" for item in svc.get_queue_snapshot()["items"])

    cleared_canceled = asyncio.run(svc.clear_canceled_queue_items())
    assert cleared_canceled["removed"] == 3
    assert cleared_canceled["removed_by_state"] == {"canceled": 3}
    assert [(item["id"], item["state"]) for item in svc.get_queue_snapshot()["items"]] == [
        ("d", "done"),
        ("e", "error"),
    ]

    cleared_done = asyncio.run(svc.clear_completed_queue_items())
    assert cleared_done["removed"] == 1
    assert cleared_done["removed_by_state"] == {"done": 1}
    remaining = svc.get_queue_snapshot()["items"]
    assert [(item["id"], item["state"]) for item in remaining] == [("e", "error")]


def test_clear_queue_items_by_state_rejects_invalid_states_without_saving(
    tmp_path,
    monkeypatch,
):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    items = [
        {"id": "done", "state": "done"},
        {"id": "error", "state": "error"},
    ]
    svc._queue = {"paused": False, "items": list(items)}

    with pytest.raises(ValueError, match="只能清理已完成或已取消"):
        asyncio.run(svc.clear_queue_items_by_state({"error"}))
    with pytest.raises(ValueError, match="只能清理已完成或已取消"):
        asyncio.run(svc.clear_queue_items_by_state(set()))

    assert svc._queue["items"] == items
    assert not (queue_dir / "queue.json").exists()


def test_clear_finished_queue_items_keeps_compatibility_and_runtime_files(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    done_runtime = _runtime_payload(tmp_path, "done")
    canceled_runtime = _runtime_payload(tmp_path, "canceled")
    done_cache = Path(done_runtime["dataset_cache_dir"]) / "keep.npz"
    canceled_log = Path(canceled_runtime["logs_dir"]) / "keep.log"
    done_cache.write_text("cache", encoding="utf-8")
    canceled_log.write_text("log", encoding="utf-8")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {"id": "done", "state": "done", "runtime_info": done_runtime},
            {"id": "canceled", "state": "canceled", "runtime_info": canceled_runtime},
            {"id": "error", "state": "error", "runtime_info": done_runtime},
        ],
    }

    cleared = asyncio.run(svc.clear_finished_queue_items())

    assert cleared["removed"] == 2
    assert cleared["removed_by_state"] == {"canceled": 1, "done": 1}
    assert [(item["id"], item["state"]) for item in svc.get_queue_snapshot()["items"]] == [("error", "error")]
    assert Path(done_runtime["run_dir"]).exists()
    assert Path(canceled_runtime["run_dir"]).exists()
    assert done_cache.read_text(encoding="utf-8") == "cache"
    assert canceled_log.read_text(encoding="utf-8") == "log"


def test_cancel_all_queue_items_cancels_waiting_and_stale_running(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._current_queue_item_id = "running"
    svc._queue = {
        "paused": False,
        "items": [
            {"id": "running", "state": "running"},
            {"id": "a", "state": "queued"},
            {"id": "b", "state": "queued"},
            {"id": "done", "state": "done"},
            {"id": "error", "state": "error"},
        ],
    }

    canceled = asyncio.run(svc.cancel_all_queue_items())

    assert canceled["canceled"] == 3
    assert canceled["canceled_waiting"] == 2
    assert canceled["stopped_running"] == 0
    states = {item["id"]: item["state"] for item in svc.get_queue_snapshot()["items"]}
    assert states == {
        "running": "canceled",
        "a": "canceled",
        "b": "canceled",
        "done": "done",
        "error": "error",
    }


def test_cancel_all_queue_items_stops_active_running_item(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "items": [
            {"id": "running", "state": "running"},
            {"id": "waiting", "state": "queued"},
        ],
    }
    svc._queue_paused = False
    svc._current_queue_item_id = "running"
    svc.status = "running"
    svc.current_job = "training"

    class FakeProcess:
        pid = 123
        returncode = None

    class FakePsutilProcess:
        def children(self, recursive=True):
            return []

        def terminate(self):
            return None

    monkeypatch.setattr(training_service.psutil, "Process", lambda pid: FakePsutilProcess())
    monkeypatch.setattr(training_service.psutil, "wait_procs", lambda family, timeout: (family, []))
    svc.process = FakeProcess()

    canceled = asyncio.run(svc.cancel_all_queue_items())

    assert canceled["canceled"] == 2
    assert canceled["canceled_waiting"] == 1
    assert canceled["stopped_running"] == 1
    snapshot = svc.get_queue_snapshot()
    assert snapshot["paused"] is True
    states = {item["id"]: item["state"] for item in snapshot["items"]}
    assert states == {"running": "canceled", "waiting": "canceled"}


def test_abort_queue_after_current_cancels_waiting_and_keeps_running(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "items": [
            {"id": "running", "state": "running"},
            {"id": "waiting", "state": "queued"},
            {"id": "done", "state": "done"},
        ],
    }
    svc._queue_paused = False
    svc._current_queue_item_id = "running"

    async def forbidden_stop():
        raise AssertionError("abort_queue_after_current must not stop the active task")

    monkeypatch.setattr(svc, "stop", forbidden_stop)
    payload = asyncio.run(svc.abort_queue_after_current())

    assert payload["ok"] is True
    assert payload["paused"] is True
    assert payload["canceled_waiting"] == 1
    assert payload["running_kept"] == 1
    states = {item["id"]: item["state"] for item in svc.get_queue_snapshot()["items"]}
    assert states == {"running": "running", "waiting": "canceled", "done": "done"}


def test_force_abort_queue_stops_active_running_and_waiting_items(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "items": [
            {"id": "running", "state": "running"},
            {"id": "waiting", "state": "queued"},
        ],
    }
    svc._queue_paused = False
    svc._current_queue_item_id = "running"
    svc.status = "running"
    svc.current_job = "training"

    class FakeProcess:
        pid = 123
        returncode = None

    class FakePsutilProcess:
        def children(self, recursive=True):
            return []

        def terminate(self):
            return None

    monkeypatch.setattr(training_service.psutil, "Process", lambda pid: FakePsutilProcess())
    monkeypatch.setattr(training_service.psutil, "wait_procs", lambda family, timeout: (family, []))
    svc.process = FakeProcess()

    payload = asyncio.run(svc.force_abort_queue())

    assert payload["ok"] is True
    assert payload["paused"] is True
    assert payload["canceled_waiting"] == 1
    assert payload["stopped_running"] == 1
    states = {item["id"]: item["state"] for item in svc.get_queue_snapshot()["items"]}
    assert states == {"running": "canceled", "waiting": "canceled"}


def test_force_abort_queue_stops_non_queue_active_process(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": []}
    svc._queue_paused = False
    svc.status = "running"
    svc.current_job = "training"

    class FakeProcess:
        pid = 123
        returncode = None

    class FakePsutilProcess:
        def children(self, recursive=True):
            return []

        def terminate(self):
            return None

    monkeypatch.setattr(training_service.psutil, "Process", lambda pid: FakePsutilProcess())
    monkeypatch.setattr(training_service.psutil, "wait_procs", lambda family, timeout: (family, []))
    svc.process = FakeProcess()

    payload = asyncio.run(svc.force_abort_queue())

    assert payload["ok"] is True
    assert payload["paused"] is True
    assert payload["canceled"] == 1
    assert payload["stopped_running"] == 1
    assert svc.status == "idle"


def test_force_abort_queue_waits_for_launch_lock_then_stops_started_process(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "items": [
            {"id": "q1", "state": "queued"},
            {"id": "q2", "state": "queued"},
        ],
    }
    svc._queue_paused = False

    class FakeProcess:
        pid = 123
        returncode = None

    class FakePsutilProcess:
        def children(self, recursive=True):
            return []

        def terminate(self):
            return None

    ready = asyncio.Event()
    release = asyncio.Event()
    launched = []

    async def fake_start_queue_item(item):
        launched.append(item["id"])
        svc.status = "running"
        svc.current_job = "training"
        svc._current_queue_item_id = item["id"]
        svc.process = FakeProcess()
        ready.set()
        await release.wait()

    async def fake_broadcast_queue():
        return None

    monkeypatch.setattr(training_service.psutil, "Process", lambda pid: FakePsutilProcess())
    monkeypatch.setattr(training_service.psutil, "wait_procs", lambda family, timeout: (family, []))
    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast_queue)
    monkeypatch.setattr(svc, "_start_queue_item", fake_start_queue_item)

    async def run():
        dispatch_task = asyncio.create_task(svc._dispatch_queue())
        await ready.wait()
        abort_task = asyncio.create_task(svc.force_abort_queue())
        await asyncio.sleep(0)
        assert not abort_task.done()
        release.set()
        await dispatch_task
        return await abort_task

    payload = asyncio.run(run())

    assert launched == ["q1"]
    assert payload["stopped_running"] == 1
    assert payload["canceled_waiting"] == 1
    assert svc._queue_launching_item_id == ""
    snapshot = svc.get_queue_snapshot()
    assert snapshot["paused"] is True
    states = {item["id"]: item["state"] for item in snapshot["items"]}
    assert states == {"q1": "canceled", "q2": "canceled"}


def test_queue_launch_lock_serializes_manual_and_queue_start(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "queued"}]}
    svc._queue_paused = False
    ready = asyncio.Event()
    release = asyncio.Event()
    launched = []

    async def fake_start_queue_item(item):
        launched.append(item["id"])
        ready.set()
        await release.wait()
        svc.status = "running"

    monkeypatch.setattr(svc, "_start_queue_item", fake_start_queue_item)

    async def run():
        dispatch_task = asyncio.create_task(svc._dispatch_queue())
        await ready.wait()
        manual_task = asyncio.create_task(svc.start("manual", "default"))
        await asyncio.sleep(0)
        assert manual_task.done() is False
        release.set()
        await dispatch_task
        with pytest.raises(RuntimeError, match="已有任务在运行中"):
            await manual_task

    asyncio.run(run())

    assert launched == ["q1"]
    assert svc._queue_launching_item_id == ""


def test_delete_terminal_queue_item_only_removes_that_record(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {"id": "waiting", "state": "queued"},
            {"id": "failed", "state": "error", "history_task_ids": ["hist-a"]},
            {"id": "done", "state": "done"},
        ],
    }

    deleted = asyncio.run(svc.cancel_queue_item("failed"))

    assert deleted["deleted"] == 1
    assert deleted["message"] == "已从队列列表移除"
    assert [item["id"] for item in svc.get_queue_snapshot()["items"]] == ["waiting", "done"]


def test_delete_terminal_queue_item_can_remove_runtime_dir(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }

    deleted = asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert deleted["deleted"] == 1
    assert deleted["deleted_runtime"] is True
    assert not Path(runtime["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"] == []


def test_delete_terminal_queue_item_marks_cleanup_before_runtime_delete_failure(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }
    saves: list[dict] = []
    original_save = svc._save_queue

    def record_save():
        saves.append(json.loads(json.dumps(svc._queue)))
        original_save()

    def fail_rmtree(path):
        raise OSError("boom")

    monkeypatch.setattr(svc, "_save_queue", record_save)
    monkeypatch.setattr(training_service.shutil, "rmtree", fail_rmtree)

    with pytest.raises(OSError, match="boom"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    item = svc.get_queue_snapshot()["items"][0]
    assert item["cleanup_state"] == "error"
    assert item["cleanup_error"] == "boom"
    assert Path(runtime["run_dir"]).exists()
    assert any(
        saved["items"][0].get("cleanup_state") == "deleting_runtime"
        for saved in saves
    )


def test_delete_terminal_queue_item_rejects_incomplete_runtime_marker(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    training_service.shutil.rmtree(Path(runtime["run_dir"]) / "dataset_cache")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }

    with pytest.raises(ValueError, match="runtime 标记"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert Path(runtime["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"][0]["id"] == "failed"


def test_delete_terminal_queue_item_rejects_runtime_config_mismatch(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "old-run")
    other = _runtime_payload(tmp_path, "other-run")
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": {
                    **runtime,
                    "run_dir": other["run_dir"],
                    "runtime_config_file": runtime["runtime_config_file"],
                },
            },
        ],
    }

    with pytest.raises(ValueError, match="runtime 配置不匹配"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert Path(other["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"][0]["id"] == "failed"


def test_delete_terminal_queue_item_rejects_runtime_outside_output_root(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "outside")
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "other-runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }

    with pytest.raises(ValueError, match="输出根目录"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert Path(runtime["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"][0]["id"] == "failed"


def test_handle_queue_start_uses_enqueue_service(monkeypatch):
    class FakeService:
        def __init__(self):
            self.calls = []

        async def enqueue_training(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"ok": True, "items": [{"id": "q"}], "paused": False}

    svc = FakeService()
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(training_routes, "is_web_runtime_config", lambda value: value.endswith("config.runtime.toml"))
    req = _FakeJsonRequest(
        {
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "output/runs/demo/config.runtime.toml",
            "confirmed": True,
            "gpu_whitelist": [1],
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_queue_start(req))

    assert response.status == 200
    assert len(svc.calls) == 1
    args, kwargs = svc.calls[0]
    assert args[:3] == ("demo", "default", "imported")
    assert kwargs["requires_preprocess"] is False
    assert kwargs["gpu_whitelist"] == [1]


def test_handle_queue_batch_start_validates_and_uses_enqueue_service(monkeypatch):
    class FakeService:
        def __init__(self):
            self.calls = []

        async def enqueue_training_batch(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {
                "ok": True,
                "items": [{"id": "q1"}, {"id": "q2"}],
                "queued_count": 2,
                "paused": True,
            }

    svc = FakeService()
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(training_routes, "is_web_runtime_config", lambda value: False)
    req = _FakeJsonRequest(
        {
            "preset": "default",
            "start_paused": True,
            "gpu_whitelist": [1],
            "items": [
                {
                    "variant": "first",
                    "methods_subdir": "imported",
                    "config_file": "configs/imported/first.toml",
                },
                {
                    "variant": "nested/second",
                    "preset": "low_vram",
                    "methods_subdir": "gui-methods",
                    "config_file": "configs/gui-methods/nested/second.toml",
                },
            ],
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_queue_batch_start(req))

    assert response.status == 200
    assert len(svc.calls) == 1
    args, kwargs = svc.calls[0]
    entries = args[0]
    assert entries[0]["variant"] == "first"
    assert entries[0]["preset"] == "default"
    assert entries[0]["methods_subdir"] == "imported"
    assert entries[0]["config_file"] == "configs/imported/first.toml"
    assert entries[0]["requires_preprocess"] is True
    assert entries[1]["variant"] == "nested/second"
    assert entries[1]["preset"] == "low_vram"
    assert kwargs["default_preset"] == "default"
    assert kwargs["gpu_whitelist"] == [1]
    assert kwargs["start_paused"] is True


def test_handle_queue_post_dispatches_batch_payload(monkeypatch):
    class FakeService:
        def __init__(self):
            self.calls = []

        async def enqueue_training_batch(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"ok": True, "items": [], "queued_count": 1, "paused": True}

    svc = FakeService()
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(training_routes, "is_web_runtime_config", lambda value: False)
    req = _FakeJsonRequest(
        {
            "preset": "default",
            "items": [{
                "variant": "first",
                "methods_subdir": "imported",
                "config_file": "configs/imported/first.toml",
                "label": "导入配置 first",
            }],
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_queue_post(req))

    assert response.status == 200
    entries = svc.calls[0][0][0]
    assert entries[0]["label"] == "导入配置 first"


def test_queue_batch_routes_include_nested_static_alias():
    routes_source = Path(training_routes.__file__).read_text(encoding="utf-8")

    assert '"/api/training/queue"' in routes_source
    assert '"/api/training/queue/batch/start"' in routes_source
    assert '"/api/training/queue/batch-start"' in routes_source
    assert '"/api/training/queue/abort-after-current"' in routes_source
    assert '"/api/training/queue/force-abort"' in routes_source
    assert routes_source.index('"/api/training/queue/batch/start"') < routes_source.index('"/api/training/queue/{item_id}/move"')
    assert routes_source.index('"/api/training/queue/abort-after-current"') < routes_source.index('"/api/training/queue/{item_id}/move"')
    assert routes_source.index('"/api/training/queue/force-abort"') < routes_source.index('"/api/training/queue/{item_id}/move"')
    assert routes_source.index('"/api/training/queue/pause"') < routes_source.index('"/api/training/queue/{item_id}/move"')


def test_handle_queue_abort_controls_call_service_methods():
    class FakeService:
        def __init__(self):
            self.calls = []

        async def abort_queue_after_current(self):
            self.calls.append("abort-after-current")
            return {"ok": True, "message": "after current", "items": [], "paused": True}

        async def force_abort_queue(self):
            self.calls.append("force-abort")
            return {"ok": True, "message": "force", "items": [], "paused": True}

    svc = FakeService()

    abort_response = asyncio.run(training_routes.handle_queue_abort_after_current(
        _FakeJsonRequest({}, {"training_service": svc})
    ))
    force_response = asyncio.run(training_routes.handle_queue_force_abort(
        _FakeJsonRequest({}, {"training_service": svc})
    ))

    assert abort_response.status == 200
    assert force_response.status == 200
    assert svc.calls == ["abort-after-current", "force-abort"]


def test_handle_queue_resume_uses_history_checkpoint_service():
    class FakeService:
        async def enqueue_resume_from_history_task(
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
                "items": [],
                "paused": False,
            }

    req = _FakeJsonRequest(
        {
            "task_id": "task-a",
            "checkpoint": "state-dir",
            "duration_overrides": {"max_train_epochs": 1},
            "gpu_whitelist": [1],
        },
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_queue_resume(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["task_id"] == "task-a"
    assert payload["checkpoint"] == "state-dir"
    assert payload["duration_overrides"] == {"max_train_epochs": 1}
    assert payload["gpu_whitelist"] == [1]


def test_queue_management_routes_call_service():
    class FakeService:
        def __init__(self):
            self.calls = []

        async def set_queue_settings(self, **kwargs):
            self.calls.append(("settings", kwargs))
            return {"ok": True, "paused": kwargs.get("paused"), "failure_policy": kwargs.get("failure_policy")}

        async def retry_queue_item(self, item_id):
            self.calls.append(("retry", item_id))
            return {"ok": True, "item_id": item_id}

        async def cancel_waiting_queue_items(self):
            self.calls.append(("cancel-waiting", None))
            return {"ok": True, "canceled": 2}

        async def cancel_all_queue_items(self):
            self.calls.append(("cancel-all", None))
            return {"ok": True, "canceled": 4}

        async def clear_finished_queue_items(self):
            self.calls.append(("clear", None))
            return {"ok": True, "removed": 3}

        async def clear_completed_queue_items(self):
            self.calls.append(("clear-completed", None))
            return {"ok": True, "removed": 1}

        async def clear_canceled_queue_items(self):
            self.calls.append(("clear-canceled", None))
            return {"ok": True, "removed": 2}

        async def cancel_queue_item(self, item_id, *, delete_runtime=False):
            self.calls.append(("cancel", item_id, delete_runtime))
            return {"ok": True, "item_id": item_id, "deleted_runtime": delete_runtime}

    svc = FakeService()
    app = {"training_service": svc}

    settings = asyncio.run(training_routes.handle_queue_settings(
        _FakeJsonRequest({"paused": True, "failure_policy": "pause"}, app)
    ))
    retry = asyncio.run(training_routes.handle_queue_retry(
        _FakeJsonRequest({}, app, {"item_id": "q1"})
    ))
    cancel_all = asyncio.run(training_routes.handle_queue_cancel_all(_FakeJsonRequest({}, app)))
    cancel = asyncio.run(training_routes.handle_queue_cancel_waiting(_FakeJsonRequest({}, app)))
    clear = asyncio.run(training_routes.handle_queue_clear(_FakeJsonRequest({}, app)))
    clear_completed = asyncio.run(training_routes.handle_queue_clear_completed(_FakeJsonRequest({}, app)))
    clear_canceled = asyncio.run(training_routes.handle_queue_clear_canceled(_FakeJsonRequest({}, app)))
    delete = asyncio.run(training_routes.handle_queue_cancel(
        _FakeJsonRequest({"delete_runtime": True}, app, {"item_id": "q2"})
    ))

    assert (
        settings.status
        == retry.status
        == cancel_all.status
        == cancel.status
        == clear.status
        == clear_completed.status
        == clear_canceled.status
        == delete.status
        == 200
    )
    assert svc.calls == [
        ("settings", {"paused": True, "failure_policy": "pause", "auto_retry": None, "max_attempts": None, "retry_backoff_sec": None}),
        ("retry", "q1"),
        ("cancel-all", None),
        ("cancel-waiting", None),
        ("clear", None),
        ("clear-completed", None),
        ("clear-canceled", None),
        ("cancel", "q2", True),
    ]


def test_queue_auto_retry_clones_until_max_attempts(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    retry_root = tmp_path / "retry-runs"
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: retry_root)
    runtime = _runtime_payload(tmp_path, "auto-retry-run")
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "continue",
        "auto_retry": True,
        "max_attempts": 3,
        "retry_backoff_sec": 0,
        "items": [{
            "id": "q1",
            "state": "running",
            "kind": "training",
            "requires_preprocess": False,
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "runtime_config_file": runtime["runtime_config_file"],
            "source_config_file": "configs/imported/source.toml",
            "extra_args": [],
            "gpu_whitelist": [1],
            "continue_info": {},
            "resume_info": {},
            "history_task_ids": ["h1"],
            "attempt": 1,
        }],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "continue"
    svc._queue_auto_retry = True
    svc._queue_max_attempts = 3
    svc._queue_retry_backoff_sec = 0
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"
    dispatch_calls: list[bool] = []
    monkeypatch.setattr(svc, "_schedule_queue_dispatch", lambda: dispatch_calls.append(True))

    class FakeStdout:
        async def read(self, _size):
            return b""

    class FakeProcess:
        stdout = FakeStdout()

        async def wait(self):
            return 7

    svc.process = FakeProcess()
    asyncio.run(svc._read_output())

    snapshot = svc.get_queue_snapshot()
    assert snapshot["auto_retry"] is True
    assert snapshot["max_attempts"] == 3
    items = snapshot["items"]
    assert any(item["id"] == "q1" and item["state"] == "error" for item in items)
    retries = [item for item in items if item.get("retry_of") == "q1"]
    assert len(retries) == 1
    assert retries[0]["attempt"] == 2
    assert retries[0]["state"] == "queued"
    assert dispatch_calls == [True]

    # second failure should still clone once more (attempt 3)
    svc._current_queue_item_id = retries[0]["id"]
    retries[0]["state"] = "running"
    svc.status = "running"
    svc.current_job = "training"
    svc.process = FakeProcess()
    asyncio.run(svc._read_output())
    snapshot = svc.get_queue_snapshot()
    retries = [item for item in snapshot["items"] if item.get("retry_of") == "q1"]
    assert len(retries) == 2
    assert max(item["attempt"] for item in retries) == 3

    # third failure at attempt 3 should stop cloning
    third = max(retries, key=lambda item: item["attempt"])
    svc._current_queue_item_id = third["id"]
    third["state"] = "running"
    svc.status = "running"
    svc.current_job = "training"
    svc.process = FakeProcess()
    asyncio.run(svc._read_output())
    snapshot = svc.get_queue_snapshot()
    retries = [item for item in snapshot["items"] if item.get("retry_of") == "q1"]
    assert len(retries) == 2


def test_queue_dispatch_skips_items_before_next_run_at(tmp_path, monkeypatch):
    import time

    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    future = time.time() + 3600
    svc._queue = {
        "paused": False,
        "failure_policy": "continue",
        "auto_retry": True,
        "max_attempts": 3,
        "retry_backoff_sec": 30,
        "items": [
            {
                "id": "later",
                "state": "queued",
                "next_run_at": future,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
            },
            {
                "id": "ready",
                "state": "queued",
                "next_run_at": None,
                "variant": "demo2",
                "preset": "default",
                "methods_subdir": "imported",
            },
        ],
    }
    svc._queue_paused = False
    started: list[str] = []

    async def fake_start(item):
        started.append(str(item.get("id")))
        item["state"] = "running"

    monkeypatch.setattr(svc, "_start_queue_item", fake_start)
    asyncio.run(svc._dispatch_queue())
    assert started == ["ready"]
