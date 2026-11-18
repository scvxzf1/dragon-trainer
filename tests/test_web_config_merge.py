from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import toml
from PIL import Image

from web.routes import config as config_routes
from web.services import config_service
from web.services.config import _legacy as legacy_config
from web.services.config import datasets as config_datasets
from web.services.config import merge as config_merge
from web.services.config import metadata as config_metadata
from web.services.config import paths as config_paths


# Split: raw_files / merge / legacy shims

from tests import web_config_test_support as _web_config_support

globals().update(
    {
        name: value
        for name, value in vars(_web_config_support).items()
        if name == "Path" or not name.startswith("__")
    }
)


def test_merge_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.merge as merge; "
        "assert callable(merge.list_methods); "
        "assert 'lora' in merge.list_methods.__wrapped__(); "
        "assert 'web.services.config_service' not in sys.modules; "
        "assert 'web.services.config._legacy' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_merge_helpers_remain_available_from_legacy_module(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    methods = configs / "methods"
    methods.mkdir()
    (methods / "spd.toml").write_text('output_name = "spd"\n', encoding="utf-8")
    gui_methods = configs / "gui-methods"
    gui_methods.mkdir()
    (gui_methods / "lora.toml").write_text(
        "\n".join([
            'output_name = "lora"',
            "[variant]",
            'family = "lora"',
            "order = 1",
            "",
        ]),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", gui_methods)
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    expected_shims = (
        "list_methods",
        "list_variants",
        "list_all_variants",
        "list_presets",
        "load_merged_config",
        "suggest_data_dirs",
        "suggest_dataset_dirs",
        "apply_auto_data_dirs",
    )
    assert tuple(legacy_config._MERGE_SHIM_NAMES) == expected_shims
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._MERGE_SHIMS[name]
        assert (
            getattr(legacy_config, name).__doc__
            == f"Compatibility shim forwarding to web.services.config.merge.{name}."
        )
    legacy_config._restore_raw_files_shims()
    raw_file_shims = legacy_config._RAW_FILES_SHIMS
    assert legacy_config.load_raw_file is raw_file_shims["load_raw_file"]

    assert config_service.list_variants("spd") == ["spd"]
    for name, shim in raw_file_shims.items():
        assert getattr(legacy_config, name) is shim

    assert "spd" in legacy_config.list_methods()
    assert legacy_config.list_variants("spd") == ["spd"]
    assert legacy_config.list_variants("lora") == ["lora"]
    assert legacy_config.list_all_variants() == ["lora"]
    assert legacy_config.list_presets() == ["default"]

    merged = legacy_config.load_merged_config("lora", "default")
    assert merged["max_train_steps"] == 0
    assert merged["source_image_dir"] == "image_dataset"
    assert legacy_config.suggest_data_dirs("image_dataset/hero") == {
        "ok": True,
        "source_image_dir": "image_dataset/hero",
        "resized_image_dir": "image_dataset/hero_resized",
        "lora_cache_dir": "image_dataset/hero_lora_cache",
    }

    source_dir = legacy_config.ROOT / "image_dataset" / "inspect"
    source_dir.mkdir(parents=True)
    (source_dir / "hero.png").write_bytes(b"png")
    (source_dir / "notes.txt").write_text("caption", encoding="utf-8")
    inspected = legacy_config.suggest_data_dirs("image_dataset/inspect", inspect=True)
    assert inspected["source_exists"] is True
    assert inspected["source_is_dir"] is True
    assert inspected["source_image_count"] == 1
    assert inspected["source_inspection_error"] == ""
    assert legacy_config.suggest_dataset_dirs(["image_dataset/hero"]) == {
        "ok": True,
        "datasets": [{
            "index": 0,
            "source_dir": "image_dataset/hero",
            "image_dir": "image_dataset/hero_resized",
            "cache_dir": "image_dataset/hero_lora_cache",
        }],
    }

    auto_dirs = legacy_config.apply_auto_data_dirs({"source_image_dir": "image_dataset/hero"})
    assert auto_dirs["resized_image_dir"] == "image_dataset/hero_resized"
    assert auto_dirs["lora_cache_dir"] == "image_dataset/hero_lora_cache"
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._MERGE_SHIMS[name]


def test_legacy_merge_private_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import merge as merge_impl

    sentinel_path = Path("configs/gui-methods/lora.toml")
    monkeypatch.setattr(
        merge_impl,
        "_builtin_variants_by_family",
        lambda: {"lora": [(7, "sentinel")]},
    )
    monkeypatch.setattr(
        merge_impl,
        "_read_variant_metadata",
        lambda path: {"path": str(path)},
    )
    monkeypatch.setattr(
        merge_impl,
        "_legacy_exact_variant_for_method",
        lambda method: [f"legacy-{method}"],
    )
    monkeypatch.setattr(
        merge_impl,
        "_custom_gui_variants",
        lambda: ["custom/sentinel"],
    )

    assert legacy_config._builtin_variants_by_family() == {"lora": [(7, "sentinel")]}
    assert legacy_config._read_variant_metadata(sentinel_path) == {"path": str(sentinel_path)}
    assert legacy_config._legacy_exact_variant_for_method("lora") == ["legacy-lora"]
    assert legacy_config._custom_gui_variants() == ["custom/sentinel"]


def test_merge_common_path_helpers_forward_to_common_module(monkeypatch):
    from web.services.config import common as common_impl
    from web.services.config import merge as merge_impl

    root = Path("/tmp/anima-test-root")
    configs = root / "configs"
    monkeypatch.setattr(merge_impl, "ROOT", root)
    monkeypatch.setattr(merge_impl, "CONFIGS_DIR", configs)
    calls: dict[str, tuple[tuple[Any, ...], Path, Path]] = {}

    def sentinel(name: str, result):
        def impl(*args):
            calls[name] = (args, common_impl.ROOT, common_impl.CONFIGS_DIR)
            return result

        return impl

    source_path = Path("image_dataset/hero")
    expected_path = Path("sentinel")
    helper_args = {
        "_load": ((Path("configs/base.toml"),), {"loaded": True}),
        "_safe_config_subdir": (("gui-methods",), expected_path),
        "_resolve_project_path": (("image_dataset/hero",), expected_path),
        "_auto_data_dir_for_key": (("", source_path, "resized"), expected_path),
        "_derived_data_dir": ((source_path, "resized"), expected_path),
        "_is_builtin_default_data_dir": (("post_image_dataset/resized",), True),
        "_display_path": ((source_path,), "image_dataset/hero"),
    }
    for name, (_args, result) in helper_args.items():
        monkeypatch.setattr(common_impl, name, sentinel(name, result))

    for name, (args, result) in helper_args.items():
        assert getattr(merge_impl, name)(*args) == result
        assert calls[name] == (args, root, configs)


def test_handle_merged_uses_selected_training_config_file(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    (configs / "imported" / "selected.toml").write_text(
        "\n".join(
            [
                'output_name = "selected_form"',
                "max_train_epochs = 9",
                "max_train_steps = 0",
                "train_batch_size = 3",
                "gradient_accumulation_steps = 2",
                "sample_ratio = 0.5",
            ]
        ),
        encoding="utf-8",
    )

    response = asyncio.run(config_routes.handle_merged(_QueryRequest({
        "variant": "lora",
        "preset": "default",
        "methods_subdir": "imported",
        "config_file": "configs/imported/selected.toml",
    })))
    body = json.loads(response.text)

    assert response.status == 200
    assert body["output_name"] == "selected_form"
    assert body["max_train_epochs"] == 9
    assert body["max_train_steps"] == 0
    assert body["train_batch_size"] == 3
    assert body["gradient_accumulation_steps"] == 2
    assert body["sample_ratio"] == 0.5


def test_web_and_library_merge_agree_on_gui_method_defaults(tmp_path: Path, monkeypatch):
    """Web load_merged_config and library load_method_preset share core defaults."""
    from library.config.io import load_method_preset
    from web.services.config import merge as web_merge

    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    # Provide a realistic gui-methods lora.toml + base/presets for both paths.
    (configs / "gui-methods").mkdir(parents=True, exist_ok=True)
    (configs / "base.toml").write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset"',
                'resized_image_dir = "post_image_dataset/resized"',
                'lora_cache_dir = "post_image_dataset/lora"',
                "max_train_steps = 1000",
                'pretrained_model_name_or_path = "models/dit.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text(
        "[default]\nlearning_rate = 1e-4\n",
        encoding="utf-8",
    )
    (configs / "gui-methods" / "lora.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                "network_dim = 16",
                "network_alpha = 16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(web_merge, "CONFIGS_DIR", configs, raising=False)
    monkeypatch.setattr(web_merge, "PRESETS_FILE", configs / "presets.toml", raising=False)
    monkeypatch.setattr(web_merge, "GUI_METHODS_DIR", configs / "gui-methods", raising=False)

    web_cfg = web_merge.load_merged_config("lora", "default", "gui-methods")
    lib_cfg = load_method_preset(
        "lora",
        "default",
        configs_dir=str(configs),
        methods_subdir="gui-methods",
    )

    core_keys = [
        "network_module",
        "network_dim",
        "network_alpha",
        "learning_rate",
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "max_train_steps",
    ]
    for key in core_keys:
        assert web_cfg.get(key) == lib_cfg.get(key), key


def test_list_methods_omits_missing_known_files(tmp_path: Path, monkeypatch) -> None:
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "lora.toml").write_text("[network]\n", encoding="utf-8")
    monkeypatch.setattr(config_merge, "CONFIGS_DIR", tmp_path)

    names = config_merge.list_methods.__wrapped__()

    assert "lora" in names
    assert "lokr" not in names
    assert "tlora" not in names
    assert "hydralora" not in names
    assert "spd" not in names


def test_list_methods_includes_disk_extras(tmp_path: Path, monkeypatch) -> None:
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "turbo.toml").write_text("[network]\n", encoding="utf-8")
    (methods / "lora.toml").write_text("[network]\n", encoding="utf-8")
    monkeypatch.setattr(config_merge, "CONFIGS_DIR", tmp_path)

    names = config_merge.list_methods.__wrapped__()

    assert "turbo" in names
    assert "lora" in names
    assert names.index("lora") < names.index("turbo")


def test_list_methods_empty_dir_returns_empty(tmp_path: Path, monkeypatch) -> None:
    methods = tmp_path / "methods"
    methods.mkdir()
    monkeypatch.setattr(config_merge, "CONFIGS_DIR", tmp_path)

    assert config_merge.list_methods.__wrapped__() == []


def test_web_list_presets_includes_custom_preset_files(tmp_path: Path, monkeypatch) -> None:
    configs = tmp_path / "configs"
    (configs / "custom" / "presets").mkdir(parents=True)
    (configs / "presets.toml").write_text("[default]\nlearning_rate = 1e-4\n", encoding="utf-8")
    (configs / "custom" / "presets" / "v100_fast.toml").write_text(
        "learning_rate = 2e-4\n", encoding="utf-8"
    )
    monkeypatch.setattr(config_merge, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_merge, "PRESETS_FILE", configs / "presets.toml")

    assert config_merge.list_presets.__wrapped__() == ["default", "v100_fast"]


def test_parse_bool_unifies_network_registry_flags() -> None:
    from library.config.normalize import is_flag_enabled, parse_bool
    from networks.registry import resolve_network_spec

    for raw in (True, 1, "1", "true", "TRUE", " yes ", "on"):
        assert parse_bool(raw) is True
    for raw in (False, 0, "0", "false", "no", "off", None, "", "none"):
        assert parse_bool(raw) is False

    assert is_flag_enabled("shared_A") is True
    assert is_flag_enabled("off") is False

    assert resolve_network_spec({"use_ortho": "1"}).name == "ortho"
    assert resolve_network_spec({"use_ortho": "on"}).name == "ortho"
    assert resolve_network_spec({"dora_wd": 1}).name == "dora"
