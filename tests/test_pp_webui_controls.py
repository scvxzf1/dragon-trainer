"""Regression coverage for PP card persistence and read-only preflight."""

from types import SimpleNamespace

import pytest
import toml

from library.training.pp.krea2_batch import batch_divisibility_message
from web.services import settings_service
from web.services.config.preflight_compat import _compat_web_message
from web.services.config.preflight_pp import cached_runtime_vae_is_unused, krea2_batch_checks
from web.services.training import distributed
from web.services.training.distributed_inputs import apply_diagnosis_overrides


def test_partial_distributed_updates_preserve_other_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.toml"
    path.write_text('[global]\noutput_root = "keep-me"\n', encoding="utf-8")
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", path)
    settings_service.save_distributed_settings({
        "mode": "pp", "gpu_whitelist": [1, 2],
        "pipeline_parallel_microbatches": 2,
        "pipeline_parallel_blocks_to_swap": 4,
    })
    settings_service.save_distributed_settings({"pipeline_parallel_microbatches": 8})
    actual = settings_service.get_distributed_settings()
    assert actual["mode"] == "pp"
    assert actual["gpu_whitelist"] == [1, 2]
    assert actual["pipeline_parallel_blocks_to_swap"] == 4
    settings_service.save_distributed_settings({"pipeline_parallel_blocks_to_swap": 0})
    actual = settings_service.get_distributed_settings()
    assert actual["pipeline_parallel_blocks_to_swap"] == 0
    assert actual["pipeline_parallel_microbatches"] == 8
    assert toml.loads(path.read_text())["global"]["output_root"] == "keep-me"


def test_failed_distributed_update_does_not_write(tmp_path, monkeypatch):
    path = tmp_path / "settings.toml"
    path.write_text("[distributed]\npipeline_parallel_blocks_to_swap = 4\n")
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", path)
    before = path.read_bytes()
    with pytest.raises(settings_service.DistributedSettingsError):
        settings_service.save_distributed_settings({"pipeline_parallel_blocks_to_swap": -1})
    assert path.read_bytes() == before


def test_diagnosis_reads_current_draft_without_changing_saved_config(monkeypatch):
    original = {"model_family": "anima", "pipeline_parallel": False}
    observed = []
    monkeypatch.setattr(
        "web.services.config_service._load_training_config_for_web_run",
        lambda *args, **kwargs: original,
    )
    monkeypatch.setattr(
        distributed, "diagnose_pp_config",
        lambda cfg, *args, **kwargs: observed.append(dict(cfg)) or [],
    )
    monkeypatch.setattr(distributed, "diagnose_pp_dataset_readiness", lambda cfg: [])
    monkeypatch.setattr(distributed, "hardware_diagnosis", lambda *args: {})
    result = distributed.diagnose_distributed_training(
        "lora", "default", mode="pp", gpu_whitelist=[1, 2],
        config_overrides={
            "model_family": "krea2_raw", "pipeline_parallel": True,
            "pipeline_parallel_blocks_to_swap": 4,
        },
    )
    assert result["ok"]
    assert observed[0]["pipeline_parallel_blocks_to_swap"] == 4
    assert observed[0]["model_family"] == "krea2_raw"
    assert original == {"model_family": "anima", "pipeline_parallel": False}


@pytest.mark.parametrize("overrides", [
    {"output_dir": "/somewhere"}, {"datasets": []},
    {"pipeline_parallel_blocks_to_swap": []}, [],
])
def test_diagnosis_overrides_cannot_change_paths_or_structures(overrides):
    with pytest.raises(ValueError):
        apply_diagnosis_overrides({}, overrides)


@pytest.mark.parametrize("batch,microbatches,suggestion", [
    (1, 4, "batch_size=2"),
    (2, 4, "--pipeline_parallel_microbatches=2"),
    (3, 2, "--pipeline_parallel_microbatches=3"),
])
def test_microbatch_error_has_a_valid_action(batch, microbatches, suggestion):
    message = batch_divisibility_message(batch, microbatches)
    assert suggestion in message
    assert "--pipeline_parallel_microbatches=1" not in message


def test_preflight_uses_the_batch_webui_will_materialize(monkeypatch):
    monkeypatch.setattr(
        "web.services.config.datasets._dataset_rows_for_estimate",
        lambda cfg: [{"settings": {"batch_size": 1}}],
    )
    cfg = {
        "model_family": "krea2_raw", "train_batch_size": 8,
        "pipeline_parallel_microbatches": 4,
    }
    assert krea2_batch_checks(cfg) == []
    cfg["train_batch_size"] = 1
    checks = krea2_batch_checks(cfg)
    assert checks[0]["level"] == "error"
    assert "train batch size 1" in checks[0]["message"]


def test_valid_microbatch_split_and_other_families_have_no_extra_checks(monkeypatch):
    monkeypatch.setattr(
        "web.services.config.datasets._dataset_rows_for_estimate", lambda cfg: [],
    )
    assert krea2_batch_checks({
        "model_family": "krea2_raw", "train_batch_size": 4,
        "pipeline_parallel_microbatches": 2,
    }) == []
    assert krea2_batch_checks({"model_family": "anima"}) == []


def test_preflight_preserves_the_specific_cross_family_refusal():
    detail = "pipeline_parallel_blocks_to_swap is only implemented for krea2/krea2_raw"
    item = SimpleNamespace(code="pipeline_parallel_config", message=detail)
    message = _compat_web_message(item)
    assert "流水线配置无效" in message
    assert detail in message


@pytest.mark.parametrize("change,runtime_file,complete,allowed", [
    ({}, True, True, True),
    ({}, False, True, False),
    ({}, True, False, False),
    ({"model_family": "anima"}, True, True, False),
    ({"cache_latents": False}, True, True, False),
    ({"skip_cache_check": True}, True, True, False),
    ({"sample_prompts": "prompts.txt", "sample_every_n_steps": 1}, True, True, False),
    ({"use_cmmd": True}, True, True, False),
])
def test_only_fully_cached_runtime_can_skip_unused_vae(
    change, runtime_file, complete, allowed, monkeypatch,
):
    monkeypatch.setattr(
        "web.services.config.pp_cache_validation.krea2_runtime_latents_complete",
        lambda cfg: True,
    )
    cfg = {
        "model_family": "krea2_raw", "pipeline_parallel": True,
        "cache_latents": True, "cache_text_encoder_outputs": True,
        "use_cmmd": False, **change,
    }
    checks = [
        {"key": "latent_cache", "level": "ok" if complete else "warning"},
        {"key": "text_cache", "level": "ok"},
    ]
    assert cached_runtime_vae_is_unused(
        cfg, checks, runtime_file=runtime_file, dataset_count=1,
    ) is allowed


def test_strict_latent_audit_keeps_cache_identity_and_nonempty_requirements(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    from library import cache_identity
    from web.services.config.pp_cache_validation import krea2_runtime_latents_complete

    dataset = tmp_path / "dataset.toml"
    dataset.write_text(toml.dumps({
        "datasets": [{"batch_size": 2, "subsets": [{"image_dir": str(tmp_path)}]}],
    }))
    monkeypatch.setattr(
        "web.services.config.datasets._dataset_config_path_from_cfg",
        lambda cfg: dataset,
    )
    cfg = {"model_family": "krea2_raw", "dataset_config": str(dataset), "vae": "unused"}
    assert krea2_runtime_latents_complete(cfg) is False
    Image.new("RGB", (896, 1200)).save(tmp_path / "a.png")
    path = tmp_path / "a_0896x1200_anima.npz"
    values = {
        "latents_150x112": np.zeros((16, 150, 112), dtype=np.float32),
        "original_size_150x112": np.array([896, 1200]),
        "crop_ltrb_150x112": np.array([0, 0, 896, 1200]),
    }
    np.savez(path, **values)
    assert krea2_runtime_latents_complete(cfg) is True
    assert krea2_runtime_latents_complete({**cfg, "require_cache_identity": True}) is False
    identity = cache_identity.latent_cache_identity(family="krea2_raw", vae="different-vae")
    np.savez(path, **values, cache_identity=cache_identity.encode_npz(identity))
    assert krea2_runtime_latents_complete(cfg) is False


def test_worker_receives_physical_external_config_path(tmp_path, monkeypatch):
    from web.services.training import launcher_runtime

    actual = tmp_path / "external" / "config.runtime.toml"
    monkeypatch.setattr(launcher_runtime, "_resolve_display_path", lambda name: actual)
    assert launcher_runtime._training_config_argument("configs/imported/config.runtime.toml") == str(actual)


def test_krea2_pp_batch_mismatch_is_reported_once_and_world_size_forwarded():
    from web.services.config.preflight_compat import _check_pipeline_trainer_admission

    cfg = {
        "model_family": "krea2_raw",
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 4,
        "train_batch_size": 2,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_vae_cache": True,
        "use_text_cache": True,
        "network_train_unet_only": True,
    }
    recorded: list[tuple[str, str, str, dict]] = []
    _check_pipeline_trainer_admission(
        cfg,
        lambda level, key, msg, **kw: recorded.append((level, key, msg, kw)),
        world_size=2,
    )
    errors = [row for row in recorded if row[0] == "error"]
    assert len(errors) == 1
    assert errors[0][1] == "pipeline_parallel_microbatches"

    diag_checks = distributed.diagnose_pp_config(cfg, world_size=2, device_indices=[1, 2])
    divisibility_errors = [
        c for c in diag_checks if c["level"] == "error" and "batch" in c["message"].lower()
    ]
    assert len(divisibility_errors) == 1


def test_schema_gate_accepts_moss_max_token_length_tiers():
    from web.services.config import schema_gate

    errors, warnings = schema_gate.validate_patch_values({"max_token_length": 1024})
    assert errors == []
    assert warnings == []


def test_frontend_moss_allows_fp16_and_pp_fallback():
    import json
    import shutil
    import subprocess
    from pathlib import Path

    if not shutil.which("node"):
        pytest.skip("node is required for frontend checks")
    root = Path(__file__).resolve().parents[1]
    family_uri = (
        root / "web/frontend-next/src/features/training-config/domain/model-family.js"
    ).as_uri()
    script = f"""
const family = await import({json.dumps(family_uri)});
console.log(JSON.stringify({{
    mossFp16: family.modelFamilyOptionSupported('mixed_precision', 'moss_image', 'fp16'),
    mossBf16: family.modelFamilyOptionSupported('mixed_precision', 'moss_image', 'bf16'),
    mossPp: family.modelFamilySupportsPipelineParallel('moss_image'),
    krea2Pp: family.modelFamilySupportsPipelineParallel('krea2'),
    animaPp: family.modelFamilySupportsPipelineParallel('anima'),
}}));
"""
    res = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=root, check=True, capture_output=True, text=True, timeout=20,
    )
    payload = json.loads(res.stdout)
    assert payload == {
        "mossFp16": True,
        "mossBf16": True,
        "mossPp": True,
        "krea2Pp": True,
        "animaPp": True,
    }
