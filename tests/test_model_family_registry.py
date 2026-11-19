from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library.models.family_registry import (
    MODEL_FAMILY_REGISTRY,
    ModelFamilyHandlerError,
    TextCacheSpec,
    dispatch_model_family,
    get_model_family_spec,
    normalize_registered_family,
)


def _register_third_family(monkeypatch, **overrides):
    spec = replace(
        get_model_family_spec("krea2_raw"),
        name="third_model",
        display_name="Third Model",
        aliases=frozenset({"third", "third_model"}),
        text_cache=TextCacheSpec("_third_te.safetensors", "third_te_v1"),
        **overrides,
    )
    monkeypatch.setitem(MODEL_FAMILY_REGISTRY, spec.name, spec)
    return spec


def test_registry_normalizes_declared_aliases_only(monkeypatch) -> None:
    _register_third_family(monkeypatch)

    assert normalize_registered_family("third_model") == "third_model"
    assert normalize_registered_family("third", allow_aliases=True) == "third_model"
    with pytest.raises(ValueError, match="model_family"):
        normalize_registered_family("third")


def test_dispatch_rejects_incomplete_handler_table(monkeypatch) -> None:
    _register_third_family(monkeypatch)

    with pytest.raises(ModelFamilyHandlerError, match="missing=third_model"):
        dispatch_model_family(
            "anima",
            operation="test operation",
            handlers={
                "anima": object(), "krea2_raw": object(), "z_image": object(),
                "qwen_image_2_1": object(), "moss_image": object(),
            },
        )


def test_registered_family_cannot_fall_back_to_anima_loader(monkeypatch) -> None:
    _register_third_family(monkeypatch)
    from library.training import model_loading

    monkeypatch.setattr(
        model_loading,
        "_load_anima_dit",
        lambda *args, **kwargs: pytest.fail("third family fell back to Anima"),
    )

    with pytest.raises(ModelFamilyHandlerError, match="missing=third_model"):
        model_loading.load_unet_lazily(
            None,
            SimpleNamespace(model_family="third_model"),
            None,
            None,
            None,
        )


def test_registered_family_capabilities_are_enforced(monkeypatch) -> None:
    _register_third_family(monkeypatch, plain_lora_only=True)
    from library.training.adapter_resolver import resolve_adapters
    from library.training.compat_matrix import check_training_compat

    result = check_training_compat(
        {
            "model_family": "third_model",
            "network_module": "networks.methods.ip_adapter",
            "use_ip_adapter": True,
        }
    )
    assert {item.code for item in result.errors} == {"family_plain_lora_only"}

    args = SimpleNamespace(
        model_family="third_model",
        use_ip_adapter=True,
        use_easycontrol=False,
        use_byg=False,
    )
    network = SimpleNamespace(_contrastive_target_weight=0.0)
    with pytest.raises(ValueError, match="Third Model.*method adapters"):
        resolve_adapters(args, network)


def test_krea_cache_writes_family_schema_metadata(tmp_path, monkeypatch) -> None:
    from library.models.krea2_raw.strategy import (
        Krea2TextEncoderOutputsCachingStrategy,
    )

    cache_path = tmp_path / "sample_krea2_te.safetensors"
    strategy = Krea2TextEncoderOutputsCachingStrategy(cache_to_disk=True)
    monkeypatch.setattr(
        strategy,
        "_encode_captions",
        lambda *args, **kwargs: (
            torch.zeros(1, 4, 2560, dtype=torch.bfloat16),
            torch.ones(1, 4, dtype=torch.bool),
        ),
    )
    info = SimpleNamespace(
        caption="test",
        caption_variants=None,
        cache_caption_variants=False,
        caption_dropout_rate=0.0,
        text_encoder_outputs_npz=str(cache_path),
    )

    strategy.cache_batch_outputs(None, [], None, [info])

    with safe_open(cache_path, framework="pt") as handle:
        assert handle.metadata() == {
            "model_family": "krea2_raw",
            "cache_schema": "krea2_te_v1",
        }


def test_anima_cache_writes_family_schema_metadata(tmp_path, monkeypatch) -> None:
    from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy

    cache_path = tmp_path / "sample_anima_te.safetensors"
    strategy = AnimaTextEncoderOutputsCachingStrategy(True, 1, False)
    monkeypatch.setattr(
        strategy,
        "_encode_to_tensors",
        lambda *args, **kwargs: (
            torch.zeros(1, 4, 8, dtype=torch.bfloat16),
            torch.ones(1, 4, dtype=torch.int32),
            torch.zeros(1, 4, dtype=torch.int64),
            torch.ones(1, 4, dtype=torch.int32),
            None,
        ),
    )
    info = SimpleNamespace(
        caption="test",
        caption_dropout_rate=0.0,
        text_encoder_outputs_npz=str(cache_path),
    )

    strategy.cache_batch_outputs(None, [], None, [info])

    with safe_open(cache_path, framework="pt") as handle:
        assert handle.metadata() == {
            "model_family": "anima",
            "cache_schema": "anima_te_v1",
        }


def test_krea_cache_accepts_legacy_and_rejects_explicit_wrong_family(tmp_path) -> None:
    from library.models.krea2_raw.strategy import (
        Krea2TextEncoderOutputsCachingStrategy,
    )

    tensors = {
        "hiddens": torch.zeros(4, 1, 8, dtype=torch.bfloat16),
        "mask": torch.ones(4, dtype=torch.bool),
        "caption_dropout_rate": torch.tensor(0.0),
    }
    strategy = Krea2TextEncoderOutputsCachingStrategy(cache_to_disk=True)

    legacy_path = tmp_path / "legacy.safetensors"
    save_file(tensors, legacy_path)
    assert strategy.is_disk_cached_outputs_expected(str(legacy_path))
    assert len(strategy.load_outputs_npz(str(legacy_path))) == 3

    wrong_path = tmp_path / "wrong.safetensors"
    save_file(
        tensors,
        wrong_path,
        metadata={"model_family": "anima", "cache_schema": "anima_te_v1"},
    )
    assert not strategy.is_disk_cached_outputs_expected(str(wrong_path))
    with pytest.raises(ValueError, match="text cache family mismatch"):
        strategy.load_outputs_npz(str(wrong_path))


def test_krea_cache_rejects_malformed_tensor_layout(tmp_path) -> None:
    from library.models.krea2_raw.strategy import (
        Krea2TextEncoderOutputsCachingStrategy,
    )

    cache_path = tmp_path / "malformed.safetensors"
    save_file(
        {
            "hiddens": torch.zeros(4, 1, 1, dtype=torch.bfloat16),
            "mask": torch.ones(4, dtype=torch.int32),
            "caption_dropout_rate": torch.tensor(0.0),
        },
        cache_path,
    )
    strategy = Krea2TextEncoderOutputsCachingStrategy(cache_to_disk=True)

    assert not strategy.is_disk_cached_outputs_expected(str(cache_path))
    with pytest.raises(ValueError, match="invalid Krea text cache tensor layout"):
        strategy.load_outputs_npz(str(cache_path))


def test_family_registry_and_capability_specs_stay_torch_free() -> None:
    import subprocess
    import sys

    code = (
        "import sys; "
        "import library.models.capability_specs; "
        "from library.models.family_registry import model_family_capability_catalog; "
        "catalog = model_family_capability_catalog(); "
        "assert len(catalog) >= 5; "
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if m.startswith('torch'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_ssot_capability_consumers_match_registry() -> None:
    from library.models.family_registry import model_family_capability_catalog
    from library.quantization import policy as quant_policy
    from library.training.pp.gate import TRAINER_PP_WIRED_FAMILIES
    from library.training.pp.trainer_adapters import ADAPTERS

    assert TRAINER_PP_WIRED_FAMILIES == frozenset(ADAPTERS)
    assert quant_policy.FAMILIES == {
        "anima": ("blocks", "mlp", ("layer1", "layer2")),
        "krea2_raw": ("blocks", "mlp", ("gate", "up", "down")),
        "z_image": ("layers", "feed_forward", ("w1", "w2", "w3")),
        "qwen_image_2_1": ("transformer_blocks", "img_mlp", ("gate_layer", "proj", "out")),
    }

    catalog = {item["name"]: item for item in model_family_capability_catalog()}
    assert catalog["anima"]["max_blocks_to_swap"] == 26
    assert catalog["krea2_raw"]["max_blocks_to_swap"] == 26
    assert catalog["z_image"]["max_blocks_to_swap"] == 28
    assert catalog["qwen_image_2_1"]["max_blocks_to_swap"] == 30
    assert catalog["moss_image"]["max_blocks_to_swap"] == 40
    assert catalog["qwen_image_2_1"]["pixel_alignment"] == 32
    assert catalog["anima"]["pixel_alignment"] == 16
    assert catalog["krea2_raw"]["pipeline_parallel"]["supports_stage_block_swap"] is True
    assert catalog["anima"]["pipeline_parallel"]["supports_stage_block_swap"] is False
    assert catalog["anima"]["pipeline_parallel"]["requires_batch_divisible_by_microbatches"] is True
    assert catalog["krea2_raw"]["pipeline_parallel"]["requires_batch_divisible_by_microbatches"] is True
    assert catalog["moss_image"]["pipeline_parallel"]["requires_batch_divisible_by_microbatches"] is True
    assert catalog["z_image"]["pipeline_parallel"]["trainer_wired"] is False
    assert get_model_family_spec("krea2_raw").attention.forbids_xformers_flag is False
    assert get_model_family_spec("z_image").attention.forbids_xformers_flag is True
    assert get_model_family_spec("qwen_image_2_1").attention.forbids_xformers_flag is True
    assert get_model_family_spec("moss_image").attention.forbids_xformers_flag is True


def test_family_plugin_loader_is_lazy_and_registered() -> None:
    import subprocess
    import sys

    code = (
        "import sys; "
        "from library.models.family_plugins import get_family_plugin; "
        "assert 'library.models.z_image.strategy' not in sys.modules; "
        "plugin = get_family_plugin('z_image'); "
        "assert plugin.family == 'z_image'; "
        "assert 'library.models.z_image.strategy' not in sys.modules; "
        "assert get_family_plugin('z_image') is plugin"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    from library.models.family_plugins import get_family_plugin

    with pytest.raises(ValueError, match="family plugin"):
        get_family_plugin("unknown_family")


def test_family_plugins_declare_text_preprocess_contracts() -> None:
    from library.models.family_plugins import get_family_plugin

    expected = {
        "anima": ("module", "scripts.preprocess.cache_text_embeddings"),
        "krea2_raw": ("module", "scripts.krea2.preprocess_te_cache"),
        "z_image": ("module", "scripts.z_image.preprocess_te_cache"),
        "qwen_image_2_1": ("special", "scripts.qwen_image_2_1.preprocess_te_cache"),
        "moss_image": ("special", "scripts.moss_image.preprocess_te_cache"),
    }
    for family, (runner, module) in expected.items():
        spec = get_family_plugin(family).text_preprocess_spec
        assert (spec.runner, spec.module) == (runner, module)


def test_family_plugin_loader_fails_closed_for_incomplete_plugin(monkeypatch) -> None:
    import library.models.family_plugins as family_plugins

    class IncompletePlugin:
        family = "z_image"
        text_preprocess_spec = family_plugins.TextPreprocessSpec(runner="module")

    monkeypatch.delitem(family_plugins._PLUGINS, "z_image", raising=False)
    monkeypatch.setitem(family_plugins._PLUGIN_MODULES, "z_image", "test.plugin")
    monkeypatch.setattr(
        family_plugins.importlib,
        "import_module",
        lambda _name: SimpleNamespace(PLUGIN=IncompletePlugin()),
    )

    with pytest.raises(TypeError, match="missing callable contract methods"):
        family_plugins.get_family_plugin("z_image")


def test_family_latent_cache_strategies_use_registry_spec_and_plugin_encoder() -> None:
    from library.models.family_plugins import get_family_plugin
    from library.strategies.latent_cache import FamilyLatentCacheStrategy

    args = SimpleNamespace(
        cache_latents_to_disk=False,
        vae_batch_size=1,
        skip_cache_check=False,
        vae="local-vae",
        require_cache_identity=False,
    )
    for family in MODEL_FAMILY_REGISTRY:
        plugin = get_family_plugin(family)
        strategy = plugin.build_latent_cache_strategy(args)
        spec = get_model_family_spec(family).latent_space

        assert strategy.MODEL_FAMILY == family
        assert strategy.latent_space_spec is spec
        assert strategy.latents_stride == spec.vae_spatial_compression
        assert strategy.cache_suffix == spec.cache_suffix
        assert callable(strategy._encode_by_vae)


def test_non_anima_latent_cache_strategy_has_no_anima_strategy_ancestry() -> None:
    from library.anima.strategy import AnimaLatentsCachingStrategy
    from library.models.family_plugins import get_family_plugin
    from library.strategies.latent_cache import FamilyLatentCacheStrategy

    args = SimpleNamespace(
        cache_latents_to_disk=False,
        vae_batch_size=1,
        skip_cache_check=False,
        vae="local-vae",
        require_cache_identity=False,
    )
    for family in ("krea2_raw", "z_image", "qwen_image_2_1", "moss_image"):
        strategy = get_family_plugin(family).build_latent_cache_strategy(args)
        assert not isinstance(strategy, AnimaLatentsCachingStrategy)
        assert isinstance(strategy, FamilyLatentCacheStrategy)
