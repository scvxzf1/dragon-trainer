from __future__ import annotations

from types import SimpleNamespace

import pytest

from library.models.krea2_raw.inference_runner import (
    require_krea2_checkpoint_family,
    validate_krea2_inference_args,
)


def _args(**overrides):
    values = {
        "sampler": "euler",
        "flow_shift": 3.0,
        "smc_cfg": False,
        "smc_cfg_lambda": 5.0,
        "smc_cfg_alpha": 0.2,
        "cns": None,
        "cns_strength": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_krea2_accepts_single_prompt_official_euler_defaults() -> None:
    validate_krea2_inference_args(_args(), mode="single")


@pytest.mark.parametrize("mode", ["batch", "interactive"])
def test_krea2_rejects_unsupported_inference_modes(mode: str) -> None:
    with pytest.raises(SystemExit, match="single-prompt"):
        validate_krea2_inference_args(_args(), mode=mode)


@pytest.mark.parametrize("sampler", ["er_sde", "lcm"])
def test_krea2_rejects_unsupported_samplers(sampler: str) -> None:
    with pytest.raises(SystemExit, match="only euler"):
        validate_krea2_inference_args(_args(sampler=sampler))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("flow_shift", 1.0, "flow_shift"),
        ("soft_tokens_weight", "soft.safetensors", "soft_tokens_weight"),
        ("smc_cfg", True, "smc_cfg"),
        ("smc_cfg_lambda", 4.0, "smc_cfg_lambda"),
        ("smc_cfg_alpha", 0.0, "smc_cfg_alpha"),
        ("cns", "auto", "--cns"),
        ("cns_strength", 0.0, "cns_strength"),
    ],
)
def test_krea2_rejects_unconsumed_options(key: str, value, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        validate_krea2_inference_args(_args(**{key: value}))


def test_krea2_rejects_existing_anima_only_extras() -> None:
    with pytest.raises(SystemExit, match="IP-Adapter"):
        validate_krea2_inference_args(_args(ip_adapter_weight="ip.safetensors"))


@pytest.mark.parametrize("family", [None, "anima", "unknown"])
def test_krea2_rejects_checkpoint_without_matching_family(family) -> None:
    network = SimpleNamespace(cfg=SimpleNamespace(model_family=family))
    with pytest.raises(ValueError, match="ss_model_family=krea2_raw"):
        require_krea2_checkpoint_family(network)


def test_krea2_accepts_checkpoint_with_matching_family() -> None:
    network = SimpleNamespace(cfg=SimpleNamespace(model_family="krea2_raw"))
    require_krea2_checkpoint_family(network)


@pytest.mark.parametrize(
    ("from_file", "interactive"),
    [("prompts.txt", False), (None, True)],
)
def test_inference_main_rejects_krea_modes_before_anima_strategy(
    monkeypatch, from_file, interactive
) -> None:
    import inference
    from library.inference import family_runtime

    args = _args(
        model_family="krea2_raw",
        latent_path=None,
        device="cpu",
        from_file=from_file,
        interactive=interactive,
    )
    monkeypatch.setattr(inference, "parse_args", lambda: args)
    monkeypatch.setattr(
        "library.inference.anima_legacy_runner.AnimaLegacyInferenceRunner.prepare",
        lambda *args, **kwargs: pytest.fail("Anima strategy must not be installed"),
    )

    with pytest.raises(SystemExit, match="single-prompt"):
        inference.main()
