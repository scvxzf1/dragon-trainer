"""Contract tests for the generic family plugin and inference runner surface."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.inference.family_runtime import (
    build_inference_runner,
    prepare_inference_family,
    validate_family_inference_options,
)
from library.inference.family_runner import FamilyInferenceRunner
from library.models.family_plugins import (
    get_family_plugin,
    validate_plugin_contract,
    validate_runner_contract,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "family",
    ("anima", "krea2_raw", "z_image", "qwen_image_2_1", "moss_image"),
)
def test_every_family_runner_exposes_the_generic_contract(family):
    runner = build_inference_runner(SimpleNamespace(), family)
    plugin = get_family_plugin(family)
    validate_runner_contract(runner, family)
    assert runner.supports_reference_conditioning is (
        family in {"anima", "qwen_image_2_1"}
    )
    for attribute in (
        "encode_vae",
        "sample_sigmas",
        "prepare_conditions",
        "forward_velocity_4d",
    ):
        assert callable(getattr(plugin, attribute))


def test_plugin_contract_rejects_missing_inference_surface():
    class IncompletePlugin:
        family = "anima"
        text_preprocess_spec = object()

    with pytest.raises(RuntimeError, match="callable build_tokenize_strategy"):
        validate_plugin_contract(IncompletePlugin(), "anima")


def test_build_runner_rejects_incomplete_runner(monkeypatch):
    plugin = get_family_plugin("anima")

    class IncompleteRunner:
        family = "anima"
        supports_reference_conditioning = False

        def validate(self, args, *, mode="single"):
            return None

    monkeypatch.setattr(
        plugin,
        "build_inference_runner",
        lambda args: IncompleteRunner(),
    )
    with pytest.raises(RuntimeError, match="callable prepare"):
        build_inference_runner(SimpleNamespace(), "anima")


def test_runner_prepare_is_called_once_by_generic_entrypoint(monkeypatch):
    plugin = get_family_plugin("anima")
    calls = []

    class Runner:
        family = "anima"
        supports_reference_conditioning = False

        def validate(self, args, *, mode="single"):
            calls.append(("validate", mode))

        def prepare(self, args, *, mode="single"):
            calls.append(("prepare", mode))

        def generate_latent(self, args, gen_settings):
            return None

        def save_output(self, args, latent, device):
            return None

    monkeypatch.setattr(plugin, "build_inference_runner", lambda args: Runner())
    runner = prepare_inference_family(SimpleNamespace(), "anima", "single")
    assert isinstance(runner, Runner)
    assert calls == [("validate", "single"), ("prepare", "single")]


def test_main_generates_through_family_runner_only_in_single_mode(
    monkeypatch,
    tmp_path,
):
    import inference

    calls = []

    class Runner:
        family = "anima"
        supports_reference_conditioning = False

        def generate_latent(self, args, gen_settings):
            calls.append("generate")
            return object()

        def save_output(self, args, latent, device):
            calls.append("save")

    def prepare(args, family, mode):
        calls.append(("prepare", family, mode))
        return Runner()

    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("prompt\n", encoding="utf-8")

    def run_case(args):
        calls.clear()
        monkeypatch.setattr(inference, "parse_args", lambda: args)
        monkeypatch.setattr(inference, "validate_inference_family_options", lambda *_: None)
        monkeypatch.setattr(inference, "resolve_model_family", lambda _args: "anima")
        monkeypatch.setattr(inference, "prepare_inference_family", prepare)
        monkeypatch.setattr(
            inference,
            "get_generation_settings",
            lambda _args: SimpleNamespace(
                device=torch.device("cpu"),
                runtime_dtype=torch.bfloat16,
            ),
        )
        monkeypatch.setattr(inference, "resolve_seed", lambda _args: 7)
        monkeypatch.setattr(
            inference,
            "preprocess_prompts_for_batch",
            lambda _lines, _args: ["prompt"],
        )
        monkeypatch.setattr(
            inference,
            "process_batch_prompts",
            lambda _prompts, _args: calls.append("batch"),
        )
        monkeypatch.setattr(
            inference,
            "process_interactive",
            lambda _args: calls.append("interactive"),
        )
        inference.main()
        return list(calls)

    base = {
        "latent_path": None,
        "device": "cpu",
        "seed": 7,
    }
    single_calls = run_case(
        SimpleNamespace(
            **base,
            from_file=None,
            interactive=False,
            prompt="prompt",
        )
    )
    batch_calls = run_case(
        SimpleNamespace(
            **base,
            from_file=str(prompt_file),
            interactive=False,
            prompt=None,
        )
    )
    interactive_calls = run_case(
        SimpleNamespace(
            **base,
            from_file=None,
            interactive=True,
            prompt=None,
        )
    )

    assert single_calls == [("prepare", "anima", "single"), "generate", "save"]
    assert batch_calls == [("prepare", "anima", "batch"), "batch"]
    assert interactive_calls == [
        ("prepare", "anima", "interactive"),
        "interactive",
    ]


def test_runtime_and_family_runners_do_not_restore_reverse_dependencies():
    runtime = (
        ROOT / "library" / "inference" / "family_runtime.py"
    ).read_text(encoding="utf-8")
    common = (
        ROOT / "library" / "training" / "sample_preview_common.py"
    ).read_text(encoding="utf-8")
    runner_paths = (
        ROOT / "library" / "models" / "krea2_raw" / "inference_runner.py",
        ROOT / "library" / "models" / "qwen_image_2_1" / "inference_runner.py",
    )

    assert "handlers={" not in runtime
    assert "from library.anima.training import decode_samples_for_live_preview" not in common
    assert "from inference import" not in "\n".join(
        path.read_text(encoding="utf-8") for path in runner_paths
    )


@pytest.mark.parametrize(
    "family",
    ("anima", "krea2_raw", "z_image", "qwen_image_2_1", "moss_image"),
)
def test_every_family_plugin_exposes_a_preview_decoder(family, tmp_path):
    plugin = get_family_plugin(family)
    plugin.decode_preview_samples(
        SimpleNamespace(
            is_main_process=True,
            device=torch.device("cpu"),
        ),
        SimpleNamespace(),
        None,
    )
    plugin.decode_pending_samples(
        SimpleNamespace(),
        SimpleNamespace(model_family=family, output_dir=str(tmp_path)),
        None,
    )


@pytest.mark.parametrize(
    "family,option",
    [
        ("krea2_raw", "ip_image"),
        ("z_image", "ip_adapter_weight"),
        ("moss_image", "ip_scale"),
        ("qwen_image_2_1", "easycontrol_image"),
        ("z_image", "easycontrol_weight"),
        ("moss_image", "easycontrol_image_match_size"),
        ("anima", "reference_image"),
        ("krea2_raw", "qwen_image_2_1_task"),
    ],
)
def test_inference_option_groups_reject_unowned_options(family, option):
    value = True if option.endswith("match_size") else "reference-or-weight"
    with pytest.raises(SystemExit, match="does not support these options"):
        validate_family_inference_options(SimpleNamespace(**{option: value}), family)


@pytest.mark.parametrize(
    "family,options",
    [
        (
            "anima",
            {"ip_adapter_weight": "adapter.safetensors", "ip_image": "reference.png", "ip_scale": 0.5},
        ),
        (
            "anima",
            {"easycontrol_weight": "adapter.safetensors", "easycontrol_image": "reference.png", "easycontrol_scale": 0.5},
        ),
        (
            "qwen_image_2_1",
            {"qwen_image_2_1_task": "edit", "reference_image": "reference.png"},
        ),
    ],
)
def test_inference_option_groups_allow_family_owned_options(family, options):
    validate_family_inference_options(SimpleNamespace(**options), family)


@pytest.mark.parametrize(
    "family,options",
    [
        ("anima", {"ip_adapter_weight": "adapter.safetensors"}),
        ("anima", {"ip_image": "reference.png"}),
        ("anima", {"easycontrol_weight": "adapter.safetensors"}),
        ("anima", {"easycontrol_image": "reference.png"}),
    ],
)
def test_reference_adapter_and_image_options_must_be_paired(family, options):
    with pytest.raises(SystemExit, match="must be supplied together"):
        validate_family_inference_options(SimpleNamespace(**options), family)


@pytest.mark.parametrize("family", ("anima", "krea2_raw", "z_image", "moss_image"))
def test_reference_options_are_fail_closed_for_non_reference_runners(family):
    args = SimpleNamespace(
        qwen_image_2_1_task="t2i",
        reference_image=None,
        reference_images=None,
    )
    args.qwen_image_2_1_task = None
    args.reference_images = ["reference.png"]
    with pytest.raises(SystemExit, match="does not support these options"):
        prepare_inference_family(args, family, "single")


def test_text_stage_cleans_allocator_on_success():
    cleanup = []
    runtime = FamilyInferenceRunner(
        "test_family",
        torch.device("cpu"),
        cleanup=cleanup.append,
    )
    result = runtime.run_text(
        lambda: torch.ones(1),
        lambda value: value + 1,
    )
    assert result.item() == 2
    assert cleanup == [torch.device("cpu")]


def test_moss_latent_images_writes_latent_and_renders_image(monkeypatch):
    from library.models.moss_image.inference_runner import MossImageInferenceRunner

    saved = []
    rendered = []
    monkeypatch.setattr(
        "library.inference.output.save_latent",
        lambda latent, args, height, width: saved.append(
            (latent, args.output_type, height, width)
        ),
    )
    monkeypatch.setattr(
        "library.models.moss_image.inference_runner.load_moss_vae",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "library.models.moss_image.inference_runner.decode_moss_latent",
        lambda vae, latent: "decoded",
    )
    monkeypatch.setattr(
        "library.models.moss_image.inference_runner.save_pil_image",
        lambda args, image, **kwargs: rendered.append((image, kwargs["prefix"])),
    )
    args = SimpleNamespace(
        output_type="latent_images",
        image_size=[64, 96],
        vae="vae.safetensors",
        seed=7,
        runtime_dtype="bf16",
    )
    latent = torch.zeros(1, 128, 4, 6)

    MossImageInferenceRunner().save_output(args, latent, torch.device("cpu"))

    assert saved == [(latent, "latent_images", 64, 96)]
    assert rendered == [("decoded", "moss_image")]


def test_shared_flow_engine_runs_family_hooks(monkeypatch):
    from contextlib import nullcontext

    from library.training.flow_step import run_family_flow_step

    latents = torch.ones(2, 4, 3, 5)
    seen = {}
    monkeypatch.setattr(
        torch,
        "randn_like",
        lambda tensor: torch.full_like(tensor, 3.0),
    )

    class Plugin:
        def sample_sigmas(
            self, batch_size, height, width, args, device, *, context
        ):
            return torch.tensor([0.25, 0.75])

        def prepare_conditions(
            self, batch, conditions, *, context, is_train
        ):
            return conditions

        def forward_velocity_4d(
            self, dit, noisy, conditions, sigmas, *, context
        ):
            seen["noisy"] = noisy
            seen["sigmas"] = sigmas
            return torch.zeros_like(noisy)

    context = SimpleNamespace(
        args=SimpleNamespace(
            gradient_checkpointing=False,
            weighting_scheme="none",
            min_snr_gamma=None,
            p2_gamma=1.0,
            p2_k=1.0,
            max_train_steps=10,
            gradient_accumulation_steps=1,
        ),
        accelerator=SimpleNamespace(device="cpu", autocast=nullcontext),
        network=None,
        unet=None,
        weight_dtype=torch.float32,
    )
    prediction, target, sigmas, weighting = run_family_flow_step(
        SimpleNamespace(_hydra_warmup_step=0),
        context,
        Plugin(),
        latents,
        {},
        ("conditions",),
    )

    torch.testing.assert_close(
        seen["noisy"][0], torch.full_like(latents[0], 1.5)
    )
    torch.testing.assert_close(
        seen["noisy"][1], torch.full_like(latents[1], 2.5)
    )
    torch.testing.assert_close(target, torch.full_like(latents, 2.0))
    assert torch.equal(seen["sigmas"], sigmas)
    assert prediction.shape == target.shape == latents.shape
    assert weighting.shape == (2, 1, 1, 1)
