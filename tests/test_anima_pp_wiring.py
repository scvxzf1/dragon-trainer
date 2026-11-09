"""t4: hot-file wiring, compatibility gate and named codes for Anima PP v0.

Every case here pins one of the acceptance points of the wiring task:

* anima + a valid ``pipeline_parallel`` config is no longer reported as
  ``pipeline_parallel_runtime_unavailable``, while krea2_raw / z_image /
  qwen_image_2_1 stay fail-closed with a named code;
* the compatibility key stays ``pipeline_parallel`` (the WebUI resume gate and
  the multimodel tests depend on it) and a more specific ``pp_*`` code is only
  the issue's ``code``;
* no entry point can silently fall back to DDP -- ``bootstrap`` (which every
  non-CLI entry uses), ``loop``, ``checkpoints`` and ``schedulers`` each refuse
  with a named :class:`PipelineParallelError`;
* ``microbatches < stages`` is refused with a named code (torch 2.10 rejects it
  for GPipe as well, so it is not a "use GPipe instead" situation);
* the ``1/n_microbatches`` gradient factor is applied in the loss, because
  ``stage.scale_grads()`` cannot reach the LoRA adapters.
"""

from __future__ import annotations

import dataclasses
import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.models.family_registry import (
    MODEL_FAMILY_REGISTRY,
    PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B,
    model_family_capability_catalog,
)
from library.training import cli_entry, loop
from library.training.bootstrap import TrainingBootstrap
from library.training.checkpoints import publish_state_with_bundle, save_checkpoint_state
from library.training.compat_matrix import check_training_compat
from library.training.pp import errors as pp_errors
from library.training.pp.errors import PipelineParallelError
from library.training.pp.gate import (
    PP_GRAD_SCALE_OWNER,
    lr_scheduler_num_processes,
    pipeline_parallel_runtime_issue,
    require_accelerate_checkpoint,
    require_pipeline_runtime,
    require_trainer_topology,
)
from library.training.pp.schedule import build_anima_schedule, make_last_stage_loss_fn
from library.training.schedulers import get_scheduler_fix


def _config(family: str = "anima", **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_family": family,
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 4,
        "pipeline_parallel_schedule": "1f1b",
        "pipeline_parallel_split": "balanced",
        "blocks_to_swap": 0,
        "torch_compile": False,
        "selective_checkpoint": "off",
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
        "network_train_unet_only": True,
        # Read immediately after ``require_trainer_topology`` in
        # ``bootstrap.prepare_with_accelerator``. Without them a mutated gate
        # fails on ``AttributeError: 'Namespace' object has no attribute
        # 'full_fp16'`` instead of on the assertion that the DDP path was
        # entered (t18 mutation report, F2).
        "full_fp16": False,
        "full_bf16": False,
        "mixed_precision": "fp16",
    }
    config.update(overrides)
    return config


def _pipeline_issues(result):
    return [item for item in result.errors if item.key == "pipeline_parallel"]


# ---------------------------------------------------------------------------
# compatibility gate (compat_matrix)
# ---------------------------------------------------------------------------


def test_compat_matrix_releases_anima_pipeline_parallel() -> None:
    result = check_training_compat(_config(), world_size=2)

    assert _pipeline_issues(result) == []
    assert "pipeline_parallel_runtime_unavailable" not in {
        item.code for item in result.errors
    }
    # Still an experiment: the release is announced, not silent.
    warnings = [item for item in result.warnings if item.key == "pipeline_parallel"]
    assert [item.code for item in warnings] == ["pipeline_parallel_experimental_runtime"]
    assert "pp_trainer_path_unavailable" in warnings[0].message


@pytest.mark.parametrize("family", ["krea2_raw", "z_image"])
def test_compat_matrix_keeps_unimplemented_families_fail_closed(family: str) -> None:
    issues = _pipeline_issues(check_training_compat(_config(family), world_size=2))

    assert [item.code for item in issues] == ["pipeline_parallel_runtime_unavailable"]
    assert all(item.key == "pipeline_parallel" for item in issues)
    assert "Refusing to fall back" in issues[0].message


def test_compat_matrix_rejects_pipeline_for_family_without_planner() -> None:
    issues = _pipeline_issues(
        check_training_compat(_config("qwen_image_2_1"), world_size=2)
    )

    assert [item.code for item in issues] == ["pipeline_parallel_config"]
    assert all(item.key == "pipeline_parallel" for item in issues)


def test_compat_matrix_names_microbatches_below_stages() -> None:
    issues = _pipeline_issues(
        check_training_compat(
            _config(pipeline_parallel_microbatches=1),
            world_size=2,
        )
    )

    # New detail code, unchanged key: PP v0 refuses to build any stock schedule.
    assert [item.code for item in issues] == [
        "pipeline_parallel_microbatches_below_stages"
    ]
    assert all(item.key == "pipeline_parallel" for item in issues)
    assert "ScheduleGPipe" in issues[0].message


def test_compat_matrix_gate_follows_the_family_switch(monkeypatch) -> None:
    """The switch and the gate are one decision, not two."""

    spec = MODEL_FAMILY_REGISTRY["anima"]
    downgraded = dataclasses.replace(
        spec,
        pipeline_parallel=dataclasses.replace(spec.pipeline_parallel, runtime_backend=None),
    )
    monkeypatch.setitem(MODEL_FAMILY_REGISTRY, "anima", downgraded)

    issues = _pipeline_issues(check_training_compat(_config(), world_size=2))

    assert [item.code for item in issues] == ["pipeline_parallel_runtime_unavailable"]


def test_compat_matrix_fails_closed_without_the_torch_pipelining_api(monkeypatch) -> None:
    from library.training.pp import contract

    def _unavailable():
        raise PipelineParallelError(
            pp_errors.CODE_RUNTIME_UNAVAILABLE, "torch.distributed.pipelining missing"
        )

    monkeypatch.setattr(contract, "require_pipelining_api", _unavailable)

    issues = _pipeline_issues(check_training_compat(_config(), world_size=2))

    assert [item.code for item in issues] == ["pipeline_parallel_runtime_unavailable"]
    assert "pipelining missing" in issues[0].message


def test_capability_catalog_exposes_the_anima_runtime_switch() -> None:
    catalog = {item["name"]: item for item in model_family_capability_catalog()}

    anima = catalog["anima"]["pipeline_parallel"]
    assert anima["runtime_available"] is True
    assert anima["runtime_backend"] == PIPELINE_PARALLEL_RUNTIME_BACKEND_1F1B
    for family in ("krea2_raw", "z_image"):
        assert catalog[family]["pipeline_parallel"]["runtime_available"] is False
        assert catalog[family]["pipeline_parallel"]["runtime_backend"] is None


# ---------------------------------------------------------------------------
# trainer path gates: never a silent DDP fallback
# ---------------------------------------------------------------------------


def test_require_trainer_topology_refuses_ddp_for_a_valid_pp_config() -> None:
    with pytest.raises(PipelineParallelError) as excinfo:
        require_trainer_topology(_config(), where="unit-test")

    assert excinfo.value.code == "pp_trainer_path_unavailable"
    message = str(excinfo.value)
    assert "unit-test" in message
    assert "library.training.pp.build_anima_pipeline_runtime" in message
    assert "Refusing to fall back to DDP" in message


def test_require_trainer_topology_is_a_no_op_without_pipeline_parallel() -> None:
    assert require_trainer_topology({"pipeline_parallel": False}, where="unit") is None
    assert require_trainer_topology(Namespace(), where="unit") is None


def test_require_trainer_topology_reports_config_errors_first() -> None:
    with pytest.raises(PipelineParallelError) as excinfo:
        require_trainer_topology(
            _config(pipeline_parallel_microbatches=1), where="unit-test"
        )

    assert excinfo.value.code == "pp_microbatches_below_stages"


def test_malformed_pipeline_toggle_is_a_named_error() -> None:
    with pytest.raises(PipelineParallelError) as excinfo:
        require_trainer_topology(_config(pipeline_parallel="maybe"), where="unit")

    assert excinfo.value.code == "pp_config_invalid"


class _RefusingAccelerator:
    """An accelerator that fails the test if the DDP path is entered."""

    num_processes = 2

    def print(self, *_args, **_kwargs) -> None:
        raise AssertionError("the data-parallel path must not be entered")

    def prepare(self, *_args, **_kwargs):
        raise AssertionError("accelerator.prepare must not run under pipeline_parallel")

    def save_state(self, *_args, **_kwargs):
        raise AssertionError(
            "accelerator.save_state must not run under pipeline_parallel"
        )

    def __getattr__(self, name: str):
        # The checkpoint protocol reads ``device`` / ``wait_for_everyone`` / ... on
        # its way to ``save_state``; report the violation instead of an
        # ``AttributeError`` about whichever fixture field happened to be missing.
        raise AssertionError(
            f"the data-parallel checkpoint path must not be entered (read {name!r})"
        )


class _RefusingObject:
    """Stands in for a model that the DDP path must never touch.

    ``prepare_with_accelerator`` walks ``network`` / ``unet`` /
    ``text_encoders`` right after its gate. Passing ``None`` there turned a
    mutated gate into ``AttributeError: 'NoneType' object has no attribute
    'requires_grad_'``, which says nothing about *why* the test failed; any
    attribute access here reports the violation instead (t18 F2).
    """

    def __getattr__(self, name: str):
        raise AssertionError(
            f"the data-parallel path must not be entered (touched {name!r})"
        )


class _RefusingLoopState:
    """Loop state that fails the test the moment the DDP loop body is entered.

    ``run_training_loop`` / ``_run_microbatch`` touch dozens of ``LoopState``
    fields after their gate (``epoch_to_start``, ``network``, ``optimizer`` ...).
    Fabricating all of them would make the fixture a second implementation of the
    loop; any unexpected read reports the violation instead, so a mutated gate
    fails on "the data-parallel loop must not be entered" rather than on a
    missing fixture attribute (t18 F2).
    """

    def __init__(self, args, accelerator=None) -> None:
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "accelerator", accelerator)

    def __getattr__(self, name: str):
        raise AssertionError(
            f"the data-parallel loop must not be entered (read {name!r})"
        )


def test_bootstrap_refuses_pp_before_accelerator_prepare() -> None:
    """The entry that bypasses ``cli_entry`` (WebUI queue / daemon / resume)."""

    args = Namespace(**_config())
    with pytest.raises(PipelineParallelError) as excinfo:
        TrainingBootstrap().prepare_with_accelerator(
            None,  # trainer
            args,
            _RefusingAccelerator(),
            _RefusingObject(),  # network
            None,  # optimizer
            None,  # train_dataloader
            None,  # val_dataloader
            None,  # lr_scheduler
            _RefusingObject(),  # unet
            [],  # text_encoders
            None,  # text_encoder
            None,  # vae
            torch.float16,  # vae_dtype
            torch.float16,  # weight_dtype
            False,  # train_unet
            False,  # train_text_encoder
            True,  # cache_latents
        )

    assert excinfo.value.code == "pp_trainer_path_unavailable"
    assert "bootstrap.prepare_with_accelerator" in str(excinfo.value)


def test_bootstrap_refuses_pp_before_building_optimizer() -> None:
    with pytest.raises(PipelineParallelError) as excinfo:
        TrainingBootstrap().setup_optimizer_and_dataloader(
            Namespace(**_config()),
            _RefusingAccelerator(),
            None,
            None,
            None,
            None,
        )

    assert excinfo.value.code == "pp_trainer_path_unavailable"
    assert "bootstrap.setup_optimizer_and_dataloader" in str(excinfo.value)


def test_training_loop_refuses_pp_before_the_first_epoch() -> None:
    state = _RefusingLoopState(Namespace(**_config()), _RefusingAccelerator())

    with pytest.raises(PipelineParallelError) as excinfo:
        loop.run_training_loop(None, state)

    assert excinfo.value.code == "pp_trainer_path_unavailable"
    assert "loop.run_training_loop" in str(excinfo.value)


def test_microbatch_body_refuses_pp_before_process_batch() -> None:
    state = _RefusingLoopState(Namespace(**_config()), _RefusingAccelerator())

    with pytest.raises(PipelineParallelError) as excinfo:
        loop._run_microbatch(None, state, None)

    assert excinfo.value.code == "pp_trainer_path_unavailable"
    assert "loop._run_microbatch" in str(excinfo.value)


def test_checkpoints_refuse_all_rank_save_under_pp(tmp_path) -> None:
    args = Namespace(**_config(), output_dir=str(tmp_path), output_name="run")
    # A refusing accelerator (instead of ``None``) makes a mutated checkpoint
    # guard report "accelerator.save_state must not run under pipeline_parallel"
    # rather than ``AttributeError: 'NoneType' object has no attribute
    # 'save_state'`` (t18 F2).
    accelerator = _RefusingAccelerator()

    with pytest.raises(PipelineParallelError) as excinfo:
        save_checkpoint_state(args, accelerator, epoch_no=1, global_step=10)
    assert excinfo.value.code == "pp_accelerate_checkpoint_unsupported"

    with pytest.raises(PipelineParallelError) as excinfo:
        publish_state_with_bundle(
            args,
            accelerator,
            str(tmp_path / "state"),
            epoch_no=None,
            global_step=10,
            loader_cursor=None,
        )
    assert excinfo.value.code == "pp_accelerate_checkpoint_unsupported"
    # Nothing was created for a run that cannot be resumed.
    assert list(tmp_path.iterdir()) == []


def test_require_accelerate_checkpoint_is_a_no_op_without_pipeline_parallel() -> None:
    assert require_accelerate_checkpoint({"pipeline_parallel": False}, where="unit") is None


# ---------------------------------------------------------------------------
# LR plan: one pipeline step is one step, not world_size steps
# ---------------------------------------------------------------------------


def test_lr_scheduler_num_processes_is_one_under_pp() -> None:
    assert lr_scheduler_num_processes(_config(), 2) == 1
    assert lr_scheduler_num_processes({"pipeline_parallel": False}, 2) == 2
    assert lr_scheduler_num_processes(Namespace(), 1) == 1


def _scheduler_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "optimizer_type": "adamw",
        "lr_scheduler": "cosine",
        "lr_scheduler_type": None,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 0,
        "lr_scheduler_num_cycles": 1,
        "lr_scheduler_power": 1.0,
        "lr_scheduler_timescale": None,
        "lr_scheduler_min_lr_ratio": None,
        "lr_scheduler_args": None,
        "max_train_steps": 10,
    }
    values.update(overrides)
    return Namespace(**values)


def _cosine_factor(args: Namespace, num_processes: int) -> float:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = get_scheduler_fix(args, optimizer, num_processes)
    return float(scheduler.lr_lambdas[0](args.max_train_steps - 1))


def test_get_scheduler_fix_does_not_multiply_the_lr_plan_under_pp() -> None:
    data_parallel = _cosine_factor(_scheduler_args(), 2)
    pipeline = _cosine_factor(_scheduler_args(**_config()), 2)

    # transformers' cosine uses 0.5*(1+cos(pi*progress)). With the plan doubled
    # (DP) the last configured step is only halfway down the schedule; under PP
    # it is the end of it. Same args, same world size, different plan length.
    total = _scheduler_args().max_train_steps
    last = total - 1
    expected_dp = 0.5 * (1.0 + math.cos(math.pi * last / (2 * total)))
    expected_pp = 0.5 * (1.0 + math.cos(math.pi * last / total))
    assert data_parallel == pytest.approx(expected_dp, abs=1e-6)
    assert pipeline == pytest.approx(expected_pp, abs=1e-6)
    assert pipeline < data_parallel / 10.0


# ---------------------------------------------------------------------------
# CLI preflight (cli_entry) and microbatches < stages
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, config: dict[str, object], *, created: list[object] | None = None):
    args = Namespace(**config)

    class _Parser:
        def parse_args(self, _argv):
            return args

    created = [] if created is None else created

    class _Trainer:
        def train(self, trained_args):
            created.append(trained_args)

    monkeypatch.setattr(
        cli_entry._config_schema, "populate_schema", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(cli_entry, "verify_command_line_training_args", lambda _args: None)
    monkeypatch.setattr(cli_entry, "read_config_from_file", lambda parsed, _parser: parsed)
    monkeypatch.setenv("WORLD_SIZE", "2")

    cli_entry.run_training_cli(
        setup_parser=_Parser,
        trainer_factory=_Trainer,
        install_stop_signal_handlers=lambda: None,
        install_crash_reporter=lambda _argv: None,
        argv=["train.py"],
    )
    return created


def test_cli_preflight_rejects_a_valid_anima_pipeline_config(monkeypatch) -> None:
    """A valid PP config is refused at argument-parse time (F6/F10).

    The compat layer releases anima (warning only), so the CLI is where the run
    has to stop -- and it must stop *before* the trainer is created, because
    ``bootstrap.prepare_with_accelerator`` only refuses after the DiT / VAE /
    text encoder are loaded. ``trainer_factory`` is the first model-loading step,
    so "not called" is the observable proof that no model was loaded.
    """

    with pytest.raises(PipelineParallelError) as excinfo:
        _run_cli(monkeypatch, {**_config(), "attn_mode": "torch"})

    assert excinfo.value.code == "pp_trainer_path_unavailable"
    assert "cli_entry" in str(excinfo.value)
    assert "Refusing to fall back to DDP" in str(excinfo.value)


def test_cli_preflight_never_reaches_the_trainer_for_a_pipeline_config(monkeypatch) -> None:
    """No trainer is constructed, so nothing can load a model before the refusal."""

    created: list[object] = []
    try:
        _run_cli(monkeypatch, {**_config(), "attn_mode": "torch"}, created=created)
    except PipelineParallelError:
        pass
    assert created == []


def test_cli_preflight_names_the_missing_runtime_for_other_families(monkeypatch) -> None:
    with pytest.raises(PipelineParallelError) as excinfo:
        _run_cli(monkeypatch, {**_config("krea2_raw"), "attn_mode": "torch"})

    assert excinfo.value.code == "pp_runtime_unavailable"
    assert "Refusing to fall back to ordinary Accelerate data parallelism" in str(
        excinfo.value
    )


def test_cli_preflight_rejects_microbatches_below_stages(monkeypatch) -> None:
    with pytest.raises(PipelineParallelError) as excinfo:
        _run_cli(
            monkeypatch,
            {**_config(pipeline_parallel_microbatches=1), "attn_mode": "torch"},
        )

    assert excinfo.value.code == "pp_microbatches_below_stages"


def test_require_pipeline_runtime_rejects_microbatches_below_stages() -> None:
    with pytest.raises(PipelineParallelError) as excinfo:
        require_pipeline_runtime(
            _config(pipeline_parallel_microbatches=1), world_size=2
        )

    assert excinfo.value.code == "pp_microbatches_below_stages"


# ---------------------------------------------------------------------------
# F6/F10 enabler: the CLI default must not look like a quantized base
# ---------------------------------------------------------------------------


def test_convrot_group_size_alone_is_not_a_quantized_base() -> None:
    """``--convrot_group_size`` defaults to 256 on *every* CLI invocation.

    ConvRot itself is switched on only by
    ``base_compute=w8a16_convrot|w8a8_convrot``
    (``bootstrap.maybe_apply_convrot_base``), so counting the tuning parameter
    alone made every default CLI config fail as ``pp_quantization_unsupported``
    before it could reach the intended ``pp_trainer_path_unavailable`` refusal.

    The rule lives in ``library.training.pp.contract`` only; the gate calls it.
    """

    from library.training.pp.contract import quantization_markers
    from library.training.pp.gate import require_pipeline_runtime

    cli_defaults = {**_config(), "convrot_group_size": 256, "base_compute": "bf16"}
    assert quantization_markers(cli_defaults) == []
    # The whole runtime contract must accept the CLI default shape.
    runtime = require_pipeline_runtime(cli_defaults, world_size=2)
    assert runtime.stages == 2 and runtime.microbatches == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_compute": "w8a16_convrot", "convrot_group_size": 256},
        {"base_compute": "w8a8_convrot"},
        {"base_compute": "nf4"},
        {"nf4": True},
        {"fp8": True},
        {"convrot_prequant_path": "/tmp/prequant.safetensors"},
    ],
)
def test_quantized_bases_are_still_refused_by_the_gate(
    overrides: dict[str, object],
) -> None:
    """Negative control: the gate must keep refusing every quantized base."""

    from library.training.pp.gate import require_pipeline_runtime

    with pytest.raises(PipelineParallelError) as excinfo:
        require_pipeline_runtime({**_config(), **overrides}, world_size=2)

    assert excinfo.value.code == "pp_quantization_unsupported"


# ---------------------------------------------------------------------------
# F11: a directly constructed stage module raises the package error type
# ---------------------------------------------------------------------------


def _stage_inputs(**model_overrides):
    """A ``model`` + ``block_stage`` pair good enough for the capability guards.

    Only the guard path is exercised here (no forward), so a namespace is enough:
    ``library.anima.mini_forward`` reads a handful of attributes and
    ``mod_guidance_probe`` accepts any object carrying them.
    """

    model = SimpleNamespace(
        enable_pooled_text_modulation=False,
        attn_mode="torch",
        _native_flatten=False,
        _mod_guidance_delta=torch.zeros(1),
        _mod_guidance_final_w=torch.zeros(1),
        _mod_guidance_schedule=torch.zeros(2),
    )
    for key, value in model_overrides.items():
        setattr(model, key, value)
    block_stage = SimpleNamespace(blocks=[nn.Identity(), nn.Identity()])
    return model, block_stage


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"enable_pooled_text_modulation": True}, "pp_pooled_text_modulation_unsupported"),
        ({"attn_mode": "flex"}, "pp_flex_attention_unsupported"),
        ({"_native_flatten": True}, "pp_native_flatten_unsupported"),
        (
            {
                "_mod_guidance_delta": torch.ones(1),
                "_mod_guidance_schedule": torch.zeros(7),
            },
            "pp_mod_guidance_unsupported",
        ),
    ],
)
def test_direct_stage_construction_raises_the_package_error_type(
    overrides: dict[str, object], code: str
) -> None:
    """F11: ``AnimaFirstStage`` must not leak ``AnimaPipelineUnsupported``.

    ``library.training.pp`` documents one error type for every "PP cannot own
    this" case (``PipelineParallelError`` with a stable ``pp_*`` code). The
    capability guards live in ``library.anima.mini_forward`` and raise their own
    type, so a stage module constructed directly -- bypassing
    ``placement.build_anima_stage_module``, which already converts -- used to
    leak it. The code is preserved; only the type is normalized.
    """

    from library.anima.mini_forward import AnimaPipelineUnsupported
    from library.training.pp.stages import AnimaFirstStage

    model, block_stage = _stage_inputs(**overrides)
    with pytest.raises(PipelineParallelError) as excinfo:
        AnimaFirstStage(
            model=model,
            block_stage=block_stage,
            stage_index=0,
            stages=2,
            block_range=(0, 2),
            conditioning="local",
        )

    assert excinfo.value.code == code
    assert not isinstance(excinfo.value, AnimaPipelineUnsupported)
    # The named code is still the one the mini-forward authority uses.
    assert code in str(excinfo.value)


def test_stage_construction_succeeds_for_a_supported_contract() -> None:
    """The guard conversion must not reject the ordinary configuration."""

    from library.training.pp.stages import AnimaFirstStage

    model, block_stage = _stage_inputs()
    stage = AnimaFirstStage(
        model=model,
        block_stage=block_stage,
        stage_index=0,
        stages=2,
        block_range=(0, 2),
        conditioning="local",
    )

    assert stage.is_first is True
    assert stage.stage_index == 0


# ---------------------------------------------------------------------------
# adapter gradient scaling: the loss owns 1/n_microbatches
# ---------------------------------------------------------------------------


def test_pp_schedule_disables_stage_scale_grads(monkeypatch) -> None:
    from library.training.pp import schedule as schedule_module

    captured: dict[str, object] = {}

    def _fake_1f1b(stage, **kwargs):
        captured.update(kwargs)
        return "1f1b-schedule"

    monkeypatch.setattr(
        schedule_module,
        "require_pipelining_api",
        lambda: (object, _fake_1f1b, lambda *_a, **_k: "gpipe-schedule"),
    )

    stage = SimpleNamespace(num_stages=2)
    built = build_anima_schedule(
        stage, microbatches=2, schedule="1f1b", loss_fn=lambda *_a: None
    )

    assert built == "1f1b-schedule"
    assert captured["n_microbatches"] == 2
    # scale_grads only walks stage.submod.parameters(); the LoRA adapters are not
    # in that tree, so the schedule must not be trusted with the divide.
    assert captured["scale_grads"] is False


def test_last_stage_loss_applies_one_over_microbatches() -> None:
    loss_fn = make_last_stage_loss_fn(is_last=True, microbatches=2)
    prediction = torch.zeros(2, 3, 4, 4)
    target = torch.ones(2, 3, 4, 4)

    loss = loss_fn(prediction, target)

    assert float(loss) == pytest.approx(0.5)
    assert PP_GRAD_SCALE_OWNER.endswith("make_last_stage_loss_fn")


def test_last_stage_loss_refuses_to_run_off_the_last_stage() -> None:
    loss_fn = make_last_stage_loss_fn(is_last=False, microbatches=2)

    with pytest.raises(PipelineParallelError) as excinfo:
        loss_fn(torch.zeros(1, 1), torch.zeros(1, 1))

    assert excinfo.value.code == "pp_loss_stage_violation"


def test_pipeline_parallel_runtime_issue_is_none_for_anima() -> None:
    assert pipeline_parallel_runtime_issue(_config(), world_size=2) is None
