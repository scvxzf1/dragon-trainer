"""MOSS PP trainer-path wiring tests (task t2, CPU only).

What this file proves without a GPU:

* the wired MOSS family reaches the dedicated PP trainer path from both the
  CLI preflight and the train_session dispatch decision, while every unwired
  family and every DDP-interior code path keeps the named refusal;
* the process-group contract (``cpu:gloo`` / ``cuda:nccl``, decided once at
  Accelerator construction, never rebuilt) is enforced against a real two-rank
  Gloo group;
* the trainer-level step semantics from the frozen contract §4 hold on real
  MOSS PP runtimes: ``global_step`` counts consumed steps (skips included),
  ``effective_step`` and the LR scheduler advance only on a successful step;
* the loader contract: unsharded per-epoch-deterministic batches at loader
  wrapper size 1 (the dataset item already is a batch).
"""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from library.training.pp.errors import PipelineParallelError
from library.training.pp.gate import (
    pipeline_parallel_enabled,
    require_trainer_pp_admission,
    require_trainer_topology,
    trainer_pp_wired,
)
from library.training.pp.trainer_admission import require_moss_trainer_admission
from library.training.pp.trainer_session import (
    PpLoopCounters,
    build_pp_train_dataloader,
    derive_run_id,
    pp_train_step,
    run_identity,
    validate_pp_process_group,
)

from tests.moss_pp_test_support import (
    HEIGHT,
    MAX_TEXT_TOKENS,
    WIDTH,
    init_gloo,
    tiny_batch,
    tiny_lora,
    tiny_model,
)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_family": "moss_image",
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 2,
        "pipeline_parallel_schedule": "1f1b",
        "pipeline_parallel_split": "balanced",
        "blocks_to_swap": 0,
        "torch_compile": False,
        "selective_checkpoint": "off",
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
        "network_train_unet_only": True,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_vae_cache": True,
        "use_text_cache": True,
        "network_module": "networks.lora_anima",
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# entry routing: wired MOSS reaches the PP path, nothing else does
# ---------------------------------------------------------------------------


def test_trainer_pp_wired_families():
    assert trainer_pp_wired(_config())
    # t2 generalization: Anima publishes a 1F1B runtime and now has a wired
    # trainer path (tests/test_anima_pp_trainer.py owns its admission cases).
    assert trainer_pp_wired({**_config(), "model_family": "anima"})
    assert not trainer_pp_wired({**_config(), "pipeline_parallel": False})
    # Krea-2 gained its own 1F1B runtime (verified on 2xV100), so it is wired too.
    assert trainer_pp_wired({**_config(), "model_family": "krea2_raw"})
    for family in ("z_image", "qwen_image_2_1"):
        assert not trainer_pp_wired({**_config(), "model_family": family})


def test_entry_admission_admits_wired_families_and_refuses_others():
    require_trainer_pp_admission(_config(), where="cli_entry")
    require_trainer_pp_admission(
        {**_config(), "model_family": "anima", "network_train_unet_only": True},
        where="cli_entry",
    )
    # A family without any stage-local runtime is refused one step earlier by
    # the shared runtime gate. Either way nothing reaches a DDP fallback.
    require_trainer_pp_admission({**_config(), "model_family": "krea2_raw"}, where="cli_entry")
    for family in ("z_image", "qwen_image_2_1"):
        with pytest.raises(PipelineParallelError) as excinfo:
            require_trainer_pp_admission({**_config(), "model_family": family}, where="cli_entry")
        # A named PP refusal either way (family contract or runtime gate) --
        # never a DDP fallback.
        assert excinfo.value.code.startswith("pp_")


def test_ddp_interior_guards_stay_armed_for_the_wired_family():
    for where in (
        "bootstrap.setup_optimizer_and_dataloader",
        "bootstrap.prepare_with_accelerator",
        "loop.run_training_loop",
        "loop._run_microbatch",
    ):
        with pytest.raises(PipelineParallelError) as excinfo:
            require_trainer_topology(_config(), where=where)
        assert excinfo.value.code == "pp_trainer_path_unavailable"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"gradient_accumulation_steps": 2}, "pp_config_invalid"),
        ({"mixed_precision": "bf16"}, "pp_config_invalid"),
        ({"sample_prompts": "prompt.txt"}, "pp_config_invalid"),
        ({"save_every_n_epochs": 1}, "pp_config_invalid"),
        ({"persistent_data_loader_workers": True}, "pp_config_invalid"),
    ],
)
def test_admission_names_unsupported_trainer_knobs(overrides, expected):
    with pytest.raises(PipelineParallelError) as excinfo:
        require_moss_trainer_admission(
            _config(**overrides), world_size=2, where="unit-test"
        )
    assert excinfo.value.code == expected


def test_admission_keeps_the_stage_schedule_and_oom_supervisor_out():
    from library.training.stage_schedule import stage_schedule_enabled

    assert not stage_schedule_enabled(_config())
    with pytest.raises(PipelineParallelError) as excinfo:
        require_moss_trainer_admission(
            _config(stage_schedule="0:100", stage_schedule_enabled=True), world_size=2, where="unit-test"
        )
    assert excinfo.value.code == "pp_config_invalid"
    with pytest.raises(PipelineParallelError) as excinfo:
        require_moss_trainer_admission(
            _config(adaptive_oom_retry=True), world_size=2, where="unit-test"
        )
    assert excinfo.value.code == "pp_config_invalid"


# ---------------------------------------------------------------------------
# process group contract against a real two-rank Gloo pair
# ---------------------------------------------------------------------------


def _fake_accelerator(rank: int):
    return SimpleNamespace(
        num_processes=2,
        device=torch.device("cpu"),
        is_main_process=(rank == 0),
        process_index=rank,
    )


def _worker_validate_group(rank, init_path, output_dir):
    init_gloo(init_path, rank)
    try:
        validate_pp_process_group(_fake_accelerator(rank))
    finally:
        torch.distributed.destroy_process_group()


def test_process_group_contract_holds_on_a_real_gloo_pair(tmp_path):
    from tests.moss_pp_test_support import run_pair

    run_pair(tmp_path, _worker_validate_group)


def test_process_group_contract_refuses_missing_group(tmp_path):
    """No gloo-rebuilding shim exists: without an initialized group, refuse."""

    with pytest.raises(PipelineParallelError) as excinfo:
        validate_pp_process_group(_fake_accelerator(0))
    assert excinfo.value.code == "pp_backend_multibackend_required"


def test_process_group_contract_refuses_world_smaller_than_stages():
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_pp_process_group(SimpleNamespace(num_processes=1, device=torch.device("cpu")))
    assert excinfo.value.code == "pp_world_size_mismatch"


# ---------------------------------------------------------------------------
# loader contract: unsharded, wrapper size 1, per-epoch determinism
# ---------------------------------------------------------------------------


class _FakeDatasetGroup(list):
    batch_size = 1

    def set_current_strategies(self):
        pass


def test_build_pp_train_dataloader_is_unsharded_and_deterministic():
    args = Namespace(
        seed=7,
        max_data_loader_n_workers=0,
        dataloader_pin_memory=False,
        train_batch_size=1,
    )
    group = _FakeDatasetGroup([{"latents": torch.ones(1, 128, HEIGHT, WIDTH)}] * 6)
    loader = build_pp_train_dataloader(args, group, collator=None)
    assert loader.batch_size == 1  # the dataset item already is a batch
    first = [batch["latents"].clone() for batch in loader]
    loader.generator.manual_seed(7 + 0)
    second = [batch["latents"].clone() for batch in loader]
    assert [b.shape for b in first] == [b.shape for b in second]
    same = all(torch.equal(a, b) for a, b in zip(first, second))
    assert same or len(first) == 1  # order reproducible for a single-item group


def test_run_identity_is_deterministic_and_rank_free():
    args = Namespace(seed=3, output_name="run", max_train_steps=10, output_dir="out")
    assert derive_run_id(args) == derive_run_id(args)
    identity = run_identity(derive_run_id(args), args)
    # Identity values must be sha256 content digests (validate_identity), and
    # one pipeline-wide fact: no rank component, so the checkpoint identity
    # agreement across stages cannot fail on the identity itself.
    assert identity == {"run": derive_run_id(args)}
    import re
    assert re.fullmatch(r"[0-9a-f]{64}", derive_run_id(args))


# ---------------------------------------------------------------------------
# §4 step semantics on a real two-rank Gloo MOSS PP runtime
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """Only what the cached-latent/TE batch prep needs (identity shift/scale)."""

    def shift_scale_latents(self, args, latents):
        return latents


def _pair_config():
    return dict(
        pipeline_parallel=True,
        pipeline_parallel_stages=2,
        pipeline_parallel_microbatches=2,
        pipeline_parallel_schedule="1f1b",
        mixed_precision="fp16",
    )


def _build_pair_runtime(rank):
    from library.training.pp.contract import PipelineRuntimeConfig
    from library.training.pp.moss_runtime import build_moss_pipeline_runtime

    model = tiny_model().to(dtype=torch.float16)
    network = tiny_lora(model)  # fp32 LoRA master weights on an fp16 base
    latents, _text, _mask, _sigmas, _times, _target = tiny_batch()
    config = PipelineRuntimeConfig.from_config(_pair_config())
    return build_moss_pipeline_runtime(
        model,
        config=config,
        rank=rank,
        world_size=2,
        device=torch.device("cpu"),
        latent_shape=tuple(latents.shape),
        max_text_tokens=MAX_TEXT_TOKENS,
        dtype=torch.float16,
        network=network,
        scaler=torch.amp.GradScaler("cpu") if rank == 1 else None,
        optimizer_factory=lambda module: torch.optim.AdamW(
            list(module.trainable_parameters()), lr=1e-3
        ),
        max_grad_norm=1.0,
    )


def _pair_inputs(rank, *, overflow: bool = False):
    from library.training.pp.trainer_inputs import build_step_inputs

    latents, text, mask, _sigmas, _times, _target = tiny_batch()
    scale = 400.0 if overflow else 1.0
    batch = {
        "latents": (latents * scale).to(torch.float16),
        "text_encoder_outputs_list": [text, mask],
    }
    return build_step_inputs(
        _FakeTrainer(),
        Namespace(seed=0),
        batch,
        accelerator=_fake_accelerator(rank),
        vae=None,
        vae_dtype=None,
        weight_dtype=torch.float16,
        is_first=(rank == 0),
    )


def _worker_step_semantics(rank, init_path, output_dir):
    from library.training import precision_report

    def run(trainer, args):
        runtime = _build_pair_runtime(rank)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            runtime.optimizer, lambda step: 1.0
        )
        counters = PpLoopCounters()
        precision_report.observe("accelerator_ready", _fake_accelerator(rank))
        precision_report.observe("pp_runtime_ready", runtime.scale)
        precision_report.notify("loop_ready", counters)
        pp_train_step(runtime, _pair_inputs(rank), counters, scheduler=scheduler)
        assert counters.global_step == 1 and counters.effective_step == 1
        assert counters.skipped == 0 and scheduler.last_epoch == 1
        if rank == 1:
            assert counters.last_loss is not None
            assert counters.last_loss == counters.last_loss
        params_before = [p.detach().clone() for p in runtime.stage_module.parameters()]

        # A real overflow skip: the huge target drives fp16 gradients past
        # 65504, the transaction's all-reduced verdict skips the step, the LR
        # plan and the parameters must not move.
        pp_train_step(
            runtime, _pair_inputs(rank, overflow=True), counters, scheduler=scheduler
        )
        assert counters.global_step == 2, counters
        assert counters.effective_step == 1, counters
        assert counters.skipped == 1, counters
        assert scheduler.last_epoch == 1, counters
        params_after = [p.detach() for p in runtime.stage_module.parameters()]
        for before, after in zip(params_before, params_after):
            assert torch.equal(before, after), "a skipped step must not move parameters"
        precision_report.notify("finish", counters)

    init_gloo(init_path, rank)
    try:
        precision_report.reported_session(_FakeTrainer(), Namespace(output_dir=output_dir), run)
    finally:
        torch.distributed.destroy_process_group()


def test_step_semantics_global_effective_and_skip(tmp_path):
    """§4 on real ranks: consumed steps, success steps and overflow skips."""

    from tests.moss_pp_test_support import run_pair

    run_pair(tmp_path, _worker_step_semantics)
    report_path, = tmp_path.glob("precision_report*.json")
    report = json.loads(report_path.read_text())
    assert [report["updates"][key] for key in ("attempted", "effective", "skipped", "total_global_step")] == [2, 1, 1, 2]
    assert report["amp"]["scaler_initial"] == 65536.0
    assert report["amp"]["scaler_final"] == 32768.0
