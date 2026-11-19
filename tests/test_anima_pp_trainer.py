"""Anima PP trainer-path wiring tests (task t2, CPU only).

The Anima counterpart of ``tests/test_moss_image_pp_trainer.py``:

* the family adapter table routes ``anima`` to the Anima adapter and keeps
  every other unwired family on the historical refusal;
* the Anima admission (contract §3) names every unsupported trainer knob and
  inherits the shared MOSS-v0 table unchanged;
* the 5D wire boundary: 4D cached latents gain the explicit ``T=1`` axis, the
  max-padded ``crossattn_emb`` is never trimmed, and the padding mask is the
  all-zero attention-sink mask;
* §4 step semantics hold on a real two-rank Gloo Anima PP runtime through the
  *generic* family engine (``build_family_step_inputs`` + ``pp_train_step``),
  with the noise/timestep math coming from the same sampler registry the
  non-PP Anima trainer uses.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from library.training.pp.errors import PipelineParallelError
from library.training.pp.gate import (
    require_trainer_pp_admission,
    trainer_pp_wired,
)
from library.training.pp.trainer_admission import (
    require_anima_trainer_admission,
    require_family_trainer_admission,
    require_moss_trainer_admission,
)
from library.training.pp.trainer_adapters import (
    AnimaTrainerAdapter,
    resolve_trainer_adapter,
    trainer_resume_format,
)
from library.training.pp.trainer_session import (
    PpLoopCounters,
    pp_train_step,
)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_family": "anima",
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 2,
        "pipeline_parallel_schedule": "1f1b",
        "pipeline_parallel_split": "balanced",
        "blocks_to_swap": 0,
        "torch_compile": False,
        "compile_seq_bands": False,
        "compile_dynamic_seq": False,
        "selective_checkpoint": "off",
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
        "network_train_unet_only": True,
        "mixed_precision": "fp16",
        "use_vae_cache": True,
        "use_text_cache": True,
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
        "gradient_checkpointing": True,
        "network_module": "networks.lora_anima",
        "router_source": "none",
        "use_moe_style": False,
        "caption_dropout_rate": 0.0,
        "ip_noise_gamma": None,
        "gradient_accumulation_steps": 1,
        "weighting_scheme": "none",
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# entry routing and adapter dispatch
# ---------------------------------------------------------------------------


def test_trainer_pp_wired_includes_anima():
    assert trainer_pp_wired(_config())
    assert not trainer_pp_wired({**_config(), "pipeline_parallel": False})
    assert trainer_pp_wired({**_config(), "model_family": "krea2_raw"})
    for family in ("z_image", "qwen_image_2_1"):
        assert not trainer_pp_wired({**_config(), "model_family": family})


def test_adapter_table_dispatch():
    assert isinstance(resolve_trainer_adapter(_config()), AnimaTrainerAdapter)
    # Krea-2 gained its own adapter; families without one still refuse by name.
    assert resolve_trainer_adapter({**_config(), "model_family": "krea2_raw"}).name == "krea2_raw"
    with pytest.raises(PipelineParallelError) as excinfo:
        resolve_trainer_adapter({**_config(), "model_family": "z_image"})
    assert excinfo.value.code == "pp_trainer_path_unavailable"


def test_entry_admission_admits_anima():
    require_trainer_pp_admission(_config(), where="cli_entry")
    require_family_trainer_admission(_config(), world_size=2, where="unit-test")
    require_anima_trainer_admission(_config(), world_size=2, where="unit-test")


# ---------------------------------------------------------------------------
# Anima admission: named refusals (contract §3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"gradient_accumulation_steps": 2},
        {"mixed_precision": "bf16"},
        {"sample_prompts": "prompt.txt"},
        {"save_every_n_epochs": 1},
        {"persistent_data_loader_workers": True},
        {"torch_compile": True},
        {"compile_seq_bands": True},
        {"compile_dynamic_seq": True},
        {"router_source": "fei"},
        {"use_moe_style": "shared_A"},
        {"use_ip_adapter": True},
        {"use_easycontrol": True},
        {"use_byg": True},
        {"caption_dropout_rate": 0.1},
        {"ip_noise_gamma": 0.1},
        {"network_train_unet_only": False},
        {"loss_type": "huber"},
        {"weighting_scheme": "uniform"},
        {"stage_schedule": "0:100", "stage_schedule_enabled": True},
        {"adaptive_oom_retry": True},
        {"use_vae_cache": False, "cache_latents": False},
        {"use_text_cache": False, "cache_text_encoder_outputs": False},
    ],
)
def test_anima_admission_names_unsupported_trainer_knobs(overrides):
    with pytest.raises(PipelineParallelError) as excinfo:
        require_anima_trainer_admission(_config(**overrides), world_size=2, where="unit-test")
    assert excinfo.value.code == "pp_config_invalid", overrides


def test_moss_admission_unchanged_by_the_generalization():
    # The MOSS table keeps its exact historical shape: the Anima-only knobs
    # (compile bands / router / caption dropout / ip-noise) are invisible
    # there, while every shared knob still refuses by name.
    moss_config = {
        "model_family": "moss_image",
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 2,
        "pipeline_parallel_schedule": "1f1b",
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_vae_cache": True,
        "use_text_cache": True,
        "network_train_unet_only": True,
        "compile_seq_bands": True,
        "router_source": "fei",
        "caption_dropout_rate": 0.1,
        "ip_noise_gamma": 0.1,
    }
    require_moss_trainer_admission(moss_config, world_size=2, where="unit-test")
    with pytest.raises(PipelineParallelError) as excinfo:
        require_moss_trainer_admission(
            {**moss_config, "gradient_accumulation_steps": 2}, world_size=2, where="unit-test"
        )
    assert excinfo.value.code == "pp_config_invalid"


# ---------------------------------------------------------------------------
# wire boundary: 5D latents, untrimmed padding, zero mask
# ---------------------------------------------------------------------------


def test_wire_latents_gain_the_explicit_time_axis():
    adapter = AnimaTrainerAdapter()
    latents = torch.randn(2, 4, 8, 6)
    wire = adapter.wire_latents(latents)
    assert wire.shape == (2, 4, 1, 8, 6)
    assert adapter.wire_latents(wire) is wire  # idempotent on 5D
    with pytest.raises(PipelineParallelError):
        adapter.wire_latents(torch.randn(2, 4))


def test_wire_shape_from_batch_uses_the_full_padded_token_count():
    adapter = AnimaTrainerAdapter()
    batch = {
        "latents": torch.randn(2, 4, 8, 6),
        "text_encoder_outputs_list": [
            torch.randn(2, 12, 24),  # prompt_embeds placeholder
            None,
            None,
            None,
            torch.randn(2, 32, 24),  # max-padded crossattn_emb
        ],
    }
    latent_shape, max_text_tokens = adapter.wire_shape_from_batch(batch)
    assert latent_shape == (2, 4, 1, 8, 6)
    assert max_text_tokens == 32
    with pytest.raises(PipelineParallelError) as excinfo:
        adapter.wire_shape_from_batch({"latents": torch.randn(2, 4, 8, 6), "text_encoder_outputs_list": [None]})
    assert excinfo.value.code == "pp_config_invalid"
    with pytest.raises(PipelineParallelError, match="cache_llm_adapter_outputs"):
        adapter.wire_shape_from_batch({
            "latents": torch.randn(2, 4, 8, 6),
            "text_encoder_outputs_list": [
                torch.randn(2, 12, 24),
                None,
                None,
                None,
                torch.tensor(0.0),
            ],
        })


def test_padding_mask_is_all_zero_attention_sink():
    from library.training.pp.trainer_adapters import anima_padding_mask

    latents = torch.randn(2, 4, 1, 8, 6)
    mask = anima_padding_mask(latents)
    assert mask.shape == (2, 1, 8, 6)
    assert bool((mask == 0).all())


def test_prepare_conditions_serves_only_the_cached_max_padded_crossattn():
    adapter = AnimaTrainerAdapter()
    args = Namespace()
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    good = {
        "text_encoder_outputs_list": [None, None, None, None, torch.randn(2, 32, 24)],
    }
    conditions = adapter.prepare_conditions(
        _FakeTrainer(), args, good, accelerator=accelerator, weight_dtype=torch.float32
    )
    assert set(conditions) == {"crossattn_emb"}
    with pytest.raises(PipelineParallelError) as excinfo:
        adapter.prepare_conditions(
            _FakeTrainer(), args,
            {"text_encoder_outputs_list": [None, None, None, None, None]},
            accelerator=accelerator, weight_dtype=torch.float32,
        )
    assert excinfo.value.code == "pp_config_invalid"


# ---------------------------------------------------------------------------
# resume format stamp per family (§4)
# ---------------------------------------------------------------------------


def test_resume_format_is_family_instanceized():
    assert trainer_resume_format("anima") == "anima_pp_trainer_resume"
    assert trainer_resume_format("moss_image") == "moss_image_pp_trainer_resume"
    from library.training.pp.trainer_resume import TRAINER_RESUME_FORMAT

    assert TRAINER_RESUME_FORMAT == "moss_image_pp_trainer_resume"


def test_cross_family_trainer_payload_is_refused():
    from library.training.pp.trainer_resume import (
        capture_trainer_payload,
        validate_trainer_payload,
    )

    payload = capture_trainer_payload(
        rank=0, run_id="abc", family="anima", global_step=2,
        effective_step=2, epoch=0, batch_index=2, lr_scheduler=None, seed=1,
    )
    assert payload["format"] == "anima_pp_trainer_resume"
    validate_trainer_payload(payload, rank=0, run_id="abc", family="anima", expected_step=2)
    # An Anima run reading a MOSS stamp payload (or vice versa) is a named,
    # pre-apply refusal.
    payload["format"] = "moss_image_pp_trainer_resume"
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_trainer_payload(payload, rank=0, run_id="abc", family="anima", expected_step=2)
    assert excinfo.value.code == "pp_checkpoint_state_mismatch"


# ---------------------------------------------------------------------------
# §4 step semantics on a real two-rank Gloo Anima PP runtime
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """Only what the cached batch prep + sampler registry need."""

    _network = None
    _adapters: list = []

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_scheduler(self, args, device):
        from library.runtime.noise import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )


def _sampling_args(**overrides) -> Namespace:
    return Namespace(
        sampler="default",
        timestep_sampling="shift",
        discrete_flow_shift=2.0,
        sigmoid_scale=1.0,
        sigmoid_bias=0.0,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        t_min=None,
        t_max=None,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=None,
        **overrides,
    )


def _fake_accelerator(rank: int):
    return SimpleNamespace(
        num_processes=2,
        device=torch.device("cpu"),
        is_main_process=(rank == 0),
        process_index=rank,
    )


def _anima_batch(count=4, height=4, width=4, channels=4, context_tokens=6):
    generator = torch.Generator().manual_seed(1234)
    return {
        "latents": torch.randn(count, channels, height, width, generator=generator),
        "text_encoder_outputs_list": [
            torch.randn(count, context_tokens, 24, generator=generator),
            None,
            None,
            None,
            torch.randn(count, context_tokens, 24, generator=generator),
        ],
    }


def _pair_inputs(rank, *, scale: float = 1.0):
    from library.training.pp.trainer_adapters import ADAPTERS
    from library.training.pp.trainer_inputs import build_family_step_inputs

    batch = _anima_batch()
    batch["latents"] = batch["latents"] * scale
    return build_family_step_inputs(
        ADAPTERS["anima"],
        _FakeTrainer(),
        _sampling_args(),
        batch,
        accelerator=_fake_accelerator(rank),
        vae=None,
        vae_dtype=None,
        weight_dtype=torch.float32,
        is_first=(rank == 0),
    )


def _worker_step_semantics(rank, init_path, output_dir):
    from tests.pp_checkpoint_test_support import (
        build_distributed_model,
        init_group,
        local_scaler,
    )
    from library.training.pp.trainer_adapters import ADAPTERS
    from library.training.pp.trainer_inputs import build_family_step_inputs

    torch.set_num_threads(1)
    try:
        init_group(init_path, rank)
        model, network = build_distributed_model()
        from tests.pp_checkpoint_test_support import build_runtime

        runtime = build_runtime(rank, model, network)
        scheduler = torch.optim.lr_scheduler.LambdaLR(runtime.optimizer, lambda step: 1.0)
        counters = PpLoopCounters()
        inputs = _pair_inputs(rank)
        assert inputs.noisy_input.ndim == 5, "the Anima wire is 5D"
        assert inputs.batch_digest
        pp_train_step(runtime, inputs, counters, scheduler=scheduler)
        assert counters.global_step == 1 and counters.effective_step == 1
        assert counters.skipped == 0 and scheduler.last_epoch == 1
        if rank == 1:
            assert counters.last_loss is not None
            assert counters.last_loss == counters.last_loss  # finite
        params_before = [p.detach().clone() for p in runtime.stage_module.trainable_parameters()]

        # Same step again on identical inputs: consumed step advances, params move.
        inputs2 = _pair_inputs(rank)
        assert inputs2.batch_digest == inputs.batch_digest
        pp_train_step(runtime, inputs2, counters, scheduler=scheduler)
        assert counters.global_step == 2 and counters.effective_step == 2
        moved = any(
            not torch.equal(before, after.detach())
            for before, after in zip(params_before, runtime.stage_module.trainable_parameters())
        )
        assert moved, "an effective step must move the trainable parameters"
    except BaseException:
        from pathlib import Path as _P
        import traceback as _tb

        (_P(output_dir) / f"rank{rank}.error").write_text(_tb.format_exc())
        raise
    finally:
        torch.distributed.destroy_process_group()


def test_step_semantics_on_a_real_gloo_anima_pair(tmp_path):
    from tests.pp_checkpoint_test_support import run_pair

    run_pair(tmp_path, _worker_step_semantics)


def _worker_engine_contract(rank, init_path, output_dir):
    """Digest agreement + deterministic noise sharing across the two ranks."""

    from tests.pp_checkpoint_test_support import init_group

    torch.set_num_threads(1)
    try:
        init_group(init_path, rank)
        inputs = _pair_inputs(rank)
        gathered: list = [None] * 2
        torch.distributed.all_gather_object(gathered, (inputs.batch_digest, inputs.timesteps.tolist()))
        assert gathered[0] == gathered[1], (
            "both stages must consume the same logical batch and the same "
            "rank0-sampled timesteps"
        )
    except BaseException:
        from pathlib import Path as _P
        import traceback as _tb

        (_P(output_dir) / f"rank{rank}.error").write_text(_tb.format_exc())
        raise
    finally:
        torch.distributed.destroy_process_group()


def test_engine_shares_batch_digest_and_timesteps_across_ranks(tmp_path):
    from tests.pp_checkpoint_test_support import run_pair

    run_pair(tmp_path, _worker_engine_contract)
