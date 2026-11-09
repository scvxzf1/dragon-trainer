"""CPU/Gloo acceptance for ``library/training/pp`` (task t2).

What this file proves
---------------------
1. **Two-rank forward + backward works** on CPU/Gloo for both conditioning
   transports, and the pipeline result matches the single-process authority
   (``Anima.forward_mini_train_dit``) for the prediction *and* for every adapter
   gradient.  This is the actual acceptance criterion for the PP v0 core.
2. **The frozen-boundary grad wire is paired.**  ``conditioning="wire"`` puts the
   cached conditioning on the wire; the later stage detaches it and the earlier
   stage allocates gradient buffers for the hidden state only.  The buffers are
   asserted directly (``grad_send_info`` / ``grad_recv_info``) so a regression
   that reintroduces the torch 2.10 one-sided override fails loudly instead of
   hanging.
3. **The fp16 transaction is symmetric.**  One rank overflowing must skip the
   step on *both* ranks, and both ranks must advance the fp16 scale identically.
4. **A one-sided frozen-slot declaration is refused**, not deadlocked.
5. **PP never degrades to DDP.**  Every unavailable/invalid configuration raises
   a :class:`PipelineParallelError` with a stable ``pp_*`` code.

The micro-model fixture de-zeros the AdaLN modulation projections: ``Block`` and
``FinalLayer`` zero-initialise their last AdaLN projection, so on a fresh model
the whole timestep-modulation path is a no-op and any numeric comparison would
pass for the wrong reason.
"""

from __future__ import annotations

import re
import traceback
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from library.anima.models import Anima
from library.training.pp import (
    CONDITIONING_LOCAL,
    CONDITIONING_WIRE,
    AnimaPipelineRuntime,
    PipelineParallelError,
    PipelineRuntimeConfig,
    assert_uniform_token_bucket,
    boundary_metas,
    build_anima_pipeline_runtime,
    build_anima_schedule,
    build_anima_stage_layout,
    build_anima_stage_module,
    build_pipeline_stage,
    latent_to_model_5d,
    make_last_stage_loss_fn,
    model_pred_to_loss_4d,
    quantization_markers,
    release_foreign_blocks,
    token_family,
    PpComputePolicy,
    validate_distributed_backend,
    validate_master_weight_dtype,
    validate_mixed_precision,
    validate_no_quantization,
    validate_runtime_config,
)
from library.training.pp.errors import (
    CODE_BACKEND_MULTIBACKEND_REQUIRED,
    CODE_FP16_MASTER_WEIGHTS_REQUIRED,
)
from library.training.pp.stages import (
    WIRE_FROZEN_INDICES,
    FrozenBoundaryPipelineStage,
    assert_frozen_boundary_agreement,
)

BLOCKS = 4
CHANNELS = 24
BATCH = 4
SPLIT_STAGE = 2


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _tiny_model(seed: int = 123) -> Anima:
    """A 4-block Anima small enough to run the whole forward on CPU.

    The AdaLN output projections are de-zeroed because the stock model is
    *conditioning-dead*: ``Block`` zeroes ``adaln_up_self_attn`` /
    ``adaln_up_cross_attn`` / ``adaln_up_mlp`` (``models.py:1468``) and
    ``FinalLayer`` zeroes its last modulation projection (``models.py:1472-1474``).
    De-zeroing only ``adaln_modulation`` leaves the *per-block* timestep path dead
    -- measured: zeroing the block-stack embedding moved the output by exactly
    ``0.0`` -- so a pipeline-vs-baseline comparison would stay green even if the
    first stage fed the wrong microbatch timesteps.  Both name families are
    de-zeroed here, mirroring ``tests/pp_equivalence_test_support.py``.
    """

    torch.manual_seed(seed)
    model = Anima(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=CHANNELS,
        num_blocks=BLOCKS,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=CHANNELS,
        pos_emb_learnable=True,
        use_adaln_lora=True,
        adaln_lora_dim=8,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    model.requires_grad_(False)
    _dezero_adaln(model, seed=seed)
    return model.train()


def _dezero_adaln(model, *, seed: int) -> None:
    """Replace every zero-initialised AdaLN output projection with ``N(0, 0.1)``.

    ``adaln_modulation`` covers ``FinalLayer``'s modulation stack; ``adaln_up_``
    covers the per-block AdaLN-LoRA up projections (``adaln_up_self_attn`` /
    ``adaln_up_cross_attn`` / ``adaln_up_mlp``).  Missing the latter is what made
    the earlier fixture's equivalence assertions vacuous.
    """

    generator = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() < 2:
                continue
            if "adaln_modulation" not in name and "adaln_up_" not in name:
                continue
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _tiny_lora(model, *, seed: int = 124):
    from scripts.experiments.anima_parallel.common import create_mlp_lora

    network = create_mlp_lora(model, seed=seed, rank_dim=4, alpha=4.0)
    generator = torch.Generator().manual_seed(seed + 3)
    with torch.no_grad():
        for lora in network.unet_loras:
            lora.lora_up.weight.copy_(
                torch.randn(lora.lora_up.weight.shape, generator=generator) * 0.02
            )
    return network


def _tiny_batch(count: int = BATCH):
    generator = torch.Generator().manual_seed(999)
    latents = torch.randn(count, 4, 1, 4, 4, generator=generator)
    timesteps = torch.rand(count, generator=generator)
    context = torch.randn(count, 3, CHANNELS, generator=generator)
    padding_mask = torch.zeros(count, 1, 4, 4)
    target = torch.randn(count, 4, 4, 4, generator=generator)
    return latents, timesteps, context, padding_mask, target


def _baseline_grads(latents, timesteps, context, padding_mask, target, *, seed=123):
    """Single-process reference: the authoritative forward, one backward."""

    model = _tiny_model(seed)
    network = _tiny_lora(model, seed=seed + 1)
    model.zero_grad(set_to_none=True)
    prediction = model.forward_mini_train_dit(
        latents, timesteps, context, padding_mask=padding_mask
    )
    loss = F.mse_loss(model_pred_to_loss_4d(prediction), target)
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in network.named_parameters()
        if parameter.grad is not None
    }
    prediction4d = model_pred_to_loss_4d(prediction).detach().clone()
    return prediction4d, grads, float(loss.detach())


def _index_from_name(name: str) -> int:
    match = re.search(r"lora_unet_blocks_(\d+)_", name)
    if match is None:
        raise AssertionError(f"cannot resolve block index from {name!r}")
    return int(match.group(1))


def _stage_lora_parameters(network, block_range):
    owned = set(range(block_range[0], block_range[1]))
    return [
        parameter
        for lora in network.unet_loras
        if _index_from_name(lora.lora_name) in owned
        for parameter in lora.parameters()
    ]


# ---------------------------------------------------------------------------
# single-process contract tests
# ---------------------------------------------------------------------------


def test_latent_contract_never_uses_a_bare_squeeze():
    latent = torch.randn(1, 4, 4, 4)
    five = latent_to_model_5d(latent)
    assert five.shape == (1, 4, 1, 4, 4)
    back = model_pred_to_loss_4d(five)
    assert back.shape == (1, 4, 4, 4)
    # A batch of 1 must survive: ``squeeze()`` would have collapsed dim 0 too.
    assert back.shape[0] == 1
    with pytest.raises(PipelineParallelError) as excinfo:
        model_pred_to_loss_4d(torch.randn(1, 4, 2, 4, 4))
    assert excinfo.value.code == "pp_shape_contract_violation"


def test_single_token_bucket_contract():
    hidden = (2, 1, 60, 70, CHANNELS)
    assert token_family(hidden) == "4200"
    assert token_family((2, 1, 63, 64, CHANNELS)) == "4032"
    assert assert_uniform_token_bucket([hidden, hidden]) == 4200
    with pytest.raises(PipelineParallelError) as excinfo:
        assert_uniform_token_bucket([hidden, (2, 1, 63, 64, CHANNELS)])
    assert excinfo.value.code == "pp_token_bucket_mismatch"


def test_runtime_config_fails_closed():
    config = PipelineRuntimeConfig(stages=2, microbatches=1)
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_runtime_config(config, world_size=2)
    assert excinfo.value.code == "pp_microbatches_below_stages"

    with pytest.raises(PipelineParallelError) as excinfo:
        validate_runtime_config(
            PipelineRuntimeConfig(stages=2, microbatches=2), world_size=3
        )
    assert excinfo.value.code == "pp_world_size_mismatch"

    with pytest.raises(PipelineParallelError) as excinfo:
        validate_runtime_config(
            PipelineRuntimeConfig(stages=2, microbatches=2, schedule="interleaved")
        )
    assert excinfo.value.code == "pp_schedule_unsupported"

    with pytest.raises(PipelineParallelError) as excinfo:
        validate_mixed_precision(PipelineRuntimeConfig(mixed_precision="bf16"), has_scaler=True)
    assert excinfo.value.code == "pp_mixed_precision_unsupported"

    with pytest.raises(PipelineParallelError) as excinfo:
        validate_mixed_precision(PipelineRuntimeConfig(), has_scaler=False)
    assert excinfo.value.code == "pp_grad_scaler_missing"


@pytest.mark.parametrize(
    "config",
    [
        {"nf4": True},
        {"fp8": True},
        {"use_nf4": True},
        {"quantize_base": True},
        {"nf4_prequantized_path": "/tmp/x.safetensors"},
        {"convrot_prequant_path": "/tmp/y.safetensors"},
        # ``base_compute`` is the runtime's own switch for a quantized base.
        {"base_compute": "w8a16_convrot"},
        {"base_compute": "w8a8_convrot"},
        {"base_compute": "w8a16"},  # alias, normalized by the runtime authority
        {"base_compute": "nf4"},
        # ConvRot *selected*, so the group size is real information.
        {"base_compute": "w8a16_convrot", "convrot_group_size": 128},
    ],
)
def test_quantized_bases_are_refused(config):
    assert quantization_markers(config)
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_no_quantization(config)
    assert excinfo.value.code == "pp_quantization_unsupported"


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"base_compute": "bf16"},
        {"base_compute": "fp16"},  # alias of bf16 in normalize_base_compute
        {"base_compute": "none"},
        # ``--convrot_group_size`` defaults to 256 on *every* CLI invocation
        # (library/training/cli_args.py); on its own it is a tuning parameter,
        # not a quantized base.  Counting it unconditionally rejected every
        # default configuration.
        {"convrot_group_size": 256},
        {"base_compute": "bf16", "convrot_group_size": 256},
    ],
)
def test_unquantized_defaults_are_not_refused(config):
    assert quantization_markers(config) == []
    validate_no_quantization(config)  # must not raise


def test_assert_runtime_capabilities_refuses_a_convrot_base():
    """Negative control: unifying the rule must not weaken the real refusal.

    ``assert_runtime_capabilities`` is the pre-construction preflight; it has to
    reject a quantized base *and* accept the default CLI shape, where
    ``convrot_group_size=256`` is only the argparse default.
    """

    from library.training.pp import assert_runtime_capabilities

    assert_runtime_capabilities({"convrot_group_size": 256, "base_compute": "bf16"})

    for config in (
        {"base_compute": "w8a16_convrot", "convrot_group_size": 256},
        {"base_compute": "w8a8_convrot"},
        {"base_compute": "nf4"},
        {"nf4": True},
        {"fp8": True},
        {"nf4_prequantized_path": "/tmp/x.safetensors"},
    ):
        with pytest.raises(PipelineParallelError) as excinfo:
            assert_runtime_capabilities(config)
        assert excinfo.value.code == "pp_quantization_unsupported"


def test_convrot_group_size_counts_only_when_convrot_is_selected():
    """The single rule behind both ``quantization_markers`` and the gate."""

    from library.training.pp.contract import convrot_selected

    assert convrot_selected({"base_compute": "bf16"}) is False
    assert convrot_selected({"convrot_group_size": 256}) is False
    assert convrot_selected({"base_compute": "fp16"}) is False  # alias of bf16
    assert convrot_selected({"base_compute": "w8a16"}) is True  # alias
    assert convrot_selected({"base_compute": "w8a8_convrot"}) is True
    assert convrot_selected({"base_compute": "nf4"}) is False  # quantized, not ConvRot


def test_gpipe_also_requires_microbatches_at_least_stages():
    class _Stage:
        num_stages = 2

    for schedule in ("1f1b", "gpipe"):
        with pytest.raises(PipelineParallelError) as excinfo:
            build_anima_schedule(
                _Stage(), microbatches=1, schedule=schedule, loss_fn=lambda *_args, **_kwargs: None
            )
        assert excinfo.value.code == "pp_microbatches_below_stages"


def test_loss_carries_the_one_over_microbatches_factor():
    """The 1/n factor must live in the loss, not in ``scale_grads``.

    ``PipelineStage.scale_grads`` only walks ``self.submod.parameters()``, and the
    LoRA adapters are attached to the adapter network rather than to the blocks
    they patch, so ``scale_grads=True`` is a silent no-op for them: every adapter
    gradient comes out ``n_microbatches`` times too large.  PP v0 therefore
    divides inside the loss and passes ``scale_grads=False``; this test pins the
    factor so a future "cleanup" cannot move it back.
    """

    prediction = torch.randn(2, 4, 4, 4)
    target = torch.randn(2, 4, 4, 4)
    raw = F.mse_loss(prediction.float(), target.float())
    for microbatches in (2, 3, 7):
        loss_fn = make_last_stage_loss_fn(is_last=True, microbatches=microbatches)
        value = loss_fn(prediction, target)
        torch.testing.assert_close(value, raw / microbatches, rtol=0, atol=0)


def test_fixture_conditioning_paths_are_live():
    """A de-zeroed fixture must actually depend on the timestep modulation.

    Two levels are checked, because they fail independently:

    * the **whole forward** must move when the timesteps change (that is
      ``FinalLayer``'s modulation path);
    * the **block stack** must move when its embedding input is zeroed.  This is
      the per-block path: it is dead whenever ``adaln_up_*`` stays at its
      zero-initialised value, and a dead per-block path makes the pipeline-vs-
      baseline equivalence assertions in this file vacuous -- a first stage that
      fed the wrong microbatch timesteps would still compare green.
    """

    from library.anima import mini_forward

    model = _tiny_model()
    latents, timesteps, context, padding_mask, _target = _tiny_batch(2)
    with torch.no_grad():
        first = model.forward_mini_train_dit(
            latents, timesteps, context, padding_mask=padding_mask
        )
        second = model.forward_mini_train_dit(
            latents, torch.flip(timesteps, dims=(0,)), context, padding_mask=padding_mask
        )
    assert not torch.equal(first, second), (
        "the fixture ignores the timestep embedding, so every conditioning "
        "comparison in this file would be vacuous"
    )

    with torch.no_grad():
        hidden, embedding, params, kwargs = mini_forward.prepare_block_inputs(
            model, latents, timesteps, context, padding_mask
        )
        conditioned = mini_forward.run_local_blocks(
            model, hidden, embedding, context, params, kwargs
        )
        unconditioned = mini_forward.run_local_blocks(
            model, hidden, torch.zeros_like(embedding), context, params, kwargs
        )
    block_diff = float((conditioned - unconditioned).abs().max())
    assert block_diff > 1e-3, (
        "the per-block timestep conditioning path is dead (block-stack output "
        f"differs by only {block_diff:.3e} when the embedding is zeroed); the "
        "fixture must de-zero adaln_up_* as well as adaln_modulation"
    )


def test_loss_fn_refuses_a_non_last_stage():
    loss_fn = make_last_stage_loss_fn(is_last=False, microbatches=2)
    with pytest.raises(PipelineParallelError) as excinfo:
        loss_fn(torch.zeros(2, 4, 4, 4), torch.zeros(2, 4, 4, 4))
    assert excinfo.value.code == "pp_loss_stage_violation"


def test_placement_is_non_destructive_and_owns_no_parameters():
    model = _tiny_model()
    network = _tiny_lora(model)
    block_ids = [id(block) for block in model.blocks]
    first_layout, first = build_anima_stage_module(
        model, stage_index=0, stages=2, conditioning=CONDITIONING_LOCAL, network=network
    )
    second_layout, second = build_anima_stage_module(
        model, stage_index=1, stages=2, conditioning=CONDITIONING_LOCAL, network=network
    )
    # The production model keeps every block: placement must not steal modules.
    assert [id(block) for block in model.blocks] == block_ids
    assert len(model.blocks) == BLOCKS
    assert first_layout.block_range == (0, SPLIT_STAGE)
    assert second_layout.block_range == (SPLIT_STAGE, BLOCKS)
    assert first_layout.is_first and second_layout.is_last
    # The stage must not register the frozen base model as a child.
    assert list(first.parameters()) == []
    assert list(second.parameters()) == []
    assert first.state_dict() == {}
    owned_ids: set[int] = set()
    for stage, layout in ((first, first_layout), (second, second_layout)):
        owned = {
            id(parameter)
            for parameter in _stage_lora_parameters(network, layout.block_range)
        }
        assert owned
        assert {id(parameter) for parameter in stage.trainable_parameters()} == owned
        assert not owned_ids & owned
        owned_ids |= owned
    # Every adapter is owned by exactly one stage.
    assert owned_ids == {id(p) for p in network.parameters()}


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda model: setattr(model, "enable_pooled_text_modulation", True),
         "pp_pooled_text_modulation_unsupported"),
        (lambda model: setattr(model, "attn_mode", "flex"),
         "pp_flex_attention_unsupported"),
        (lambda model: setattr(model, "_native_flatten", True),
         "pp_native_flatten_unsupported"),
        (lambda model: setattr(model, "blocks_to_swap", 2),
         "pp_stage_contract_violation"),
        (lambda model: model._mod_guidance_delta.fill_(0.5),
         "pp_mod_guidance_unsupported"),
    ],
)
def test_placement_refuses_unsupported_capabilities(mutate, code):
    model = _tiny_model()
    mutate(model)
    with pytest.raises(PipelineParallelError) as excinfo:
        build_anima_stage_module(model, stage_index=0, stages=2)
    assert excinfo.value.code == code


def test_release_foreign_blocks_is_opt_in():
    model = _tiny_model()
    layout, stage = build_anima_stage_module(model, stage_index=1, stages=2)
    assert len(model.blocks) == BLOCKS
    release_foreign_blocks(model, layout)
    assert len(model.blocks) == BLOCKS - SPLIT_STAGE
    # The stage still holds its own slice after the release.
    assert len(stage.block_stage.blocks) == BLOCKS - SPLIT_STAGE


def test_boundary_metas_are_per_microbatch_shapes():
    model = _tiny_model()
    layout, _ = build_anima_stage_module(model, stage_index=0, stages=2)
    inputs, outputs = boundary_metas(
        model,
        layout,
        latent_shape=(4, 4, 1, 4, 4),
        microbatches=2,
        policy=PpComputePolicy.from_dtype(torch.float32),
        conditioning=CONDITIONING_LOCAL,
    )
    assert tuple(inputs[0].shape) == (2, 4, 1, 4, 4)
    assert tuple(outputs[0].shape) == (2, 1, 2, 2, CHANNELS)
    wire_inputs, wire_outputs = boundary_metas(
        model,
        layout,
        latent_shape=(4, 4, 1, 4, 4),
        microbatches=2,
        policy=PpComputePolicy.from_dtype(torch.float32),
        conditioning=CONDITIONING_WIRE,
        timesteps_shape=(4,),
        context_shape=(4, 3, CHANNELS),
    )
    assert len(wire_outputs) == 4
    assert tuple(wire_outputs[1].shape) == (2, 1, CHANNELS)
    assert tuple(wire_outputs[2].shape) == (2, 3, CHANNELS)
    assert tuple(wire_outputs[3].shape) == (2, 1, 3 * CHANNELS)
    assert tuple(wire_inputs[0].shape) == (2, 4, 1, 4, 4)


# ---------------------------------------------------------------------------
# D3 / D4: process-group backend and fp16 master weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["gloo", "nccl", "", None])
def test_cuda_pp_requires_a_multi_backend_process_group(backend):
    """D3: a CUDA PP run needs ``cpu:gloo,cuda:nccl``, not a single backend.

    NCCL is required for the device activations and Gloo for the CPU scalars the
    fp16 transaction exchanges; a single-backend group cannot serve both.
    """

    with pytest.raises(PipelineParallelError) as excinfo:
        validate_distributed_backend(backend, device_type="cuda", world_size=2)
    assert excinfo.value.code == CODE_BACKEND_MULTIBACKEND_REQUIRED
    assert "cpu:gloo,cuda:nccl" in str(excinfo.value)


@pytest.mark.parametrize(
    "backend", ["cpu:gloo,cuda:nccl", "cuda:nccl,cpu:gloo", "gloo,nccl"]
)
def test_multi_backend_process_group_is_accepted(backend):
    validate_distributed_backend(backend, device_type="cuda", world_size=2)


def test_cpu_pp_does_not_require_a_multi_backend_group():
    """The Gloo-only CPU harness is legitimate; the rule is CUDA-only."""

    validate_distributed_backend("gloo", device_type="cpu", world_size=2)


def test_fp16_pp_requires_fp32_master_weights():
    """D4: GradScaler refuses fp16 gradients, so fp16 PP needs fp32 masters."""

    fp32 = [torch.nn.Parameter(torch.zeros(2))]
    bf16 = [torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))]
    validate_master_weight_dtype(fp32, mixed_precision="fp16")
    validate_master_weight_dtype(bf16, mixed_precision=None)
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_master_weight_dtype(bf16, mixed_precision="fp16")
    assert excinfo.value.code == CODE_FP16_MASTER_WEIGHTS_REQUIRED
    assert "torch.bfloat16" in str(excinfo.value)


def test_frozen_fp16_base_model_is_allowed_under_fp16():
    """Only *trainable* parameters need fp32 masters; the frozen base may be fp16."""

    frozen = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16), requires_grad=False)
    validate_master_weight_dtype([frozen], mixed_precision="fp16")


# ---------------------------------------------------------------------------
# D1: use_fp32 is one pipeline decision, not a boundary inference
# ---------------------------------------------------------------------------


def test_use_fp32_is_decided_once_from_the_pre_block_dtype():
    """D1: the fp16 tail must not re-derive ``use_fp32`` from the boundary.

    ``models.py:2665`` decides ``use_fp32`` from the **pre-block hidden** dtype
    (fp16 -> ``True``) and applies it to every block.  When it is true the block
    stack promotes the residual stream (``models.py:1519``), so the tensor that
    crosses the PP boundary is **fp32 even though the run is fp16**.  A stage that
    re-derived the flag from the tensor it received therefore got ``False`` and ran
    its blocks in fp16 -- T8 measured ``8.665e4 > 65504`` -> non-finite, first bad
    block ``idx=32``.

    This test pins the rule rather than a constant: the value every stage uses must
    equal ``resolve_use_fp32(pre_block_dtype)``, and the boundary-derived value is
    asserted to be *different* so the assertion cannot pass by accident on a
    fixture where the two sources happen to agree.
    """

    from library.anima.mini_forward import resolve_use_fp32

    model = _tiny_model()
    latents, timesteps, context, padding_mask, _target = _tiny_batch(2)
    policy = PpComputePolicy.from_dtype(torch.float16)
    assert policy.use_fp32 is True
    assert policy.boundary_dtype is torch.float32

    _, first = build_anima_stage_module(
        model, stage_index=0, stages=2, compute_policy=policy
    )
    _, last = build_anima_stage_module(
        model, stage_index=1, stages=2, compute_policy=policy
    )
    # Built without a policy the module falls back to the model dtype (fp32),
    # which is exactly the value the old boundary inference produced.
    _, legacy_last = build_anima_stage_module(model, stage_index=1, stages=2)

    with torch.autocast("cpu", dtype=torch.float16):
        boundary = first(
            latents, timesteps=timesteps, crossattn_emb=context, padding_mask=padding_mask
        )
        prediction = last(
            boundary,
            timesteps=timesteps,
            crossattn_emb=context,
            padding_mask=padding_mask,
        )

    # The dtype topology that makes the old inference wrong: fp16 run, fp32 wire.
    assert boundary.dtype is torch.float32
    assert prediction.dtype is torch.float16
    assert resolve_use_fp32(torch.float16) is True
    assert resolve_use_fp32(boundary.dtype) is False, (
        "the fixture is not in the D1 regime: the boundary dtype must disagree "
        "with the pre-block dtype"
    )

    assert first.use_fp32 is True
    assert last.use_fp32 is True
    assert first.use_fp32 is last.use_fp32
    assert first.use_fp32 is resolve_use_fp32(torch.float16)
    assert legacy_last.use_fp32 is False, (
        "the boundary-derived value must still be False, otherwise this test no "
        "longer demonstrates what D1 fixed"
    )
    assert torch.isfinite(prediction).all()


# ---------------------------------------------------------------------------
# two-rank Gloo workers
# ---------------------------------------------------------------------------


def _init(init_path, rank):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )


def _distributed_model(seed=123):
    model = _tiny_model(seed)
    network = _tiny_lora(model, seed=seed + 1)
    return model, network


def _build_runtime(
    rank,
    model,
    network,
    *,
    conditioning,
    microbatches,
    batch,
    scaler_factory,
    dtype=torch.float32,
    mixed_precision="fp16",
):
    layout, _ = build_anima_stage_module(
        model,
        stage_index=rank,
        stages=2,
        conditioning=conditioning,
        network=network,
        compute_policy=PpComputePolicy.from_dtype(dtype),
    )
    params = _stage_lora_parameters(network, layout.block_range)
    assert params, "each stage must own at least one adapter parameter"
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    config = PipelineRuntimeConfig(
        stages=2,
        microbatches=microbatches,
        conditioning=conditioning,
        mixed_precision=mixed_precision,
    )
    return build_anima_pipeline_runtime(
        model,
        config=config,
        world_size=2,
        rank=rank,
        device=torch.device("cpu"),
        latent_shape=(batch, 4, 1, 4, 4),
        dtype=dtype,
        timesteps_shape=(batch,),
        context_shape=(batch, 3, CHANNELS),
        scaler=scaler_factory(rank),
        optimizer=optimizer,
        network=network,
        group=dist.group.WORLD,
    )


def _local_scaler(rank):
    # Only the last stage owns the loss, so only it owns a GradScaler.
    if rank != 1:
        return None
    return torch.amp.GradScaler("cpu", init_scale=8.0)


def _worker_pipeline(rank, init_path, conditioning, microbatches, batch, root):
    torch.set_num_threads(1)
    _init(init_path, rank)
    try:
        model, network = _distributed_model()
        runtime = _build_runtime(
            rank,
            model,
            network,
            conditioning=conditioning,
            microbatches=microbatches,
            batch=batch,
            scaler_factory=_local_scaler,
        )
        latents, timesteps, context, padding_mask, target = _tiny_batch(batch)
        runtime.zero_grad()
        result = runtime.step(
            latents,
            timesteps=timesteps,
            crossattn_emb=context,
            target=target,
            padding_mask=padding_mask,
            losses=[],
        )
        outcome = runtime.optimizer_step()
        assert outcome.skipped is False, outcome
        payload = {
            "prediction": None if result.prediction is None else result.prediction.detach(),
            "losses": result.losses,
            "grads": {
                name: parameter.grad.detach().clone()
                for name, parameter in network.named_parameters()
                if parameter.grad is not None
            },
            "scale": runtime.scale,
            "skipped": outcome.skipped,
            "frozen_input_indices": list(runtime.stage.frozen_input_indices),
            "frozen_output_indices": list(runtime.stage.frozen_output_indices),
            "n_grad_recv": len(runtime.stage.grad_recv_info[0])
            if runtime.stage.grad_recv_info
            else 0,
            "grad_send_info": None
            if runtime.stage.grad_send_info is None
            else list(runtime.stage.grad_send_info),
            "topology": runtime.topology(),
        }
        torch.save(payload, Path(root) / f"rank{rank}.pt")
        (Path(root) / f"rank{rank}.ok").touch()
    except Exception:
        (Path(root) / f"rank{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def _worker_fp16_policy(rank, init_path, root):
    """D1+D2 end to end: an fp16 pipeline step with a fp32 boundary.

    The two stages must report the same ``use_fp32`` (resolved from the fp16
    pre-block dtype, not from the fp32 boundary), and the boundary metas must be
    able to say "activation fp32, prediction fp16" -- with one shared dtype the
    first fp16 step raised ``PipeliningShapeError ... expected torch.float32
    actual torch.float16``.
    """

    torch.set_num_threads(1)
    _init(init_path, rank)
    try:
        model, network = _distributed_model()
        runtime = _build_runtime(
            rank,
            model,
            network,
            conditioning=CONDITIONING_LOCAL,
            microbatches=2,
            batch=2,
            scaler_factory=_local_scaler,
            dtype=torch.float16,
            mixed_precision="fp16",
        )
        latents, timesteps, context, padding_mask, target = _tiny_batch(2)
        runtime.zero_grad()
        with torch.autocast("cpu", dtype=torch.float16):
            result = runtime.step(
                latents,
                timesteps=timesteps,
                crossattn_emb=context,
                target=target,
                padding_mask=padding_mask,
                losses=[],
            )
        prediction = result.prediction
        torch.save(
            {
                "rank": rank,
                "runtime_use_fp32": runtime.use_fp32,
                "stage_use_fp32": runtime.stage_module.use_fp32,
                "policy": runtime.compute_policy.as_dict(),
                "inputs_meta": [str(meta.dtype) for meta in runtime.stage.inputs_meta],
                # torch stores the forward output metadata as ``_outputs_meta``
                "outputs_meta": [
                    str(meta.dtype) for meta in runtime.stage._outputs_meta
                ],
                "prediction_dtype": None if prediction is None else str(prediction.dtype),
                "prediction_finite": (
                    None if prediction is None else bool(torch.isfinite(prediction).all())
                ),
            },
            Path(root) / f"fp16_{rank}.pt",
        )
        (Path(root) / f"fp16_{rank}.ok").touch()
    except Exception:
        (Path(root) / f"fp16_{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def _worker_scale_policy(rank, init_path, root):
    """Non-default fp16 policy: both stages must stay at the same scale.

    Regression for the silent 2x: the non-last stage used to fall back to the
    torch defaults (growth_interval=2000) while the last stage grew every 10
    clean steps, so after 12 steps the last stage unscaled by 16.0 and the first
    stage by 8.0 -- a factor-of-two error on the boundary gradients with nothing
    raising.
    """

    torch.set_num_threads(1)
    _init(init_path, rank)
    try:
        model, network = _distributed_model()

        def factory(stage_rank):
            if stage_rank != 1:
                return None
            return torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=10)

        runtime = _build_runtime(
            rank,
            model,
            network,
            conditioning=CONDITIONING_LOCAL,
            microbatches=2,
            batch=2,
            scaler_factory=factory,
        )
        latents, timesteps, context, padding_mask, target = _tiny_batch(2)
        trace = []
        for _ in range(12):
            runtime.zero_grad()
            runtime.step(
                latents,
                timesteps=timesteps,
                crossattn_emb=context,
                target=target,
                padding_mask=padding_mask,
                losses=[],
            )
            outcome = runtime.optimizer_step()
            trace.append(
                {
                    "scale": outcome.scale,
                    "next_scale": outcome.next_scale,
                    "skipped": outcome.skipped,
                }
            )
        torch.save(
            {
                "trace": trace,
                "final_scale": runtime.scale,
                "growth_interval": runtime.transaction.growth_interval,
                "growth_factor": runtime.transaction.growth_factor,
                "backoff_factor": runtime.transaction.backoff_factor,
            },
            Path(root) / f"policy{rank}.pt",
        )
        (Path(root) / f"policy{rank}.ok").touch()
    except Exception:
        (Path(root) / f"policy{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def _worker_overflow(rank, init_path, root):
    """One rank overflows; both ranks must skip and keep the same fp16 scale."""

    torch.set_num_threads(1)
    _init(init_path, rank)
    try:
        model, network = _distributed_model()
        runtime = _build_runtime(
            rank,
            model,
            network,
            conditioning=CONDITIONING_LOCAL,
            microbatches=2,
            batch=BATCH,
            scaler_factory=_local_scaler,
        )
        latents, timesteps, context, padding_mask, target = _tiny_batch()
        before = {n: p.detach().clone() for n, p in network.named_parameters()}
        runtime.zero_grad()
        runtime.step(
            latents,
            timesteps=timesteps,
            crossattn_emb=context,
            target=target,
            padding_mask=padding_mask,
            losses=[],
        )
        if rank == 0:
            first = next(
                p for p in network.parameters() if p.grad is not None
            )
            first.grad.fill_(float("inf"))
        outcome = runtime.optimizer_step()
        after = {n: p.detach().clone() for n, p in network.named_parameters()}
        moved = [n for n in before if not torch.equal(before[n], after[n])]
        # The skipped step must leave the pipeline in lockstep: a second, clean
        # step has to run at the backed-off scale and reproduce the single-process
        # gradients exactly (the weights never moved).
        runtime.zero_grad()
        runtime.step(
            latents,
            timesteps=timesteps,
            crossattn_emb=context,
            target=target,
            padding_mask=padding_mask,
            losses=[],
        )
        recovery = runtime.optimizer_step()
        payload = {
            "skipped": outcome.skipped,
            "local_overflow": outcome.local_overflow,
            "agreed_overflow": outcome.agreed_overflow,
            "scale": outcome.scale,
            "next_scale": outcome.next_scale,
            "runtime_scale": runtime.scale,
            "moved": moved,
            "recovery_skipped": recovery.skipped,
            "recovery_scale": recovery.scale,
            "grads": {
                name: parameter.grad.detach().clone()
                for name, parameter in network.named_parameters()
                if parameter.grad is not None
            },
        }
        torch.save(payload, Path(root) / f"overflow{rank}.pt")
        (Path(root) / f"overflow{rank}.ok").touch()
    except Exception:
        (Path(root) / f"overflow{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def _worker_asymmetric(rank, init_path, root):
    """A one-sided frozen-slot declaration must be refused at build time."""

    torch.set_num_threads(1)
    _init(init_path, rank)
    try:
        model, _network = _distributed_model()
        layout, stage_module = build_anima_stage_module(
            model, stage_index=rank, stages=2, conditioning=CONDITIONING_WIRE
        )
        inputs, outputs = boundary_metas(
            model,
            layout,
            latent_shape=(BATCH, 4, 1, 4, 4),
            microbatches=2,
            policy=PpComputePolicy.from_dtype(torch.float32),
            conditioning=CONDITIONING_WIRE,
            timesteps_shape=(BATCH,),
            context_shape=(BATCH, 3, CHANNELS),
        )
        stage = build_pipeline_stage(
            stage_module,
            layout,
            device=torch.device("cpu"),
            input_args=inputs,
            output_args=outputs,
            conditioning=CONDITIONING_WIRE,
        )
        if rank == 1:
            # Deliberately drop the paired send-side suppression.
            stage._frozen_input_indices = ()
        try:
            assert_frozen_boundary_agreement(stage, group=dist.group.WORLD)
        except PipelineParallelError as exc:
            assert exc.code == "pp_frozen_boundary_mismatch", exc
            (Path(root) / f"asym{rank}.ok").touch()
        else:
            (Path(root) / f"asym{rank}.missed").touch()
    except Exception:
        (Path(root) / f"asym{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def _run_pair(tmp_path, worker, *args):
    ctx = get_context("spawn")
    init_path = str(tmp_path / "init")
    processes = [
        ctx.Process(target=worker, args=(rank, init_path, *args, str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=90)
        errors = "\n".join(path.read_text() for path in tmp_path.glob("*.error"))
        assert all(process.exitcode == 0 for process in processes), errors
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.mark.integration
@pytest.mark.probe
@pytest.mark.parametrize(
    ("conditioning", "microbatches", "batch"),
    # Every microbatch must hold the same batch size (the PP wire buffers are
    # fixed-shape), so the batch is a multiple of the microbatch count.
    [(CONDITIONING_LOCAL, 2, BATCH), (CONDITIONING_LOCAL, 3, 6), (CONDITIONING_WIRE, 2, BATCH)],
)
def test_two_rank_pipeline_matches_single_process(
    tmp_path, conditioning, microbatches, batch
):
    _run_pair(tmp_path, _worker_pipeline, conditioning, microbatches, batch)
    payloads = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in range(2)]
    assert len(list(tmp_path.glob("*.ok"))) == 2

    latents, timesteps, context, padding_mask, target = _tiny_batch(batch)
    expected_prediction, expected_grads, expected_loss = _baseline_grads(
        latents, timesteps, context, padding_mask, target
    )

    last = payloads[1]
    assert last["prediction"] is not None
    assert payloads[0]["prediction"] is None
    torch.testing.assert_close(last["prediction"], expected_prediction, rtol=1e-5, atol=1e-6)
    # ``losses`` holds one *fp16-scaled, 1/n-divided* loss per microbatch, so the
    # whole-batch loss is sum(losses) / scale.
    assert len(last["losses"]) == microbatches
    unscaled = sum(last["losses"]) / last["scale"]
    assert unscaled == pytest.approx(expected_loss, rel=1e-5)

    seen = set()
    for payload in payloads:
        for name, grad in payload["grads"].items():
            assert name not in seen
            seen.add(name)
            torch.testing.assert_close(grad, expected_grads[name], rtol=2e-4, atol=2e-7)
    assert seen == set(expected_grads)

    # The fp16 scale is one pipeline-wide value, identical on both stages.
    assert payloads[0]["scale"] == payloads[1]["scale"] == 8.0
    for payload in payloads:
        assert payload["skipped"] is False

    if conditioning == CONDITIONING_WIRE:
        assert payloads[0]["frozen_output_indices"] == list(WIRE_FROZEN_INDICES)
        assert payloads[1]["frozen_input_indices"] == list(WIRE_FROZEN_INDICES)
        # The send half drops the frozen slots ...
        assert payloads[1]["grad_send_info"] == [0, None, None, None]
        # ... and the receive half allocates one gradient buffer, not four.
        assert payloads[0]["n_grad_recv"] == 1
    else:
        assert payloads[0]["frozen_output_indices"] == []
        assert payloads[1]["frozen_input_indices"] == []
        assert payloads[0]["n_grad_recv"] == 1


@pytest.mark.integration
@pytest.mark.probe
def test_overflow_skips_both_stages_symmetrically(tmp_path):
    _run_pair(tmp_path, _worker_overflow)
    payloads = [
        torch.load(tmp_path / f"overflow{rank}.pt", weights_only=False) for rank in range(2)
    ]
    assert len(list(tmp_path.glob("overflow*.ok"))) == 2
    for payload in payloads:
        assert payload["skipped"] is True
        assert payload["agreed_overflow"] is True
        assert payload["moved"] == []
    assert payloads[0]["local_overflow"] is True
    assert payloads[1]["local_overflow"] is False
    # Same scale before and after on both stages: the backoff is derived from the
    # agreed decision, not from each rank's local verdict.
    assert payloads[0]["scale"] == payloads[1]["scale"] == 8.0
    assert payloads[0]["next_scale"] == payloads[1]["next_scale"] == 4.0
    # The recovery step ran at the backed-off scale on both stages and stepped.
    for payload in payloads:
        assert payload["recovery_skipped"] is False
        assert payload["recovery_scale"] == 4.0
        assert payload["runtime_scale"] == 4.0
    # ... and its gradients are the single-process gradients again: the skipped
    # step left no residue behind.
    latents, timesteps, context, padding_mask, target = _tiny_batch()
    _prediction, expected_grads, _loss = _baseline_grads(
        latents, timesteps, context, padding_mask, target
    )
    seen = set()
    for payload in payloads:
        for name, grad in payload["grads"].items():
            assert name not in seen
            seen.add(name)
            torch.testing.assert_close(grad, expected_grads[name], rtol=2e-4, atol=2e-7)
    assert seen == set(expected_grads)


@pytest.mark.integration
@pytest.mark.probe
def test_fp16_pipeline_shares_one_use_fp32_and_splits_boundary_dtypes(tmp_path):
    """D1 + D2 on a real two-rank fp16 step.

    Rank 0 feeds an fp16 pre-block, the block stack promotes the residual stream,
    so the wire carries fp32 while the prediction comes back fp16.  Both stages
    must therefore resolve ``use_fp32`` from the fp16 pre-block dtype, and the
    boundary metas must express the two different dtypes.
    """

    _run_pair(tmp_path, _worker_fp16_policy)
    payloads = [
        torch.load(tmp_path / f"fp16_{rank}.pt", weights_only=False) for rank in range(2)
    ]
    assert len(list(tmp_path.glob("fp16*.ok"))) == 2

    for payload in payloads:
        assert payload["runtime_use_fp32"] is True, payload
        assert payload["stage_use_fp32"] is True, payload
        assert payload["policy"]["use_fp32"] is True, payload
        assert payload["policy"]["activation_dtype"] == "torch.float16", payload
        assert payload["policy"]["boundary_dtype"] == "torch.float32", payload
        assert payload["policy"]["prediction_dtype"] == "torch.float16", payload

    assert payloads[0]["policy"] == payloads[1]["policy"], (
        "the stages resolved different compute policies: "
        f"{payloads[0]['policy']} vs {payloads[1]['policy']}"
    )

    # D2: one dtype could not describe both ends of the last stage.
    assert payloads[0]["inputs_meta"] == ["torch.float16"], payloads[0]
    assert payloads[0]["outputs_meta"] == ["torch.float32"], payloads[0]
    assert payloads[1]["inputs_meta"] == ["torch.float32"], payloads[1]
    assert payloads[1]["outputs_meta"] == ["torch.float16"], payloads[1]

    assert payloads[1]["prediction_dtype"] == "torch.float16"
    assert payloads[1]["prediction_finite"] is True, payloads[1]


@pytest.mark.integration
@pytest.mark.probe
def test_scale_policy_is_broadcast_so_both_stages_stay_in_lockstep(tmp_path):
    """A non-default growth interval must not split the pipeline-wide scale.

    Before the fix the non-last stage used the torch defaults (2.0 / 0.5 / 2000)
    instead of the last stage's scaler policy, so with ``growth_interval=10`` it
    stayed at 8.0 while the last stage grew to 16.0 and unscaled the boundary
    gradients by twice the factor the loss was scaled with -- silently.
    """

    _run_pair(tmp_path, _worker_scale_policy)
    payloads = [
        torch.load(tmp_path / f"policy{rank}.pt", weights_only=False) for rank in range(2)
    ]
    assert len(list(tmp_path.glob("policy*.ok"))) == 2

    for payload in payloads:
        assert payload["growth_interval"] == 10, payload
        assert payload["growth_factor"] == 2.0
        assert payload["backoff_factor"] == 0.5
    assert payloads[0]["final_scale"] == payloads[1]["final_scale"], (
        "the fp16 scale split between the stages: "
        f"stage0={payloads[0]['final_scale']} stage1={payloads[1]['final_scale']}"
    )
    # init_scale=8.0 with growth_interval=10 over 12 clean steps: exactly one
    # growth, on both stages, and no skip anywhere.
    assert payloads[0]["final_scale"] == 16.0, payloads[0]
    for index, (first, second) in enumerate(
        zip(payloads[0]["trace"], payloads[1]["trace"], strict=True)
    ):
        assert first["skipped"] is False, (index, first)
        assert first["scale"] == second["scale"], (index, first, second)
        assert first["next_scale"] == second["next_scale"], (index, first, second)
    assert [step["scale"] for step in payloads[0]["trace"]][:10] == [8.0] * 10
    assert [step["scale"] for step in payloads[0]["trace"]][10:] == [16.0, 16.0]


@pytest.mark.integration
@pytest.mark.probe
def test_one_sided_frozen_slots_are_refused_not_deadlocked(tmp_path):
    _run_pair(tmp_path, _worker_asymmetric)
    assert len(list(tmp_path.glob("asym*.ok"))) == 2
    assert not list(tmp_path.glob("asym*.missed"))


# ---------------------------------------------------------------------------
# API surface guards
# ---------------------------------------------------------------------------


def test_runtime_never_wraps_the_model_in_ddp():
    """No module in the package may reach for a data-parallel wrapper."""

    root = Path("library/training/pp")
    offenders = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for marker in (
            "DistributedDataParallel",
            "torch.nn.parallel",
            "accelerator.backward",
            "DDP(",
        ):
            if marker in text:
                offenders.append(f"{path.name}:{marker}")
    assert offenders == []


def test_public_runtime_types_are_exported():
    assert issubclass(PipelineParallelError, RuntimeError)
    assert AnimaPipelineRuntime is not None
    assert FrozenBoundaryPipelineStage is not None
    assert build_anima_stage_layout(stage_index=0, stages=2, num_blocks=BLOCKS).block_range == (0, 2)
