"""MOSS-Image PP v0 runtime acceptance (task t2).

Layers proven here (frozen entrypoint, see docs/experimental/
moss_image_pipeline_parallel_v0.md):

* **L0** -- shape/wire-capacity contracts and named rejections (registry-backed
  plan, tail batch, odd-but-legal packed edges, non-binary masks).
* **L1 (CPU/Gloo)** -- a real two-rank 1F1B run over the production tiny MOSS
  model with a nonzero applied plain LoRA: the pipeline loss and every adapter
  gradient match the non-PP production-forward reference (magnitudes and
  directions, thresholds recorded in :mod:`tests.moss_pp_test_support` and
  never derived from a candidate's error); stage-local optimizers only update
  their own stage; the ``1/n_microbatches`` factor is applied exactly once; a
  single-side overflow skips the step on both ranks.
* **L2 (marked hardware, not run in this task)** --
  ``test_two_rank_cuda_fp16_runtime_and_resume`` is the frozen dual-VCC tiny
  node (fp16 forward/backward, one-sided overflow, checkpoint restart/replay,
  per-rank peak allocated/reserved).  It is only executed by the verification
  task on the authorized GPUs; ``-m 'not hardware'`` excludes it everywhere
  else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from library.models.moss_image.pipeline_parallel import (
    build_moss_stage_module,
    resolve_wire_capacity,
    split_moss_batch,
    validate_moss_latent_shape,
)
from library.training.pp import (
    MossPipelineRuntime,
    PipelineParallelError,
    PipelineRuntimeConfig,
    build_moss_pipeline_runtime,
    checkpoint_format_for_family,
)
from library.training.pp.contract import PpComputePolicy
from library.training.pp.errors import (
    CODE_SHAPE_CONTRACT,
    CODE_STAGE_CONTRACT,
    CODE_TOKEN_BUCKET_MISMATCH,
)
from library.training.pp.moss_runtime import resolve_moss_wire_spec

from tests.moss_pp_test_support import (
    BATCH,
    GRAD_ATOL,
    GRAD_RTOL,
    HEIGHT,
    MAX_TEXT_TOKENS,
    MODEL_SEED,
    PRED_ATOL,
    PRED_RTOL,
    WIDTH,
    reference_forward_and_grads,
    reference_noise_baseline,
    run_pair,
    run_pair_expect_failure,
    tiny_batch,
    tiny_lora,
    tiny_model,
)


PP_CONFIG = dict(
    model_family="moss_image",
    pipeline_parallel=True,
    pipeline_parallel_stages=2,
    pipeline_parallel_microbatches=2,
    pipeline_parallel_schedule="1f1b",
    pipeline_parallel_split="balanced",
    mixed_precision="fp16",
)


def _runtime_config() -> PipelineRuntimeConfig:
    return PipelineRuntimeConfig.from_config(PP_CONFIG)


# ---------------------------------------------------------------------------
# L0: shape and wire-capacity contracts
# ---------------------------------------------------------------------------


def test_latent_shape_contract():
    assert validate_moss_latent_shape((2, 128, 2, 3)) == (2, 128, 2, 3)
    for shape in ((2, 128, 3), (2, 64, 2, 3), (0, 128, 2, 3), (2, 128, 0, 3), (2, 128, 2)):
        with pytest.raises(PipelineParallelError) as excinfo:
            validate_moss_latent_shape(shape)
        assert excinfo.value.code == CODE_SHAPE_CONTRACT


def test_wire_capacity_and_split_contract():
    assert resolve_wire_capacity(HEIGHT, WIDTH, MAX_TEXT_TOKENS) == HEIGHT * WIDTH + MAX_TEXT_TOKENS
    assert split_moss_batch(4, 2) == 2
    with pytest.raises(PipelineParallelError) as excinfo:
        split_moss_batch(3, 2)  # tail batch
    assert excinfo.value.code == CODE_TOKEN_BUCKET_MISMATCH
    with pytest.raises(PipelineParallelError) as excinfo:
        split_moss_batch(0, 2)
    assert excinfo.value.code == CODE_TOKEN_BUCKET_MISMATCH


def test_wire_spec_rejects_non_divisible_batch():
    model = tiny_model()
    with pytest.raises(PipelineParallelError) as excinfo:
        resolve_moss_wire_spec(
            model, latent_shape=(3, 128, HEIGHT, WIDTH),
            max_text_tokens=MAX_TEXT_TOKENS, microbatches=2,
        )
    assert excinfo.value.code == CODE_TOKEN_BUCKET_MISMATCH
    spec = resolve_moss_wire_spec(
        model, latent_shape=(2, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, microbatches=2,
    )
    assert spec.token_cap == HEIGHT * WIDTH + MAX_TEXT_TOKENS
    assert spec.latent_shape == (2, 128, HEIGHT, WIDTH)
    assert spec.microbatch_latent_shape == (1, 128, HEIGHT, WIDTH)


def test_moss_family_plan_is_balanced_over_42_layers():
    from library.models.pipeline_parallel import make_pipeline_plan

    plan = make_pipeline_plan(family="moss_image", stages=2, num_blocks=42)
    assert plan.ranges == ((0, 21), (21, 42))
    assert plan.block_container == "layers"


def test_adapter_ownership_covers_every_adapter_exactly_once():
    model = tiny_model()
    network = tiny_lora(model)
    names = [lora.lora_name for lora in network.unet_loras]
    assert len(names) == 168
    assert all(name.startswith("lora_unet_layers_") for name in names)
    policy = PpComputePolicy.from_dtype(torch.float32)
    _, first = build_moss_stage_module(
        model, stage_index=0, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, network=network, compute_policy=policy,
    )
    _, last = build_moss_stage_module(
        model, stage_index=1, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, network=network, compute_policy=policy,
    )
    first_names = {m.lora_name for m in first.adapter_modules}
    last_names = {m.lora_name for m in last.adapter_modules}
    assert len(first_names) == 84 and len(last_names) == 84
    assert not (first_names & last_names)
    assert first_names | last_names == set(names)
    # The ownership cross-check ran against the real applied network (the
    # monkey-patched Linear targets) inside build_moss_stage_module; a name
    # whose target is missing from the model would have raised above.
    assert len({m.lora_name for m in first.adapter_modules}) == 84


def test_stage_contract_rejects_block_swap():
    model = tiny_model()
    model.blocks_to_swap = 2
    with pytest.raises(PipelineParallelError) as excinfo:
        build_moss_stage_module(
            model, stage_index=0, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
            max_text_tokens=MAX_TEXT_TOKENS,
        )
    assert excinfo.value.code == CODE_STAGE_CONTRACT


# ---------------------------------------------------------------------------
# L1 (single process): split forward equals the production forward
# ---------------------------------------------------------------------------


def test_staged_forward_matches_production_bitwise():
    latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
    model = tiny_model()
    network = tiny_lora(model)
    policy = PpComputePolicy.from_dtype(torch.float32)
    _, first = build_moss_stage_module(
        model, stage_index=0, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, network=network, compute_policy=policy,
    )
    _, last = build_moss_stage_module(
        model, stage_index=1, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, network=network, compute_policy=policy,
    )
    network.zero_grad(set_to_none=True)
    reference_prediction, reference_grads, reference_loss = (
        reference_forward_and_grads(latents, text, mask, times, target)
    )
    staged = first(latents, timesteps=times, text_hiddens=text, text_mask=mask)
    hidden, lengths, mod = staged
    assert hidden.shape == (BATCH, HEIGHT * WIDTH + MAX_TEXT_TOKENS, 64)
    assert lengths.tolist() == [4, 2]
    assert hidden.requires_grad is True
    prediction = last(hidden, lengths, mod, timesteps=times)
    torch.testing.assert_close(
        prediction, reference_prediction, rtol=PRED_RTOL, atol=PRED_ATOL
    )
    loss = F.mse_loss(prediction.float(), target.float())
    assert loss.item() == reference_loss
    loss.backward()
    worst_name, worst_value = max(
        (
            (name, (network.get_parameter(name).grad - grad).abs().max().item())
            for name, grad in reference_grads.items()
        ),
        key=lambda item: item[1],
    )
    assert worst_value <= GRAD_ATOL + GRAD_RTOL * reference_grads[worst_name].abs().max().item(), (
        f"staged gradient diverged on {worst_name}: {worst_value}"
    )
    # The boundary hidden is differentiable back into the text path: the
    # stage-0-owned adapters must receive gradient although the loss lives on
    # stage 1 (this one-process path backwardss through the real boundary).
    stage0_grad_sum = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in network.named_parameters()
        if parameter.grad is not None and "layers_0_proj" in name
    )
    assert stage0_grad_sum > 0.0


def test_transport_padding_never_enters_attention():
    latents, text, mask, _sigmas, times, _target = tiny_batch(BATCH)
    model = tiny_model()
    network = tiny_lora(model)
    policy = PpComputePolicy.from_dtype(torch.float32)
    _, first = build_moss_stage_module(
        model, stage_index=0, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, network=network, compute_policy=policy,
    )
    _, last = build_moss_stage_module(
        model, stage_index=1, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, network=network, compute_policy=policy,
    )
    with torch.no_grad():
        hidden, lengths, mod = first(
            latents, timesteps=times, text_hiddens=text, text_mask=mask
        )
        baseline = last(hidden, lengths, mod, timesteps=times)
        # Poison the *text* transport padding: stage 0 slices by the mask, so
        # the boundary hidden (and therefore stage 1) is unchanged.
        corrupted = text.clone()
        corrupted[~mask] = float("inf")
        hidden2, lengths2, mod2 = first(
            latents, timesteps=times, text_hiddens=corrupted, text_mask=mask
        )
        torch.testing.assert_close(hidden2, hidden, rtol=0, atol=0)
        poisoned = last(hidden2, lengths2, mod2, timesteps=times)
        torch.testing.assert_close(poisoned, baseline, rtol=0, atol=0)
        assert lengths2.tolist() == lengths.tolist()
        # Poison the *boundary* padding: stage 1 slices to the true lengths,
        # so padded positions cannot influence q/k/v or the RMSNorm statistics.
        padded = hidden.clone()
        for index in range(BATCH):
            valid = HEIGHT * WIDTH + int(lengths[index])
            padded[index, valid:] = float("nan")
        result = last(padded, lengths, mod, timesteps=times)
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)


def test_non_binary_or_empty_mask_is_refused():
    latents, text, _mask, _sigmas, times, _target = tiny_batch(BATCH)
    model = tiny_model()
    policy = PpComputePolicy.from_dtype(torch.float32)
    _, first = build_moss_stage_module(
        model, stage_index=0, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, compute_policy=policy,
    )
    sneaky = torch.zeros(BATCH, MAX_TEXT_TOKENS)
    sneaky[0, :4] = 1.0
    sneaky[1, 1] = 0.5  # not binary: the true text length becomes ambiguous
    sneaky[1, 2] = 1.0
    with pytest.raises(PipelineParallelError) as excinfo:
        first(latents, timesteps=times, text_hiddens=text, text_mask=sneaky)
    assert excinfo.value.code == CODE_SHAPE_CONTRACT
    empty = torch.zeros(BATCH, MAX_TEXT_TOKENS, dtype=torch.bool)
    empty[0, 0] = True  # sample 1 has no valid token
    with pytest.raises(PipelineParallelError) as excinfo:
        first(latents, timesteps=times, text_hiddens=text, text_mask=empty)
    assert excinfo.value.code == CODE_STAGE_CONTRACT


def test_latent_geometry_is_frozen_per_runtime():
    latents, text, mask, _sigmas, times, _target = tiny_batch(BATCH)
    model = tiny_model()
    policy = PpComputePolicy.from_dtype(torch.float32)
    _, first = build_moss_stage_module(
        model, stage_index=0, stages=2, latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS, compute_policy=policy,
    )
    with pytest.raises(PipelineParallelError) as excinfo:
        first(
            torch.randn(BATCH, 128, 2, 4),
            timesteps=times, text_hiddens=torch.randn(BATCH, MAX_TEXT_TOKENS, 32),
            text_mask=mask,
        )
    assert excinfo.value.code == CODE_TOKEN_BUCKET_MISMATCH
    with pytest.raises(PipelineParallelError) as excinfo:
        first(
            latents.to(torch.float64),
            timesteps=times, text_hiddens=text, text_mask=mask,
        )
    assert excinfo.value.code == "pp_compute_policy_mismatch"


def test_moss_checkpoint_format_mapping():
    assert checkpoint_format_for_family("moss_image") == "moss-image-pp-stage-checkpoint"
    assert checkpoint_format_for_family("anima") == "anima-pp-stage-checkpoint"
    assert checkpoint_format_for_family(None) == "anima-pp-stage-checkpoint"


# ---------------------------------------------------------------------------
# L1 (two-rank Gloo): schedule, transaction, stage-local optimizer
# ---------------------------------------------------------------------------


def _worker_build_failure(rank, init_path, tmp_path):
    """Rank 1's optimizer factory raises locally; both ranks must refuse."""
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        override = (
            (lambda _stage: (_ for _ in ()).throw(ValueError("boom: broken optimizer factory")))
            if rank == 1
            else None
        )
        try:
            _build_runtime(
                rank, model, network, microbatches=2,
                optimizer_factory_override=override,
            )
            raise AssertionError("the broken factory must refuse the build")
        except Exception as exc:  # noqa: BLE001 - negotiated below
            (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
            (Path(tmp_path) / f"rank{rank}.code").write_text(
                getattr(exc, "code", "no_pp_code")
            )
            raise
        finally:
            dist.destroy_process_group()
    except (ValueError, AssertionError, RuntimeError):
        raise  # the expected, negotiated refusal
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


@pytest.mark.integration
def test_build_failure_negotiates_before_any_collective(tmp_path):
    """C2: a rank-local construction failure is a symmetric, fast refusal.

    Rank 1's optimizer factory raises; both ranks must exit with the same
    named code well before any process-group timeout could fire (the refusal
    happens at the first negotiation barrier, not inside a collective).
    """
    import time

    started = time.monotonic()
    run_pair_expect_failure(tmp_path, _worker_build_failure)
    elapsed = time.monotonic() - started
    codes = [
        (Path(tmp_path) / f"rank{rank}.code").read_text()
        for rank in range(2)
    ]
    assert set(codes) == {"pp_stage_contract_violation"}, codes
    for rank in range(2):
        text = (Path(tmp_path) / f"rank{rank}.error").read_text()
        assert "boom: broken optimizer factory" in text, text
    assert elapsed < 60, f"refusal took {elapsed:.1f}s; the ranks deadlocked"


def _worker_empty_sample(rank, init_path, tmp_path):
    """Step precheck: one sample with zero valid tokens must abort both ranks."""
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network, microbatches=2)
        latents, text, _mask, _sigmas, times, target = tiny_batch(BATCH)
        # Total valid count (4) >= batch, so a batch-sum check alone would
        # pass; sample 1 has no valid token at all.
        sneaky_mask = torch.zeros(BATCH, MAX_TEXT_TOKENS, dtype=torch.bool)
        sneaky_mask[0, :4] = True
        try:
            runtime.step(
                latents,
                timesteps=times,
                text_hiddens=text,
                text_mask=sneaky_mask,
                target=target if runtime.is_last else None,
            )
            raise AssertionError("the empty sample must refuse the step")
        except PipelineParallelError as exc:
            (Path(tmp_path) / f"rank{rank}.code").write_text(exc.code)
            (Path(tmp_path) / f"rank{rank}.error").write_text(f"negotiated: {exc}")
            raise
        finally:
            dist.destroy_process_group()
    except PipelineParallelError:
        raise
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


@pytest.mark.integration
def test_empty_sample_refused_symmetrically_before_schedule(tmp_path):
    """C1: per-sample true-length precheck aborts both ranks identically."""
    run_pair_expect_failure(tmp_path, _worker_empty_sample)
    codes = [
        (Path(tmp_path) / f"rank{rank}.code").read_text()
        for rank in range(2)
    ]
    assert set(codes) == {CODE_STAGE_CONTRACT}, codes
    for rank in range(2):
        text = (Path(tmp_path) / f"rank{rank}.error").read_text()
        assert "sample 1" in text and "no valid text token" in text


def _worker_missing_target(rank, init_path, tmp_path):
    """Last stage without target must abort both ranks before the schedule."""
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network, microbatches=2)
        latents, text, mask, _sigmas, times, _target = tiny_batch(BATCH)
        try:
            runtime.step(
                latents,
                timesteps=times,
                text_hiddens=text,
                text_mask=mask,
                target=None if runtime.is_last else None,
                losses=[] if runtime.is_last else None,
            )
            raise AssertionError("the missing target must refuse the step")
        except PipelineParallelError as exc:
            (Path(tmp_path) / f"rank{rank}.code").write_text(exc.code)
            (Path(tmp_path) / f"rank{rank}.error").write_text(f"negotiated: {exc}")
            raise
        finally:
            dist.destroy_process_group()
    except PipelineParallelError:
        raise
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


@pytest.mark.integration
def test_missing_target_refused_symmetrically_before_schedule(tmp_path):
    """F1: a last stage without target aborts both ranks identically."""
    run_pair_expect_failure(tmp_path, _worker_missing_target)
    codes = [
        (Path(tmp_path) / f"rank{rank}.code").read_text()
        for rank in range(2)
    ]
    assert set(codes) == {CODE_STAGE_CONTRACT}, codes
    for rank in range(2):
        text = (Path(tmp_path) / f"rank{rank}.error").read_text()
        assert "needs the batch target" in text


def _worker_bad_timesteps(rank, init_path, tmp_path):
    """Wrong timestep count must abort both ranks (rank 0 included)."""
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network, microbatches=2)
        latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
        bad_times = torch.zeros(BATCH - 1)
        try:
            runtime.step(
                latents,
                timesteps=bad_times,
                text_hiddens=text,
                text_mask=mask,
                target=target if runtime.is_last else None,
                losses=[] if runtime.is_last else None,
            )
            raise AssertionError("the short timesteps must refuse the step")
        except PipelineParallelError as exc:
            (Path(tmp_path) / f"rank{rank}.code").write_text(exc.code)
            (Path(tmp_path) / f"rank{rank}.error").write_text(f"negotiated: {exc}")
            raise
        finally:
            dist.destroy_process_group()
    except PipelineParallelError:
        raise
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


@pytest.mark.integration
def test_bad_timesteps_refused_symmetrically_on_both_ranks(tmp_path):
    """F1: timestep count mismatch aborts both ranks (rank 0 validated too)."""
    run_pair_expect_failure(tmp_path, _worker_bad_timesteps)
    codes = [
        (Path(tmp_path) / f"rank{rank}.code").read_text()
        for rank in range(2)
    ]
    assert set(codes) == {CODE_SHAPE_CONTRACT}, codes
    for rank in range(2):
        text = (Path(tmp_path) / f"rank{rank}.error").read_text()
        assert "timesteps must carry one value per sample" in text


def _build_runtime(rank, model, network, *, microbatches=2, optimizer_factory_override=None):
    from library.training.pp.moss_runtime import build_moss_pipeline_runtime

    device = torch.device("cpu")
    config = PipelineRuntimeConfig(
        stages=2, microbatches=microbatches, schedule="1f1b",
        conditioning="local", mixed_precision="fp16",
    )
    scaler = torch.amp.GradScaler("cpu") if rank == 1 else None
    optimizer_factory = optimizer_factory_override or (
        lambda stage: torch.optim.AdamW(list(stage.trainable_parameters()), lr=1e-3)
    )
    return build_moss_pipeline_runtime(
        model,
        config=config,
        rank=rank,
        world_size=2,
        device=device,
        latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS,
        dtype=torch.float32,
        network=network,
        scaler=scaler,
        optimizer_factory=optimizer_factory,
        group=None,
    )


def _worker_equivalence(rank, init_path, microbatches, tmp_path):
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo, write_error

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network, microbatches=microbatches)
        latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
        owned_names = {m.lora_name for m in runtime.stage_module.adapter_modules}
        runtime.zero_grad()
        losses: list = []
        runtime.step(
            latents,
            timesteps=times,
            text_hiddens=text,
            text_mask=mask,
            target=target if runtime.is_last else None,
            losses=losses,
        )
        outcome = runtime.optimizer_step()
        assert outcome.skipped is False
        assert outcome.local_overflow is False
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in network.named_parameters()
            if parameter.grad is not None
        }
        # Stage-local ownership: this rank's gradient set must be a subset of
        # the adapters its layers own (keyed by the real lora_name).
        assert all(
            name.split(".lora_")[0] in owned_names for name in grads
        ), sorted(grads)[:4]
        scale = float(runtime.transaction.scale)
        payload = {
            "losses": [float(value) for value in losses],
            "grads": grads,
            "scale": scale,
            "owned": sorted(owned_names),
            "rank": rank,
            "topology": runtime.topology(),
        }
        torch.save(payload, f"{tmp_path}/rank{rank}.pt")
        dist.all_reduce(torch.tensor([1.0]))
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001 - recorded for the parent assert
        write_error(tmp_path, rank, exc)
        raise


@pytest.mark.integration
def test_two_rank_gloo_matches_reference_and_stage_local_optimizer(tmp_path):
    microbatches = 2
    run_pair(tmp_path, _worker_equivalence, microbatches)
    payloads = [
        torch.load(f"{tmp_path}/rank{rank}.pt", weights_only=False) for rank in range(2)
    ]
    latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
    prediction, reference_grads, reference_loss = reference_forward_and_grads(
        latents, text, mask, times, target
    )
    baseline = reference_noise_baseline(latents, text, mask, times, target)
    print(
        "recorded noise baseline:",
        {key: float(value) for key, value in baseline.items()},
    )
    last = payloads[1]
    unscaled = sum(last["losses"]) / last["scale"]
    # The 1/n_microbatches factor is applied exactly once: sum(losses)/scale
    # equals the whole-batch mean loss.
    assert unscaled == pytest.approx(reference_loss, rel=1e-5)
    assert payloads[0]["scale"] == payloads[1]["scale"]
    # Every adapter gradient appears on exactly one rank and matches the
    # non-PP reference in magnitude and direction.
    seen: dict[str, torch.Tensor] = {}
    for payload in payloads:
        for name, grad in payload["grads"].items():
            assert name not in seen, name
            seen[name] = grad
    assert set(seen) == set(reference_grads)
    for name, grad in seen.items():
        ref = reference_grads[name]
        torch.testing.assert_close(grad, ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)
        cosine = F.cosine_similarity(
            grad.flatten().float(), ref.flatten().float(), dim=0
        )
        assert cosine.item() >= 1.0 - 1e-5, name
    # The wire/topology is the MOSS contract, family-tagged for checkpoints.
    for payload in payloads:
        topology = payload["topology"]
        assert topology["family"] == "moss_image"
        assert topology["token_cap"] == HEIGHT * WIDTH + MAX_TEXT_TOKENS
        assert topology["frozen_output_indices" if payload["rank"] == 0
                         else "frozen_input_indices"] == [1, 2]


def _worker_overflow(rank, init_path, tmp_path):
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo, write_error

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network, microbatches=2)
        latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
        runtime.zero_grad()
        losses: list = []
        runtime.step(
            latents,
            timesteps=times,
            text_hiddens=text,
            text_mask=mask,
            target=target if runtime.is_last else None,
            losses=losses,
        )
        before = float(runtime.transaction.scale)
        if rank == 0:
            # One-sided overflow: inject a non-finite gradient on this rank only.
            parameter = next(runtime.stage_module.trainable_parameters())
            parameter.grad = torch.full_like(parameter, float("inf"))
        outcome = runtime.optimizer_step()
        payload = {
            "skipped": outcome.skipped,
            "agreed_overflow": outcome.agreed_overflow,
            "scale": float(runtime.transaction.scale),
            "before_scale": before,
            "last_skip": runtime.transaction.last_skip,
        }
        torch.save(payload, f"{tmp_path}/rank{rank}.pt")
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        write_error(tmp_path, rank, exc)
        raise


@pytest.mark.integration
def test_single_side_overflow_skips_on_both_ranks(tmp_path):
    run_pair(tmp_path, _worker_overflow)
    payloads = [
        torch.load(f"{tmp_path}/rank{rank}.pt", weights_only=False) for rank in range(2)
    ]
    assert all(payload["skipped"] is True for payload in payloads)
    assert all(payload["agreed_overflow"] is True for payload in payloads)
    assert payloads[0]["last_skip"] is True and payloads[1]["last_skip"] is True
    # Both ranks back off the shared scale by the same policy factor.
    assert payloads[0]["scale"] == payloads[1]["scale"]
    assert payloads[0]["scale"] < payloads[0]["before_scale"]


# ---------------------------------------------------------------------------
# L2 (frozen hardware node; not executed by this task)
# ---------------------------------------------------------------------------


MOSS_CUDA_REPLAY_RTOL = 1e-2  # pre-registered fp16 prior; never candidate-fitted


def _cuda_runtime(rank, device, model, network, *, growth_interval):
    # LoRA lives outside the borrowed base model; place it before the factory
    # captures parameter references. A device-only move preserves fp32 masters.
    network.to(device=device)
    config = PipelineRuntimeConfig(
        stages=2, microbatches=2, schedule="1f1b",
        conditioning="local", mixed_precision="fp16",
    )
    scaler = torch.amp.GradScaler("cuda") if rank == 1 else None
    if scaler is not None:
        scaler.set_growth_interval(growth_interval)
    return build_moss_pipeline_runtime(
        model,
        config=config,
        rank=rank,
        world_size=2,
        device=device,
        latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS,
        dtype=torch.float16,
        network=network,
        scaler=scaler,
        optimizer_factory=lambda stage: torch.optim.AdamW(
            list(stage.trainable_parameters()), lr=1e-3
        ),
    )


def _cuda_batch(rank_device):
    latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
    return (
        latents.to(rank_device).to(torch.float16),
        text.to(rank_device),
        mask.to(rank_device),
        times.to(rank_device),
        target.to(rank_device).to(torch.float16),
    )


def _cuda_run_steps(runtime, latents, text, mask, times, target, count):
    """Run ``count`` clean autocast fp16 steps, recording per-step evidence.

    Each record keeps the scale the step's loss was multiplied with, so the
    comparison restores unscaled losses per step (C5) instead of dividing a
    later scale through earlier losses.
    """
    records: list = []
    for _ in range(count):
        runtime.zero_grad()
        step_losses: list = []
        with torch.autocast("cuda", dtype=torch.float16):
            runtime.step(
                latents,
                timesteps=times,
                text_hiddens=text,
                text_mask=mask,
                target=target if runtime.is_last else None,
                losses=step_losses,
            )
        assert len(step_losses) == (2 if runtime.is_last else 0)
        assert all(torch.isfinite(loss).all() for loss in step_losses)
        outcome = runtime.optimizer_step()
        assert outcome.skipped is False
        records.append({
            "scale_used": float(outcome.scale),
            "losses": [float(value) for value in step_losses],
        })
    return records


def _worker_cuda_save(rank, init_path, root, tmp_path):
    import datetime

    import torch.distributed as dist
    from library.training.pp import (
        capture_stage_snapshot,
        save_distributed_checkpoint,
    )
    from tests.moss_pp_test_support import write_error

    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{init_path}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=120),
        )
        model = tiny_model().to(device=device, dtype=torch.float16)
        network = tiny_lora(model)  # fp32 master weights
        runtime = _cuda_runtime(rank, device, model, network, growth_interval=4)
        latents, text, mask, times, target = _cuda_batch(device)
        # Steps 1-2 (clean, checkpoint window).
        save_steps = _cuda_run_steps(runtime, latents, text, mask, times, target, 2)
        before_overflow = capture_stage_snapshot(runtime, step=2).digest()
        # One-side overflow: the whole pipeline must skip this step.
        runtime.zero_grad()
        overflow_losses: list = []
        with torch.autocast("cuda", dtype=torch.float16):
            runtime.step(
                latents,
                timesteps=times,
                text_hiddens=text,
                text_mask=mask,
                target=target if runtime.is_last else None,
                losses=overflow_losses,
            )
        if rank == 0:
            parameter = next(runtime.stage_module.trainable_parameters())
            parameter.grad = torch.full_like(parameter, float("inf"))
        outcome = runtime.optimizer_step()
        assert outcome.skipped is True
        after_overflow = capture_stage_snapshot(runtime, step=3).digest()
        assert after_overflow["adapter_hash"] == before_overflow["adapter_hash"]
        assert after_overflow["optimizer_hash"] == before_overflow["optimizer_hash"]
        save_distributed_checkpoint(root, runtime, step=3)
        scale_at_save = float(runtime.transaction.scale)
        tracker_at_save = runtime.transaction.growth_tracker
        saved_digest = capture_stage_snapshot(runtime, step=3).digest()
        # C5: the uninterrupted continuation (steps 4-5) recorded AFTER the
        # save, on the same trajectory the fresh resume must reproduce.
        continuation_steps = _cuda_run_steps(
            runtime, latents, text, mask, times, target, 2
        )
        after_digest = capture_stage_snapshot(runtime, step=5).digest()
        peak = torch.cuda.max_memory_allocated() / (1 << 20)
        reserved = torch.cuda.max_memory_reserved() / (1 << 20)
        torch.save(
            {
                "save_steps": save_steps,
                "continuation_steps": continuation_steps,
                "saved_digest": saved_digest,
                "after_continuation_digest": after_digest,
                "scale_at_save": scale_at_save,
                "tracker_at_save": tracker_at_save,
                "scale_after_continuation": float(runtime.transaction.scale),
                "tracker_after_continuation": runtime.transaction.growth_tracker,
                "growth_interval": runtime.transaction.growth_interval,
                "skipped": True,
                "peak_allocated_mib": peak,
                "peak_reserved_mib": reserved,
            },
            f"{tmp_path}/cuda_save_rank{rank}.pt",
        )
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        write_error(tmp_path, rank, exc)
        raise


def _worker_cuda_restore(rank, init_path, root, tmp_path):
    """Phase 2: a *fresh* process pair restores and replays the clean steps.

    The constructor interval (777) differs from the checkpointed one (4), so
    the growth policy must come from the checkpoint, not this process.
    """
    import datetime

    import torch.distributed as dist
    from library.training.pp import (
        capture_stage_snapshot,
        load_distributed_checkpoint,
    )
    from tests.moss_pp_test_support import write_error

    try:
        assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{init_path}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=120),
        )
        model = tiny_model().to(device=device, dtype=torch.float16)
        network = tiny_lora(model)
        runtime = _cuda_runtime(rank, device, model, network, growth_interval=777)
        state = load_distributed_checkpoint(root, runtime)
        assert state.step == 3
        restored_digest = capture_stage_snapshot(runtime, step=3).digest()
        latents, text, mask, times, target = _cuda_batch(device)
        # Same window as the save-side continuation: the two steps after the
        # resume point, replayed on the fresh pair (steps 4-5).
        replay_steps = _cuda_run_steps(runtime, latents, text, mask, times, target, 2)
        after_digest = capture_stage_snapshot(runtime, step=5).digest()
        peak = torch.cuda.max_memory_allocated() / (1 << 20)
        reserved = torch.cuda.max_memory_reserved() / (1 << 20)
        torch.save(
            {
                "replay_steps": replay_steps,
                "restored_digest": restored_digest,
                "after_replay_digest": after_digest,
                "scale_after_replay": float(runtime.transaction.scale),
                "tracker_after_replay": runtime.transaction.growth_tracker,
                "growth_interval": runtime.transaction.growth_interval,
                "dtype": str(
                    next(p.dtype for p in network.parameters() if p.requires_grad)
                ),
                "peak_allocated_mib": peak,
                "peak_reserved_mib": reserved,
            },
            f"{tmp_path}/cuda_restore_rank{rank}.pt",
        )
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        write_error(tmp_path, rank, exc)
        raise


@pytest.mark.hardware
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    reason="requires the two authorized CUDA GPUs (verification task only)",
)
def test_two_rank_cuda_fp16_runtime_and_resume(tmp_path):
    """Frozen hardware node: dual tiny fp16 runtime + fresh-process resume.

    Phase 1 runs two clean autocast fp16 steps (steps 1-2, losses recorded),
    one one-side-overflow skip, the checkpoint save and an *uninterrupted
    continuation* (steps 4-5, per-step records); phase 2 is a *fresh pair of
    processes* that restores with a different constructor growth interval and
    replays the same post-resume window (steps 4-5).  Comparisons are
    same-window with per-step recorded scales (C5).  Each GPU must stay
    <= 2 GiB (asserted from the recorded peaks).  ``MOSS_CUDA_REPLAY_RTOL``
    is pre-registered from the fp16 accumulation prior; evidence records the
    observation and must not be used to adjust that threshold.
    """

    import json
    from time import monotonic

    root = tmp_path / "cuda_root"
    deadline = monotonic() + 45  # shared phases + cleanup/report before outer 60s
    run_pair(tmp_path, _worker_cuda_save, root, deadline=deadline)
    run_pair(tmp_path, _worker_cuda_restore, root, deadline=deadline)
    saved = [
        torch.load(f"{tmp_path}/cuda_save_rank{rank}.pt", weights_only=False)
        for rank in range(2)
    ]
    restored = [
        torch.load(f"{tmp_path}/cuda_restore_rank{rank}.pt", weights_only=False)
        for rank in range(2)
    ]
    for rank in range(2):
        assert restored[rank]["dtype"] == "torch.float32", (
            "fp16 runs require fp32 LoRA master weights"
        )
        # The policy came from the checkpoint (save=4), not the constructor
        # (restore=777); scale and tracker resume where the save stopped.
        assert restored[rank]["growth_interval"] == saved[rank]["growth_interval"] == 4
        assert restored[rank]["scale_after_replay"] == saved[rank]["scale_after_continuation"]
        assert restored[rank]["tracker_after_replay"] == saved[rank]["tracker_after_continuation"]
        assert restored[rank]["restored_digest"] == saved[rank]["saved_digest"]
        assert restored[rank]["after_replay_digest"] == saved[rank]["after_continuation_digest"]
        assert all(payload["peak_allocated_mib"] <= 2048 for payload in (saved[rank], restored[rank]))
        assert all(payload["peak_reserved_mib"] <= 2048 for payload in (saved[rank], restored[rank]))
        # Same-window per-step comparison (steps 4-5 on both trajectories):
        # the scale each step used must match, and unscaled losses are
        # restored per step with that step's own scale.
        for step_index in range(2):
            cont = saved[rank]["continuation_steps"][step_index]
            rep = restored[rank]["replay_steps"][step_index]
            assert rep["scale_used"] == cont["scale_used"]
            assert len(rep["losses"]) == len(cont["losses"]) == (2 if rank == 1 else 0)
            for loss_cont, loss_rep in zip(cont["losses"], rep["losses"]):
                unscaled_cont = loss_cont / cont["scale_used"]
                unscaled_rep = loss_rep / rep["scale_used"]
                assert unscaled_rep == pytest.approx(
                    unscaled_cont, rel=MOSS_CUDA_REPLAY_RTOL
                )
    # Cross-rank skip symmetry evidence stays in the recorded payloads
    # (skipped=True, per-rank scale trajectories in cuda_node_evidence.json).
    assert all(payload["skipped"] for payload in saved)
    (tmp_path / "cuda_node_evidence.json").write_text(
        json.dumps({"save": saved, "restore": restored}, indent=2)
    )
