"""MOSS-Image PP v0 checkpoint acceptance (task t2).

Frozen entrypoint.  What this file proves:

1. **A distinct, family-tagged checkpoint format.**  MOSS saves
   ``moss-image-pp-stage-checkpoint``; the Anima loader refuses it and the MOSS
   loader refuses ``anima-pp-stage-checkpoint`` -- both *before* any payload
   field is read (named ``pp_checkpoint_incomplete``).
2. **Save -> fresh-process restart -> load is exact** and a resumed next step
   replays: the restored adapter/optimizer/fp16 state is compared per tensor
   against the state the first pair recorded, and the post-resume step loss
   matches the uninterrupted run within the recorded relative tolerance.
3. **Cross-rank negotiation.**  A corrupted artifact or a stage whose adapter
   ownership does not fit aborts *both* ranks before any rank applies state.
4. **fp16 policy recovery**: scale, growth tracker, last skip and the broadcast
   growth/backoff/interval policy are restored exactly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from library.training.pp import (
    MOSS_IMAGE_PP_CHECKPOINT_FORMAT,
    PipelineParallelError,
    PipelineRuntimeConfig,
    capture_stage_snapshot,
    checkpoint_format_for_family,
    content_hash_file,
    load_distributed_checkpoint,
    read_manifest,
    save_distributed_checkpoint,
)
from library.training.pp.errors import (
    CODE_CHECKPOINT_IDENTITY_MISMATCH,
    CODE_CHECKPOINT_INCOMPLETE,
    CODE_CHECKPOINT_STATE_MISMATCH,
    CODE_STAGE_CONTRACT,
)

from tests.moss_pp_test_support import (
    BATCH,
    HEIGHT,
    MAX_TEXT_TOKENS,
    MODEL_SEED,
    WIDTH,
    reference_forward_and_grads,
    run_pair,
    run_pair_expect_failure,
    tiny_batch,
    tiny_lora,
    tiny_model,
)

MOSS_CONFIG = dict(
    model_family="moss_image",
    pipeline_parallel=True,
    pipeline_parallel_stages=2,
    pipeline_parallel_microbatches=2,
    pipeline_parallel_schedule="1f1b",
    pipeline_parallel_split="balanced",
    mixed_precision="fp16",
)


def _build_runtime(rank, model, network, *, growth_interval=2000):
    from library.training.pp import build_moss_pipeline_runtime

    scaler = torch.amp.GradScaler("cpu") if rank == 1 else None
    if scaler is not None:
        scaler.set_growth_interval(growth_interval)
    return build_moss_pipeline_runtime(
        model,
        config=PipelineRuntimeConfig(
            stages=2, microbatches=2, schedule="1f1b",
            conditioning="local", mixed_precision="fp16",
        ),
        rank=rank,
        world_size=2,
        device=torch.device("cpu"),
        latent_shape=(BATCH, 128, HEIGHT, WIDTH),
        max_text_tokens=MAX_TEXT_TOKENS,
        dtype=torch.float32,
        network=network,
        scaler=scaler,
        optimizer_factory=lambda stage: torch.optim.AdamW(
            list(stage.trainable_parameters()), lr=1e-3
        ),
    )


def _run_one_step(runtime, latents, text, mask, times, target):
    """One step; returns (outcome, per-step record {scale used, losses})."""
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
    return outcome, {
        # outcome.scale is the scale THIS step's loss was multiplied with
        # (C4: unscale per step with its own scale, never with a later one).
        "scale_used": float(outcome.scale),
        "losses": [float(value) for value in losses],
    }


def _worker_save(rank, init_path, root, growth_interval, tmp_path):
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo, write_error

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network, growth_interval=growth_interval)
        latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
        steps: list = []
        outcome, record = _run_one_step(runtime, latents, text, mask, times, target)
        assert outcome.skipped is False
        steps.append(record)
        outcome, record = _run_one_step(runtime, latents, text, mask, times, target)
        assert outcome.skipped is False
        steps.append(record)
        result = save_distributed_checkpoint(root, runtime, step=2)
        manifest = read_manifest(
            root, format_name=MOSS_IMAGE_PP_CHECKPOINT_FORMAT
        )
        saved_state = capture_stage_snapshot(runtime, step=2)
        scale_at_save = float(runtime.transaction.scale)
        tracker_at_save = runtime.transaction.growth_tracker
        # Continue the uninterrupted run two more steps (steps 3 and 4): the
        # fresh-process restore must replay exactly these steps -- and both
        # independent trajectories must cross the scale-growth boundary
        # (tracker reaches 4 -> scale doubles at the 4th step).
        continuation: list = []
        outcome, record = _run_one_step(runtime, latents, text, mask, times, target)
        assert outcome.skipped is False
        continuation.append(record)
        outcome, record = _run_one_step(runtime, latents, text, mask, times, target)
        assert outcome.skipped is False
        continuation.append(record)
        after_growth = capture_stage_snapshot(runtime, step=4)
        torch.save(
            {
                "saved_digest": saved_state.digest(),
                "scale_at_save": scale_at_save,
                "tracker_at_save": tracker_at_save,
                "save_steps": steps,
                "scale_after_continuation": float(runtime.transaction.scale),
                "tracker_after_continuation": runtime.transaction.growth_tracker,
                "after_growth_digest": after_growth.digest(),
                "continuation_steps": continuation,
                "manifest": manifest,
            },
            f"{tmp_path}/rank{rank}.pt",
        )
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


def _worker_restore(rank, init_path, root, restore_interval, tmp_path):
    """A *fresh* process pair rebuilds the fixture and restores + replays.

    ``restore_interval`` is deliberately **different** from the save-side
    interval: the growth policy must come back from the checkpoint (re-derived
    from the restored GradScaler), not from this process's constructor.
    """
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo, write_error

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network, growth_interval=restore_interval)
        state = load_distributed_checkpoint(root, runtime)
        assert state.step == 2
        restored_digest = capture_stage_snapshot(runtime, step=2).digest()
        latents, text, mask, _sigmas, times, target = tiny_batch(BATCH)
        replay: list = []
        _outcome, record = _run_one_step(runtime, latents, text, mask, times, target)
        replay.append(record)
        _outcome, record = _run_one_step(runtime, latents, text, mask, times, target)
        replay.append(record)
        after_replay = capture_stage_snapshot(runtime, step=4)
        torch.save(
            {
                "restored_digest": restored_digest,
                "replay_steps": replay,
                "scale_after_replay": float(runtime.transaction.scale),
                "tracker_after_replay": runtime.transaction.growth_tracker,
                "after_replay_digest": after_replay.digest(),
                "manifest_format": state.manifest["format"],
                "last_skip": runtime.transaction.last_skip,
            },
            f"{tmp_path}/restored_rank{rank}.pt",
        )
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


@pytest.mark.integration
def test_moss_checkpoint_restart_and_replay(tmp_path):
    root = tmp_path / "run"
    growth_interval = 4  # tracker reaches 4 inside the 4 total steps, so the
    # scale-growth boundary is crossed on both the save-continuation and the
    # replay paths
    run_pair(tmp_path, _worker_save, root, growth_interval)
    saved = [
        torch.load(f"{tmp_path}/rank{rank}.pt", weights_only=False) for rank in range(2)
    ]
    manifest = saved[0]["manifest"]
    # The manifest is family-tagged and carries the MOSS topology per rank.
    assert manifest["format"] == MOSS_IMAGE_PP_CHECKPOINT_FORMAT
    assert manifest["plan"]["family"] == "moss_image"
    assert manifest["plan"]["ranges"] == [[0, 21], [21, 42]]
    for entry in manifest["ranks"]:
        topology = entry["topology"]
        assert topology["family"] == "moss_image"
        assert topology["token_cap"] == HEIGHT * WIDTH + MAX_TEXT_TOKENS
        assert topology["adapter_names"], "adapter ownership must be recorded"
    # Fresh processes restore and replay; their constructor interval differs
    # from the save-side one, so the policy must come from the checkpoint.
    run_pair(tmp_path, _worker_restore, root, 777)
    restored = [
        torch.load(f"{tmp_path}/restored_rank{rank}.pt", weights_only=False)
        for rank in range(2)
    ]
    for rank in range(2):
        assert restored[rank]["manifest_format"] == MOSS_IMAGE_PP_CHECKPOINT_FORMAT
        # C4: the scale-growth boundary is actually crossed on *both*
        # independent trajectories (interval 4, tracker reaches 4 -> doubles).
        assert saved[rank]["scale_after_continuation"] == saved[rank]["scale_at_save"] * 2
        assert restored[rank]["scale_after_replay"] == saved[rank]["scale_after_continuation"]
        assert restored[rank]["tracker_after_replay"] == saved[rank]["tracker_after_continuation"]
        # Restore exactness: the pre-replay digest equals the saved state.
        before = saved[rank]["saved_digest"]
        after = restored[rank]["restored_digest"]
        assert before == after, (
            f"rank {rank} state diverged after restart: "
            + "; ".join(
                f"{key}: {before.get(key)!r} != {after.get(key)!r}"
                for key in sorted(set(before) | set(after))
                if before.get(key) != after.get(key)
            )
        )
        # C5: same window (steps 3-4 on both trajectories), per-step unscaled
        # losses restored with the scale each step actually used; the
        # post-update state of the independent trajectories (fresh resume vs
        # uninterrupted continuation) is compared as a full snapshot digest.
        for step_index in range(2):
            cont = saved[rank]["continuation_steps"][step_index]
            rep = restored[rank]["replay_steps"][step_index]
            # The scale each step used must match between the trajectories.
            assert rep["scale_used"] == cont["scale_used"]
            assert rep["losses"] == cont["losses"], (
                f"rank {rank} step {step_index} replay losses diverged"
            )
            unscaled_rep = [loss / rep["scale_used"] for loss in rep["losses"]]
            unscaled_cont = [loss / cont["scale_used"] for loss in cont["losses"]]
            assert unscaled_rep == pytest.approx(unscaled_cont, rel=1e-6)
        assert restored[rank]["after_replay_digest"] == saved[rank]["after_growth_digest"], (
            f"rank {rank} post-update state diverged between the fresh resume and "
            "the uninterrupted continuation"
        )
    # Replay exactness: per-step losses are bitwise equal to the uninterrupted
    # continuation (asserted per rank above); nothing else is claimed about
    # post-training losses here (the fixture weights have moved by then).


def _worker_corrupt_then_load(rank, init_path, root, tmp_path):
    import torch.distributed as dist
    from tests.moss_pp_test_support import init_gloo, write_error

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        runtime = _build_runtime(rank, model, network)
        if rank == 1:
            # Corrupt this rank's artifact bytes after the manifest was written.
            artifact = root / "pp_stage-rank1.pt"
            payload = torch.load(artifact, weights_only=True)
            payload["step"] = 999
            torch.save(payload, artifact)
        try:
            load_distributed_checkpoint(root, runtime)
            raise AssertionError("the corrupted artifact must abort the load")
        except PipelineParallelError as exc:
            (Path(tmp_path) / f"rank{rank}.code").write_text(exc.code)
            (Path(tmp_path) / f"rank{rank}.error").write_text(
                f"negotiated refusal: {exc}"
            )
            raise
        finally:
            dist.destroy_process_group()
    except PipelineParallelError:
        raise  # the expected, negotiated refusal
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


@pytest.mark.integration
def test_corrupted_artifact_aborts_both_ranks(tmp_path):
    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    errors = run_pair_expect_failure(tmp_path, _worker_corrupt_then_load, root)
    # Both ranks recorded the identity mismatch: neither rank applied state
    # while the other refused.
    codes = [
        (tmp_path / f"rank{rank}.code").read_text()
        for rank in range(2)
        if (tmp_path / f"rank{rank}.code").is_file()
    ]
    assert len(errors) == 2
    assert set(codes) == {CODE_CHECKPOINT_IDENTITY_MISMATCH}, codes


# ---------------------------------------------------------------------------
# format refusal, both directions, before any payload field is read
# ---------------------------------------------------------------------------


def _write_anima_format_root(root: Path) -> None:
    """A minimal *Anima-format* run root (format check fires before fields)."""

    root.mkdir()
    artifact = {
        "format": "anima-pp-stage-checkpoint",
        "version": 1,
        "step": 1,
        "topology": {"family": "anima"},
        "identity": {},
        "adapters": {},
        "optimizer": {},
        "scaler": None,
        "scale": None,
        "growth_tracker": 0,
        "last_skip": None,
    }
    torch.save(artifact, root / "pp_stage-rank0.pt")
    (root / "pp_checkpoint.json").write_text(
        json.dumps(
            {
                "format": "anima-pp-stage-checkpoint",
                "version": 1,
                "world_size": 2,
                "stages": 2,
                "step": 1,
                "identity": {},
                "ranks": [
                    {
                        "rank": 0,
                        "file": "pp_stage-rank0.pt",
                        "sha256": "0" * 64,
                        "topology": {"family": "anima"},
                    }
                ],
            }
        )
    )


def _moss_probe_runtime(tmp_path):
    """A full MOSS runtime without a distributed peer (world_size=1 gloo).

    ``torch.distributed.pipelining.PipelineStage`` requires an initialized
    default group even in single-process use, so the probe initializes a
    one-rank Gloo group; the checkpoint format checks only read topology.
    """

    from types import SimpleNamespace

    import torch.distributed as dist

    model = tiny_model()
    network = tiny_lora(model)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path}/pg",
        rank=0,
        world_size=1,
        timeout=__import__("datetime").timedelta(seconds=60),
    )
    try:
        runtime = _build_runtime(0, model, network)
        return SimpleNamespace(
            topology=runtime.topology, stage_module=runtime.stage_module
        )
    finally:
        dist.destroy_process_group()


def test_moss_loader_refuses_anima_format_fixture(tmp_path):
    from library.training.pp import read_stage_checkpoint

    root = tmp_path / "anima-format"
    _write_anima_format_root(root)
    runtime = _moss_probe_runtime(tmp_path)
    with pytest.raises(PipelineParallelError) as excinfo:
        read_stage_checkpoint(root, runtime)
    assert excinfo.value.code == CODE_CHECKPOINT_INCOMPLETE
    assert "moss-image-pp-stage-checkpoint" in str(excinfo.value)


def test_anima_loader_refuses_moss_format(tmp_path):
    """The real MOSS run root read with the Anima schema is refused by name."""

    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    from types import SimpleNamespace

    from library.training.pp import read_stage_checkpoint

    anima_probe = SimpleNamespace(
        topology=lambda: {"family": "anima", "stages": 2, "stage_index": 0},
        stage_module=None,
    )
    with pytest.raises(PipelineParallelError) as excinfo:
        read_stage_checkpoint(root, anima_probe)
    assert excinfo.value.code == CODE_CHECKPOINT_INCOMPLETE
    assert "anima-pp-stage-checkpoint" in str(excinfo.value)


def test_artifact_payload_carries_the_moss_format(tmp_path):
    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    payload = torch.load(root / "pp_stage-rank0.pt", weights_only=True)
    assert payload["format"] == MOSS_IMAGE_PP_CHECKPOINT_FORMAT
    assert content_hash_file(root / "pp_stage-rank0.pt") == (
        read_manifest(root, format_name=MOSS_IMAGE_PP_CHECKPOINT_FORMAT)
        ["ranks"][0]["sha256"]
    )


# ---------------------------------------------------------------------------
# ownership must fit before anything is applied
# ---------------------------------------------------------------------------


def _worker_half_network(rank, init_path, root, tmp_path):
    """Rank 1 rebuilds with a network that misses its own layers 21..41."""
    import torch.distributed as dist
    from library.training.pp import build_moss_pipeline_runtime
    from networks.lora_anima import create_network
    from library.models.moss_image.lora_targets import moss_image_target_kwargs
    from tests.moss_pp_test_support import init_gloo, write_error

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        if rank == 1:
            kwargs = moss_image_target_kwargs()
            # Keep the default exclude (everything but main projections) and
            # additionally exclude layers 21..30 so the rebuilt rank 1 still
            # owns adapters (the runtime can build) but misses the adapters
            # the checkpoint recorded for those layers.
            kwargs["exclude_patterns"] = [
                r"(?!layers\.\d+\.proj\.[qkvo]$).*",
                r"layers\.(?:2[1-9]|30)\.proj\.[qkvo]$",
            ]
            network = create_network(
                1.0, 4, 4.0, None, [], model,
                model_family="moss_image", **kwargs,
            )
            network.apply_to([], model, False, True)
        else:
            network = tiny_lora(model)
        runtime = build_moss_pipeline_runtime(
            model,
            config=PipelineRuntimeConfig(
                stages=2, microbatches=2, schedule="1f1b",
                conditioning="local", mixed_precision="fp16",
            ),
            rank=rank,
            world_size=2,
            device=torch.device("cpu"),
            latent_shape=(BATCH, 128, HEIGHT, WIDTH),
            max_text_tokens=MAX_TEXT_TOKENS,
            dtype=torch.float32,
            network=network,
            scaler=torch.amp.GradScaler("cpu") if rank == 1 else None,
            optimizer_factory=lambda stage: torch.optim.AdamW(
                list(stage.trainable_parameters()), lr=1e-3
            ),
        )
        before = {
            module.lora_name: {
                key: value.detach().clone()
                for key, value in module.state_dict().items()
            }
            for module in runtime.stage_module.adapter_modules
        }
        try:
            load_distributed_checkpoint(root, runtime)
            raise AssertionError("ownership mismatch must refuse the load")
        except PipelineParallelError as exc:
            after = {
                module.lora_name: {
                    key: value.detach().clone()
                    for key, value in module.state_dict().items()
                }
                for module in runtime.stage_module.adapter_modules
            }
            for name, state in before.items():
                for key, tensor in state.items():
                    assert torch.equal(tensor, after[name][key]), (
                        f"rank {rank}: {name}.{key} changed although the load "
                        "was refused; state must not be applied"
                    )
            (Path(tmp_path) / f"rank{rank}.code").write_text(exc.code)
            (Path(tmp_path) / f"rank{rank}.error").write_text(
                f"negotiated refusal: {exc}"
            )
            raise
        finally:
            dist.destroy_process_group()
    except PipelineParallelError:
        raise
    except Exception as exc:  # noqa: BLE001
        (Path(tmp_path) / f"rank{rank}.error").write_text(repr(exc))
        raise


@pytest.mark.integration
def test_adapter_ownership_mismatch_refuses_before_apply(tmp_path):
    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    run_pair_expect_failure(tmp_path, _worker_half_network, root)
    codes = [
        (tmp_path / f"rank{rank}.code").read_text()
        for rank in range(2)
        if (tmp_path / f"rank{rank}.code").is_file()
    ]
    assert set(codes) == {"pp_checkpoint_topology_mismatch"}, codes
    # The refusal text names the adapter ownership difference explicitly and
    # fires before any state is applied (the negotiated payload carries it).
    error_text = Path(tmp_path, "rank0.error").read_text()
    assert "adapter_names" in error_text


def _worker_rank1_optimizer_mismatch(rank, init_path, root, tmp_path):
    """Rank 1 rebuilds with two AdamW param groups; apply must abort both."""
    import torch.distributed as dist
    from library.training.pp import build_moss_pipeline_runtime
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        if rank == 1:
            optimizer_factory = lambda stage: torch.optim.AdamW(
                [
                    {"params": [next(stage.trainable_parameters())]},
                    {"params": list(stage.trainable_parameters())[1:]},
                ],
                lr=1e-3,
            )
        else:
            optimizer_factory = lambda stage: torch.optim.AdamW(
                list(stage.trainable_parameters()), lr=1e-3
            )
        runtime = build_moss_pipeline_runtime(
            model,
            config=PipelineRuntimeConfig(
                stages=2, microbatches=2, schedule="1f1b",
                conditioning="local", mixed_precision="fp16",
            ),
            rank=rank,
            world_size=2,
            device=torch.device("cpu"),
            latent_shape=(BATCH, 128, HEIGHT, WIDTH),
            max_text_tokens=MAX_TEXT_TOKENS,
            dtype=torch.float32,
            network=network,
            scaler=torch.amp.GradScaler("cpu") if rank == 1 else None,
            optimizer_factory=optimizer_factory,
        )
        before = {
            module.lora_name: {
                key: value.detach().clone()
                for key, value in module.state_dict().items()
            }
            for module in runtime.stage_module.adapter_modules
        }
        try:
            load_distributed_checkpoint(root, runtime)
            raise AssertionError("optimizer mismatch must refuse the apply")
        except PipelineParallelError as exc:
            after = {
                module.lora_name: {
                    key: value.detach().clone()
                    for key, value in module.state_dict().items()
                }
                for module in runtime.stage_module.adapter_modules
            }
            for name, state in before.items():
                for key, tensor in state.items():
                    assert torch.equal(tensor, after[name][key]), (
                        f"rank {rank}: {name}.{key} changed although the apply "
                        "was refused; state must not be mutated"
                    )
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
def test_single_side_optimizer_mismatch_refuses_symmetrically(tmp_path):
    """C3: a one-rank optimizer shape mismatch aborts both ranks, unmutated."""
    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    run_pair_expect_failure(tmp_path, _worker_rank1_optimizer_mismatch, root)
    codes = [
        (Path(tmp_path) / f"rank{rank}.code").read_text()
        for rank in range(2)
        if (Path(tmp_path) / f"rank{rank}.code").is_file()
    ]
    assert set(codes) == {CODE_CHECKPOINT_STATE_MISMATCH}, codes


def _worker_rank1_no_transaction(rank, init_path, root, tmp_path):
    """Rank 1 builds without optimizer/transaction; the build must abort both.

    The fp16 transaction opens with broadcasts, so a half-transactional
    pipeline cannot even be constructed: the builder's capability negotiation
    refuses both ranks before any collective a lone rank could enter.
    """
    import torch.distributed as dist
    from library.training.pp import build_moss_pipeline_runtime
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        try:
            runtime = build_moss_pipeline_runtime(
                model,
                config=PipelineRuntimeConfig(
                    stages=2, microbatches=2, schedule="1f1b",
                    conditioning="local", mixed_precision="fp16",
                ),
                rank=rank,
                world_size=2,
                device=torch.device("cpu"),
                latent_shape=(BATCH, 128, HEIGHT, WIDTH),
                max_text_tokens=MAX_TEXT_TOKENS,
                dtype=torch.float32,
                network=network,
                scaler=torch.amp.GradScaler("cpu") if rank == 1 else None,
                optimizer=None,
                optimizer_factory=None if rank == 1 else (
                    lambda stage: torch.optim.AdamW(
                        list(stage.trainable_parameters()), lr=1e-3
                    )
                ),
            )
            raise AssertionError("the half-transactional build must refuse")
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
def test_single_side_missing_transaction_refuses_symmetrically(tmp_path):
    """C3/C2: a stage without its optimizer transaction aborts both ranks."""
    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    run_pair_expect_failure(tmp_path, _worker_rank1_no_transaction, root)
    codes = [
        (Path(tmp_path) / f"rank{rank}.code").read_text()
        for rank in range(2)
        if (Path(tmp_path) / f"rank{rank}.code").is_file()
    ]
    assert set(codes) == {CODE_STAGE_CONTRACT}, codes


def _worker_rank1_scaler_state_stripped(rank, init_path, root, tmp_path):
    """Rank 1's checkpoint artifact loses its scaler state; both refuse.

    The loss is negotiated in the validate phase (before any mutation), so
    both ranks keep their adapter state bitwise unchanged.
    """
    import torch.distributed as dist
    from library.training.pp import build_moss_pipeline_runtime
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        if rank == 1:
            # Strip the scaler payload from this rank's artifact AND keep the
            # manifest content-hash consistent, so the load itself succeeds and
            # only the validate phase can catch the missing scaler state.
            import hashlib

            import json

            artifact = root / "pp_stage-rank1.pt"
            payload = torch.load(artifact, weights_only=True)
            payload.pop("scaler", None)
            torch.save(payload, artifact)
            manifest_file = root / "pp_checkpoint.json"
            manifest = json.loads(manifest_file.read_text())
            for entry in manifest["ranks"]:
                if entry["rank"] == 1:
                    entry["sha256"] = hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest()
            manifest_file.write_text(json.dumps(manifest, indent=2))
        runtime = build_moss_pipeline_runtime(
            model,
            config=PipelineRuntimeConfig(
                stages=2, microbatches=2, schedule="1f1b",
                conditioning="local", mixed_precision="fp16",
            ),
            rank=rank,
            world_size=2,
            device=torch.device("cpu"),
            latent_shape=(BATCH, 128, HEIGHT, WIDTH),
            max_text_tokens=MAX_TEXT_TOKENS,
            dtype=torch.float32,
            network=network,
            scaler=torch.amp.GradScaler("cpu") if rank == 1 else None,
            optimizer_factory=lambda stage: torch.optim.AdamW(
                list(stage.trainable_parameters()), lr=1e-3
            ),
        )
        before = {
            module.lora_name: {
                key: value.detach().clone()
                for key, value in module.state_dict().items()
            }
            for module in runtime.stage_module.adapter_modules
        }
        try:
            load_distributed_checkpoint(root, runtime)
            raise AssertionError("the stripped scaler state must refuse the apply")
        except PipelineParallelError as exc:
            after = {
                module.lora_name: {
                    key: value.detach().clone()
                    for key, value in module.state_dict().items()
                }
                for module in runtime.stage_module.adapter_modules
            }
            for name, state in before.items():
                for key, tensor in state.items():
                    assert torch.equal(tensor, after[name][key]), (
                        f"rank {rank}: {name}.{key} changed although the apply "
                        "was refused; state must not be mutated"
                    )
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
def test_single_side_scaler_state_missing_refuses_symmetrically(tmp_path):
    """C3: a one-rank scaler/transaction payload mismatch aborts both ranks."""
    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    run_pair_expect_failure(tmp_path, _worker_rank1_scaler_state_stripped, root)
    codes = [
        (Path(tmp_path) / f"rank{rank}.code").read_text()
        for rank in range(2)
        if (Path(tmp_path) / f"rank{rank}.code").is_file()
    ]
    assert set(codes) == {CODE_CHECKPOINT_STATE_MISMATCH}, codes


def _mutate_payload(payload: dict, mutation: str) -> None:
    """Apply one F2 content mutation to a rank's checkpoint payload."""
    if mutation == "scale_nan":
        payload["scale"] = float("nan")
    elif mutation == "scale_negative":
        payload["scale"] = -65536.0
    elif mutation == "tracker_negative":
        payload["growth_tracker"] = -1
    elif mutation == "scaler_missing_key":
        payload["scaler"].pop("growth_interval")
    elif mutation == "scaler_bad_interval":
        payload["scaler"]["growth_interval"] = 0
    elif mutation == "scaler_missing_tracker":
        payload["scaler"].pop("_growth_tracker")
    elif mutation == "scaler_scale_mismatch":
        payload["scaler"]["scale"] = payload["scale"] / 2
    elif mutation == "optimizer_step_nan":
        optimizer = payload["optimizer"]
        first_key = optimizer["param_groups"][0]["params"][0]
        optimizer["state"][first_key]["step"] = torch.tensor(float("nan"))
    elif mutation in ("exp_avg_shape", "optimizer_id_moment_shape"):
        optimizer = payload["optimizer"]
        optimizer_state = optimizer["state"]
        first_key = optimizer["param_groups"][0]["params"][0]
        if mutation == "optimizer_id_moment_shape":
            new_id = max(p for g in optimizer["param_groups"] for p in g["params"]) + 91
            optimizer_state[new_id] = optimizer_state.pop(first_key)
            optimizer["param_groups"][0]["params"][0] = new_id
            first_key = new_id
        moment = optimizer_state[first_key]["exp_avg"]
        wrong_shape = tuple(moment.shape[:-1]) + (moment.shape[-1] + 1,)
        optimizer_state[first_key]["exp_avg"] = torch.zeros(wrong_shape)
    else:
        raise AssertionError(f"unknown mutation {mutation}")


def _worker_rank1_content_mutation(rank, init_path, root, mutation, tmp_path):
    """Rank 1's payload carries one illegal content mutation; both refuse.

    The manifest content hash is re-computed so the *load* succeeds and only
    the pure validate phase can catch the mutation -- before any rank mutates
    its adapter/optimizer/scaler state.
    """
    import hashlib

    import json

    import torch.distributed as dist
    from library.training.pp import build_moss_pipeline_runtime
    from tests.moss_pp_test_support import init_gloo

    try:
        init_gloo(init_path, rank)
        if rank == 1:
            artifact = root / "pp_stage-rank1.pt"
            payload = torch.load(artifact, weights_only=True)
            _mutate_payload(payload, mutation)
            torch.save(payload, artifact)
            manifest_file = root / "pp_checkpoint.json"
            manifest = json.loads(manifest_file.read_text())
            for entry in manifest["ranks"]:
                if entry["rank"] == 1:
                    entry["sha256"] = hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest()
            manifest_file.write_text(json.dumps(manifest, indent=2))
        model = tiny_model()
        network = tiny_lora(model)
        runtime = build_moss_pipeline_runtime(
            model,
            config=PipelineRuntimeConfig(
                stages=2, microbatches=2, schedule="1f1b",
                conditioning="local", mixed_precision="fp16",
            ),
            rank=rank,
            world_size=2,
            device=torch.device("cpu"),
            latent_shape=(BATCH, 128, HEIGHT, WIDTH),
            max_text_tokens=MAX_TEXT_TOKENS,
            dtype=torch.float32,
            network=network,
            scaler=torch.amp.GradScaler("cpu") if rank == 1 else None,
            optimizer_factory=lambda stage: torch.optim.AdamW(
                list(stage.trainable_parameters()), lr=1e-3
            ),
        )
        before = capture_stage_snapshot(runtime, step=0).digest()
        try:
            load_distributed_checkpoint(root, runtime)
            raise AssertionError(f"mutation {mutation} must refuse the load")
        except PipelineParallelError as exc:
            assert capture_stage_snapshot(runtime, step=0).digest() == before, (
                f"rank {rank}: adapter/optimizer/scaler state changed although "
                f"{mutation} was refused before apply"
            )
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


def test_content_mutation_refused_symmetrically_before_apply(tmp_path):
    """F2: illegal payload content aborts both ranks, state untouched.

    One checkpoint save, then one fresh restore pair per mutation -- the
    unique rendezvous paths (F3) keep consecutive pairs independent.
    """
    root = tmp_path / "run"
    run_pair(tmp_path, _worker_save, root, 2000)
    for mutation in (
        "scale_nan",
        "scale_negative",
        "tracker_negative",
        "scaler_missing_key",
        "scaler_bad_interval",
        "scaler_missing_tracker",
        "scaler_scale_mismatch",
        "exp_avg_shape",
        "optimizer_id_moment_shape",
        "optimizer_step_nan",
    ):
        case_tmp = tmp_path / mutation
        case_tmp.mkdir()
        # Every mutation starts from the same valid saved payload.  Otherwise a
        # missing key left by an earlier case can mask the next case's defect.
        case_root = shutil.copytree(root, case_tmp / "checkpoint")
        run_pair_expect_failure(
            case_tmp, _worker_rank1_content_mutation, case_root, mutation
        )
        codes = [
            (case_tmp / f"rank{rank}.code").read_text()
            for rank in range(2)
        ]
        assert codes == [CODE_CHECKPOINT_STATE_MISMATCH] * 2, (
            mutation,
            codes,
            [p.read_text() for p in case_tmp.glob("*.error")],
        )


def test_checkpoint_format_mapping():
    assert checkpoint_format_for_family("moss_image") == MOSS_IMAGE_PP_CHECKPOINT_FORMAT
    assert checkpoint_format_for_family("anima") == "anima-pp-stage-checkpoint"
