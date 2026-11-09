"""Two-rank Gloo fixture and workers for the PP checkpoint roundtrip (task t5).

The distributed half of ``tests/test_training_pp_checkpoint.py``.  It is a
separate module for the same reason ``tests/pp_equivalence_dist_test_support.py``
is: the fixture and the spawned workers are long, and the pytest file should stay
readable.

What the two phases prove
-------------------------
``worker_save`` runs in one pair of processes: it builds the runtime, takes two
real optimizer steps, saves the per-rank checkpoint into a **new** run root and
dumps this rank's full :class:`PpStageSnapshot` as the ground truth.

``worker_restore`` runs in a **second, freshly spawned** pair of processes: it
rebuilds the fixture from the same seeds, loads the checkpoint, and compares the
restored state against the ground truth with ``torch.equal`` per tensor and ``==``
per scalar.  It then replays the ``benchmark.verify_resume`` shape
(save -> step -> load -> step) and requires an adapter ``max_abs`` of exactly zero
and a loss delta of exactly zero.

The micro-model fixture mirrors ``tests/test_training_pp_runtime.py`` (same
``Anima`` sizes, same de-zeroed AdaLN modulation, same seeded LoRA); it lives here
rather than being imported from that test module so the two acceptance files stay
independently runnable.
"""

from __future__ import annotations

import json
import traceback
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from library.anima.models import Anima
from library.training.pp import (
    CONDITIONING_LOCAL,
    PipelineRuntimeConfig,
    PpStageSnapshot,
    build_anima_pipeline_runtime,
    build_anima_stage_module,
    capture_stage_snapshot,
    compare_snapshots,
    content_hash_tensors,
    load_distributed_checkpoint,
    save_distributed_checkpoint,
    stage_adapter_state,
)

BLOCKS = 4
CHANNELS = 24
BATCH = 4
MODEL_SEED = 123
MICROBATCHES = 2
#: Steps taken before the save, so the optimizer really has state to restore.
SAVE_AFTER_STEPS = 2
DIST_TIMEOUT = timedelta(seconds=90)
JOIN_TIMEOUT_SECONDS = 100


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


def tiny_model(seed: int = MODEL_SEED) -> Anima:
    """A 4-block Anima small enough to run the whole forward on CPU.

    The AdaLN output projections are de-zeroed because the stock model is
    *conditioning-dead*: ``Block`` zeroes ``adaln_up_self_attn`` /
    ``adaln_up_cross_attn`` / ``adaln_up_mlp`` (``models.py:1468``) and
    ``FinalLayer`` zeroes its last modulation projection (``models.py:1472-1474``).
    De-zeroing only ``adaln_modulation`` leaves the per-block timestep path dead,
    so a per-rank checkpoint round-trip would prove bit-exactness of weights whose
    conditioning contribution is identically zero.  Both name families are
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
    generator = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dim() < 2:
                continue
            if "adaln_modulation" not in name and "adaln_up_" not in name:
                continue
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model.train()


def tiny_lora(model, *, seed: int = MODEL_SEED + 1):
    from scripts.experiments.anima_parallel.common import create_mlp_lora

    network = create_mlp_lora(model, seed=seed, rank_dim=4, alpha=4.0)
    generator = torch.Generator().manual_seed(seed + 3)
    with torch.no_grad():
        for lora in network.unet_loras:
            lora.lora_up.weight.copy_(
                torch.randn(lora.lora_up.weight.shape, generator=generator) * 0.02
            )
    return network


def tiny_batch(count: int = BATCH):
    generator = torch.Generator().manual_seed(999)
    latents = torch.randn(count, 4, 1, 4, 4, generator=generator)
    timesteps = torch.rand(count, generator=generator)
    context = torch.randn(count, 3, CHANNELS, generator=generator)
    padding_mask = torch.zeros(count, 1, 4, 4)
    target = torch.randn(count, 4, 4, 4, generator=generator)
    return latents, timesteps, context, padding_mask, target


def build_distributed_model(seed: int = MODEL_SEED):
    model = tiny_model(seed)
    network = tiny_lora(model, seed=seed + 1)
    return model, network


def run_identity(model) -> dict[str, str]:
    """Content hash of the frozen base this checkpoint belongs to (no paths)."""

    return {"base": content_hash_tensors(dict(model.state_dict()))}


def stage_lora_parameters(network, block_range):
    from library.training.pp import adapter_block_index

    owned = set(range(block_range[0], block_range[1]))
    return [
        parameter
        for lora in network.unet_loras
        if adapter_block_index(lora.lora_name) in owned
        for parameter in lora.parameters()
    ]


def local_scaler(rank: int):
    # Only the last stage owns the loss, so only it owns a GradScaler.
    if rank != 1:
        return None
    return torch.amp.GradScaler("cpu", init_scale=8.0)


def build_runtime(rank: int, model, network, *, batch=BATCH, microbatches=MICROBATCHES):
    layout, _ = build_anima_stage_module(
        model,
        stage_index=rank,
        stages=2,
        conditioning=CONDITIONING_LOCAL,
        network=network,
    )
    parameters = stage_lora_parameters(network, layout.block_range)
    assert parameters, "each stage must own at least one adapter parameter"
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    return build_anima_pipeline_runtime(
        model,
        config=PipelineRuntimeConfig(
            stages=2,
            microbatches=microbatches,
            conditioning=CONDITIONING_LOCAL,
        ),
        world_size=2,
        rank=rank,
        device=torch.device("cpu"),
        latent_shape=(batch, 4, 1, 4, 4),
        dtype=torch.float32,
        timesteps_shape=(batch,),
        context_shape=(batch, 3, CHANNELS),
        scaler=local_scaler(rank),
        optimizer=optimizer,
        network=network,
        group=dist.group.WORLD,
    )


def one_step(runtime, batch) -> dict[str, Any]:
    """One full PP step plus its optimizer boundary; returns what to compare."""

    latents, timesteps, context, padding_mask, target = batch
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
    return {
        # Only the last stage computes a loss; earlier stages legitimately see none.
        "loss": float(sum(result.losses)),
        "scale": float(runtime.scale),
        "skipped": bool(outcome.skipped),
    }


# ---------------------------------------------------------------------------
# process plumbing
# ---------------------------------------------------------------------------


def init_group(init_path: str, rank: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=DIST_TIMEOUT,
    )


def run_pair(workdir: Path, worker, *args) -> None:
    """Spawn two Gloo ranks, join them and surface any rank traceback."""

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    init_path = str(workdir / "init")
    context = get_context("spawn")
    processes = [
        context.Process(target=worker, args=(rank, init_path, *args, str(workdir)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
        errors = "\n".join(path.read_text() for path in sorted(workdir.glob("*.error")))
        assert all(
            process.exitcode == 0 for process in processes
        ), f"PP checkpoint worker failed:\n{errors}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def run_pair_expect_failure(workdir: Path, worker, *args) -> str:
    """Spawn two Gloo ranks that must *both* fail, and return their tracebacks.

    The point is the *shape* of the failure: a refusal that one rank raises while
    the other waits in a collective shows up here as a join timeout instead of a
    clean non-zero exit on both ranks.
    """

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    init_path = str(workdir / "init")
    context = get_context("spawn")
    processes = [
        context.Process(target=worker, args=(rank, init_path, *args, str(workdir)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
        assert all(
            process.exitcode is not None for process in processes
        ), "a PP rank did not exit; the refusal deadlocked instead of failing"
        assert all(
            process.exitcode != 0 for process in processes
        ), "both ranks must refuse the save, not just one"
        return "\n".join(path.read_text() for path in sorted(workdir.glob("*.error")))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------


def worker_save(rank: int, init_path: str, root: str, workdir: str) -> None:
    """Phase 1: take real steps, save the per-rank checkpoint, dump ground truth."""

    torch.set_num_threads(1)
    init_group(init_path, rank)
    try:
        model, network = build_distributed_model()
        runtime = build_runtime(rank, model, network)
        batch = tiny_batch()
        losses = [one_step(runtime, batch) for _ in range(SAVE_AFTER_STEPS)]
        identity = run_identity(model)
        saved = save_distributed_checkpoint(
            root, runtime, step=SAVE_AFTER_STEPS, identity=identity
        )
        # Captured after the save: nothing mutates state between the two calls.
        snapshot = capture_stage_snapshot(
            runtime, step=SAVE_AFTER_STEPS, identity=identity
        )
        torch.save(snapshot, Path(workdir) / f"expected-rank{rank}.pt")
        _write_json(
            Path(workdir) / f"save-rank{rank}.json",
            {
                "rank": rank,
                "step": saved.step,
                "root": str(saved.root),
                "files": sorted(path.name for path in Path(root).iterdir()),
                "world_size": saved.manifest["world_size"],
                "steps": losses,
                "digest": snapshot.digest(),
                "topology": snapshot.topology,
                "adapters": sorted(snapshot.adapters),
            },
        )
    except Exception:
        (Path(workdir) / f"save-rank{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def worker_restore(
    rank: int, init_path: str, root: str, expected_dir: str, workdir: str
) -> None:
    """Phase 2 (fresh processes): load, prove bit equality, replay a resume."""

    torch.set_num_threads(1)
    init_group(init_path, rank)
    try:
        model, network = build_distributed_model()
        runtime = build_runtime(rank, model, network)
        identity = run_identity(model)
        expected: PpStageSnapshot = torch.load(
            Path(expected_dir) / f"expected-rank{rank}.pt", weights_only=False
        )
        state = load_distributed_checkpoint(root, runtime, identity=identity)
        restored = capture_stage_snapshot(runtime, step=state.step, identity=identity)

        # torch.equal per tensor, == per scalar: the acceptance claim itself.
        tensors_equal = all(
            torch.equal(expected.adapters[name][key], restored.adapters[name][key])
            for name in expected.adapters
            for key in expected.adapters[name]
        )
        optimizer_tensors_equal = _optimizer_tensors_equal(
            expected.optimizer, restored.optimizer
        )

        # ``benchmark.verify_resume`` shape: save -> step -> load -> step.
        batch = tiny_batch()
        scratch = Path(workdir) / "replay"
        save_distributed_checkpoint(scratch, runtime, step=state.step, identity=identity)
        after_save = one_step(runtime, batch)
        moved = stage_adapter_state(runtime.stage_module)
        load_distributed_checkpoint(scratch, runtime, identity=identity)
        after_reload = one_step(runtime, batch)
        replayed = stage_adapter_state(runtime.stage_module)
        replay_max_abs = max(
            float((moved[name][key] - replayed[name][key]).abs().max())
            for name in moved
            for key in moved[name]
        )
        replay_loss_delta = abs(after_save["loss"] - after_reload["loss"])

        report = {
            "rank": rank,
            "step": state.step,
            "restored_step": restored.step,
            "tensors_equal": tensors_equal,
            "optimizer_tensors_equal": optimizer_tensors_equal,
            "comparison": compare_snapshots(expected, restored),
            "digest_equal": expected.digest() == restored.digest(),
            "restored_digest": restored.digest(),
            "replay_adapter_max_abs": replay_max_abs,
            "replay_loss_delta": replay_loss_delta,
            "replay_scale_delta": abs(after_save["scale"] - after_reload["scale"]),
            "replay_skipped": [after_save["skipped"], after_reload["skipped"]],
            "root_files": sorted(path.name for path in Path(root).iterdir()),
        }
        _write_json(Path(workdir) / f"restore-rank{rank}.json", report)
    except Exception:
        (Path(workdir) / f"restore-rank{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def _optimizer_tensors_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _optimizer_tensors_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _optimizer_tensors_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    return left == right


__all__ = [
    "BATCH",
    "CHANNELS",
    "MICROBATCHES",
    "MODEL_SEED",
    "SAVE_AFTER_STEPS",
    "build_distributed_model",
    "build_runtime",
    "init_group",
    "local_scaler",
    "one_step",
    "run_identity",
    "run_pair",
    "run_pair_expect_failure",
    "stage_lora_parameters",
    "tiny_batch",
    "tiny_lora",
    "tiny_model",
    "worker_restore",
    "worker_save",
]
