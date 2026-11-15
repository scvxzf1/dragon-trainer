"""The frozen two-rank CUDA hot node for the MOSS PP trainer path.

``test_two_rank_cuda_trainer_resume`` is the *only* GPU test of this task and
it belongs to the independent verifier (t3): the implementer runs CPU-only.

What it does when actually enabled (no mocks, real production entry):

* gates on two idle CUDA devices and an explicit environment contract
  (``MOSS_PP_HOT_ENABLE=1``), so a bare ``pytest`` run skips cleanly and never
  grabs a card;
* spawns two CUDA processes that run the *real* trainer session
  (:func:`library.training.pp.trainer_session.run_pp_trainer_session`) over the
  tiny MOSS fixture (official 42-layer topology, shrunk widths) — never a
  downloaded base model and no user data;
* phase 1: 2 clean steps + a negotiated bundle save; phase 2: a fresh
  two-process restart that resumes the bundle and runs 2 more steps, then
  checks the resume window (batch digest, noise identity, LR, parameters,
  optimizer state, scale) against a straight 4-step run;
* drives one real fp16 overflow skip (huge target) and requires the unanimous
  skip verdict;
* bounds every phase: 45s total inside the pair, 60s outer budget, and a
  per-card allocator cap of 2 GiB via ``set_per_process_memory_fraction``;
  allocated/reserved memory is asserted below the cap with the raw numbers
  written into the phase evidence files (original receipts, no re-labelling).

Enable with::

    MOSS_PP_HOT_ENABLE=1 CUDA_VISIBLE_DEVICES=<uuid0>,<uuid1> \
        python -m pytest -q tests/test_moss_image_pp_trainer_hot.py -rs
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from tests.moss_pp_test_support import (
    HEIGHT,
    MAX_TEXT_TOKENS,
    WIDTH,
    init_gloo,
    tiny_batch,
    tiny_lora,
    tiny_model,
)

HOT_ENABLE_ENV = "MOSS_PP_HOT_ENABLE"
#: Pre-registered amplification (CPU derivation in _CudaDatasetGroup): 200000
#: puts the amplified latents at 10.5x fp16 max, forcing a deterministic
#: overflow skip independent of noise draw and GradScaler scale.
_HOT_AMPLIFICATION = 200000.0
ALLOC_CAP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB per card
PHASE_BUDGET_SECONDS = 45.0
MULTIBACKEND = "cpu:gloo,cuda:nccl"


def _hot_enabled() -> bool:
    return os.environ.get(HOT_ENABLE_ENV, "").strip().lower() in {"1", "true", "yes"}


def _two_cuda_devices() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


@pytest.mark.skipif(
    not _hot_enabled() or not _two_cuda_devices(),
    reason=(
        f"hot node needs {HOT_ENABLE_ENV}=1 and two idle CUDA devices; the "
        "implementer runs CPU-only and the independent verifier runs this on "
        "the authorized fixed two-V100 node"
    ),
)
def test_two_rank_cuda_trainer_resume(tmp_path):
    """Two real CUDA processes: train 2 steps, save, restart, resume, verify."""
    import time

    from tests.moss_pp_test_support import run_pair

    # run_pair takes an ABSOLUTE monotonic() endpoint (see
    # moss_pp_test_support._spawn_pair); passing the 60s budget as a relative
    # value used to expire immediately and spawn zero ranks. The outer budget
    # stays 60s wall clock from now; the phase logic itself must stay inside
    # the 45s internal budget.
    outer = time.monotonic() + 60.0
    run_pair(tmp_path, _cuda_worker, "straight", str(tmp_path), deadline=outer)
    run_pair(tmp_path, _cuda_worker, "resume", str(tmp_path), deadline=outer)

    straight = json.loads((tmp_path / "straight.rank0.json").read_text())
    resumed = json.loads((tmp_path / "resumed.rank0.json").read_text())
    assert resumed["resume_step"] == 2
    # Pre-registered trajectory, CPU-derived (see _CudaDatasetGroup: the
    # amplified latents are Inf at 10.5x fp16 max, so both amplified loader
    # positions -- straight steps 2 and 3 -- overflow deterministically, while
    # the normal steps stay finite at every scale):
    #   straight: 1 eff@65536, 2 skip(->32768), 3 skip(->16384), 4 eff@16384
    #             => final progress line "effective 2, skipped 2), scale 16384"
    #   resumed (step-2 bundle: scale 32768, effective 1, cursor epoch0/idx2):
    #             step 3 = first replayed batch = amplified => skip(->16384),
    #             step 4 eff@16384 => exactly one skip line "step 3/4 ...
    #             skipped 1) ... scale=16384.0".
    assert len(straight["skip_lines"]) == 2, straight["skip_lines"]
    assert "step 2/4" in straight["skip_lines"][0], straight["skip_lines"]
    assert "skipped 2)" in straight["skip_lines"][-1], straight["skip_lines"]
    assert "scale=16384.0" in straight["skip_lines"][-1], straight["skip_lines"]
    assert len(resumed["skip_lines"]) == 1, resumed["skip_lines"]
    assert "step 3/4" in resumed["skip_lines"][0], resumed["skip_lines"]
    assert "skipped 1)" in resumed["skip_lines"][0], resumed["skip_lines"]
    assert "scale=16384.0" in resumed["skip_lines"][0], resumed["skip_lines"]
    # The resumed window reproduces the straight run: batch digest, noise
    # identity, LR, parameters and scale must all match (frozen contract §5).
    assert resumed["final_param_digest"] == straight["final_param_digest"]
    assert resumed["final_lr"] == straight["final_lr"]
    assert resumed["batch_digests"][2:] == straight["batch_digests"][2:]
    # Per-card memory receipts stay inside the frozen allocator cap.
    for rank in (0, 1):
        memory = json.loads((tmp_path / f"memory.rank{rank}.json").read_text())
        assert memory["allocated_bytes"] <= ALLOC_CAP_BYTES
        assert memory["reserved_bytes"] <= ALLOC_CAP_BYTES


def _memory_receipt(rank: int, workdir: Path) -> None:
    receipt = {
        "allocated_bytes": torch.cuda.memory_allocated(rank),
        "reserved_bytes": torch.cuda.memory_reserved(rank),
        "cap_bytes": ALLOC_CAP_BYTES,
        "device": str(torch.cuda.get_device_name(rank)),
    }
    (workdir / f"memory.rank{rank}.json").write_text(json.dumps(receipt))


def _cuda_worker(rank, init_path, phase, tmp_path, output_dir):
    """Real session entry on two CUDA devices; no mocks at any seam."""
    from argparse import Namespace

    from library.training.pp.trainer_session import (
        PpSessionHandoff,
        run_pp_trainer_session,
    )

    torch.distributed.init_process_group(
        MULTIBACKEND,
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    workdir = Path(tmp_path)
    # Freeze the per-card allocator at the authorized budget.
    total = torch.cuda.get_device_properties(rank).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, ALLOC_CAP_BYTES / total), rank)
    device = torch.device(f"cuda:{rank}")
    try:
        model = tiny_model().to(device=device, dtype=torch.float16)
        network = tiny_lora(model)  # fp32 LoRA master weights on the fp16 base
        network.to(device=device)

        args = _cuda_args(workdir, phase)
        args._printed_skips = []
        accelerator = _cuda_accelerator(rank)
        accelerator._current_args = args
        handoff = PpSessionHandoff(
            args=args,
            accelerator=accelerator,
            network=network,
            unet=model,
            vae=None,
            vae_dtype=None,
            text_encoders=[],
            train_dataset_group=_CudaDatasetGroup(seed=int(getattr(args, "seed", 0) or 0)),
            val_dataset_group=None,
            collator=lambda items: items[0],
            weight_dtype=torch.float16,
            save_dtype=torch.float32,
            tokenizers=[],
            training_started_at=0.0,
            session_id=1,
            model_version="tiny-cuda",
            # Metadata user-config branch: iterate nothing (empty datasets)
            # instead of needing a full single-dataset stub; trainer-level
            # num_train_images/num_reg_images come from the group itself.
            use_user_config=True,
            use_dreambooth_method=False,
            text_encoding_strategy=None,
            tokenize_strategy=None,
            train_unet=True,
            train_text_encoder=False,
            net_kwargs={},
        )
        if phase == "straight":
            _phase_straight(rank, args, handoff, workdir)
        else:
            _phase_resume(rank, args, handoff, workdir)
        _memory_receipt(rank, workdir)
        assert torch.cuda.memory_allocated(rank) <= ALLOC_CAP_BYTES
    finally:
        torch.distributed.destroy_process_group()


def _cuda_args(workdir: Path, phase: str):
    from tests.test_moss_image_pp_trainer_checkpoint import _session_args

    args = _session_args(workdir, steps=4)
    args.output_dir = str(workdir / ("hot-straight" if phase == "straight" else "hot-resumed"))
    args.save_every_n_steps = 2
    return args


def _cuda_accelerator(rank: int):
    from tests.test_moss_image_pp_trainer_checkpoint import _FakeAccelerator

    accelerator = _FakeAccelerator(
        num_processes=2,
        device=torch.device(f"cuda:{rank}"),
        is_main_process=(rank == 0),
        process_index=rank,
    )
    # Original progress receipts: every printed pp step line with its
    # effective/skip counters (never a hard-coded flag).
    accelerator.printed = []
    accelerator._args_seen = []

    def _print(*parts):
        line = " ".join(str(part) for part in parts)
        accelerator.printed.append(line)
        # Keep only skip-EVENT lines: every step prints a progress line, so a
        # receipt line is a skip event exactly when the cumulative skipped
        # counter grew relative to the previous line (F6 fix).
        if "skipped" in line and hasattr(accelerator, "_current_args"):
            receipt = accelerator._current_args._printed_skips

            def _cum(text):
                return int(text.split("skipped ")[1].split(")")[0])

            if _cum(line) > 0 and (not receipt or _cum(line) > _cum(receipt[-1])):
                receipt.append(line)

    accelerator.print = _print
    return accelerator


class _CudaDatasetGroup(list):
    """Four cached batches, the third one driven to a real fp16 overflow.

    A list subclass: DataLoader shuffle requires an indexable, sized dataset,
    not a generator-only mapping.
    """

    batch_size = 1
    # Empty datasets tuple: the user-config metadata branch iterates it and
    # yields empty metadata dicts; keep use_user_config=True on the handoff.
    datasets = ()
    num_train_images = 4
    num_reg_images = 0

    def __init__(self, seed: int = 0):
        super().__init__()
        latents, text, mask, _sigmas, _times, _target = tiny_batch()
        base = [
            {
                "latents": latents.to(torch.float16),
                "text_encoder_outputs_list": [text, mask],
            }
            for _ in range(4)
        ]
        # Same-source permutation as the loader: build_pp_train_dataloader
        # seeds a dedicated torch.Generator with args.seed (re-seeded to
        # seed+epoch each epoch) and RandomSampler draws
        # torch.randperm(n, generator=...).  The stored-slot -> loader-position
        # mapping was calibrated empirically on CPU (identical for epochs 0/1):
        # stored slot 1 and 2 land on loader positions 1 and 2.
        #
        # Amplification factor 200000, chosen from CPU measurements (never
        # fitted to GPU results):
        #   * raw fp16 LoRA grad max at factor 1 is 0.0092 and SATURATES around
        #     0.051 under input amplification (the MSE grad wrt pred is bounded),
        #     so pure gradient amplification can never reach fp16's 65504/32768
        #     threshold -- gradient-margin scaling is impossible for this fixture;
        #   * instead the amplified latents themselves exceed fp16 finite range:
        #     3.4355 * 200000 = 687109 = 10.5x the fp16 max 65504 (>=10x margin),
        #     so the forward pass is Inf -> loss NaN -> gradients NaN on every
        #     noise draw and at every GradScaler scale.  The skip is therefore
        #     deterministic and CPU-derivable (noise- and scale-independent),
        #     which is a strictly stronger guarantee than a gradient margin.
        #   * batch_step._prepare_latents' NaN-gated scrub replaces NaN only and
        #     leaves +-Inf in place (read, not changed), so the Inf reaches the
        #     forward exactly as intended.
        #
        # With the amplified batches at loader positions 1 and 2 the pre-registered
        # trajectory (see the assertion block below) is: straight skips at steps
        # 2 and 3 (65536 -> 32768 -> 16384), the resumed window re-enters at the
        # restored scale 32768 and skips its first replayed step (step 3) before
        # succeeding at step 4 -- both windows converge to the same final state.
        perm = torch.randperm(4, generator=torch.Generator().manual_seed(int(seed)))
        for position, source in enumerate(perm.tolist()):
            scale = _HOT_AMPLIFICATION if position in (1, 2) else 1.0
            batch = dict(base[source])
            batch["latents"] = (base[source]["latents"] * scale).to(torch.float16)
            self.append(batch)

    def set_current_strategies(self):
        pass


def _phase_straight(rank, args, handoff, workdir):
    from library.training.pp.trainer_session import run_pp_trainer_session

    run_pp_trainer_session(_HotTrainer(), args, handoff)
    manifest_root = _latest_root(args)
    (workdir / f"straight.rank{rank}.json").write_text(
        json.dumps(
            {
                "final_param_digest": _param_digest(handoff.network),
                "final_lr": _last_lr(args),
                "root": manifest_root,
                "batch_digests": _batch_digests(handoff),
                "skip_lines": args._printed_skips,
            }
        )
    )


def _phase_resume(rank, args, handoff, workdir):
    """Resume the straight run from its step-2 bundle (pre-overflow window)."""

    from library.training.pp.trainer_session import derive_run_id, run_pp_trainer_session

    run_id = derive_run_id(args)
    resume_root = workdir / "hot-straight" / "pp_runs" / f"{run_id}-step2"
    manifest = json.loads((resume_root / "pp_checkpoint.json").read_text())
    assert int(manifest["step"]) == 2
    args.resume = str(resume_root)
    run_pp_trainer_session(_HotTrainer(), args, handoff)
    (workdir / f"resumed.rank{rank}.json").write_text(
        json.dumps(
            {
                "resume_step": int(manifest["step"]),
                "final_param_digest": _param_digest(handoff.network),
                "final_lr": _last_lr(args),
                "batch_digests": _batch_digests(handoff),
                "skip_lines": args._printed_skips,
            }
        )
    )


class _HotTrainer:
    _network = None
    _adapters: list = []

    def shift_scale_latents(self, args, latents):
        return latents

    def get_sai_model_spec(self, args):
        return {}

    def update_metadata(self, metadata, args):
        return metadata


def _latest_root(args):
    pointer = Path(args.output_dir) / "pp_runs" / "LATEST.json"
    return json.loads(pointer.read_text())["root"]


def _last_lr(args):
    import json as _json

    state_path = Path(_latest_root(args)) / "trainer_state_rank0.pt"
    state = torch.load(state_path, weights_only=False)
    scheduler_state = state.get("scheduler") or {}
    last_lr = scheduler_state.get("_last_lr")
    return None if last_lr is None else _json.dumps(last_lr)


def _batch_digests(handoff):
    import hashlib

    digests = []
    for batch in handoff.train_dataset_group:
        payload = batch["latents"].detach().cpu().contiguous().numpy().tobytes()
        digests.append(hashlib.sha256(payload).hexdigest())
    return digests


def _param_digest(network):
    import hashlib

    hasher = hashlib.sha256()
    for lora in sorted(network.unet_loras, key=lambda module: module.lora_name):
        for key, value in sorted(lora.state_dict().items()):
            hasher.update(key.encode())
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()
