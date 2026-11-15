"""Two-rank CUDA harness for the Anima PP trainer hot test (§6/§7).

Spawned only by ``tests/test_anima_pp_trainer_hot.py::test_two_rank_cuda_trainer_resume``
on the two authorized V100s.  The worker mirrors the CPU/Gloo session pair in
``tests/test_anima_pp_trainer_checkpoint.py`` but runs on ``cuda`` with the
production fp16 path (real GradScaler, fp32 LoRA masters), straight run +
resume run against the same tiny Anima fixture.
"""

from __future__ import annotations

import json
import traceback
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path

import torch
import torch.distributed as dist

V100_UUIDS = (
    "GPU-b4b014b2-953b-28e8-b8c1-425c4a401165",
    "GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061",
)
DIST_TIMEOUT = timedelta(seconds=45)
JOIN_TIMEOUT_SECONDS = 60


def _authorized_devices() -> bool:
    """Both visible CUDA devices must be exactly the two V100s (no 1050)."""

    wanted = {value.split("GPU-")[1] for value in V100_UUIDS}
    visible = set()
    for index in range(torch.cuda.device_count()):
        # str(torch._C._CUuuid) is the canonical dashed form.
        visible.add(str(torch.cuda.get_device_properties(index).uuid))
    return visible == wanted


def _cuda_session_worker(rank, init_path, workdir):
    from types import SimpleNamespace

    from tests.pp_checkpoint_test_support import BATCH, CHANNELS, tiny_lora, tiny_model
    from library.training.pp.trainer_session import PpSessionHandoff, run_pp_trainer_session

    torch.set_num_threads(1)
    workdir = Path(workdir)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    def _trainer():
        instance = SimpleNamespace(_network=None, _adapters=[])
        instance.shift_scale_latents = lambda args, latents: latents

        def get_noise_scheduler(args, dev):
            from library.runtime.noise import FlowMatchEulerDiscreteScheduler

            return FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000, shift=args.discrete_flow_shift
            )

        instance.get_noise_scheduler = get_noise_scheduler
        instance.get_sai_model_spec = lambda args: {}
        instance.update_metadata = lambda metadata, args: metadata
        return instance

    def _session_args(steps, resume=None):
        from tests.test_anima_pp_trainer_checkpoint import _session_args

        args = _session_args(workdir, steps=steps, resume=resume)
        args.output_dir = str(workdir / ("straight" if resume is None else "resumed"))
        args.gradient_checkpointing = False
        return args

    class _DatasetGroup(list):
        batch_size = 1
        num_train_images = BATCH
        num_reg_images = 0
        # Empty datasets tuple: the user-config metadata branch iterates it
        # (same stub contract as the MOSS hot fixture).
        datasets = ()

        def set_current_strategies(self):
            pass

    def _dataset():
        generator = torch.Generator().manual_seed(4242)
        latents = torch.randn(BATCH, 4, 4, 4, generator=generator)
        crossattn = torch.randn(BATCH, 6, CHANNELS, generator=generator)
        item = {
            "latents": latents,
            "text_encoder_outputs_list": [
                torch.randn(BATCH, 6, CHANNELS, generator=generator),
                None,
                None,
                None,
                crossattn,
            ],
        }
        group = _DatasetGroup()
        group.extend([dict(item) for _ in range(6)])
        return group

    def _handoff(args):
        model = tiny_model().to(dtype=torch.float16, device=device)
        network = tiny_lora(model)
        accelerator = SimpleNamespace(
            num_processes=2,
            device=device,
            is_main_process=(rank == 0),
            process_index=rank,
            print=print,
            unwrap_model=lambda network: getattr(network, "module", network),
            end_training=lambda: None,
        )
        return PpSessionHandoff(
            args=args,
            accelerator=accelerator,
            network=network,
            unet=model,
            vae=None,
            vae_dtype=None,
            text_encoders=[],
            train_dataset_group=_dataset(),
            val_dataset_group=None,
            collator=lambda items: items[0],
            weight_dtype=torch.float16,
            save_dtype=torch.float32,
            tokenizers=[],
            training_started_at=0.0,
            session_id=1,
            model_version="tiny",
            use_user_config=True,
            use_dreambooth_method=False,
            text_encoding_strategy=None,
            tokenize_strategy=None,
            train_unet=True,
            train_text_encoder=False,
            net_kwargs={},
        )

    def _digest(network):
        import hashlib

        hasher = hashlib.sha256()
        for lora in sorted(network.unet_loras, key=lambda m: m.lora_name):
            for key, value in sorted(lora.state_dict().items()):
                hasher.update(key.encode())
                hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()

    def _latest(args):
        pointer = Path(args.output_dir) / "pp_runs" / "LATEST.json"
        return str(json.loads(pointer.read_text())["root"])

    try:
        # Same multi-backend group the production accelerator injects
        # (library/runtime/accelerator.py): CPU scalars via gloo, device
        # activations via nccl -- an NCCL-only group fails the fp16
        # transaction's CPU tensor collectives.
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"file://{init_path}",
            rank=rank,
            world_size=2,
            timeout=DIST_TIMEOUT,
        )
        args = _session_args(6)
        run_pp_trainer_session(_trainer(), args, _handoff(args))
        straight_digest = _digest(_handoff(args).network)
        straight_bundle = _latest(args)

        args = _session_args(4)
        # Separate root: reusing the straight directory would make the step-4
        # periodic save collide with the straight run's bundle root and be
        # idempotently skipped (LATEST.json would keep pointing at step 6).
        args.output_dir = str(workdir / "split")
        run_pp_trainer_session(_trainer(), args, _handoff(args))
        bundle = _latest(args)
        manifest = json.loads((Path(bundle) / "pp_checkpoint.json").read_text())

        args = _session_args(6, resume=bundle)
        run_pp_trainer_session(_trainer(), args, _handoff(args))
        resumed_digest = _digest(_handoff(args).network)
        assert resumed_digest == straight_digest, (
            "split(4)+resume(2) must reproduce straight(6) on the CUDA pair"
        )
        (workdir / f"hot.rank{rank}.json").write_text(json.dumps({
            "rank": rank,
            "straight_bundle": straight_bundle,
            "split_step": manifest["step"],
            "resumed_digest": resumed_digest,
        }))
    except BaseException:
        (workdir / f"hot.rank{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def run_cuda_session_pair(workdir) -> dict:
    """Spawn the two CUDA ranks; return the straight/resume digest verdict."""

    assert _authorized_devices(), (
        "the Anima PP hot test requires exactly the two authorized V100s; the "
        "GTX 1050 must never be visible to this process"
    )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    init_path = str(workdir / "init")
    context = get_context("spawn")
    processes = [
        context.Process(target=_cuda_session_worker, args=(rank, init_path, str(workdir)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
        errors = "\n".join(path.read_text() for path in sorted(workdir.glob("hot.rank*.error")))
        assert all(process.exitcode == 0 for process in processes), (
            f"Anima PP CUDA hot worker failed:\n{errors}"
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    rank0 = json.loads((workdir / "hot.rank0.json").read_text())
    rank1 = json.loads((workdir / "hot.rank1.json").read_text())
    assert rank0["split_step"] == 4 and rank1["split_step"] == 4
    return rank0
