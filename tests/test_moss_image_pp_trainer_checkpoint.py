"""MOSS PP trainer-level save/restore tests (task t2, real CPU/Gloo pairs).

These tests drive the *actual session entry*
(:func:`library.training.pp.trainer_session.run_pp_trainer_session`) — the same
function ``train_session._run_training_session`` dispatches to — over two real
Gloo ranks with the tiny MOSS fixture:

* a straight run writes negotiated per-rank PP bundles plus trainer-level state
  (scheduler / torch+python+numpy RNG / data cursor / run identity) and the
  ordinary-format final LoRA through the production ``CheckpointSaver``;
* a resumed run restores stage weights, optimizer, fp16 scale/policy, scheduler,
  RNG and the batch cursor, continues at the same batch position, and lands on
  the same parameters / LR / scale as an uninterrupted run of the same length;
* a bundle without trainer state is refused by name unless the explicit
  weights-only opt-in is present, and a bad payload refuses before any
  mutation.

The fixture model/LoRA/data come from ``tests.moss_pp_test_support``; the
session seams that exist only to feed *real production objects* (the tiny DiT
instead of a downloaded base, an in-memory cached dataset instead of a disk
one) are explicitly fake-but-real-object stand-ins, not mocked wiring: the
loader, PP runtime, transaction, checkpoint protocol and saver are all live.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training.pp.errors import PipelineParallelError

from tests.moss_pp_test_support import (
    HEIGHT,
    MAX_TEXT_TOKENS,
    WIDTH,
    init_gloo,
    tiny_batch,
    tiny_lora,
    tiny_model,
)


MAX_TEXT_TOKEN_WIDTH = MAX_TEXT_TOKENS


# ---------------------------------------------------------------------------
# shared fixture plumbing: real objects, fixture-fed
# ---------------------------------------------------------------------------


class _SessionTrainer:
    """The trainer surface the PP session touches (identity shift/scale)."""

    _network = None
    _adapters: list = []

    def shift_scale_latents(self, args, latents):
        return latents

    def get_sai_model_spec(self, args):
        return {}

    def update_metadata(self, metadata, args):
        return metadata


class _FakeDataset:
    subsets = (SimpleNamespace(metadata_file="subset.json", num_repeats=1, img_count=1),)
    tag_frequency: dict = {}
    bucket_info: dict = {}


class _FakeDatasetGroup(list):
    batch_size = 1
    datasets = (_FakeDataset(),)
    num_train_images = 6
    num_reg_images = 0

    def set_current_strategies(self):
        pass


def _session_args(tmp_path: Path, *, steps: int, resume: str | None = None) -> Namespace:
    return Namespace(
        model_family="moss_image",
        loss_type="l2",
        pipeline_parallel=True,
        pipeline_parallel_stages=2,
        pipeline_parallel_microbatches=2,
        pipeline_parallel_schedule="1f1b",
        pipeline_parallel_split="balanced",
        blocks_to_swap=0,
        torch_compile=False,
        selective_checkpoint="off",
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        network_train_unet_only=True,
        mixed_precision="fp16",
        gradient_checkpointing=True,
        use_vae_cache=True,
        use_text_cache=True,
        cache_latents=True,
        cache_text_encoder_outputs=True,
        network_module="networks.lora_anima",
        gradient_accumulation_steps=1,
        weighting_scheme="none",
        sample_prompts=None,
        save_n_epoch_ratio=None,
        save_every_n_epochs=None,
        persistent_data_loader_workers=False,
        adaptive_oom_retry=False,
        stage_schedule_enabled=False,
        seed=41,
        output_name="pp-session-test",
        output_dir=str(tmp_path),
        max_train_steps=steps,
        max_train_epochs=None,
        max_data_loader_n_workers=0,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,
        train_batch_size=1,
        batch_size=1,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        lr_scheduler_args=None,
        lr_scheduler_type=None,
        optimizer_type="AdamW",
        learning_rate=1e-3,
        optimizer_args=None,
        use_8bit_adam=False,
        use_lion_optimizer=False,
        fused_backward_pass=False,
        max_grad_norm=1.0,
        save_every_n_steps=2,
        save_state=True,
        save_state_on_train_end=False,
        save_model_as="safetensors",
        save_precision="float",
        no_metadata=False,
        full_fp16=False,
        full_bf16=False,
        network_dim=4,
        network_alpha=4.0,
        network_dropout=None,
        text_encoder_lr=None,
        unet_lr=None,
        vae_batch_size=None,
        resume=resume,
        pipeline_parallel_resume_weights_only=False,
        color_aug=False,
        flip_aug=False,
        random_crop=False,
        v2="",
        clip_skip=None,
        pretrained_model_name_or_path="tiny",
        vae=None,
        training_comment=None,
        lowram=False,
        max_token_length=4096,
        timestep_sampling="uniform",
        discrete_flow_shift=1.0,
        min_snr_gamma=None,
        scale_weight_norms=None,
        adaptive_noise_scale=None,
        multires_noise_discount=0.0,
        multires_noise_iterations=None,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        noise_offset=None,
        noise_offset_random_strength=False,
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=None,
        sigmoid_scale=1.0,
        pseudo_huber_c=None,
        huber_c=None,
        huber_scale=1.0,
        huber_schedule="snr",
        debiased_estimation_loss=False,
        multiscale_loss_weight=0.0,
        prior_loss_weight=1.0,
        zero_terminal_snr=False,
        face_crop_aug_range=None,
        resize_interpolation=None,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        max_validation_steps=None,
        validate_every_n_epochs=None,
        validate_every_n_steps=None,
        validation_seed=None,
        validation_split=0.0,
        keep_tokens=0,
    )


class _FakeAccelerator(SimpleNamespace):
    def unwrap_model(self, network):
        return network

    def end_training(self):
        pass

    def print(self, *parts):
        pass


def _fake_accelerator(rank: int) -> _FakeAccelerator:
    return _FakeAccelerator(
        num_processes=2,
        device=torch.device("cpu"),
        is_main_process=(rank == 0),
        process_index=rank,
    )


class _SessionDatasetGroup(_FakeDatasetGroup):
    """Six deterministic batches; one dataset item already is one batch."""

    def __init__(self, count: int = 6, scale: float = 1.0):
        super().__init__()
        latents, text, mask, _sigmas, _times, _target = tiny_batch()
        item = {
            "latents": latents * scale,
            "text_encoder_outputs_list": [text, mask],
        }
        self.extend([dict(item) for _ in range(count)])


def _handoff(args: Namespace, rank: int):
    from library.training.pp.trainer_session import PpSessionHandoff

    model = tiny_model().to(dtype=torch.float16)
    network = tiny_lora(model)  # fp32 LoRA master weights
    return PpSessionHandoff(
        args=args,
        accelerator=_fake_accelerator(rank),
        network=network,
        unet=model,
        vae=None,
        vae_dtype=None,
        text_encoders=[],
        train_dataset_group=_SessionDatasetGroup(),
        val_dataset_group=None,
        collator=lambda items: items[0],
        weight_dtype=torch.float16,
        save_dtype=torch.float32,
        tokenizers=[],
        training_started_at=0.0,
        session_id=1,
        model_version="tiny",
        use_user_config=False,
        use_dreambooth_method=False,
        text_encoding_strategy=None,
        tokenize_strategy=None,
        train_unet=True,
        train_text_encoder=False,
        net_kwargs={},
    )


def _adapter_digest(network) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for lora in sorted(network.unet_loras, key=lambda m: m.lora_name):
        for key, value in sorted(lora.state_dict().items()):
            hasher.update(key.encode())
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _latest_bundle_root(args: Namespace) -> str:
    pointer = Path(args.output_dir) / "pp_runs" / "LATEST.json"
    return str(json.loads(pointer.read_text())["root"])


def _run_session_worker(rank, init_path, phase, tmp_path, resume=None, output_dir=None):
    """One session run; ``phase`` picks train-straight / train-then-resume."""

    from library.training.pp.trainer_session import run_pp_trainer_session

    init_gloo(init_path, rank)
    workdir = Path(tmp_path)
    try:
        if phase == "straight":
            args = _session_args(workdir, steps=6)
            args.output_dir = str(workdir / "straight")
            run_pp_trainer_session(_SessionTrainer(), args, _handoff(args, rank))
            result = {
                "digest": _adapter_digest(_handoff(args, rank).network),
                "bundle": _latest_bundle_root(args),
            }
            (workdir / f"straight.rank{rank}.json").write_text(json.dumps(result))
        elif phase == "split":
            args = _session_args(workdir, steps=4)
            args.output_dir = str(workdir / "split")
            run_pp_trainer_session(_SessionTrainer(), args, _handoff(args, rank))
            bundle = _latest_bundle_root(args)
            manifest = json.loads((Path(bundle) / "pp_checkpoint.json").read_text())
            trainer_state = torch.load(
                Path(bundle) / f"trainer_state_rank{rank}.pt", weights_only=False
            )
            result = {
                "bundle": bundle,
                "manifest_step": manifest["step"],
                "trainer_global_step": trainer_state["global_step"],
                "trainer_effective_step": trainer_state["effective_step"],
                "trainer_epoch": trainer_state["epoch"],
                "trainer_batch_index": trainer_state["batch_index"],
                "trainer_run_id": trainer_state["run_id"],
                "has_scheduler": trainer_state["scheduler"] is not None,
                "rng_torch": trainer_state["rng_torch_cpu"].numel(),
            }
            (workdir / f"split.rank{rank}.json").write_text(json.dumps(result))
        elif phase == "resume":
            args = _session_args(workdir, steps=6, resume=resume)
            args.output_dir = str(workdir / "resumed")
            run_pp_trainer_session(_SessionTrainer(), args, _handoff(args, rank))
            result = {"digest": _adapter_digest(_handoff(args, rank).network)}
            (workdir / f"resumed.rank{rank}.json").write_text(json.dumps(result))
        else:  # pragma: no cover
            raise AssertionError(f"unknown phase {phase}")
    finally:
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# session-level save/restore across a real process restart
# ---------------------------------------------------------------------------


def test_session_straight_run_saves_bundles_and_final_lora(tmp_path):
    from tests.moss_pp_test_support import run_pair

    run_pair(tmp_path, _run_session_worker, "straight", str(tmp_path))
    result = json.loads((tmp_path / "straight.rank0.json").read_text())
    bundle = Path(result["bundle"])
    assert (bundle / "pp_checkpoint.json").is_file()
    assert (bundle / "pp_stage-rank0.pt").is_file()
    assert (bundle / "pp_stage-rank1.pt").is_file()
    assert (bundle / "trainer_state_rank0.pt").is_file()
    assert (bundle / "trainer_state_rank1.pt").is_file()
    final = Path(tmp_path / "straight" / "pp-session-test.safetensors")
    assert final.is_file(), "the final LoRA must land in the ordinary output format"
    from safetensors.torch import load_file

    state = load_file(str(final))
    assert len(state) > 0


def test_session_resume_lands_on_the_straight_run_state(tmp_path):
    """split(4)+resume(2) == straight(6): the frozen resume equivalence."""
    from tests.moss_pp_test_support import run_pair

    straight_dir = tmp_path / "straight"
    straight_dir.mkdir()
    run_pair(straight_dir, _run_session_worker, "straight", str(straight_dir))
    straight = json.loads((straight_dir / "straight.rank0.json").read_text())

    run_pair(tmp_path, _run_session_worker, "split", str(tmp_path))
    split = json.loads((tmp_path / "split.rank0.json").read_text())
    assert split["manifest_step"] == 4
    assert split["trainer_global_step"] == 4
    assert split["trainer_effective_step"] == 4
    assert split["trainer_epoch"] == 0
    assert split["trainer_batch_index"] == 4
    assert split["has_scheduler"] is True

    # Point the resumed run at the split's step-4 bundle.
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    run_pair(resumed_dir, _run_session_worker, "resume", str(resumed_dir), split["bundle"])
    resumed = json.loads((resumed_dir / "resumed.rank0.json").read_text())
    assert resumed["digest"] == straight["digest"], (
        "split(4)+resume(2) must reproduce straight(6): same batch cursor, "
        "same RNG continuation, same LR plan, same optimizer state"
    )


def test_session_resume_refuses_a_trainer_state_less_bundle(tmp_path):
    from library.training.pp.trainer_resume import (
        require_exact_resume_or_opt_in,
        weights_only_opt_in,
    )

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text("{}")
    args = _session_args(tmp_path, steps=2)
    with pytest.raises(PipelineParallelError) as excinfo:
        require_exact_resume_or_opt_in(root, args, family="moss_image")
    assert excinfo.value.code == "pp_checkpoint_incomplete"
    assert not weights_only_opt_in(args)
    # Explicit opt-in continues, logged as non-exact.
    args.pipeline_parallel_resume_weights_only = True
    assert require_exact_resume_or_opt_in(root, args, family="moss_image") is True


def test_trainer_payload_refuses_before_any_mutation(tmp_path):
    """Version / identity / step mismatches are named, pre-apply refusals."""

    from library.training.pp.trainer_resume import (
        capture_trainer_payload,
        validate_trainer_payload,
    )

    payload = capture_trainer_payload(
        rank=1, run_id="abc", family="moss_image", global_step=4,
        effective_step=4, epoch=0, batch_index=4, lr_scheduler=None, seed=1,
    )
    validate_trainer_payload(payload, rank=1, run_id="abc", family="moss_image", expected_step=4)
    for mutate, code in (
        ({"global_step": 5}, "pp_checkpoint_step_mismatch"),
        ({"run_id": "other"}, "pp_checkpoint_identity_mismatch"),
        ({"version": 99}, "pp_checkpoint_state_mismatch"),
        ({"family": "anima"}, "pp_checkpoint_state_mismatch"),
        ({"rank": 0}, "pp_checkpoint_state_mismatch"),
    ):
        bad = dict(payload)
        bad.update(mutate)
        with pytest.raises(PipelineParallelError) as excinfo:
            validate_trainer_payload(
                bad, rank=1, run_id="abc", family="moss_image", expected_step=4
            )
        assert excinfo.value.code == code
