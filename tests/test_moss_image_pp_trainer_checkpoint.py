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
from safetensors.torch import load_file

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


def _handoff(args: Namespace, rank: int, batches_per_epoch: int):
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
        train_dataset_group=_SessionDatasetGroup(count=batches_per_epoch),
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


def _run_session_worker(
    rank, init_path, steps, batches_per_epoch, save_every, resume, output_dir
):
    """Exercise the reported production session on both ranks."""

    from library.training import precision_report
    from library.training.pp.trainer_session import run_pp_trainer_session

    init_gloo(init_path, rank)
    try:
        args = _session_args(Path(output_dir), steps=steps, resume=resume)
        args.save_every_n_steps = save_every
        handoff = _handoff(args, rank, batches_per_epoch)

        def run(trainer, args):
            precision_report.observe("accelerator_ready", handoff.accelerator)
            run_pp_trainer_session(trainer, args, handoff)

        precision_report.reported_session(_SessionTrainer(), args, run)
    finally:
        torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# session-level save/restore across a real process restart
# ---------------------------------------------------------------------------


def test_session_straight_run_saves_bundles_and_final_lora(tmp_path):
    from tests.moss_pp_test_support import run_pair

    run_pair(tmp_path, _run_session_worker, 6, 6, 2, None)
    result = json.loads((tmp_path / "pp_runs" / "LATEST.json").read_text())
    bundle = Path(result["root"])
    assert (bundle / "pp_checkpoint.json").is_file()
    assert (bundle / "pp_stage-rank0.pt").is_file()
    assert (bundle / "pp_stage-rank1.pt").is_file()
    assert (bundle / "trainer_state_rank0.pt").is_file()
    assert (bundle / "trainer_state_rank1.pt").is_file()
    final = tmp_path / "pp-session-test.safetensors"
    assert final.is_file(), "the final LoRA must land in the ordinary output format"
    state = load_file(str(final))
    assert len(state) > 0
    report_path, = tmp_path.glob("precision_report*.json")
    report = json.loads(report_path.read_text())
    assert (report["status"], report["stage"]) == ("completed", "finish")
    assert [report["updates"][key] for key in ("attempted", "effective", "skipped", "total_global_step")] == [6, 6, 0, 6]
    assert report["amp"]["scaler_source"] == "pp_transaction"
    assert report["amp"]["scaler_present"] is True


@pytest.mark.parametrize("batches_per_epoch,checkpoint_step,total_steps", [(6, 4, 6), (1, 5, 10)])
def test_session_resume_lands_on_the_straight_run_state(tmp_path, batches_per_epoch, checkpoint_step, total_steps):
    """Resume both within an epoch and after the only batch of an epoch."""
    from tests.moss_pp_test_support import run_pair

    straight_dir = tmp_path / "straight"
    straight_dir.mkdir()
    run_pair(straight_dir, _run_session_worker, total_steps, batches_per_epoch, checkpoint_step, None)
    bundle, = (straight_dir / "pp_runs").glob(f"*-step{checkpoint_step}")
    trainer_state = torch.load(bundle / "trainer_state_rank0.pt", weights_only=False)
    assert trainer_state["global_step"] == trainer_state["effective_step"] == checkpoint_step
    assert trainer_state["epoch"] == (checkpoint_step - 1) // batches_per_epoch
    assert trainer_state["batch_index"] == (checkpoint_step - 1) % batches_per_epoch + 1
    assert trainer_state["scheduler"] is not None

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    run_pair(resumed_dir, _run_session_worker, total_steps, batches_per_epoch, checkpoint_step, str(bundle))
    straight = load_file(straight_dir / "pp-session-test.safetensors")
    resumed = load_file(resumed_dir / "pp-session-test.safetensors")
    assert resumed.keys() == straight.keys()
    assert all(torch.equal(resumed[key], value) for key, value in straight.items())
    assert json.loads((resumed_dir / "pp_runs" / "LATEST.json").read_text())["step"] == total_steps
    report_path, = resumed_dir.glob("precision_report*.json")
    report = json.loads(report_path.read_text())
    new_steps = total_steps - checkpoint_step
    assert (report["status"], report["stage"]) == ("completed", "finish")
    assert report["resume"]["inherited_global_step"] == checkpoint_step
    assert [report["updates"][key] for key in ("attempted", "effective", "skipped", "total_global_step")] == [
        new_steps, new_steps, 0, total_steps,
    ]


def test_session_refuses_resume_at_target_without_publishing(tmp_path):
    from tests.moss_pp_test_support import run_pair, run_pair_expect_failure

    original = tmp_path / "original"
    original.mkdir()
    run_pair(original, _run_session_worker, 2, 1, 2, None)
    bundle, = (original / "pp_runs").glob("*-step2")
    refused = tmp_path / "refused"
    refused.mkdir()
    errors = run_pair_expect_failure(
        refused, _run_session_worker, 2, 1, 2, str(bundle)
    )
    assert all("pp_resume_no_work" in error for error in errors)
    assert not (refused / "pp_runs").exists()
    assert not list(refused.glob("*.safetensors"))
    report_path, = refused.glob("precision_report*.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["updates"]["attempted"] == 0
    assert report["updates"]["total_global_step"] == 2


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
