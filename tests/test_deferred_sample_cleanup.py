from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
ANIMA_TRAINING = ROOT / "library" / "anima" / "training.py"


def _section(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    return source[start_index: source.index(end, start_index)]


def test_deferred_sample_decode_runs_on_interrupted_training_loop() -> None:
    source = (ROOT / "library" / "training" / "train_session.py").read_text(encoding="utf-8")
    train_tail = source[source.index("with run_scope(trainer.progress_sink") :]

    assert "training_loop_completed = False" in train_tail
    assert "finally:" in train_tail
    assert "if not training_loop_completed:" in train_tail
    assert "decode_deferred_samples_safely(" in train_tail
    assert "optimizer_eval_fn=optimizer_eval_fn" in train_tail


def test_deferred_sample_decode_still_runs_on_normal_completion() -> None:
    source = (ROOT / "library" / "training" / "train_session.py").read_text(encoding="utf-8")
    train_tail = source[source.index("accelerator.end_training()") :]

    assert "optimizer_eval_fn()" in train_tail
    assert "decode_deferred_samples_safely(accelerator, args, loop_state, vae)" in train_tail


def test_normal_completion_writes_final_model() -> None:
    source = (ROOT / "library" / "training" / "train_session.py").read_text(
        encoding="utf-8"
    )
    train_tail = source[source.index("accelerator.end_training()") :]

    assert "saver.cleanup_resumable()" in train_tail
    assert (
        "saver.save_final(loop_state.network, loop_state.global_step, num_train_epochs)"
        in train_tail
    )


def test_step_limit_runs_epoch_end_artifacts_before_break() -> None:
    source = (ROOT / "library" / "training" / "loop.py").read_text(encoding="utf-8")
    loop_body = _section(source, "def run_training_loop(", "def _current_stage_fields(")

    limit_index = loop_body.index("reached_step_limit =")
    save_index = loop_body.index("state.saver.maybe_save_epoch(")
    sample_index = loop_body.index("trainer.sample_images(")
    break_index = loop_body.index("if reached_step_limit:")
    assert limit_index < save_index < sample_index < break_index


def test_sigterm_uses_keyboard_interrupt_cleanup_path() -> None:
    source = TRAIN_PY.read_text(encoding="utf-8")

    assert "signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)" in source
    assert "_install_stop_signal_handlers()" in source


def test_live_sample_preview_decode_runs_after_all_ranks_finish() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")
    common = (ROOT / "library" / "training" / "sample_preview_common.py").read_text(
        encoding="utf-8"
    )
    sample_body = _section(source, "def sample_images(", "def _sample_image_inference(")

    assert "finish_preview_batch(" in sample_body
    assert "accelerator.wait_for_everyone()" in common
    assert "decode(accelerator, args, vae, dit=dit, network=network)" in common
    assert common.index("accelerator.wait_for_everyone()") < common.index(
        "decode(accelerator, args, vae, dit=dit, network=network)"
    )
    assert common.index(
        "decode(accelerator, args, vae, dit=dit, network=network)"
    ) < common.index("accelerator.wait_for_everyone()", common.index("decode("))


def test_live_sample_preview_restores_block_swap_safely() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")

    assert "def _restore_dit_training_device" in source
    assert "move_to_device_except_swap_blocks" in source
    assert "_restore_dit_training_device(dit, dit_device)" in source


def test_training_sample_sampler_is_consumed_by_preview_loop() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")
    sample_fn = _section(source, "def do_sample(", "def sample_images(")
    inference_fn = _section(source, "def _sample_image_inference(", "def _module_device(")

    assert '_TRAINING_SAMPLE_SAMPLERS = {"euler", "er_sde", "lcm"}' in source
    assert "def normalize_training_sample_sampler" in source
    assert 'sampler: str = "euler"' in sample_fn
    assert "sampler = normalize_training_sample_sampler(sampler)" in sample_fn
    assert "inference_sampling.ERSDESampler" in sample_fn
    assert "inference_sampling.LCMSampler" in sample_fn
    assert "sampler_stepper.step(x, denoised, i)" in sample_fn
    assert 'getattr(args, "sample_sampler", "euler")' in inference_fn
    assert "sample_sampler," in inference_fn


def test_training_sample_prompts_cfg_and_dual_schedule_are_used() -> None:
    source = ANIMA_TRAINING.read_text(encoding="utf-8")
    sample_images_fn = _section(source, "def sample_images(", "def _sample_image_inference(")
    inference_fn = _section(source, "def _sample_image_inference(", "def _module_device(")

    assert 'scale = prompt_dict.get("guidance_scale", prompt_dict.get("scale", 7.5))' in inference_fn
    assert "sample_this_call = False" in sample_images_fn
    assert "if epoch is not None and args.sample_every_n_epochs is not None:" in sample_images_fn
    assert "sample_this_call = epoch % args.sample_every_n_epochs == 0" in sample_images_fn
    assert "if epoch is None and args.sample_every_n_steps is not None:" in sample_images_fn
    assert "sample_this_call = steps % args.sample_every_n_steps == 0" in sample_images_fn


def test_sample_prompts_without_schedule_do_not_trigger_qwen3_or_te_cache() -> None:
    train_source = TRAIN_PY.read_text(encoding="utf-8")
    cache_source = (ROOT / "library" / "training" / "text_encoder_cache.py").read_text(encoding="utf-8")
    session_source = (ROOT / "library" / "training" / "train_session.py").read_text(encoding="utf-8")
    bootstrap_source = (ROOT / "library" / "training" / "train_bootstrap.py").read_text(encoding="utf-8")

    cache_fn = cache_source[cache_source.index("def cache_text_encoder_outputs_if_needed(") :]
    encoder_decision = _section(
        session_source,
        "sampling_enabled = sample_preview_enabled(args)",
        "# Prepare accelerator",
    )

    assert "def sample_preview_enabled(args) -> bool:" in bootstrap_source
    assert "if text_encoders[0] is not None and sample_preview_enabled(args):" in cache_fn
    assert "sampling_enabled = sample_preview_enabled(args)" in encoder_decision
    assert "or sampling_enabled" in encoder_decision
    assert 'getattr(args, "sample_prompts", None)' not in encoder_decision
    assert "from library.training.text_encoder_cache import (" in train_source


def test_text_encoder_outputs_caching_strategy_receives_weight_dtype() -> None:
    train_source = TRAIN_PY.read_text(encoding="utf-8")
    session_source = (ROOT / "library" / "training" / "train_session.py").read_text(encoding="utf-8")
    strategy_fn = _section(
        train_source,
        "def get_text_encoder_outputs_caching_strategy(",
        "    def get_models_for_text_encoding",
    )
    encoder_decision = _section(
        session_source,
        "whether the Qwen3 text encoder needs loading at all.",
        "# Prepare accelerator",
    )

    assert "def get_text_encoder_outputs_caching_strategy(self, args, weight_dtype: torch.dtype):" in strategy_fn
    assert "cache_dtype=weight_dtype" in strategy_fn
    assert "weight_dtype, _save_dtype = prepare_dtype(args)" in encoder_decision
    assert "trainer.get_text_encoder_outputs_caching_strategy(args, weight_dtype)" in encoder_decision


def test_cached_sample_prompt_snapshot_survives_prompt_file_edits() -> None:
    train_source = TRAIN_PY.read_text(encoding="utf-8")
    cache_source = (ROOT / "library" / "training" / "text_encoder_cache.py").read_text(encoding="utf-8")
    anima_source = ANIMA_TRAINING.read_text(encoding="utf-8")
    cache_fn = cache_source[cache_source.index("def cache_text_encoder_outputs_if_needed(") :]
    sample_images_fn = _section(anima_source, "def sample_images(", "def _sample_image_inference(")

    assert "self.sample_prompts_snapshot = None" in train_source
    assert "trainer.sample_prompts_snapshot = [dict(prompt) for prompt in prompts]" in cache_fn
    assert "sample_prompts_snapshot=self.sample_prompts_snapshot" in train_source
    assert "sample_prompts_snapshot=None" in sample_images_fn
    assert "use_cached_prompt_snapshot = (" in sample_images_fn
    assert "and text_encoder is None" in sample_images_fn
    assert "prompts = [dict(prompt) for prompt in sample_prompts_snapshot]" in sample_images_fn
