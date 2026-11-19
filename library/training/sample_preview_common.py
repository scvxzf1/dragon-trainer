"""Shared scheduling and prompt-loading helpers for family training previews."""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

import torch

from library import train_util

logger = logging.getLogger(__name__)


def should_sample(args, epoch, steps: int) -> bool:
    if steps == 0:
        return bool(args.sample_at_first)
    if epoch is not None and args.sample_every_n_epochs is not None:
        return epoch % args.sample_every_n_epochs == 0
    if epoch is None and args.sample_every_n_steps is not None:
        return steps % args.sample_every_n_steps == 0
    return False


def failed_on_any_process(accelerator, failed: bool, num_processes: int) -> bool:
    if num_processes <= 1:
        return failed
    failure_count = accelerator.reduce(
        torch.tensor(
            [int(failed)],
            device=accelerator.device,
            dtype=torch.int32,
        ),
        reduction="sum",
    )
    return bool(failure_count.item())


def load_prompts(
    accelerator,
    args,
    *,
    text_encoder,
    sample_prompts_te_outputs,
    sample_prompts_snapshot,
    num_processes: int,
) -> list[dict[str, Any]]:
    use_cached_snapshot = (
        sample_prompts_snapshot is not None
        and sample_prompts_te_outputs is not None
        and text_encoder is None
    )
    prompt_file_missing = not use_cached_snapshot and not os.path.isfile(
        args.sample_prompts
    )
    if failed_on_any_process(accelerator, prompt_file_missing, num_processes):
        if getattr(accelerator, "is_main_process", True):
            logger.error(
                "Sample prompt file is unavailable on at least one process: %s",
                args.sample_prompts,
            )
        return []

    prompt_error: Optional[Exception] = None
    prompts: list[dict[str, Any]] = []
    try:
        if use_cached_snapshot:
            prompts = [dict(prompt) for prompt in sample_prompts_snapshot]
        else:
            prompts = train_util.load_prompts(args.sample_prompts)
    except Exception as exc:  # keep every rank on the same collective path
        prompt_error = exc
    if failed_on_any_process(accelerator, prompt_error is not None, num_processes):
        if prompt_error is not None:
            raise prompt_error
        raise RuntimeError("Sample prompt loading failed on another process")
    return prompts


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(exc).lower()
    )


def finish_preview_batch(
    accelerator,
    args,
    *,
    processes: int,
    run_local,
    failure_message: str,
    vae,
    dit=None,
    network=None,
    decode=None,
    failure_check=None,
) -> None:
    """Coordinate local preview, rank failure, decode, and barriers."""
    check_failure = failure_check or failed_on_any_process
    sample_error: Optional[Exception] = None
    try:
        result = run_local()
        if isinstance(result, Exception):
            sample_error = result
    except Exception as exc:
        sample_error = exc
    if sample_error is not None and is_cuda_oom(sample_error):
        from library.runtime.device import clean_memory_on_device

        clean_memory_on_device(accelerator.device)
    if check_failure(accelerator, sample_error is not None, processes):
        if sample_error is not None:
            raise sample_error
        raise RuntimeError(failure_message)

    if decode is None:
        from library.training.preview_live_decode import decode_live_preview_for_family

        def decode(accelerator, args, vae, *, dit=None, network=None):
            return decode_live_preview_for_family(
                accelerator,
                args,
                vae,
                dit=dit,
                network=network,
            )
    accelerator.wait_for_everyone()
    decode_error: Optional[Exception] = None
    try:
        if getattr(accelerator, "is_main_process", True):
            decode(accelerator, args, vae, dit=dit, network=network)
    except Exception as exc:
        decode_error = exc
    if check_failure(accelerator, decode_error is not None, processes):
        if decode_error is not None:
            raise decode_error
        raise RuntimeError("Training preview decode failed on another process")
    accelerator.wait_for_everyone()
