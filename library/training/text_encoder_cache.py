"""Text-encoder output caching helpers extracted from AnimaTrainer.

Preserves live-encoding vs disk-cache behavior and sample-prompt TE caching.
"""

from __future__ import annotations

import logging

import torch
from accelerate import Accelerator

from library import train_util
from library.anima import text_strategies
from library.datasets import DatasetGroup
from library.runtime.device import clean_memory_on_device
from library.training.sample_prompt_cache import save_sample_prompt_cache
from library.training.train_bootstrap import sample_preview_enabled
from library.strategies.text_cache import text_encoder_cache_session

logger = logging.getLogger(__name__)


def cache_text_encoder_outputs_if_needed(
    trainer,
    args,
    accelerator: Accelerator,
    text_encoders,
    dataset: DatasetGroup,
):
    if not args.cache_text_encoder_outputs:
        # Live-encoding mode (e.g. IP-Adapter cache_text_encoder_outputs=false):
        # move the text encoder to device for per-step encoding.
        text_encoders[0].to(accelerator.device)
        return

    # With caching on, the on-disk cache is guaranteed complete (asserted in
    # train(), including the LLM adapter's crossattn_emb outputs, which
    # preprocess writes). The dataset thus never needs encoding here — run
    # the pass with no model purely to populate
    # ImageInfo.text_encoder_outputs_npz (forms no batches).
    dataset.new_cache_text_encoder_outputs([None], accelerator)

    # The text encoder is in memory only to encode sample prompts when
    # sampling will actually run. It is None when no preview needs it.
    preview_cache_ready = bool(
        trainer.sample_prompts_snapshot is not None
        and trainer.sample_prompts_te_outputs is not None
    )
    if preview_cache_ready and sample_preview_enabled(args):
        logger.info("using persistent Text Encoder outputs for sample prompts")
    elif text_encoders[0] is not None and sample_preview_enabled(args):
        logger.info(
            f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}"
        )
        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

        from library.env import resolve_model_family

        if resolve_model_family(args) == "qwen_image_2_1":
            from library.models.qwen_image_2_1.preview_conditions import cache_preview_text
            from library.models.qwen_image_2_1.text_encoder_runtime import loaded_text_encoder_for_cache
            from library.training.sample_preview_common import failed_on_any_process

            error = None
            try:
                policy = getattr(args, "qwen_text_encoder_cache_policy", "auto")
                with loaded_text_encoder_for_cache(
                    text_encoders[0],
                    dtype=next(text_encoders[0].parameters()).dtype,
                    device="cpu" if policy == "cpu" else str(accelerator.device),
                    cache_policy=policy,
                ) as model:
                    with accelerator.autocast(), torch.no_grad():
                        cache_preview_text(trainer, args, model, tokenize_strategy, text_encoding_strategy)
            except Exception as exc:
                error = exc
            finally:
                clean_memory_on_device(accelerator.device)
            if failed_on_any_process(accelerator, error is not None, getattr(accelerator, "num_processes", 1)):
                raise RuntimeError("Qwen preview text caching failed on at least one process") from error
            accelerator.wait_for_everyone()
            return

        logger.info("move text encoder to gpu")
        with text_encoder_cache_session(text_encoders[0], accelerator.device):
            prompts = train_util.load_prompts(args.sample_prompts)
            trainer.sample_prompts_snapshot = [dict(prompt) for prompt in prompts]
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [
                        prompt_dict.get("prompt", ""),
                        prompt_dict.get("negative_prompt", ""),
                    ]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"  cache TE outputs for: {p}")
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            encoded = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                text_encoders,
                                tokens_and_masks,
                            )
                            sample_prompts_te_outputs[p] = tuple(
                                torch.as_tensor(value).detach().to("cpu").contiguous()
                                for value in encoded
                            )
            trainer.sample_prompts_te_outputs = sample_prompts_te_outputs

            if accelerator.is_main_process:
                try:
                    save_sample_prompt_cache(
                        args,
                        trainer.sample_prompts_snapshot,
                        sample_prompts_te_outputs,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning("Could not persist sample-prompt TE cache: %s", exc)

        clean_memory_on_device(accelerator.device)

    accelerator.wait_for_everyone()
