"""Small adapters between the shared trainer and MOSS component APIs."""

import torch


def load_text_encoder(args, weight_dtype):
    from .text_encoder_runtime import load_moss_text_encoder

    return load_moss_text_encoder(
        args.qwen3, dtype=torch.bfloat16,
        quantization=getattr(args, "moss_te_quantization", "nf4"),
    )


def load_vae(args, weight_dtype):
    from library.training.precision_policy import resolve_vae_dtype
    from .weights import load_moss_vae

    return load_moss_vae(args.vae, dtype=resolve_vae_dtype(args, weight_dtype))


def load_dit(trainer, args, weight_dtype, accelerator, text_encoders):
    from library.runtime.block_swap_controller import BlockSwapConfig
    from .block_swap import enable_moss_block_swap
    from .weights import load_moss_transformer

    swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    trainer.is_swapping_blocks = bool(swap)
    model = load_moss_transformer(
        args.pretrained_model_name_or_path, dtype=weight_dtype,
        device="cpu" if swap else accelerator.device,
    )
    model.set_attn_mode(getattr(args, "attn_mode", "torch"))
    if getattr(args, "gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
    if swap:
        enable_moss_block_swap(
            model,
            swap,
            accelerator.device,
            **BlockSwapConfig.from_args(args).to_offloader_kwargs(),
        )
    trainer._use_unsloth_offload_checkpointing = False
    return model, text_encoders


def encoding_profile(args):
    from library.training.sample_prompt_cache import _path_fingerprint
    from .cache_policy import resolve_tokenizer_path, text_cache_metadata

    return dict(
        max_length=int(getattr(args, "max_token_length", None) or 4096),
        hidden_layer="last",
        tokenizer=_path_fingerprint(resolve_tokenizer_path(args.qwen3)),
        **text_cache_metadata(getattr(args, "moss_te_quantization", "nf4")),
    )


def reject_inference(*args, **kwargs):
    raise RuntimeError("MOSS-Image v1 supports training previews; standalone inference is not registered")


def reject_pipeline_parallel(*args, **kwargs):
    raise NotImplementedError("MOSS-Image pipeline parallel is not implemented")
