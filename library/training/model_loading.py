"""Model loading helpers extracted from AnimaTrainer.

Keeps DiT / TE / VAE load order and freeze-merge behavior identical to train.py.
Family dispatch (stage 6): ``resolve_model_family(args)`` selects Anima
(Qwen3+T5 cross-attn DiT), Krea-2-Raw (Qwen3-VL single-stream MMDiT), or
Z-Image (Diffusers flow transformer). Adaptive precision islands are installed
in each family loader before adapter and offloader setup.
"""

from __future__ import annotations

import torch
from torch import nn

from library.anima import weights as anima_utils
from library.env import resolve_model_family
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.models.family_registry import dispatch_model_family, get_model_family_spec
from library.quantization.training import validate_mixed_nf4_training
from library.training.probes import maybe_probe_components as _maybe_probe_components
from library.training.train_bootstrap import resolve_block_swap_profile_jsonl
from library.training.v100_flash import (
    flash_attn_v100_doc,
    resolve_debug_finite_checks,
    resolve_v100_flash_stability,
)
import logging

logger = logging.getLogger(__name__)


def _adaptive_islands_enabled(args) -> bool:
    from library.training.adaptive_runtime.training_config import islands_enabled

    return islands_enabled(args)


def _install_adaptive_precision(model, args, accelerator, family: str, *, trainer=None) -> None:
    """Install frozen precision islands before adapter/offloader setup."""
    if not _adaptive_islands_enabled(args):
        from library.training.adaptive_runtime.native_runtime import NativeStartupRuntime
        from library.training.adaptive_runtime.native_precision import native_enabled

        runtime = getattr(trainer, "_adaptive_training_runtime", None)
        if native_enabled(args) and family == "krea2_raw":
            from library.training.adaptive_runtime.native_family import install_guards
            install_guards(model, family)
        if native_enabled(args) and type(runtime) is NativeStartupRuntime:
            runtime.capture_model(model)
            runtime.validate_swap_capacity(model, family)
            trainer._native_precision_model = model
        return
    from library.training.adaptive_runtime.training_config import require_training_contract
    from library.training.adaptive_runtime.training_precision import (
        install_training_precision,
        register_precision_checkpoint,
    )

    require_training_contract(args, world_size=accelerator.num_processes)
    manifest = install_training_precision(model, args, model_family=family)
    register_precision_checkpoint(accelerator, manifest)
    if trainer is not None:
        runtime = getattr(trainer, "_adaptive_training_runtime", None)
        if runtime is not None:
            runtime.validate_swap_capacity(model, family)
    # Keep the original base model visible to the adaptive runtime.  The
    # runtime re-hashes these tensors after setup and after accelerator
    # preparation so an offloader dtype drift cannot hide behind the request ID.
    if trainer is not None:
        trainer._adaptive_precision_manifest_model = model
    logger.info(
        "Adaptive precision islands installed: family=%s linear_units=%d",
        family,
        len(manifest["assignments"]),
    )


def _load_anima_text_encoder(args, weight_dtype):
    logger.info("Loading Qwen3 text encoder...")
    model, _ = anima_utils.load_qwen3_text_encoder(
        args.qwen3, dtype=weight_dtype, device="cpu"
    )
    return model


def _load_krea2_text_encoder(args, weight_dtype):
    from library.models.krea2_raw.strategy import load_krea2_text_encoder

    logger.info("Loading Krea-2 Qwen3-VL text encoder...")
    model, _ = load_krea2_text_encoder(
        args.qwen3, dtype=weight_dtype, device="cpu"
    )
    return model


def _load_z_image_text_encoder(args, weight_dtype):
    from library.models.z_image.weights import load_z_image_text_encoder

    logger.info("Loading Z-Image Qwen3 text encoder...")
    return load_z_image_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")


def _load_moss_text_encoder(*args, **kwargs):
    from library.models.moss_image.integration import load_text_encoder

    return load_text_encoder(*args, **kwargs)


def _load_qwen_image_2_1_text_encoder(args, weight_dtype):
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_text_encoder

    return load_qwen_image_2_1_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")


def _load_qwen_vae(args, weight_dtype):
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
    )
    vae.to(weight_dtype)
    vae.eval()
    return vae


def _load_z_image_vae(args, weight_dtype):
    from library.models.z_image.weights import load_z_image_vae

    return load_z_image_vae(args.vae, dtype=weight_dtype, device="cpu")


def _load_moss_vae(*args, **kwargs):
    from library.models.moss_image.integration import load_vae

    return load_vae(*args, **kwargs)


def _load_qwen_image_2_1_vae(args, weight_dtype):
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_vae

    return load_qwen_image_2_1_vae(args.vae, dtype=weight_dtype, device="cpu")


def load_target_model(trainer, args, weight_dtype, accelerator, load_qwen3=True, load_vae=True):
    family = resolve_model_family(args)
    trainer.is_swapping_blocks = (
        args.blocks_to_swap is not None and args.blocks_to_swap > 0
    )

    # Load text encoder. Anima uses Qwen3 (+T5 adapter downstream); Krea-2 uses
    # Qwen3-VL-4B directly (hidden_dim=2560=DiT txtdim, no adapter). Skipped
    # when every text-encoder output is already cached and no live encoding
    # (sampling / TE training / cache disabled) needs it.
    if load_qwen3:
        loader = dispatch_model_family(
            family,
            operation="training text-encoder loading",
            handlers={
                "anima": _load_anima_text_encoder,
                "krea2_raw": _load_krea2_text_encoder,
                "z_image": _load_z_image_text_encoder,
                "qwen_image_2_1": _load_qwen_image_2_1_text_encoder,
                "moss_image": _load_moss_text_encoder,
            },
        )
        qwen3_text_encoder = loader(args, weight_dtype)
        qwen3_text_encoder.eval()
    else:
        logger.info(
            "Skipping text encoder load: all text-encoder outputs cached."
        )
        qwen3_text_encoder = None

    # Load VAE. Shared across families (same AutoencoderKLQwenImage).
    if load_vae:
        spec = get_model_family_spec(family)
        logger.info("Loading %s VAE...", spec.display_name)
        loader = dispatch_model_family(
            family,
            operation="training VAE loading",
            handlers={
                "anima": _load_qwen_vae,
                "krea2_raw": _load_qwen_vae,
                "z_image": _load_z_image_vae,
                "qwen_image_2_1": _load_qwen_image_2_1_vae,
                "moss_image": _load_moss_vae,
            },
        )
        vae = loader(args, weight_dtype)
    else:
        logger.info("Skipping VAE load: all latents cached and no sampling.")
        vae = None

    # Return format: (model_type, text_encoders, vae, unet). model_type is the
    # family name stamped into training metadata (anima | krea2_raw).
    return family, [qwen3_text_encoder], vae, None  # unet loaded lazily


def load_unet_lazily(trainer, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
    family = resolve_model_family(args)
    loader = dispatch_model_family(
        family,
        operation="training DiT loading",
        handlers={
            "anima": _load_anima_dit,
            "krea2_raw": _load_krea2_dit,
            "z_image": _load_z_image_dit,
            "qwen_image_2_1": _load_qwen_image_2_1_dit,
            "moss_image": _load_moss_dit,
        },
    )
    return loader(trainer, args, weight_dtype, accelerator, text_encoders)


def _load_moss_dit(*args, **kwargs):
    from library.models.moss_image.integration import load_dit

    return load_dit(*args, **kwargs)


def _load_qwen_image_2_1_dit(trainer, args, weight_dtype, accelerator, text_encoders):
    from library.models.qwen_image_2_1.attention_backend import prepare_qwen_image_2_1_attention
    from library.models.qwen_image_2_1.block_swap import enable_qwen_image_2_1_block_swap
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer

    trainer.is_swapping_blocks = bool(getattr(args, "blocks_to_swap", 0) or 0)
    from library.models.qwen_image_2_1.preview_memory import needs_preview_offloader

    preview_offloader = needs_preview_offloader(args)
    loading_device = "cpu" if trainer.is_swapping_blocks else accelerator.device
    model = load_qwen_image_2_1_transformer(
        args.pretrained_model_name_or_path, dtype=weight_dtype, device=loading_device
    )
    validate_mixed_nf4_training(model, args)
    attn_mode = prepare_qwen_image_2_1_attention(
        model,
        getattr(args, "attn_mode", None),
        dtype=weight_dtype,
    )
    logger.info(
        "Using Qwen Image 2.1 attention mode: %s%s",
        attn_mode,
        " (FlashAttention varlen for image segments; native SDPA for causal text segments)"
        if attn_mode == "flash" else "",
    )
    if getattr(args, "gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
    if trainer.is_swapping_blocks or preview_offloader:
        enable_qwen_image_2_1_block_swap(
            model,
            getattr(args, "blocks_to_swap", 0) or 0,
            accelerator.device,
            profile_jsonl=resolve_block_swap_profile_jsonl(args),
            transfer_dtype=getattr(args, "block_swap_transfer_dtype", None),
            restore_mode=getattr(args, "block_swap_restore_mode", None),
            int8_restore_mode=getattr(args, "block_swap_int8_restore_mode", None),
            int8_scope=getattr(args, "block_swap_int8_scope", None),
            int8_staging_ring=getattr(args, "block_swap_int8_staging_ring", None),
            int8_dual_stream=getattr(args, "block_swap_int8_dual_stream", None),
            int8_restore_chunk_rows=getattr(
                args, "block_swap_int8_restore_chunk_rows", None
            ),
        )
    trainer._use_unsloth_offload_checkpointing = False
    return model, text_encoders


def _load_z_image_dit(trainer, args, weight_dtype, accelerator, text_encoders):
    from library.models.z_image.attention_backend import prepare_z_image_attention
    from library.models.z_image.block_swap import enable_z_image_block_swap
    from library.models.z_image.weights import load_z_image_transformer

    trainer.is_swapping_blocks = bool(
        args.blocks_to_swap is not None and args.blocks_to_swap > 0
    )
    loading_device = "cpu" if trainer.is_swapping_blocks else accelerator.device
    logger.info(
        "Loading Z-Image transformer from %s...",
        args.pretrained_model_name_or_path,
    )
    model = load_z_image_transformer(
        args.pretrained_model_name_or_path,
        dtype=weight_dtype,
        device=loading_device,
    )
    validate_mixed_nf4_training(model, args)
    _install_adaptive_precision(model, args, accelerator, "z_image", trainer=trainer)
    attn_mode = prepare_z_image_attention(
        model,
        getattr(args, "attn_mode", None),
        dtype=weight_dtype,
    )
    logger.info(
        "Using Z-Image attention mode: %s%s",
        attn_mode,
        " (Diffusers flash_varlen)" if attn_mode == "flash" else "",
    )
    if getattr(args, "gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
    trainer._use_unsloth_offload_checkpointing = False
    if trainer.is_swapping_blocks:
        profile_jsonl = resolve_block_swap_profile_jsonl(args)
        enable_z_image_block_swap(
            model,
            args.blocks_to_swap,
            accelerator.device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=args.block_swap_transfer_dtype,
            restore_mode=args.block_swap_restore_mode,
            int8_restore_mode=getattr(args, "block_swap_int8_restore_mode", None),
            int8_scope=getattr(args, "block_swap_int8_scope", None),
            int8_staging_ring=getattr(args, "block_swap_int8_staging_ring", None),
            int8_dual_stream=getattr(args, "block_swap_int8_dual_stream", None),
            int8_restore_chunk_rows=getattr(
                args, "block_swap_int8_restore_chunk_rows", None
            ),
        )
        _maybe_probe_components(
            trainer,
            "block_swap_enabled",
            unet=model,
            device=accelerator.device,
            phase="setup",
            blocks_to_swap=args.blocks_to_swap,
            block_swap_profile_jsonl=profile_jsonl or "off",
            block_swap_transfer_dtype=args.block_swap_transfer_dtype,
            block_swap_restore_mode=args.block_swap_restore_mode,
        )
    _maybe_probe_components(
        trainer,
        "dit_loaded",
        unet=model,
        device=accelerator.device,
        phase="setup",
        loading_device=loading_device,
        loading_dtype=weight_dtype,
        attn_mode=attn_mode,
    )
    return model, text_encoders


def _load_krea2_dit(trainer, args, weight_dtype, accelerator, text_encoders):
    """Krea-2 single-stream MMDiT loader (stage 6).

    Reuses the anima block-swap / probe / selective-checkpoint / VR-loss
    plumbing where the SingleStreamDiT exposes the same hooks as Anima
    (enable_block_swap / enable_selective_checkpointing / enable_peak_probe —
    added in stage 6 block-swap probe). ``attn_mode=torch`` selects the original
    cuDNN SDPA path; ``attn_mode=flash`` selects packed FlashAttention varlen.
    """
    from library.models.krea2_raw.attention_backend import (
        prepare_krea2_attention,
        validate_krea2_attention_mode,
    )
    from library.models.krea2_raw.weights import load_krea2_dit
    from library.training.adaptive_runtime.training_config import islands_enabled

    from library.training.adaptive_runtime.native_precision import native_enabled
    adaptive = islands_enabled(args)
    if adaptive or native_enabled(args):
        from library.models.krea2_raw.quantize import inspect_nf4_checkpoint
        if inspect_nf4_checkpoint(str(args.pretrained_model_name_or_path)).is_nf4:
            raise ValueError("FP16/FP32 training requires nonquantized base weights, not NF4")
    loading_dtype = torch.bfloat16 if adaptive else weight_dtype
    # PP must slice on CPU: the stage cut keeps 13/15 of the 28 blocks, so the
    # full 25 GB model cannot be materialised on a single 16 GB card.  Loading
    # straight to the accelerator OOM'd inside .to() before the PP dispatch
    # ever ran (measured: 15.18 GiB allocated, 192 MiB short).
    from library.training.pp.gate import pipeline_parallel_enabled

    loading_device = (
        "cpu"
        if (trainer.is_swapping_blocks or pipeline_parallel_enabled(args))
        else accelerator.device
    )
    # NF4 (Krea-2 QLoRA): compat_matrix 对 nf4×block_swap 已降级 warning (方向 A
    # 端到端探针验证通过). NF4 v1 通过 nf4_prequantized_path 覆盖 BF16;
    # 自包含 v2 可用同一参数，也可直接作为 pretrained_model_name_or_path，
    # 两者都跳过在线量化。未给磁盘 NF4 时才回退 quantize_dit_to_nf4。
    nf4_active = (
        str(getattr(args, "base_compute", "bf16") or "bf16").strip().lower()
        == "nf4"
    )
    nf4_path = getattr(args, "nf4_prequantized_path", None)
    requested_attn_mode = validate_krea2_attention_mode(
        getattr(args, "attn_mode", None),
        dtype=loading_dtype,
        compile_enabled=bool(getattr(args, "torch_compile", False)),
    )
    if nf4_active and nf4_path:
        import os as _os

        if not _os.path.exists(str(nf4_path)):
            raise FileNotFoundError(
                f"--nf4_prequantized_path 不存在: {nf4_path}. "
                f"先用 scripts/krea2/probe_nf4_save.py 生成, 或去掉该参数走在线量化."
            )
        logger.info(f"  [NF4 磁盘加载] {nf4_path} (跳过在线量化)")
    logger.info(
        f"Loading Krea-2 DiT (family=krea2_raw) from "
        f"{args.pretrained_model_name_or_path}..."
        + (" [NF4]" if nf4_active else "")
    )
    model = load_krea2_dit(
        args.pretrained_model_name_or_path,
        device=loading_device,
        dtype=loading_dtype,
        eval=False,
        nf4=nf4_active,
        nf4_path=nf4_path if nf4_active else None,
    )
    validate_mixed_nf4_training(model, args)
    _install_adaptive_precision(model, args, accelerator, "krea2_raw", trainer=trainer)
    attn_mode = prepare_krea2_attention(
        model,
        requested_attn_mode,
        dtype=weight_dtype if adaptive else loading_dtype,
        compile_enabled=bool(getattr(args, "torch_compile", False)),
    )
    logger.info("Krea-2 attention mode: %s", attn_mode)
    _maybe_probe_components(
        trainer,
        "dit_loaded",
        unet=model,
        device=accelerator.device,
        phase="setup",
        loading_device=loading_device,
        loading_dtype=loading_dtype,
        attn_mode=attn_mode,
    )
    if trainer.peak_probe is not None and hasattr(model, "enable_peak_probe"):
        model.enable_peak_probe(trainer.peak_probe)
        trainer.peak_probe.write(
            {
                "ev": "peak_probe_config",
                "label": "dit_peak_probe_attached",
                "phase": "setup",
                "block_count": len(getattr(model, "blocks", []) or []),
            }
        )

    trainer._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing
    selective_checkpoint = getattr(args, "selective_checkpoint", "off") or "off"
    if selective_checkpoint != "off":
        selective_checkpoint_blocks = getattr(
            args, "selective_checkpoint_blocks", ""
        )
        logger.info(
            "enable selective checkpoint: "
            f"{selective_checkpoint}"
            + (
                f" blocks={selective_checkpoint_blocks or 'auto'}"
                if str(selective_checkpoint).startswith("peak_blocks_")
                else ""
            )
        )
        model.enable_selective_checkpointing(
            selective_checkpoint,
            blocks=selective_checkpoint_blocks,
        )

    # Block swap (reuses anima ModelOffloader; SingleStreamBlock forward is
    # transparent to the offloader — stage 6 block-swap probe verified).
    trainer.is_swapping_blocks = (
        args.blocks_to_swap is not None and args.blocks_to_swap > 0
    )
    if trainer.is_swapping_blocks:
        profile_jsonl = resolve_block_swap_profile_jsonl(args)
        logger.info(
            "enable block swap: "
            f"blocks_to_swap={args.blocks_to_swap}, "
            f"transfer_dtype={args.block_swap_transfer_dtype}, "
            f"restore_mode={args.block_swap_restore_mode}, "
            f"profile_jsonl={profile_jsonl or 'off'}"
        )
        model.enable_block_swap(
            args.blocks_to_swap,
            accelerator.device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=args.block_swap_transfer_dtype,
            restore_mode=args.block_swap_restore_mode,
            int8_restore_mode=getattr(args, "block_swap_int8_restore_mode", None),
            int8_scope=getattr(args, "block_swap_int8_scope", None),
            int8_staging_ring=getattr(args, "block_swap_int8_staging_ring", None),
            int8_dual_stream=getattr(args, "block_swap_int8_dual_stream", None),
            int8_restore_chunk_rows=getattr(
                args, "block_swap_int8_restore_chunk_rows", None
            ),
        )
        _maybe_probe_components(
            trainer,
            "block_swap_enabled",
            unet=model,
            device=accelerator.device,
            phase="setup",
            blocks_to_swap=args.blocks_to_swap,
            block_swap_profile_jsonl=profile_jsonl or "off",
            block_swap_transfer_dtype=args.block_swap_transfer_dtype,
            block_swap_restore_mode=args.block_swap_restore_mode,
        )

    if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
        logger.info(
            f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
            f"using trainable DiT with multiplier=0 as the control variate"
        )

    return model, text_encoders


def _load_anima_dit(trainer, args, weight_dtype, accelerator, text_encoders):
    from library.anima.checkpoint import (
        anima_checkpoint_sha256,
        apply_layout_to_args,
        inspect_anima_checkpoint,
    )
    from library.anima.compat import (
        ANIMA29_PREVIEW_V1_SHA256,
        require_training_compatibility,
    )

    loading_dtype = weight_dtype
    loading_device = "cpu" if trainer.is_swapping_blocks else accelerator.device

    attn_mode = "flash"
    if args.xformers:
        attn_mode = "xformers"
    if args.attn_mode is not None:
        attn_mode = args.attn_mode

    v100_flash_stability = resolve_v100_flash_stability(args)
    debug_finite_checks = resolve_debug_finite_checks(
        args, v100_flash_stability
    )

    if attn_mode == "flash4":
        # Flash Attention 4 (flash-attention-sm120) is not supported yet.
        raise RuntimeError(
            "attn_mode='flash4' is not supported yet -- the flash-attention-sm120 "
            "kernel is disabled in this build. Use 'flash', 'torch', 'flex', "
            "'sageattn', or 'xformers' instead."
        )
    elif attn_mode == "flash":
        from networks.attention_dispatch import (
            flash_attn,
            flash_attn_func,
            flash_attn_v100_provider,
        )

        if flash_attn_func is None:
            raise RuntimeError(
                "attn_mode='flash' requested but flash_attn is not available."
            )
        flash_doc, doc_reports_v100 = flash_attn_v100_doc(flash_attn)
        is_v100_fork = flash_attn_v100_provider or doc_reports_v100
        try:
            major, minor = torch.cuda.get_device_capability(accelerator.device)
        except Exception:
            major, minor = -1, -1
        logger.info(
            "Using Flash Attention 2 (flash_attn %s), gpu_sm=%s.%s, "
            "v100_fork=%s, v100_flash_stability=%s, debug_finite_checks=%s%s",
            getattr(flash_attn, "__version__", "unknown"),
            major,
            minor,
            is_v100_fork,
            v100_flash_stability,
            debug_finite_checks,
            f", doc={flash_doc}" if flash_doc else "",
        )
        if major == 7 and minor == 0 and is_v100_fork:
            if loading_dtype != torch.float16:
                raise RuntimeError(
                    "flash-attention-v100 only supports FP16 inputs; use "
                    "mixed_precision='fp16' or attn_mode='torch'."
                )
            logger.warning(
                "Detected flash-attention-v100 on Volta/V100. Its strict "
                "production validation is not approved; use attn_mode='torch' "
                "for production and Flash only for diagnostics."
            )
    else:
        logger.info(f"Using attention mode: {attn_mode}")

    # Frozen LoRA: merged into DiT weights at load time (no runtime hooks).
    # Used by postfix runs that train on top of a fixed LoRA.
    lora_weights_list = None
    lora_multipliers = None
    if getattr(args, "lora_path", None):
        from safetensors.torch import load_file

        logger.info(
            f"merging frozen LoRA from {args.lora_path} into DiT weights "
            f"(multiplier={args.lora_multiplier})"
        )
        lora_sd = load_file(args.lora_path)
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
        lora_weights_list = [lora_sd]
        lora_multipliers = [args.lora_multiplier]

    layout = inspect_anima_checkpoint(args.pretrained_model_name_or_path)
    base_sha256 = None
    if layout.num_blocks == 40:
        base_sha256 = anima_checkpoint_sha256(args.pretrained_model_name_or_path)
        if base_sha256.lower() != ANIMA29_PREVIEW_V1_SHA256:
            raise ValueError(
                "40-block Anima checkpoint is structurally valid but not the "
                "certified Anima-2.9B-preview-v1 base"
            )
    apply_layout_to_args(args, layout, base_sha256)
    profile = require_training_compatibility(args, layout)
    logger.info(
        "Detected Anima checkpoint: variant=%s, blocks=%s, profile=%s",
        layout.variant,
        layout.num_blocks,
        profile,
    )

    # Load DiT
    attn_softmax_scale = getattr(args, "attn_softmax_scale", None)
    logger.info(
        f"Loading Anima DiT model with attn_softmax_scale: {attn_softmax_scale}..."
    )
    model = anima_utils.load_anima_model(
        accelerator.device,
        args.pretrained_model_name_or_path,
        attn_mode,
        loading_device,
        loading_dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        attn_softmax_scale=attn_softmax_scale,
        v100_flash_stability=v100_flash_stability,
        debug_finite_checks=debug_finite_checks,
        checkpoint_layout=layout,
        anima_base_sha256=base_sha256,
    )
    _install_adaptive_precision(model, args, accelerator, "anima", trainer=trainer)
    _maybe_probe_components(
        trainer,
        "dit_loaded",
        unet=model,
        device=accelerator.device,
        phase="setup",
        attn_mode=attn_mode,
        loading_device=loading_device,
        loading_dtype=loading_dtype,
    )
    if trainer.peak_probe is not None and hasattr(model, "enable_peak_probe"):
        model.enable_peak_probe(trainer.peak_probe)
        trainer.peak_probe.write(
            {
                "ev": "peak_probe_config",
                "label": "dit_peak_probe_attached",
                "phase": "setup",
                "block_count": len(getattr(model, "blocks", []) or []),
            }
        )

    # Store unsloth preference so that when the base trainer calls
    # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
    trainer._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

    selective_checkpoint = getattr(args, "selective_checkpoint", "off") or "off"
    if selective_checkpoint != "off":
        selective_checkpoint_blocks = getattr(
            args, "selective_checkpoint_blocks", ""
        )
        logger.info(
            "enable selective checkpoint: "
            f"{selective_checkpoint}"
            + (
                f" blocks={selective_checkpoint_blocks or 'auto'}"
                if str(selective_checkpoint).startswith("peak_blocks_")
                else ""
            )
        )
        model.enable_selective_checkpointing(
            selective_checkpoint,
            blocks=selective_checkpoint_blocks,
        )

    # torch.compile is intentionally delayed until after
    # network.apply_to/load_weights and gradient-checkpoint setup. Otherwise
    # Dynamo can trace the bare DiT path instead of the adapter-patched
    # Linear forwards, and checkpoint recompute may not match the original
    # forward graph. See library.runtime.harness.compile_blocks_for_training.

    # Block swap
    trainer.is_swapping_blocks = (
        args.blocks_to_swap is not None and args.blocks_to_swap > 0
    )
    if trainer.is_swapping_blocks:
        profile_jsonl = resolve_block_swap_profile_jsonl(args)
        logger.info(
            "enable block swap: "
            f"blocks_to_swap={args.blocks_to_swap}, "
            f"transfer_dtype={args.block_swap_transfer_dtype}, "
            f"restore_mode={args.block_swap_restore_mode}, "
            f"profile_jsonl={profile_jsonl or 'off'}"
        )
        model.enable_block_swap(
            args.blocks_to_swap,
            accelerator.device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=args.block_swap_transfer_dtype,
            restore_mode=args.block_swap_restore_mode,
            int8_restore_mode=getattr(args, "block_swap_int8_restore_mode", None),
            int8_scope=getattr(args, "block_swap_int8_scope", None),
            int8_staging_ring=getattr(args, "block_swap_int8_staging_ring", None),
            int8_dual_stream=getattr(args, "block_swap_int8_dual_stream", None),
            int8_restore_chunk_rows=getattr(
                args, "block_swap_int8_restore_chunk_rows", None
            ),
        )
        _maybe_probe_components(
            trainer,
            "block_swap_enabled",
            unet=model,
            device=accelerator.device,
            phase="setup",
            blocks_to_swap=args.blocks_to_swap,
            block_swap_profile_jsonl=profile_jsonl or "off",
            block_swap_transfer_dtype=args.block_swap_transfer_dtype,
            block_swap_restore_mode=args.block_swap_restore_mode,
        )

    # Variance-reduced FM loss: the "frozen reference" is the trainable
    # DiT itself with ``network.set_multiplier(0)`` during the no-grad
    # forward — works because base weights are frozen and LoRA-family
    # adapters are additive. See ``get_noise_pred_and_target`` for the
    # bypass. Saves ~5 GB VRAM vs holding a second DiT copy.
    if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
        logger.info(
            f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
            f"using trainable DiT with multiplier=0 as the control variate"
        )

    return model, text_encoders
