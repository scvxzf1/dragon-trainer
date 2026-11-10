"""Validate and mutate training CLI args before dataset/model setup."""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

from library.datasets import DatasetGroup, MinimalDataset
from library.env import resolve_model_family
from library.runtime.offloading import (
    normalize_block_swap_restore_mode,
    normalize_block_swap_transfer_dtype,
)
from library.training.compat_matrix import (
    apply_training_compat_mutations,
    check_training_compat,
)

logger = logging.getLogger(__name__)


def assert_training_extra_args(
    args,
    train_dataset_group: Union[DatasetGroup, MinimalDataset],
    val_dataset_group: Optional[DatasetGroup],
):
    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning(
            "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled"
        )
        args.cache_text_encoder_outputs = True

    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True
        ), (
            "when caching Text Encoder output, token_warmup_step or caption_tag_dropout_rate cannot be used"
        )
        if getattr(args, "cache_llm_adapter_outputs", False):
            # Adapter output caching is only valid when the adapter is frozen (no LoRA on adapter).
            if args.network_args is not None and any(
                "train_llm_adapter" in a and "true" in a.lower()
                for a in args.network_args
            ):
                raise ValueError(
                    "--cache_llm_adapter_outputs is incompatible with --network_args train_llm_adapter=True"
                )
    else:
        assert not getattr(args, "cache_llm_adapter_outputs", False), (
            "--cache_llm_adapter_outputs requires --cache_text_encoder_outputs"
        )

    assert args.network_train_unet_only or not args.cache_text_encoder_outputs, (
        "network for Text Encoder cannot be trained with caching Text Encoder outputs"
    )

    args.selective_checkpoint = (
        str(getattr(args, "selective_checkpoint", "off") or "off").strip().lower()
    )
    args.selective_checkpoint_blocks = str(
        getattr(args, "selective_checkpoint_blocks", "") or ""
    ).strip()
    args.block_swap_transfer_dtype = normalize_block_swap_transfer_dtype(
        getattr(args, "block_swap_transfer_dtype", "bf16")
    )
    args.block_swap_restore_mode = normalize_block_swap_restore_mode(
        getattr(args, "block_swap_restore_mode", "slab")
    )

    # Materialize the runtime family before entering the pure compatibility
    # matrix so env fallback and explicit CLI/TOML selection share one contract.
    args.model_family = resolve_model_family(args)
    if args.model_family == "z_image":
        rates = [
            float(getattr(subset, "caption_dropout_rate", 0.0) or 0.0)
            for group in (train_dataset_group, val_dataset_group)
            if group is not None
            for dataset in getattr(group, "datasets", ())
            for subset in getattr(dataset, "subsets", ())
        ]
        if any(rate > 0 for rate in rates):
            raise ValueError(
                "Z-Image caption dropout is not supported in v1; set every "
                "dataset subset caption_dropout_rate=0"
            )
    compat = check_training_compat(args)
    warning_codes = {warning.code for warning in compat.warnings}
    for warning in compat.warnings:
        logger.warning(warning.message)
    for mutation in compat.mutations:
        if mutation.code not in warning_codes:
            logger.warning(mutation.message)
    apply_training_compat_mutations(args, compat)
    if compat.errors:
        raise ValueError("\n".join(issue.message for issue in compat.errors))

    # Propagate inversion_dir to datasets for functional-loss supervision (postfix-func).
    inversion_dir = getattr(args, "inversion_dir", None)
    if inversion_dir:
        num_runs = getattr(args, "functional_loss_num_runs", 3)
        for dataset in train_dataset_group.datasets:
            dataset.inversion_dir = inversion_dir
            dataset.inversion_num_runs = num_runs
        if val_dataset_group is not None:
            for dataset in val_dataset_group.datasets:
                dataset.inversion_dir = inversion_dir
                dataset.inversion_num_runs = num_runs

    # BYG consumes pre-built per-image edit tuples. Filter out images with
    # no tuple before dataloader construction so every sampled batch has a
    # complete BYG conditioning surface.
    if getattr(args, "use_byg", False):
        byg_text_dir = getattr(args, "byg_text_dir", None) or os.path.join(
            "post_image_dataset", "byg"
        )
        for dataset in train_dataset_group.datasets:
            dataset.byg_text_dir = byg_text_dir
            kept, dropped = dataset.restrict_to_byg_tuples()
            if dropped:
                logger.info(
                    f"BYG: kept {kept} images with edit-tuple sidecars, "
                    f"dropped {dropped} without BYG tuple."
                )
        train_dataset_group.refresh_concat_state()
        if val_dataset_group is not None:
            for dataset in val_dataset_group.datasets:
                dataset.byg_text_dir = byg_text_dir
                dataset.restrict_to_byg_tuples()
            val_dataset_group.refresh_concat_state()

    # Propagate IP-Adapter feature-cache flag so datasets load
    # {stem}_anima_{encoder}.safetensors sidecars into batch["ip_features"].
    if getattr(args, "ip_features_cache_to_disk", False):
        ip_encoder = getattr(args, "ip_encoder", "pe")
        for dataset in train_dataset_group.datasets:
            dataset.ip_features_cache_to_disk = True
            dataset.ip_features_encoder = ip_encoder
        if val_dataset_group is not None:
            for dataset in val_dataset_group.datasets:
                dataset.ip_features_cache_to_disk = True
                dataset.ip_features_encoder = ip_encoder

    # IP-Adapter live PE encoding (PE-LoRA, or no cached features) needs
    # batch["images"] every step. With cache_latents=true the dataset
    # would normally skip image loading; this flag forces it to keep
    # decoding the source image alongside the cached latent so the live
    # PE forward has its input. VAE encoding still runs from cache.
    if getattr(args, "use_ip_adapter", False) and not getattr(
        args, "ip_features_cache_to_disk", False
    ):
        for dataset in train_dataset_group.datasets:
            dataset.force_load_images_for_ip = True
        if val_dataset_group is not None:
            for dataset in val_dataset_group.datasets:
                dataset.force_load_images_for_ip = True

    # IP-Adapter distinct-pair (identity) training. When opted in
    # (ip_pair_mode != "self") each dataset draws the IP-path reference from
    # a *different* image of the target's identity instead of the target
    # itself, removing the self-pair copy shortcut. Requires cached PE
    # features (the pairing is a stem swap on disk). See
    # docs/proposal/ip-adapter-identity-pairs.md.
    ip_pair_mode = str(getattr(args, "ip_pair_mode", "self") or "self")
    if getattr(args, "use_ip_adapter", False) and ip_pair_mode != "self":
        if not getattr(args, "ip_features_cache_to_disk", False):
            raise ValueError(
                "ip_pair_mode requires ip_features_cache_to_disk=true "
                "(distinct-pair training swaps which stem's cached PE "
                "features feed the IP path). PE-LoRA's live encoder is "
                "incompatible — set pe_lora_enabled=false."
            )
        index_path = getattr(
            args,
            "ip_pair_index",
            "post_image_dataset/captions/caption_index.json",
        )
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"ip_pair_index not found: {index_path}. Run "
                "`python tasks.py download-tagger`, then "
                "`python tasks.py caption-index`."
            )
        pair_kwargs = dict(
            index_path=index_path,
            mode=ip_pair_mode,
            prob=float(getattr(args, "ip_pair_prob", 0.8)),
            min_level=str(getattr(args, "ip_pair_min_level", "artist")),
            caption_strip_p=float(getattr(args, "ip_pair_caption_strip_p", 0.0)),
        )
        for dataset in train_dataset_group.datasets:
            dataset.setup_identity_pairs(is_validation=False, **pair_kwargs)
        if val_dataset_group is not None:
            for dataset in val_dataset_group.datasets:
                dataset.setup_identity_pairs(is_validation=True, **pair_kwargs)
        logger.info(
            f"IP-Adapter distinct pairs: mode={ip_pair_mode} "
            f"prob={pair_kwargs['prob']} min_level={pair_kwargs['min_level']} "
            f"caption_strip_p={pair_kwargs['caption_strip_p']} "
            f"index={index_path}"
        )

    # Soft-tokens contrastive negatives. The objective's knobs live in
    # ``network_args`` (see configs/methods/soft_tokens.toml); preview them
    # here to decide whether
    # the dataset should surface cached negative text embeddings. Off unless
    # contrastive_weight > 0. See docs/proposal/soft_tokens_contrastive.md.
    if str(getattr(args, "network_module", "") or "") == (
        "networks.methods.soft_tokens"
    ):
        net_arg_preview: dict[str, str] = {}
        for na in args.network_args or []:
            if "=" in na:
                pk, pv = na.split("=", 1)
                net_arg_preview[pk] = pv
        con_weight = float(net_arg_preview.get("contrastive_weight", 0.0) or 0.0)
        if con_weight > 0.0:
            con_k = int(net_arg_preview.get("contrastive_k", 1) or 1)
            con_mode = str(net_arg_preview.get("contrastive_negative_mode", "shuffled"))
            # The negative grouping always comes from the shared caption
            # index `make caption-index` writes — not a user knob.
            con_index = "post_image_dataset/captions/caption_index.json"
            if not os.path.exists(con_index):
                raise FileNotFoundError(
                    f"contrastive_index not found: {con_index}. "
                    "Run `python tasks.py download-tagger`, then "
                    "`python tasks.py caption-index`."
                )
            if not getattr(args, "cache_llm_adapter_outputs", False):
                raise ValueError(
                    "soft_tokens contrastive requires "
                    "cache_llm_adapter_outputs=true (negatives are cached "
                    "crossattn_emb swapped off disk)."
                )
            # Negatives only feed the training-step contrastive forward; the
            # validation FM-MSE stays a clean baseline, so val datasets are
            # left untouched.
            for dataset in train_dataset_group.datasets:
                dataset.setup_contrastive_negatives(
                    con_index, k=con_k, mode=con_mode, is_validation=False
                )
            logger.info(
                f"Soft-tokens contrastive: weight={con_weight} k={con_k} "
                f"mode={con_mode} index={con_index}"
            )

    train_dataset_group.verify_bucket_reso_steps(
        16
    )  # WanVAE spatial downscale = 8 and patch size = 2
    if val_dataset_group is not None:
        val_dataset_group.verify_bucket_reso_steps(16)


def recheck_compat_after_precision_resolve(args) -> None:
    """在 ``resolve_mixed_precision`` 改写之后复跑一次兼容检查（P0-1）。

    ``assert_training_extra_args`` 里的 ``check_training_compat`` 跑在
    ``resolve_mixed_precision`` **之前**，而 ``precision_policy`` 会在 sm<80 上
    把 ``bf16`` 就地改写成 ``fp16``（NF4 路径豁免）。于是所有以原始
    ``mixed_precision`` 为条件的 fail-closed 规则都被绕过：

    - ``z_image_bf16_only`` 看见 bf16 放行，实际以 fp16 运行；
    - ``qwen_image_2_1_mixed_precision_flash`` 同理，本该在启动期报错的硬 code
      从未生效，问题推迟到首次 forward 才由 attention backend 抛出。

    走到这里的调用点已经通过了一次检查（有 error 早在那时抛了），所以**任何**
    新增 error 都是改写引入的，不会出现「原配置错一遍、改写后又错一遍」的重复
    报错。这里只复检 error，不重复应用 mutation（第一次已应用）。
    """
    compat = check_training_compat(args)
    if not compat.errors:
        return
    details = "\n".join(f"  - [{issue.code}] {issue.message}" for issue in compat.errors)
    raise ValueError(
        "mixed_precision 被自动改写后触发了模型族限制：当前 GPU 不支持原生 bf16"
        "（sm<80），bfloat16 已被 precision_policy 降级为 float16，而该模型族不允许"
        "当前的 fp16 组合。\n"
        "原配置在改写前已通过兼容检查，以下冲突由自动降级引入：\n"
        f"{details}\n"
        "请改用支持原生 bf16 的硬件（sm80+），或按上述提示调整配置。"
    )
