"""Network/training-hook wrappers used by AnimaTrainer."""

from __future__ import annotations

import argparse
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator

from library import train_util
from library.anima import strategy as strategy_anima
from library.datasets import DatasetGroup, MinimalDataset
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.runtime import noise as noise_utils
from library.training.contexts import TrainCtx


class TrainerNetworkMixin:
    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        from library.training.extra_args import assert_training_extra_args

        assert_training_extra_args(args, train_dataset_group, val_dataset_group)

    def load_target_model(
        self, args, weight_dtype, accelerator, load_qwen3=True, load_vae=True
    ):
        from library.training.model_loading import (
            load_target_model as _load_target_model,
        )

        return _load_target_model(
            self,
            args,
            weight_dtype,
            accelerator,
            load_qwen3=load_qwen3,
            load_vae=load_vae,
        )

    def load_unet_lazily(
        self, args, weight_dtype, accelerator, text_encoders
    ) -> tuple[nn.Module, list[nn.Module]]:
        from library.training.model_loading import load_unet_lazily as _load_unet_lazily

        return _load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders)

    def get_tokenize_strategy(self, args):
        from library.training.anima_strategies import get_tokenize_strategy as _impl

        return _impl(args)

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        from library.training.anima_strategies import get_tokenizers as _impl

        return _impl(tokenize_strategy)

    def get_latents_caching_strategy(self, args):
        from library.training.anima_strategies import (
            get_latents_caching_strategy as _impl,
        )

        return _impl(args)

    def get_text_encoding_strategy(self, args):
        from library.training.anima_strategies import (
            get_text_encoding_strategy as _impl,
        )

        return _impl(args)

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        from library.training.anima_strategies import (
            get_models_for_text_encoding as _impl,
        )

        return _impl(args, accelerator, text_encoders)

    def _ensure_uncond_crossattn(
        self,
        args: argparse.Namespace,
        accelerator,
        weight_dtype: torch.dtype,
    ) -> None:
        from library.training.uncond_sidecar import ensure_uncond_crossattn

        return ensure_uncond_crossattn(self, args, accelerator, weight_dtype)

    @staticmethod
    def _prior_preservation_enabled(args: argparse.Namespace) -> bool:
        from library.training.prior_preservation import prior_preservation_enabled

        return prior_preservation_enabled(args)

    @staticmethod
    def _blank_prompt_preservation_enabled(args: argparse.Namespace) -> bool:
        from library.training.prior_preservation import (
            blank_prompt_preservation_enabled,
        )

        return blank_prompt_preservation_enabled(args)

    @staticmethod
    def _diff_output_preservation_enabled(args: argparse.Namespace) -> bool:
        from library.training.prior_preservation import diff_output_preservation_enabled

        return diff_output_preservation_enabled(args)

    @staticmethod
    def _inverted_mask_prior_enabled(args: argparse.Namespace, batch: dict) -> bool:
        from library.training.prior_preservation import inverted_mask_prior_enabled

        return inverted_mask_prior_enabled(args, batch)

    def get_noise_scheduler(
        self, args: argparse.Namespace, device: torch.device
    ) -> Any:
        noise_scheduler = noise_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        from library.env import resolve_model_family
        from library.models.family_registry import dispatch_model_family

        def encode_qwen_vae(loaded_vae, pixels):
            loaded_vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
            return loaded_vae.encode_pixels_to_latents(pixels)

        def encode_z_image_vae(loaded_vae, pixels):
            from library.models.z_image.latent import encode_z_image_latents

            return encode_z_image_latents(loaded_vae, pixels)

        def encode_qwen_image_2_1_vae(loaded_vae, pixels):
            from library.models.qwen_image_2_1.latent import encode_qwen_image_2_1_latents

            return encode_qwen_image_2_1_latents(loaded_vae, pixels)

        def encode_moss_vae(loaded_vae, pixels):
            from library.models.moss_image.latent import encode_moss_latents

            return encode_moss_latents(loaded_vae, pixels)

        encoder = dispatch_model_family(
            resolve_model_family(args),
            operation="live VAE encoding",
            handlers={
                "anima": encode_qwen_vae,
                "krea2_raw": encode_qwen_vae,
                "z_image": encode_z_image_vae,
                "qwen_image_2_1": encode_qwen_image_2_1_vae,
                "moss_image": encode_moss_vae,
            },
        )
        return encoder(vae, images)  # Keep 4D for input/output

    def shift_scale_latents(self, args, latents):
        # Latents already normalized by vae.encode with scale
        return latents

    def get_noise_pred_and_target(
        self,
        ctx: TrainCtx,
        latents,
        batch,
        text_encoder_conds,
        *,
        is_train=True,
    ):
        from library.env import resolve_model_family
        from library.models.family_registry import get_family_plugin

        plugin = get_family_plugin(resolve_model_family(ctx.args), capability="flow")
        return plugin.compute_noise_pred_and_target(
            self, ctx, latents, batch, text_encoder_conds, is_train=is_train
        )

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        from library.training.unet_prepare import (
            prepare_unet_with_accelerator as _prepare_unet_with_accelerator,
        )

        return _prepare_unet_with_accelerator(self, args, accelerator, unet)

    def on_validation_step_end(self, ctx: TrainCtx, batch):
        from library.training.unet_prepare import on_validation_step_end as _impl

        return _impl(self, ctx, batch)

    def process_batch(
        self,
        ctx: TrainCtx,
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        """Override base process_batch to surface caption_dropout_rates for on-device dropout."""
        from library.training.batch_preprocess import split_cached_text_encoder_outputs

        batch = split_cached_text_encoder_outputs(batch)
        return self._process_batch_inner(ctx, batch, is_train=is_train)

    def _process_batch_inner(
        self,
        ctx: TrainCtx,
        batch,
        *,
        is_train=True,
    ) -> torch.Tensor:
        from library.training.batch_step import process_batch_inner

        return process_batch_inner(self, ctx, batch, is_train=is_train)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        from library.training.functional_hooks import post_process_network

        return post_process_network(
            self, args, accelerator, network, text_encoders, unet
        )

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec_dataclass(
            args, lora=True
        ).to_metadata_dict()

    def update_metadata(self, metadata, args):
        from library.training.metadata import update_anima_metadata

        return update_anima_metadata(metadata, args)

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set first parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug
        first_param = next(text_encoder.parameters())
        first_param.requires_grad_(True)

    def get_text_encoders_train_flags(self, args, text_encoders):
        return (
            [True] * len(text_encoders)
            if self.is_train_text_encoder(args)
            else [False] * len(text_encoders)
        )

    def on_step_start(self, ctx: TrainCtx, batch, *, is_train: bool = True):
        from library.training.step_hooks import on_step_start as _on_step_start

        return _on_step_start(self, ctx, batch, is_train=is_train)

    def run_after_backward(self, ctx: TrainCtx):
        """Dispatch the post-backward hook to adapters (between
        ``accelerator.backward`` and gradient clipping)."""
        from library.training.step_hooks import (
            run_after_backward as _run_after_backward,
        )

        return _run_after_backward(self, ctx)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cast_text_encoder(self, args):
        return True

    def cast_vae(self, args):
        return True

    def cast_unet(self, args):
        return True

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        **kwargs,
    ):
        from library.training.call_unet import call_unet as _call_unet

        return _call_unet(
            args,
            accelerator,
            unet,
            noisy_latents,
            timesteps,
            text_conds,
            batch,
            weight_dtype,
            **kwargs,
        )

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator: Accelerator,
        text_encoders,
        dataset: DatasetGroup,
    ):
        from library.training.text_encoder_cache import (
            cache_text_encoder_outputs_if_needed as _cache_text_encoder_outputs_if_needed,
        )

        return _cache_text_encoder_outputs_if_needed(
            self,
            args,
            accelerator,
            text_encoders,
            dataset,
        )
