"""Qwen3-VL T2I prompt encoding and isolated text/latent caches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from library.anima import strategy as anima_strategy
from library.anima.text_strategies import TextEncodingStrategy, TokenizeStrategy
from library.io.cache import resolve_cache_path
from library.models.family_registry import get_model_family_spec
from library.strategies.text_cache import FamilySafetensorsTextCacheStrategy
from library.models.latent_space import QWEN_IMAGE_21_F16C64_P1
from library.models.qwen_image_2_1.latent import encode_qwen_image_2_1_latents
from library.datasets.qwen_image_edit import (
    EDIT_TEXT_CACHE_SUFFIX,
    edit_cache_suffix,
    edit_condition_fingerprint,
    prepare_reference_image,
    reference_size_for_bucket,
)


SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
MAX_PROMPT_LENGTH = 512


def _qwen_execution_device(model: Any) -> torch.device:
    """Resolve the device used by a normal or Accelerate-offloaded encoder.

    ``accelerate.cpu_offload`` intentionally replaces parameters with meta
    tensors.  Looking at ``next(model.parameters()).device`` therefore gives
    ``meta`` and sends inputs to an unusable device.  The cache runtime stores
    the real execution device on the outer model; regular eager models keep
    the historical parameter based fallback.
    """
    configured = getattr(model, "_qwen_execution_device", None)
    if configured is not None:
        return torch.device(configured)
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - malformed test double
        raise ValueError("Qwen Image 2.1 text encoder has no parameters") from exc


def _qwen_encoder_forward(model: Any, kwargs: dict[str, Any]) -> Any:
    """Run the core encoder while retaining compatibility with tiny test doubles."""
    core = getattr(model, "model", None)
    if callable(core):
        return core(**kwargs)
    return model(**kwargs)


def _qwen_last_hidden_state(outputs: Any) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    states = getattr(outputs, "hidden_states", None)
    if states:
        return states[-1]
    raise ValueError("Qwen3-VL encoder returned no last hidden state")


def resolve_tokenizer_path(text_encoder_path: str) -> str:
    path = Path(text_encoder_path).expanduser()
    for parent in (path.parent, *path.parents):
        for candidate in (parent / "tokenizer", parent / "comfy" / "text_encoders" / "qwen25_tokenizer"):
            if (candidate / "tokenizer_config.json").is_file():
                return str(candidate)
    raise FileNotFoundError(
        "Qwen Image 2.1 needs a local Qwen3-VL tokenizer; supply a sibling tokenizer/ "
        "or a ComfyUI installation with comfy/text_encoders/qwen25_tokenizer"
    )


class QwenImage21TokenizeStrategy(TokenizeStrategy):
    def __init__(self, text_encoder_path: str, max_length: int = MAX_PROMPT_LENGTH):
        from transformers import AutoTokenizer

        self.tokenizer_path = resolve_tokenizer_path(text_encoder_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        self.drop_idx = len(self.tokenizer.encode(prefix, add_special_tokens=False))
        self.max_length = max_length
        if self.tokenizer.pad_token_id != 151643:
            raise ValueError("Qwen Image 2.1 tokenizer must use Qwen3-VL pad token 151643")

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        captions = [text] if isinstance(text, str) else list(text)
        prompts = [
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{caption or ' '}<|im_end|>\n<|im_start|>assistant\n"
            for caption in captions
        ]
        encoded = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=self.max_length + self.drop_idx,
            return_tensors="pt", add_special_tokens=False,
        )
        return [encoded.input_ids, encoded.attention_mask]

    def tokenize_with_weights(self, text: Union[str, List[str]]) -> tuple:
        tokens = self.tokenize(text)
        return tokens, [torch.ones_like(tokens[0], dtype=torch.float32)]


class QwenImage21EditTokenizeStrategy(QwenImage21TokenizeStrategy):
    """Qwen3-VL processor setup used only by paired reference-image training."""

    def __init__(self, text_encoder_path: str, max_length: int = MAX_PROMPT_LENGTH):
        super().__init__(text_encoder_path, max_length)
        from transformers import Qwen3VLProcessor
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

        self.processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                min_pixels=32 * 32,
                max_pixels=16_777_216,
            ),
            tokenizer=self.tokenizer,
            video_processor=Qwen3VLVideoProcessor(),
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if self.image_token_id is None or self.image_token_id < 0:
            raise ValueError("Qwen Image 2.1 Edit tokenizer is missing <|image_pad|>")


def encode_edit_prompt(
    model: Any,
    tokenize_strategy: QwenImage21EditTokenizeStrategy,
    instruction: str,
    image: Image.Image | list[Image.Image] | tuple[Image.Image, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    processor = tokenize_strategy.processor
    images = [image] if isinstance(image, Image.Image) else list(image)
    if not images or any(not isinstance(item, Image.Image) for item in images):
        raise ValueError("Qwen Image 2.1 Edit requires one or more PIL reference images")
    if len(images) == 1:
        image_prefix = "<|vision_start|><|image_pad|><|vision_end|>"
    else:
        image_prefix = " ".join(
            f"<image{index}><|vision_start|><|image_pad|><|vision_end|>"
            for index in range(1, len(images) + 1)
        )
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{image_prefix}"
        f"{instruction or ' '}<|im_end|>\n<|im_start|>assistant\n"
    )
    inputs = processor(
        text=[prompt],
        images=images,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        images_kwargs={"do_resize": False},
    )
    device = _qwen_execution_device(model)
    inputs = inputs.to(device)
    forward_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "output_hidden_states": False,
        "use_cache": False,
    }
    for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        value = getattr(inputs, key, None)
        if value is not None:
            forward_kwargs[key] = value

    language_model = getattr(model.model, "language_model", model.model)
    handle = language_model.norm.register_forward_hook(
        lambda _module, args, _output: args[0]
    )
    try:
        with torch.no_grad():
            outputs = _qwen_encoder_forward(model, forward_kwargs)
    finally:
        handle.remove()

    valid = inputs.attention_mask[0].bool()
    ids = inputs.input_ids[0][valid][tokenize_strategy.drop_idx:]
    hiddens = _qwen_last_hidden_state(outputs)[0][valid][tokenize_strategy.drop_idx:]
    slots = ids.eq(tokenize_strategy.image_token_id)
    if not slots.any():
        raise ValueError("Qwen Image 2.1 Edit processor emitted no reference image slots")
    return hiddens, torch.ones_like(slots, dtype=torch.bool), slots


def _expected_edit_reference_latent_shape(info) -> tuple[int, int, int]:
    if not info.reference_image_path:
        raise ValueError("Qwen edit reference image path is missing")
    bucket_tuple = tuple(int(v) for v in info.bucket_reso)
    cached_key = getattr(info, "_edit_reference_size_key", None)
    cached_size = getattr(info, "edit_reference_size", None)
    if (
        cached_size is not None
        and len(cached_size) == 2
        and (cached_key is None or cached_key == (info.reference_image_path, bucket_tuple))
    ):
        width, height = int(cached_size[0]), int(cached_size[1])
    else:
        with Image.open(info.reference_image_path) as reference:
            width, height = reference_size_for_bucket(reference.size, bucket_tuple)
        try:
            info.edit_reference_size = (width, height)
            info._edit_reference_size_key = (info.reference_image_path, bucket_tuple)
        except Exception:
            pass
    space = QWEN_IMAGE_21_F16C64_P1
    stride = space.vae_spatial_compression
    if width % stride or height % stride:
        raise ValueError("Qwen edit reference resize is not aligned to the VAE stride")
    return space.latent_channels, height // stride, width // stride


class QwenImage21TextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(
        self, tokenize_strategy: QwenImage21TokenizeStrategy,
        models: List[Any], tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        model = models[0]
        input_ids, attention_mask = tokens
        device = _qwen_execution_device(model)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_model = model.model.language_model
        handle = text_model.norm.register_forward_hook(lambda _module, args, _output: args[0])
        try:
            with torch.no_grad():
                outputs = _qwen_encoder_forward(model, {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "output_hidden_states": False,
                    "use_cache": False,
                })
        finally:
            handle.remove()
        rows = [
            hidden[valid.bool()][tokenize_strategy.drop_idx:]
            for hidden, valid in zip(_qwen_last_hidden_state(outputs), attention_mask)
        ]
        if any(row.shape[0] == 0 for row in rows):
            raise ValueError("Qwen Image 2.1 prompt became empty after system-prefix removal")
        width = max(row.shape[0] for row in rows)
        hiddens = torch.stack([
            torch.cat([row, row.new_zeros(width - len(row), row.shape[1])])
            for row in rows
        ])
        mask = torch.stack([
            torch.arange(width, device=device) < len(row) for row in rows
        ])
        return [hiddens, mask]

    def apply_caption_dropout_inplace(self, caption_dropout_rates, **_kwargs) -> None:
        if torch.as_tensor(caption_dropout_rates).gt(0).any():
            raise ValueError("Qwen Image 2.1 caption dropout is not supported")


class QwenImage21TextCache(FamilySafetensorsTextCacheStrategy):
    MODEL_FAMILY = "qwen_image_2_1"
    CACHE_LABEL = "Qwen Image 2.1"
    EXPECTED_HIDDEN_RANK = 2

    def get_outputs_npz_path(
        self, image_abs_path: str, cache_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> str:
        suffix = get_model_family_spec(self.MODEL_FAMILY).text_cache.suffix
        return resolve_cache_path(image_abs_path, suffix, cache_dir=cache_dir, image_dir=image_dir)


class QwenImage21EditTextCache(QwenImage21TextCache):
    """Reference-aware text cache; its identity includes image, instruction and bucket."""

    EDIT_SUFFIX = EDIT_TEXT_CACHE_SUFFIX
    EDIT_CACHE_SCHEMA = "qwen-image-2.1-edit-v1"

    def get_outputs_npz_path(self, image_abs_path, cache_dir=None, image_dir=None):
        return resolve_cache_path(
            image_abs_path, self.EDIT_SUFFIX, cache_dir=cache_dir, image_dir=image_dir
        )

    def get_outputs_npz_path_for_info(self, info, subset) -> str:
        fingerprint = edit_condition_fingerprint(
            info.reference_image_path, info.caption, info.bucket_reso
        )
        suffix = edit_cache_suffix(
            task="te", fingerprint=fingerprint, target_bucket=info.bucket_reso
        )
        return resolve_cache_path(
            info.absolute_path,
            suffix,
            cache_dir=getattr(subset, "text_cache_dir", None)
            or getattr(subset, "cache_dir", None),
            image_dir=getattr(subset, "image_dir", None),
        )

    def is_expected_for_info(self, path: str, info) -> bool:
        if not self.cache_to_disk or not os.path.exists(path):
            return False
        expected = edit_condition_fingerprint(
            info.reference_image_path, info.caption, info.bucket_reso
        )
        try:
            with safe_open(path, framework="pt") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("edit_condition_fingerprint") != expected:
                    return False
                if metadata.get("edit_cache_schema") != self.EDIT_CACHE_SCHEMA:
                    return False
                keys = set(handle.keys())
                if not {"hiddens", "mask", "image_slots", "caption_dropout_rate"} <= keys:
                    return False
                hidden_shape = handle.get_slice("hiddens").get_shape()
                if not (
                    len(hidden_shape) == 2
                    and hidden_shape[-1] == 4096
                    and handle.get_slice("mask").get_shape() == [hidden_shape[0]]
                    and handle.get_slice("image_slots").get_shape() == [hidden_shape[0]]
                ):
                    return False
                _, ref_height, ref_width = _expected_edit_reference_latent_shape(info)
                image_slots = handle.get_tensor("image_slots")
                return int(image_slots.bool().sum()) == ref_height * ref_width // 4
        except Exception:
            return False

    def load_outputs_npz(self, path: str) -> List[torch.Tensor]:
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("edit_cache_schema") != self.EDIT_CACHE_SCHEMA:
                raise ValueError(f"invalid Qwen Image 2.1 Edit text cache: {path}")
            outputs = [
                handle.get_tensor("hiddens"),
                handle.get_tensor("mask"),
                handle.get_tensor("image_slots"),
                handle.get_tensor("caption_dropout_rate"),
            ]
        hiddens, mask, image_slots, dropout = outputs
        if (
            hiddens.ndim != 2
            or hiddens.shape[-1] != 4096
            or mask.shape != hiddens.shape[:1]
            or image_slots.shape != hiddens.shape[:1]
            or dropout.numel() != 1
            or not torch.isfinite(hiddens).all()
        ):
            raise ValueError(f"invalid Qwen Image 2.1 Edit text cache tensors: {path}")
        return outputs

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, batch):
        model = models[0]
        spec = get_model_family_spec(self.MODEL_FAMILY)
        for info in batch:
            image, _pixels = prepare_reference_image(
                info.reference_image_path, info.bucket_reso
            )
            hiddens, mask, slots = encode_edit_prompt(
                model, tokenize_strategy, info.caption, image
            )
            fingerprint = edit_condition_fingerprint(
                info.reference_image_path, info.caption, info.bucket_reso
            )
            tensors = {
                "hiddens": hiddens.to(dtype=torch.bfloat16, device="cpu").contiguous(),
                "mask": mask.to(dtype=torch.bool, device="cpu").contiguous(),
                "image_slots": slots.to(dtype=torch.bool, device="cpu").contiguous(),
                "caption_dropout_rate": torch.tensor(0.0, dtype=torch.float32),
            }
            if self.cache_to_disk:
                save_file(
                    tensors,
                    info.text_encoder_outputs_npz,
                    metadata={
                        **spec.text_cache.metadata(spec.name),
                        "edit_cache_schema": self.EDIT_CACHE_SCHEMA,
                        "edit_condition_fingerprint": fingerprint,
                    },
                )
            else:
                info.text_encoder_outputs = list(tensors.values())


class QwenImage21LatentCache(anima_strategy.AnimaLatentsCachingStrategy):
    ANIMA_LATENTS_NPZ_SUFFIX = QWEN_IMAGE_21_F16C64_P1.cache_suffix
    EDIT_LATENT_CACHE_SCHEMA = "qwen-image-2.1-edit-reference-v1"

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        return self._default_is_disk_cached_latents_expected(
            16, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(self, npz_path, bucket_reso):
        return self._default_load_latents_from_disk(16, npz_path, bucket_reso)

    def get_image_size_from_disk_cache_path(self, absolute_path, npz_path):
        stem = npz_path[: -len(self.cache_suffix)]
        width, height = stem.rsplit("_", 1)[-1].split("x")
        return int(width), int(height)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype

        def encode(images):
            return encode_qwen_image_2_1_latents(vae, images).to("cpu")

        self._default_cache_batch_latents(
            encode, vae_device, vae_dtype, image_infos,
            flip_aug, alpha_mask, random_crop, multi_resolution=True,
        )

        if any(getattr(info, "reference_image_path", None) for info in image_infos):
            from library.datasets.image_utils import IMAGE_TRANSFORMS

            for info in image_infos:
                if not info.reference_image_path:
                    continue
                image, pixels = prepare_reference_image(
                    info.reference_image_path, info.bucket_reso
                )
                info.edit_reference_size = image.size
                tensor = IMAGE_TRANSFORMS(pixels).unsqueeze(0).to(
                    device=vae_device, dtype=vae_dtype
                )
                with torch.no_grad():
                    latent = encode_qwen_image_2_1_latents(vae, tensor)[0].to("cpu")
                if self.cache_to_disk:
                    if not info.edit_reference_latent_path:
                        raise ValueError("Qwen edit reference latent cache path was not initialized")
                    fingerprint = edit_condition_fingerprint(
                        info.reference_image_path, info.caption, info.bucket_reso
                    )
                    save_file(
                        {"latent": latent.contiguous()},
                        info.edit_reference_latent_path,
                        metadata={
                            "model_family": "qwen_image_2_1",
                            "edit_cache_schema": self.EDIT_LATENT_CACHE_SCHEMA,
                            "edit_condition_fingerprint": fingerprint,
                        },
                    )
                else:
                    info.edit_reference_latent = latent

    def get_edit_reference_latent_path(self, info, subset) -> str:
        fingerprint = edit_condition_fingerprint(
            info.reference_image_path, info.caption, info.bucket_reso
        )
        suffix = edit_cache_suffix(
            task="ref", fingerprint=fingerprint, target_bucket=info.bucket_reso
        )
        return resolve_cache_path(
            info.absolute_path,
            suffix,
            cache_dir=getattr(subset, "cache_dir", None),
            image_dir=getattr(subset, "image_dir", None),
        )

    def is_edit_reference_cache_expected(self, info, subset) -> bool:
        if not info.reference_image_path:
            return True
        path = self.get_edit_reference_latent_path(info, subset)
        if not self.cache_to_disk or not os.path.exists(path):
            return False
        expected = edit_condition_fingerprint(
            info.reference_image_path, info.caption, info.bucket_reso
        )
        try:
            with safe_open(path, framework="pt") as handle:
                metadata = handle.metadata() or {}
                shape = handle.get_slice("latent").get_shape()
                expected_shape = _expected_edit_reference_latent_shape(info)
                return (
                    metadata.get("edit_cache_schema") == self.EDIT_LATENT_CACHE_SCHEMA
                    and metadata.get("edit_condition_fingerprint") == expected
                    and tuple(shape) == expected_shape
                )
        except Exception:
            return False

    def load_edit_reference_latent(self, info, subset):
        if not info.reference_image_path:
            return None
        in_memory_latent = getattr(info, "edit_reference_latent", None)
        verified_disk_key = None
        if in_memory_latent is not None:
            latent = in_memory_latent
            path = info.reference_image_path
        else:
            path = info.edit_reference_latent_path or self.get_edit_reference_latent_path(
                info, subset
            )
            if not getattr(info, "edit_reference_latent_path", None):
                try:
                    info.edit_reference_latent_path = path
                except Exception:
                    pass
            try:
                st = os.stat(path)
                verified_disk_key = (path, int(st.st_mtime_ns), int(st.st_size))
            except OSError:
                verified_disk_key = None
            with safe_open(path, framework="pt") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("edit_cache_schema") != self.EDIT_LATENT_CACHE_SCHEMA:
                    raise ValueError(f"invalid Qwen edit reference latent cache: {path}")
                latent = handle.get_tensor("latent")
        if tuple(latent.shape) != _expected_edit_reference_latent_shape(info):
            raise ValueError(f"Qwen edit reference latent shape mismatch: {path}")
        if (
            verified_disk_key is None
            or getattr(info, "_edit_ref_latent_verified_key", None) != verified_disk_key
        ):
            if not torch.isfinite(latent).all():
                raise ValueError(f"Qwen edit reference latent cache contains non-finite values: {path}")
            if verified_disk_key is not None:
                try:
                    info._edit_ref_latent_verified_key = verified_disk_key
                except Exception:
                    pass
        return latent.float()
