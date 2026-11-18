"""Shared safetensors text-cache mechanics that are family-independent."""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library import cache_identity as _identity
from library.models.family_registry import (
    get_model_family_spec,
    validate_text_cache_metadata,
)

logger = logging.getLogger(__name__)

_warned_legacy_variants_cache: set[str] = set()
_IN_MEMORY_VARIANTS_KEY = "__text_cache_caption_variants__"


class VariantSafetensorsTextCacheStrategy:
    """Mixin for selecting one caption variant from a safetensors cache.

    Family strategies retain ownership of tensor names, schema validation, and
    encoder-specific padding/length semantics. This mixin only owns the shared
    ``multi_source``/``v0_intact`` selection contract.
    """

    def _select_variant_index(
        self,
        num_variants: int,
        *,
        multi_source: bool,
        v0_intact: bool,
        cache_path: str | None = None,
    ) -> int:
        if num_variants < 1:
            raise ValueError(
                f"invalid {self.CACHE_LABEL} caption variant count in "
                f"{cache_path or 'memory'}: {num_variants}"
            )
        if multi_source:
            return random.randint(0, num_variants - 1)
        if not self.use_shuffled_caption_variants:
            return 0
        if not v0_intact:
            label = str(self.CACHE_LABEL)
            if label not in _warned_legacy_variants_cache:
                logger.warning(
                    "Loaded a legacy %s multi-variant TE cache without "
                    "`v0_intact` (%s); sampling variants uniformly. Re-run "
                    "preprocess-te for 20%% pristine / 80%% shuffled sampling.",
                    label,
                    cache_path or "memory",
                )
                _warned_legacy_variants_cache.add(label)
            return random.randint(0, num_variants - 1)
        if num_variants == 1 or random.random() < 0.2:
            return 0
        return random.randint(1, num_variants - 1)


class FamilySafetensorsTextCacheStrategy(VariantSafetensorsTextCacheStrategy):
    """Shared cache mechanics for fixed-width hidden-state text encoders."""

    MODEL_FAMILY = ""
    CACHE_LABEL = "text encoder"
    EXPECTED_HIDDEN_RANK = 2
    CACHE_DTYPE = torch.bfloat16

    def __init__(
        self,
        cache_to_disk: bool = True,
        batch_size: int | str | None = None,
        skip_disk_cache_validity_check: bool = False,
        use_shuffled_caption_variants: bool = False,
        identity_inputs: dict | None = None,
        require_cache_identity: bool = False,
    ) -> None:
        self._cache_to_disk = cache_to_disk
        self._batch_size = batch_size
        self.skip_disk_cache_validity_check = skip_disk_cache_validity_check
        self.use_shuffled_caption_variants = use_shuffled_caption_variants
        self.identity_inputs = dict(identity_inputs or {})
        self.require_cache_identity = bool(require_cache_identity)
        self._preprocess_batcher = None

    @property
    def cache_to_disk(self):
        return self._cache_to_disk

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_partial(self):
        return False

    @property
    def is_weighted(self):
        return False

    @property
    def preprocess_batch_size(self) -> int:
        if self.batch_size == "auto":
            return self._preprocess_batcher.batch_size if self._preprocess_batcher else 1
        return max(1, int(self.batch_size or 1))

    def get_outputs_npz_path(
        self,
        image_abs_path: str,
        cache_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def _observe_cached_hidden(
        self, source: torch.Tensor, stored: torch.Tensor
    ) -> None:
        del source, stored

    def expected_cache_identity(self) -> dict | None:
        if not self.identity_inputs:
            return None
        allowed = {
            "text_encoder",
            "tokenizer",
            "t5_tokenizer",
            "vocab_pack",
            "dit",
            "llm_adapter_outputs",
            "compute_dtype",
            "caption_shuffle_variants",
            "tag_dropout_rate",
        }
        inputs = {
            key: value
            for key, value in self.identity_inputs.items()
            if key in allowed
        }
        return _identity.text_cache_identity(
            family=self.MODEL_FAMILY,
            **inputs,
        )

    def _cache_metadata(self) -> dict[str, str]:
        metadata = get_model_family_spec(self.MODEL_FAMILY).text_cache.metadata(
            self.MODEL_FAMILY
        )
        return _identity.stamp(metadata, self.expected_cache_identity())

    def _validate_cache_metadata(self, metadata, path: str) -> None:
        validate_text_cache_metadata(
            metadata, family=self.MODEL_FAMILY, path=path
        )
        expected = self.expected_cache_identity()
        if expected is None:
            return
        recorded = _identity.read_stamp(metadata)
        if _identity.evaluate(
            recorded,
            expected,
            require_identity=self.require_cache_identity,
        ) == "stale":
            raise ValueError(f"stale {self.CACHE_LABEL} text cache identity: {path}")
        if _identity.compare(recorded, expected) == "absent":
            _identity.warn_legacy("text encoder", path)

    @classmethod
    def _validate_tensor_layout(
        cls,
        handle,
        hidden_key: str,
        mask_key: str,
        *,
        cache_path: str,
        stamped: bool,
    ) -> None:
        hidden_slice = handle.get_slice(hidden_key)
        mask_slice = handle.get_slice(mask_key)
        hidden_shape = hidden_slice.get_shape()
        mask_shape = mask_slice.get_shape()
        expected_width = get_model_family_spec(cls.MODEL_FAMILY).text_cache.hidden_width
        valid = (
            len(hidden_shape) == cls.EXPECTED_HIDDEN_RANK
            and len(mask_shape) == 1
            and hidden_shape[0] == mask_shape[0]
            and hidden_slice.get_dtype() == "BF16"
            and mask_slice.get_dtype() == "BOOL"
            and (
                not stamped
                or expected_width is None
                or hidden_shape[-1] == expected_width
            )
        )
        if not valid:
            raise ValueError(
                f"invalid {cls.CACHE_LABEL} text cache tensor layout for {cache_path}: "
                f"{hidden_key} shape={hidden_shape} dtype={hidden_slice.get_dtype()}, "
                f"{mask_key} shape={mask_shape} dtype={mask_slice.get_dtype()}"
            )

    def load_outputs_npz(self, npz_path: str) -> List[torch.Tensor]:
        with safe_open(npz_path, framework="pt") as handle:
            metadata = handle.metadata()
            self._validate_cache_metadata(metadata, npz_path)
            keys = set(handle.keys())
            if "num_variants" in keys:
                num_variants = int(handle.get_tensor("num_variants"))
                variant_index = self._select_variant_index(
                    num_variants,
                    multi_source="caption_multi_source" in keys,
                    v0_intact="v0_intact" in keys,
                    cache_path=npz_path,
                )
                hidden_key = f"hiddens_v{variant_index}"
                mask_key = f"mask_v{variant_index}"
            else:
                hidden_key = "hiddens"
                mask_key = "mask"
            self._validate_tensor_layout(
                handle,
                hidden_key,
                mask_key,
                cache_path=npz_path,
                stamped=bool((metadata or {}).get("cache_schema")),
            )
            hiddens = handle.get_tensor(hidden_key)
            mask = handle.get_tensor(mask_key)
            rate = (
                handle.get_tensor("caption_dropout_rate")
                if "caption_dropout_rate" in keys
                else torch.tensor(0.0, dtype=torch.float32)
            )
        return [hiddens, mask, rate]

    def is_disk_cached_outputs_expected(
        self,
        npz_path: str,
        *,
        expected_num_variants: Optional[int] = None,
        expected_caption_shuffle_variants: Optional[int] = None,
        expected_caption_tag_dropout_rate: Optional[float] = None,
        expected_multi_source: Optional[bool] = None,
    ) -> bool:
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            with safe_open(npz_path, framework="pt") as handle:
                metadata = handle.metadata()
                self._validate_cache_metadata(metadata, npz_path)
                keys = set(handle.keys())
                has_variants = "num_variants" in keys
                num_variants = int(handle.get_tensor("num_variants")) if has_variants else 0
                if has_variants and num_variants < 1:
                    return False
                if self.use_shuffled_caption_variants and not has_variants:
                    return False
                if expected_num_variants is not None:
                    if num_variants != expected_num_variants:
                        return False
                    if expected_multi_source is not None:
                        if ("caption_multi_source" in keys) != expected_multi_source:
                            return False
                    if expected_num_variants > 0:
                        if "v0_intact" not in keys:
                            return False
                        if expected_caption_shuffle_variants is not None:
                            if "caption_shuffle_variants" not in keys or int(
                                handle.get_tensor("caption_shuffle_variants")
                            ) != int(expected_caption_shuffle_variants):
                                return False
                        if expected_caption_tag_dropout_rate is not None:
                            if "caption_tag_dropout_rate" not in keys:
                                return False
                            cached_rate = float(
                                handle.get_tensor("caption_tag_dropout_rate")
                            )
                            if abs(cached_rate - expected_caption_tag_dropout_rate) > 1e-7:
                                return False
                if "caption_dropout_rate" not in keys:
                    return False
                layouts = (
                    [(f"hiddens_v{index}", f"mask_v{index}") for index in range(num_variants)]
                    if num_variants > 0
                    else [("hiddens", "mask")]
                )
                for hidden_key, mask_key in layouts:
                    if hidden_key not in keys or mask_key not in keys:
                        return False
                    self._validate_tensor_layout(
                        handle,
                        hidden_key,
                        mask_key,
                        cache_path=npz_path,
                        stamped=bool((metadata or {}).get("cache_schema")),
                    )
                rate_slice = handle.get_slice("caption_dropout_rate")
                if rate_slice.get_shape() != [] or rate_slice.get_dtype() != "F32":
                    return False
            return True
        except Exception:
            return False

    def select_in_memory_outputs(self, outputs):
        if not isinstance(outputs, dict) or not outputs.get(_IN_MEMORY_VARIANTS_KEY):
            return outputs
        num_variants = int(outputs["hiddens"].shape[0])
        variant_index = self._select_variant_index(
            num_variants,
            multi_source=bool(outputs["caption_multi_source"]),
            v0_intact=True,
        )
        return [
            outputs["hiddens"][variant_index],
            outputs["mask"][variant_index],
            outputs["caption_dropout_rate"],
        ]

    def _encode_captions(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        captions: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def encode(group):
            hiddens, mask = text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokenize_strategy.tokenize(group)
            )
            stored_hiddens = hiddens.detach().to(
                dtype=self.CACHE_DTYPE, device="cpu"
            )
            self._observe_cached_hidden(hiddens.detach(), stored_hiddens)
            return (
                stored_hiddens,
                mask.detach().to(dtype=torch.bool, device="cpu"),
            )

        if self.batch_size == "auto":
            from library.preprocess.adaptive_batch import AutoBatcher

            if self._preprocess_batcher is None:
                configured_device = getattr(models[0], "_qwen_execution_device", None)
                device = (
                    torch.device(configured_device)
                    if configured_device is not None
                    else next(models[0].parameters()).device
                )
                self._preprocess_batcher = AutoBatcher(
                    device, label=f"{self.MODEL_FAMILY}/text"
                )
            return self._preprocess_batcher.encode_all(captions, encode)

        batch_size = max(1, int(self.batch_size or len(captions) or 1))
        chunks = [encode(captions[start : start + batch_size]) for start in range(0, len(captions), batch_size)]
        return (
            torch.cat([chunk[0] for chunk in chunks], dim=0),
            torch.cat([chunk[1] for chunk in chunks], dim=0),
        )

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: List,
    ) -> None:
        variants_by_info = [
            list(getattr(info, "caption_variants", None) or [info.caption])
            for info in batch
        ]
        variant_layout = [
            bool(getattr(info, "cache_caption_variants", False)) or len(variants) > 1
            for info, variants in zip(batch, variants_by_info)
        ]
        captions = [caption for variants in variants_by_info for caption in variants]
        hiddens, mask = self._encode_captions(
            tokenize_strategy, models, text_encoding_strategy, captions
        )
        offset = 0
        spec = get_model_family_spec(self.MODEL_FAMILY)
        for info, variants, use_variants in zip(batch, variants_by_info, variant_layout):
            rate = torch.tensor(info.caption_dropout_rate, dtype=torch.float32)
            if use_variants:
                save_dict = {
                    "num_variants": torch.tensor(len(variants), dtype=torch.int64),
                    "v0_intact": torch.tensor(1, dtype=torch.int8),
                    "caption_shuffle_variants": torch.tensor(
                        int(getattr(info, "caption_shuffle_variants", len(variants))),
                        dtype=torch.int64,
                    ),
                    "caption_tag_dropout_rate": torch.tensor(
                        float(getattr(info, "caption_tag_dropout_rate", 0.0)),
                        dtype=torch.float32,
                    ),
                    "caption_dropout_rate": rate,
                }
                if getattr(info, "caption_multi_source", False):
                    save_dict["caption_multi_source"] = torch.tensor(1, dtype=torch.int8)
                for index in range(len(variants)):
                    save_dict[f"hiddens_v{index}"] = hiddens[offset + index].contiguous()
                    save_dict[f"mask_v{index}"] = mask[offset + index].contiguous()
            else:
                save_dict = {
                    "hiddens": hiddens[offset].contiguous(),
                    "mask": mask[offset].contiguous(),
                    "caption_dropout_rate": rate,
                }
            if self.cache_to_disk:
                save_file(
                    save_dict,
                    info.text_encoder_outputs_npz,
                    metadata=self._cache_metadata(),
                )
            elif use_variants:
                end = offset + len(variants)
                info.text_encoder_outputs = {
                    _IN_MEMORY_VARIANTS_KEY: True,
                    "hiddens": hiddens[offset:end].clone(),
                    "mask": mask[offset:end].clone(),
                    "caption_multi_source": bool(
                        getattr(info, "caption_multi_source", False)
                    ),
                    "caption_dropout_rate": rate,
                }
            else:
                info.text_encoder_outputs = (
                    hiddens[offset].clone(),
                    mask[offset].clone(),
                    rate,
                )
            offset += len(variants)


@contextmanager
def text_encoder_cache_session(
    model: torch.nn.Module,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
    restore: bool = True,
) -> Iterator[torch.nn.Module]:
    """Temporarily place a regular text encoder on a cache device.

    This session deliberately handles only eager, non-offloaded encoders. Qwen
    Image 2.1's meta-parameter/Accelerate path remains in its dedicated runtime
    because it has stronger cleanup and partial-install failure semantics.
    """
    target = torch.device(device)
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    tensors = parameters + buffers
    original_specs = {
        id(tensor): (tensor.device, tensor.dtype) for tensor in tensors
    }
    try:
        if dtype is None:
            model.to(target)
        else:
            model.to(device=target, dtype=dtype)
        yield model
    finally:
        if restore:
            for tensor in (*model.parameters(), *model.buffers()):
                original_device, original_dtype = original_specs[id(tensor)]
                tensor.data = tensor.data.to(
                    device=original_device,
                    dtype=original_dtype,
                )
        if target.type == "cuda":
            torch.cuda.empty_cache()
