# Krea-2-Raw 文本链路 (阶段 1)
#
# Qwen3-VL-4B-Instruct 文本编码 + 12 层 MFA + caching。
#
# 与 anima (library/anima/strategy.py) 的关键差异 (R1 定论见
# docs/findings/krea2_raw_migration_stage0_findings.md §R1):
#   - anima 用 Qwen3 纯文本 LLM + LLM Adapter 桥到 T5 space, padding 位
#     "zero 作 cross-attn sink" (strategy.py:137 [~mask]=0 + models.py:2736)。
#   - Krea-2 用 Qwen3-VL-4B 直接, hidden_dim=2560 = DiT txtdim, 无需 adapter。
#     padding 用 attention mask 屏蔽 (DiT 内部不二次置零)。
#   - anima max-pad 到 512; Krea-2 ChatML 模板 pad 到 541 (512+34-5) + cat suffix。
#
# encoder.py 核实 (krea-ai/krea-2 commit db3984f, 见阶段1 子代理核实):
#   - Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
#   - select_layers=(2,5,8,11,14,17,20,23,26,29,32,35), stack dim=2
#   - prompt_template_encode_start_idx=34 (切掉 system prompt 前 34 token)
#   - max_length=512, padding 长度 = 512+34-5 = 541 (suffix_start_idx=5)
#   - system prompt 固定 "Describe the image...", suffix "<|im_end|>\n<|im_start|>assistant\n"
#   - MFA projector (Linear(12,1)) 和 2560→6144 投影都在 DiT 的 txtfusion/txtmlp, encoder 零可训练参数
#
# 单文件 safetensors 加载 (同 anima load_qwen3_text_encoder 模式):
# bundled config 目录 library/models/krea2_raw/configs/qwen3vl_4b/ + 单 safetensors。

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from safetensors import safe_open

from library.anima.text_strategies import (
    TextEncodingStrategy,
    TextEncoderOutputsCachingStrategy,
    TokenizeStrategy,
)
from library.io.cache import resolve_cache_path
from library.models.family_registry import (
    get_model_family_spec,
    validate_text_cache_metadata,
)
from library.log import setup_logging
from safetensors.torch import save_file as _save_safetensors

setup_logging()
logger = logging.getLogger(__name__)


# ---- 常量 (核实自 krea-ai/krea-2 encoder.py) ----

KREA2_TE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
KREA2_TE_BUNDLED_CONFIG_DIR = str(
    Path(__file__).resolve().parent / "configs" / "qwen3vl_4b"
)

KREA2_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_NUM_TXT_LAYERS = len(KREA2_SELECT_LAYERS)  # 12
KREA2_PREFIX_IDX = (
    34  # prompt_template_encode_start_idx: 切掉 system prompt 前 34 token
)
KREA2_SUFFIX_START_IDX = 5  # prompt_template_encode_suffix_start_idx (padding 公式用)
KREA2_MAX_LENGTH = 512  # user prompt 正文 max_length
# padding 长度 = max_length + prefix_idx - suffix_start_idx = 512+34-5 = 541
KREA2_PAD_LENGTH = KREA2_MAX_LENGTH + KREA2_PREFIX_IDX - KREA2_SUFFIX_START_IDX

KREA2_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
KREA2_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

KREA2_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = get_model_family_spec(
    "krea2_raw"
).text_cache.suffix
KREA2_CACHE_DTYPE = torch.bfloat16

_warned_legacy_variants_cache = False
_IN_MEMORY_VARIANTS_KEY = "__krea2_caption_variants__"


def _observe_cache_cast(name: str, source: torch.Tensor, stored: torch.Tensor) -> None:
    """Count non-finite Krea-2 cache tensors without touching the values.

    Krea-2 stores Qwen3-VL hidden states after a cast to ``KREA2_CACHE_DTYPE``; a value
    that is finite before the cast but overflows the narrower dtype is a *cast overflow*
    and must be distinguishable from a NaN/Inf the encoder already produced. Observation
    only: the cast, the dtype and the cache bytes are unchanged.
    """
    from library.training import numerical_observation

    stored_summary = numerical_observation.summarise(stored)
    if not stored_summary["nonfinite"]:
        return
    source_summary = numerical_observation.summarise(source)
    common = {
        "elements": stored_summary["elements"],
        "before_dtype": source.dtype,
        "after_dtype": KREA2_CACHE_DTYPE,
        "shape": tuple(stored.shape),
        "identity": f"krea2_te.{name}",
    }
    site = f"krea2_raw.strategy.{name}"
    if source_summary["nan"]:
        numerical_observation.record(
            site, numerical_observation.KIND_SOURCE_NAN,
            nonfinite=source_summary["nan"], **common,
        )
    if source_summary["inf"]:
        numerical_observation.record(
            site, numerical_observation.KIND_SOURCE_INF,
            nonfinite=source_summary["inf"], **common,
        )
    introduced = stored_summary["nonfinite"] - source_summary["nonfinite"]
    if introduced > 0:
        numerical_observation.record(
            site, numerical_observation.KIND_CAST_OVERFLOW,
            nonfinite=introduced, detail="finite before the storage cast", **common,
        )


class Krea2TokenizeStrategy(TokenizeStrategy):
    """Qwen3-VL ChatML tokenize (prompt pad 到 541 + cat suffix, 不裁剪)."""

    def __init__(
        self,
        max_length: int = KREA2_MAX_LENGTH,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_length = max_length + KREA2_PREFIX_IDX - KREA2_SUFFIX_START_IDX
        config_dir = tokenizer_path or KREA2_TE_BUNDLED_CONFIG_DIR
        from transformers import AutoTokenizer

        logger.info(f"load Krea-2 tokenizer from {config_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config_dir, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        if isinstance(text, str):
            text = [text]
        # ChatML: system prefix + user prompt, 一起 tokenize + pad 到 pad_length
        full = [KREA2_PROMPT_PREFIX + t for t in text]
        inputs = self.tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=self.pad_length,
            return_tensors="pt",
        )
        # suffix 单独 tokenize (不 pad), cat 到右侧
        suffix_inputs = self.tokenizer(
            [KREA2_PROMPT_SUFFIX] * len(text),
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = torch.cat([inputs["input_ids"], suffix_inputs["input_ids"]], dim=1)
        attn_mask = torch.cat(
            [inputs["attention_mask"], suffix_inputs["attention_mask"]], dim=1
        )
        return [input_ids, attn_mask]

    def tokenize_with_weights(self, text: Union[str, List[str]]) -> tuple:
        # Krea-2 首日不支持 weighted prompt (无 T5 weighted tokenize). 直走 tokenize.
        tokens = self.tokenize(text)
        weights = [torch.ones_like(tokens[0], dtype=torch.float32)]
        return tokens, weights


class Krea2TextEncodingStrategy(TextEncodingStrategy):
    """Qwen3-VL forward + 12 层 stack dim=2 + 切 prefix, 返回 (hiddens, mask)."""

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        qwen = models[0]  # Qwen3VLForConditionalGeneration, eval + frozen
        input_ids, attn_mask = tokens[0], tokens[1]
        device = next(qwen.parameters()).device
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            states = qwen(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
        # hidden_states tuple 长 37 (embedding + 36 层); 选 12 层 stack dim=2
        hiddens = torch.stack(
            [states.hidden_states[i] for i in KREA2_SELECT_LAYERS], dim=2
        )
        # 切掉 system prompt 前 34 token
        hiddens = hiddens[:, KREA2_PREFIX_IDX:]
        mask = attn_mask[:, KREA2_PREFIX_IDX:].bool()
        # 注: 不二次置零 padding (R1 定论: Krea-2 用 mask 屏蔽, 非 anima zero-sink)
        return [hiddens, mask]

    def apply_caption_dropout_inplace(
        self,
        caption_dropout_rates: torch.Tensor,
        *,
        hiddens: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Drop complete Krea text conditions per sample on the model device."""
        rates = caption_dropout_rates.to(hiddens.device, non_blocking=True).reshape(-1)
        if rates.numel() != hiddens.shape[0]:
            raise ValueError(
                "caption dropout rate batch does not match Krea text batch: "
                f"{rates.numel()} != {hiddens.shape[0]}"
            )
        drop_mask = torch.rand(rates.shape[0], device=hiddens.device) < rates
        hiddens[drop_mask] = 0
        mask[drop_mask] = False


class Krea2TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """{stem}_krea2_te.safetensors caching (suffix 隔离, 不污染 _anima_te)."""

    MODEL_FAMILY = "krea2_raw"
    CACHE_LABEL = "Krea"
    EXPECTED_HIDDEN_RANK = 3

    def __init__(
        self,
        cache_to_disk: bool = True,
        batch_size: int | str | None = None,
        skip_disk_cache_validity_check: bool = False,
        use_shuffled_caption_variants: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk=cache_to_disk,
            batch_size=batch_size,
            skip_disk_cache_validity_check=skip_disk_cache_validity_check,
        )
        self.use_shuffled_caption_variants = use_shuffled_caption_variants
        self._preprocess_batcher = None

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
        return resolve_cache_path(
            image_abs_path,
            KREA2_TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
            cache_dir=cache_dir,
            image_dir=image_dir,
        )

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        # Lazy per-tensor read via safe_open (same pattern as anima's
        # AnimaTextEncoderOutputsCachingStrategy.load_outputs_npz). Returns
        # torch tensors in the [hiddens, mask, caption_dropout_rate] order
        # that family.compute_noise_pred_and_target unpacks
        # (conds[0]=hiddens, conds[1]=mask; the trailing rate mirrors anima's
        # cache layout so split_cached_text_encoder_outputs can surface it as
        # batch["caption_dropout_rates"] instead of eating the mask).
        # bfloat16 hiddens have no numpy representation; the anima base class
        # signature is nominal — the real contract carries torch tensors.
        with safe_open(npz_path, framework="pt") as f:
            metadata = f.metadata()
            validate_text_cache_metadata(
                metadata, family=self.MODEL_FAMILY, path=npz_path
            )
            keys = set(f.keys())
            if "num_variants" in keys:
                num_variants = int(f.get_tensor("num_variants"))
                vi = self._select_variant_index(
                    num_variants,
                    multi_source="caption_multi_source" in keys,
                    v0_intact="v0_intact" in keys,
                    cache_path=npz_path,
                )
                hidden_key = f"hiddens_v{vi}"
                mask_key = f"mask_v{vi}"
            else:
                hidden_key = "hiddens"
                mask_key = "mask"
            self._validate_tensor_layout(
                f,
                hidden_key,
                mask_key,
                cache_path=npz_path,
                stamped=bool((metadata or {}).get("cache_schema")),
            )
            hiddens = f.get_tensor(hidden_key)
            mask = f.get_tensor(mask_key)
            caption_dropout_rate = (
                f.get_tensor("caption_dropout_rate")
                if "caption_dropout_rate" in keys
                else torch.tensor(0.0, dtype=torch.float32)
            )
        return [hiddens, mask, caption_dropout_rate]

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
            with safe_open(npz_path, framework="pt") as f:
                metadata = f.metadata()
                validate_text_cache_metadata(
                    metadata, family=self.MODEL_FAMILY, path=npz_path
                )
                keys = set(f.keys())
                has_variants = "num_variants" in keys
                num_variants = int(f.get_tensor("num_variants")) if has_variants else 0
                if has_variants and num_variants < 1:
                    return False
                if self.use_shuffled_caption_variants and not has_variants:
                    return False
                if expected_num_variants is not None:
                    if num_variants != expected_num_variants:
                        return False
                    if expected_multi_source is not None:
                        has_multi_source = "caption_multi_source" in keys
                        if has_multi_source != expected_multi_source:
                            return False
                    if expected_num_variants > 0:
                        if "v0_intact" not in keys:
                            return False
                        if expected_caption_shuffle_variants is not None:
                            if "caption_shuffle_variants" not in keys or int(
                                f.get_tensor("caption_shuffle_variants")
                            ) != int(expected_caption_shuffle_variants):
                                return False
                        if expected_caption_tag_dropout_rate is not None:
                            if "caption_tag_dropout_rate" not in keys:
                                return False
                            cached_rate = float(
                                f.get_tensor("caption_tag_dropout_rate")
                            )
                            if (
                                abs(cached_rate - expected_caption_tag_dropout_rate)
                                > 1e-7
                            ):
                                return False
                if "caption_dropout_rate" not in keys:
                    return False
                layouts = (
                    [(f"hiddens_v{vi}", f"mask_v{vi}") for vi in range(num_variants)]
                    if num_variants > 0
                    else [("hiddens", "mask")]
                )
                if not all(hidden in keys and mask in keys for hidden, mask in layouts):
                    return False
                for hidden, mask in layouts:
                    self._validate_tensor_layout(
                        f,
                        hidden,
                        mask,
                        cache_path=npz_path,
                        stamped=bool((metadata or {}).get("cache_schema")),
                    )
                rate_slice = f.get_slice("caption_dropout_rate")
                if rate_slice.get_shape() != [] or rate_slice.get_dtype() != "F32":
                    return False
            return True
        except Exception:
            return False

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
                f"{cache_path or 'memory'}: "
                f"{num_variants}"
            )
        if multi_source:
            return random.randint(0, num_variants - 1)
        if not self.use_shuffled_caption_variants:
            return 0
        if not v0_intact:
            global _warned_legacy_variants_cache
            if not _warned_legacy_variants_cache:
                logger.warning(
                    "Loaded a legacy %s multi-variant TE cache without "
                    "`v0_intact` (%s); sampling variants uniformly. "
                    "Re-run preprocess-te for 20%% pristine / 80%% "
                    "shuffled sampling.",
                    self.CACHE_LABEL,
                    cache_path,
                )
                _warned_legacy_variants_cache = True
            return random.randint(0, num_variants - 1)
        if num_variants == 1 or random.random() < 0.2:
            return 0
        return random.randint(1, num_variants - 1)

    def select_in_memory_outputs(self, outputs):
        """Resolve one caption variant from an in-memory cache on each sample read."""
        if not isinstance(outputs, dict) or not outputs.get(_IN_MEMORY_VARIANTS_KEY):
            return outputs
        num_variants = int(outputs["hiddens"].shape[0])
        vi = self._select_variant_index(
            num_variants,
            multi_source=bool(outputs["caption_multi_source"]),
            v0_intact=True,
        )
        return [
            outputs["hiddens"][vi],
            outputs["mask"][vi],
            outputs["caption_dropout_rate"],
        ]

    def _encode_captions(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        captions: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode flattened variants without multiplying the configured batch size."""
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
                    device, label=f"{self.MODEL_FAMILY}/text",
                )

            def encode(group):
                hiddens, mask = text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokenize_strategy.tokenize(group),
                )
                stored_hiddens = hiddens.detach().to(dtype=KREA2_CACHE_DTYPE, device="cpu")
                _observe_cache_cast("hiddens", hiddens.detach(), stored_hiddens)
                return (stored_hiddens,
                        mask.detach().to(dtype=torch.bool, device="cpu"))

            return self._preprocess_batcher.encode_all(captions, encode)
        batch_size = max(1, int(self.batch_size or len(captions) or 1))
        hidden_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        for start in range(0, len(captions), batch_size):
            caption_batch = captions[start : start + batch_size]
            hiddens, mask = text_encoding_strategy.encode_tokens(
                tokenize_strategy,
                models,
                tokenize_strategy.tokenize(caption_batch),
            )
            stored_hiddens = hiddens.detach().to(dtype=KREA2_CACHE_DTYPE, device="cpu")
            _observe_cache_cast("hiddens", hiddens.detach(), stored_hiddens)
            hidden_chunks.append(stored_hiddens)
            mask_chunks.append(mask.detach().to(dtype=torch.bool, device="cpu"))
        return torch.cat(hidden_chunks, dim=0), torch.cat(mask_chunks, dim=0)

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: List,
    ) -> None:
        """缓存 Krea-2 文本编码 (hiddens + mask) 到 safetensors.

        与 anima 的 _cache_batch_outputs_single 同构: 批量 encode 一组 caption,
        拆回 per-sample 写盘 (或塞 info.text_encoder_outputs 供 in-memory 使用).
        不二次置零 padding (R1 契约: mask 屏蔽即可).
        """
        variants_by_info: list[list[str]] = []
        variant_layout_by_info: list[bool] = []
        for info in batch:
            variants = list(getattr(info, "caption_variants", None) or [info.caption])
            variants_by_info.append(variants)
            variant_layout_by_info.append(
                bool(getattr(info, "cache_caption_variants", False))
                or len(variants) > 1
            )

        flattened_captions = [
            caption for variants in variants_by_info for caption in variants
        ]
        hiddens, mask = self._encode_captions(
            tokenize_strategy,
            models,
            text_encoding_strategy,
            flattened_captions,
        )

        offset = 0
        for info, variants, use_variant_layout in zip(
            batch,
            variants_by_info,
            variant_layout_by_info,
        ):
            caption_dropout_rate = torch.tensor(
                info.caption_dropout_rate, dtype=torch.float32
            )
            if use_variant_layout:
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
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if getattr(info, "caption_multi_source", False):
                    save_dict["caption_multi_source"] = torch.tensor(
                        1, dtype=torch.int8
                    )
                for vi in range(len(variants)):
                    save_dict[f"hiddens_v{vi}"] = hiddens[offset + vi].contiguous()
                    save_dict[f"mask_v{vi}"] = mask[offset + vi].contiguous()
            else:
                save_dict = {
                    "hiddens": hiddens[offset].contiguous(),
                    "mask": mask[offset].contiguous(),
                    "caption_dropout_rate": caption_dropout_rate,
                }
            if self.cache_to_disk:
                spec = get_model_family_spec(self.MODEL_FAMILY)
                _save_safetensors(
                    save_dict,
                    info.text_encoder_outputs_npz,
                    metadata=spec.text_cache.metadata(spec.name),
                )
            elif use_variant_layout:
                end = offset + len(variants)
                info.text_encoder_outputs = {
                    _IN_MEMORY_VARIANTS_KEY: True,
                    "hiddens": hiddens[offset:end].clone(),
                    "mask": mask[offset:end].clone(),
                    "caption_multi_source": bool(
                        getattr(info, "caption_multi_source", False)
                    ),
                    "caption_dropout_rate": caption_dropout_rate,
                }
            else:
                info.text_encoder_outputs = (
                    hiddens[offset].clone(),
                    mask[offset].clone(),
                    caption_dropout_rate,
                )
            offset += len(variants)


def load_krea2_text_encoder(
    te_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> tuple:
    """加载 Qwen3-VL-4B (单 safetensors + bundled config).

    返回 (model, tokenizer). model 是 Qwen3VLForConditionalGeneration,
    eval + frozen + no cache. visual 权重在但 text path 不激活.
    """
    import transformers

    te_path = str(te_path)
    logger.info(f"Loading Krea-2 Qwen3-VL text encoder from {te_path}")

    config = transformers.Qwen3VLConfig.from_pretrained(
        KREA2_TE_BUNDLED_CONFIG_DIR, local_files_only=True
    )
    model = transformers.Qwen3VLForConditionalGeneration(config)

    state_dict = load_safetensors(te_path, device="cpu")
    # 单文件权重 key 形如 "model.language_model.*", "model.visual.*";
    # Qwen3VLForConditionalGeneration 的 state_dict key 形如 "model.language_model.*"
    # ( transformers 会给顶层 model. 前缀 ). 去掉可能的 "model." 重复或保留.
    new_sd = {}
    for k, v in state_dict.items():
        # transformers Qwen3VLForConditionalGeneration 期望 "model." 前缀
        if k.startswith("model."):
            new_sd[k] = v
        else:
            new_sd[f"model.{k}"] = v
    info = model.load_state_dict(new_sd, strict=False)
    n_missing = len(info.missing_keys)
    n_unexpected = len(info.unexpected_keys)
    # visual 权重在, 但若 config 没 visual 字段会 missing; 阶段 1 只用 LM 部分,
    # 视觉 missing/unexpected 在容忍范围. 这里只 log 不硬断.
    logger.info(f"Qwen3-VL state dict: missing={n_missing}, unexpected={n_unexpected}")

    model.config.use_cache = False
    model = model.to(device, dtype=dtype).eval().requires_grad_(False)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        KREA2_TE_BUNDLED_CONFIG_DIR, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        f"Loaded Krea-2 Qwen3-VL. Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model, tokenizer


def load_safetensors(path: str, device: str = "cpu") -> dict:
    """单文件 safetensors 加载 (CPU, 不占 GPU)."""
    from safetensors.torch import load_file

    return load_file(path, device=device)
