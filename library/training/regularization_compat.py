"""Shared no-dataset regularization (prior preservation / DOP / mask prior) validator.

Consumed by both CLI ``verify_training_args`` and WebUI
``_check_no_dataset_regularization_config`` so their validation conditions
cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from library.config.normalize import parse_bool


@dataclass(frozen=True)
class RegularizationCompatIssue:
    code: str
    key: str
    cli_message: str
    web_message: str


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _nonneg_float(value: Any) -> float:
    try:
        parsed = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0.0 else 0.0


def validate_no_dataset_regularization(
    config: Mapping[str, Any] | object,
) -> list[RegularizationCompatIssue]:
    prior_weight = _nonneg_float(_get(config, "prior_preservation_weight", 0.0))
    mask_weight = _nonneg_float(_get(config, "inverted_mask_prior_weight", 0.0))
    blank_enabled = parse_bool(_get(config, "blank_prompt_preservation", False))
    dop_trigger = str(_get(config, "diff_output_preservation_trigger", "") or "").strip()
    dop_class = str(_get(config, "diff_output_preservation_class", "") or "").strip()
    use_text_cache = parse_bool(_get(config, "use_text_cache", False))
    cache_llm_adapter_outputs = parse_bool(
        _get(config, "cache_llm_adapter_outputs", False)
    )

    issues: list[RegularizationCompatIssue] = []
    if prior_weight > 0.0 and dop_trigger and not dop_class and not blank_enabled:
        issues.append(
            RegularizationCompatIssue(
                code="dop_missing_class",
                key="diff_output_preservation_class",
                cli_message=(
                    "prior_preservation_weight requires blank_prompt_preservation=true or "
                    "diff_output_preservation_class to select a prior-preservation mode."
                ),
                web_message=(
                    "DOP 已填写触发词，但未填写类提示；请填写 woman / character / style "
                    "等类提示，或关闭 prior_preservation_weight。"
                ),
            )
        )
    elif prior_weight > 0.0 and not (blank_enabled or dop_class):
        issues.append(
            RegularizationCompatIssue(
                code="prior_missing_mode",
                key="prior_preservation_weight",
                cli_message=(
                    "prior_preservation_weight requires blank_prompt_preservation=true or "
                    "diff_output_preservation_class to select a prior-preservation mode."
                ),
                web_message=(
                    "prior_preservation_weight 大于 0 时，需要开启 "
                    "blank_prompt_preservation 或填写 DOP 类提示。"
                ),
            )
        )

    if prior_weight > 0.0 and blank_enabled and dop_class:
        issues.append(
            RegularizationCompatIssue(
                code="prior_mode_mutex",
                key="blank_prompt_preservation",
                cli_message=(
                    "blank_prompt_preservation and diff_output_preservation_class "
                    "are mutually exclusive."
                ),
                web_message="blank_prompt_preservation 不能和 DOP 类提示同时使用。",
            )
        )

    if (prior_weight > 0.0 or mask_weight > 0.0) and not use_text_cache:
        cli_msg = (
            "prior preservation requires use_text_cache=true so prior text "
            "conditions are cached before the text encoder is released."
            if prior_weight > 0.0
            else "inverted_mask_prior requires use_text_cache=true so the base "
            "reference forward can reuse cached text conditions."
        )
        issues.append(
            RegularizationCompatIssue(
                code="regularization_requires_text_cache",
                key="use_text_cache",
                cli_message=cli_msg,
                web_message="无数据集正则化需要 use_text_cache=true；请开启文本缓存后重新预处理。",
            )
        )

    prior_needs_adapter_cache = prior_weight > 0.0 and (blank_enabled or bool(dop_class))
    if (prior_needs_adapter_cache or mask_weight > 0.0) and not cache_llm_adapter_outputs:
        cli_msg = (
            "prior preservation requires cache_llm_adapter_outputs=true so training "
            "batches carry max-padded crossattn_emb without re-running the text encoder."
            if prior_needs_adapter_cache
            else "inverted_mask_prior requires cache_llm_adapter_outputs=true so training "
            "batches carry max-padded crossattn_emb for the adapter-disabled base forward."
        )
        issues.append(
            RegularizationCompatIssue(
                code="regularization_requires_adapter_cache",
                key="cache_llm_adapter_outputs",
                cli_message=cli_msg,
                web_message=(
                    "无数据集正则化需要 cache_llm_adapter_outputs=true；"
                    "请开启 LLM adapter 输出缓存后重新预处理。"
                ),
            )
        )
    return issues
