"""LoRA target discovery helpers.

Keep this module limited to pure candidate collection. Build orchestration lives
in ``builders.py``; module-class selection and constructor kwargs live in
``module_builders.py``; runtime buffer wiring lives in ``routing_state.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Pattern, Sequence

import torch

_BLOCK_IDX_RE = re.compile(r"(?:^|\.)(?:blocks|layers|transformer_blocks)\.(\d+)\.")


@dataclass(frozen=True)
class LoRATargetCandidate:
    lora_name: str
    child_module: Optional[torch.nn.Module]
    dim: Optional[int]
    alpha: Optional[float]
    original_name: str
    skipped: bool


def compile_lora_target_patterns(
    patterns: Optional[Sequence[str]],
    *,
    logger: Optional[logging.Logger] = None,
) -> list[Pattern[str]]:
    re_patterns: list[Pattern[str]] = []
    if patterns is None:
        return re_patterns
    for pattern in patterns:
        try:
            re_patterns.append(re.compile(pattern))
        except re.error as exc:
            if logger is not None:
                logger.error(f"Invalid pattern '{pattern}': {exc}")
    return re_patterns


def collect_lora_target_candidates(
    *,
    root_module: torch.nn.Module,
    prefix: str,
    target_replace_modules: Optional[Sequence[str]],
    exclude_patterns: Sequence[Pattern[str]],
    include_patterns: Sequence[Pattern[str]],
    is_unet: bool,
    layer_start: Optional[int],
    layer_end: Optional[int],
    modules_dim: Optional[dict[str, int]],
    modules_alpha: Optional[dict[str, float]],
    reg_dims: Optional[dict[str, int]],
    reg_alphas: Optional[dict[str, float]] = None,
    default_dim: Optional[int],
    lora_dim: int,
    alpha: float,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> list[LoRATargetCandidate]:
    candidates: list[LoRATargetCandidate] = []
    for name, module in root_module.named_modules():
        if (
            target_replace_modules is None
            or module.__class__.__name__ in target_replace_modules
        ):
            if target_replace_modules is None:
                module = root_module

            for child_name, child_module in module.named_modules():
                is_linear = isinstance(child_module, torch.nn.Linear)
                is_conv2d = isinstance(child_module, torch.nn.Conv2d)
                is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                if is_linear or is_conv2d:
                    original_name = (name + "." if name else "") + child_name
                    original_name = original_name.replace("_orig_mod.", "")
                    lora_name = f"{prefix}.{original_name}".replace(".", "_")

                    excluded = any(
                        pattern.fullmatch(original_name)
                        for pattern in exclude_patterns
                    )
                    included = any(
                        pattern.fullmatch(original_name)
                        for pattern in include_patterns
                    )
                    if excluded and not included:
                        if verbose and logger is not None:
                            logger.info(f"exclude: {original_name}")
                        continue

                    if is_unet and (layer_start is not None or layer_end is not None):
                        block_match = _BLOCK_IDX_RE.match(original_name)
                        if block_match:
                            block_idx = int(block_match.group(1))
                            if layer_start is not None and block_idx < layer_start:
                                if verbose and logger is not None:
                                    logger.info(
                                        f"layer_range exclude: {original_name} "
                                        f"(block {block_idx} < {layer_start})"
                                    )
                                continue
                            if layer_end is not None and block_idx >= layer_end:
                                if verbose and logger is not None:
                                    logger.info(
                                        f"layer_range exclude: {original_name} "
                                        f"(block {block_idx} >= {layer_end})"
                                    )
                                continue

                    dim = None
                    alpha_val = None

                    if modules_dim is not None:
                        if lora_name in modules_dim:
                            dim = modules_dim[lora_name]
                            if modules_alpha is None:
                                raise KeyError(lora_name)
                            alpha_val = modules_alpha[lora_name]
                    else:
                        if reg_dims is not None:
                            for reg, d in reg_dims.items():
                                if re.fullmatch(reg, original_name):
                                    dim = d
                                    alpha_val = alpha
                                    if logger is not None:
                                        logger.info(
                                            f"Module {original_name} matched "
                                            f"with regex '{reg}' -> dim: {dim}"
                                        )
                                    break
                        if dim is None and (is_linear or is_conv2d_1x1):
                            dim = default_dim if default_dim is not None else lora_dim
                            alpha_val = alpha
                        # Per-pattern alpha override (completes the reg_dims /
                        # reg_lrs trio). Independent of reg_dims: applies whether
                        # the dim came from a reg_dims match or the network
                        # default — e.g. a scale-preserving alpha on adaln
                        # modules built at their own rank.
                        if reg_alphas is not None and dim:
                            for reg, a in reg_alphas.items():
                                if re.fullmatch(reg, original_name):
                                    alpha_val = a
                                    if logger is not None:
                                        logger.debug(
                                            f"Module {original_name} matched "
                                            f"with regex '{reg}' -> alpha: {a}"
                                        )
                                    break

                    if dim is None or dim == 0:
                        if is_linear or is_conv2d_1x1:
                            candidates.append(
                                LoRATargetCandidate(
                                    lora_name=lora_name,
                                    child_module=None,
                                    dim=None,
                                    alpha=None,
                                    original_name=original_name,
                                    skipped=True,
                                )
                            )
                        continue

                    candidates.append(
                        LoRATargetCandidate(
                            lora_name=lora_name,
                            child_module=child_module,
                            dim=dim,
                            alpha=alpha_val,
                            original_name=original_name,
                            skipped=False,
                        )
                    )

            if target_replace_modules is None:
                break

    return candidates
