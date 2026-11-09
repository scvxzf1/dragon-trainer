"""Two-stage, one-microbatch Anima pipeline probe (consumer module).

The split-forward numerics used to be defined *here*.  Task t10 moved the
authority into :mod:`library.anima.mini_forward`, because those rules are
properties of the Anima DiT forward (``models.py:2665`` / ``2668-2674`` /
``2778-2780`` / ``2745-2765``) and ``library/training/pp/numerics.py`` must not
import from ``scripts`` -- a ``library`` -> ``scripts`` reverse dependency breaks
the package boundary and makes an installed ``anima_lora`` fail to import when
``scripts/`` is absent.

This module now only:

1. re-exports the promoted names so the existing probe call sites
   (``stages.py``, ``anima_parallel_probe.py``) and the t9/t3 test suites keep
   working unchanged, and
2. keeps :class:`PipelineCommunication`, which is experiment-only measurement
   plumbing (it ``cuda.synchronize``s around every transfer and therefore must
   not be mistaken for a production communication primitive).

Do not add numerics back here: the single authority is
:mod:`library.anima.mini_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from library.anima.mini_forward import (
    PP_FLEX_ATTENTION_UNSUPPORTED,
    PP_MOD_GUIDANCE_CONTRACT_UNSUPPORTED,
    PP_MOD_GUIDANCE_UNSUPPORTED,
    PP_NATIVE_FLATTEN_UNSUPPORTED,
    PP_POOLED_TEXT_MODULATION_UNSUPPORTED,
    AnimaPipelineUnsupported,
    assert_mini_forward_supported,
    assert_mod_guidance_indexable,
    block_sdpa_scope,
    cast_rope_for_compute,
    finish_model,
    normalize_block_kwargs,
    prepare_block_inputs,
    resolve_block_compute_dtype,
    resolve_use_fp32,
    run_local_blocks,
)

__all__ = [
    "PP_FLEX_ATTENTION_UNSUPPORTED",
    "PP_MOD_GUIDANCE_CONTRACT_UNSUPPORTED",
    "PP_MOD_GUIDANCE_UNSUPPORTED",
    "PP_NATIVE_FLATTEN_UNSUPPORTED",
    "PP_POOLED_TEXT_MODULATION_UNSUPPORTED",
    "AnimaPipelineUnsupported",
    "PipelineCommunication",
    "assert_mini_forward_supported",
    "assert_mod_guidance_indexable",
    "block_sdpa_scope",
    "cast_rope_for_compute",
    "finish_model",
    "normalize_block_kwargs",
    "prepare_block_inputs",
    "resolve_block_compute_dtype",
    "resolve_use_fp32",
    "run_local_blocks",
]


@dataclass
class PipelineCommunication:
    payload_bytes: int = 0
    calls: int = 0
    seconds: float = 0.0

    def transfer(self, fn, tensor: torch.Tensor) -> None:
        torch.cuda.synchronize(tensor.device)
        start = perf_counter()
        fn(tensor)
        torch.cuda.synchronize(tensor.device)
        self.seconds += perf_counter() - start
        self.payload_bytes += tensor.numel() * tensor.element_size()
        self.calls += 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "payload_bytes_per_rank": self.payload_bytes,
            "wire_bytes_per_rank": self.payload_bytes,
            "collective_calls": self.calls,
            "quantize_seconds": 0.0,
            "collective_seconds": self.seconds,
            "dequantize_seconds": 0.0,
            "communication_seconds": self.seconds,
        }
