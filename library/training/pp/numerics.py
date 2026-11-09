"""PP-side adaptation of the authoritative split-forward numerics.

The numerics themselves live in :mod:`library.anima.mini_forward`, because they
are properties of the Anima DiT forward (``models.py:2665`` / ``2668-2674`` /
``2778-2780`` / ``2745-2765``), not of pipeline parallelism.  That module is the
single authority for:

* ``use_fp32`` -- a *function of the compute dtype*, never a caller flag;
* the RoPE cache dtype;
* the SDPA backend scope selected by ``_dynamic_seq_strict_marks``;
* the fail-closed capability guards (pooled text modulation / flex attention /
  native flatten / modulation guidance on a sliced stage).

This module re-exports those names (so ``library.training.pp`` consumers have one
import site) and adds the three things that are genuinely PP-specific:

* :func:`stage_block_kwargs` -- rename ``adaln_lora_B_T_3D`` to
  ``adaln_lora``, the keyword :class:`library.anima.pipeline_parallel.AnimaBlockStage`
  speaks, *without* re-deriving the dtype rules;
* :func:`run_stage_blocks` -- run a contiguous block *slice* under the same dtype
  and SDPA rules, with the ``requires_grad`` normalization the full stack does;
* :func:`mod_guidance_probe` -- a ``model``-shaped view that carries the stage
  slice, because the placement builder is non-destructive and therefore keeps the
  full ``blocks`` list on the production model.

Import direction is ``library.training.pp`` -> ``library.anima.mini_forward`` ->
``torch``.  Nothing under ``library/`` may import from ``scripts/``; that reverse
dependency existed here until task t10 and broke the package boundary (an
installed ``anima_lora`` failed to import whenever ``scripts/`` was absent).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
    "assert_mini_forward_supported",
    "assert_mod_guidance_indexable",
    "block_sdpa_scope",
    "cast_rope_for_compute",
    "finish_model",
    "mod_guidance_probe",
    "normalize_block_kwargs",
    "prepare_block_inputs",
    "resolve_block_compute_dtype",
    "resolve_use_fp32",
    "run_local_blocks",
    "run_stage_blocks",
    "stage_block_kwargs",
]


def stage_block_kwargs(
    compute_dtype,
    *,
    rope_cos_sin,
    adaln_lora,
) -> dict[str, Any]:
    """Authoritative block kwargs for a *sliced* stage block stack.

    ``compute_dtype`` is the **pre-block hidden** dtype of the pipeline (the value
    :class:`~library.training.pp.contract.PpComputePolicy` carries), never the
    dtype of the tensor this stage received.  Under fp16 the block stack promotes
    the residual stream (``models.py:1519``), so the boundary tensor is fp32 while
    ``use_fp32`` is still true; deriving the flag from the boundary would return
    ``False`` and run the tail in fp16 (T8: ``8.665e4 > 65504`` -> NaN at block
    32).  The parameter was called ``hidden_dtype`` and was fed the boundary dtype
    -- that is exactly the D1 bug, so the name now says which dtype it must be.

    ``normalize_block_kwargs`` speaks the ``_run_blocks`` keyword names
    (``adaln_lora_B_T_3D``); :class:`AnimaBlockStage` speaks ``adaln_lora``.
    Renaming the two entries is the whole adaptation -- ``use_fp32`` and the RoPE
    dtype still come from the single authority, so a caller cannot pass a stale
    ``use_fp32`` here.
    """

    normalized = normalize_block_kwargs(
        compute_dtype,
        {"rope_cos_sin": rope_cos_sin, "adaln_lora_B_T_3D": adaln_lora},
    )
    return {
        "rope_cos_sin": normalized["rope_cos_sin"],
        "adaln_lora": normalized["adaln_lora_B_T_3D"],
        "use_fp32": normalized["use_fp32"],
    }


def run_stage_blocks(
    block_stage,
    model,
    hidden,
    embedding,
    crossattn_emb,
    attn_params,
    *,
    compute_dtype,
    rope_cos_sin,
    adaln_lora,
    block_embeddings=None,
):
    """Run this stage's block slice with the authoritative dtype/SDPA rules.

    ``block_stage`` is an :class:`AnimaBlockStage` (a contiguous slice produced by
    ``build_anima_block_stage``), so the block loop itself is not re-implemented
    here.  ``model`` is only read for the SDPA scope decision.

    ``_run_blocks`` normalizes ``requires_grad`` once at stack entry
    (``models.py:2545``); the slice needs the same normalization or the first
    stage's frozen patch-embed output would not carry a gradient back across the
    boundary.

    ``compute_dtype`` is required and keyword-only on purpose: it is the pipeline's
    pre-block dtype, not ``hidden.dtype`` (the received boundary tensor).  See
    :func:`stage_block_kwargs`.
    """

    kwargs = stage_block_kwargs(
        compute_dtype, rope_cos_sin=rope_cos_sin, adaln_lora=adaln_lora
    )
    if block_embeddings is None:
        with block_sdpa_scope(model):
            return block_stage(
                hidden.requires_grad_(),
                embedding,
                crossattn_emb,
                attn_params,
                **kwargs,
            )
    with block_sdpa_scope(model):
        return block_stage(
            hidden.requires_grad_(),
            embedding,
            crossattn_emb,
            attn_params,
            block_embeddings=block_embeddings,
            **kwargs,
        )


def mod_guidance_probe(model, blocks) -> Any:
    """A ``model``-shaped view whose ``blocks`` is the *stage slice*.

    ``assert_mod_guidance_indexable`` compares ``len(model.blocks)`` against the
    per-block schedule length to detect a sliced stage.  The production model
    keeps all blocks (the placement builder is non-destructive), so the probe has
    to carry the slice explicitly or the check would pass on a stage that cannot
    honour the global schedule.
    """

    return SimpleNamespace(
        blocks=blocks,
        _mod_guidance_delta=getattr(model, "_mod_guidance_delta", None),
        _mod_guidance_final_w=getattr(model, "_mod_guidance_final_w", None),
        _mod_guidance_schedule=getattr(model, "_mod_guidance_schedule", None),
    )
