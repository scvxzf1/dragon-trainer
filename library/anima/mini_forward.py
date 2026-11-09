"""Split-forward numerics for :class:`library.anima.models.Anima`.

Why this module lives in ``library/anima/``
-------------------------------------------
Every rule here is a property of the **Anima DiT forward**, derived from
:meth:`library.anima.models.Anima.forward_mini_train_dit`:

* ``use_fp32`` is a *function of the compute dtype* (``models.py:2665``), never a
  caller-supplied flag;
* the RoPE cache is cast to the block compute dtype once (``models.py:2668-2674``);
* the post-block region adds the modulation-guidance final-weight term
  (``models.py:2778-2780``);
* the block stack runs inside the SDPA backend scope that
  ``_dynamic_seq_strict_marks`` selects (``models.py:2745-2765``).

They are *model* semantics, not training semantics: a single-GPU forward, an
inference path or a split (pipeline-parallel) forward all need the same answers.
Owning them under ``library/training/pp`` would make a training package the
authority for model behaviour and would force any non-PP consumer to import the
whole training PP stack.  The dependency direction is therefore

    library.training.pp  ->  library.anima.mini_forward  ->  torch

which is also the direction the rest of the repository uses (training depends on
models, never the reverse).

History: these functions were written inside
``scripts/experiments/anima_parallel/pipeline_parallel.py`` (task t9) and
imported from there by ``library/training/pp/numerics.py``.  That was a
``library`` -> ``scripts`` reverse dependency -- the only one in the repository --
and it made an installed ``anima_lora`` fail to import whenever ``scripts/`` was
absent.  Task t10 moved the authority here; the scripts module is now a consumer
that re-exports these names for its probe call sites.

The three region functions (:func:`prepare_block_inputs`,
:func:`run_local_blocks`, :func:`finish_model`) are the split of
``forward_mini_train_dit`` into pre-block / block-stack / post-block.  They are
deliberately co-located with the rules so the duplication with ``models.py`` is
visible in one place and cannot drift silently.

Capabilities the split forward cannot represent are rejected with a named,
fail-closed error instead of being ignored: pooled text modulation, flex
attention, native-flatten compile, and per-block modulation guidance on a sliced
stage.  A silently ignored capability would keep producing wrong numbers without
ever raising.

Known scope limits (documented, not silently dropped): ``fps``, ``h_offset`` /
``w_offset`` (tiled diffusion), ``t5_input_ids`` (LLM adapter) and
``pooled_text_override`` are not parameters of the split forward, so a caller
cannot request them here at all.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

import torch

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
    "normalize_block_kwargs",
    "prepare_block_inputs",
    "resolve_block_compute_dtype",
    "resolve_use_fp32",
    "run_local_blocks",
]

# --- fail-closed capability codes -------------------------------------------
#
# These are the machine-readable codes the PP contract uses for "the split
# forward cannot reproduce this capability".  Keep them stable: they surface in
# preflight output and in test assertions.

PP_POOLED_TEXT_MODULATION_UNSUPPORTED = "pp_pooled_text_modulation_unsupported"
PP_FLEX_ATTENTION_UNSUPPORTED = "pp_flex_attention_unsupported"
PP_NATIVE_FLATTEN_UNSUPPORTED = "pp_native_flatten_unsupported"
PP_MOD_GUIDANCE_UNSUPPORTED = "pp_mod_guidance_unsupported"
PP_MOD_GUIDANCE_CONTRACT_UNSUPPORTED = "pp_mod_guidance_contract_unsupported"


class AnimaPipelineUnsupported(RuntimeError):
    """Named fail-closed error for a capability the split forward cannot own."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --- dtype rules (pure; no model needed) -------------------------------------


def resolve_use_fp32(dtype: torch.dtype) -> bool:
    """Whether the block / final layers must promote to fp32 (models.py:2665).

    ``use_fp32`` is not a tuning knob: the authority derives it from the compute
    dtype, so any caller-supplied value is a bug.  fp16 is the only dtype whose
    AdaLN modulation and residual stream need the fp32 path.
    """

    return dtype == torch.float16


def resolve_block_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Dtype the RoPE cache must be materialised in (models.py:2668-2674)."""

    return torch.float32 if resolve_use_fp32(dtype) else dtype


def cast_rope_for_compute(
    rope: tuple[torch.Tensor, torch.Tensor] | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Cast a ``(cos, sin)`` RoPE pair to the block compute dtype, once."""

    if rope is None:
        return None
    compute_dtype = resolve_block_compute_dtype(dtype)
    return tuple(value.to(compute_dtype) for value in rope)


def normalize_block_kwargs(
    dtype: torch.dtype,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Force the dtype-derived block kwargs to their authoritative values.

    ``use_fp32`` and the RoPE dtype are functions of ``dtype``, so a caller that
    passes a stale value (for example a stage wrapper that hard-coded
    ``use_fp32=False``) cannot make the block stack run in the wrong precision.
    """

    normalized = dict(kwargs)
    normalized["use_fp32"] = resolve_use_fp32(dtype)
    rope = normalized.get("rope_cos_sin")
    if rope is not None:
        normalized["rope_cos_sin"] = cast_rope_for_compute(rope, dtype)
    return normalized


# --- capability guards -------------------------------------------------------


def _attn_mode(model) -> str:
    return str(getattr(model, "attn_mode", "") or "").strip().lower()


def assert_mini_forward_supported(model) -> None:
    """Reject capabilities the split forward cannot reproduce.

    Raising here is the whole point: ignoring one of these would keep training
    and silently produce numbers that differ from the single-GPU path.
    """

    if bool(getattr(model, "enable_pooled_text_modulation", False)):
        raise AnimaPipelineUnsupported(
            PP_POOLED_TEXT_MODULATION_UNSUPPORTED,
            "the PP mini forward does not inject the pooled text embedding into "
            "the modulation path (models.py:2688-2699); set "
            "enable_pooled_text_modulation=false for pipeline_parallel",
        )
    if _attn_mode(model) == "flex":
        raise AnimaPipelineUnsupported(
            PP_FLEX_ATTENTION_UNSUPPORTED,
            "the PP mini forward does not build the per-sample flex "
            "cross-attention BlockMask (models.py:2717-2741); use "
            "attn_mode=flash or torch for pipeline_parallel",
        )
    if bool(getattr(model, "_native_flatten", False)):
        raise AnimaPipelineUnsupported(
            PP_NATIVE_FLATTEN_UNSUPPORTED,
            "the PP mini forward exchanges the unflattened 5D block grid; the "
            "native-flatten fake-5D shape cannot cross a stage boundary",
        )


def assert_mod_guidance_indexable(model) -> None:
    """Per-block guidance indexes *global* block ids, so a slice cannot use it.

    ``_run_blocks`` enumerates ``self.blocks`` and looks up
    ``_mod_guidance_schedule[block_idx]`` with the local index.  On a stage that
    owns a sliced ``blocks`` list the local index no longer matches the global
    one, so an active schedule would be applied to the wrong blocks.  Detecting
    the slice by comparing the schedule length to the local block count keeps
    this check self-contained (no need to know the stage's block offset).
    """

    delta = getattr(model, "_mod_guidance_delta", None)
    final_w = getattr(model, "_mod_guidance_final_w", None)
    schedule = getattr(model, "_mod_guidance_schedule", None)
    if delta is None or final_w is None or schedule is None:
        raise AnimaPipelineUnsupported(
            PP_MOD_GUIDANCE_CONTRACT_UNSUPPORTED,
            "model does not expose _mod_guidance_delta / _mod_guidance_final_w / "
            "_mod_guidance_schedule, so the PP mini forward cannot reproduce the "
            "modulation-guidance post-block term (models.py:2778-2780)",
        )
    if not (bool(delta.count_nonzero()) or bool(final_w.count_nonzero())):
        return
    if int(schedule.numel()) != len(model.blocks):
        raise AnimaPipelineUnsupported(
            PP_MOD_GUIDANCE_UNSUPPORTED,
            "modulation guidance is active but this stage owns "
            f"{len(model.blocks)} blocks while the per-block schedule holds "
            f"{int(schedule.numel())} entries; _run_blocks indexes the schedule "
            "with local block ids, so the global schedule would be misapplied",
        )


def block_sdpa_scope(model):
    """Same SDPA backend restriction the authority applies (models.py:2745-2765).

    PyTorch's generic SDPA lowering can add a ``seq % 8`` guard; the strict
    per-band path may hold arbitrary native token counts, so the authority pins
    the alignment-tolerant kernels.  Reproducing the scope here keeps the split
    forward on the same backend set as the single-GPU path.
    """

    if not bool(getattr(model, "_dynamic_seq_strict_marks", False)):
        return nullcontext()
    if _attn_mode(model) != "torch":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )
    except (ImportError, AttributeError, RuntimeError):
        # Older builds may not expose scoped SDPA selection. Preserve the
        # historical dispatcher there rather than failing startup.
        return nullcontext()


# --- the split forward -------------------------------------------------------


def prepare_block_inputs(model, x, timesteps, context, padding_mask):
    """Pre-block region: embed latents, RoPE, timestep embedding, attn params."""

    from networks import attention_dispatch

    assert_mini_forward_supported(model)
    hidden, rope = model.prepare_embedded_sequence(x, padding_mask=padding_mask)
    rope = cast_rope_for_compute(rope, hidden.dtype)
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    embedding, adaln = model.t_embedder(timesteps)
    embedding = model.t_embedding_norm(embedding)
    params = attention_dispatch.AttentionParams.create_attention_params(
        model.attn_mode,
        model.attn_softmax_scale,
        v100_flash_stability=model.v100_flash_stability,
        debug_finite_checks=model.debug_finite_checks,
    )
    kwargs = {
        "rope_cos_sin": rope,
        "adaln_lora_B_T_3D": adaln,
        "use_fp32": resolve_use_fp32(hidden.dtype),
    }
    return hidden, embedding, params, kwargs


def run_local_blocks(model, hidden, embedding, context, params, kwargs):
    """Run this stage's block stack with the authoritative dtype rules applied."""

    assert_mini_forward_supported(model)
    block_kwargs = normalize_block_kwargs(hidden.dtype, kwargs)
    with block_sdpa_scope(model):
        return model._run_blocks(hidden, embedding, context, params, **block_kwargs)


def finish_model(model, hidden, embedding, adaln, *, use_fp32: bool | None = None):
    """Post-block region: modulation-guidance final term, final layer, unpatchify.

    ``use_fp32`` is the pipeline-wide decision
    (:class:`library.training.pp.contract.PpComputePolicy`).  When it is ``None``
    the flag is inferred from ``hidden.dtype``, which is only correct for a
    single-process call: with ``use_fp32`` true the block stack promotes the
    residual stream, so ``hidden`` is **fp32 even though the stack ran fp16** and
    the inference returns ``False`` where ``models.py:2665`` decided ``True``.  A
    PP stage must therefore pass the flag explicitly rather than let the last
    stage re-derive it from the tensor it received.
    """

    assert_mini_forward_supported(model)
    assert_mod_guidance_indexable(model)
    if use_fp32 is None:
        use_fp32 = resolve_use_fp32(hidden.dtype)
    t_emb_final = embedding + (
        model._mod_guidance_final_w * model._mod_guidance_delta
    ).unsqueeze(1)
    projected = model.final_layer(
        hidden,
        t_emb_final,
        adaln_lora_B_T_3D=adaln,
        use_fp32=use_fp32,
    )
    return model.unpatchify(projected)
