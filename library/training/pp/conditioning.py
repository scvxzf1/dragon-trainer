"""Stage-local Anima conditioning for the PP split.

Why the conditioning is rebuilt on each stage
---------------------------------------------
In the single-process forward the modulation tensors (``t_embedding`` /
``adaln_lora``) and the RoPE cache are cheap, *frozen* products of
``timesteps`` and the latent geometry.  Shipping them across the pipeline
boundary would put tensors on the wire that can never carry a useful gradient,
and torch 2.10's ``PipelineStage`` assumes every received input needs one (see
:class:`library.training.pp.stages.FrozenBoundaryPipelineStage`).  PP v0
therefore rebuilds them locally on whichever stage needs them:

* ``timesteps`` is a per-microbatch *keyword* argument that the schedule splits
  and hands to every stage, so no collective is involved;
* RoPE is a pure function of the block-grid shape (``models.py:2385-2393``) and
  is cached by ``(T, H, W, fps)`` inside ``pos_embedder``, so both stages derive
  bit-identical values without talking to each other.

Every helper here is a thin adapter over the authoritative split forward in
:mod:`library.training.pp.numerics`; none of them re-derives a dtype rule.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .errors import CODE_STAGE_CONTRACT, PipelineParallelError


def attention_params(model):
    """``AttentionParams`` for this model, exactly as the authority builds them."""

    from networks import attention_dispatch

    return attention_dispatch.AttentionParams.create_attention_params(
        model.attn_mode,
        model.attn_softmax_scale,
        v100_flash_stability=model.v100_flash_stability,
        debug_finite_checks=model.debug_finite_checks,
    )


def rope_for_shape(
    model,
    hidden_shape,
    *,
    fps: Optional[Tensor] = None,
    h_offset: int = 0,
    w_offset: int = 0,
) -> tuple[Tensor, Tensor]:
    """RoPE ``(cos, sin)`` for a block-grid shape, with no autograd edge.

    ``prepare_embedded_sequence`` calls ``pos_embedder(x, fps)``, and
    ``VideoPositionEmb.forward`` only reads ``x.shape``.  Calling the generator
    with the shape directly keeps the *received* activation (which requires grad
    on the later stage) out of the positional-embedding graph, so no gradient can
    leak into a frozen RoPE table.
    """

    embedder = getattr(model, "pos_embedder", None)
    if embedder is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "model does not expose pos_embedder, so the PP stage cannot rebuild "
            "the RoPE cache locally",
        )
    shape = torch.Size(tuple(int(value) for value in hidden_shape))
    if h_offset or w_offset:
        return embedder.generate_embeddings_with_offset(
            shape, h_offset=h_offset, w_offset=w_offset, fps=fps
        )
    return embedder.generate_embeddings(shape, fps=fps)


def timestep_modulation(model, timesteps) -> tuple[Tensor, Optional[Tensor]]:
    """``(t_embedding, adaln_lora)`` — the pre-block modulation tensors.

    Mirrors ``models.py:2676-2679``.  The pooled-text branch right after it
    (``models.py:2688-2699``) is unreachable here because
    ``assert_mini_forward_supported`` rejects
    ``enable_pooled_text_modulation`` for PP.
    """

    embedder = getattr(model, "t_embedder", None)
    norm = getattr(model, "t_embedding_norm", None)
    if embedder is None or norm is None:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            "model does not expose t_embedder / t_embedding_norm, so the PP stage "
            "cannot rebuild the timestep modulation locally",
        )
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    if timesteps.ndim != 2:
        raise PipelineParallelError(
            CODE_STAGE_CONTRACT,
            f"timesteps must be (B,) or (B, T), got shape {tuple(timesteps.shape)}",
        )
    embedding, adaln_lora = embedder(timesteps)
    return norm(embedding), adaln_lora


__all__ = ["attention_params", "rope_for_shape", "timestep_modulation"]
