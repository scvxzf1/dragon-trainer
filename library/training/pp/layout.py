"""Latent shape contract for the Anima pipeline-parallel boundary.

``Anima`` has two latent conventions and they are not interchangeable:

* the VAE cache, the dataloader and the loss composer use 4D ``(B, C, H, W)``;
* ``Anima.forward`` / ``forward_mini_train_dit`` use 5D ``(B, C, T=1, H, W)``
  where dim 2 is the *singleton* temporal axis.

Crossing that boundary with a bare ``squeeze()`` / ``squeeze(0)`` silently
removes the wrong axis as soon as a batch or channel dimension happens to be 1,
so every crossing in this package goes through :func:`latent_to_model_5d` /
:func:`model_pred_to_loss_4d`, which assert the rank first and always name the
axis.

The second contract here is the *wire* contract: ``PipelineStage`` allocates its
receive buffers once, from the metadata handed to the constructor, and then
reuses them for every microbatch.  A PP step therefore carries exactly one token
bucket (the 4032/4200 families of ``library.datasets.buckets``), never a mix.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .errors import (
    CODE_SHAPE_CONTRACT,
    CODE_TOKEN_BUCKET_MISMATCH,
    PipelineParallelError,
)

#: The singleton temporal axis of the Anima 5D latent contract.
TIME_AXIS = 2

#: ``TimestepEmbedding`` emits ``3 * model_channels`` AdaLN-LoRA values per token
#: (``models.py:1125``: ``nn.Linear(out_features, 3 * out_features)``).
ADALN_CHUNKS = 3

#: Token counts of the two constant-token families (``library/datasets/buckets.py``).
TOKEN_FAMILIES = {4032: "4032", 4200: "4200"}


def _require_rank(tensor, rank: int, *, what: str) -> None:
    if tensor.ndim != rank:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"{what} must be {rank}D, got shape {tuple(tensor.shape)}",
        )


def latent_to_model_5d(latent):
    """``(B, C, H, W)`` -> ``(B, C, 1, H, W)`` with an explicit ``unsqueeze(2)``."""

    _require_rank(latent, 4, what="training latent")
    return latent.unsqueeze(TIME_AXIS)


def model_pred_to_loss_4d(prediction):
    """``(B, C, T=1, H, W)`` -> ``(B, C, H, W)`` with an explicit ``squeeze(2)``.

    The temporal axis is asserted to be exactly 1 first; a bare ``squeeze()``
    would also drop a batch or channel axis of size 1.
    """

    _require_rank(prediction, 5, what="model prediction")
    if prediction.shape[TIME_AXIS] != 1:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            "model prediction temporal axis must be 1 to cross to the 4D loss "
            f"contract, got shape {tuple(prediction.shape)}",
        )
    return prediction.squeeze(TIME_AXIS)


def validate_latent_5d_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Validate a 5D latent shape and return it as a plain tuple."""

    shape = tuple(int(value) for value in shape)
    if len(shape) != 5:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"Anima PP latents must be 5D (B, C, T, H, W), got {shape}",
        )
    if shape[TIME_AXIS] != 1:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            "Anima PP v0 runs image latents only; the temporal axis must be 1, "
            f"got shape {shape}",
        )
    return shape


def boundary_hidden_shape(
    latent_5d_shape: Sequence[int],
    *,
    patch_spatial: int,
    patch_temporal: int,
    model_channels: int,
) -> tuple[int, ...]:
    """Shape of the tensor that crosses the PP boundary: the block grid.

    This mirrors ``PatchEmbed`` exactly: ``(B, C, T, H, W)`` ->
    ``(B, T // pt, H // ps, W // ps, model_channels)``.
    """

    batch, _channels, time, height, width = validate_latent_5d_shape(latent_5d_shape)
    patch_spatial = int(patch_spatial)
    patch_temporal = int(patch_temporal)
    if patch_spatial < 1 or patch_temporal < 1:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"patch sizes must be >= 1, got spatial={patch_spatial} temporal={patch_temporal}",
        )
    if time % patch_temporal or height % patch_spatial or width % patch_spatial:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"latent {tuple(latent_5d_shape)} is not divisible by patch sizes "
            f"(spatial={patch_spatial}, temporal={patch_temporal})",
        )
    return (
        batch,
        time // patch_temporal,
        height // patch_spatial,
        width // patch_spatial,
        int(model_channels),
    )


def prediction_shape(latent_5d_shape: Sequence[int]) -> tuple[int, ...]:
    """Shape of the 4D prediction the last stage hands to the loss."""

    batch, channels, _time, height, width = validate_latent_5d_shape(latent_5d_shape)
    return (batch, channels, height, width)


def timestep_embedding_shape(
    timesteps_shape: Sequence[int], model_channels: int
) -> tuple[int, ...]:
    """Shape of ``t_embedder`` output for a given ``timesteps`` shape.

    ``forward_mini_train_dit`` unsqueezes a 1D ``(B,)`` timestep to ``(B, 1)``
    before the embedder, so the modulation tensor is ``(B, T, D)``.
    """

    shape = tuple(int(value) for value in timesteps_shape)
    if len(shape) == 1:
        return (shape[0], 1, int(model_channels))
    if len(shape) == 2:
        return (shape[0], shape[1], int(model_channels))
    raise PipelineParallelError(
        CODE_SHAPE_CONTRACT,
        f"timesteps must be (B,) or (B, T), got {shape}",
    )


def adaln_lora_shape(
    timesteps_shape: Sequence[int], model_channels: int, *, chunks: int = ADALN_CHUNKS
) -> tuple[int, ...]:
    """Shape of the ``adaln_lora_B_T_3D`` modulation tensor.

    ``TimestepEmbedding`` returns ``3 * model_channels`` values per token, which
    ``Block`` reshapes into the three AdaLN chunks -- it is *not* ``(B, T, 3, D)``.
    """

    embedding = timestep_embedding_shape(timesteps_shape, model_channels)
    return (embedding[0], embedding[1], int(chunks) * int(model_channels))


def token_count(hidden_shape: Sequence[int]) -> int:
    """Number of DiT tokens in a boundary block grid."""

    shape = tuple(int(value) for value in hidden_shape)
    if len(shape) != 5:
        raise PipelineParallelError(
            CODE_SHAPE_CONTRACT,
            f"boundary hidden state must be 5D, got {shape}",
        )
    return shape[1] * shape[2] * shape[3]


def token_family(hidden_shape: Sequence[int]) -> str:
    """``"4032"`` / ``"4200"`` for a constant-token bucket, else ``"other:<n>"``."""

    count = token_count(hidden_shape)
    return TOKEN_FAMILIES.get(count, f"other:{count}")


def assert_uniform_token_bucket(hidden_shapes: Iterable[Sequence[int]]) -> int:
    """Fail closed unless every microbatch has the same boundary block grid.

    ``PipelineStage`` freezes its receive buffers from the first microbatch, so a
    mixed-bucket step would either trip ``validate_tensors_metadata`` or corrupt
    the stream.  Detecting it here turns a transport-level failure into a named
    contract violation.
    """

    shapes = [tuple(int(value) for value in shape) for shape in hidden_shapes]
    if not shapes:
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH, "no boundary shapes to validate"
        )
    first = shapes[0]
    for index, shape in enumerate(shapes[1:], start=1):
        if shape != first:
            raise PipelineParallelError(
                CODE_TOKEN_BUCKET_MISMATCH,
                "one PP step may only carry a single token bucket; microbatch 0 "
                f"has {first} (tokens={token_count(first)}, family={token_family(first)}) "
                f"but microbatch {index} has {shape} "
                f"(tokens={token_count(shape)}, family={token_family(shape)})",
            )
    return token_count(first)


def split_batch_shape(full_shape: Sequence[int], microbatches: int) -> tuple[int, ...]:
    """Per-microbatch shape for a full-batch shape, validating divisibility."""

    shape = tuple(int(value) for value in full_shape)
    microbatches = int(microbatches)
    if microbatches < 1:
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"microbatches must be >= 1, got {microbatches}",
        )
    if not shape or shape[0] % microbatches:
        raise PipelineParallelError(
            CODE_TOKEN_BUCKET_MISMATCH,
            f"batch {shape[0] if shape else '?'} is not divisible by "
            f"pipeline_parallel_microbatches={microbatches}; the PP wire buffers "
            "are fixed-shape, so every microbatch must hold the same batch size",
        )
    return (shape[0] // microbatches, *shape[1:])


__all__ = [
    "ADALN_CHUNKS",
    "TIME_AXIS",
    "TOKEN_FAMILIES",
    "adaln_lora_shape",
    "assert_uniform_token_bucket",
    "boundary_hidden_shape",
    "latent_to_model_5d",
    "model_pred_to_loss_4d",
    "prediction_shape",
    "split_batch_shape",
    "timestep_embedding_shape",
    "token_count",
    "token_family",
    "validate_latent_5d_shape",
]
