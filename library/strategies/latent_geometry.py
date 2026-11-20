"""Geometry validation for image latent caches."""

from collections.abc import Sequence

from library.models.latent_space import LatentSpaceSpec


def validate_latent_shape(
    shape: Sequence[int],
    bucket_reso: tuple[int, int],
    spec: LatentSpaceSpec,
    *,
    label: str,
) -> None:
    stride = spec.vae_spatial_compression
    width, height = bucket_reso
    if min(width, height) <= 0 or width % stride or height % stride:
        raise ValueError(f"{label}: bucket_reso {bucket_reso} is not aligned to latent stride {stride}")
    expected = (height // stride, width // stride)
    shape = tuple(shape)
    valid_shapes = ((spec.latent_channels, *expected), (spec.latent_channels, 1, *expected))
    if shape not in valid_shapes:
        raise ValueError(
            f"{label} geometry {shape} does not match {valid_shapes} "
            f"for bucket_reso {bucket_reso} and stride {stride}"
        )


def validate_latent_batch_geometry(
    latents,
    bucket_sizes: Sequence[tuple[int, int]],
    spec: LatentSpaceSpec,
    *,
    label: str,
) -> None:
    shape = getattr(latents, "shape", ())
    if len(shape) not in (4, 5):
        raise ValueError(f"{label} geometry requires a 4D or singleton-time 5D batch, got {shape}")
    if shape[0] != len(bucket_sizes):
        raise ValueError(f"{label} batch {shape[0]} for {len(bucket_sizes)} image(s)")
    for bucket_reso in set(bucket_sizes):
        validate_latent_shape(shape[1:], bucket_reso, spec, label=label)
