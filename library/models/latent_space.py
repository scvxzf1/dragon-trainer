"""Latent-space geometry specs for multi-space DiT training.

DC-Gen migrates a pretrained DiT from its original latent space (Anima:
f8 / 16 channels / patch2) into a higher-compression latent space (DC-AE:
f32 / 32 channels / patch1). Every piece of geometry that used to be a module
constant (``8``, ``16``, ``2``) is described here as one :class:`LatentSpaceSpec`
so cache naming, DiT construction and token-count math read from one source.

A spec deliberately owns *no tensors*. It is the shared vocabulary for:

* ``library/models/dc_ae.py``  — DC-AE loading + training-space encode
* ``library/io/cache_names.py`` — per-space latent sidecar suffixes
* ``library/anima/weights.py`` — parameterized Anima DiT construction
* ``scripts/dcgen/*``          — dual-latent cache + alignment probes
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatentSpaceSpec:
    """Geometry + cache identity for one DiT latent space.

    ``name`` is the stable cache/stamp key and must not change once sidecars
    have been written, because it selects the on-disk latent files.
    """

    name: str
    vae_spatial_compression: int
    latent_channels: int
    patch_spatial: int
    patch_temporal: int = 1
    # Suffix of the latent sidecar, e.g. ``_anima.npz`` / ``_dcgen_f32c32.npz``.
    cache_suffix: str = "_anima.npz"
    # How ``encode -> DiT input`` normalization is applied.
    #   "anima_mean_std": VAE encode already returns normalized latents.
    #   "scaling_factor": raw VAE latent must be multiplied by ``scaling_factor``.
    normalization: str = "anima_mean_std"
    scaling_factor: float | None = None
    shift_factor: float | None = None
    pixel_alignment_override: int | None = None

    def __post_init__(self) -> None:
        if self.vae_spatial_compression <= 0:
            raise ValueError("vae_spatial_compression must be positive")
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if self.patch_spatial <= 0:
            raise ValueError("patch_spatial must be positive")
        if self.normalization == "scaling_factor" and self.scaling_factor is None:
            raise ValueError("scaling_factor is required for scaling_factor normalization")
        if self.normalization == "shift_and_scale":
            if self.scaling_factor is None or self.shift_factor is None:
                raise ValueError(
                    "scaling_factor and shift_factor are required for "
                    "shift_and_scale normalization"
                )

    @property
    def pixel_stride(self) -> int:
        """One DiT patch token covers ``pixel_stride`` source pixels per side."""
        return self.vae_spatial_compression * self.patch_spatial

    @property
    def pixel_alignment(self) -> int:
        """Required source-image pixel divisibility (e.g. 32px for Qwen 2x2 slots)."""
        return int(self.pixel_alignment_override or self.pixel_stride)

    @property
    def patch_embed_in_channels(self) -> int:
        """PatchEmbed input channels including the concatenated padding mask."""
        return self.latent_channels + 1

    @property
    def patch_embed_in_features(self) -> int:
        """PatchEmbed Linear input width: (C + mask) * patch^2."""
        return (
            self.patch_embed_in_channels
            * self.patch_spatial
            * self.patch_spatial
            * self.patch_temporal
        )

    @property
    def final_layer_out_features(self) -> int:
        """FinalLayer Linear output width: C * patch^2."""
        return (
            self.latent_channels
            * self.patch_spatial
            * self.patch_spatial
            * self.patch_temporal
        )


# Original Anima space: 8x VAE, 16 channels, patch 2. Effective pixel stride 16.
ANIMA_F8C16_P2 = LatentSpaceSpec(
    name="anima",
    vae_spatial_compression=8,
    latent_channels=16,
    patch_spatial=2,
    cache_suffix="_anima.npz",
    normalization="anima_mean_std",
)

# DC-Gen target space for the first Anima forge: 32x VAE, 32 channels, patch 1.
# Token count at equal resolution is (8*2 / 32*1)^2 = 1/4 of the original.
DCGEN_F32C32_P1 = LatentSpaceSpec(
    name="dcgen_f32c32",
    vae_spatial_compression=32,
    latent_channels=32,
    patch_spatial=1,
    cache_suffix="_dcgen_f32c32.npz",
    normalization="scaling_factor",
    scaling_factor=0.41407,  # mit-han-lab/dc-ae-f32c32-sana-1.1(-diffusers)
)

# Z-Image uses Flux VAE latents with an affine normalization distinct from
# Anima's Qwen VAE space. Geometry matches Anima, cache identity does not.
Z_IMAGE_F8C16_P2 = LatentSpaceSpec(
    name="z_image",
    vae_spatial_compression=8,
    latent_channels=16,
    patch_spatial=2,
    cache_suffix="_z_image.npz",
    normalization="shift_and_scale",
    scaling_factor=0.3611,
    shift_factor=0.1159,
)

QWEN_IMAGE_21_F16C64_P1 = LatentSpaceSpec(
    name="qwen_image_2_1",
    vae_spatial_compression=16,
    latent_channels=64,
    patch_spatial=1,
    cache_suffix="_qwen_image_2_1.npz",
    normalization="qwen_image_2_1_mean_std",
    pixel_alignment_override=32,
)

MOSS_IMAGE_F16C128_P1 = LatentSpaceSpec(
    name="moss_image",
    vae_spatial_compression=16,
    latent_channels=128,
    patch_spatial=1,
    cache_suffix="_moss_image.npz",
    normalization="moss_image_packed_bn",
)

ALL_SPACES = (
    ANIMA_F8C16_P2, DCGEN_F32C32_P1, Z_IMAGE_F8C16_P2,
    QWEN_IMAGE_21_F16C64_P1, MOSS_IMAGE_F16C128_P1,
)


def get_latent_space(name: str) -> LatentSpaceSpec:
    for spec in ALL_SPACES:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown latent space {name!r}; known: {[s.name for s in ALL_SPACES]}")
