"""Pure declarative capability sub-specs for registered model families.

Keep this module strictly free of ``torch`` or model-weight imports so CLI,
config schema, and WebUI capability discovery can load it in sub-millisecond
startup paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class TopologyCapabilitySpec:
    """Block container layout, swap bounds, and main-MLP quantization targets."""

    block_container: str
    default_num_blocks: int
    known_num_blocks: frozenset[int]
    min_resident_blocks: int = 2
    supports_adaptive_swap_capacity: bool = False
    mixed_nf4_mlp_spec: tuple[str, tuple[str, ...]] | None = None
    supports_all_linear_nf4: bool = False

    @property
    def max_blocks_to_swap(self) -> int:
        return self.swap_limit_for(self.default_num_blocks)

    def swap_limit_for(self, block_count: int) -> int:
        return max(0, int(block_count) - self.min_resident_blocks)


@dataclass(frozen=True)
class AttentionCapabilitySpec:
    """Canonical attention backends and accepted config aliases per family."""

    canonical_modes: frozenset[str]
    mode_aliases: Mapping[str, str] = field(default_factory=dict)
    forbids_xformers_flag: bool = False

    @property
    def accepted_modes(self) -> frozenset[str]:
        return self.canonical_modes | frozenset(self.mode_aliases.keys())


@dataclass(frozen=True)
class CompileCapabilitySpec:
    """Per-family ``torch.compile`` support envelope."""

    supported: bool
    inductor_modes: frozenset[str] = frozenset({"default"})
    supports_dynamic_seq: bool = False
    supports_seq_bands: bool = False
    allows_cudagraphs: bool = False


@dataclass(frozen=True)
class QuantizationCapabilitySpec:
    """Runtime precision, base quantization, and checkpointing envelope."""

    runtime_base_compute: frozenset[str] = frozenset({"bf16"})
    mixed_precisions: frozenset[str] = frozenset({"bf16", "fp16", "no"})
    block_swap_transfer_dtypes: frozenset[str] = frozenset({"bf16", "fp8_e4m3", "int8"})
    selective_checkpoint_modes: frozenset[str] = frozenset({"off"})
    supports_checkpoint_offload: bool = False
    supports_v100_flash_stability: bool = False
