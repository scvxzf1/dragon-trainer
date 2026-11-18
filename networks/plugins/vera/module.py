"""VeRA (Vector-based Random Matrix Adaptation) plugin module.

This follows the PEFT VeRA layout for Linear layers:

    Δy = lambda_b * B @ (lambda_d * (A @ x))

A and B are frozen random projections shared by shape within the adapter run;
only the two vectors are trainable per adapted Linear.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan

from ...lora_modules.base import BaseLoRAModule

logger = logging.getLogger(__name__)


class VeraProjectionBank(torch.nn.Module):
    """Frozen random projection bank sliced by each VeRA module.

    The tensors are intentionally non-trainable.  ``persistent=False`` matches
    the PEFT ``save_projection=False`` workflow: checkpoints stay tiny and the
    projections are deterministically regenerated from ``projection_prng_key``.
    """

    def __init__(
        self,
        *,
        rank: int,
        max_in_features: int,
        max_out_features: int,
        projection_prng_key: int = 0,
        save_projection: bool = False,
    ) -> None:
        super().__init__()
        rank = int(rank)
        max_in_features = int(max_in_features)
        max_out_features = int(max_out_features)
        if rank <= 0:
            raise ValueError(f"VeRA rank must be > 0, got {rank}")
        if max_in_features <= 0 or max_out_features <= 0:
            raise ValueError(
                "VeRA projection bank requires positive max dimensions, got "
                f"in={max_in_features}, out={max_out_features}"
            )

        generator = torch.Generator(device="cpu").manual_seed(
            int(projection_prng_key)
        )
        vera_A = _kaiming_init((rank, max_in_features), generator=generator)
        vera_B = _kaiming_init((max_out_features, rank), generator=generator)
        self.register_buffer("vera_A", vera_A, persistent=bool(save_projection))
        self.register_buffer("vera_B", vera_B, persistent=bool(save_projection))
        self.rank = rank
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.projection_prng_key = int(projection_prng_key)
        self.save_projection = bool(save_projection)

    def slice(self, in_features: int, out_features: int) -> tuple[torch.Tensor, torch.Tensor]:
        if in_features > self.vera_A.shape[1]:
            raise ValueError(
                f"VeRA vera_A width {self.vera_A.shape[1]} < required {in_features}"
            )
        if out_features > self.vera_B.shape[0]:
            raise ValueError(
                f"VeRA vera_B height {self.vera_B.shape[0]} < required {out_features}"
            )
        return self.vera_A[:, :in_features], self.vera_B[:out_features, :]


def _kaiming_init(
    tensor_or_shape: torch.Tensor | tuple[int, ...],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Kaiming-uniform init with an explicit CPU generator, matching PEFT."""

    tensor = torch.empty(tensor_or_shape) if isinstance(tensor_or_shape, tuple) else tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


def make_projection_bank(
    *,
    rank: int,
    max_in_features: int,
    max_out_features: int,
    projection_prng_key: int,
    save_projection: bool = False,
) -> VeraProjectionBank:
    return VeraProjectionBank(
        rank=rank,
        max_in_features=max_in_features,
        max_out_features=max_out_features,
        projection_prng_key=projection_prng_key,
        save_projection=save_projection,
    )


class VeRAModule(BaseLoRAModule):
    """PEFT-style VeRA adapter for ``torch.nn.Linear`` layers."""

    supports_conv2d = False
    supports_packed_qkv = True

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
        projection_bank: Optional[VeraProjectionBank] = None,
        d_initial: float = 0.1,
    ):
        if not isinstance(org_module, torch.nn.Linear):
            raise ValueError("VeRAModule only supports torch.nn.Linear modules")
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        self.in_features = int(org_module.in_features)
        self.out_features = int(org_module.out_features)
        self.projection_bank = projection_bank

        self.vera_lambda_b = torch.nn.Parameter(torch.empty(self.out_features))
        self.vera_lambda_d = torch.nn.Parameter(torch.empty(self.lora_dim))
        with torch.no_grad():
            self.vera_lambda_b.zero_()
            self.vera_lambda_d.fill_(float(d_initial))

        if channel_scale is not None:
            logger.warning(
                "VeRA module %s: channel_scale ignored because the frozen "
                "shared projection cannot safely absorb per-input-column scale",
                lora_name,
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def attach_projection_bank(self, bank: VeraProjectionBank) -> None:
        if bank.rank < self.lora_dim:
            raise ValueError(
                f"VeRA bank rank {bank.rank} < module rank {self.lora_dim}"
            )
        # Validate once so failures happen at construction/load time.
        bank.slice(self.in_features, self.out_features)
        self.projection_bank = bank

    def _projection_slices(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.projection_bank is None:
            raise RuntimeError(
                f"VeRA module {self.lora_name} has no projection bank attached"
            )
        return self.projection_bank.slice(self.in_features, self.out_features)

    @staticmethod
    def _broadcast_output_scale(scale: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        while scale.dim() < y.dim():
            scale = scale.unsqueeze(0)
        return scale

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)
        if self._skip_module():
            return org_forwarded

        vera_A, vera_B = self._projection_slices()
        x_vera = F.dropout(x, p=self.dropout) if self.training and self.dropout else x

        # Training keeps the bottleneck math in fp32 for DiT stability, mirroring
        # the local LoRA implementation. Eval follows input dtype for speed.
        if self.training:
            hidden = F.linear(x_vera.float(), vera_A.float())
            hidden = hidden * self.vera_lambda_d.float() * self._timestep_mask.to(
                device=hidden.device, dtype=hidden.dtype
            )
            hidden, scale = self._apply_rank_dropout(hidden)
            delta = F.linear(hidden, vera_B.float())
            lambda_b = self.vera_lambda_b.float()
        else:
            dtype = x_vera.dtype
            hidden = F.linear(x_vera, vera_A.to(device=x_vera.device, dtype=dtype))
            hidden = hidden * self.vera_lambda_d.to(device=hidden.device, dtype=dtype)
            delta = F.linear(hidden, vera_B.to(device=hidden.device, dtype=dtype))
            scale = self.scale
            lambda_b = self.vera_lambda_b.to(device=delta.device, dtype=dtype)

        lambda_b = self._broadcast_output_scale(lambda_b, delta)
        delta = delta * lambda_b
        return org_forwarded + (delta * self.multiplier * scale).to(org_forwarded.dtype)

    def compute_delta(
        self,
        x: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        *,
        base_output: torch.Tensor | None = None,
        base_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del base_weight
        output_dtype = out_dtype or (base_output.dtype if base_output is not None else x.dtype)
        if not self.enabled or self._fused:
            if base_output is None:
                raise RuntimeError("inactive VeRA delta requires base_output")
            return torch.zeros_like(base_output)
        if self._skip_module():
            if base_output is not None:
                return torch.zeros_like(base_output)
            return torch.zeros((*x.shape[:-1], self.out_features), device=x.device, dtype=output_dtype)
        vera_A, vera_B = self._projection_slices()
        x_vera = F.dropout(x, p=self.dropout) if self.training and self.dropout else x
        if self.training:
            hidden = F.linear(x_vera.float(), vera_A.float())
            hidden = hidden * self.vera_lambda_d.float() * self._timestep_mask.to(
                device=hidden.device, dtype=hidden.dtype
            )
            hidden, scale = self._apply_rank_dropout(hidden)
            delta = F.linear(hidden, vera_B.float())
            lambda_b = self.vera_lambda_b.float()
        else:
            hidden = F.linear(x_vera, vera_A.to(device=x_vera.device, dtype=x_vera.dtype))
            hidden = hidden * self.vera_lambda_d.to(device=hidden.device, dtype=x_vera.dtype)
            delta = F.linear(hidden, vera_B.to(device=hidden.device, dtype=x_vera.dtype))
            scale = self.scale
            lambda_b = self.vera_lambda_b.to(device=delta.device, dtype=delta.dtype)
        delta = delta * self._broadcast_output_scale(lambda_b, delta)
        return (delta * self.multiplier * scale).to(output_dtype)

    def _compute_weight(self) -> torch.Tensor:
        vera_A, vera_B = self._projection_slices()
        return (self.vera_lambda_b.float().unsqueeze(-1) * vera_B.float()) @ (
            self.vera_lambda_d.float().unsqueeze(-1) * vera_A.float()
        )

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier
        return self._compute_weight().float() * multiplier * self.scale

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            vera_A, vera_B = self._projection_slices()
            lambda_b = sd["vera_lambda_b"].to(torch.float).to(device)
            lambda_d = sd["vera_lambda_d"].to(torch.float).to(device)
            vera_A = vera_A.to(torch.float).to(device)
            vera_B = vera_B.to(torch.float).to(device)
            delta = (lambda_b.unsqueeze(-1) * vera_B) @ (lambda_d.unsqueeze(-1) * vera_A)
            weight.data.add_((delta * self.multiplier * self.scale).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False
