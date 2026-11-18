"""LoHa (Low-Rank Hadamard Product) plugin module."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from ...lora_modules.base import BaseLoRAModule
from .autograd import loha_linear, make_hada_weight

logger = logging.getLogger(__name__)


class LoHaModule(BaseLoRAModule):
    """PEFT/LyCORIS-style LoHa adapter for Linear layers."""

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
    ):
        if not isinstance(org_module, torch.nn.Linear):
            raise ValueError("LoHaModule only supports torch.nn.Linear modules")
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

        in_features = int(org_module.in_features)
        out_features = int(org_module.out_features)

        self.hada_w1_a = torch.nn.Parameter(torch.empty(out_features, self.lora_dim))
        self.hada_w1_b = torch.nn.Parameter(torch.empty(self.lora_dim, in_features))
        self.hada_w2_a = torch.nn.Parameter(torch.empty(out_features, self.lora_dim))
        self.hada_w2_b = torch.nn.Parameter(torch.empty(self.lora_dim, in_features))

        torch.nn.init.kaiming_uniform_(self.hada_w1_a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.hada_w1_b, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.hada_w2_a, a=math.sqrt(5))
        torch.nn.init.zeros_(self.hada_w2_b)

        if channel_scale is not None:
            logger.warning(
                "LoHa module %s: channel_scale ignored because Hadamard "
                "weights cannot safely absorb per-input-column scale",
                lora_name,
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _compute_weight(self) -> torch.Tensor:
        return make_hada_weight(
            self.hada_w1_a,
            self.hada_w1_b,
            self.hada_w2_a,
            self.hada_w2_b,
        )

    @staticmethod
    def _drop_output_rows(weight: torch.Tensor, p: float | None) -> torch.Tensor:
        if p is None or p <= 0.0:
            return weight
        keep = (torch.rand(weight.size(0), device=weight.device) > p).to(weight.dtype)
        keep = keep.view(-1, *([1] * (weight.dim() - 1)))
        return weight * keep / keep.mean().clamp_min(1e-6)

    def _scalar_timestep_gate(self, y: torch.Tensor) -> torch.Tensor:
        gate = self._timestep_mask[:, :1].to(device=y.device, dtype=y.dtype)
        while gate.dim() < y.dim():
            gate = gate.unsqueeze(1)
        return gate

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)
        if self._skip_module():
            return org_forwarded

        x_loha = F.dropout(x, p=self.dropout) if self.training and self.dropout else x
        effective_scale = float(self.multiplier) * float(self.scale)

        # rank_dropout masks output rows of the materialized ΔW. Keep the legacy
        # path when it is active so dropout semantics stay bit-compatible.
        if self.training and self.rank_dropout:
            weight = self._compute_weight()
            weight = self._drop_output_rows(weight, self.rank_dropout)
            y = F.linear(x_loha.float(), weight.float()) * effective_scale
            y = y.to(org_forwarded.dtype)
        else:
            y = loha_linear(
                x_loha,
                self.hada_w1_a,
                self.hada_w1_b,
                self.hada_w2_a,
                self.hada_w2_b,
                scale=effective_scale,
            )

        if self.training:
            y = y * self._scalar_timestep_gate(y)
        return org_forwarded + y.to(org_forwarded.dtype)

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
                raise RuntimeError("inactive LoHa delta requires base_output")
            return torch.zeros_like(base_output)
        if self._skip_module():
            if base_output is not None:
                return torch.zeros_like(base_output)
            return torch.zeros((*x.shape[:-1], self.hada_w1_a.shape[0]), device=x.device, dtype=output_dtype)

        x_loha = F.dropout(x, p=self.dropout) if self.training and self.dropout else x
        effective_scale = float(self.multiplier) * float(self.scale)
        if self.training and self.rank_dropout:
            weight = self._drop_output_rows(self._compute_weight(), self.rank_dropout)
            delta = F.linear(x_loha.float(), weight.float()) * effective_scale
        else:
            delta = loha_linear(
                x_loha, self.hada_w1_a, self.hada_w1_b,
                self.hada_w2_a, self.hada_w2_b, scale=effective_scale,
            )
        if self.training:
            delta = delta * self._scalar_timestep_gate(delta)
        return delta.to(output_dtype)

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

            delta = make_hada_weight(
                sd["hada_w1_a"].to(device),
                sd["hada_w1_b"].to(device),
                sd["hada_w2_a"].to(device),
                sd["hada_w2_b"].to(device),
            )
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
