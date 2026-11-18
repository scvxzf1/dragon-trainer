"""DoRA extension for the standard Anima LoRA module.

DoRA keeps the LoRA low-rank direction but trains a per-output magnitude
parameter. At forward/merge time the adapted weight ``W + ΔW`` is normalized
by its row norm and rescaled by the learned magnitude.
"""

from __future__ import annotations

from typing import Optional

import torch

from .lora import LoRAModule
from .weight_access import is_nf4_weight, materialize_module_weight


class DoRALoRAModule(LoRAModule):
    supports_conv2d = True
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
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            channel_scale=channel_scale,
        )
        base_weight = materialize_module_weight(org_module, dtype=torch.float32)
        magnitude = self._compute_weight_norm(base_weight)
        self.magnitude = torch.nn.Parameter(magnitude.contiguous())
        self._nf4_base = is_nf4_weight(org_module.weight)
        if self._nf4_base:
            # The packed NF4 payload cannot be read as a logical matrix during
            # forward.  Cache only the base row norm; the cross term is
            # evaluated through the native quantized Linear in
            # ``_nf4_merged_norm`` below.
            base_sq = base_weight.reshape(base_weight.shape[0], -1).square().sum(dim=1)
            self.register_buffer("_base_weight_norm_sq", base_sq, persistent=False)
        else:
            self._base_weight_norm_sq = None
        del base_weight
        self._fused_delta: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse=True):
        """Keep the cached NF4 base norm in FP32 while moving devices."""

        original = self._base_weight_norm_sq
        result = super()._apply(fn, recurse=recurse)
        if self._nf4_base and original is not None:
            self._base_weight_norm_sq = original.to(
                device=self._base_weight_norm_sq.device, dtype=torch.float32
            )
        return result

    @staticmethod
    def _compute_weight_norm(weight: torch.Tensor) -> torch.Tensor:
        flat = weight.reshape(weight.shape[0], -1)
        return torch.linalg.norm(flat, dim=1).clamp_min(1e-6)

    def _delta_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        return super().get_weight(multiplier=multiplier)

    def _nf4_merged_norm(self, org_forwarded: torch.Tensor) -> torch.Tensor:
        """Compute ``||W + a U D||`` without materializing the NF4 matrix.

        ``W`` remains inside bitsandbytes' native matmul.  The cached base
        norm plus the two low-rank terms are algebraically equivalent to the
        dense row norm for Linear weights and keep the packed base untouched.
        """

        org_module = self.org_module_ref[0]
        if not hasattr(self.lora_down, "weight") or self.lora_down.weight.ndim != 2:
            raise RuntimeError("NF4 DoRA currently supports Linear weights only")
        down = self.lora_down.weight.detach().to(
            device=org_forwarded.device, dtype=torch.float32
        )
        if self._has_channel_scale:
            down = down * self.inv_scale.to(down).unsqueeze(0)
        up = self.lora_up.weight.detach().to(
            device=org_forwarded.device, dtype=torch.float32
        )

        # The norm is intentionally detached (the existing DoRA training rule).
        # Probe without bias: subtracting a BF16 bias afterward loses information.
        import bitsandbytes as bnb

        with torch.no_grad(), torch.autocast(org_forwarded.device.type, enabled=False):
            probe = bnb.matmul_4bit(
                down.to(dtype=org_module.compute_dtype),
                org_module.weight.t(),
                quant_state=org_module.weight.quant_state,
            ).float()
            cross = probe.transpose(0, 1)
            gram = down @ down.transpose(0, 1)
            coeff = float(self.scale)
            cross_term = (up * cross).sum(dim=1)
            lowrank_sq = ((up @ gram) * up).sum(dim=1)
            norm_sq = (
                self._base_weight_norm_sq.to(device=org_forwarded.device)
                + 2.0 * coeff * cross_term
                + coeff * coeff * lowrank_sq
            ).clamp_min(1e-12)
            return norm_sq.sqrt()

    def _merged_weight(
        self,
        *,
        multiplier: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        org_module = self.org_module_ref[0]
        multiplier = self.multiplier if multiplier is None else multiplier
        if self._nf4_base:
            org_weight = materialize_module_weight(
                org_module, device=device, dtype=dtype
            )
        else:
            org_weight = org_module.weight.to(device=device, dtype=dtype)
        if multiplier == 0:
            return org_weight
        delta = self._delta_weight(multiplier=1.0).to(
            device=org_weight.device, dtype=dtype
        )
        merged = org_weight + delta
        row_scale = (
            self.magnitude.to(device=org_weight.device, dtype=dtype)
            / self._compute_weight_norm(merged).detach()
        ).clamp_min(1e-6)
        view_shape = [merged.shape[0]] + [1] * (merged.ndim - 1)
        adapted = merged * row_scale.view(*view_shape)
        return org_weight + multiplier * (adapted - org_weight)

    @staticmethod
    def _view_output_channel_param(
        value: torch.Tensor,
        out: torch.Tensor,
        *,
        is_conv2d: bool,
    ) -> torch.Tensor:
        if out.ndim < 2:
            raise ValueError(f"Unsupported DoRA output ndim: {out.ndim}")
        if is_conv2d:
            return value.view(1, -1, *([1] * (out.ndim - 2)))
        return value.view(*([1] * (out.ndim - 1)), -1)

    def get_weight(self, multiplier=None):
        org_module = self.org_module_ref[0]
        if self._nf4_base:
            org_weight = materialize_module_weight(org_module, dtype=torch.float32)
        else:
            org_weight = org_module.weight.to(torch.float32)
        merged = self._merged_weight(
            multiplier=multiplier,
            device=org_weight.device,
            dtype=torch.float32,
        )
        return merged - org_weight

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            if self._nf4_base or is_nf4_weight(weight):
                raise RuntimeError(
                    "DoRA merge_to is not supported for NF4 Params4bit weights; "
                    "export a dense base before merging."
                )
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            down_weight = sd["lora_down.weight"].to(torch.float32).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float32).to(device)
            magnitude_tensor = sd.get(
                "magnitude", sd.get("dora_scale", sd.get("dora_magnitude"))
            )
            if magnitude_tensor is None:
                raise KeyError("DoRA checkpoint is missing magnitude/dora_scale")
            magnitude = magnitude_tensor.to(torch.float32).to(device)

            if "inv_scale" in sd and down_weight.dim() == 2:
                inv_scale = sd["inv_scale"].to(torch.float32).to(device)
                down_weight = down_weight * inv_scale.unsqueeze(0)

            if len(down_weight.size()) == 2:
                delta = up_weight @ down_weight
            elif down_weight.size()[2:4] == (1, 1):
                delta = (
                    up_weight.squeeze(3).squeeze(2)
                    @ down_weight.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                delta = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)

            base = weight.data.to(torch.float32).to(device)
            merged = base + delta * self.scale
            row_scale = (magnitude / self._compute_weight_norm(merged).detach()).clamp_min(
                1e-6
            )
            view_shape = [merged.shape[0]] + [1] * (merged.ndim - 1)
            adapted = merged * row_scale.view(*view_shape)
            weight.data.copy_((base + self.multiplier * (adapted - base)).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        if self._nf4_base or is_nf4_weight(org_module.weight):
            raise RuntimeError(
                "DoRA fuse_weight is not supported for NF4 Params4bit weights; "
                "keep the adapter unfused."
            )
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused_delta = delta.detach().clone()
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        if self._nf4_base or is_nf4_weight(org_module.weight):
            raise RuntimeError(
                "DoRA unfuse_weight is not supported for NF4 Params4bit weights."
            )
        if self._fused_delta is not None:
            org_module.weight.data -= self._fused_delta.to(org_module.weight.dtype)
        self._fused_delta = None
        self._fused = False

    def forward(self, x):
        if not self.enabled or self._fused or self.multiplier == 0:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if self.training and self._skip_module():
            return org_forwarded

        x_lora = self._rebalance(x)
        lx = self.lora_down(x_lora)
        # T-LoRA is training-only; eval/inference keep full rank.
        if self.training:
            lx = lx * self._timestep_mask

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)
        delta_out = self.lora_up(lx) * scale

        org_module = self.org_module_ref[0]
        bias = getattr(org_module, "bias", None)
        is_conv2d = org_module.__class__.__name__ == "Conv2d"
        bias_view = None
        if bias is not None:
            bias_view = self._view_output_channel_param(
                bias.to(device=org_forwarded.device, dtype=org_forwarded.dtype),
                org_forwarded,
                is_conv2d=is_conv2d,
            )

        base_without_bias = (
            org_forwarded if bias_view is None else org_forwarded - bias_view
        )
        if self._nf4_base:
            merged_norm = self._nf4_merged_norm(org_forwarded).detach()
        else:
            merged_norm = self._compute_weight_norm(
                self.org_module_ref[0].weight.to(
                    device=org_forwarded.device, dtype=torch.float32
                )
                + self._delta_weight(multiplier=1.0).to(
                    device=org_forwarded.device, dtype=torch.float32
                )
            ).detach()
        row_scale = (
            self.magnitude.to(device=org_forwarded.device, dtype=org_forwarded.dtype)
            / merged_norm.to(device=org_forwarded.device, dtype=org_forwarded.dtype)
        ).clamp_min(1e-6)
        row_scale = self._view_output_channel_param(
            row_scale,
            org_forwarded,
            is_conv2d=is_conv2d,
        )

        # Residual form preserves the quantized base exactly at zero init,
        # even when subtracting and re-adding a BF16 bias would round twice.
        edit = (row_scale - 1) * base_without_bias
        edit = edit + delta_out.to(org_forwarded.dtype) * row_scale
        adapted = org_forwarded + edit
        if self.multiplier == 1:
            return adapted
        return org_forwarded + self.multiplier * (adapted - org_forwarded)

    def compute_delta(
        self,
        x: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        *,
        base_output: torch.Tensor | None = None,
        base_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled or self._fused or self.multiplier == 0:
            if base_output is None:
                raise RuntimeError("inactive DoRA delta requires base_output")
            return torch.zeros_like(base_output)
        if base_output is None or base_weight is None:
            raise RuntimeError("DoRA packed delta requires base_output and base_weight")
        output_dtype = out_dtype or base_output.dtype
        if self.training and self._skip_module():
            return torch.zeros_like(base_output)

        x_lora = self._rebalance(x)
        lx = self.lora_down(x_lora)
        if self.training:
            lx = lx * self._timestep_mask
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx, scale = self._apply_rank_dropout(lx)
        delta_out = self.lora_up(lx) * scale

        delta_weight = self._delta_weight(multiplier=1.0).to(
            device=base_output.device, dtype=torch.float32
        )
        merged_norm = self._compute_weight_norm(
            base_weight.to(device=base_output.device, dtype=torch.float32) + delta_weight
        ).detach()
        row_scale = (
            self.magnitude.to(device=base_output.device, dtype=output_dtype)
            / merged_norm.to(device=base_output.device, dtype=output_dtype)
        ).clamp_min(1e-6)
        row_scale = row_scale.view(*([1] * (base_output.ndim - 1)), -1)
        edit = (row_scale - 1) * base_output
        edit = edit + delta_out.to(output_dtype) * row_scale
        return (edit * self.multiplier).to(output_dtype)
