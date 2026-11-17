"""LoKr (Low-Rank Kronecker Product) plugin module."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from library.runtime.peak_probe import record_peak_probe_event
from ...attn_fuse import match_fused_spec
from ...lora_modules.base import BaseLoRAModule
from .autograd import (
    DEFAULT_LOKR_GROUPED_DELTA_BACKEND,
    DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
    DEFAULT_LOKR_PROJECT_CHUNK_BYTES,
    lokr_add_grouped_delta_,
    normalize_lokr_grouped_delta_backward_backend,
    lokr_project_factor_group,
    normalize_lokr_grouped_delta_backend,
)

logger = logging.getLogger(__name__)


def _factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Split a dimension into a near-square ``(outer, inner)`` factor pair."""

    if factor > 0 and dimension % factor == 0:
        outer = factor
        inner = dimension // factor
        if outer > inner:
            inner, outer = outer, inner
        return outer, inner
    if factor < 0:
        factor = dimension
    outer, inner = 1, dimension
    best_sum = outer + inner
    while outer < inner:
        candidate = outer + 1
        while dimension % candidate != 0:
            candidate += 1
        other = dimension // candidate
        if candidate + other > best_sum or candidate > factor:
            break
        outer, inner = candidate, other
        best_sum = outer + inner
    if outer > inner:
        inner, outer = outer, inner
    return outer, inner


class LoKrModule(BaseLoRAModule):
    """LyCORIS-style LoKr adapter for Linear layers."""

    supports_conv2d = False

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
        factor=8,
        lokr_factor_group_size=8,
        lokr_project_chunk_bytes=DEFAULT_LOKR_PROJECT_CHUNK_BYTES,
        lokr_grouped_delta_backend=DEFAULT_LOKR_GROUPED_DELTA_BACKEND,
        lokr_grouped_delta_backward_backend=DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
        lokr_use_einsum=True,
        lokr_decompose_w2=False,
        lokr_full_factor=False,
        explicit_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        if not isinstance(org_module, torch.nn.Linear):
            raise ValueError("LoKrModule only supports torch.nn.Linear modules")
        if lokr_full_factor and lokr_decompose_w2:
            raise ValueError(
                "LoKR full_factor and lokr_decompose_w2 are mutually exclusive"
            )
        if not lokr_use_einsum and lokr_decompose_w2:
            raise ValueError(
                "LoKR lokr_decompose_w2=true requires lokr_use_einsum=true"
            )
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
        self.lokr_use_einsum = bool(lokr_use_einsum)
        self.lokr_full_factor = bool(lokr_full_factor)
        fused_spec = match_fused_spec(str(lora_name))
        n_comp = (
            len(fused_spec.component_letters)
            if fused_spec is not None and out_features % len(fused_spec.component_letters) == 0
            else 1
        )
        single_out = out_features // n_comp

        has_explicit = (
            isinstance(explicit_shapes, dict)
            and "w1" in explicit_shapes
            and (
                "w2" in explicit_shapes
                or ("w2_a" in explicit_shapes and "w2_b" in explicit_shapes)
            )
        )
        if has_explicit:
            self.out_a, self.in_a = (int(v) for v in explicit_shapes["w1"])
            self.factor = max(1, self.out_a // n_comp)
            if "w2_a" in explicit_shapes and "w2_b" in explicit_shapes:
                self.out_b, decomp_rank = (int(v) for v in explicit_shapes["w2_a"])
                _, self.in_b = (int(v) for v in explicit_shapes["w2_b"])
                self.in_dim = self.in_b
                self.out_dim = self.out_b
                self.lokr_w1 = torch.nn.Parameter(torch.empty(self.out_a, self.in_a))
                self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_b, decomp_rank))
                self.lokr_w2_b = torch.nn.Parameter(torch.empty(decomp_rank, self.in_b))
                self._use_decomposed_w2 = True
                self.lokr_use_einsum = True
                torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lokr_w2_b)
            else:
                self.out_b, self.in_b = (int(v) for v in explicit_shapes["w2"])
                self.in_dim = self.in_b
                self.out_dim = self.out_b
                if self.out_a // n_comp != self.in_a:
                    self.lokr_use_einsum = True
                self.lokr_w1 = torch.nn.Parameter(torch.empty(self.out_a, self.in_a))
                self.lokr_w2 = torch.nn.Parameter(torch.empty(self.out_b, self.in_b))
                self._use_decomposed_w2 = False
                torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lokr_w2)
        elif self.lokr_use_einsum:
            sub_out_a, self.out_b = _factorization(single_out, int(factor))
            self.out_a = n_comp * sub_out_a
            self.in_a, self.in_b = _factorization(in_features, int(factor))
            self.factor = sub_out_a
            self.in_dim = self.in_b
            self.out_dim = self.out_b

            self.lokr_w1 = torch.nn.Parameter(torch.empty(self.out_a, self.in_a))
            # full_factor forces both Kronecker factors complete, independent of
            # lora_dim. Without it, lokr_decompose_w2 may split the larger factor.
            decompose_w2 = bool(lokr_decompose_w2) and not self.lokr_full_factor
            if decompose_w2:
                self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_b, lora_dim))
                self.lokr_w2_b = torch.nn.Parameter(torch.empty(lora_dim, self.in_b))
                self._use_decomposed_w2 = True
            else:
                self.lokr_w2 = torch.nn.Parameter(torch.empty(self.out_b, self.in_b))
                self._use_decomposed_w2 = False

            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            if self._use_decomposed_w2:
                torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lokr_w2_b)
            else:
                torch.nn.init.zeros_(self.lokr_w2)
        else:
            sub_factor = self._find_factor(in_features, single_out, int(factor))
            self.factor = sub_factor
            self.in_dim = in_features // sub_factor
            self.out_dim = single_out // sub_factor
            self.in_a = sub_factor
            self.out_a = n_comp * sub_factor
            self.in_b = self.in_dim
            self.out_b = self.out_dim

            self.lokr_w1 = torch.nn.Parameter(torch.empty(self.out_a, self.in_a))
            self.lokr_w2 = torch.nn.Parameter(torch.empty(self.out_dim, self.in_dim))
            self._use_decomposed_w2 = False

            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2)

        if channel_scale is not None:
            logger.warning(
                "LoKr module %s: channel_scale ignored because Kronecker "
                "weights cannot safely absorb per-input-column scale",
                lora_name,
            )
        if rank_dropout is not None:
            logger.warning(
                "LoKr module %s: rank_dropout ignored because LoKr has no "
                "explicit low-rank bottleneck",
                lora_name,
            )

        self.org_module_ref = [org_module]
        self._fused = False
        self.use_custom_lokr_autograd = False
        self.lokr_factor_group_size = max(1, int(lokr_factor_group_size))
        self.lokr_project_chunk_bytes = max(1, int(lokr_project_chunk_bytes))
        self.lokr_grouped_delta_backend = normalize_lokr_grouped_delta_backend(
            lokr_grouped_delta_backend
        )
        self.lokr_grouped_delta_backward_backend = (
            normalize_lokr_grouped_delta_backward_backend(
                lokr_grouped_delta_backward_backend
            )
        )
        self._peak_probe = None

    @staticmethod
    def _find_factor(in_features: int, out_features: int, target_factor: int) -> int:
        target_factor = max(1, int(target_factor))
        candidates = [target_factor]
        candidates.extend(f for f in (16, 8, 4, 2, 1) if f < target_factor)
        for factor in dict.fromkeys(candidates):
            if factor > 0 and in_features % factor == 0 and out_features % factor == 0:
                return factor
        return 1

    def _compute_weight(self) -> torch.Tensor:
        return torch.kron(self.lokr_w1, self._get_w2())

    def _get_w2(self) -> torch.Tensor:
        if self._use_decomposed_w2:
            return self.lokr_w2_a @ self.lokr_w2_b
        return self.lokr_w2

    def _can_use_fused_grouped_delta(
        self,
        x: torch.Tensor,
        base: torch.Tensor,
        gate_scale: torch.Tensor,
    ) -> bool:
        if self._use_decomposed_w2 or self.dropout is not None:
            return False
        if gate_scale.numel() != 1:
            return False
        if self.out_a != self.in_a:
            return False
        if self.lokr_w1.shape != (self.out_a, self.in_a):
            return False
        if self.lokr_w2.shape != (self.out_b, self.in_b):
            return False
        return x.is_contiguous() and base.is_contiguous()

    def _apply_fused_grouped_delta(
        self,
        base: torch.Tensor,
        x: torch.Tensor,
        gate_scale: torch.Tensor,
    ) -> torch.Tensor:
        group_size = max(1, min(int(self.lokr_factor_group_size), self.out_a))
        return lokr_add_grouped_delta_(
            base,
            x,
            self.lokr_w1,
            self.lokr_w2,
            gate_scale,
            self.out_a,
            self.in_b,
            self.out_b,
            group_size,
            self.lokr_project_chunk_bytes,
            backend=self.lokr_grouped_delta_backend,
            backward_backend=self.lokr_grouped_delta_backward_backend,
        )

    def _einsum_delta(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x_lokr = x.to(dtype)
        w1 = self.lokr_w1.to(dtype)
        w2 = self._get_w2().to(dtype)
        out_a, in_a = w1.shape
        out_b, in_b = w2.shape
        x_mat = x_lokr.reshape(-1, in_a, in_b)
        delta = torch.einsum("oi,nij,bj->nob", w1.to(x_mat), x_mat, w2.to(x_mat))
        out_features = out_a * out_b
        if x_lokr.dim() == 2:
            return delta.reshape(x_lokr.shape[0], out_features)
        if x_lokr.dim() == 3:
            return delta.reshape(x_lokr.shape[0], x_lokr.shape[1], out_features)
        return delta.reshape(x_lokr.shape[:-1] + (out_features,))

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)
        original_name = getattr(self, "original_name", None)
        peak_probe = self._peak_probe
        record_lokr = bool(getattr(peak_probe, "record_lokr", False))
        if record_lokr:
            record_peak_probe_event(
                peak_probe,
                "lokr_after_base",
                tensor=org_forwarded,
                module_type="lokr",
                lora_name=self.lora_name,
                original_name=original_name,
                factor=self.factor,
                group_size=self.lokr_factor_group_size,
                chunk_bytes=self.lokr_project_chunk_bytes,
            )
        if self._skip_module():
            return org_forwarded

        x_lokr = F.dropout(x, p=self.dropout) if self.training and self.dropout else x
        if record_lokr:
            record_peak_probe_event(
                peak_probe,
                "lokr_before_delta_apply",
                tensor=x_lokr,
                module_type="lokr",
                lora_name=self.lora_name,
                original_name=original_name,
                factor=self.factor,
                group_size=self.lokr_factor_group_size,
                chunk_bytes=self.lokr_project_chunk_bytes,
            )
        if self.lokr_use_einsum:
            if self.training:
                work = self._rank_compute_dtype(org_forwarded)
                with self._rank_autocast_context(x, work):
                    x_r = self._rebalance(x.to(work))
                    gate_scale = (
                        self._timestep_mask[:, :1].float()
                        * self.multiplier
                        * self.scale
                    )
                    if (
                        self.lokr_grouped_delta_backend == "triton"
                        and self._can_use_fused_grouped_delta(
                            x_r, org_forwarded, gate_scale
                        )
                    ):
                        result = self._apply_fused_grouped_delta(
                            org_forwarded,
                            x_r,
                            gate_scale,
                        )
                        if record_lokr:
                            record_peak_probe_event(
                                peak_probe,
                                "lokr_after_delta_apply",
                                tensor=result,
                                module_type="lokr",
                                lora_name=self.lora_name,
                                original_name=original_name,
                                factor=self.factor,
                                group_size=self.lokr_factor_group_size,
                                chunk_bytes=self.lokr_project_chunk_bytes,
                            )
                        return result
                    lx = self._einsum_delta(x_r, work)
                    lx = lx * self._timestep_mask[:, :1].to(lx)
                    if self.dropout is not None:
                        lx = F.dropout(lx, p=self.dropout)
            else:
                lx = self._einsum_delta(self._rebalance(x), x.dtype)
            result = org_forwarded + (lx * self.multiplier * self.scale).to(
                org_forwarded.dtype
            )
            if record_lokr:
                record_peak_probe_event(
                    peak_probe,
                    "lokr_after_delta_apply",
                    tensor=result,
                    module_type="lokr",
                    lora_name=self.lora_name,
                    original_name=original_name,
                    factor=self.factor,
                    group_size=self.lokr_factor_group_size,
                    chunk_bytes=self.lokr_project_chunk_bytes,
                )
            return result

        if self.training:
            if self.use_custom_lokr_autograd and self.out_a == self.in_a:
                gate_scale = self._timestep_mask[:, :1] * self.multiplier * self.scale
                group_size = max(1, min(int(self.lokr_factor_group_size), self.factor))
                if gate_scale.numel() == 1 and org_forwarded.is_contiguous():
                    result = lokr_add_grouped_delta_(
                        org_forwarded,
                        x_lokr,
                        self.lokr_w1,
                        self.lokr_w2,
                        gate_scale,
                        self.factor,
                        self.in_dim,
                        self.out_dim,
                        group_size,
                        self.lokr_project_chunk_bytes,
                        backend=self.lokr_grouped_delta_backend,
                        backward_backend=self.lokr_grouped_delta_backward_backend,
                    )
                else:
                    # Preserve the existing broadcast semantics for rare
                    # non-scalar T-LoRA masks; the fused path intentionally
                    # optimizes the standard scalar LoKr gate used by Anima.
                    result = org_forwarded
                    for out_start in range(0, self.factor, group_size):
                        out_count = min(group_size, self.factor - out_start)
                        start = out_start * self.out_dim
                        end = start + out_count * self.out_dim
                        lx_slice = lokr_project_factor_group(
                            x_lokr,
                            self.lokr_w1,
                            self.lokr_w2,
                            out_start,
                            out_count,
                            self.factor,
                            self.in_dim,
                            self.out_dim,
                            self.lokr_project_chunk_bytes,
                        )
                        result[..., start:end].add_(
                            (lx_slice * gate_scale).to(org_forwarded.dtype)
                        )
                if record_lokr:
                    record_peak_probe_event(
                        peak_probe,
                        "lokr_after_delta_apply",
                        tensor=result,
                        module_type="lokr",
                        lora_name=self.lora_name,
                        original_name=original_name,
                        factor=self.factor,
                        group_size=self.lokr_factor_group_size,
                        chunk_bytes=self.lokr_project_chunk_bytes,
                    )
                return result
            else:
                lx = F.linear(x_lokr.float(), self._compute_weight().float())
            # LoKr has no rank axis; T-LoRA masks reduce to a scalar gate.
            lx = lx * self._timestep_mask[:, :1]
            lx = lx.to(org_forwarded.dtype)
        else:
            weight = self._compute_weight()
            lx = F.linear(x_lokr, weight.to(x_lokr.dtype))

        result = org_forwarded + lx * self.multiplier * self.scale
        if record_lokr:
            record_peak_probe_event(
                peak_probe,
                "lokr_after_delta_apply",
                tensor=result,
                module_type="lokr",
                lora_name=self.lora_name,
                original_name=original_name,
                factor=self.factor,
                group_size=self.lokr_factor_group_size,
                chunk_bytes=self.lokr_project_chunk_bytes,
            )
        return result

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

            w1 = sd["lokr_w1"].to(torch.float).to(device)
            if "lokr_w2" in sd:
                w2 = sd["lokr_w2"].to(torch.float).to(device)
            else:
                w2a = sd["lokr_w2_a"].to(torch.float).to(device)
                w2b = sd["lokr_w2_b"].to(torch.float).to(device)
                w2 = w2a @ w2b
            delta = torch.kron(w1, w2)
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
