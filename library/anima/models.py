# Anima Model Architecture
# Original code: NVIDIA CORPORATION & AFFILIATES, licensed under Apache-2.0

import math
from contextlib import nullcontext
from numbers import Integral
from typing import Any, Optional, Sequence, Tuple

import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint as torch_checkpoint,
    create_selective_checkpoint_contexts,
)

from library.runtime import offloading as custom_offloading_utils
from library.runtime.block_swap_controller import ExplicitBlockSwapMixin
from library.anima.finite_checks import (
    assert_finite_tensor as _assert_finite_tensor,
    finite_checks_enabled as _finite_checks_enabled,
)
from library.runtime.peak_probe import record_peak_probe_event
from library.runtime.token_counts import (
    ANIMA_VAE_SPATIAL_COMPRESSION,
    pixel_bucket_token_counts,
)
from networks import attention_dispatch


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


# Unsloth Offloaded Gradient Checkpointing
# Based on Unsloth Zoo by Daniel Han-Chen & the Unsloth team
try:
    from deepspeed.runtime.activation_checkpointing.checkpointing import detach_variable
except ImportError:

    def detach_variable(inputs, device=None):
        """Detach tensors from computation graph, optionally moving to a device.

        Reimplementation of deepspeed.runtime.activation_checkpointing.checkpointing.detach_variable
        for environments without DeepSpeed installed.
        """
        if isinstance(inputs, tuple):
            out = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    out.append(inp)
                    continue
                requires_grad = inp.requires_grad
                if device is not None:
                    x = inp.to(device=device)
                else:
                    x = inp
                x = x.detach()
                x.requires_grad = requires_grad
                out.append(x)
            return tuple(out)
        else:
            raise RuntimeError(
                "Only tuple of tensors is supported. Got Unsupported input type: ",
                type(inputs).__name__,
            )


class UnslothOffloadedGradientCheckpointer(torch.autograd.Function):
    """Saves VRAM by offloading activations to CPU RAM using non-blocking transfers.

    Compared to standard cpu_offload_checkpointing which uses blocking transfers,
    this uses non_blocking=True to hide CPU<->GPU transfer latency behind compute.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, forward_function, hidden_states, *args):
        # Remember the original device for backward pass (multi-GPU support)
        ctx.input_device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        # NOTE: args stored directly on ctx (not via save_for_backward) because
        # the training loop holds references to these tensors, preventing GC.
        # Using save_for_backward for all args would add complexity for no benefit.
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.input_device, non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(
            outputs if isinstance(outputs, tuple) else (outputs,),
            grads if isinstance(grads, tuple) else (grads,),
        ):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    """Wrapper for UnslothOffloadedGradientCheckpointer."""
    return UnslothOffloadedGradientCheckpointer.apply(function, *args)


def _mark_seq_axis_dynamic(
    tensor: torch.Tensor,
    axis: int,
    lo: int,
    hi: int,
    *,
    strict: bool = False,
) -> None:
    """Mark one sequence axis, retaining the peak-probe compatibility path.

    The historical union-range path uses ``maybe_mark_dynamic`` because the
    optional op-level peak probe deliberately creates graph breaks. Per-band
    dispatch opts into strict bounds so each specialization retains its tight
    ``(lo, hi)`` contract; the scoped SDPA backend in ``forward_mini_train_dit``
    avoids the alignment guard that strict arbitrary ranges can otherwise hit.
    """
    if strict:
        torch._dynamo.mark_dynamic(tensor, axis, min=int(lo), max=int(hi))
        return
    maybe_mark_dynamic = getattr(torch._dynamo, "maybe_mark_dynamic", None)
    if callable(maybe_mark_dynamic):
        maybe_mark_dynamic(tensor, axis)
    else:
        torch._dynamo.mark_dynamic(tensor, axis, min=int(lo), max=int(hi))


def _make_dynamic_seq_forward(
    compiled_inner, bands, hi: Optional[int] = None, *, strict_marks: bool = False
):
    """Wrap a compiled ``Block._forward`` with recompute-safe dynamic marks.

    Under gradient checkpointing, backward recompute detaches tensor inputs into
    fresh tensors. Any ``mark_dynamic`` done once before ``Block.forward`` is
    therefore lost on recompute, while non-tensor inputs such as the RoPE tuple
    keep their previous dynamic metadata. Marking inside the checkpointed
    callable makes forward and recompute see the same symbolic seq axis.
    """

    # Keep the old ``(compiled, lo, hi)`` call form usable for external probes
    # while the training path passes a list of explicit bands.
    if hi is not None:
        normalized_bands = [(int(bands), int(hi))]
    elif isinstance(bands, tuple) and len(bands) == 2 and all(
        isinstance(value, Integral) for value in bands
    ):
        normalized_bands = [(int(bands[0]), int(bands[1]))]
    else:
        normalized_bands = sorted(
            (int(lo), int(high)) for lo, high in (bands or [])
        )
    if not normalized_bands:
        raise ValueError("dynamic-seq requires at least one token band")
    for band_lo, band_hi in normalized_bands:
        if band_lo < 1 or band_hi < band_lo:
            raise ValueError(f"invalid dynamic-seq token band {(band_lo, band_hi)}")
    for previous, current in zip(normalized_bands, normalized_bands[1:]):
        if current[0] <= previous[1]:
            raise ValueError(
                "dynamic-seq token bands must be non-overlapping; "
                f"got {normalized_bands}"
            )
    strict_marks = bool(strict_marks)

    from library.datasets.buckets import band_for_seq

    warned_oob: set[int] = set()

    @torch._disable_dynamo
    def marked_forward(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
        use_fp32: bool = False,
    ):
        # native_flatten shape is (B, 1, seq_len, 1, D), so seq_len is dim 2.
        if (
            x_B_T_H_W_D.ndim == 5
            and int(x_B_T_H_W_D.shape[1]) == 1
            and int(x_B_T_H_W_D.shape[3]) == 1
        ):
            seq_len = int(x_B_T_H_W_D.shape[2])
            band = band_for_seq(normalized_bands, seq_len)
            if band is None:
                if not strict_marks:
                    lo, high = normalized_bands[0][0], normalized_bands[-1][1]
                    raise RuntimeError(
                        f"compile_dynamic_seq seq_len {seq_len} is outside "
                        f"derived range [{lo}, {high}]"
                    )
                if seq_len not in warned_oob:
                    warned_oob.add(seq_len)
                    logger.warning(
                        "dynamic-seq: %s tokens falls outside every compiled band "
                        "%s; running an unmarked (static) specialization",
                        seq_len,
                        normalized_bands,
                    )
            else:
                lo, high = band
                _mark_seq_axis_dynamic(
                    x_B_T_H_W_D, 2, lo, high, strict=strict_marks
                )
                if rope_cos_sin is not None:
                    _mark_seq_axis_dynamic(
                        rope_cos_sin[0], 0, lo, high, strict=strict_marks
                    )
                    _mark_seq_axis_dynamic(
                        rope_cos_sin[1], 0, lo, high, strict=strict_marks
                    )
        return compiled_inner(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            rope_cos_sin,
            adaln_lora_B_T_3D,
            use_fp32,
        )

    return marked_forward


ADAPTER_AWARE_CHECKPOINT_MAX_SAVE_NUMEL = 1_048_576


def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def _safe_numel_from_shape(shape) -> Optional[int]:
    try:
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)
    except Exception:
        return None


def _tensor_numel(tensor: torch.Tensor) -> Optional[int]:
    try:
        return int(tensor.numel())
    except Exception:
        return _safe_numel_from_shape(getattr(tensor, "shape", ()))


def _broadcast_shape_numel(*shapes) -> Optional[int]:
    try:
        shape = torch.broadcast_shapes(*[torch.Size(s) for s in shapes])
    except Exception:
        return None
    return _safe_numel_from_shape(shape)


def _matmul_output_numel(lhs: torch.Tensor, rhs: torch.Tensor) -> Optional[int]:
    """Best-effort matmul output size inference for checkpoint policy.

    The policy must make the same decision during forward and recompute.  For
    common projection ops we infer output sizes from input shapes instead of
    looking at ``ctx.op_output`` (which is unavailable during recompute).
    """
    try:
        a = tuple(lhs.shape)
        b = tuple(rhs.shape)
        if len(a) == 1 and len(b) == 1:
            return 1
        if len(a) == 1:
            batch = b[:-2]
            out_shape = (*batch, b[-1])
        elif len(b) == 1:
            batch = a[:-2]
            out_shape = (*batch, a[-2])
        else:
            batch = torch.broadcast_shapes(torch.Size(a[:-2]), torch.Size(b[:-2]))
            out_shape = (*batch, a[-2], b[-1])
        return _safe_numel_from_shape(out_shape)
    except Exception:
        return None


def _einsum_output_numel(equation: str, operands) -> Optional[int]:
    """Infer the few adapter einsums used by LoRA-family expert modules."""
    try:
        tensors = list(operands)
        if equation == "...i,eri->...er":
            x, w = tensors
            return _safe_numel_from_shape(
                (*tuple(x.shape[:-1]), w.shape[0], w.shape[1])
            )
        if equation == "...er,eor->...o":
            x, w = tensors
            return _safe_numel_from_shape((*tuple(x.shape[:-2]), w.shape[1]))
        if equation == "...j,eij->...ei":
            x, w = tensors
            return _safe_numel_from_shape(
                (*tuple(x.shape[:-1]), w.shape[0], w.shape[1])
            )
        if equation == "ejr,...er->...j":
            w, x = tensors
            return _safe_numel_from_shape((*tuple(x.shape[:-2]), w.shape[1]))
        if equation in {
            "be,eor->bor",
            "be,eod->bod",
            "bc,cor->bor",
            "bf,for->bor",
        }:
            gate, w = tensors
            return _safe_numel_from_shape((gate.shape[0], w.shape[1], w.shape[2]))
    except Exception:
        return None
    return None


def _adapter_aware_estimated_output_numel(op, args, kwargs) -> Optional[int]:
    name = str(op)
    try:
        if name == "aten.mm.default":
            return _safe_numel_from_shape((args[0].shape[0], args[1].shape[1]))
        if name == "aten.addmm.default":
            return _safe_numel_from_shape((args[1].shape[0], args[2].shape[1]))
        if name == "aten.bmm.default":
            return _safe_numel_from_shape(
                (args[0].shape[0], args[0].shape[1], args[1].shape[2])
            )
        if name == "aten.matmul.default":
            return _matmul_output_numel(args[0], args[1])
        if name == "aten.linear.default":
            inp, weight = args[0], args[1]
            return _safe_numel_from_shape((*tuple(inp.shape[:-1]), weight.shape[0]))
        if name == "aten.linalg_solve.default":
            return _tensor_numel(args[1])
        if name in {"aten.linalg_solve_ex.default", "aten._linalg_solve_ex.default"}:
            return _tensor_numel(args[1])
        if name == "aten.einsum.default":
            operands = args[1] if len(args) > 1 else kwargs.get("operands")
            return _einsum_output_numel(str(args[0]), operands)
        if name == "aten.cat.default":
            return sum((_tensor_numel(t) or 0) for t in args[0])
        if name == "aten.stack.default":
            return sum((_tensor_numel(t) or 0) for t in args[0])
        if name == "aten.expand.default":
            return _safe_numel_from_shape(args[1])
    except Exception:
        return None

    # Unary / elementwise / view-like ops: output is no larger than the first
    # tensor input.  Saving small rank/router tensors here lets LoRA custom
    # autograd boundaries keep their cheap intermediates while large DiT
    # activations still recompute.
    first_tensor = next(_iter_tensors(args), None)
    if first_tensor is None:
        return None
    same_shape_prefixes = (
        "aten._to_copy.",
        "aten.clone.",
        "aten.contiguous.",
        "aten.detach.",
        "aten.view.",
        "aten.reshape.",
        "aten.permute.",
        "aten.transpose.",
        "aten.t.",
        "aten.unsqueeze.",
        "aten.squeeze.",
        "aten.slice.",
        "aten.select.",
        "aten.add.",
        "aten.sub.",
        "aten.mul.",
        "aten.div.",
        "aten.neg.",
        "aten.pow.",
        "aten.sqrt.",
        "aten.rsqrt.",
        "aten.gelu.",
        "aten.silu.",
        "aten.relu.",
        "aten._softmax.",
        "aten.softmax.",
        "aten.dropout.",
        "aten.native_dropout.",
        "aten.mean.",
        "aten.sum.",
    )
    if name.startswith(same_shape_prefixes):
        if name.startswith(("aten.add.", "aten.sub.", "aten.mul.", "aten.div.")):
            tensor_args = [t for t in _iter_tensors(args)]
            return _broadcast_shape_numel(*[tuple(t.shape) for t in tensor_args])
        return _tensor_numel(first_tensor)
    return None


def _adapter_aware_checkpoint_policy(max_save_numel: int):
    max_save_numel = int(max(0, max_save_numel))

    def policy(ctx, op, *args, **kwargs):
        if max_save_numel <= 0:
            return CheckpointPolicy.PREFER_RECOMPUTE
        if not any(getattr(t, "requires_grad", False) for t in _iter_tensors(args)):
            return CheckpointPolicy.PREFER_RECOMPUTE
        numel = _adapter_aware_estimated_output_numel(op, args, kwargs)
        if numel is not None and 0 < numel <= max_save_numel:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy


def _adapter_aware_checkpoint_context(
    max_save_numel: int = ADAPTER_AWARE_CHECKPOINT_MAX_SAVE_NUMEL,
):
    """Selective activation-checkpoint context for adapter training.

    Large DiT activations (attention Q/K/V, attention outputs, MLP hidden
    states) are recomputed.  Small trainable intermediates are cached: LoRA
    rank activations, router logits/weights, and Ortho/FeRA small matrices.
    """
    return create_selective_checkpoint_contexts(
        _adapter_aware_checkpoint_policy(max_save_numel)
    )


@torch.compiler.disable(recursive=True)
def _unflatten_native_shape(x, flatten_info):
    """Restore the fake-5D flattened sequence back to (B, T, H, W, D).

    Disabled from dynamo tracing on purpose: flatten_info is a 4-tuple of Python
    ints (T_s, H_s, W_s, seq_len) computed from the input's pre-flatten shape, so
    if this ran inside the compiled frame each bucket would specialize
    ``flatten_info[1] == H_s`` (per-value guard) and narrow the symbolic range
    on ``flatten_info[3]`` (per-bucket seq_len guard). Running it eagerly keeps
    the returned tensor's shape as the only signal crossing back into the
    compile zone — downstream ops (final_layer, unpatchify) then pick up
    symbolic T/H/W from the tensor itself, not from Python ints.
    """
    T_s, H_s, W_s, seq_len = flatten_info
    x = x.squeeze(3).squeeze(1)
    x = x[:, :seq_len, :]
    x = x.unflatten(1, (T_s, H_s, W_s))
    return x


from library.log import setup_logging  # noqa: E402

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# Utility functions: RoPE for DiT
def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def apply_rotary_pos_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
    tensor_format: str = "sbhd",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k using precomputed (cos, sin) tensors.

    RoPE is generated for the same sequence length as the current q/k. Avoid
    slicing by a symbolic ``cur_seq_len`` here: under ``compile_dynamic_seq`` the
    seq axis is already marked dynamic, and an extra symbolic slice can create a
    different graph between forward and checkpoint recompute.
    """
    cos_, sin_ = rope_cos_sin

    if tensor_format == "bshd":
        cos_ = cos_.transpose(0, 1)
        sin_ = sin_.transpose(0, 1)

    rot_dim = cos_.shape[-1]

    cos_q = cos_.to(q.dtype)
    sin_q = sin_.to(q.dtype)
    # For Anima, dim_t+dim_h+dim_w sum to head_dim by construction (see the
    # assert in VideoRopePosition3DEmb.__init__), so rot_dim == head_dim and the
    # pass-through slice is empty. Skip the torch.cat in that case — it would
    # otherwise allocate+copy a full (B,L,H,D) tensor per Q/K per block for
    # nothing. `rot_dim == q.shape[-1]` is a compile-time constant (head_dim
    # never varies across buckets), so the branch resolves once under
    # torch.compile and never recompiles per bucket.
    q_rot = q[..., :rot_dim]
    q_emb = (q_rot * cos_q) + (_rotate_half(q_rot, False) * sin_q)
    q = q_emb if rot_dim == q.shape[-1] else torch.cat((q_emb, q[..., rot_dim:]), dim=-1)

    cos_k = cos_q if k.dtype == q.dtype else cos_.to(k.dtype)
    sin_k = sin_q if k.dtype == q.dtype else sin_.to(k.dtype)
    k_rot = k[..., :rot_dim]
    k_emb = (k_rot * cos_k) + (_rotate_half(k_rot, False) * sin_k)
    k = k_emb if rot_dim == k.shape[-1] else torch.cat((k_emb, k[..., rot_dim:]), dim=-1)

    return q, k


# Basic building blocks
class RMSNorm(torch.nn.Module):
    """RMS Normalization for DiT blocks."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variance in fp32 for numerical stability, then immediately restore
        # ``x.dtype`` *before* the affine multiply so inductor cannot keep a
        # float32 Q/K path into FlashAttention (compile + ConvRot scope=all).
        # Equivalent to ``(self._norm(x.float()) * weight).to(x.dtype)`` up to
        # half-precision rounding on the final mul.
        output = self._norm(x.float()).to(dtype=x.dtype)
        return output * self.weight.to(dtype=x.dtype)


class GPT2FeedForward(nn.Module):
    """GELU feedforward network."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)
        self.layer1_checkpointing = False

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def _layer1_activation(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        return self.activation(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.layer1_checkpointing:
            x = torch_checkpoint(
                self._layer1_activation,
                x,
                use_reentrant=False,
            )
        else:
            x = self._layer1_activation(x)
        x = self.layer2(x)
        return x


# Attention module for DiT
class Attention(nn.Module):
    """Multi-head attention supporting both self-attention and cross-attention.

    Uses QK-norm (RMSNorm on q/k) and optional RoPE (only for self-attention).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 16,
        head_dim: int = 128,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        if self.is_selfattn:
            self.qkv_proj = nn.Linear(query_dim, 3 * inner_dim, bias=False)
        else:
            self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
            self.kv_proj = nn.Linear(context_dim, 2 * inner_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        if self.is_selfattn:
            # Self-attention: query_dim == context_dim, single std for fused QKV
            std = 1.0 / math.sqrt(self._query_dim)
            torch.nn.init.trunc_normal_(
                self.qkv_proj.weight, std=std, a=-3 * std, b=3 * std
            )
        else:
            std = 1.0 / math.sqrt(self._query_dim)
            torch.nn.init.trunc_normal_(
                self.q_proj.weight, std=std, a=-3 * std, b=3 * std
            )
            std = 1.0 / math.sqrt(self._context_dim)
            torch.nn.init.trunc_normal_(
                self.kv_proj.weight, std=std, a=-3 * std, b=3 * std
            )

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(
            self.output_proj.weight, std=std, a=-3 * std, b=3 * std
        )

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple:
        if self.is_selfattn:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.unflatten(-1, (3, self.n_heads, self.head_dim)).unbind(dim=-3)
        else:
            q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim))
            kv = self.kv_proj(context)
            k, v = kv.unflatten(-1, (2, self.n_heads, self.head_dim)).unbind(dim=-3)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_cos_sin is not None:
            q, k = apply_rotary_pos_emb_qk(
                q, k, rope_cos_sin, tensor_format=self.qkv_format
            )

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        attn_params: attention_dispatch.AttentionParams,
        context: torch.Tensor,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        attn_params = attn_params.for_attention_kind(is_selfattn=self.is_selfattn)
        kind = "self_attn" if self.is_selfattn else "cross_attn"
        check_finite = _finite_checks_enabled(attn_params.debug_finite_checks) or (
            attn_params.v100_flash_stability == "safe"
            and attn_params.attn_mode == "flash"
        )

        q, k, v = self.compute_qkv(x, context, rope_cos_sin=rope_cos_sin)
        if check_finite:
            _assert_finite_tensor(
                q, f"{kind}.q before attention backend={attn_params.attn_mode}"
            )
            _assert_finite_tensor(
                k, f"{kind}.k before attention backend={attn_params.attn_mode}"
            )
            _assert_finite_tensor(
                v, f"{kind}.v before attention backend={attn_params.attn_mode}"
            )
        # Align Q/K/V dtypes for backends that reject mixed or fp32 stacks.
        # Flash hard-cast also lives in attention_dispatch (compile can still
        # promote all three to fp32); this covers mixed-dtype before dispatch.
        if not attn_params.supports_fp32 or attn_params.requires_same_dtype:
            half = (torch.float16, torch.bfloat16)
            if q.dtype != v.dtype or k.dtype != v.dtype or q.dtype not in half:
                if v.dtype in half:
                    target_dtype = v.dtype
                elif torch.is_autocast_enabled():
                    get_dtype = getattr(torch, "get_autocast_dtype", None)
                    if get_dtype is not None:
                        try:
                            target_dtype = get_dtype("cuda")
                        except (TypeError, RuntimeError, ValueError):
                            target_dtype = torch.bfloat16
                    else:
                        get_gpu = getattr(torch, "get_autocast_gpu_dtype", None)
                        target_dtype = (
                            get_gpu() if get_gpu is not None else torch.bfloat16
                        )
                    if target_dtype not in half:
                        target_dtype = torch.bfloat16
                else:
                    target_dtype = torch.bfloat16
                if q.dtype != target_dtype:
                    q = q.to(dtype=target_dtype)
                if k.dtype != target_dtype:
                    k = k.to(dtype=target_dtype)
                if v.dtype != target_dtype:
                    v = v.to(dtype=target_dtype)
        # return self.compute_attention(q, k, v)
        qkv = [q, k, v]
        del q, k, v
        result = attention_dispatch.dispatch_attention(qkv, attn_params=attn_params)
        if check_finite:
            _assert_finite_tensor(
                result,
                f"{kind}.attention_output backend={attn_params.attn_mode}",
            )
        out = self.output_dropout(self.output_proj(result))
        if check_finite:
            _assert_finite_tensor(
                out,
                f"{kind}.output_projection backend={attn_params.attn_mode}",
            )
        return out


# Positional Embeddings
class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(
        self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B_T_H_W_C = x_B_T_H_W_C.shape
        return self.generate_embeddings(B_T_H_W_C, fps=fps)

    def generate_embeddings(
        self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]
    ) -> Any:
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    """3D Rotary Position Embedding for video (T, H, W) dimensions."""

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.register_buffer(
            "seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float)
        )
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, (
            f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        )
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self._cos_sin_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = (
            torch.arange(max(self.max_h, self.max_w, self.max_t))
            .float()
            .to(self.dim_spatial_range.device)
        )
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)]
            .float()
            .to(self.dim_spatial_range.device)
            / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)]
            .float()
            .to(self.dim_spatial_range.device)
            / dim_t
        )

    def _cache_key(
        self,
        T: int,
        H: int,
        W: int,
        fps: Optional[torch.Tensor],
        h_offset: int,
        w_offset: int,
    ) -> tuple:
        fps_val = None if fps is None else fps[:1].item()
        return (T, H, W, fps_val, h_offset, w_offset)

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, W, _ = B_T_H_W_C

        # Skip Python dict cache inside compiled code — dict mutations cause dynamo
        # guard failures and recompilation.  The RoPE computation is pure tensor math
        # that dynamo traces cleanly; under native flatten the block graph keys on
        # token count so there is no recompilation from shape guards.
        _compiling = torch.compiler.is_compiling()

        if not _compiling:
            if h_ntk_factor is None and w_ntk_factor is None and t_ntk_factor is None:
                key = self._cache_key(T, H, W, fps, 0, 0)
                cached = self._cos_sin_cache.get(key)
                if cached is not None:
                    return cached

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        )
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            if fps is None:
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(
                    self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs
                )
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        freqs = em_T_H_W_D.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        result = (torch.cos(freqs), torch.sin(freqs))

        if not _compiling:
            if (
                h_ntk_factor == self.h_ntk_factor
                and w_ntk_factor == self.w_ntk_factor
                and t_ntk_factor == self.t_ntk_factor
            ):
                key = self._cache_key(T, H, W, fps, 0, 0)
                self._cos_sin_cache[key] = result

        return result

    def generate_embeddings_with_offset(
        self,
        B_T_H_W_C: torch.Size,
        h_offset: int = 0,
        w_offset: int = 0,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate RoPE (cos, sin) with spatial offsets for tiled diffusion."""
        B, T, H, W, _ = B_T_H_W_C

        _compiling = torch.compiler.is_compiling()

        if not _compiling:
            if h_ntk_factor is None and w_ntk_factor is None and t_ntk_factor is None:
                key = self._cache_key(T, H, W, fps, h_offset, w_offset)
                cached = self._cos_sin_cache.get(key)
                if cached is not None:
                    return cached

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        assert h_offset + H <= self.max_h, (
            f"h_offset + H ({h_offset + H}) exceeds max_h ({self.max_h})"
        )
        assert w_offset + W <= self.max_w, (
            f"w_offset + W ({w_offset + W}) exceeds max_w ({self.max_w})"
        )
        half_emb_h = torch.outer(self.seq[h_offset : h_offset + H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[w_offset : w_offset + W], w_spatial_freqs)

        # Temporal dimension always starts at 0
        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            if fps is None:
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(
                    self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs
                )
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        freqs = em_T_H_W_D.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        result = (torch.cos(freqs), torch.sin(freqs))

        if not _compiling:
            if (
                h_ntk_factor == self.h_ntk_factor
                and w_ntk_factor == self.w_ntk_factor
                and t_ntk_factor == self.t_ntk_factor
            ):
                key = self._cache_key(T, H, W, fps, h_offset, w_offset)
                self._cos_sin_cache[key] = result

        return result

    @property
    def seq_dim(self) -> int:
        return 0


# Timestep Embedding
class Timesteps(nn.Module):
    """Sinusoidal timestep features."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb.to(dtype=in_dtype).reshape(
            timesteps_B_T.shape[0], timesteps_B_T.shape[1], -1
        )


class TimestepEmbedding(nn.Module):
    """Projects timestep features to model dimension, with optional AdaLN-LoRA."""

    def __init__(
        self, in_features: int, out_features: int, use_adaln_lora: bool = False
    ):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(
            self.linear_1.weight, std=std, a=-3 * std, b=3 * std
        )
        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(
            self.linear_2.weight, std=std, a=-3 * std, b=3 * std
        )

    def forward(
        self, sample: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


# Commented out Fourier Features (not used in Anima). Kept for reference.
# class FourierFeatures(nn.Module):
#     """Fourier feature transform: [B] -> [B, D]."""

#     def __init__(self, num_channels: int, bandwidth: int = 1, normalize: bool = False):
#         super().__init__()
#         self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
#         self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
#         self.gain = np.sqrt(2) if normalize else 1
#         self.bandwidth = bandwidth
#         self.num_channels = num_channels
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         generator = torch.Generator()
#         generator.manual_seed(0)
#         self.freqs = 2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
#         self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

#     def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
#         in_dtype = x.dtype
#         x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
#         x = x.cos().mul(self.gain * gain).to(in_dtype)
#         return x


# Patch Embedding
class PatchEmbed(nn.Module):
    """Patch embedding: (B, C, T, H, W) -> (B, T', H', W', D)"""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 17,
        out_channels: int = 2048,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels
                * spatial_patch_size
                * spatial_patch_size
                * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )
        self.dim = (
            in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0, (
            f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


# Final Layer
class FinalLayer(nn.Module):
    """Final layer with AdaLN modulation + unpatchify."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size
            * spatial_patch_size
            * temporal_patch_size
            * out_channels,
            bias=False,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(
                    adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False
                ),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False),
            )

        # Runtime flag, not weight state. Must not live in init_weights().
        self.fp32_residual = False

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(
                self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std
            )
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def _fp32_project(self, x_modulated: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x_modulated.device.type, enabled=False):
            return torch.nn.functional.linear(x_modulated, self.linear.weight.float())

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        use_fp32: bool = False,
    ):
        mod_input = emb_B_T_D.float() if use_fp32 else emb_B_T_D
        use_autocast_fp32 = use_fp32 and x_B_T_H_W_D.device.type != "cpu"
        with torch.autocast(
            device_type=x_B_T_H_W_D.device.type,
            dtype=torch.float32,
            enabled=use_autocast_fp32,
        ):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(mod_input)
                    + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size].to(mod_input.dtype)
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(mod_input).chunk(
                    2, dim=-1
                )

        shift_B_T_1_1_D = shift_B_T_D[:, :, None, None, :]
        scale_B_T_1_1_D = scale_B_T_D[:, :, None, None, :]

        if self.fp32_residual:
            normed = self.layer_norm(x_B_T_H_W_D)
            x_modulated = (
                normed.float() * (1.0 + scale_B_T_1_1_D.float())
                + shift_B_T_1_1_D.float()
            )
            return self._fp32_project(x_modulated)

        x_B_T_H_W_D = (
            self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
        )
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


# Transformer Block (DiT Block)
class Block(nn.Module):
    """Transformer block with self-attention + cross-attention + MLP, each modulated by AdaLN.

    Each sublayer: x = x + gate * sublayer(norm(x) * (1 + scale) + shift)
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-6
        )
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_cross_attn = nn.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-6
        )
        self.cross_attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_fused_down = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * adaln_lora_dim, bias=False),
            )
            self.adaln_up_self_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
            self.adaln_up_cross_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
            self.adaln_up_mlp = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = False
        self.adapter_aware_checkpoint_max_save_numel = (
            ADAPTER_AWARE_CHECKPOINT_MAX_SAVE_NUMEL
        )
        self.mlp_checkpointing = False
        self.mlp.layer1_checkpointing = False
        self._peak_probe = None
        self._block_idx = None
        self.fp32_residual = False

    def enable_gradient_checkpointing(
        self, cpu_offload: bool = False, unsloth_offload: bool = False
    ):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp.layer1_checkpointing = False

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp.layer1_checkpointing = False

    def enable_mlp_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = True
        self.mlp.layer1_checkpointing = False

    def enable_mlp_layer1_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp.layer1_checkpointing = True

    def enable_adapter_aware_checkpointing(
        self,
        *,
        max_save_numel: int = ADAPTER_AWARE_CHECKPOINT_MAX_SAVE_NUMEL,
    ) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = True
        self.adapter_aware_checkpoint_max_save_numel = int(max_save_numel)
        self.mlp_checkpointing = False
        self.mlp.layer1_checkpointing = False

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(
                self.adaln_fused_down[1].weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            torch.nn.init.zeros_(self.adaln_up_self_attn.weight)
            torch.nn.init.zeros_(self.adaln_up_cross_attn.weight)
            torch.nn.init.zeros_(self.adaln_up_mlp.weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    # Production `_forward` uses `_gated_residual_add`; this is the non-gated helper.
    def _residual_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self.fp32_residual:
            return a + b
        return a.float() + b.float()

    def _gated_residual_add(
        self,
        residual: torch.Tensor,
        gate: torch.Tensor,
        branch: torch.Tensor,
    ) -> torch.Tensor:
        if not self.fp32_residual:
            return residual + gate * branch
        return residual.float() + gate.float() * branch.float()

    def _forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention_dispatch.AttentionParams,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        use_fp32: bool = False,
    ) -> torch.Tensor:
        peak_probe = self._peak_probe
        record_ops = bool(getattr(peak_probe, "record_block_ops", False))
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "block_enter",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="block",
            )
        if use_fp32:
            x_B_T_H_W_D = x_B_T_H_W_D.float()
        mod_input = emb_B_T_D.float() if use_fp32 else emb_B_T_D
        use_autocast_fp32 = use_fp32 and x_B_T_H_W_D.device.type != "cpu"
        # Compute AdaLN modulation parameters
        with torch.autocast(
            device_type=x_B_T_H_W_D.device.type,
            dtype=torch.float32,
            enabled=use_autocast_fp32,
        ):
            if self.use_adaln_lora:
                fused_down = self.adaln_fused_down(mod_input)
                down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
                adaln_lora = (
                    adaln_lora_B_T_3D.to(down_self.dtype)
                    if adaln_lora_B_T_3D is not None
                    else None
                )
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_up_self_attn(down_self) + adaln_lora
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_up_cross_attn(down_cross) + adaln_lora
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_up_mlp(down_mlp) + adaln_lora
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(mod_input).chunk(3, dim=-1)
                )
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(mod_input).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(mod_input).chunk(3, dim=-1)
                )

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        shift_self_attn_B_T_1_1_D = shift_self_attn_B_T_D[:, :, None, None, :]
        scale_self_attn_B_T_1_1_D = scale_self_attn_B_T_D[:, :, None, None, :]
        gate_self_attn_B_T_1_1_D = gate_self_attn_B_T_D[:, :, None, None, :]

        shift_cross_attn_B_T_1_1_D = shift_cross_attn_B_T_D[:, :, None, None, :]
        scale_cross_attn_B_T_1_1_D = scale_cross_attn_B_T_D[:, :, None, None, :]
        gate_cross_attn_B_T_1_1_D = gate_cross_attn_B_T_D[:, :, None, None, :]

        shift_mlp_B_T_1_1_D = shift_mlp_B_T_D[:, :, None, None, :]
        scale_mlp_B_T_1_1_D = scale_mlp_B_T_D[:, :, None, None, :]
        gate_mlp_B_T_1_1_D = gate_mlp_B_T_D[:, :, None, None, :]

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln_fn(_x, _norm_layer, _scale, _shift):
            return _norm_layer(_x) * (1 + _scale) + _shift

        # 1. Self-attention
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "self_attn_before_norm",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="self_attn",
            )
        normalized_x = _adaln_fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        x_flat = normalized_x.flatten(1, 3)
        result = self.self_attn(
            x_flat,
            attn_params,
            x_flat,
            rope_cos_sin=rope_cos_sin,
        ).unflatten(1, (T, H, W))
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "self_attn_after_projection",
                tensor=result,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="self_attn",
            )
        x_B_T_H_W_D = self._gated_residual_add(
            x_B_T_H_W_D, gate_self_attn_B_T_1_1_D, result
        )
        if _finite_checks_enabled(attn_params.debug_finite_checks):
            _assert_finite_tensor(
                x_B_T_H_W_D, "self_attn_residual", block=self
            )
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "self_attn_after_residual",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="self_attn",
            )

        # 2. Cross-attention
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "cross_attn_before_norm",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="cross_attn",
            )
        normalized_x = _adaln_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        result = self.cross_attn(
            normalized_x.flatten(1, 3),
            attn_params,
            crossattn_emb,
            rope_cos_sin=rope_cos_sin,
        ).unflatten(1, (T, H, W))
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "cross_attn_after_projection",
                tensor=result,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="cross_attn",
            )
        x_B_T_H_W_D = self._gated_residual_add(
            x_B_T_H_W_D, gate_cross_attn_B_T_1_1_D, result
        )
        if _finite_checks_enabled(attn_params.debug_finite_checks):
            _assert_finite_tensor(
                x_B_T_H_W_D, "cross_attn_residual", block=self
            )
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "cross_attn_after_residual",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="cross_attn",
            )

        # 3. MLP
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "mlp_before_norm",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="mlp",
            )
        normalized_x = _adaln_fn(
            x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D
        )
        if self.training and self.mlp_checkpointing:
            result = torch_checkpoint(
                self.mlp,
                normalized_x,
                use_reentrant=False,
            )
        else:
            result = self.mlp(normalized_x)
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "mlp_after_projection",
                tensor=result,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="mlp",
            )
        x_B_T_H_W_D = self._gated_residual_add(
            x_B_T_H_W_D, gate_mlp_B_T_1_1_D, result
        )
        if _finite_checks_enabled(attn_params.debug_finite_checks):
            _assert_finite_tensor(x_B_T_H_W_D, "mlp_residual", block=self)
        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "mlp_after_residual",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="mlp",
            )

        if record_ops:
            record_peak_probe_event(
                peak_probe,
                "block_exit",
                tensor=x_B_T_H_W_D,
                module_type="block",
                block_idx=self._block_idx,
                block_phase="block",
            )
        return x_B_T_H_W_D

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention_dispatch.AttentionParams,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        use_fp32: bool = False,
    ) -> torch.Tensor:
        if (
            torch.is_grad_enabled()
            and self.training
            and self.adapter_aware_checkpointing
        ):
            return torch_checkpoint(
                self._forward,
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                rope_cos_sin,
                adaln_lora_B_T_3D,
                use_fp32,
                use_reentrant=False,
                context_fn=lambda: _adapter_aware_checkpoint_context(
                    self.adapter_aware_checkpoint_max_save_numel
                ),
            )
        if torch.is_grad_enabled() and self.training and self.gradient_checkpointing:
            if self.unsloth_offload_checkpointing:
                # Unsloth: async non-blocking CPU RAM offload (fastest offload method)
                return unsloth_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    use_fp32,
                )
            elif self.cpu_offload_checkpointing:
                # Standard cpu offload: blocking transfers
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        # Determine original device from first tensor input
                        device = next(
                            t.device for t in inputs if isinstance(t, torch.Tensor)
                        )
                        device_inputs = to_device(inputs, device)
                        outputs = func(*device_inputs)
                        return to_cpu(outputs)

                    return custom_forward

                return torch_checkpoint(
                    create_custom_forward(self._forward),
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    use_fp32,
                    use_reentrant=False,
                )
            else:
                # Standard gradient checkpointing (no offload)
                return torch_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    use_fp32,
                    use_reentrant=False,
                )
        else:
            return self._forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                rope_cos_sin,
                adaln_lora_B_T_3D,
                use_fp32,
            )


# Main DiT Model: MiniTrainDIT (renamed to Anima)
class Anima(ExplicitBlockSwapMixin, nn.Module):
    """Cosmos-Predict2 DiT model for image/video generation.

    28 transformer blocks with AdaLN-LoRA modulation, 3D RoPE, and optional LLM Adapter.
    """

    LATENT_CHANNELS = 16
    VAE_SPATIAL_COMPRESSION = ANIMA_VAE_SPATIAL_COMPRESSION
    _block_swap_switch_banner = "Anima"

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        vae_spatial_compression: int = ANIMA_VAE_SPATIAL_COMPRESSION,
        concat_padding_mask: bool = True,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_learnable: bool = True,
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 4.0,
        rope_w_extrapolation_ratio: float = 4.0,
        rope_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = False,
        use_llm_adapter: bool = True,
        attn_mode: str = "torch",
        attn_softmax_scale: Optional[float] = None,
        v100_flash_stability: str = "off",
        debug_finite_checks: bool = False,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.vae_spatial_compression = vae_spatial_compression
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_learnable = pos_emb_learnable
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.use_llm_adapter = use_llm_adapter

        self.attn_mode = attn_mode
        self.attn_softmax_scale = attn_softmax_scale
        self.v100_flash_stability = v100_flash_stability
        self.debug_finite_checks = debug_finite_checks

        # Block swap support
        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None
        # Stashed blocks_to_swap while paused (e.g. during eval). None = not paused.
        self._paused_blocks_to_swap: Optional[int] = None
        self.selective_checkpoint = "off"
        self._peak_probe = None

        # Native-shape flattening for torch.compile. Flipped True by
        # compile_blocks(): the forward flattens each bucket's patch sequence to
        # a fake-5D (B, 1, seq_len, 1, D) shape so dynamo keys the block graph on
        # token count alone (2 families) instead of guarding H and W separately
        # (one graph per resolution). Eager forwards leave it False and skip the
        # reshape — bit-exact to the flattened path, slightly cheaper.
        self._native_flatten: bool = False
        # Dynamic-seq state is populated by compile_blocks().  A classic union
        # compile stores one band; per-band mode stores one tight range per
        # cluster.  Static/eager paths keep the value unset.
        self._dynamic_seq: bool = False
        self._dynamic_seq_range: Optional[tuple[int, int]] = None
        self._dynamic_seq_bands: Optional[list[tuple[int, int]]] = None
        self._dynamic_seq_strict_marks: bool = False

        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(
                model_channels, model_channels, use_adaln_lora=use_adaln_lora
            ),
        )

        if self.use_llm_adapter:
            self.llm_adapter = LLMAdapter(
                source_dim=1024,
                target_dim=1024,
                model_dim=1024,
                num_layers=6,
                self_attn=True,
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        for idx, block in enumerate(self.blocks):
            block._block_idx = idx

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        # Modulation guidance: project pooled crossattn_emb into modulation space.
        # Zero-initialized so it's a no-op before distillation training.
        self.pooled_text_proj = nn.Sequential(
            nn.Linear(crossattn_emb_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        # Whether the per-forward pooled_text_proj path runs. Default off: the
        # base DiT checkpoint never carries these weights (re-zeroed on load), so
        # the proj is a no-op and the max-reduce + 2 linears are pure per-step
        # overhead. Flipped True only where the module becomes active —
        # ``load_pooled_text_proj`` (inference / DCW) and distill-mod training.
        # A plain Python bool set once at load: it guards once under compile (like
        # ``_native_flatten``) and never toggles per-forward, so no recompile churn.
        self.enable_pooled_text_modulation = False

        # Modulation guidance runtime state as non-persistent buffers (zeros = off).
        # Registered unconditionally so the forward can do unconditional arithmetic
        # (``t_emb + schedule[l] * delta``) without a Python-level None/zero branch —
        # branches here guard-fire under torch.compile per bucket/block combination.
        # Setters live on library/inference/mod_guidance.py; reset via reset_mod_guidance().
        self.register_buffer(
            "_mod_guidance_delta",
            torch.zeros(1, model_channels),
            persistent=False,
        )
        self.register_buffer(
            "_mod_guidance_schedule",
            torch.zeros(num_blocks),
            persistent=False,
        )
        self.register_buffer(
            "_mod_guidance_final_w",
            torch.zeros(()),
            persistent=False,
        )

        self.init_weights()

    def reset_mod_guidance(self) -> None:
        """Disable modulation guidance by zeroing the runtime buffers."""
        self._mod_guidance_delta.zero_()
        self._mod_guidance_schedule.zero_()
        self._mod_guidance_final_w.zero_()

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()
        # Zero-init pooled_text_proj output layer so it's a no-op at init
        nn.init.zeros_(self.pooled_text_proj[-1].weight)
        nn.init.zeros_(self.pooled_text_proj[-1].bias)

    def enable_gradient_checkpointing(
        self, cpu_offload: bool = False, unsloth_offload: bool = False
    ):
        for block in self.blocks:
            block.enable_gradient_checkpointing(
                cpu_offload=cpu_offload, unsloth_offload=unsloth_offload
            )

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.disable_gradient_checkpointing()

    def enable_fp32_residual(self) -> None:
        """Promote residual stream / final projection to fp32-safe path for fp16."""
        for block in self.blocks:
            block.fp32_residual = True
        self.final_layer.fp32_residual = True

    def enable_peak_probe(self, probe=None):
        """Attach an opt-in fine-grained peak probe to DiT blocks."""
        self._peak_probe = probe
        for idx, block in enumerate(self.blocks):
            block._peak_probe = probe
            block._block_idx = idx

    @staticmethod
    def _parse_selective_checkpoint_blocks(
        blocks_spec: Optional[str],
        *,
        num_blocks: int,
        default_last: int = 3,
    ) -> set[int]:
        value = str(blocks_spec or "").strip().lower()
        if value in {"", "auto", "last3"}:
            start = max(0, int(num_blocks) - int(default_last))
            return set(range(start, int(num_blocks)))

        out: set[int] = set()
        for raw_part in value.replace(";", ",").split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
                if start > end:
                    start, end = end, start
                out.update(range(start, end + 1))
            else:
                out.add(int(part))
        invalid = sorted(idx for idx in out if idx < 0 or idx >= int(num_blocks))
        if invalid:
            raise ValueError(
                f"selective_checkpoint_blocks contains invalid block index {invalid}; "
                f"valid range is 0..{int(num_blocks) - 1}"
            )
        return out

    def enable_selective_checkpointing(
        self,
        mode: str = "off",
        blocks: Optional[str] = None,
    ):
        mode = str(mode or "off").strip().lower()
        valid_modes = {
            "off",
            "adapter_aware",
            "every_other",
            "mlp_only",
            "mlp_layer1_only",
            "peak_blocks_adapter_aware",
            "peak_blocks_mlp",
            "peak_blocks_mlp_layer1",
        }
        if mode not in valid_modes:
            raise ValueError(
                "selective_checkpoint must be one of "
                f"{', '.join(sorted(valid_modes))}; got {mode!r}"
            )
        self.selective_checkpoint = mode
        peak_block_indices: set[int] = set()
        if mode.startswith("peak_blocks_"):
            peak_block_indices = self._parse_selective_checkpoint_blocks(
                blocks,
                num_blocks=len(self.blocks),
            )
        for idx, block in enumerate(self.blocks):
            block.disable_gradient_checkpointing()
            if mode == "adapter_aware":
                block.enable_adapter_aware_checkpointing()
            elif mode == "every_other" and idx % 2 == 0:
                block.enable_gradient_checkpointing(
                    cpu_offload=False, unsloth_offload=False
                )
            elif mode == "mlp_only":
                block.enable_mlp_checkpointing()
            elif mode == "mlp_layer1_only":
                block.enable_mlp_layer1_checkpointing()
            elif mode == "peak_blocks_adapter_aware" and idx in peak_block_indices:
                block.enable_adapter_aware_checkpointing()
            elif mode == "peak_blocks_mlp" and idx in peak_block_indices:
                block.enable_mlp_checkpointing()
            elif mode == "peak_blocks_mlp_layer1" and idx in peak_block_indices:
                block.enable_mlp_layer1_checkpointing()

    def compile_blocks(
        self,
        backend: str = "inductor",
        mode: Optional[str] = None,
        bucket_resolutions: Optional[Sequence[Tuple[int, int]]] = None,
        n_token_families: Optional[int] = None,
        dynamic_seq: bool = False,
        seq_range: Optional[tuple[int, int]] = None,
        compile_block_scope: str = "resident",
        seq_bands: Optional[Sequence[tuple[int, int]]] = None,
    ):
        """Enable native-shape flattening and torch.compile each block's _forward.

        Two coupled effects, both owned by this one call:

        1. Sets ``self._native_flatten = True`` so the forward flattens every
           bucket's patch sequence to a fake-5D ``(B, 1, seq_len, 1, D)`` shape.
           This is what keys the block graph on token count alone — the shipped
           ``CONSTANT_TOKEN_BUCKETS`` collapses to 2 token-count families (the
           4032 and 4200 groups) — instead of guarding H and W separately (one
           graph per resolution, ~24 buckets). Native shapes, no padding, so
           flash self-attention sees no padded tokens (bit-exact to the eager
           5D path; the gap=0 control of the retired pad-leak probe verified it).

        2. Compiles ``_forward`` (the actual attention/MLP computation) rather
           than ``forward`` (the checkpointing wrapper). This is critical because
           unsloth_checkpoint has @torch._disable_dynamo, which causes an
           immediate graph break if forward itself is compiled — dynamo compiles
           nothing useful but still checks shape guards, causing recompile storms.

        **Block-swap coexistence**: by default, when ``self.blocks_to_swap > 0``
        only the resident head blocks are compiled; the tail swap blocks keep
        their eager ``_forward``. The offloader swaps weights via
        ``.weight.data`` reassignment, and Dynamo guards each Parameter on its
        dispatch key (device), so compiling swapped blocks can recompile every
        step. ``compile_block_scope="all"`` restores the older all-block compile
        path for tight-VRAM runs where Inductor's activation planning is worth
        the possible recompilation overhead.

        Also raises the dynamo recompile budget to fit those token-count
        families. ``2 * n + 8``: the ``2 *`` covers fwd+bwd sharing the one
        ``_forward`` bytecode, the ``+ 8`` covers requires_grad / stride /
        num_threads specializations. The raise is pinned across compile contexts
        so checkpoint backward recompute does not fall back to Dynamo's default
        limit of 8. ``bucket_resolutions`` is kept for existing callers;
        ``n_token_families`` lets training pass a pre-derived budget directly.

        ``mode`` maps to torch.compile's inductor preset (e.g. ``reduce-overhead``
        to enable per-block CUDAGraphs). ``None`` leaves it unset (inductor default).

        ``dynamic_seq`` keeps ``torch.compile(dynamic=False)`` but marks only the
        native-flattened sequence axis dynamic via an eager wrapper around the
        compiled inner. The wrapper is the checkpointed callable, so marks are
        re-applied during backward recompute as well as the original forward.
        ``seq_bands`` optionally supplies sorted, non-overlapping ``(lo, hi)``
        ranges. Inputs are dispatched to their containing range, giving each
        token-count cluster a tight dynamic specialization. It is inert when
        ``dynamic_seq`` is false.
        """
        self._native_flatten = True

        # Local import: library.datasets.buckets does not import models, so
        # importing it here (rather than at module top) avoids a circular import.
        from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS
        from library.runtime.dynamo import pin_dynamo_limit

        resos = bucket_resolutions or CONSTANT_TOKEN_BUCKETS

        counts = pixel_bucket_token_counts(
            resos,
            patch_spatial=self.patch_spatial,
            vae_spatial_compression=self.vae_spatial_compression,
        )
        n = int(n_token_families) if n_token_families is not None else len(counts)
        limit = pin_dynamo_limit("recompile_limit", 2 * n + 8)

        self._dynamic_seq = bool(dynamic_seq)
        self._dynamic_seq_range = None
        self._dynamic_seq_bands = None
        self._dynamic_seq_strict_marks = bool(self._dynamic_seq and seq_bands)
        if self._dynamic_seq:
            if seq_range is not None:
                self._dynamic_seq_range = (int(seq_range[0]), int(seq_range[1]))
            else:
                self._dynamic_seq_range = (min(counts), max(counts))
            if seq_bands:
                normalized_bands = sorted(
                    (int(lo), int(hi)) for lo, hi in seq_bands
                )
                for band_lo, band_hi in normalized_bands:
                    if band_lo < 1 or band_hi < band_lo:
                        raise ValueError(
                            f"invalid dynamic-seq token band {(band_lo, band_hi)}"
                        )
                for previous, current in zip(
                    normalized_bands, normalized_bands[1:]
                ):
                    if current[0] <= previous[1]:
                        raise ValueError(
                            "dynamic-seq token bands must be sorted and "
                            f"non-overlapping; got {normalized_bands}"
                        )
                self._dynamic_seq_bands = normalized_bands
            else:
                self._dynamic_seq_bands = [self._dynamic_seq_range]
            # Inductor's mix-order-reduction fusion (torch 2.12, default-on) is
            # incompatible with the strict seq marks: its profitability check
            # calls guard_or_true(Ge(nrow, 4096)) where nrow is the symbolic seq
            # axis (it fires on backward graphs that pair a seq-axis reduction
            # with an elementwise grad — e.g. any LoRA on a broadcast-consumed
            # Linear like adaln_up, whose shift/scale/gate grads reduce over
            # seq). The recorded guard (either branch: Ge(seq, 4096) or its
            # negation seq <= 4095, per the first-traced hint) contradicts any
            # mark range straddling 4096 → ConstraintViolationError at guard
            # build. MUST be pinned via pin_inductor_flag, not plain assignment:
            # inductor config overrides are thread-local ContextVars, and the
            # grad-enabled step-0 compile (grad-ckpt recompute / AOT backward
            # path) schedules in a context where a plain override is absent and
            # the read falls back to the env-derived default True.
            straddles_4096 = (
                any(lo < 4096 <= hi for lo, hi in self._dynamic_seq_bands)
                if self._dynamic_seq_strict_marks
                else True
            )
            import torch._inductor.config as _inductor_config

            if _inductor_config.triton.mix_order_reduction and straddles_4096:
                from library.runtime.dynamo import pin_inductor_flag

                pin_inductor_flag("triton.mix_order_reduction", False)
                print(
                    "Anima: inductor mix_order_reduction disabled — default pinned "
                    "(hint-derived 4096-boundary guard breaks strict dynamic-seq marks)"
                )

        compile_kwargs = {"backend": backend, "dynamic": False}
        if mode is not None:
            compile_kwargs["mode"] = mode
        n_swap = int(self.blocks_to_swap or 0)
        max_swappable = self.num_blocks - 2
        if n_swap < 0 or n_swap > max_swappable:
            raise ValueError(
                f"Invalid blocks_to_swap={n_swap}; expected 0 <= blocks_to_swap <= "
                f"{max_swappable} for {self.num_blocks} blocks."
            )
        compile_scope = str(compile_block_scope or "resident").strip().lower()
        if compile_scope not in {"resident", "all"}:
            raise ValueError("compile_block_scope must be one of: resident, all")
        n_resident = self.num_blocks - n_swap
        n_compile = self.num_blocks if compile_scope == "all" else n_resident
        for block_idx, block in enumerate(self.blocks):
            if block_idx >= n_compile:
                continue
            # Compile the ORIGINAL forward, never the currently-installed one.
            # ensure_training_compile_seq_range re-enters compile_blocks to widen
            # a dynamic-seq range mid-run; without this the second pass would
            # wrap the first pass's compiled (and dynamic-seq-marked) callable,
            # nesting a graph inside a graph and re-marking an already-marked
            # seq axis against the stale bounds.
            if not hasattr(block, "_anima_compile_base_forward"):
                block._anima_compile_base_forward = block._forward
            base_forward = block._anima_compile_base_forward
            compiled_inner = torch.compile(base_forward, **compile_kwargs)
            if self._dynamic_seq:
                block._forward = _make_dynamic_seq_forward(
                    compiled_inner,
                    self._dynamic_seq_bands,
                    strict_marks=self._dynamic_seq_strict_marks,
                )
            else:
                block._forward = compiled_inner
        if (
            self._dynamic_seq_strict_marks
            and len(self._dynamic_seq_bands) > 1
        ):
            graph_mode = (
                f"dynamic-seq per-band mark_dynamic seq∈{self._dynamic_seq_bands} "
                f"({len(self._dynamic_seq_bands)} graphs)"
            )
        elif self._dynamic_seq:
            graph_mode = (
                f"dynamic-seq mark_dynamic seq∈{self._dynamic_seq_range} (1 graph)"
            )
        else:
            graph_mode = f"static ({n} graphs)"
        if n_swap and compile_scope == "all":
            swap_note = f"{n_resident} resident + {n_swap} swapped compiled"
        elif n_swap:
            swap_note = f"{n_resident} resident compiled / {n_swap} swapped (eager)"
        else:
            swap_note = f"{len(self.blocks)} block._forward"
        print(
            f"Anima: native_flatten on, {n} token-count families, {graph_mode} "
            f"(recompile_limit={limit}); compiled {swap_note} "
            f"with backend={backend}, mode={mode}"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def build_patch_embed(self) -> None:
        in_channels = (
            self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        )
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
        )

    def build_pos_embed(self) -> None:
        self.pos_embedder = VideoRopePosition3DEmb(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        h_offset: int = 0,
        w_offset: int = 0,
    ) -> Tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.concat_padding_mask:
            if padding_mask is None:
                raise ValueError(
                    "padding_mask must be provided when concat_padding_mask is True"
                )
            if padding_mask.ndim != 4:
                raise ValueError(
                    f"padding_mask must be 4D (B, 1, H, W), got shape {tuple(padding_mask.shape)}"
                )
            if padding_mask.shape[-2:] != x_B_C_T_H_W.shape[-2:]:
                from torchvision import transforms

                padding_mask = transforms.functional.resize(
                    padding_mask,
                    list(x_B_C_T_H_W.shape[-2:]),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )

            # (B, 1, H, W) -> (B, 1, T, H, W) without materializing a repeated tensor
            padding_mask_B_1_T_H_W = padding_mask.unsqueeze(2).expand(
                -1, -1, x_B_C_T_H_W.shape[2], -1, -1
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask_B_1_T_H_W], dim=1)
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if h_offset != 0 or w_offset != 0:
            rope_cos_sin = self.pos_embedder.generate_embeddings_with_offset(
                x_B_T_H_W_D.shape, h_offset=h_offset, w_offset=w_offset, fps=fps
            )
        else:
            rope_cos_sin = self.pos_embedder(x_B_T_H_W_D, fps=fps)
        return x_B_T_H_W_D, rope_cos_sin

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        B, T, H, W, M = x_B_T_H_W_M.shape
        p1 = self.patch_spatial
        p2 = self.patch_spatial
        pt = self.patch_temporal
        C = M // (p1 * p2 * pt)
        # (B,T,H,W, p1*p2*pt*C) → (B,T,H,W, p1,p2,pt,C) → (B,C, T,pt, H,p1, W,p2)
        #                                                    → (B,C, T*pt, H*p1, W*p2)
        x_B_C_Tt_Hp_Wp = (
            x_B_T_H_W_M.unflatten(-1, (p1, p2, pt, C))
            .permute(0, 7, 1, 6, 2, 4, 3, 5)
            .reshape(B, C, T * pt, H * p1, W * p2)
        )
        return x_B_C_Tt_Hp_Wp

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        **offloader_kwargs,
    ):
        self._init_explicit_block_swap(
            num_blocks,
            device,
            offloader_cls=custom_offloading_utils.ModelOffloader,
            **offloader_kwargs,
        )
        logger.info(
            "Anima: Block swap enabled. "
            f"Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, "
            f"device: {device}, transfer_dtype: {self.offloader.transfer_dtype}, "
            f"restore_mode: {self.offloader.restore_mode}."
        )

    def _run_blocks(
        self,
        x_padded: torch.Tensor,
        t_embedding_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params,
        **block_kwargs,
    ) -> torch.Tensor:
        """The block loop — the per-block compiled hot path (see compile_blocks).

        Inputs from the eager pre-blocks region:
        - ``x_padded``: ``(B, 1, seq_len, 1, D)`` when native-flattened, else the
          plain ``(B, T, H, W, D)`` grid (eager forwards skip the flatten)
        - ``t_embedding_B_T_D``: ``(B, 1, D)``
        - ``crossattn_emb``: ``(B, max_text_len, D)`` (padded to max_length)
        - ``attn_params``: attention params (no self-attn mask in native mode)
        - ``block_kwargs["rope_cos_sin"]``: each ``(seq_len, 1, 1, D_head)``
        - ``block_kwargs["adaln_lora_B_T_3D"]``: ``(B, 1, 3, D)``

        Mod-guidance is applied via buffers on ``self`` (zero = off) so the
        per-block ``t_emb`` arithmetic is unconditional. No Python branches.
        """
        # Normalize requires_grad once at the stack entry. Block 0 receives
        # requires_grad=False (frozen patch_embed output) while blocks 1+
        # receive True (LoRA-enhanced); a mismatch would fragment guards if
        # the loop were ever traced per-block. requires_grad_(True) is a no-op
        # under torch.no_grad().
        x = x_padded.requires_grad_()
        peak_probe = self._peak_probe
        record_boundaries = bool(getattr(peak_probe, "record_block_boundaries", False))
        for block_idx, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_idx)

            if record_boundaries:
                record_peak_probe_event(
                    peak_probe,
                    "block_before",
                    tensor=x,
                    module_type="block_boundary",
                    block_idx=block_idx,
                    block_phase="block",
                )

            # Unconditional: zero buffers collapse to identity when guidance
            # is off; avoids a data-dependent branch inside the compiled frame.
            t_emb_block = t_embedding_B_T_D + (
                self._mod_guidance_schedule[block_idx] * self._mod_guidance_delta
            ).unsqueeze(1)

            x = block(
                x,
                t_emb_block,
                crossattn_emb,
                attn_params,
                **block_kwargs,
            )

            if record_boundaries:
                record_peak_probe_event(
                    peak_probe,
                    "block_after",
                    tensor=x,
                    module_type="block_boundary",
                    block_idx=block_idx,
                    block_phase="block",
                )

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, block_idx)
        return x

    def forward_mini_train_dit(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attn_mask: Optional[torch.Tensor] = None,
        crossattn_seqlens: Optional[torch.Tensor] = None,
        h_offset: int = 0,
        w_offset: int = 0,
        pooled_text_override: Optional[torch.Tensor] = None,
        skip_pooled_text_proj: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x_B_C_T_H_W: (B, C, T, H, W) noisy latents
            timesteps_B_T: (B,) or (B, T) timesteps
            crossattn_emb: (B, N, D) cross-attention embeddings (or raw Qwen3 prompt_embeds if t5_input_ids provided)
            fps: Optional frames per second
            padding_mask: Optional padding mask
            source_attention_mask: Optional attention mask for Qwen3 embeddings (used with LLM adapter)
            t5_input_ids: Optional T5 token IDs (triggers LLM adapter when provided)
            t5_attn_mask: Optional T5 attention mask
            crossattn_seqlens: Optional per-sample text token counts [B] for flex cross-attention masking
            h_offset: Height offset in patched space for tiled diffusion RoPE
            w_offset: Width offset in patched space for tiled diffusion RoPE
            pooled_text_override: Optional pre-computed pooled text (B, 1024) for modulation guidance.
                Use to decouple modulation from prefix/postfix tokens in crossattn_emb.
        """
        if (
            t5_input_ids is not None
            and self.use_llm_adapter
            and hasattr(self, "llm_adapter")
        ):
            crossattn_emb = self.llm_adapter(
                source_hidden_states=crossattn_emb,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=source_attention_mask,
            )
            if t5_attn_mask is not None:
                crossattn_emb[~t5_attn_mask.bool()] = 0

        x_B_T_H_W_D, rope_cos_sin = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            h_offset=h_offset,
            w_offset=w_offset,
        )

        # --- Native-shape flattening: flatten 5D → fake-5D so the block graph
        # keys on token count, not on H/W separately ---
        # Enabled by compile_blocks(). The fake-5D shape (B, 1, seq_len, 1, D) is
        # compatible with existing Block code because rearrange("b t h w d -> b (t h w) d")
        # with t=1, w=1 produces the same flat sequential order as the original,
        # so the block stack sees a flat (B,1,L,1,D) sequence keyed solely by
        # token count. Otherwise dynamo guards on H and W separately and recompiles
        # per *resolution* (24 buckets) instead of per token-count (2 families).
        # No padding → native flash, bit-exact to the eager 5D path (the gap=0
        # control of the retired pad-leak probe verified it). Eager forwards leave
        # _native_flatten False and skip the reshape entirely.
        _native_flatten_info = None
        if self._native_flatten:
            B_s, T_s, H_s, W_s, D_s = x_B_T_H_W_D.shape
            seq_len = T_s * H_s * W_s
            _native_flatten_info = (T_s, H_s, W_s, seq_len)

            # Flatten 5D → 2D, then reshape to fake-5D: (B, 1, seq_len, 1, D).
            x_B_T_H_W_D = x_B_T_H_W_D.flatten(1, 3)
            x_B_T_H_W_D = x_B_T_H_W_D.unsqueeze(1).unsqueeze(3)

        use_fp32 = x_B_T_H_W_D.dtype == torch.float16

        # Cast RoPE cos/sin to the block compute dtype once. Without this, every
        # block's apply_rotary_pos_emb_qk re-materializes the fp32 cache in bf16.
        if rope_cos_sin is not None:
            compute_dtype = torch.float32 if use_fp32 else x_B_T_H_W_D.dtype
            rope_cos_sin = (
                rope_cos_sin[0].to(compute_dtype),
                rope_cos_sin[1].to(compute_dtype),
            )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Modulation guidance: inject pooled text embedding into modulation path.
        # - pooled_text_override: use this tensor instead of computing from crossattn_emb
        #   (used to decouple modulation from prefix/postfix tokens)
        # - skip_pooled_text_proj: disable entirely (for distillation teacher forward)
        # The enable flag short-circuits the whole max/proj path when no trained
        # pooled_text_proj is loaded (the common LoRA train/infer case) — bit-exact
        # to running it, since the output layer is zero-init there.
        if self.enable_pooled_text_modulation and not skip_pooled_text_proj:
            if pooled_text_override is not None:
                pooled_text = pooled_text_override
            elif crossattn_emb is not None:
                pooled_text = crossattn_emb.max(dim=1).values  # (B, 1024)
            else:
                pooled_text = None
            if pooled_text is not None:
                t_embedding_B_T_D = t_embedding_B_T_D + self.pooled_text_proj(
                    pooled_text
                ).unsqueeze(1)

        # Phase 2: modulation guidance delta.
        # The steering delta (proj_pos - proj_neg) is NOT baked into the shared
        # t_embedding here — it is applied per-block below via _mod_guidance_schedule,
        # so early tonal-DC blocks and the final compensation layer can be skipped.
        # Buffers are zeros when guidance is off (see __init__), so the arithmetic
        # below is an unconditional identity in that case — no Python branch.
        # See docs/methods/mod-guidance.md for the rationale.

        block_kwargs = {
            "rope_cos_sin": rope_cos_sin,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "use_fp32": use_fp32,
        }

        attn_params = attention_dispatch.AttentionParams.create_attention_params(
            self.attn_mode,
            self.attn_softmax_scale,
            v100_flash_stability=self.v100_flash_stability,
            debug_finite_checks=self.debug_finite_checks,
        )

        # Pre-compute cross-attention BlockMask once for all blocks (flex mode only)
        if (
            self.attn_mode == "flex"
            and crossattn_seqlens is not None
            and attention_dispatch.create_block_mask is not None
        ):
            B, T, H, W, _D = x_B_T_H_W_D.shape
            q_len = T * H * W
            kv_len = crossattn_emb.shape[1]
            seqlens = crossattn_seqlens

            def _crossattn_mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < seqlens[b]

            attn_params.crossattn_block_mask = attention_dispatch.create_block_mask(
                _crossattn_mask_mod,
                B,
                None,
                q_len,
                kv_len,
                device=x_B_T_H_W_D.device,
            )

        # No self-attention pad-mask: native shapes never have padded KV
        # positions, so attn_params.selfattn_block_mask stays None (the legacy
        # pad-to-static path masked padded positions; that path is gone).

        # Block stack runs in _run_blocks — a split point kept so pre/post-block
        # regions stay eager while the block loop is the compiled hot path.
        # PyTorch 2.12's generic SDPA lowering can add a seq % 8 guard.  A
        # strict per-band range may contain arbitrary native token counts, so
        # select the alignment-tolerant cuDNN/efficient/mathematical kernels for
        # this dynamic path without changing the token shape or padding it.
        sdpa_scope = nullcontext()
        if self._dynamic_seq_strict_marks and self.attn_mode == "torch":
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                sdpa_scope = sdpa_kernel(
                    [
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ]
                )
            except (ImportError, AttributeError, RuntimeError):
                # Older builds may not expose scoped SDPA selection. Preserve
                # the historical dispatcher there rather than failing startup.
                sdpa_scope = nullcontext()
        with sdpa_scope:
            x_B_T_H_W_D = self._run_blocks(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                attn_params,
                **block_kwargs,
            )

        # --- Native flatten: restore the original 5D shape ---
        # Delegated to a @torch.compiler.disable'd helper so the bucket-
        # dependent tuple (T_s, H_s, W_s, seq_len) never enters the compile
        # zone. See _unflatten_native_shape for rationale.
        if _native_flatten_info is not None:
            x_B_T_H_W_D = _unflatten_native_shape(x_B_T_H_W_D, _native_flatten_info)

        # Unconditional: zero buffers collapse to identity when guidance is off.
        t_emb_final = t_embedding_B_T_D + (
            self._mod_guidance_final_w * self._mod_guidance_delta
        ).unsqueeze(1)
        record_peak_probe_event(
            self._peak_probe,
            "final_layer_before",
            tensor=x_B_T_H_W_D,
            module_type="anima",
            block_idx="final",
            block_phase="final_projection",
            op_name="final_layer",
        )
        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D,
            t_emb_final,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            use_fp32=use_fp32,
        )
        record_peak_probe_event(
            self._peak_probe,
            "final_layer_after",
            tensor=x_B_T_H_W_O,
            module_type="anima",
            block_idx="final",
            block_phase="final_projection",
            op_name="final_layer",
        )
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        record_peak_probe_event(
            self._peak_probe,
            "unpatchify_after",
            tensor=x_B_C_Tt_Hp_Wp,
            module_type="anima",
            block_idx="final",
            block_phase="unpatchify",
            op_name="unpatchify",
        )
        return x_B_C_Tt_Hp_Wp

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        crossattn_seqlens: Optional[torch.Tensor] = None,
        h_offset: int = 0,
        w_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if crossattn_seqlens is None:
            # Compute seqlens from mask inside _preprocess_text_embeds
            context, crossattn_seqlens = self._preprocess_text_embeds(
                context, target_input_ids, target_attention_mask, source_attention_mask
            )
        return self.forward_mini_train_dit(
            x,
            timesteps,
            context,
            fps=fps,
            padding_mask=padding_mask,
            crossattn_seqlens=crossattn_seqlens,
            h_offset=h_offset,
            w_offset=w_offset,
            **kwargs,
        )

    def _preprocess_text_embeds(
        self,
        source_hidden_states,
        target_input_ids,
        target_attention_mask=None,
        source_attention_mask=None,
    ):
        if target_input_ids is not None and target_input_ids.shape[-1] > 0:
            context = self.llm_adapter(
                source_hidden_states,
                target_input_ids,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
            )
            crossattn_mask = target_attention_mask
            # Adapter may have appended T5-side postfix tokens (dual mode) — extend mask to match
            if (
                crossattn_mask is not None
                and context.shape[1] > crossattn_mask.shape[-1]
            ):
                num_extra = context.shape[1] - crossattn_mask.shape[-1]
                extra_mask = torch.ones(
                    crossattn_mask.shape[0],
                    num_extra,
                    device=crossattn_mask.device,
                    dtype=crossattn_mask.dtype,
                )
                crossattn_mask = torch.cat([crossattn_mask, extra_mask], dim=-1)
            context[~crossattn_mask.bool()] = 0  # zero out padding tokens
        else:
            # Adapter skipped (pre-cached output or no adapter) — use source mask
            context = source_hidden_states
            crossattn_mask = source_attention_mask

        # Compute per-sample text token counts from the mask. Used only by the
        # flex-attention BlockMask path (attn_mode="flex"); the default sink-
        # padded attention modes ignore it and treat zero keys as attention
        # sinks, which is what the pretrained model expects.
        crossattn_seqlens = None
        if crossattn_mask is not None:
            crossattn_seqlens = crossattn_mask.sum(dim=-1).to(torch.int32)
        return context, crossattn_seqlens


# LLM Adapter: Bridges Qwen3 embeddings to T5-compatible space
class LLMAdapterRMSNorm(nn.Module):
    """RMSNorm specifically for the LLM Adapter (T5-style, no mean subtraction)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def _adapter_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _adapter_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (_adapter_rotate_half(x) * sin)
    return x_embed


class AdapterRotaryEmbedding(nn.Module):
    """Rotary embedding for LLM Adapter."""

    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # inv_freq is registered as fp32 but a parent .to(bf16) casts it too —
        # force fp32 here so the matmul matches position_ids_expanded.
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    """Attention module for LLM Adapter with QK-norm and separate RoPE for query/key."""

    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = LLMAdapterRMSNorm(self.head_dim)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = LLMAdapterRMSNorm(self.head_dim)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self,
        x,
        q_mask=None,
        kv_mask=None,
        context=None,
        position_embeddings=None,
        position_embeddings_context=None,
    ):
        """
        Args:
            x: Query input [B, L_q, D].
            q_mask: Optional 2-D bool mask [B, L_q] — True = valid token.
            kv_mask: Optional 2-D bool mask [B, L_kv] — True = valid token.
            context: Key/Value input [B, L_kv, D]. Defaults to x (self-attention).
            position_embeddings: (cos, sin) for query RoPE.
            position_embeddings_context: (cos, sin) for key RoPE.
        """
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape))
        key_states = self.k_norm(self.k_proj(context).view(kv_shape))
        value_states = self.v_proj(context).view(kv_shape)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            # RoPE expects [B, H, L, D] layout
            query_states = _adapter_apply_rotary_pos_emb(
                query_states.transpose(1, 2), cos, sin
            ).transpose(1, 2)
            cos, sin = position_embeddings_context
            key_states = _adapter_apply_rotary_pos_emb(
                key_states.transpose(1, 2), cos, sin
            ).transpose(1, 2)

        can_use_flash = attention_dispatch.flash_attn_available_for_tensor(
            query_states
        )
        if (
            not can_use_flash
            and query_states.dtype == torch.bfloat16
            and attention_dispatch.flash_attn_v100_provider
        ):
            can_use_flash = attention_dispatch.flash_attn_supports_outer_dtype(
                query_states.dtype, device=query_states.device
            )

        if can_use_flash and q_mask is None and kv_mask is None:
            # No masking — simple flash attention, [B, L, H, D] layout
            outer_dtype = query_states.dtype
            query_states, key_states, value_states = (
                attention_dispatch._cast_qkv_for_flash(
                    query_states, key_states, value_states
                )
            )
            attn_output = attention_dispatch.flash_attn_func(
                query_states, key_states, value_states
            )
            if (
                attention_dispatch.flash_attn_v100_provider
                and outer_dtype == torch.bfloat16
            ):
                attn_output = attn_output.to(dtype=outer_dtype)
        elif can_use_flash:
            # Varlen flash attention: pack valid tokens, attend, unpack
            B, L_q = query_states.shape[:2]
            L_kv = key_states.shape[1]

            eff_q_mask = (
                q_mask
                if q_mask is not None
                else query_states.new_ones(B, L_q, dtype=torch.bool)
            )
            eff_kv_mask = (
                kv_mask
                if kv_mask is not None
                else key_states.new_ones(B, L_kv, dtype=torch.bool)
            )

            q_seqlens = eff_q_mask.sum(dim=1, dtype=torch.int32)
            kv_seqlens = eff_kv_mask.sum(dim=1, dtype=torch.int32)

            cu_seqlens_q = F.pad(q_seqlens.cumsum(0, dtype=torch.int32), (1, 0))
            cu_seqlens_kv = F.pad(kv_seqlens.cumsum(0, dtype=torch.int32), (1, 0))

            # Pack by removing padding: [B, L, H, D] → [total_valid, H, D]
            q_packed = query_states[eff_q_mask]
            k_packed = key_states[eff_kv_mask]
            v_packed = value_states[eff_kv_mask]

            # Pass the padded lengths as max_seqlen_q/k. Slightly over-sizes the
            # flash kernel's metadata vs the true batch maxima but avoids a
            # host-device sync from .item() on every adapter layer.
            outer_dtype = query_states.dtype
            q_packed, k_packed, v_packed = attention_dispatch._cast_qkv_for_flash(
                q_packed, k_packed, v_packed
            )
            out_packed = attention_dispatch.flash_attn_varlen_func(
                q_packed,
                k_packed,
                v_packed,
                cu_seqlens_q,
                cu_seqlens_kv,
                L_q,
                L_kv,
            )
            if (
                attention_dispatch.flash_attn_v100_provider
                and outer_dtype == torch.bfloat16
            ):
                out_packed = out_packed.to(dtype=outer_dtype)

            # Unpack: [total_valid_q, H, D] → [B, L_q, H, D]
            attn_output = query_states.new_zeros(B, L_q, self.n_heads, self.head_dim)
            attn_output[eff_q_mask] = out_packed
        else:
            # Fallback to PyTorch SDPA: needs [B, H, L, D] layout
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            # Expand kv_mask to 4D for SDPA broadcasting: [B, L] → [B, 1, 1, L]
            sdpa_mask = kv_mask[:, None, None, :] if kv_mask is not None else None
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=sdpa_mask
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class LLMAdapterTransformerBlock(nn.Module):
    """Transformer block for LLM Adapter: optional self-attn + cross-attn + MLP."""

    def __init__(
        self,
        source_dim,
        model_dim,
        num_heads=16,
        mlp_ratio=4.0,
        self_attn=False,
        layer_norm=False,
    ):
        super().__init__()
        self.has_self_attn = self_attn

        if self.has_self_attn:
            self.norm_self_attn = (
                nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
            )
            self.self_attn = LLMAdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim // num_heads,
            )

        self.norm_cross_attn = (
            nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        )
        self.cross_attn = LLMAdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim // num_heads,
        )

        self.norm_mlp = (
            nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(
        self,
        x,
        context,
        target_attention_mask=None,
        source_attention_mask=None,
        position_embeddings=None,
        position_embeddings_context=None,
    ):
        if self.has_self_attn:
            # Self-attention: target_attention_mask is not expected to be all zeros
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(
                normed,
                q_mask=target_attention_mask,
                kv_mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(
            normed,
            q_mask=target_attention_mask,
            kv_mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)


class LLMAdapter(nn.Module):
    """Bridge module: Qwen3 embeddings (source) → T5-compatible space (target).

    Uses T5 token IDs as target input, embeds them, and cross-attends to Qwen3 hidden states.
    """

    def __init__(
        self,
        source_dim,
        target_dim,
        model_dim,
        num_layers=6,
        num_heads=16,
        embed=None,
        self_attn=False,
        layer_norm=False,
    ):
        super().__init__()
        if embed is not None:
            self.embed = nn.Embedding.from_pretrained(embed.weight)
        else:
            self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                LLMAdapterTransformerBlock(
                    source_dim,
                    model_dim,
                    num_heads=num_heads,
                    self_attn=self_attn,
                    layer_norm=layer_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = LLMAdapterRMSNorm(target_dim)

    def forward(
        self,
        source_hidden_states,
        target_input_ids,
        target_attention_mask=None,
        source_attention_mask=None,
    ):
        # Keep masks as 2D [B, L] bool tensors — the attention layer handles
        # expansion to 4D for SDPA or packing for flash_attn_varlen_func.
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 4:
                target_attention_mask = target_attention_mask.squeeze(1).squeeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 4:
                source_attention_mask = source_attention_mask.squeeze(1).squeeze(1)

        x = self.in_proj(self.embed(target_input_ids))

        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(
            context.shape[1], device=x.device
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


# Not used currently, but kept for reference
