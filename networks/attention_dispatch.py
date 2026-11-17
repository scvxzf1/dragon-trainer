# Unified attention function supporting various implementations

import importlib
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Optional, Union

try:
    import flash_attn
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    flash_attn_func = None

def _is_v100_flash_provider(module, func) -> bool:
    provider_module = getattr(func, "__module__", "")
    doc = getattr(module, "__doc__", None) or ""
    return provider_module.startswith("flash_attn_v100.") or "Tesla V100" in doc


flash_attn_v100_provider = _is_v100_flash_provider(flash_attn, flash_attn_func)
flash_attn_v100_compat_active = False
if flash_attn_v100_provider:
    try:
        from networks.flash_attn_v100_compat import (
            flash_attn_func,
            flash_attn_varlen_func,
        )

        flash_attn_v100_compat_active = True
    except (ImportError, AttributeError) as exc:
        warnings.warn(
            "flash-attention-v100 is installed, but anima_lora could not enable "
            f"its torch.compile compatibility layer: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

# The V100 fork exposes the public FA2 facade but not all Tri Dao private
# wrappers. Keep those optional so their absence does not disable Flash.
_flash_interface = None
if flash_attn is not None:
    try:
        _flash_interface = importlib.import_module("flash_attn.flash_attn_interface")
    except ImportError:
        pass

_flash_attn_forward = getattr(_flash_interface, "_flash_attn_forward", None)
_wrapped_flash_attn_forward = getattr(
    _flash_interface, "_wrapped_flash_attn_forward", None
)
_wrapped_flash_attn_backward = getattr(
    _flash_interface, "_wrapped_flash_attn_backward", None
)

# Flash Attention 4 (flash-attention-sm120) is disabled; see docs/optimizations/fa4.md.
_flash_attn_4_func_raw = None
flash_attn_4_func = None
flash_attn_4_varlen_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask,
    )

    # Do NOT pre-compile flex_attention here. When blocks are individually
    # compiled (compile_blocks / native-flatten mode), the outer torch.compile
    # already traces into _flex_attention and fuses it.
    # Pre-compiling causes nested compilation which exhausts dynamo's
    # recompile limit (grad_mode guard × mask variants) and falls back to
    # the slow unfused path.
    compiled_flex_attention = _flex_attention

except ImportError:
    compiled_flex_attention = None
    create_block_mask = None


def sdpa_backend_context(attn_mode: Optional[str]):
    """Scope an explicitly requested PyTorch SDPA backend to one call."""
    if attn_mode == "mem_efficient":
        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    return nullcontext()


def flash_attn_device_supported(
    device: torch.device | int | str | None = None,
) -> bool:
    """Return whether the installed Flash build can run on the target/current GPU.

    The dtype/provider checks below only prove that a Python provider exists.
    A wheel built for Ampere+ still imports on Turing and then raises
    "FlashAttention only supports Ampere GPUs or newer" at kernel launch, so
    callers that only know a dtype need this guard to fall back to SDPA instead
    of crashing mid-forward. The bundled V100 provider targets sm_70, so it is
    the one pre-Ampere build this dispatch treats as usable.
    """
    if not torch.cuda.is_available():
        return False
    try:
        if device is not None:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type != "cuda":
                return False
            major, _minor = torch.cuda.get_device_capability(dev)
        else:
            major, _minor = torch.cuda.get_device_capability()
    except (RuntimeError, AssertionError, ValueError):
        return False
    if flash_attn_v100_provider:
        return major >= 7
    return major >= 8


def flash_attn_available_for_dtype(
    dtype: torch.dtype,
    device: torch.device | int | str | None = None,
) -> bool:
    """Return whether the imported Flash provider supports this dtype here."""
    if flash_attn_func is None or flash_attn_varlen_func is None:
        return False
    if device is None:
        supported = flash_attn_device_supported()
    else:
        try:
            supported = flash_attn_device_supported(device)
        except TypeError:
            supported = flash_attn_device_supported()
    if not supported:
        return False
    if dtype == torch.float16:
        return True
    return dtype == torch.bfloat16 and not flash_attn_v100_provider


_ORIG_FLASH_ATTN_AVAILABLE_FOR_DTYPE = flash_attn_available_for_dtype


def flash_attn_available_for_tensor(tensor: torch.Tensor) -> bool:
    """Check Flash availability against ``tensor.dtype`` and ``tensor.device``."""
    if flash_attn_available_for_dtype is not _ORIG_FLASH_ATTN_AVAILABLE_FOR_DTYPE:
        return flash_attn_available_for_dtype(tensor.dtype)
    dev = tensor.device if tensor.device.type == "cuda" else None
    return flash_attn_available_for_dtype(tensor.dtype, device=dev)


_FLASH_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _preferred_flash_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Pick fp16/bf16 for FlashAttention when inputs may be float32.

    Preference order:
    1. Any already-half Q/K/V dtype (keeps mixed stacks consistent)
    2. Active autocast GPU dtype when it is half
    3. bfloat16 (default mixed-precision training target)
    """
    if flash_attn_v100_provider:
        return torch.float16
    for tensor in tensors:
        if tensor is not None and tensor.dtype in _FLASH_HALF_DTYPES:
            return tensor.dtype
    if torch.is_autocast_enabled():
        # Prefer the public helper; fall back for older torch builds.
        get_dtype = getattr(torch, "get_autocast_dtype", None)
        if get_dtype is not None:
            try:
                autocast_dtype = get_dtype("cuda")
            except (TypeError, RuntimeError, ValueError):
                autocast_dtype = None
        else:
            get_gpu = getattr(torch, "get_autocast_gpu_dtype", None)
            autocast_dtype = get_gpu() if get_gpu is not None else None
        if autocast_dtype in _FLASH_HALF_DTYPES:
            return autocast_dtype
    return torch.bfloat16


def _cast_qkv_for_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unconditionally cast Q/K/V to fp16/bf16 before FlashAttention.

    FlashAttention kernels reject float32. Under ``torch.compile`` + ConvRot
    ``scope=all`` the inductor graph can materialize Q/K/V as float32 even when
    training mixed_precision is bf16, and the weak Attention.forward guard
    (cast only when q.dtype != v.dtype *and* autocast is on) does not fire.
    Casting here is the safety net for that four-lock combo.
    """
    if flash_attn_v100_provider and any(
        tensor.dtype == torch.bfloat16 for tensor in (q, k, v)
    ):
        raise RuntimeError(
            "flash-attention-v100 only supports FP16; BF16 Q/K/V are invalid"
        )
    target = _preferred_flash_dtype(q, k, v)
    if q.dtype != target:
        q = q.to(dtype=target)
    if k.dtype != target:
        k = k.to(dtype=target)
    if v.dtype != target:
        v = v.to(dtype=target)
    return q, k, v


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None
    softmax_scale: Optional[float] = (
        None  # custom softmax scale (default: 1/sqrt(head_dim))
    )
    crossattn_block_mask: Optional[object] = (
        None  # pre-computed BlockMask for cross-attention (flex mode only)
    )
    selfattn_block_mask: Optional[object] = (
        None  # pre-computed BlockMask for self-attention padding (flex mode, static-shape training)
    )
    uniform_seqlens: bool = (
        False  # caller guarantees all seqlens are equal (skips GPU sync check)
    )
    v100_flash_stability: str = "off"
    debug_finite_checks: bool = False

    @property
    def supports_fp32(self) -> bool:
        # flash4 is not supported yet, but keep it in the exclusion list for parity.
        return self.attn_mode not in ["flash", "flash4", "mem_efficient"]

    @property
    def requires_same_dtype(self) -> bool:
        return self.attn_mode in ["xformers"]

    def for_attention_kind(self, *, is_selfattn: bool) -> "AttentionParams":
        """Specialize shared parameters for one self/cross attention call."""
        if (
            self.v100_flash_stability == "hybrid"
            and self.attn_mode == "flash"
            and not is_selfattn
        ):
            return replace(self, attn_mode="torch")
        return self

    @staticmethod
    def create_attention_params(
        attn_mode: Optional[str],
        softmax_scale: Optional[float] = None,
        *,
        v100_flash_stability: str = "off",
        debug_finite_checks: bool = False,
    ) -> "AttentionParams":
        return AttentionParams(
            attn_mode,
            softmax_scale=softmax_scale,
            v100_flash_stability=v100_flash_stability,
            debug_finite_checks=debug_finite_checks,
        )

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str],
        img_len: Optional[int],
        attention_mask: Optional[torch.Tensor],
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
            max_seqlen = attention_mask.shape[1] + img_len

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens = torch.zeros(
                [2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device
            )
            offsets = (
                torch.arange(
                    batch_size, dtype=torch.int32, device=attention_mask.device
                )
                * max_seqlen
            )
            cu_seqlens[1::2] = offsets + seqlens  # end of valid tokens per batch
            cu_seqlens[2::2] = offsets + max_seqlen  # end of all tokens per batch

            # Expand attention mask to include image tokens
            attention_mask = torch.nn.functional.pad(
                attention_mask, (img_len, 0), value=1
            )  # [B, img_len + L]

            if attn_mode == "xformers":
                seqlens_list = seqlens.cpu().tolist()
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens_list, seqlens_list, device=attention_mask.device
                )
            elif attn_mode in ("torch", "mem_efficient", "flex"):
                attention_mask = attention_mask[:, None, None, :].to(
                    torch.bool
                )  # [B, 1, 1, img_len + L]

            return AttentionParams(
                attn_mode,
                img_len,
                attention_mask,
                seqlens,
                cu_seqlens,
                max_seqlen,
            )


def dispatch_attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_params: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, (
            "k and v must be provided if qkv_or_q is a tensor"
        )
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch")

    # Flex attention: early return using BlockMask for variable-length handling (compile-friendly)
    if attn_params.attn_mode == "flex":
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = attn_params.softmax_scale
        B, H, Q_LEN, D = q.shape
        KV_LEN = k.shape[2]

        block_mask = None
        if Q_LEN != KV_LEN and attn_params.crossattn_block_mask is not None:
            # Cross-attention with pre-computed BlockMask (skips padding in text tokens)
            block_mask = attn_params.crossattn_block_mask
        elif Q_LEN == KV_LEN and attn_params.selfattn_block_mask is not None:
            # Self-attention with pre-computed BlockMask (static-shape padding)
            block_mask = attn_params.selfattn_block_mask
        elif attn_params.seqlens is not None:
            # Variable-length: mask padding positions via BlockMask
            seqlens = attn_params.seqlens

            def mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < seqlens[b]

            block_mask = create_block_mask(
                mask_mod, B, H, Q_LEN, KV_LEN, device=q.device
            )
        elif attn_params.attention_mask is not None and isinstance(
            attn_params.attention_mask, torch.Tensor
        ):
            bool_mask = attn_params.attention_mask.squeeze(1).squeeze(1)  # [B, L]

            def mask_mod(b, h, q_idx, kv_idx):
                return bool_mask[b, kv_idx]

            block_mask = create_block_mask(
                mask_mod, B, H, Q_LEN, KV_LEN, device=q.device
            )

        x = compiled_flex_attention(q, k, v, block_mask=block_mask, scale=scale)
        del q, k, v
        x = x.transpose(1, 2)  # [B, L, H, D]
        x = x.flatten(2)  # [B, L, H*D]
        return x

    # If attention mask is provided and all sequence lengths are the same, we can trim the sequence
    seqlen_trimmed = False
    if attn_params.attention_mask is not None and attn_params.seqlens is not None:
        if attn_params.uniform_seqlens or torch.all(
            attn_params.seqlens == attn_params.seqlens[0]
        ):
            seqlen = attn_params.seqlens[0].item()
            q = q[:, :seqlen]
            k = k[:, :seqlen]
            v = v[:, :seqlen]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(
                attn_params.attn_mode,
                softmax_scale=attn_params.softmax_scale,
                v100_flash_stability=attn_params.v100_flash_stability,
                debug_finite_checks=attn_params.debug_finite_checks,
            )  # do not in-place modify
            attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
            seqlen_trimmed = True

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode in ("torch", "mem_efficient") or (
        attn_params.attn_mode == "sageattn" and attn_params.cu_seqlens is None
    ):

        def transpose_fn(x):
            return x.transpose(
                1, 2
            )  # [B, H, L, D] for SDPA and sageattn with fixed length
    else:

        def transpose_fn(x):
            return x  # [B, L, H, D] for other implementations

    q = transpose_fn(q)
    k = transpose_fn(k)
    v = transpose_fn(v)

    scale = attn_params.softmax_scale  # None = default 1/sqrt(head_dim)

    if attn_params.attn_mode == "torch":
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_params.attention_mask,
            dropout_p=drop_rate,
            scale=scale,
        )
        del q, k, v

    elif attn_params.attn_mode == "mem_efficient":
        # Scope backend forcing to this call. Process-global CUDA backend flags
        # would also affect the text encoder, VAE, and third-party components.
        # The public context manager is traceable by torch.compile (covered by
        # tests/test_mem_efficient_sdpa.py).
        with sdpa_backend_context(attn_params.attn_mode):
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_params.attention_mask,
                dropout_p=drop_rate,
                scale=scale,
            )
        del q, k, v

    elif attn_params.attn_mode == "xformers":
        x = xops.memory_efficient_attention(
            q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate, scale=scale
        )
        del q, k, v

    elif attn_params.attn_mode == "sageattn":
        if attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v, sm_scale=scale)  # B, L, H, D. No dropout support
            del q, k, v
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv. No dropout support
            x = sageattn_varlen(
                q,
                k,
                v,
                attn_params.cu_seqlens,
                attn_params.cu_seqlens,
                attn_params.max_seqlen,
                attn_params.max_seqlen,
                sm_scale=scale,
            )
            del q, k, v

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        # Hard half cast: FlashAttention rejects fp32. Required when compile
        # materializes Q/K/V as float32 (e.g. w8a16_convrot scope=all).
        q, k, v = _cast_qkv_for_flash(q, k, v)
        if attn_params.cu_seqlens is None:  # all tokens are valid
            x = flash_attn_func(q, k, v, drop_rate, softmax_scale=scale)  # B, L, H, D
            del q, k, v
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
            x = flash_attn_varlen_func(
                q,
                k,
                v,
                attn_params.cu_seqlens,
                attn_params.cu_seqlens,
                attn_params.max_seqlen,
                attn_params.max_seqlen,
                drop_rate,
                softmax_scale=scale,
            )
            del q, k, v

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash4":
        raise NotImplementedError(
            "attn_mode='flash4' is disabled in this build "
            "(see docs/optimizations/fa4.md). "
            "Use 'flash', 'torch', 'mem_efficient', 'flex', 'sageattn', or 'xformers'."
        )

    else:
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.flatten(2)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(
            x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0
        )  # pad back to max_seqlen

    return x


def attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mode: str,
    softmax_scale: Optional[float] = None,
    drop_rate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SDPA that returns ``(out, softmax_lse)``.

    Used by EasyControl's LSE-decomposed extended self-attention to combine
    target+cond attention legs without materializing the
    ``[B, H, S_q, S_t+S_c]`` attention matrix that the masked-SDPA path falls
    into when a per-key bias forces dispatch to the math kernel.

    Args:
        q, k, v: BLHD tensors. K_q and K_kv may differ.
        attn_mode: only ``"flash"`` (FA2) is supported. Other backends raise
            ``NotImplementedError``; callers should fall back to the masked
            path.
        softmax_scale: 1/sqrt(d) by default.
        drop_rate: attention dropout (typically 0).

    Returns:
        out: ``[B, S_q, H, D]``, same dtype as ``q``.
        lse: ``[B, H, S_q]`` log-sum-exp of scaled QK^T over keys, fp32.

    Caveat: FA2's standard ``FlashAttnFunc.backward`` does NOT propagate
    gradient through the returned ``lse`` (the upstream gradient on ``lse``
    is silently dropped — see flash_attn 2.x source). Callers that need
    correct gradients through ``lse`` must use a custom
    ``torch.autograd.Function`` that calls FA's lower-level ops directly —
    see ``networks/methods/easycontrol.py:_ExtendedSelfAttnLSEFunc``.
    """
    if attn_mode != "flash" or flash_attn_func is None:
        raise NotImplementedError(
            f"attention_with_lse currently requires attn_mode='flash' with "
            f"flash-attn installed (got attn_mode={attn_mode!r}, "
            f"flash_attn={'installed' if flash_attn is not None else 'missing'})."
        )
    q, k, v = _cast_qkv_for_flash(q, k, v)
    out, lse, _ = flash_attn_func(
        q, k, v, drop_rate, softmax_scale=softmax_scale, return_attn_probs=True
    )
    return out, lse
