# Krea-2-Raw Single-Stream MMDiT
#
# 原始来源：krea-ai/krea-2 的 mmdit.py（路径 B 裸移植，非 diffusers）。
# 移植原则（搬家型重构）：保持原始计算语义不变，只做最小适配——
#   1. 去掉 @torch.compile(fullgraph=True) 装饰器；anima 的 compile_blocks()
#      统一在 network.apply_to 之后编译 block._forward，内置装饰器会双重编译。
#   2. attention() 保留 SDPA CUDNN backend（Krea-2 原始选择），加 fallback：
#      若当前环境无 CUDNN attention，退回默认 SDPA（自动选 flash/cudnn/math）。
#   3. config 用从权重反推的值做默认（见 weights.py 的 shape 核验）。
#
# 关键差异（vs anima DiT）：
#   - single-stream：text/image 拼接后共享 attention，无独立 cross_attn。
#   - GQA 48:12（query 48 头，KV 12 头，headdim=128）。
#   - light modulation：mod.lin 是单 Parameter(6*dim) 的 bias（不是 AdaLN-LoRA）。
#   - attention 门控：sigmoid(gate) 乘性作用于 attention 输出。
#   - 3D RoPE：headdim 按轴拆 [96, 16, 16]（headdim=128 时）。

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library.models.krea2_raw.attention_backend import (
    normalize_krea2_attention_mode,
    run_krea2_attention,
)
from library.runtime import offloading as custom_offloading_utils


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def _mask(mask: Tensor) -> Tensor:
    """Expand a (B, L) key-padding mask into a (B, 1, L, L) attention mask."""
    return mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20

    @classmethod
    def krea2_raw(cls) -> "SingleMMDiTConfig":
        """从 krea2_raw_bf16.safetensors 权重反推的默认 config。

        shape 核验见 docs/findings/krea2_raw_migration_stage0_findings.md §R4。
        """
        return cls(
            features=6144,
            tdim=256,
            txtdim=2560,
            heads=48,
            multiplier=4,
            layers=28,
            patch=2,
            channels=16,
            bias=False,
            theta=1e3,
            kvheads=12,
            txtlayers=12,
            txtheads=20,
            txtkvheads=20,
        )


class SimpleModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = torch.nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class RMSNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = torch.nn.Linear(features, mlpdim, bias=bias)
        self.up = torch.nn.Linear(features, mlpdim, bias=bias)
        self.down = torch.nn.Linear(mlpdim, features, bias=bias)
        # Mirrors SingleStreamBlock.gradient_checkpointing; the block's
        # enable/disable toggles this too.
        self.gradient_checkpointing = False

    def _mlp_chunk(self, hidden: Tensor) -> Tensor:
        # Pin the dtype at the chunk entry.  With chunk(2) a fp16 gate/up output
        # is 2176*16384*2 = 68 MiB, while fp32 is 136 MiB -- the exact size the
        # backward OOM asked for, i.e. the chunk math was running in fp32.
        hidden = hidden.to(self.gate.weight.dtype)
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))

    def forward(self, x: Tensor) -> Tensor:
        # The MLP is token-independent along the sequence dim, so under gradient
        # checkpointing we split the sequence and run *complete* MLPs per chunk,
        # concatenating only the projected 6144-dim result.
        #
        # Chunking here -- rather than inside a gate/up-only helper -- is what
        # removes every 136 MiB tensor: [1, 4352, 16384] fp16 becomes
        # [1, 2176, 16384], and because ``down`` (16384 -> 6144) also runs
        # inside the chunk, even its input is halved.  The outer cat only ever
        # holds the small 6144-dim output.  down(cat(a, b)) == cat(down(a),
        # down(b)) holds exactly since every op here is per-token.
        #
        # Also intentionally *not* an in-place ``up(x).mul_(...)`` fusion:
        # MulBackward0 needs the pre-update ``u`` and would raise "modified by
        # an inplace operation".
        checkpointing = (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        )
        # 4 chunks: 4352 / 4 = 1088 = 17 * 64 tokens, still tensor-core aligned,
        # and every 16384-dim intermediate drops from 68 MiB to 34 MiB.  The
        # splits are applied on *both* branches so the req=False precompute pass
        # never materialises the whole 4352-long sequence either.
        chunks = x.chunk(4, dim=-2)
        if not checkpointing:
            return torch.cat([self._mlp_chunk(chunk) for chunk in chunks], dim=-2)
        return torch.cat(
            [
                torch_checkpoint(self._mlp_chunk, chunk, use_reentrant=False)
                for chunk in chunks
            ],
            dim=-2,
        )


class Attention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = torch.nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = torch.nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = torch.nn.Linear(dim, dim, bias=bias)
        self.attn_mode = "torch"

    def forward(
        self, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        out = self.wo(
            run_krea2_attention(
                q, k, v, mask=mask, gqa=self.gqa, mode=self.attn_mode
            )
            * F.sigmoid(gate)
        )

        return out


class LastLayer(torch.nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = torch.nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        if getattr(self, "fp32_residual", False):
            from .native_precision import final_forward
            return final_forward(self, x, tvec)
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(torch.nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if getattr(self, "fp32_residual", False):
            from .native_precision import text_forward
            return text_forward(self, x, mask)
        x = x + self.attn(self.prenorm(x), mask=mask)
        x = x + self.mlp(self.postnorm(x))

        return x


class TextFusionTransformer(torch.nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth — that's fixed at 2 + 2 blocks.
    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = torch.nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = torch.nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        if getattr(self.layerwise_blocks[0], "fp32_residual", False):
            x = x.to(self.projector.weight.dtype)
        x = self.projector(x)
        x = x.squeeze(-1)

        for block in self.refiner_blocks:
            x = block(x, mask=mask)

        return x


class SingleStreamBlock(nn.Module):
    # Residual adapters must run inside the checkpoint/compile computation.
    reft_forward_method = "_forward"

    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.features = features
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)
        # Gradient checkpointing (移植自 anima models.py:1329/1342). SingleStream
        # block 是 1024×1024 训练显存主因 (28 blocks × 激活), grad-ckpt 用重算换
        # 激活显存——block swap 只搬权重救不了激活 (阶段6 findings). Krea-2 首
        # 日只实现标准 grad-ckpt (无 cpu_offload / unsloth / adapter-aware 变体,
        # 那些是 anima 专属优化路径).
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True
        mlp = getattr(self, "mlp", None)
        if mlp is not None and hasattr(mlp, "gradient_checkpointing"):
            mlp.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        mlp = getattr(self, "mlp", None)
        if mlp is not None and hasattr(mlp, "gradient_checkpointing"):
            mlp.gradient_checkpointing = False

    def _forward(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        if getattr(self, "fp32_residual", False):
            from .native_precision import block_forward
            return block_forward(self, x, vec, freqs, mask)
        compute_dtype = x.dtype
        modulation = tuple(value.to(compute_dtype) for value in self.mod(vec))
        prescale, preshift, pregate, postscale, postshift, postgate = modulation

        def _attn_step(hidden: Tensor) -> Tensor:
            attn_input = ((1 + prescale) * self.prenorm(hidden) + preshift).to(
                compute_dtype
            )
            return self.attn(attn_input, freqs, mask).to(compute_dtype)

        # Nested checkpoint on the *first* half (attention), use_reentrant=True.
        # Backward runs mlp first and attention second, so checkpointing the tail
        # would recompute mlp while attention's ~400 MB of internals are still
        # resident -- zero saving.  Wrapping attention instead lets the mlp pass
        # finish and release its ~800 MB before attention gets recomputed, so the
        # two never coexist.  use_reentrant=True keeps the first pass under
        # torch.no_grad(), so it does not stack with the outer checkpoint's
        # saved_tensors_hooks, and it only holds this one block's input.
        if (
            torch.is_grad_enabled()
            and self.training
            and self.gradient_checkpointing
            and x.requires_grad
        ):
            attn_output = torch_checkpoint(
                _attn_step, x, use_reentrant=False
            )
        else:
            attn_output = _attn_step(x)
        x = x + pregate * attn_output
        mlp_input = ((1 + postscale) * self.postnorm(x) + postshift).to(
            compute_dtype
        )
        mlp_output = self.mlp(mlp_input).to(compute_dtype)
        x = x + postgate * mlp_output
        return x

    def forward(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        if (
            torch.is_grad_enabled()
            and self.training
            and self.gradient_checkpointing
        ):
            return torch_checkpoint(
                self._forward,
                x,
                vec,
                freqs,
                mask,
                use_reentrant=False,
            )
        return self._forward(x, vec, freqs, mask)


class SingleStreamDiT(nn.Module):
    """Krea-2-Raw 单流 MMDiT。

    forward 签名与 anima DiT 对齐到「5D latent 输入」契约：
    img latent shape (B, C, T=1, H, W)，单例时间轴是 dim 2（anima 不变量）。
    内部 patchify 后拉平成 (B, L, D) 序列送入 transformer。
    """

    LATENT_CHANNELS = 16
    VAE_SPATIAL_COMPRESSION = 8
    PATCH_SIZE = 2

    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(
            config.features, axes, theta=config.theta, ntk=1.0
        )
        self.first = nn.Linear(
            config.channels * config.patch**2, config.features, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    config.features,
                    config.heads,
                    config.multiplier,
                    config.bias,
                    config.kvheads,
                )
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers,
            config.txtdim,
            config.txtheads,
            config.multiplier,
            config.bias,
            config.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6)
        )

        # Block swap 状态 (同 anima DiT, AGENTS.md lazy loading 不变量);
        # offloader 复用 anima ModelOffloader, 对 block forward 零假设.
        self.blocks_to_swap: int | None = None
        self.offloader = None
        self._paused_blocks_to_swap: int | None = None

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    # Symmetric with anima DiT (models.py:2200-2206): training loop and
    # harness read .device/.dtype off the unet for logging + dispatch.
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        parameters = iter(self.parameters())
        first = next(parameters)
        if first.is_floating_point():
            return first.dtype
        for parameter in parameters:
            if parameter.is_floating_point():
                return parameter.dtype
        return first.dtype

    def set_attention_mode(self, mode: str) -> None:
        normalized = normalize_krea2_attention_mode(mode)
        for module in self.modules():
            if isinstance(module, Attention):
                module.attn_mode = normalized
        self.attn_mode = normalized

    # === Gradient checkpointing 接口 (移植自 anima models.py:1942-1946) ===
    # 遍历 SingleStreamBlock 调其 enable_gradient_checkpointing. 训练侧
    # model_loading.py 读 args.gradient_checkpointing 后调此方法 (鸭子类型,
    # 与 anima DiT 同名). Krea-2 首 日只支持标准 grad-ckpt (无 cpu/unsloth
    # offload 变体). block swap 与 grad-ckpt 可共存 (swap 搬权重, grad-ckpt
    # 省激活, 两个维度独立).
    def enable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            if hasattr(block, "enable_gradient_checkpointing"):
                block.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        for block in self.blocks:
            if hasattr(block, "disable_gradient_checkpointing"):
                block.disable_gradient_checkpointing()

    def enable_selective_checkpointing(
        self,
        mode: str = "off",
        blocks: str | None = None,
    ) -> None:
        """Enable Krea-2 selective activation checkpointing.

        Krea-2 currently has only whole-block checkpointing. ``every_other``
        is therefore the sole selective mode with matching semantics; the
        Anima adapter/MLP/peak modes require checkpoint surfaces that these
        blocks do not expose.
        """
        del blocks
        mode = str(mode or "off").strip().lower()
        if mode not in {"off", "every_other"}:
            raise ValueError(
                "Krea-2 selective_checkpoint supports only 'off' and "
                f"'every_other'; got {mode!r}"
            )
        self.selective_checkpoint = mode
        self.disable_gradient_checkpointing()
        if mode == "every_other":
            for block_idx, block in enumerate(self.blocks):
                if block_idx % 2 == 0:
                    block.enable_gradient_checkpointing()

    def compile_blocks(
        self,
        backend: str = "inductor",
        mode: str | None = None,
        bucket_resolutions=None,
        n_token_families: int | None = None,
        dynamic_seq: bool = False,
        seq_range: tuple[int, int] | None = None,
        compile_block_scope: str = "resident",
        seq_bands: list[tuple[int, int]] | None = None,
    ) -> None:
        """Compile block computation after adapters and checkpoint setup.

        Krea-2 pads the combined text/image sequence to a multiple of 256, so
        it does not use Anima's native-flatten bucket machinery. The matching
        arguments remain in the signature for the shared training harness;
        ``seq_bands`` is accepted only to provide an explicit incompatibility
        error if a caller bypasses the normal family compatibility gate.
        """
        del bucket_resolutions, n_token_families, seq_range
        if seq_bands:
            raise ValueError(
                "Krea-2 compile_blocks uses fixed padded sequence lengths; "
                "per-band dynamic sequence is not supported"
            )
        if dynamic_seq:
            raise ValueError(
                "Krea-2 compile_blocks currently supports fixed padded "
                "sequence lengths only"
            )
        if mode not in {None, "default"}:
            raise ValueError(
                "Krea-2 compile_blocks supports only the default Inductor "
                f"mode; got {mode!r}"
            )
        if compile_block_scope not in {"resident", "all"}:
            raise ValueError(
                "compile_block_scope must be 'resident' or 'all'; "
                f"got {compile_block_scope!r}"
            )

        resident = len(self.blocks)
        if compile_block_scope == "resident" and self.blocks_to_swap:
            resident -= int(self.blocks_to_swap)
        compile_kwargs = {"backend": backend, "dynamic": False}
        if mode:
            compile_kwargs["mode"] = mode

        self._krea_compile_options = compile_kwargs

        for block_idx, block in enumerate(self.blocks):
            if block_idx >= resident:
                continue
            base_forward = getattr(block, "_krea_compile_base_forward", None)
            if base_forward is None:
                base_forward = block._forward
                block._krea_compile_base_forward = base_forward
            block._forward = torch.compile(base_forward, **compile_kwargs)
            block._krea_dynamic_compiled_forward = block._forward

    # === Block swap 接口 (移植自 anima models.py:2291-2387, 复用 ModelOffloader) ===
    # ModelOffloader 只遍历 block.named_modules() 取 .weight + .to(device) +
    # register_full_backward_hook, 对 block forward 签名零假设, SingleStreamBlock 满足.
    # 训练/推理管线 (harness.place_dit_for_training 等) 鸭子类型调这些方法.

    def enable_block_swap(
        self,
        num_blocks: int,
        device: torch.device,
        *,
        profile_jsonl: str | None = None,
        transfer_dtype: str | None = None,
        restore_mode: str | None = None,
        int8_restore_mode: str | None = None,
        int8_staging_ring: int | None = None,
        int8_dual_stream: bool | None = None,
        int8_restore_chunk_rows: int | None = None,
        int8_scope: str | None = None,
    ):
        self.blocks_to_swap = num_blocks
        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"Cannot swap more than {self.num_blocks - 2} blocks. "
            f"Requested: {self.blocks_to_swap} blocks."
        )
        self.offloader = custom_offloading_utils.ModelOffloader(
            self.blocks,
            self.blocks_to_swap,
            device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=transfer_dtype,
            restore_mode=restore_mode,
            int8_restore_mode=int8_restore_mode,
            int8_staging_ring=int8_staging_ring,
            int8_dual_stream=int8_dual_stream,
            int8_restore_chunk_rows=int8_restore_chunk_rows,
            int8_scope=int8_scope,
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None  # skip .to() on blocks (offloader 管理)
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = save_blocks

    def switch_block_swap_for_inference(self):
        if not self.blocks_to_swap:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if not self.blocks_to_swap:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()

    def prepare_block_swap_before_forward(self, free_cache: bool = True):
        if not self.blocks_to_swap:
            return
        self.offloader.prepare_block_devices_before_forward(
            self.blocks, free_cache=free_cache
        )

    def flush_block_swap_profile(self, blocking: bool = False) -> None:
        if not self.blocks_to_swap or self.offloader is None:
            return
        self.offloader.flush_profile_events(blocking=blocking)

    def _run_blocks(
        self,
        combined: Tensor,
        tvec: Tensor,
        freqs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Block 循环 + swap 钩子 (移植自 anima _run_blocks, 改 Krea-2 签名).

        SingleStreamBlock.forward(x, vec, freqs, mask) — 与 anima Block.forward
        签名不同, 但 swap 钩子 (wait_for_block / submit_move_blocks) 对 forward
        透明, 只夹在 block 调用前后.
        """
        for block_idx, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_idx)
            combined = block(combined, tvec, freqs, mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, block_idx)
        return combined

    def forward(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        # img: (B, L_img, patch*patch*channels) 已 patchify 的 latent 序列
        # context: (B, L_txt, num_txt_layers, txtdim) Qwen3-VL 多层 hidden states
        # t: (B,) timestep
        # pos: (B, L_total, 3) 3D 位置编码（H, W, token-type 轴）
        # mask: (B, L_total) key-padding mask（True=有效）
        compute_dtype = img.dtype
        img = self.first(img).to(compute_dtype)
        t = self.tmlp(
            temb(t, self.config.tdim, device=img.device, dtype=compute_dtype)
        ).to(compute_dtype)
        tvec = self.tproj(t).to(compute_dtype)

        txtmask = _mask(mask[:, : context.shape[1]])

        context = self.txtfusion(context.to(compute_dtype), mask=txtmask).to(
            compute_dtype
        )
        context = self.txtmlp(context).to(compute_dtype)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1).to(compute_dtype)

        # Pad combined sequence to a multiple of 256 to stabilize compiled kernel shapes.
        fulllen = combined.shape[1]
        _padlen = (-fulllen) % 256
        if _padlen > 0:
            combined = F.pad(combined, (0, 0, 0, _padlen))
            mask = F.pad(mask, (0, _padlen), value=False)
            pos = F.pad(pos, (0, 0, 0, _padlen))

        mask = _mask(mask)

        freqs = self.posemb(pos)

        combined = self._run_blocks(combined, tvec, freqs, mask)

        final = self.last(combined, t).to(compute_dtype)
        output = final[:, txtlen : txtlen + imglen, :]

        return output
