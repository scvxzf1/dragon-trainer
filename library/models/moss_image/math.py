# SPDX-License-Identifier: Apache-2.0
# Adapted from MOSS-Image-SFT/pipeline_moss_image.py, HF revision
# daf706d2e1c7a6369c17b6d1658671aee6db1ebe (Apache-2.0).
"""Official MOSS normalization, SiTU, RoPE, and transformer blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _validate_skip_layers(layers, depth, name='skip_layers'):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f'{name} must contain integer block indices in [0, {depth}).')
    return frozenset(layers)


def time_sinusoid(times, freq_dim: int, device) -> torch.Tensor:
    """Training time path: float64 sinusoid, float32 into the MLP.

    Single authority for the production forward and the PP stages, so a PP
    stage cannot re-derive the precision chain (dit.py: fp64 phase, fp32
    embedding into ``time_embedding``).
    """
    times = torch.as_tensor(times, device=device, dtype=torch.float32).reshape(-1)
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float()


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


def _attention_packed(q, k, v, rope, heads, spans):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q_rot = _rotate(q.reshape(shape), rope).transpose(1, 2).to(v.dtype)
    k_rot = _rotate(k.reshape(shape), rope).transpose(1, 2).to(v.dtype)
    v_heads = v.reshape(shape).transpose(1, 2)
    parts = [
        F.scaled_dot_product_attention(
            q_rot[:, :, start:end, :],
            k_rot[:, :, start:end, :],
            v_heads[:, :, start:end, :],
        )
        for start, end in spans
    ]
    return torch.cat(parts, dim=2).transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.gradient_checkpointing = False
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(self._forward, x, rope, use_reentrant=False)
        return self._forward(x, rope)

    def _forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.gradient_checkpointing = False
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')
        })
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(self._forward, x, rope, modulation, use_reentrant=False)
        return self._forward(x, rope, modulation)

    def forward_packed(self, x, rope, modulation, spans):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(
                self._forward_packed, x, rope, modulation, spans, use_reentrant=False
            )
        return self._forward_packed(x, rope, modulation, spans)

    def _forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj['q'].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj['o'](_attention(q, k, self.proj['v'](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f[:, None]

    def _forward_packed(self, x, rope, modulation, spans):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        normed1 = self.norm1(x)
        y = torch.cat(
            [
                normed1[:, s:e] * (1 + scale_a[i : i + 1, None]) + shift_a[i : i + 1, None]
                for i, (s, e) in enumerate(spans)
            ],
            dim=1,
        )
        dtype = self.proj['q'].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        v = self.proj['v'](y)
        attn_out = _attention_packed(q, k, v, rope, self.heads, spans)
        proj_o = self.proj['o'](attn_out)
        x = x + torch.cat(
            [proj_o[:, s:e] * gate_a[i : i + 1, None] for i, (s, e) in enumerate(spans)],
            dim=1,
        )
        normed2 = self.norm2(x)
        y = torch.cat(
            [
                normed2[:, s:e] * (1 + scale_f[i : i + 1, None]) + shift_f[i : i + 1, None]
                for i, (s, e) in enumerate(spans)
            ],
            dim=1,
        )
        hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
        ffn_out = self.proj['ffn_down'](_situ(hidden, gate))
        return x + torch.cat(
            [ffn_out[:, s:e] * gate_f[i : i + 1, None] for i, (s, e) in enumerate(spans)],
            dim=1,
        )


def run_moss_block_sequence(
    layers,
    samples: list[torch.Tensor],
    ropes: list[torch.Tensor],
    modulation: torch.Tensor,
    *,
    skip_layers=frozenset(),
    offloader=None,
    blocks_to_swap: int = 0,
) -> list[torch.Tensor]:
    """Run ``layers`` outer-most over ``samples`` (unlocks ``batch > 1`` swap & GPU GEMM fusion)."""
    batch = len(samples)
    if batch == 1:
        x = samples[0]
        rope = ropes[0]
        for layer_index, layer in enumerate(layers):
            if blocks_to_swap:
                offloader.wait_for_block(layer_index)
            if layer_index not in skip_layers:
                x = layer(x, rope, modulation)
            if blocks_to_swap:
                offloader.submit_move_blocks(layers, layer_index)
        return [x]

    if samples[0].device.type != "cuda":
        # Preserve bitwise-exact fp32 CPU per-sample numerics while still
        # stepping the block-swap ring once per layer across the batch.
        cur = list(samples)
        for layer_index, layer in enumerate(layers):
            if blocks_to_swap:
                offloader.wait_for_block(layer_index)
            if layer_index not in skip_layers:
                for index in range(batch):
                    cur[index] = layer(
                        cur[index], ropes[index], modulation[index : index + 1]
                    )
            if blocks_to_swap:
                offloader.submit_move_blocks(layers, layer_index)
        return cur

    first_len = int(samples[0].shape[1])
    if all(int(sample.shape[1]) == first_len for sample in samples):
        x_batched = torch.cat(samples, dim=0)
        rope = ropes[0]
        for layer_index, layer in enumerate(layers):
            if blocks_to_swap:
                offloader.wait_for_block(layer_index)
            if layer_index not in skip_layers:
                x_batched = layer(x_batched, rope, modulation)
            if blocks_to_swap:
                offloader.submit_move_blocks(layers, layer_index)
        return list(x_batched.split(1, dim=0))

    spans: list[tuple[int, int]] = []
    cursor = 0
    for sample in samples:
        next_cursor = cursor + int(sample.shape[1])
        spans.append((cursor, next_cursor))
        cursor = next_cursor
    span_tuple = tuple(spans)
    x_packed = torch.cat(samples, dim=1)
    rope_packed = torch.cat(ropes, dim=1)
    for layer_index, layer in enumerate(layers):
        if blocks_to_swap:
            offloader.wait_for_block(layer_index)
        if layer_index not in skip_layers:
            x_packed = layer.forward_packed(
                x_packed, rope_packed, modulation, span_tuple
            )
        if blocks_to_swap:
            offloader.submit_move_blocks(layers, layer_index)
    return [x_packed[:, start:end] for start, end in span_tuple]


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer('bands', self._default_bands(torch.device('cpu')))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)
