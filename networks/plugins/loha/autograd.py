"""Memory-conscious LoHa linear autograd.

Forward still builds the Hadamard-product weight (same math as PEFT/LyCORIS),
but only the four low-rank factors and the activation are saved for backward.
The full ``out × in`` weight is recomputed in backward instead of retained on
the autograd tape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Bound transient FP32 Hadamard weight blocks to ~16 MiB (4M floats) so large
# MLP projections (e.g. 16384x4096 = 256 MiB x 3 = 768 MiB forward / 1.5 GiB
# backward) never materialize full-matrix FP32 temporaries at once.
_LOHA_WEIGHT_CHUNK_ELEMS = 4 * 1024 * 1024


def make_hada_weight(
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
) -> torch.Tensor:
    """``(w1_a @ w1_b) ⊙ (w2_a @ w2_b)`` in fp32 matmul precision."""

    return (w1_a.float() @ w1_b.float()) * (w2_a.float() @ w2_b.float())


class HadaLinearFn(torch.autograd.Function):
    """``y = scale * F.linear(x, (w1a@w1b) ⊙ (w2a@w2b))``."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1_a: torch.Tensor,
        w1_b: torch.Tensor,
        w2_a: torch.Tensor,
        w2_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        out_features = int(w1_a.shape[0])
        in_features = int(w1_b.shape[1])
        scale_f = float(scale)
        x_f = x.float()
        ctx.save_for_backward(x, w1_a, w1_b, w2_a, w2_b)
        ctx.scale = scale_f

        if out_features * in_features <= _LOHA_WEIGHT_CHUNK_ELEMS:
            weight = make_hada_weight(w1_a, w1_b, w2_a, w2_b)
            out = F.linear(x_f, weight)
            return (out * scale_f).to(dtype=x.dtype)

        chunk_out = max(1, _LOHA_WEIGHT_CHUNK_ELEMS // max(1, in_features))
        w1_a_f = w1_a.float()
        w1_b_f = w1_b.float()
        w2_a_f = w2_a.float()
        w2_b_f = w2_b.float()
        result = torch.empty(
            (*x.shape[:-1], out_features), device=x.device, dtype=x.dtype
        )
        for out_start in range(0, out_features, chunk_out):
            out_end = min(out_start + chunk_out, out_features)
            out_slice = slice(out_start, out_end)
            w_chunk = (w1_a_f[out_slice] @ w1_b_f) * (w2_a_f[out_slice] @ w2_b_f)
            result[..., out_slice] = (F.linear(x_f, w_chunk) * scale_f).to(
                dtype=x.dtype
            )
            del w_chunk
        return result

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w1_a, w1_b, w2_a, w2_b = ctx.saved_tensors
        scale = ctx.scale
        out_features = int(w1_a.shape[0])
        in_features = int(w1_b.shape[1])

        if out_features * in_features <= _LOHA_WEIGHT_CHUNK_ELEMS:
            go = grad_out.float() * scale
            x_f = x.float()

            weight = make_hada_weight(w1_a, w1_b, w2_a, w2_b)
            grad_x = go.matmul(weight).to(dtype=x.dtype)

            # dL/dW = go^T @ x  → shape (out, in)
            grad_w = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
                x_f.reshape(-1, x_f.shape[-1])
            )

            w1 = w1_a.float() @ w1_b.float()
            w2 = w2_a.float() @ w2_b.float()
            # d/dW1 (W1 ⊙ W2) = W2; d/dW2 (W1 ⊙ W2) = W1
            grad_w1 = grad_w * w2
            grad_w2 = grad_w * w1

            grad_w1_a = grad_w1.matmul(w1_b.float().transpose(0, 1)).to(dtype=w1_a.dtype)
            grad_w1_b = w1_a.float().transpose(0, 1).matmul(grad_w1).to(dtype=w1_b.dtype)
            grad_w2_a = grad_w2.matmul(w2_b.float().transpose(0, 1)).to(dtype=w2_a.dtype)
            grad_w2_b = w2_a.float().transpose(0, 1).matmul(grad_w2).to(dtype=w2_b.dtype)

            return grad_x, grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b, None

        chunk_out = max(1, _LOHA_WEIGHT_CHUNK_ELEMS // max(1, in_features))
        x_2d = x.reshape(-1, in_features).float()
        go_view = grad_out.reshape(-1, out_features)
        w1_a_f = w1_a.float()
        w1_b_f = w1_b.float()
        w2_a_f = w2_a.float()
        w2_b_f = w2_b.float()
        w1_b_t = w1_b_f.transpose(0, 1)
        w2_b_t = w2_b_f.transpose(0, 1)

        grad_x_2d = torch.zeros_like(x_2d)
        grad_w1_a_f = torch.empty_like(w1_a_f)
        grad_w1_b_f = torch.zeros_like(w1_b_f)
        grad_w2_a_f = torch.empty_like(w2_a_f)
        grad_w2_b_f = torch.zeros_like(w2_b_f)

        for out_start in range(0, out_features, chunk_out):
            out_end = min(out_start + chunk_out, out_features)
            out_slice = slice(out_start, out_end)
            go_chunk = go_view[:, out_slice].float() * scale
            w1_chunk = w1_a_f[out_slice] @ w1_b_f
            w2_chunk = w2_a_f[out_slice] @ w2_b_f
            grad_x_2d.add_(go_chunk.matmul(w1_chunk * w2_chunk))

            grad_w_chunk = go_chunk.transpose(0, 1).matmul(x_2d)
            gw1_chunk = grad_w_chunk * w2_chunk
            grad_w1_a_f[out_slice] = gw1_chunk.matmul(w1_b_t)
            grad_w1_b_f.add_(w1_a_f[out_slice].transpose(0, 1).matmul(gw1_chunk))
            del gw1_chunk, w2_chunk

            gw2_chunk = grad_w_chunk * w1_chunk
            grad_w2_a_f[out_slice] = gw2_chunk.matmul(w2_b_t)
            grad_w2_b_f.add_(w2_a_f[out_slice].transpose(0, 1).matmul(gw2_chunk))
            del gw2_chunk, w1_chunk, grad_w_chunk, go_chunk

        return (
            grad_x_2d.reshape_as(x).to(dtype=x.dtype),
            grad_w1_a_f.to(dtype=w1_a.dtype),
            grad_w1_b_f.to(dtype=w1_b.dtype),
            grad_w2_a_f.to(dtype=w2_a.dtype),
            grad_w2_b_f.to(dtype=w2_b.dtype),
            None,
        )


def loha_linear(
    x: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Public helper used by :class:`LoHaModule` forward."""

    return HadaLinearFn.apply(x, w1_a, w1_b, w2_a, w2_b, float(scale))
