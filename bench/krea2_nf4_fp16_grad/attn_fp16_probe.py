"""Krea-2 attention fp16 内层的单层数值验证（重点：深层 dO 下溢）。

findings §2.1 测出 blocks.2~27 的 attention 上游梯度 dO 在 6.5e-9 ~ 1.7e-7。
裸 ``.to(fp16)`` 会把这些值压进 subnormal 区甚至冲成 0，反向当场崩掉。
本脚本用 fp32 + 展开 MHA 作真值，对比两种实现：

- ``scaled``：``_fp16_attention``，反向用 frexp/ldexp 把 dO 平移进 fp16 正规区；
- ``bare``：直接 fp16 SDPA，不做任何缩放（预期在 dO<=1e-8 时崩）。

判据：``scaled`` 在各 dO 量级下 rel-L2 应稳定在 1e-3 量级；``bare`` 应随 dO
减小而迅速劣化。

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/attn_fp16_probe.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2FP16_GPU", "1"))

import torch
import torch.nn.functional as F
from einops import rearrange

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.krea2_raw.attention_backend import (
    _fp16_attention,
    is_krea2_attn_fp16_enabled,
    set_krea2_attn_fp16,
)

CUDA = torch.device("cuda")


def ref_attention(q, k, v, mask, scale):
    """fp32 + 展开 MHA 真值（V100 上 fp32 可走 EFFICIENT，384 MiB）。"""
    group = q.shape[1] // k.shape[1]
    k32 = k.float().repeat_interleave(group, dim=1)
    v32 = v.float().repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(
        q.float(), k32, v32, attn_mask=mask, scale=scale
    )
    return rearrange(out, "B H L D -> B L (H D)")


def scaled_fp16_attention(q, k, v, mask, scale):
    """生产实现（keyword-only 签名）的位置参数包装。"""
    return _fp16_attention(q, k, v, mask=mask, scale=scale, gqa=True)


def bare_fp16_attention(q, k, v, mask, scale):
    """不做任何反向缩放的裸 fp16 路径（对照组）。"""
    group = q.shape[1] // k.shape[1]
    k16 = k.half().repeat_interleave(group, dim=1)
    v16 = v.half().repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(
        q.half(), k16, v16, attn_mask=mask, scale=scale
    )
    return rearrange(out, "B H L D -> B L (H D)")


def run_arm(fn, q, k, v, mask, scale, do_rms):
    """跑一次前向+反向，返回 (out, dq, dk, dv) 的 fp32 副本。"""
    qq = q.detach().clone().requires_grad_(True)
    kk = k.detach().clone().requires_grad_(True)
    vv = v.detach().clone().requires_grad_(True)
    out = fn(qq, kk, vv, mask, scale)
    torch.manual_seed(0)
    grad = torch.randn_like(out.float())
    grad = grad / grad.pow(2).mean().sqrt() * do_rms
    out.backward(grad.to(out.dtype))
    return (
        out.detach().float(),
        qq.grad.detach().float(),
        kk.grad.detach().float(),
        vv.grad.detach().float(),
    )


def rel(a, b):
    denom = b.norm()
    if denom == 0:
        return float("nan")
    return ((a - b).norm() / denom).item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=48)
    ap.add_argument("--kv-heads", type=int, default=12)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument(
        "--out-dir", type=Path, default=ROOT / "output" / "tests" / "krea2_nf4_fp16_grad"
    )
    args = ap.parse_args()

    if not is_krea2_attn_fp16_enabled():
        set_krea2_attn_fp16(True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    B, L, H, KV, D = 1, args.tokens, args.heads, args.kv_heads, args.dim
    torch.manual_seed(0)
    # QK-Norm 之后 q/k/v 量级在 O(1)，模拟真实 attention 输入
    q = torch.randn(B, H, L, D, device=CUDA, dtype=torch.bfloat16)
    k = torch.randn(B, KV, L, D, device=CUDA, dtype=torch.bfloat16)
    v = torch.randn(B, KV, L, D, device=CUDA, dtype=torch.bfloat16)
    mask = torch.ones(B, 1, L, L, device=CUDA, dtype=torch.bool)
    scale = D**-0.5

    print(
        f"=== GPU: {torch.cuda.get_device_name(0)} | L={L} H={H} KV={KV} D={D} ==="
    )

    rows = []
    ref_peak = 0.0
    for do_rms in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 每个 dO 量级都要重算 fp32 真值：不同 dO 的梯度量级本来就不同，
        # 拿 dO=1e-2 的真值去比 dO=1e-8 的实现会得到无意义的 100% 误差。
        ref = run_arm(ref_attention, q, k, v, mask, scale, do_rms)
        ref_peak = max(ref_peak, torch.cuda.max_memory_allocated() / 2**30)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        scaled = run_arm(scaled_fp16_attention, q, k, v, mask, scale, do_rms)
        scaled_peak = torch.cuda.max_memory_allocated() / 2**30
        bare = run_arm(bare_fp16_attention, q, k, v, mask, scale, do_rms)

        row = {
            "dO_rms": do_rms,
            "scaled_peak_GiB": scaled_peak,
            "scaled_out_rel": rel(scaled[0], ref[0]),
            "scaled_dq_rel": rel(scaled[1], ref[1]),
            "scaled_dk_rel": rel(scaled[2], ref[2]),
            "scaled_dv_rel": rel(scaled[3], ref[3]),
            "bare_dq_rel": rel(bare[1], ref[1]),
            "bare_dk_rel": rel(bare[2], ref[2]),
            "bare_dv_rel": rel(bare[3], ref[3]),
        }
        rows.append(row)
        print(
            f"  dO={do_rms:.0e}  scaled dq/dk/dv="
            f"{row['scaled_dq_rel']:.2e}/{row['scaled_dk_rel']:.2e}/{row['scaled_dv_rel']:.2e}"
            f"   bare dq/dk/dv="
            f"{row['bare_dq_rel']:.2e}/{row['bare_dk_rel']:.2e}/{row['bare_dv_rel']:.2e}"
            f"   peak={scaled_peak:.2f} GiB"
        )

    payload = {
        "probe": "bench/krea2_nf4_fp16_grad/attn_fp16_probe.py",
        "tokens": L,
        "heads": H,
        "kv_heads": KV,
        "dim": D,
        "gpu": torch.cuda.get_device_name(0),
        "fp32_ref_peak_GiB": ref_peak,
        "rows": rows,
    }
    out_path = args.out_dir / f"attn_fp16_probe_{L}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float))
    print(f"\n[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
