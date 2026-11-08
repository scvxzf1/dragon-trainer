"""Krea-2 NF4 fp16 backward 偏差溯源：真实权重/真实梯度量级捕获探针。

背景（stage6，2026-08-09）：同 NF4 权重、同输入下只把 compute_dtype 从 bf16
换成 fp16，forward 输出 rel-L2 = 0.2885%，backward 输入梯度 rel-L2 = 35.65%、
cosine 0.9434。当年只留 6 个标量，**没有记录梯度量级 / subnormal 占比 /
参考口径**，探针也未入库。本脚本补这个缺口。

机制假设（本脚本用真实数据验证）：bnb 的 MatMul4Bit.backward 是
    grad_A = torch.matmul(grad_output, dequantize_4bit(B, qs).to(grad_output.dtype).t())
反量化恒为 bf16（qs.dtype=bfloat16），再 cast 到 grad_output 的 dtype。fp16 臂里
这条链路全程 fp16，而 flow-matching 的 MSE 是 **全元素均值**：元素数越大，
dL/dv 越小。1024² 时元素数 2.6e5，dY 量级 ~1e-5..1e-6，正好落在 fp16 的
subnormal 区间（正规数下限 6.1e-5，最小 subnormal 6e-8），逐元素有效位被截到
0-3 bit，dX 相对误差爆炸；bf16 指数范围与 fp32 同宽，不受影响。

做四件事（真实 NF4 权重 + 真实 TE/VAE 缓存）：

1. 双臂（bf16 / fp16 compute）同权重同输入单步 forward+backward，逐层抓
   grad_output(dY)/grad_input(dX) 的 L2、max|·|、<6.1e-5、<6e-8、==0 占比（R2）。
2. 逐层 intra-op 拆解：同一份真实 dY + 真实反量化权重，比较 fp32 参考 /
   bf16 操作数 / fp16 操作数三条算子，回答「35% 里有多少是 bf16 自身偏差」（gap 3）。
3. loss-scale 扫描：真实 dY × 2^k 走 fp16 仿真验单调性（R1），并在模型级跑
   fp16 + loss scaling 端到端对照。
4. 归约精度（R3）：真实 dY × 真实权重，切
   torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction 两档比。

非目标 / 边界：
- 本机无 bf16 原生卡。V100(sm70) 的 bf16 matmul 走 PyTorch 的 fp32 上采样仿真
  （操作数 bf16 舍入 + fp32 累加），语义上等价 sm86 bf16 tensor core，但**不是**
  sm86 cuBLAS 逐 bit 复现；fp16 matmul 在 V100 上是真 tensor core（fp32 累加）。
- 图像是合成图案（沿用 probe_train.make_test_image），不是真实数据集样本。
  影响激活分布细节，不影响「梯度量级 → fp16 subnormal」这条机制链。
- 只读权重、不写用户数据目录；输出写 output/tests/krea2_nf4_fp16_grad/。

典型用法（跑完即释放显存）::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/capture_real.py \
      --img 256 --tag smoke

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/capture_real.py \
      --img 1024 --loss-scale-k 4,8,12,16 --replay-layers 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2FP16_GPU", "1"))

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "krea2"))

from library.models.krea2_raw.family import (
    Krea2TextEmbedding,
    forward_for_loss,
)
from library.models.krea2_raw.strategy import (
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
    load_krea2_text_encoder,
)
from library.models.krea2_raw.weights import load_krea2_dit
from library.models.qwen_vae import load_vae
from probe_train import FIXED_SIGMA, PROMPT, make_test_image

KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"
# 自包含 NF4 v2：一次加载即得与生产一致的预量化权重（不再走 26GB bf16 在线量化）。
KREA2_NF4 = ROOT / "models" / "diffusion_models" / "krea2_raw_nf4_self_contained.safetensors"

FP16_MIN_NORMAL = 6.103515625e-05  # 2^-14
FP16_MIN_SUBNORMAL = 5.960464477539063e-08  # 2^-24
FP16_MAX = 65504.0
CUDA = torch.device("cuda")


# --------------------------------------------------------------------------- #
# 基础度量
# --------------------------------------------------------------------------- #
def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a-b|| / ||b||，fp64 累加避免度量自身失真。"""
    a, b = _flat(a).double(), _flat(b).double()
    return float((a - b).norm() / (b.norm() + 1e-300))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = _flat(a).double(), _flat(b).double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-300))


def _norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(_flat(a).double().norm() / (_flat(b).double().norm() + 1e-300))


def _stats(t: torch.Tensor) -> dict:
    """张量量级画像，专门盯 fp16 的两个门槛。"""
    x = _flat(t).float().abs()
    n = x.numel()
    below_normal = int((x < FP16_MIN_NORMAL).sum())
    below_subnormal = int((x < FP16_MIN_SUBNORMAL).sum())
    zeros = int((x == 0).sum())
    pos = x[x > 0]
    return {
        "numel": n,
        "l2": float(x.double().norm()),
        "max_abs": float(x.max()) if n else 0.0,
        "min_abs_nonzero": float(pos.min()) if pos.numel() else 0.0,
        "frac_lt_fp16_min_normal": below_normal / n if n else 0.0,
        "frac_lt_fp16_min_subnormal": below_subnormal / n if n else 0.0,
        "frac_zero": zeros / n if n else 0.0,
        "frac_fp16_subnormal": (below_normal - below_subnormal) / n if n else 0.0,
        # 能量口径：有多少"平方和"落在 fp16 正规区以下（比元素占比更能说明影响）
        "energy_share_lt_fp16_min_normal": float(
            x.double().pow(2)[x < FP16_MIN_NORMAL].sum()
        )
        / (float(x.double().pow(2).sum()) + 1e-300),
    }


def _round_to(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """模拟「操作数 cast 到 dtype」：降到目标精度再升回 fp32 参与 fp32 matmul。"""
    return t.to(dtype).to(torch.float32)


def _fp16_operand_matmul(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp16 操作数 + fp32 累加（等价 fp16 tensor core 的 fp32-accumulate 语义）。"""
    # dX = dY @ W（W 形状 [out, in]），归约维是 out；不用 .t() 表达同一算子。
    return torch.matmul(_round_to(dy, torch.float16), _round_to(w, torch.float16))


def _bf16_operand_matmul(dy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(_round_to(dy, torch.bfloat16), _round_to(w, torch.bfloat16))


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 参考 matmul；显式关 TF32，否则参考本身只剩 10 bit 尾数。"""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.matmul(a, b)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


# --------------------------------------------------------------------------- #
# 输入准备（lazy loading：TE -> encode -> free -> VAE -> free）
# --------------------------------------------------------------------------- #
def prepare_inputs(img: int, cache_dir: Path) -> dict:
    cache = cache_dir / f"inputs_{img}.pt"
    if cache.exists():
        print(f"[cache] 命中 {cache}")
        return torch.load(cache, map_location="cpu", weights_only=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[A] TE encode 文本 hiddens ...")
    te, _ = load_krea2_text_encoder(str(KREA2_TE), dtype=torch.bfloat16, device="cuda")
    tok, enc = Krea2TokenizeStrategy(), Krea2TextEncodingStrategy()
    with torch.inference_mode():
        hiddens, txtmask = enc.encode_tokens(tok, [te], tok.tokenize([PROMPT]))
    hiddens, txtmask = hiddens.to("cpu"), txtmask.to("cpu")
    del te, enc, tok
    torch.cuda.empty_cache()

    print("[B] VAE encode 图像 latent ...")
    vae = load_vae(str(KREA2_VAE), device=torch.device("cuda"), dtype=torch.bfloat16, eval=True)
    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(make_test_image(img, torch.device("cuda"), torch.bfloat16))
    latents = latents.to("cpu").to(torch.bfloat16)
    del vae
    torch.cuda.empty_cache()

    blob = {"hiddens": hiddens, "txtmask": txtmask, "latents": latents}
    torch.save(blob, cache)
    print(f"[cache] 写入 {cache}, tokens/img={latents[0].numel()}")
    return blob


def linear4bit_paths(model: torch.nn.Module) -> list[str]:
    from bitsandbytes.nn import Linear4bit

    return [n for n, m in model.named_modules() if isinstance(m, Linear4bit)]


def set_compute_dtype(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """把每个 Linear4bit 的 compute_dtype 翻到目标 dtype（权重 bit 完全不变）。"""
    from bitsandbytes.nn import Linear4bit

    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, Linear4bit):
            mod.compute_dtype = dtype
            mod.compute_type_is_set = True
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 单臂单步
# --------------------------------------------------------------------------- #
def run_arm(
    dit: torch.nn.Module,
    blob: dict,
    *,
    dtype: torch.dtype,
    grad_scale: float = 1.0,
    capture: set[str] | None = None,
    device: torch.device = CUDA,
) -> dict:
    """给定 compute dtype 跑一次 forward+backward，抓逐层 dY/dX 画像。"""
    from bitsandbytes.nn import Linear4bit

    capture = capture or set()
    n_layers = set_compute_dtype(dit, dtype)
    latents = blob["latents"].to(device=device, dtype=dtype)
    hiddens = blob["hiddens"].to(device=device, dtype=dtype)
    txtmask = blob["txtmask"].to(device=device)

    torch.manual_seed(1234)
    noise = torch.randn_like(latents)
    sigma = torch.full((latents.shape[0],), FIXED_SIGMA, device=device, dtype=dtype)
    view = (-1, 1, 1, 1)
    x_t = (1.0 - sigma.view(*view)) * latents + sigma.view(*view) * noise
    target = (noise - latents).to(dtype)
    x_t = x_t.detach().requires_grad_(True)

    layer_stats: dict[str, dict] = {}
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_pre(name: str):
        def _pre(_mod, args):
            x = args[0] if args else None
            if isinstance(x, torch.Tensor) and x.requires_grad:
                x.register_hook(
                    lambda g, n=name: layer_stats.setdefault(n, {}).__setitem__(
                        "dX", _stats(g)
                    )
                )

        return _pre

    def make_post(name: str):
        def _post(_mod, _args, out):
            if isinstance(out, torch.Tensor):
                def _grab(g, n=name):
                    layer_stats.setdefault(n, {})["dY"] = _stats(g)
                    if n in capture:
                        captured[n] = g.detach().cpu()

                out.register_hook(_grab)

        return _post

    for name, mod in dit.named_modules():
        if isinstance(mod, Linear4bit):
            handles.append(mod.register_forward_pre_hook(make_pre(name)))
            handles.append(mod.register_forward_hook(make_post(name)))

    text_emb = Krea2TextEmbedding(hiddens, txtmask)
    t0 = time.time()
    velocity = forward_for_loss(dit, x_t.unsqueeze(2), text_emb, sigma)
    loss = torch.nn.functional.mse_loss(velocity, target.unsqueeze(2))
    (loss * grad_scale).backward()
    torch.cuda.synchronize()
    secs = time.time() - t0
    for h in handles:
        h.remove()

    return {
        "compute_dtype": str(dtype).replace("torch.", ""),
        "linear4bit_layers": n_layers,
        "loss": float(loss.detach()),
        "grad_scale": grad_scale,
        "forward_backward_s": secs,
        "velocity": velocity.detach().float().cpu(),
        # 反 scale，便于跨臂直接比（loss scaling 只放大后向）
        "latent_grad": (x_t.grad.detach().float().cpu() / grad_scale),
        "layer_stats": layer_stats,
        "captured": captured,
    }


# --------------------------------------------------------------------------- #
# 逐层 intra-op 拆解
# --------------------------------------------------------------------------- #
def dequant_fp32(mod: torch.nn.Module) -> torch.Tensor:
    import bitsandbytes.functional as bnbF

    return bnbF.dequantize_4bit(mod.weight.data, mod.weight.quant_state).to(torch.float32)


def error_buckets(dx: torch.Tensor, ref: torch.Tensor) -> dict:
    """把 ||err||² 按 **参考 dX 的量级** 分桶。

    err 与 ref 同形（dY 与 dX 形状不同，不能拿 dY 的逐元素量级去索引 err）；
    这里回答的是候选②：误差是被小分母元素放大出来的，还是摊在大元素上的。
    """
    a = _flat(ref).float().abs()
    e = (_flat(dx).float() - _flat(ref).float()).pow(2)
    total_e = float(e.double().sum())
    total_a = float(a.double().pow(2).sum())
    if total_e <= 0 or total_a <= 0:
        return {"total_sq_err": total_e}
    masks = {
        "ref_below_2^-24": a < FP16_MIN_SUBNORMAL,
        "ref_subnormal_2^-24_2^-14": (a >= FP16_MIN_SUBNORMAL) & (a < FP16_MIN_NORMAL),
        "ref_normal_ge_2^-14": a >= FP16_MIN_NORMAL,
    }
    out: dict = {"total_sq_err": total_e}
    for k, m in masks.items():
        out[k] = {
            "element_share": float(m.float().mean()),
            "ref_energy_share": float(a[m].double().pow(2).sum()) / total_a,
            "sq_err_share": float(e[m].double().sum()) / total_e,
        }
    return out


def signal_restricted(dx: torch.Tensor, ref: torch.Tensor) -> dict:
    """在"大信号元素"子集上看方向保真度，判断 35% 是否主要由小分母贡献。"""

    def _rel(mask: torch.Tensor) -> dict:
        if int(mask.sum()) < 16:
            return {"elements": int(mask.sum()), "insufficient": True}
        d, r = _flat(dx)[mask], _flat(ref)[mask]
        return {
            "elements": int(mask.sum()),
            "element_share": float(mask.float().mean()),
            "rel_l2": _rel_l2(d, r),
            "cos": _cos(d, r),
            "norm_ratio": _norm_ratio(d, r),
        }

    a = _flat(ref).float().abs()
    n = a.numel()
    k_top = max(16, n // 100)
    top_idx = torch.topk(a, k_top).indices
    top_mask = torch.zeros(n, dtype=torch.bool, device=a.device)
    top_mask[top_idx] = True
    return {
        "all": _rel(torch.ones(n, dtype=torch.bool, device=a.device)),
        "ref_ge_fp16_min_normal": _rel(a >= FP16_MIN_NORMAL),
        "ref_top_1pct": _rel(top_mask),
    }


def compare_ops(dy: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """同一份 dY + 真实反量化权重，比较 fp32 参考 / bf16 / fp16 三条算子。"""
    ref = _matmul(dy, w)
    out: dict = {"dY": _stats(dy), "dX_fp32_ref": _stats(ref)}
    arms = [("bf16_emul", _bf16_operand_matmul), ("fp16_emul", _fp16_operand_matmul)]
    if dy.device.type == "cuda":
        # 真 tensor core fp16 GEMM：操作数 fp16 + fp32 累加 + **fp16 输出舍入**。
        # 比纯仿真多一层输出舍入，是 R3 与逐层真值的口径。
        arms.append(("fp16_real_gemm", lambda a, b: torch.matmul(a.half(), b.half()).float()))
    for tag, fn in arms:
        dx = fn(dy, w)
        out[tag] = {
            "rel_l2_vs_fp32_ref": _rel_l2(dx, ref),
            "cos_vs_fp32_ref": _cos(dx, ref),
            "norm_ratio_vs_fp32_ref": _norm_ratio(dx, ref),
            "dX": _stats(dx),
            "error_buckets": error_buckets(dx, ref),
            "signal_restricted": signal_restricted(dx, ref),
        }
    return ref, out


def loss_scale_scan(dy: torch.Tensor, w: torch.Tensor, ks: list[int]) -> list[dict]:
    """真实 dY × 2^k 走 fp16 仿真：检验 subnormal 解释的单调性（R1）。"""
    dy32 = _flat(dy).float().view(dy.shape)
    w16 = _round_to(w, torch.float16)
    base_max = float(dy32.abs().max())
    rows = []
    for k in ks:
        s = float(2**k)
        ref = _matmul(dy32 * s, w)
        dx = torch.matmul(_round_to(dy32 * s, torch.float16), w16)
        rows.append(
            {
                "k": k,
                "scale": s,
                "rel_l2": _rel_l2(dx, ref),
                "cos": _cos(dx, ref),
                "norm_ratio": _norm_ratio(dx, ref),
                "dY_max_abs_scaled": base_max * s,
                "dY_frac_lt_fp16_min_normal": _stats(dy32 * s)["frac_lt_fp16_min_normal"],
                "fp16_headroom": FP16_MAX / max(base_max * s, 1e-30),
            }
        )
    return rows


def reduction_scan(dy: torch.Tensor, w: torch.Tensor) -> dict:
    """R3：allow_fp16_reduced_precision_reduction 两档的真实 fp16 GEMM 对比。"""
    ref = _matmul(dy, w)
    # 必须是真 fp16 张量：喂 fp32 张量会跑 fp32 GEMM，开关就没有意义。
    dy16, w16 = dy.half(), w.half()
    prev = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    got: dict[bool, torch.Tensor] = {}
    try:
        for flag in (True, False):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
            got[flag] = torch.matmul(dy16, w16).float()
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev
    emul = _fp16_operand_matmul(dy, w)
    return {
        "reduced_reduction_on": {
            "rel_l2_vs_fp32_ref": _rel_l2(got[True], ref),
            "cos_vs_fp32_ref": _cos(got[True], ref),
        },
        "reduced_reduction_off": {
            "rel_l2_vs_fp32_ref": _rel_l2(got[False], ref),
            "cos_vs_fp32_ref": _cos(got[False], ref),
        },
        "on_vs_off_rel_l2": _rel_l2(got[True], got[False]),
        "real_vs_fp32acc_emul_rel_l2": _rel_l2(got[False], emul),
        "out_float16_frac_zero": float((got[False] == 0).float().mean()),
        "ambient_default": bool(prev),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def sample_paths(paths: list[str], count: int) -> list[str]:
    if count <= 0 or count >= len(paths):
        return list(paths)
    step = (len(paths) - 1) / (count - 1)
    idx = sorted({round(i * step) for i in range(count)})
    return [paths[i] for i in idx]


def analyze_layers(
    dit: torch.nn.Module,
    arms: dict,
    pick: list[str],
    ks: list[int],
    skip_reduction: bool,
    device: torch.device = CUDA,
) -> tuple[list[dict], str]:
    """逐层拆解。以 fp32 臂（若有）的 dY 为真值输入，否则退回 bf16 臂。"""
    truth = "fp32" if "fp32" in arms else "bf16"
    mods = dict(dit.named_modules())
    rows = []
    for name in pick:
        w = dequant_fp32(mods[name])
        dy_truth = arms[truth]["captured"].get(name)
        row: dict = {
            "layer": name,
            "shape": [mods[name].out_features, mods[name].in_features],
            "weight": _stats(w),
            "truth_arm": truth,
            "dY_l2_by_arm": {
                k: arms[k]["layer_stats"].get(name, {}).get("dY", {}).get("l2")
                for k in arms
            },
        }
        if dy_truth is not None:
            dy32 = dy_truth.float().to(device)
            _, ops = compare_ops(dy32, w)
            row["op_on_truth_dY"] = ops
            row["drift_vs_truth"] = {}
            base = dy32.double().reshape(-1)
            for k in arms:
                if k == truth:
                    continue
                cap = arms[k]["captured"].get(name)
                if cap is None:
                    continue
                other = cap.float().to(device).double().reshape(-1)
                row["drift_vs_truth"][k] = {
                    "rel_l2": float((other - base).norm() / (base.norm() + 1e-300)),
                    "cos": float(
                        torch.dot(other, base) / (other.norm() * base.norm() + 1e-300)
                    ),
                    "norm_ratio": float(other.norm() / (base.norm() + 1e-300)),
                }
            row["loss_scale_scan"] = loss_scale_scan(dy32, w, ks)
            if not skip_reduction:
                row["reduction_scan"] = reduction_scan(dy32, w)
        rows.append(row)
        dy = row.get("op_on_truth_dY", {}).get("dY", {})
        f16 = row.get("op_on_truth_dY", {}).get("fp16_real_gemm", {})
        b16 = row.get("op_on_truth_dY", {}).get("bf16_emul", {})
        print(
            f"    {name}: |dY|={dy.get('l2', 0):.3e} max={dy.get('max_abs', 0):.3e} "
            f"<2^-14={dy.get('frac_lt_fp16_min_normal', 0) * 100:.2f}% "
            f"fp16_real={f16.get('rel_l2_vs_fp32_ref', float('nan')):.4f} "
            f"bf16_emul={b16.get('rel_l2_vs_fp32_ref', float('nan')):.2e}"
        )
        del w
    return rows, truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", type=int, default=256, help="图像边长（像素）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output" / "tests" / "krea2_nf4_fp16_grad")
    ap.add_argument("--replay-layers", type=int, default=24, help="逐层拆解抽样数（0=全部）")
    ap.add_argument("--scale-ks", type=str, default="0,1,2,3,4,6,8,10,12")
    ap.add_argument("--loss-scale-k", type=str, default="", help="模型级 loss scaling 的 2^k 列表")
    ap.add_argument("--skip-reduction-scan", action="store_true")
    ap.add_argument(
        "--fp32-arm", type=int, default=1, help="1=额外跑 fp32 compute 臂当模型级参考"
    )
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    device = CUDA
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = args.out or out_dir / f"capture_img{args.img}{suffix}.json"
    ks = [int(x) for x in args.scale_ks.split(",") if x.strip()]
    lsk = [int(x) for x in args.loss_scale_k.split(",") if x.strip()]

    torch.cuda.reset_peak_memory_stats()
    print(f"=== GPU: {torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)} ===")
    blob = prepare_inputs(args.img, out_dir / "cache")

    print(f"[C] 加载自包含 NF4 权重 {KREA2_NF4.name} ...")
    t0 = time.time()
    # 必须 train()：block.forward 只在 self.training 时才走 torch_checkpoint，
    # eval 模式下 grad-ckpt 静默失效，16GB 卡会直接 OOM。
    dit = load_krea2_dit(KREA2_NF4, device=str(device), dtype=torch.bfloat16, eval=False)
    dit.train()
    dit.enable_gradient_checkpointing()
    print(f"    加载 {time.time() - t0:.1f}s alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB")

    paths = linear4bit_paths(dit)
    pick = sample_paths(paths, args.replay_layers)
    print(f"    Linear4bit 层数={len(paths)}，逐层拆解抽样={len(pick)}")

    plan: list[tuple[str, torch.dtype, float]] = []
    if args.fp32_arm:
        plan.append(("fp32", torch.float32, 1.0))
    plan.append(("bf16", torch.bfloat16, 1.0))
    plan.append(("fp16", torch.float16, 1.0))
    plan += [(f"fp16_scale2^{k}", torch.float16, float(2**k)) for k in lsk]

    arms: dict[str, dict] = {}
    for name, dtype, scale in plan:
        print(f"[D] {name} 臂 ...")
        arms[name] = run_arm(
            dit, blob, dtype=dtype, grad_scale=scale, capture=set(pick), device=device
        )
        arms[name]["headline"] = {
            "loss": arms[name]["loss"],
            "latent_grad_l2": float(arms[name]["latent_grad"].norm()),
            "secs": arms[name]["forward_backward_s"],
        }
        print(
            f"    loss={arms[name]['loss']:.6f} |dx_t|={arms[name]['headline']['latent_grad_l2']:.3e} "
            f"{arms[name]['forward_backward_s']:.2f}s"
        )
        torch.cuda.empty_cache()

    reference = "fp32" if "fp32" in arms else "bf16"
    print(f"[E] 模型级跨臂对比（参考臂={reference}）...")
    ref = arms[reference]
    cross = {"reference_arm": reference}
    for name, arm in arms.items():
        if name == reference:
            continue
        row = {
            "velocity_rel_l2": _rel_l2(arm["velocity"], ref["velocity"]),
            "velocity_cos": _cos(arm["velocity"], ref["velocity"]),
            "latent_grad_rel_l2": _rel_l2(arm["latent_grad"], ref["latent_grad"]),
            "latent_grad_cos": _cos(arm["latent_grad"], ref["latent_grad"]),
            "latent_grad_norm_ratio": _norm_ratio(arm["latent_grad"], ref["latent_grad"]),
            "velocity_delta_stats": _stats(arm["velocity"] - ref["velocity"]),
            "latent_grad_stats": _stats(arm["latent_grad"]),
            "reference_latent_grad_stats": _stats(ref["latent_grad"]),
            "signal_restricted_on_reference_grad": signal_restricted(
                arm["latent_grad"], ref["latent_grad"]
            ),
        }
        if "bf16" in arms and name != "bf16":
            row["latent_grad_rel_l2_vs_bf16"] = _rel_l2(arm["latent_grad"], arms["bf16"]["latent_grad"])
            row["latent_grad_cos_vs_bf16"] = _cos(arm["latent_grad"], arms["bf16"]["latent_grad"])
        cross[name] = row

    print("[F] 逐层 intra-op 拆解 ...")
    per_layer, truth_arm = analyze_layers(
        dit, arms, pick, ks, args.skip_reduction_scan, device
    )

    result = {
        "probe": "bench/krea2_nf4_fp16_grad/capture_real.py",
        "date": time.strftime("%Y-%m-%d"),
        "img": args.img,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "flags": {
            "allow_tf32": False,
            "allow_fp16_reduced_precision_reduction": bool(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "gradient_checkpointing": True,
            "fixed_sigma": FIXED_SIGMA,
            "prompt": PROMPT,
            "weights": KREA2_NF4.name,
            "image": "synthetic make_test_image",
            "loss": "mse over all elements (mean) -> dY scales as 1/N",
        },
        "arms": {
            k: {kk: vv for kk, vv in v.items() if kk not in {"velocity", "latent_grad", "layer_stats", "captured"}}
            for k, v in arms.items()
        },
        "cross_arm": cross,
        "truth_arm_for_per_layer": truth_arm,
        "per_layer": per_layer,
        "layer_stats_bf16": arms["bf16"]["layer_stats"],
        "layer_stats_fp16": arms["fp16"]["layer_stats"],
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float))
    print(f"\n[out] {out_path}")

    fp = cross["fp16"]
    print(f"\n=== headline（真权重 + 真激活；参考臂={cross['reference_arm']}；V100，非 sm86 复现） ===")
    print(f"  output rel-L2      = {fp['velocity_rel_l2'] * 100:.4f}%  cos={fp['velocity_cos']:.6f}")
    print(
        f"  latent-grad rel-L2 = {fp['latent_grad_rel_l2'] * 100:.4f}%  "
        f"cos={fp['latent_grad_cos']:.6f}  norm_ratio={fp['latent_grad_norm_ratio']:.4f}"
    )
    for name, row in cross.items():
        if not isinstance(row, dict):
            continue
        extra = ""
        if "latent_grad_rel_l2_vs_bf16" in row:
            extra = (
                f" | vs bf16: rel={row['latent_grad_rel_l2_vs_bf16'] * 100:.4f}% "
                f"cos={row['latent_grad_cos_vs_bf16']:.6f}"
            )
        print(
            f"  [{name}] grad rel-L2={row['latent_grad_rel_l2'] * 100:.4f}% "
            f"cos={row['latent_grad_cos']:.6f} norm={row['latent_grad_norm_ratio']:.4f}{extra}"
        )
    print(f"  peak GPU {result['peak_gpu_gb']:.2f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
