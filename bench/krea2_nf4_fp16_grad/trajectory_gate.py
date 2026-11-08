"""Krea-2 NF4 fp16 候选的**多步梯度 cosine / loss 轨迹门槛**（stage6 遗留要求）。

stage6 文档（`_archive/docs/findings/krea2_3080_speed_stage6.md`「判定」末段）要求：
任何放开 NF4 混合精度前，必须先建立多步梯度 cosine / loss 轨迹门槛。本脚本就是
这个门槛的可执行版本：同一份 NF4 权重、同一份 LoRA 初始化、同一批激活，用两个
compute dtype（bf16 / fp16）各跑 N 步真训练，逐步对账。

为什么不看单步 latent 梯度：`capture_real.py` 已经测到，单步 latent-grad 在 bf16
臂上与 fp32 参考也差 ~100%（cos 0.65）——residual 反向链的放大让该指标不可用。
梯度门槛因此必须落在**可训练参数（LoRA）**上，并看**轨迹**而不是单点。

门槛（沿用 `library/training/adaptive_runtime/precision.py` 的 Tolerances 口径）：
  * 每步 LoRA 梯度 relative-L2 ≤ 0.02
  * 每步 LoRA 梯度 cosine ≥ 0.999
  * 每步 loss 相对偏差 ≤ 1%，且末步 loss 相对偏差 ≤ 1%
  * 轨迹不塌：两臂 loss 都单调下降（末 3 步均值 < 首 3 步均值）

用法::

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/trajectory_gate.py \
      --img 256 --steps 10 --lora-dim 8

输出 JSON 默认写 `output/tests/krea2_nf4_fp16_grad/trajectory_<img>_<tag>.json`。
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss
from library.models.krea2_raw.lora_targets import krea2_target_kwargs
from library.models.krea2_raw.weights import load_krea2_dit
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import LoRANetwork
from networks.lora_modules.lora import LoRAModule
from probe_train import FIXED_SIGMA, PROMPT

from capture_real import (
    CUDA,
    KREA2_NF4,
    _cos,
    _rel_l2,
    _norm_ratio,
    prepare_inputs,
    set_compute_dtype,
)

# 门槛阈值：与 library/training/adaptive_runtime/precision.py::Tolerances 对齐
GRAD_REL_L2_MAX = 0.02
GRAD_COS_MIN = 0.999
LOSS_REL_MAX = 0.01


def install_gqa_expand_patch() -> None:
    """把 enable_gqa 的 SDPA 调用改写成显式 kv 展开的普通 MHA。

    sm70 的 memory-efficient 内核不支持 GQA（q_heads != kv_heads 时直接回退 MATH，
    4096 tokens 峰值 7424MiB）；显式 repeat_interleave 展开后走 EFFICIENT，峰值 224MiB。
    数学等价（反向梯度会正确 sum 回原 kv head 数），实测 rel-L2 ~3e-4 属 fp16 kernel 差异。
    """
    orig = torch.nn.functional.scaled_dot_product_attention

    def patched(q, k, v, *args, **kwargs):
        if kwargs.get("enable_gqa") and q.shape[1] != k.shape[1]:
            g = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
            kwargs["enable_gqa"] = False
        return orig(q, k, v, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched


def build_lora(dit: torch.nn.Module, dim: int, alpha: float) -> LoRANetwork:
    kwargs = {**krea2_target_kwargs(), "lora_dim": dim, "alpha": alpha}
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=dim,
        network_alpha=alpha,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    net = LoRANetwork(text_encoders=[], unet=dit, cfg=cfg, multiplier=1.0)
    net.apply_to(text_encoders=[], unet=dit, apply_text_encoder=False, apply_unet=True)
    return net


def _flat_grad(params) -> torch.Tensor:
    return torch.cat([p.grad.detach().float().reshape(-1) for p in params]).cpu()


def run_trajectory(
    dit: torch.nn.Module,
    net: LoRANetwork,
    blob: dict,
    *,
    dtype: torch.dtype,
    steps: int,
    lr: float,
    init_state: dict,
    loss_scale: float = 1.0,
    device: torch.device = CUDA,
) -> dict:
    """一个 compute dtype 下的 N 步真实 AdamW 轨迹。

    ``loss_scale`` 复现 GradScaler 语义：backward 前把 loss 放大，step 前把梯度
    unscale 回原量级。记录与比较用的都是 **unscale 后**的梯度，所以不同 scale
    的臂可以直接与 bf16 基线对比。
    """
    set_compute_dtype(dit, dtype)
    net.load_state_dict(init_state)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    latents = blob["latents"].to(device=device, dtype=dtype)
    hiddens = blob["hiddens"].to(device=device, dtype=dtype)
    txtmask = blob["txtmask"].to(device=device)
    text_emb = Krea2TextEmbedding(hiddens, txtmask)

    torch.manual_seed(1234)
    noise = torch.randn_like(latents)
    sigma = torch.full((latents.shape[0],), FIXED_SIGMA, device=device, dtype=dtype)
    view = (-1, 1, 1, 1)
    x_t = ((1.0 - sigma.view(*view)) * latents + sigma.view(*view) * noise).detach()
    target = (noise - latents).to(dtype)

    losses, grads, secs, nonfinite = [], [], [], []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        if getattr(dit, "blocks_to_swap", None):
            dit.prepare_block_swap_before_forward()
        velocity = forward_for_loss(dit, x_t.unsqueeze(2), text_emb, sigma)
        # AMP 最佳实践：MSE 在 fp32 里算。若留在 fp16，backward 内部先算
        # 2*(v-t)/N（≈2e-5，fp16 subnormal，只剩 0~2 位有效位）再乘 loss_scale，
        # 放大也救不回已丢的位；转 fp32 后该中间量精确，乘完 scale 才 cast 回 fp16。
        loss = torch.nn.functional.mse_loss(
            velocity.float(), target.unsqueeze(2).float()
        )
        (loss * loss_scale).backward()
        if loss_scale != 1.0:
            # GradScaler 的 unscale：必须在 clip/step 之前，否则 clip 阈值失真。
            for p in params:
                if p.grad is not None:
                    p.grad.div_(loss_scale)
        g_flat = _flat_grad(params)
        grads.append(g_flat)
        bad = bool(not torch.isfinite(g_flat).all())
        nonfinite.append(bad)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.synchronize()
        secs.append(time.time() - t0)
        losses.append(float(loss.detach()))
        flag = "  <-- 非有限" if bad else ""
        print(
            f"    step {step:2d}: loss={losses[-1]:.6f} ({secs[-1]:.2f}s){flag}"
        )
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "loss_scale": loss_scale,
        "losses": losses,
        "grads": grads,
        "nonfinite_steps": [i for i, v in enumerate(nonfinite) if v],
        "step_seconds": secs,
        "lora_state": {k: v.detach().float().cpu() for k, v in net.state_dict().items()},
    }


def evaluate(bf: dict, fp: dict, *, label: str = "fp16") -> dict:
    rows = []
    for i, (gb, gf) in enumerate(zip(bf["grads"], fp["grads"])):
        rows.append(
            {
                "step": i,
                "loss_bf16": bf["losses"][i],
                "loss_candidate": fp["losses"][i],
                "loss_rel_diff": abs(fp["losses"][i] - bf["losses"][i]) / abs(bf["losses"][i]),
                "grad_rel_l2": _rel_l2(gf, gb),
                "grad_cos": _cos(gf, gb),
                "grad_norm_ratio": _norm_ratio(gf, gb),
            }
        )
    final_lora_rel = _rel_l2(
        torch.cat([v.reshape(-1) for v in fp["lora_state"].values()]),
        torch.cat([v.reshape(-1) for v in bf["lora_state"].values()]),
    )
    checks = {
        "grad_rel_l2_every_step": all(r["grad_rel_l2"] <= GRAD_REL_L2_MAX for r in rows),
        "grad_cos_every_step": all(r["grad_cos"] >= GRAD_COS_MIN for r in rows),
        "loss_rel_every_step": all(r["loss_rel_diff"] <= LOSS_REL_MAX for r in rows),
        # steps<3 时首尾窗口重叠，该判据无意义，直接放行。
        "loss_trajectory_not_collapsed": (
            len(bf["losses"]) < 3
            or (
                sum(bf["losses"][-3:]) / 3 < sum(bf["losses"][:3]) / 3
                and sum(fp["losses"][-3:]) / 3 < sum(fp["losses"][:3]) / 3
            )
        ),
        "no_nonfinite_grad": not fp["nonfinite_steps"],
    }
    return {
        "candidate": label,
        "loss_scale": fp["loss_scale"],
        "steps": rows,
        "worst_grad_rel_l2": max(r["grad_rel_l2"] for r in rows),
        "worst_grad_cos": min(r["grad_cos"] for r in rows),
        "worst_loss_rel_diff": max(r["loss_rel_diff"] for r in rows),
        "final_lora_param_rel_l2": final_lora_rel,
        "nonfinite_steps": fp["nonfinite_steps"],
        "thresholds": {
            "grad_rel_l2_max": GRAD_REL_L2_MAX,
            "grad_cos_min": GRAD_COS_MIN,
            "loss_rel_max": LOSS_REL_MAX,
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lora-dim", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument(
        "--loss-scale-k",
        type=str,
        default="0",
        help="fp16 臂的 loss scale 2^k 列表（逗号分隔）；k=0 即不加 scale。"
        "k>15 会撞 fp16 标量上限 65504 而污染梯度",
    )
    ap.add_argument(
        "--fp32-reference",
        type=int,
        default=0,
        help="1=额外跑 fp32 compute 臂当唯一裁判（显存约翻倍，256² 在 16GB 卡上可能 OOM）",
    )
    ap.add_argument(
        "--expand-gqa",
        action="store_true",
        help="把 enable_gqa 改写成显式 kv 展开的 MHA，绕过 sm70 memory-efficient 内核"
        "不支持 GQA 的限制（1024² 免 OOM）",
    )
    ap.add_argument(
        "--skip-bf16",
        action="store_true",
        help="跳过 bf16 基准臂（bf16 在 sm70 无 memory-efficient 内核，1024² 必 OOM）；"
        "此时以第一个 fp16 臂当裁判做自收敛对比",
    )
    ap.add_argument(
        "--blocks-to-swap",
        type=int,
        default=0,
        help="0=全驻留显存；>0 走生产 block swap（换显存、慢，用于 1024² 免 OOM）",
    )
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output" / "tests" / "krea2_nf4_fp16_grad")
    args = ap.parse_args()

    if args.expand_gqa:
        install_gqa_expand_patch()
        print("    GQA expand patch 已启用")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = CUDA
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"trajectory_{args.img}{('_' + args.tag) if args.tag else ''}.json"

    print(f"=== GPU: {torch.cuda.get_device_name(0)} | img={args.img} steps={args.steps} ===")
    blob = prepare_inputs(args.img, args.out_dir / "cache")
    if args.blocks_to_swap:
        # 生产顺序：CPU 构造 -> offloader -> 只把非 swap block 搬到 GPU。
        dit = load_krea2_dit(KREA2_NF4, device="cpu", dtype=torch.bfloat16, eval=False)
        dit.train()
        dit.enable_gradient_checkpointing()
        dit.enable_block_swap(args.blocks_to_swap, device)
        dit.move_to_device_except_swap_blocks(device)
        dit.switch_block_swap_for_training()
        print(f"    block swap = {args.blocks_to_swap} blocks")
    else:
        dit = load_krea2_dit(KREA2_NF4, device=str(device), dtype=torch.bfloat16, eval=False)
        dit.train()
        dit.enable_gradient_checkpointing()

    torch.manual_seed(7)
    net = build_lora(dit, args.lora_dim, args.lora_alpha)
    net = net.to(device)
    init_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"LoRA 可训参数 {n_train / 1e6:.2f}M, dim={args.lora_dim}, lr={args.lr}")

    ks = [int(x) for x in args.loss_scale_k.split(",") if x.strip()]
    plan: list[tuple[str, torch.dtype, float]] = []
    if args.fp32_reference:
        plan.append(("fp32", torch.float32, 1.0))
    if not args.skip_bf16:
        plan.append(("bf16", torch.bfloat16, 1.0))
    for k in ks:
        name = "fp16" if k == 0 else f"fp16_scale2^{k}"
        plan.append((name, torch.float16, float(2**k)))

    results = {}
    for name, dtype, scale in plan:
        print(f"[{name}] {args.steps} 步轨迹 (loss_scale={scale:g}) ...")
        results[name] = run_trajectory(
            dit,
            net,
            blob,
            dtype=dtype,
            steps=args.steps,
            lr=args.lr,
            init_state=init_state,
            loss_scale=scale,
        )
        torch.cuda.empty_cache()

    if "fp32" in results:
        judge = "fp32"
    elif "bf16" in results:
        judge = "bf16"
    else:
        judge = next(iter(results))
    base = results[judge]
    verdicts = {
        name: evaluate(base, results[name], label=name)
        for name in results
        if name != judge
    }
    vs_bf16 = {}
    if judge != "bf16" and "bf16" in results:
        vs_bf16 = {
            name: evaluate(results["bf16"], results[name], label=name)
            for name in results
            if name not in {"bf16"}
        }
    payload = {
        "probe": "bench/krea2_nf4_fp16_grad/trajectory_gate.py",
        "date": time.strftime("%Y-%m-%d"),
        "img": args.img,
        "steps": args.steps,
        "lora_dim": args.lora_dim,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "loss_scale_ks": ks,
        "weights": KREA2_NF4.name,
        "gpu": torch.cuda.get_device_name(0),
        "prompt": PROMPT,
        "note": "LoRA 参数两臂都是 fp32 master；只有 NF4 compute_dtype 变化。"
        "loss_scale 臂在 step 前 unscale，比较用 unscale 后梯度",
        "judge_arm": judge,
        "loss_judge": base["losses"],
        "loss_by_arm": {n: r["losses"] for n, r in results.items()},
        "verdicts_vs_judge": verdicts,
        "verdicts_vs_bf16": vs_bf16,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float))
    print(f"\n[out] {out_path}")
    print(f"\n########## 裁判 = {judge} ##########")
    for name, v in verdicts.items():
        print(f"\n=== [{name}] vs {judge}, loss_scale={v['loss_scale']:g} ===")
        for kk, vv in v["checks"].items():
            print(f"  {kk}: {'PASS' if vv else 'FAIL'}")
        print(f"  worst grad rel-L2 = {v['worst_grad_rel_l2'] * 100:.3f}% (限 2%)")
        print(f"  worst grad cos    = {v['worst_grad_cos']:.6f} (下限 0.999)")
        print(f"  worst loss rel    = {v['worst_loss_rel_diff'] * 100:.3f}% (限 1%)")
        print(f"  末步 LoRA 参数 rel-L2 = {v['final_lora_param_rel_l2'] * 100:.3f}%")
        print(f"  GATE: {'PASS' if v['gate_pass'] else 'FAIL'}")
    if vs_bf16:
        print("\n########## 对照：各臂 vs bf16 ##########")
        for name, v in vs_bf16.items():
            print(
                f"  [{name}] worst grad rel-L2 = {v['worst_grad_rel_l2'] * 100:.3f}%  "
                f"cos = {v['worst_grad_cos']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
