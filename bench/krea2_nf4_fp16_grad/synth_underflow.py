"""Krea-2 NF4 fp16 backward 下溢假说的纯 CPU 数学探针。

bitsandbytes ``MatMul4Bit.backward`` 实际做的是
``grad_A = matmul(grad_output, F.dequantize_4bit(B, qs).to(grad_output.dtype).t())``：
反量化权重是 bf16（``qs.dtype=bfloat16``），再 cast 到 ``grad_output`` 的 dtype。
fp16 臂因此把 dY（grad_output）舍入到 fp16 后才参与 matmul；fp16 指数范围窄
（最小正规数 6.1e-5、最小 subnormal 6e-8），dY 量级掉到 1e-7 附近时大量元素落入
subnormal 或被冲零，dX 误差随之爆炸。本脚本把这条操作链逐位复现成纯 CPU 数学仿真，
不加载模型、不 import bitsandbytes、不 import 本仓库模块，可离线重跑：

* A. dY 量级扫描（O(1)→1e-9）：fp64 参考 / bf16 操作数臂 / fp16 操作数臂；
  A2 细分扫描定位「rel_l2≈0.3565 / cos≈0.9434」指纹出现的量级；
* B. 误差归因：按 ``fp16 normal / subnormal / 冲零`` 三桶拆 dX 误差平方和，
  给 signal-restricted rel-L2 / cosine；B2 做 dY 能量预算分解；
* C. loss-scale 扫描：dY 乘 2^k，看 fp16 臂是否单调恢复并饱和到 bf16 水平；
* D. fp16 归约（reduced precision reduction）上界**模拟**，用分块累加近似。

边界声明：仿真只覆盖「操作数舍入 + 浮点范围」这条数学链，**不代表** sm86 cuBLAS、
V100 tensor core 或任何真实 kernel 的行为，也不覆盖 TF32、归约顺序、kernel 内部
upcast；D 的归约差异是 chunked 模拟，不是真机 kernel 测量。

运行（纯 CPU，禁止使用 CUDA）::

    CUDA_VISIBLE_DEVICES='' .venv/bin/python bench/krea2_nf4_fp16_grad/synth_underflow.py

默认写同目录 ``synth_underflow.json``，终端打印人类可读表格；``--repeat-check``
跑两遍并比较两次 JSON（去掉时间字段）是否完全一致。
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch

# fp16 关键常量（IEEE 754 binary16）
FP16_MIN_NORMAL = 6.103515625e-05  # 2**-14
FP16_MIN_SUBNORMAL = 5.960464477539063e-08  # 2**-24
FP16_MAX = 65504.0

# bitsandbytes NF4 的 16 级 codebook（归一化到 absmax 的 quantile 值）
NF4_LEVELS: tuple[float, ...] = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)

DEFAULT_SCALES: tuple[float, ...] = (1.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
DEFAULT_ATTR_SCALES: tuple[float, ...] = (1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
DEFAULT_BUDGET_SCALES: tuple[float, ...] = (1e-6, 1e-7, 5e-8, 3e-8, 1e-8)
# 上游报告的实测指纹（外部输入，不是本脚本测出来的）
REPORTED_FP16_GRAD_REL_L2 = 0.3565
REPORTED_FP16_GRAD_COS = 0.9434


# --- 舍入、量化原语与问题构造 ---
def round_fp16(x: torch.Tensor) -> torch.Tensor:
    """把 fp32 张量舍入到 fp16 再无损提升回 fp32（不在 fp16 上做 matmul）。"""
    return x.to(torch.float16).to(torch.float32)


def round_bf16(x: torch.Tensor) -> torch.Tensor:
    """把 fp32 张量舍入到 bf16 再无损提升回 fp32。"""
    return x.to(torch.bfloat16).to(torch.float32)


def nf4_quant_dequant(weight: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """手搓 NF4 分块量化 + 反量化（absmax 缩放 + 16 级最近邻 codebook）。"""
    if weight.shape[-1] % block_size != 0:
        raise ValueError(f"最后一维必须能被 block_size={block_size} 整除")
    levels = torch.tensor(NF4_LEVELS, dtype=torch.float32)
    flat = weight.reshape(-1, weight.shape[-1] // block_size, block_size)
    absmax = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    idx = (flat / absmax).unsqueeze(-1).sub(levels).abs().argmin(dim=-1)
    return (levels[idx] * absmax).reshape(weight.shape)


def build_problem(m_tokens: int, out_features: int, in_features: int, seed: int) -> dict[str, Any]:
    """构造一次实验的静态张量：NF4 反量化权重 + 单位方差 dY 基向量。"""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_features, in_features, generator=gen) * 0.02  # 近似 DiT Linear 统计
    wdq = nf4_quant_dequant(w)  # fp32 反量化权重（bnb 里会先落成 bf16）
    w_bf16 = round_bf16(wdq)  # bnb 链路：dequant 输出 bf16，再 cast 到 grad_output.dtype
    return {
        "wdq": wdq,
        "w_arm": round_fp16(w_bf16),  # bf16 -> fp16 在 fp16 正规范围内无损
        "dy_base": torch.randn(m_tokens, out_features, generator=gen),
        "shape": {"M": m_tokens, "N": out_features, "K": in_features},
        "weight_operand_identical_across_arms": bool(torch.equal(w_bf16, round_fp16(w_bf16))),
        "nf4_weight_rel_l2_vs_fp32": _norm_ratio(wdq - w, w),
    }


# --- 度量 ---
def _norm_ratio(x: torch.Tensor, ref: torch.Tensor) -> float:
    """||x|| / ||ref||（fp64 计算；用于误差能量与信号能量的比例）。"""
    denom = torch.linalg.vector_norm(ref.double()).item()
    return torch.linalg.vector_norm(x.double()).item() / denom if denom else float("nan")


def compare(arm: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    """返回 rel_l2 / cosine / 范数比；arm 全零时按约定取 rel_l2=1、cosine=0。"""
    a, r = arm.double().reshape(-1), ref.double().reshape(-1)
    rn = torch.linalg.vector_norm(r)
    if rn.item() == 0.0:
        return {"rel_l2": None, "cosine": None, "norm_ratio": None, "note": "ref 全零"}
    an = torch.linalg.vector_norm(a)
    if an.item() == 0.0:
        return {"rel_l2": 1.0, "cosine": 0.0, "norm_ratio": 0.0, "note": "arm 全零（全部冲零）"}
    return {"rel_l2": (torch.linalg.vector_norm(a - r) / rn).item(),
            "cosine": torch.nn.functional.cosine_similarity(a, r, dim=0).item(),
            "norm_ratio": (an / rn).item()}


def dy_stats(dy: torch.Tensor) -> dict[str, float]:
    """dY 元素级分布指纹：亚正规数占比、冲零占比、最大幅值。"""
    n, absdy, dy16 = dy.numel(), dy.abs(), round_fp16(dy)
    return {
        "frac_below_min_normal": (absdy < FP16_MIN_NORMAL).sum().item() / n,
        "frac_below_min_subnormal": (absdy < FP16_MIN_SUBNORMAL).sum().item() / n,
        "frac_flushed_to_zero": ((dy16 == 0) & (dy != 0)).sum().item() / n,
        "frac_fp16_subnormal": ((dy16 != 0) & (dy16.abs() < FP16_MIN_NORMAL)).sum().item() / n,
        "frac_fp16_normal": (dy16.abs() >= FP16_MIN_NORMAL).sum().item() / n,
        "frac_exact_zero": (dy == 0).sum().item() / n,
        "max_abs_dy": absdy.max().item(),
        "rms_dy": dy.pow(2).mean().sqrt().item(),
    }


def _sumsq(x: torch.Tensor) -> float:
    return x.double().pow(2).sum().item()


# --- 实验 A：dY 量级扫描 ---
def run_scale_sweep(problem: dict[str, Any], scales: Sequence[float]) -> list[dict[str, Any]]:
    """逐档比较 bf16/fp16 操作数臂；``*_fp32_weight`` 变体剥离权重舍入底噪。"""
    wdq, w_arm, dy_base = problem["wdq"], problem["w_arm"], problem["dy_base"]
    wt, rows = wdq.double().t(), []
    for s in scales:
        dy = dy_base * s
        ref = torch.matmul(dy.double(), wt)
        dy16, dybf = round_fp16(dy), round_bf16(dy)
        rows.append({
            "dy_scale": s,
            "fp64_ref_norm": torch.linalg.vector_norm(ref).item(),
            "bf16_arm": compare(torch.matmul(dybf, w_arm.t()), ref),
            "fp16_arm": compare(torch.matmul(dy16, w_arm.t()), ref),
            "bf16_arm_fp32_weight": compare(torch.matmul(dybf, wdq.t()), ref),
            "fp16_arm_fp32_weight": compare(torch.matmul(dy16, wdq.t()), ref),
            "dy_stats": dy_stats(dy), "fp16_max_abs_dy": dy16.abs().max().item(),
            "fp16_overflow": bool(dy16.abs().max().item() >= FP16_MAX),
        })
    return rows


def run_fingerprint_scan(problem: dict[str, Any], lo_exp: float, hi_exp: float,
                         steps: int) -> dict[str, Any]:
    """在 log2 网格上细分扫描，定位复现「rel_l2≈0.3565、cos≈0.9434」的量级。"""
    wdq, w_arm, dy_base = problem["wdq"], problem["w_arm"], problem["dy_base"]
    wt, best, curve = wdq.double().t(), None, []
    for i in range(steps + 1):
        s = 2.0 ** (lo_exp + (hi_exp - lo_exp) * i / steps)
        dy = dy_base * s
        m = compare(torch.matmul(round_fp16(dy), w_arm.t()), torch.matmul(dy.double(), wt))
        rel = 1.0 if m["rel_l2"] is None else m["rel_l2"]
        cos = 0.0 if m["cosine"] is None else m["cosine"]
        stats = dy_stats(dy)
        curve.append({"dy_scale": s, "fp16_rel_l2": rel, "fp16_cosine": cos,
                      "frac_flushed_to_zero": stats["frac_flushed_to_zero"],
                      "frac_fp16_subnormal": stats["frac_fp16_subnormal"]})
        dist = abs(rel - REPORTED_FP16_GRAD_REL_L2) + abs(cos - REPORTED_FP16_GRAD_COS)
        if best is None or dist < best["distance"]:
            best = {"dy_scale": s, "rel_l2": rel, "cosine": cos, "distance": dist, "dy_stats": stats}
    return {"target": {"rel_l2": REPORTED_FP16_GRAD_REL_L2, "cosine": REPORTED_FP16_GRAD_COS},
            "grid_log2": [lo_exp, hi_exp, steps], "best_match": best, "curve": curve}


# --- 实验 B：误差归因分解 ---
def _bucket_masks(dy: torch.Tensor) -> dict[str, torch.Tensor]:
    """按 fp16 表示的结局分桶：normal / subnormal / 冲零。"""
    abs16 = round_fp16(dy).abs()
    return {"fp16_normal": abs16 >= FP16_MIN_NORMAL,
            "fp16_subnormal": (abs16 > 0) & (abs16 < FP16_MIN_NORMAL),
            "fp16_flushed": abs16 == 0}


def attribute_scale(problem: dict[str, Any], s: float, rel_quantile: float = 0.9) -> dict[str, Any]:
    """把「仅 dY 舍入」造成的 dX 误差平方和按 dY 桶归因，并给 restricted 度量。"""
    wdq, dy = problem["wdq"], problem["dy_base"] * s
    err_dy = round_fp16(dy) - dy  # dY 舍入误差；权重统一取 fp32 以剥离权重舍入底噪
    err_total = torch.matmul(err_dy, wdq.t())
    total_sq, buckets, additive = _sumsq(err_total), {}, 0.0
    for name, mask in _bucket_masks(dy).items():
        sq = _sumsq(torch.matmul(err_dy * mask, wdq.t()))
        additive += sq
        buckets[name] = {"element_fraction": mask.sum().item() / mask.numel(), "err_sq_norm": math.sqrt(sq),
                         "err_sq_share_additive": (sq / total_sq) if total_sq > 0 else None}
    abs_dy = dy.abs()
    thresh_q = torch.quantile(abs_dy.flatten().float(), rel_quantile).item()
    return {
        "dy_scale": s,
        "dy_rounding_only": {
            "rel_l2_vs_fp64_ref": _norm_ratio(err_total, torch.matmul(dy.double(), wdq.double().t())),
            "err_sq_total": total_sq,
            "additive_share_sum": additive / total_sq if total_sq > 0 else None,
            "cross_term_share": (total_sq - additive) / total_sq if total_sq > 0 else None,
        },
        "buckets": buckets,
        # 绝对阈值口径：|dY| >= fp16 最小正规数（小量级下该 mask 会空，属预期结果）
        "signal_restricted_abs_threshold": _restricted(dy, abs_dy >= FP16_MIN_NORMAL, wdq,
                                                       FP16_MIN_NORMAL),
        # 辅助相对阈值口径（|dY| 前 10%），保证每档都有定义
        "signal_restricted_rel_quantile": _restricted(dy, abs_dy >= thresh_q, wdq, thresh_q,
                                                      rel_quantile),
    }


def _restricted(dy: torch.Tensor, mask: torch.Tensor, wdq: torch.Tensor, threshold: float,
                quantile: float | None = None) -> dict[str, Any]:
    """只在 mask 命中的 dY 元素上重算 fp16 臂 vs 参考（其余元素两侧同样清零）。"""
    n_signal = int(mask.sum().item())
    out: dict[str, Any] = {"threshold_abs": threshold, "quantile": quantile,
                           "n_signal": n_signal, "signal_fraction": n_signal / mask.numel()}
    if n_signal == 0:
        out.update({"rel_l2": None, "cosine": None, "norm_ratio": None,
                    "note": "该量级下没有 fp16 正规元素，绝对阈值口径不可定义"})
        return out
    dy_r = dy * mask
    out.update(compare(torch.matmul(round_fp16(dy_r), wdq.t()),
                       torch.matmul(dy_r.double(), wdq.double().t())))
    return out


def run_energy_budget(problem: dict[str, Any], scales: Sequence[float]) -> dict[str, Any]:
    """B2：dY 能量预算分解（信号能量 / 误差能量按桶拆开）。

    纯「冲零损失能量」模型的解析预测是 rel_l2 ≈ sqrt(冲零元素的信号能量份额)；
    实测 rel_l2² 与该预测的差额即 subnormal 固定量化步长（6e-8）带来的额外误差。
    """
    wdq, dy_base = problem["wdq"], problem["dy_base"]
    wt, rows = wdq.double().t(), []
    for s in scales:
        dy = dy_base * s
        err = round_fp16(dy) - dy
        total_signal, total_err = _sumsq(dy), _sumsq(err)
        buckets = {name: {"signal_energy_share": _sumsq(dy * mask) / total_signal,
                          "err_energy_share": _sumsq(err * mask) / total_err}
                   for name, mask in _bucket_masks(dy).items()}
        rel = _norm_ratio(torch.matmul(err, wdq.t()), torch.matmul(dy.double(), wt))
        flushed = buckets["fp16_flushed"]["signal_energy_share"]
        rows.append({"dy_scale": s, "fp16_rel_l2": rel, "fp16_rel_l2_squared": rel**2,
                     "buckets": buckets, "flush_only_model_rel_l2": math.sqrt(flushed),
                     "granularity_residual_sq": rel**2 - flushed})
    tan_sq = 1.0 / REPORTED_FP16_GRAD_COS**2 - 1.0
    return {
        "reported_fingerprint_consistency": {
            "reported_rel_l2": REPORTED_FP16_GRAD_REL_L2,
            "reported_cosine": REPORTED_FP16_GRAD_COS,
            "rel_l2_squared": REPORTED_FP16_GRAD_REL_L2**2,
            "pure_energy_loss_prediction_tan_sq": tan_sq,
            "orthogonal_residual": math.sqrt(max(REPORTED_FP16_GRAD_REL_L2**2 - tan_sq, 0.0)),
            "interpretation": "rel_l2²≈tan²θ 说明误差几乎全是幅度（能量被压碎）损失，方向扰动很小",
        },
        "rows": rows,
    }


# --- 实验 C：loss-scale 扫描 ---
def run_loss_scale(problem: dict[str, Any], s: float, k_max: int,
                   bf16_target: float | None) -> dict[str, Any]:
    """把 dY 乘以 2^k 后重算 fp16 臂，检查误差单调性与饱和点。"""
    wdq, w_arm = problem["wdq"], problem["w_arm"]
    wt, dy0, rows = wdq.double().t(), problem["dy_base"] * s, []
    for k in range(k_max + 1):
        dy = dy0 * (2.0**k)  # 2 的幂缩放本身无损
        ref, stats = torch.matmul(dy.double(), wt), dy_stats(dy)
        rows.append({"k": k, "loss_scale": 2.0**k, "max_abs_dy": stats["max_abs_dy"],
                     "fp16_arm": compare(torch.matmul(round_fp16(dy), w_arm.t()), ref),
                     "bf16_arm_rel_l2": compare(torch.matmul(round_bf16(dy), w_arm.t()), ref)["rel_l2"],
                     "headroom_to_fp16_max": FP16_MAX / max(stats["max_abs_dy"], 1e-30),
                     "frac_fp16_normal": stats["frac_fp16_normal"], "overflow": bool(stats["max_abs_dy"] > FP16_MAX),
                     "frac_fp16_subnormal": stats["frac_fp16_subnormal"],
                     "frac_flushed_to_zero": stats["frac_flushed_to_zero"]})
    rels = [r["fp16_arm"]["rel_l2"] for r in rows]
    return {"dy_scale": s, "bf16_target_rel_l2": bf16_target, "rows": rows,
            "monotonic_non_increasing_tol_1e-3": _is_non_increasing(rels, 1e-3),
            "saturation_k_within_10pct": _first_saturated(rows, bf16_target, 1.10),
            "saturation_k_within_25pct": _first_saturated(rows, bf16_target, 1.25),
            "core_window_k0_12_rel_l2": [rels[r["k"]] for r in rows if r["k"] <= 12],
            "final_rel_l2_at_k_max": rels[-1],
            "overflow_first_k": next((r["k"] for r in rows if r["overflow"]), None)}


def _is_non_increasing(values: Sequence[float | None], rel_tol: float) -> bool:
    """相对容差下的单调不增判定（平台区存在 1e-7 级舍入抖动）。"""
    vals = [v for v in values if v is not None]
    return all(b <= a * (1.0 + rel_tol) for a, b in itertools.pairwise(vals))


def _first_saturated(rows: Sequence[dict[str, Any]], bf16_target: float | None,
                     tol: float) -> int | None:
    """首个 rel_l2 落进 bf16 目标 ``tol`` 倍以内的 k（即已回到 bf16 精度水平）。"""
    hits = [r["k"] for r in rows
            if bf16_target is not None and r["fp16_arm"]["rel_l2"] is not None
            and r["fp16_arm"]["rel_l2"] <= bf16_target * tol]
    return int(hits[0]) if hits else None


# --- 实验 D：fp16 归约上界（模拟） ---
def run_reduction_sim(problem: dict[str, Any], scales: Sequence[float],
                      chunk: int = 16) -> dict[str, Any]:
    """用分块累加模拟「fp32 累加 vs fp16 累加」，给出归约精度上界估计。"""
    wdq, wt = problem["wdq"], problem["wdq"].double().t()
    rows = []
    for s in scales:
        dy = problem["dy_base"] * s
        dy16 = round_fp16(dy)  # 两臂共用同一 fp16 操作数，只比较归约差异
        ref = torch.matmul(dy.double(), wt)
        acc32 = _chunked_accumulate(dy16, wdq, chunk, torch.float32)
        acc16 = _chunked_accumulate(dy16, wdq, chunk, torch.float16)
        rows.append({"dy_scale": s, "fp32_accum_vs_ref": compare(acc32, ref),
                     "fp16_accum_vs_ref": compare(acc16, ref),
                     "fp16_vs_fp32_accum": compare(acc16, acc32)})
    return {"simulated": True, "rows": rows, "chunk_size": chunk,
            "reduction_dim": problem["shape"]["N"],
            "note": "chunked 累加模拟，不是真机 kernel；只估算 reduced-precision reduction 量级"}


def _chunked_accumulate(dy: torch.Tensor, w: torch.Tensor, chunk: int,
                        acc_dtype: torch.dtype) -> torch.Tensor:
    """沿归约维分块累加：每块先 fp32 求和，再按 acc_dtype 舍入后叠加。"""
    out = torch.zeros(dy.shape[0], w.shape[0], dtype=acc_dtype)
    for start in range(0, dy.shape[1], chunk):
        part = torch.matmul(dy[:, start:start + chunk], w[:, start:start + chunk].t())
        out = (out.float() + part).to(acc_dtype)
    return out.float()


# --- 实验编排 ---
def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(p) for p in text.replace(";", ",").split(",") if p.strip())


def _parse_shapes(text: str) -> tuple[tuple[int, int, int], ...]:
    """解析 ``M,N,K`` 三元组；多组用 ``;`` 分隔。"""
    dims = [[int(v) for v in spec.split(",")] for spec in text.split(";") if spec.strip()]
    if not dims or any(len(d) != 3 for d in dims):
        raise ValueError(f"--shapes 需要 M,N,K 三元组: {text!r}")
    return tuple((d[0], d[1], d[2]) for d in dims)


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    """执行 A/A2/B/B2/C/D，返回待落盘的 JSON 结构。"""
    scales, attr_scales = _parse_floats(args.scales), _parse_floats(args.attr_scales)
    shapes, experiments = _parse_shapes(args.shapes), []
    for idx, (m_tokens, out_features, in_features) in enumerate(shapes):
        problem = build_problem(m_tokens, out_features, in_features, args.seed + idx)
        sweep = run_scale_sweep(problem, scales)
        by_scale = {r["dy_scale"]: r for r in sweep}
        worst = max(sweep, key=lambda r: r["fp16_arm"]["rel_l2"] or 0.0)
        # 最差档 + 指纹档各扫一遍 loss scale，便于和上游数字对照
        c_targets = [(worst["dy_scale"], worst["bf16_arm"]["rel_l2"])]
        if 1e-7 in by_scale and 1e-7 != worst["dy_scale"]:
            c_targets.append((1e-7, by_scale[1e-7]["bf16_arm"]["rel_l2"]))
        experiments.append({
            "shape": problem["shape"], "seed": args.seed + idx,
            "nf4": {"block_size": 64, "codebook_levels": len(NF4_LEVELS),
                    "weight_rel_l2_vs_fp32": problem["nf4_weight_rel_l2_vs_fp32"],
                    "weight_operand_identical_across_arms":
                        problem["weight_operand_identical_across_arms"]},
            "A_dy_scale_sweep": sweep,
            "A2_fingerprint_scan": None if args.skip_scan else run_fingerprint_scan(
                problem, args.scan_lo, args.scan_hi, args.scan_steps),
            "B_error_attribution": [attribute_scale(problem, s) for s in attr_scales if s in by_scale],
            "B2_energy_budget": run_energy_budget(problem, _parse_floats(args.budget_scales)),
            "C_loss_scale": [run_loss_scale(problem, s, args.k_max, t) for s, t in c_targets],
            "D_reduction_sim": None if args.skip_d else run_reduction_sim(
                problem, _parse_floats(args.d_scales)),
        })
    return {
        "probe": "krea2_nf4_fp16_grad/synth_underflow",
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "env": {"device": "cpu", "cpu_only": True, "cuda_used": False,
                "note": "纯 CPU 数学仿真；未初始化 CUDA，未 import bitsandbytes、未 import 本仓库模块",
                "torch_threads": torch.get_num_threads(), "python": sys.version.split()[0],
                "platform": platform.platform(), "machine": platform.machine()},
        "params": {"seed": args.seed, "scales": list(scales), "attr_scales": list(attr_scales),
                   "k_max": args.k_max,
                   "shapes": [{"M": m, "N": n, "K": k} for m, n, k in shapes],
                   "fp16_constants": {"min_normal": FP16_MIN_NORMAL,
                                      "min_subnormal": FP16_MIN_SUBNORMAL, "max": FP16_MAX},
                   "reported_fingerprints": {"grad_rel_l2": REPORTED_FP16_GRAD_REL_L2,
                                             "grad_cosine": REPORTED_FP16_GRAD_COS}},
        "experiments": experiments,
    }


# --- 终端报告 ---
def _f(value: float | None, digits: int = 5) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def print_report(payload: dict[str, Any]) -> None:
    """打印人类可读表格。"""
    print(f"probe = {payload['probe']}  torch = {payload['torch']}  date = {payload['date']}")
    print(f"env   = {payload['env']['note']}  threads = {payload['env']['torch_threads']}")
    for exp in payload["experiments"]:
        sh = exp["shape"]
        print(f"\n=== shape M={sh['M']} N={sh['N']} K={sh['K']}  seed={exp['seed']} ===")
        print(f"NF4 权重 rel_l2 = {_f(exp['nf4']['weight_rel_l2_vs_fp32'])}  "
              f"两臂权重操作数一致 = {exp['nf4']['weight_operand_identical_across_arms']}")
        _print_a(exp)
        _print_b(exp)
        _print_cd(exp)


def _print_a(exp: dict[str, Any]) -> None:
    print("\n[A] dY 量级扫描（fp16/bf16 臂均用 bnb 链路的 bf16 权重操作数）")
    print(f"{'s_dY':>9} {'bf16 rel':>9} {'fp16 rel':>9} {'fp16 cos':>9} {'fp16 norm':>9} "
          f"{'bf16*f32w':>10} {'fp16*f32w':>10} {'sub':>6} {'flush':>6} {'max|dY|':>10}")
    for r in exp["A_dy_scale_sweep"]:
        st = r["dy_stats"]
        print(f"{r['dy_scale']:>9.0e} {_f(r['bf16_arm']['rel_l2']):>9} "
              f"{_f(r['fp16_arm']['rel_l2']):>9} {_f(r['fp16_arm']['cosine']):>9} "
              f"{_f(r['fp16_arm']['norm_ratio']):>9} {_f(r['bf16_arm_fp32_weight']['rel_l2']):>10} "
              f"{_f(r['fp16_arm_fp32_weight']['rel_l2']):>10} {st['frac_fp16_subnormal']:>6.3f} "
              f"{st['frac_flushed_to_zero']:>6.3f} {st['max_abs_dy']:>10.3g}")
    scan = exp["A2_fingerprint_scan"]
    if not scan:
        return
    b = scan["best_match"]
    print(f"[A2] 指纹细分扫描（log2 网格 {scan['grid_log2']}）：目标 rel={scan['target']['rel_l2']} "
          f"cos={scan['target']['cosine']} -> 最接近 s_dY={b['dy_scale']:.4g} "
          f"rel={_f(b['rel_l2'])} cos={_f(b['cosine'])} "
          f"flush={b['dy_stats']['frac_flushed_to_zero']:.3f} sub={b['dy_stats']['frac_fp16_subnormal']:.3f}")


def _print_b(exp: dict[str, Any]) -> None:
    print("\n[B] 误差归因（仅 dY 舍入误差，权重统一 fp32）")
    for r in exp["B_error_attribution"]:
        conv = r["dy_rounding_only"]
        print(f"  s={r['dy_scale']:.0e} err_rel_l2={_f(conv['rel_l2_vs_fp64_ref'])} "
              f"additive={_f(conv['additive_share_sum'], 4)} cross={_f(conv['cross_term_share'], 4)}")
        for name, b in r["buckets"].items():
            print(f"    {name:<16} elems={b['element_fraction']:>6.3f} "
                  f"err_share={_f(b['err_sq_share_additive'], 4)}")
        for key in ("signal_restricted_abs_threshold", "signal_restricted_rel_quantile"):
            sub = r[key]
            print(f"    {key:<31} n={sub['n_signal']:>7} rel_l2={_f(sub['rel_l2'])} cos={_f(sub['cosine'])}")
    eb = exp["B2_energy_budget"]
    cons = eb["reported_fingerprint_consistency"]
    print(f"  [B2] 实测指纹 rel²={cons['rel_l2_squared']:.4f} vs 纯能量损失预测 "
          f"tan²θ={cons['pure_energy_loss_prediction_tan_sq']:.4f} -> 正交残留="
          f"{cons['orthogonal_residual']:.4f}（{cons['interpretation']}）")
    print(f"       {'s_dY':>9} {'fp16 rel':>9} {'rel²':>8} {'冲零损失模型':>13} "
          f"{'subnormal信号能量':>16} {'subnormal误差能量':>16}")
    for r in eb["rows"]:
        sub = r["buckets"]["fp16_subnormal"]
        print(f"       {r['dy_scale']:>9.0e} {_f(r['fp16_rel_l2']):>9} "
              f"{r['fp16_rel_l2_squared']:>8.4f} {r['flush_only_model_rel_l2']:>13.4f} "
              f"{sub['signal_energy_share']:>16.4f} {sub['err_energy_share']:>16.4f}")


def _print_cd(exp: dict[str, Any]) -> None:
    """打印 C（loss-scale）与 D（归约模拟）两张表。"""
    print("\n[C] loss-scale 扫描")
    for blk in exp["C_loss_scale"]:
        print(f"  s={blk['dy_scale']:.0e} bf16目标={_f(blk['bf16_target_rel_l2'])} "
              f"单调不增(1e-3)={blk['monotonic_non_increasing_tol_1e-3']} "
              f"饱和k(10%)={blk['saturation_k_within_10pct']} "
              f"饱和k(25%)={blk['saturation_k_within_25pct']} 溢出首k={blk['overflow_first_k']}")
        print(f"    {'k':>3} {'fp16 rel':>9} {'fp16 cos':>9} {'bf16 rel':>9} {'max|dY|':>10} "
              f"{'headroom':>10} {'normal':>7} {'flush':>7}")
        for r in blk["rows"]:
            print(f"    {r['k']:>3} {_f(r['fp16_arm']['rel_l2']):>9} {_f(r['fp16_arm']['cosine']):>9} "
                  f"{_f(r['bf16_arm_rel_l2']):>9} {r['max_abs_dy']:>10.3g} "
                  f"{r['headroom_to_fp16_max']:>10.3g} {r['frac_fp16_normal']:>7.3f} "
                  f"{r['frac_flushed_to_zero']:>7.3f}")
    sim = exp["D_reduction_sim"]
    if not sim:
        return
    print(f"\n[D] 归约上界（模拟，chunk={sim['chunk_size']}；{sim['note']}）")
    for r in sim["rows"]:
        print(f"  s={r['dy_scale']:.0e} fp16accum_rel={_f(r['fp16_accum_vs_ref']['rel_l2'])} "
              f"fp32accum_rel={_f(r['fp32_accum_vs_ref']['rel_l2'])} "
              f"fp16_vs_fp32={_f(r['fp16_vs_fp32_accum']['rel_l2'])}")


# --- CLI ---
def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Krea-2 NF4 fp16 backward 下溢纯 CPU 探针")
    p.add_argument("--out", default=str(here / "synth_underflow.json"))
    p.add_argument("--seed", type=int, default=20261108)
    p.add_argument("--shapes", default="1024,1024,1024", help="M,N,K；多组用 ; 分隔")
    p.add_argument("--scales", default=",".join(f"{v:g}" for v in DEFAULT_SCALES))
    p.add_argument("--attr-scales", default=",".join(f"{v:g}" for v in DEFAULT_ATTR_SCALES))
    p.add_argument("--budget-scales", default=",".join(f"{v:g}" for v in DEFAULT_BUDGET_SCALES))
    p.add_argument("--d-scales", default="1e-2,1e-4")
    p.add_argument("--k-max", type=int, default=24)
    p.add_argument("--scan-lo", type=float, default=-31.0, help="A2 扫描 log2 下界")
    p.add_argument("--scan-hi", type=float, default=-20.0, help="A2 扫描 log2 上界")
    p.add_argument("--scan-steps", type=int, default=44)
    p.add_argument("--skip-scan", action="store_true")
    p.add_argument("--skip-d", action="store_true")
    p.add_argument("--threads", type=int, default=0, help=">0 时固定 torch CPU 线程数")
    p.add_argument("--repeat-check", action="store_true", help="跑两遍并比较结果是否一致")
    return p


def _strip_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    """去掉 date/elapsed 等易变字段，用于两次重复运行的一致性比较。"""
    clone = json.loads(json.dumps(payload))
    for key in ("date", "elapsed_s", "repeat_check"):
        clone.pop(key, None)
    return clone


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    started = time.time()
    payload = run_all(args)
    if args.repeat_check:
        payload["repeat_check"] = _strip_volatile(payload) == _strip_volatile(run_all(args))
    payload["elapsed_s"] = round(time.time() - started, 2)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_report(payload)
    if args.repeat_check:
        print(f"\nrepeat_check 两次结果一致 = {payload['repeat_check']}")
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
