"""fp16 保护与 NaN 后处理的静态数学探针（P1–P9，CPU-only，不需要 GPU）。

真 torch.amp.GradScaler("cpu")，九个探针分别对应
docs/findings/fp16_nan_static_equivalence_20261021.md 里的 P1–P9 编号。
本文件是静态验收报告的数字来源，可离线重跑核对。

运行：
  CUDA_VISIBLE_DEVICES='' .venv/bin/python bench/fp16_precision/run_probe.py

说明：全部结论仅对 CPU 可判定面成立；真实 CUDA autocast、unscale kernel 数值、
跨 rank 归约与长程收敛仍需 GPU 热测（见报告「待热测清单」H1–H8）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULT: dict = {}


def rec(key, **fields):
    RESULT[key] = fields
    print(f"[{key}] " + json.dumps(fields, ensure_ascii=False, default=str))


def p1_scaler_state_machine():
    s = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.0)
    seq = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        s.scale((p * 0.0 + 1.0).sum()).backward()
        before = s.get_scale()
        s.step(opt)
        s.update()
        seq.append([before, s.get_scale()])
    opt.zero_grad(set_to_none=True)
    s.scale((p * float("inf")).sum()).backward()
    before_overflow = s.get_scale()
    s.step(opt)
    s.update()
    after_overflow = s.get_scale()
    sd = s.state_dict()
    s2 = torch.amp.GradScaler("cpu", init_scale=1.0, growth_interval=2)
    s2.load_state_dict(sd)
    rec(
        "P1_E4_scaler_state_machine",
        growth_interval=2,
        clean_step_scale_pairs=seq,
        overflow_pair=[before_overflow, after_overflow],
        backoff_is_halving=(after_overflow * 2 == before_overflow),
        state_dict_scale_roundtrip=(s2.get_scale() == s.get_scale()),
        state_dict_payload_keys=sorted(sd.keys()),
    )


def p2_skip_semantics():
    from accelerate import Accelerator, DistributedType
    from accelerate.optimizer import AcceleratedOptimizer
    from library.runtime.accelerator import patch_accelerator_for_fp16_training

    Accelerator(device_placement=False)

    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    opt = torch.optim.SGD([p], lr=0.5)
    calls = {"raw_optimizer_step": 0}
    orig_step = opt.step

    def counting_step(*a, **k):
        calls["raw_optimizer_step"] += 1
        return orig_step(*a, **k)

    opt.step = counting_step  # 必须在 AcceleratedOptimizer 捕获原始 step 之前打桩
    scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    acc = AcceleratedOptimizer(opt, device_placement=False, scaler=scaler)
    sync = bool(acc.gradient_state.sync_gradients)

    opt.zero_grad(set_to_none=True)
    scaler.scale(p.sum() * 0.001).backward()
    w0 = p.detach().clone()
    clean_calls_before = calls["raw_optimizer_step"]
    acc.step()
    clean_inner_step_invoked = calls["raw_optimizer_step"] > clean_calls_before
    clean_applied = not acc.step_was_skipped
    clean_delta = (p.detach() - w0)

    opt.zero_grad(set_to_none=True)
    overflow_loss = p.sum() * 1e38
    loss_finite = bool(torch.isfinite(overflow_loss).all())
    scaler.scale(overflow_loss).backward()
    grad_finite = bool(torch.isfinite(p.grad).all())
    w1 = p.detach().clone()
    scale_before = scaler.get_scale()
    calls_before = calls["raw_optimizer_step"]
    acc.step()  # 调用点被执行（不抛异常即 executed=True），更新由 scaler 内部决定
    inner_step_invoked = calls["raw_optimizer_step"] > calls_before
    skipped = bool(acc.step_was_skipped)
    unchanged = bool(torch.equal(p.detach(), w1))
    scale_after = scaler.get_scale()

    # fp16 参数：torch 原生拒绝；A 的 full_fp16 补丁后可用
    p16 = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float16))
    opt16 = torch.optim.SGD([p16], lr=0.5)
    s16 = torch.amp.GradScaler("cpu", init_scale=1024.0)
    a16 = AcceleratedOptimizer(opt16, device_placement=False, scaler=s16)
    s16.scale(p16.sum() * 0.001).backward()
    native_error = None
    try:
        a16.step()
    except Exception as exc:
        native_error = f"{type(exc).__name__}: {exc}"
    opt16.zero_grad(set_to_none=True)
    patch_accelerator_for_fp16_training(
        type("Proxy", (), {"distributed_type": DistributedType.NO, "scaler": s16})()
    )
    s16.scale(p16.sum() * 0.001).backward()
    w16 = p16.detach().clone()
    patched_error = None
    try:
        a16.step()
    except Exception as exc:
        patched_error = f"{type(exc).__name__}: {exc}"

    rec(
        "P2_E2_skip_semantics",
        sync_gradients=sync,
        clean_step_applied=clean_applied,
        clean_weight_delta=[float(v) for v in clean_delta],
        overflow_loss_finite_before_scaling=loss_finite,
        overflow_grad_finite_after_backward=grad_finite,
        call_site_executed=True,
        inner_optimizer_step_invoked_on_clean=clean_inner_step_invoked,
        inner_optimizer_step_invoked_on_overflow=inner_step_invoked,
        optimizer_update_applied=not skipped,
        weight_bitwise_unchanged=unchanged,
        scale_before=scale_before,
        scale_after=scale_after,
        step_was_skipped_field=skipped,
        fp16_params_native_error=native_error,
        fp16_params_after_a_patch_error=patched_error,
        fp16_params_after_a_patch_weight_moved=not torch.equal(p16.detach(), w16),
    )


def p3_owner_misclassified_as_failure():
    from accelerate import DistributedType
    from library.runtime.accelerator import patch_accelerator_for_fp16_training
    from library.training.finite_checks import check_loss_finite, check_trainable_grads_finite

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float16))

    model = Tiny()
    scaler = torch.amp.GradScaler("cpu")
    patch_accelerator_for_fp16_training(
        type("Proxy", (), {"distributed_type": DistributedType.NO, "scaler": scaler})()
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = model.w.sum() * 2.0
    check_loss_finite(loss, mixed_precision="fp16")
    scaler.scale(loss).backward()
    raised = None
    try:
        check_trainable_grads_finite(model)
    except Exception as exc:
        raised = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    w = model.w.detach().clone()
    pre = scaler.get_scale()
    scaler.step(opt)
    scaler.update()

    rec(
        "P3_E5_ownership_counterexample",
        loss_finite=bool(torch.isfinite(loss.detach()).all()),
        loss_side_fail_fast_passed=True,
        scaled_grad_finite=bool(torch.isfinite(model.w.grad).all()),
        scale_at_backward=pre,
        a_side_grad_fail_fast_raised=raised,
        real_scaler_would_skip=(scaler.get_scale() < pre and torch.equal(model.w.detach(), w)),
    )


def p4_clip_math():
    scale = 65536.0
    max_norm = 1.0
    true_grad = torch.tensor([10.0, 0.0])
    scaled_grad = true_grad * scale

    def norm(t):
        return float(torch.linalg.vector_norm(t, 2))

    correct = true_grad * min(max_norm / norm(true_grad), 1.0)
    wrong = (scaled_grad * min(max_norm / norm(scaled_grad), 1.0)) / scale
    alt = (scaled_grad * min((max_norm * scale) / norm(scaled_grad), 1.0)) / scale

    rec(
        "P4_clip_math",
        scale=scale,
        true_norm=norm(true_grad),
        correct_unscale_then_clip_norm=norm(correct),
        wrong_clip_scaled_then_unscale_norm=norm(wrong),
        wrong_over_correct_ratio=norm(wrong) / norm(correct),
        threshold_scaled_equivalent_norm=norm(alt),
    )


def p5_bitwise_no_scaler():
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer

    Accelerator(device_placement=False)
    x = torch.randn(3, 4)

    def run(with_acc):
        torch.manual_seed(1234)
        m = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        acc = AcceleratedOptimizer(opt, device_placement=False, scaler=None) if with_acc else None
        loss = (m(x) ** 2).mean()
        before = id(loss)
        loss.backward()
        same_obj = id(loss) == before
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        (acc.step() if acc else opt.step())
        return (
            [q.grad.detach().clone() for q in m.parameters()],
            [q.detach().clone() for q in m.parameters()],
            same_obj,
        )

    g0, p0, _ = run(False)
    g1, p1, same = run(True)
    rec(
        "P5_E1_bitwise_no_scaler",
        loss_object_identity_preserved=same,
        grads_bitwise_equal=all(torch.equal(a, b) for a, b in zip(g0, g1)),
        params_bitwise_equal=all(torch.equal(a, b) for a, b in zip(p0, p1)),
    )


def p6_static_assertions():
    loop_src = (REPO / "library/training/loop.py").read_text(encoding="utf-8")
    fin_src = (REPO / "library/training/finite_checks.py").read_text(encoding="utf-8")
    boot_src = (REPO / "library/training/bootstrap.py").read_text(encoding="utf-8")
    acc_src = (REPO / "library/runtime/accelerator.py").read_text(encoding="utf-8")
    lines = loop_src.splitlines()

    def line_of(token, text=None):
        src = (text if text is not None else loop_src).splitlines()
        for i, line in enumerate(src, 1):
            if token in line:
                return i
        return -1

    l_loss = line_of("check_loss_finite(")
    l_grads = line_of("check_trainable_grads_finite(")
    l_clip = line_of("accelerator.clip_grad_norm_(")
    l_step = line_of("state.optimizer.step()")
    l_sched = line_of("_step_lr_scheduler(state, loss)")
    l_gstep = line_of("state.global_step += 1")

    rec(
        "P6_static_loop",
        step_was_skipped_mentions_in_loop=loop_src.count("step_was_skipped"),
        isfinite_mentions_in_loop=loop_src.count("isfinite"),
        check_loss_finite_line=l_loss,
        check_trainable_grads_finite_line=l_grads,
        clip_grad_norm_line=l_clip,
        optimizer_step_line=l_step,
        scheduler_step_line=l_sched,
        global_step_increment_line=l_gstep,
        order_loss_grads_clip_step_sched=(0 < l_loss < l_grads < l_clip < l_step < l_sched),
        global_step_increment_guarded_by_sync=("sync_gradients" in lines[l_gstep - 3]),
        finite_checks_knows_scaler=("scaler" in fin_src),
        prepare_optimizer_line=line_of("optimizer, train_dataloader", boot_src),
        patch_call_line=line_of("patch_accelerator_for_fp16_training(accelerator)", boot_src),
        patch_def_line=line_of("def patch_accelerator_for_fp16_training", acc_src),
    )


def p7_grep_counts():
    patterns = [
        "accumulation_window",
        "window_invalid",
        "reset_scaler_step_state",
        "optimizer_step_executed",
        "optimizer_update_applied",
        "scaler_overflow_ignored",
        "scaled_gradient_overflow_is_scaler_owned",
    ]
    counts = {p: 0 for p in patterns}
    for path in (REPO / "library").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for p in patterns:
            counts[p] += text.count(p)
    rec("P7_grep_counts_library", **{k: v for k, v in counts.items()})


def p8_unscale_clip_step_order():
    """A 的实际顺序：clip_grad_norm_ 先在 scale.unscale_ 后裁剪，再 step。"""
    p = torch.nn.Parameter(torch.tensor([10.0, 0.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
    loss = p.sum()  # 真梯度 1
    scaler.scale(loss).backward()
    scaled_norm = float(p.grad.norm())
    scaler.unscale_(opt)  # accelerate.clip_grad_norm_ 内部先做 này
    unscaled_norm = float(p.grad.norm())
    torch.nn.utils.clip_grad_norm_([p], 0.5)
    clipped_norm = float(p.grad.norm())
    w = p.detach().clone()
    scaler.step(opt)
    scaler.update()
    delta = (p.detach() - w).tolist()

    # 反例：若先 clip 带 scale 的梯度再除 scale
    p2 = torch.nn.Parameter(torch.tensor([10.0, 0.0]))
    opt2 = torch.optim.SGD([p2], lr=0.1)
    s2 = torch.amp.GradScaler("cpu", init_scale=65536.0)
    s2.scale(p2.sum()).backward()
    torch.nn.utils.clip_grad_norm_([p2], 0.5)
    s2.unscale_(opt2)
    w2 = p2.detach().clone()
    s2.step(opt2)
    s2.update()
    rec(
        "P8_unscale_clip_step",
        scaled_grad_norm=scaled_norm,
        unscaled_grad_norm=unscaled_norm,
        clipped_grad_norm=clipped_norm,
        weight_delta_correct=delta,
        weight_delta_clip_before_unscale=(p2.detach() - w2).tolist(),
    )


def p9_fp16_unscale_dynamic_range():
    """A 的 full_fp16 补丁让 scaler 接受 fp16 参数，但 unscale 仍在 fp16 里做。"""
    from accelerate import DistributedType
    from library.runtime.accelerator import patch_accelerator_for_fp16_training

    true_vals = [1e-8, 1e-7, 5.96e-8, 1e-3, 0.5, 1.5]
    scale = 65536.0
    param = torch.nn.Parameter(torch.zeros(len(true_vals), dtype=torch.float16))
    opt = torch.optim.SGD([param], lr=1.0)
    scaler = torch.amp.GradScaler("cpu", init_scale=scale)
    scaler.scale(torch.zeros((), dtype=torch.float16))  # 触发懒初始化，生成 _scale
    patch_accelerator_for_fp16_training(
        type("Proxy", (), {"distributed_type": DistributedType.NO, "scaler": scaler})()
    )
    param.grad = (torch.tensor(true_vals, dtype=torch.float16) * scaler.get_scale())
    scaled_repr = param.grad.tolist()
    scaler.unscale_(opt)
    observed = param.grad.tolist()

    rows = []
    for truth, scaled, seen in zip(true_vals, scaled_repr, observed):
        rows.append(
            {
                "true_grad": truth,
                "scaled_fp16": scaled,
                "after_unscale_fp16": seen,
                "flushed_to_zero": seen == 0.0,
                "rel_error": (None if truth == 0 else abs(seen - truth) / truth),
            }
        )
    rec(
        "P9_fp16_unscale_window",
        scale=scale,
        fp16_min_subnormal=2.0 ** -24,
        fp16_max=65504.0,
        analytic_true_grad_window=[2.0 ** -24, 65504.0 / scale],
        rows=rows,
    )


def main():
    print(f"torch={torch.__version__} cuda_initialized={torch.cuda.is_initialized()}")
    for fn in (
        p1_scaler_state_machine,
        p2_skip_semantics,
        p3_owner_misclassified_as_failure,
        p4_clip_math,
        p5_bitwise_no_scaler,
        p6_static_assertions,
        p7_grep_counts,
        p8_unscale_clip_step_order,
        p9_fp16_unscale_dynamic_range,
    ):
        fn()
    print("ALL_PROBES_DONE", len(RESULT))


if __name__ == "__main__":
    main()
