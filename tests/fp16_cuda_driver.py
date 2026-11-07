"""Two-rank driver for the fp16/E3 CUDA rows (M15–M18) — run as a script, never collected.

Modes (``FP16_DRIVER_MODE``):
  * ``manual``     — M15/M16/M18: the process group is created here with manual NCCL
                     (``init_process_group("nccl")`` + explicit master address/port).
  * ``accelerate`` — M17: ``accelerate launch`` already built the process group and every
                     rank wraps its model with the framework's native DDP reducer.

Scenario families
-----------------
``*_healthy``   both ranks clean → commit/apply (M15 ①③, M16 apply 档, M17 ①, M18 healthy 档)
``m15_bad``     exactly one rank *observes* a non-finite loss (finite gradients) → world discard;
                the clean rank may only follow (L2), never claim an originating cause
``m16_no_scaler`` / ``m17_bad`` / ``m18_bad``
                exactly one rank produces **non-finite gradients with a finite loss** (fp16
                overflow of the scaled backward), which is the no-scaler guard path: the world
                drops the update on every rank.

Receipts: ``<FP16_RECEIPT_DIR>/<scenario>.rank<N>.json`` plus one cleanup receipt per rank.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist

MODE = os.environ["FP16_DRIVER_MODE"]
SCENARIOS = [name for name in os.environ["FP16_SCENARIOS"].split(",") if name]
OUT = os.environ["FP16_RECEIPT_DIR"]

#: Micro-batches per accumulation window. Deliberately > 2 so "the extra 1-element
#: reduction happens once per *boundary*, not once per micro-batch" is falsifiable: the
#: observed collective count must stay at 2 while this grows.
MICRO_BATCH = 4
#: Small input magnitudes keep a *healthy* fp16 window clear of accidental overflow.
INPUT_SCALE = 0.05
#: Finite loss, fp16-overflowing gradients: this is the no-scaler guard / M18 bad-window 档.
GRAD_OVERFLOW_FACTOR = 1e9

REDUCTIONS = {"one_element_max": []}


def _reset_reductions() -> None:
    REDUCTIONS["one_element_max"] = []


def _install_reduction_counter() -> None:
    """Count the 1-element MAX reductions (the world-agreement collectives).

    Only ``numel == 1`` + ``ReduceOp.MAX`` calls are counted, so the production manual
    gradient sync, DDP's own bucket reducer and the async buckets (all multi-element) stay
    out of the tally. This is the M16③ counting stub.
    """
    original = dist.all_reduce

    def counting(tensor, op=None, async_op=False, group=None):
        if getattr(tensor, "numel", None) and tensor.numel() == 1 and op == dist.ReduceOp.MAX:
            REDUCTIONS["one_element_max"].append(int(tensor.numel()))
        return original(tensor, op=op, async_op=async_op, group=group)

    dist.all_reduce = counting


def _write_json(name: str, payload: dict) -> None:
    """Write a receipt atomically: a crash can never leave a half-written 0-byte file."""
    path = os.path.join(OUT, name)
    temporary = f"{path}.tmp{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _hash_tensor(tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1).to(torch.float32).cpu().contiguous()
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def _weight_hash(module) -> str:
    return _hash_tensor(torch.cat([p.detach().reshape(-1) for p in module.parameters()]))


def _grad_hash(module) -> str:
    parts = []
    for parameter in module.parameters():
        grad = parameter.grad
        parts.append(
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if grad is None
            else grad.detach().reshape(-1).to(torch.float32).cpu()
        )
    return _hash_tensor(torch.cat(parts))


class Shim:
    """The accelerate surface the accounting helpers read (no DDP wrap, no scaler wrap)."""

    def __init__(self, rank, world, scaler=None):
        self.device = torch.device("cuda", rank)
        self.num_processes = world
        self.process_index = rank
        self.local_process_index = rank
        self.scaler = scaler
        self.optimizer_step_was_skipped = False
        self.mixed_precision = "fp16"
        self.use_distributed = world > 1
        self.distributed_type = "MULTI_GPU"

    def reduce(self, tensor, reduction="mean", scale=1.0):
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        if reduction == "mean":
            reduced.div_(self.num_processes)
        return reduced

    def print(self, *args, **kwargs):  # pragma: no cover - diagnostics only
        print(*args, file=sys.stderr, **kwargs)


def _build_model(device, seed=1234, *, half=False):
    """Two parameters in several buckets.

    ``half=False`` keeps fp32 master weights and runs the forward under ``autocast(fp16)`` —
    that is accelerate's real fp16 recipe, and it is the only shape GradScaler accepts
    (it refuses to unscale fp16 gradients). ``half=True`` mirrors the *no-scaler* fp16 case
    R6 warns about, where an overflow really does leave non-finite fp16 gradients.
    """
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=False),
        torch.nn.Linear(8, 8, bias=False),
    ).to(device)
    if half:
        model = model.half()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    return model, optimizer


def _micro_inputs(device, rank, micro):
    """Rank-different, deterministic inputs: gradients only agree after a real reduction."""
    generator = torch.Generator(device=device).manual_seed(1000 + 7 * rank + micro)
    base = torch.ones(2, 8, device=device) * (1.0 + 0.25 * (rank + 1) + 0.5 * micro)
    noise = 0.01 * torch.randn(2, 8, device=device, generator=generator)
    return (INPUT_SCALE * (base + noise)).to(torch.float16)


def _nonfinite_grads(model) -> bool:
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return True
    return False


def _window_decision(state, buffer, window_accounting, window_coordination):
    """The window-level E3 verdict exactly as ``_run_epoch_steps`` takes it."""
    local_reason = window_coordination.local_window_reason(state, buffer)
    reason = window_coordination.agree_failure_reason(state, local_reason)
    if reason is None:
        window_accounting.commit_window(state, buffer)
        window_coordination.note_window_decision(state, "commit")
        decision = "commit"
    else:
        window_accounting.discard_window(state, buffer, reason, local_reason=local_reason)
        window_coordination.note_window_decision(
            state, "discard", propagated=local_reason is None
        )
        decision = "discard"
    return {"local_reason": local_reason, "reason": reason, "decision": decision}


def _boundary(state, model, optimizer, *, scaler, skip_update, loss_is_finite, window_coordination):
    """The optimizer-boundary transaction, mirroring ``_run_microbatch`` line by line."""
    state.update_skipped = None
    drop_update = bool(skip_update) or (not loss_is_finite)
    agreed_drop = window_coordination.agree_drop_update(state, drop_update)
    if agreed_drop:
        state.update_skipped = True
        optimizer.zero_grad(set_to_none=True)
    else:
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            state.accelerator.optimizer_step_was_skipped = bool(
                scaler.get_scale() < scale_before
            )
        else:
            optimizer.step()
        state.update_skipped = bool(state.accelerator.optimizer_step_was_skipped)
        if state.update_skipped:
            optimizer.zero_grad(set_to_none=True)
    cause = window_coordination.note_attempt(
        state,
        route=window_coordination.ROUTE_WINDOW,
        skip_update=bool(skip_update),
        force_skip=False,
        loss_is_finite=bool(loss_is_finite),
        step_attempted=not agreed_drop,
    )
    return {"agreed_drop": bool(agreed_drop), "cause": cause}


def _receipt(state, *, scenario, rank, extra, window_coordination):
    snapshot = window_coordination.snapshot(state)
    payload = {
        "scenario": scenario,
        "rank": rank,
        "step_attempts": snapshot["step_attempts"],
        "effective": snapshot["updates_effective"],
        "updates_skipped": snapshot["updates_skipped"],
        "updates_skipped_by_scaler": snapshot["updates_skipped_by_scaler"],
        "boundary_causes": snapshot["originating_causes"],
        "propagated_drops": snapshot["propagated_drops"],
        "window_world_decisions": snapshot["window_world_decisions"],
        "windows_committed": snapshot["windows_committed"],
        "windows_discarded": snapshot["windows_discarded"],
        "window_discard_causes": snapshot["window_discard_causes"],
        "window_discard_propagated": snapshot["window_discard_propagated"],
        # M19-R2 ⑥: the whole ``windows.*`` view in one place so the combination gate can
        # compare the two ranks key by key without inventing a second ledger.
        "window_ledger": {
            "committed": snapshot["windows_committed"],
            "discarded": snapshot["windows_discarded"],
            "replayed": snapshot["window_replays"],
            "attempts": snapshot["window_attempts"],
            "consecutive_discards": snapshot["window_consecutive_discards"],
            "discard_causes": snapshot["window_discard_causes"],
            "discard_propagated": snapshot["window_discard_propagated"],
        },
        "reductions": list(REDUCTIONS["one_element_max"]),
        "reduction_calls": len(REDUCTIONS["one_element_max"]),
        "reduction_numels": list(REDUCTIONS["one_element_max"]),
        "micro_batches": MICRO_BATCH,
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(state.accelerator.device),
        "world_size": int(dist.get_world_size()),
        "backend": dist.get_backend(),
        "max_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }
    payload.update(extra)
    _write_json(f"{scenario}.rank{rank}.json", payload)
    return payload


def _state_for(accelerator):
    from library.training import window_accounting, window_coordination

    state = SimpleNamespace(
        accelerator=accelerator,
        update_skipped=None,
        global_step=0,
        loss_recorder=None,
        args=SimpleNamespace(max_nonfinite_window_discards=3, adaptive_max_nonfinite_skips=0),
    )
    window_accounting.ensure_counters(state)
    window_coordination.ensure(state)
    return state


def _poison(name: str, rank: int, micro: int) -> str:
    """Which rank/micro-batch carries the injected fault, and of which kind."""
    last = MICRO_BATCH - 1
    if name == "m15_bad" and rank == 1 and micro == last:
        return "loss"  # ledger sees a non-finite loss; gradients stay finite
    if name in ("m16_no_scaler", "m17_bad", "m18_bad") and rank == 1 and micro == last:
        return "grad"  # finite loss, fp16-overflowing gradients (guard / async bad 档)
    return "none"


def _manual_scenario(name, rank, world):
    """M15 (both modes) / M16 (both 档) — manual NCCL reduction of gradients."""
    from library.training import gradient_sync, window_accounting, window_coordination

    sync_calls = {"n": 0}
    original_sync = gradient_sync.synchronize_optimizer_gradients

    def counting_manual_sync(accelerator, optimizer):
        sync_calls["n"] += 1
        return original_sync(accelerator, optimizer)

    gradient_sync.synchronize_optimizer_gradients = counting_manual_sync
    try:
        use_scaler = name in ("m15_healthy", "m16_fp16_scaler")
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        accelerator = Shim(rank, world, scaler)
        device = accelerator.device
        model, optimizer = _build_model(device, half=not use_scaler)
        state = _state_for(accelerator)

        pre_hash = _weight_hash(model)
        buffer = window_accounting.WindowBuffer()
        _reset_reductions()
        for micro in range(MICRO_BATCH):
            inputs = _micro_inputs(device, rank, micro).float()
            if use_scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(inputs).float().pow(2).mean()
            else:
                loss = model(inputs.half()).float().pow(2).mean()
            fault = _poison(name, rank, micro)
            recorded = float("inf") if fault == "loss" else float(loss.detach())
            backward_loss = loss * GRAD_OVERFLOW_FACTOR if fault == "grad" else loss
            if fault == "loss":
                assert recorded == float("inf"), "the loss-poison 档 must record a non-finite loss"
            else:
                assert are_finite(recorded), "every other 档 keeps the loss ledger finite"
            buffer.add(object(), recorded, epoch=0, step=micro)
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
        buffer.record_attempt()
        # M15/M16 manual path: the repository's own manual mean-reduction over NCCL.
        sync_result = gradient_sync.synchronize_optimizer_gradients(accelerator, optimizer)
        skip_update = scaler is None and _nonfinite_grads(model)
        loss_is_finite = window_accounting.losses_are_finite(buffer.losses)
        grads_after = _grad_hash(model)
        grads_finite = not _nonfinite_grads(model)
        boundary = _boundary(
            state, model, optimizer,
            scaler=scaler, skip_update=skip_update,
            loss_is_finite=loss_is_finite, window_coordination=window_coordination,
        )
        decision = _window_decision(state, buffer, window_accounting, window_coordination)
        post_hash = _weight_hash(model)
        extra = {
            "pre_weight_hash": pre_hash,
            "post_weight_hash": post_hash,
            "weight_changed": pre_hash != post_hash,
            "grads_hash": grads_after,
            "grads_finite": grads_finite,
            "manual_sync_calls": sync_calls["n"],
            "manual_sync_result": [sync_result.parameter_count, sync_result.reduced_parameter_count],
            "scaler": scaler is not None,
            "agreed_drop": boundary["agreed_drop"],
            "boundary_cause": boundary["cause"],
            "update_skipped": bool(state.update_skipped),
            "windows": 1,
            "fault": _fault_kind(name),
        }
        extra.update(decision)
        return _receipt(state, scenario=name, rank=rank, extra=extra, window_coordination=window_coordination)
    finally:
        gradient_sync.synchronize_optimizer_gradients = original_sync


def _async_scenario(name, rank, world):
    """M18 — real AsyncGradientSynchronizer.begin_step()/finish_step().

    Wiring mirrors ``loop.py`` exactly: ``begin_step()`` is issued for **every micro-batch**
    (``_run_microbatch``), ``finish_step()`` once at the optimizer boundary. That is what
    makes the pairing auditable instead of inferred from the finish side alone.
    """
    from library.training import gradient_sync, window_accounting, window_coordination

    micro_batches = 1 if name == "m18_paired" else MICRO_BATCH
    windows = 1 if name == "m18_bad" else 2
    accelerator = Shim(rank, world, scaler=None)
    device = accelerator.device
    model, optimizer = _build_model(device, half=True)
    synchronizer = gradient_sync.AsyncGradientSynchronizer(accelerator, optimizer, bucket_bytes=64)
    state = _state_for(accelerator)

    after_begin, after_finish, bucket_results, decisions, boundaries = [], [], [], [], []
    pre_hash = _weight_hash(model)
    _reset_reductions()
    for window_index in range(windows):
        buffer = window_accounting.WindowBuffer()
        for micro in range(micro_batches):
            synchronizer.begin_step()
            if micro == 0:
                # Read the begin side *before* any finish: proves both sides are counted
                # on their own (M18①: the baseline only ever counted finish).
                after_begin.append(dict(synchronizer.lifecycle))
            inputs = _micro_inputs(device, rank, micro + window_index).half()
            loss = model(inputs).float().pow(2).mean()
            fault = _poison(name, rank, micro)
            backward_loss = loss * GRAD_OVERFLOW_FACTOR if fault == "grad" else loss
            buffer.add(object(), float(loss.detach()), epoch=0, step=micro)
            backward_loss.backward()
        result = synchronizer.finish_step()
        bucket_results.append([result.parameter_count, result.reduced_parameter_count, result.bucket_count])
        after_finish.append(dict(synchronizer.lifecycle))
        grads = _grad_hash(model)
        skip_update = _nonfinite_grads(model)
        loss_is_finite = window_accounting.losses_are_finite(buffer.losses)
        boundary = _boundary(
            state, model, optimizer,
            scaler=None, skip_update=skip_update,
            loss_is_finite=loss_is_finite, window_coordination=window_coordination,
        )
        decision = _window_decision(state, buffer, window_accounting, window_coordination)
        decision["grads_hash"] = grads
        decision["window_index"] = window_index
        decisions.append(decision)
        boundaries.append(boundary)
    lifecycle_final = dict(synchronizer.lifecycle)
    post_hash = _weight_hash(model)
    finite_weights = all(torch.isfinite(p).all().item() for p in model.parameters())
    synchronizer.close()
    extra = {
        "pre_weight_hash": pre_hash,
        "post_weight_hash": post_hash,
        "weight_changed": pre_hash != post_hash,
        "weights_finite": bool(finite_weights),
        "lifecycle_after_begin": after_begin,
        "lifecycle_after_finish": after_finish,
        "lifecycle_final": lifecycle_final,
        "lifecycle_after_close": dict(synchronizer.lifecycle),
        "bucket_result": bucket_results,
        "windows": windows,
        "micro_batches": micro_batches,
        "decisions": decisions,
        "grads_hash_per_window": [d["grads_hash"] for d in decisions],
        "reason": decisions[-1]["reason"],
        "decision": decisions[-1]["decision"],
        "local_reason": decisions[-1]["local_reason"],
        "agreed_drop": boundaries[-1]["agreed_drop"],
        "boundary_cause": boundaries[-1]["cause"],
        "update_skipped": bool(state.update_skipped),
        "fault": _fault_kind(name),
        "cleanup": {
            "synchronizer_closed": True,
            "pairing_balanced": lifecycle_final["begin_calls"] == lifecycle_final["finish_calls"],
            "pending_matches_gap": lifecycle_final["steps_pending"]
            == lifecycle_final["begin_calls"] - lifecycle_final["finish_calls"],
            "expected_begins": windows * micro_batches,
            "expected_finishes": windows,
        },
    }
    return _receipt(state, scenario=name, rank=rank, extra=extra, window_coordination=window_coordination)


def _ddp_scenario(name, rank, world):
    """M17 — ``accelerate`` native DDP reducer (no manual reduction anywhere)."""
    from accelerate import Accelerator

    from library.training import gradient_sync, window_accounting, window_coordination

    manual_sync = {"calls": 0}
    original_sync = gradient_sync.synchronize_optimizer_gradients

    def forbidden_manual_sync(accelerator, optimizer):
        # M17 forbids replacing the native reducer with a manual reduction: if this ever
        # runs, the row is reduced to M15 and must fail.
        manual_sync["calls"] += 1
        return original_sync(accelerator, optimizer)

    gradient_sync.synchronize_optimizer_gradients = forbidden_manual_sync
    try:
        accelerator = Accelerator(
            mixed_precision="fp16", gradient_accumulation_steps=MICRO_BATCH, cpu=False
        )
        device = accelerator.device
        model, optimizer = _build_model(device, seed=1234, half=False)
        model, optimizer = accelerator.prepare(model, optimizer)
        native_reducer = {
            "is_ddp": isinstance(model, torch.nn.parallel.DistributedDataParallel),
            "has_native_reducer": hasattr(model, "reducer"),
            "backend": dist.get_backend(),
            "num_processes": int(accelerator.num_processes),
        }
        if not (native_reducer["is_ddp"] and native_reducer["has_native_reducer"]):
            raise RuntimeError(f"accelerate did not build a native DDP reducer: {native_reducer}")
        state = _state_for(accelerator)
        unwrapped = accelerator.unwrap_model(model)

        pre_hash = _weight_hash(unwrapped)
        buffer = window_accounting.WindowBuffer()
        _reset_reductions()
        accelerator.gradient_state._set_sync_gradients(False)
        for micro in range(MICRO_BATCH):
            inputs = _micro_inputs(device, rank, micro)
            inputs = inputs.float()
            with accelerator.accumulate(model):
                loss = model(inputs).float().pow(2).mean()
                buffer.add(object(), float(loss.detach()), epoch=0, step=micro)
                # Native reducer: no manual all-reduce, no stub. The final micro-batch of
                # the accumulation cycle reduces through DDP's own buckets.
                accelerator.backward(loss)
        if _poison(name, rank, MICRO_BATCH - 1) == "grad":
            # M17 asserts the *native-reducer world consistency*, not an overflow attribution:
            # one rank ends the window with a non-finite gradient (fault injection after DDP
            # already reduced, so the other rank stays clean — exactly the asymmetric case the
            # world agreement has to survive).
            first = next(unwrapped.parameters())
            first.grad.view(-1)[0] = float("nan")
        buffer.record_attempt()
        skip_update = _nonfinite_grads(unwrapped)
        loss_is_finite = window_accounting.losses_are_finite(buffer.losses)
        grads = _grad_hash(unwrapped)
        grads_finite = not _nonfinite_grads(unwrapped)
        # The scaler belongs to accelerate here: ``optimizer.step()`` is the
        # AcceleratedOptimizer, which unscales/steps/skips internally and reports through
        # ``accelerator.optimizer_step_was_skipped``. Wrapping it in the raw scaler again
        # would double-scale, so ``_boundary`` is driven without a scaler handle.
        boundary = _boundary(
            state, unwrapped, optimizer,
            scaler=None, skip_update=skip_update,
            loss_is_finite=loss_is_finite, window_coordination=window_coordination,
        )
        decision = _window_decision(state, buffer, window_accounting, window_coordination)
        post_hash = _weight_hash(unwrapped)
        extra = {
            "native_reducer": native_reducer,
            "manual_sync_calls": manual_sync["calls"],
            "pre_weight_hash": pre_hash,
            "post_weight_hash": post_hash,
            "weight_changed": pre_hash != post_hash,
            "grads_hash": grads,
            "grads_finite": grads_finite,
            "agreed_drop": boundary["agreed_drop"],
            "boundary_cause": boundary["cause"],
            "update_skipped": bool(state.update_skipped),
            "windows": 1,
            "fault": _fault_kind(name),
            "scaler": accelerator.scaler is not None,
        }
        extra.update(decision)
        return _receipt(state, scenario=name, rank=rank, extra=extra, window_coordination=window_coordination)
    finally:
        gradient_sync.synchronize_optimizer_gradients = original_sync


def _fault_kind(name: str) -> str:
    return {
        "m15_bad": "loss",
        "m16_no_scaler": "grad",
        "m17_bad": "grad",
        "m18_bad": "grad",
    }.get(name, "none")


def are_finite(value: float) -> bool:
    import math

    return math.isfinite(value)


_SCENARIO_FNS = {
    "m15_healthy": _manual_scenario,
    "m15_bad": _manual_scenario,
    "m16_fp16_scaler": _manual_scenario,
    "m16_no_scaler": _manual_scenario,
    "m18_healthy": _async_scenario,
    "m18_paired": _async_scenario,
    "m18_bad": _async_scenario,
    "m17_healthy": _ddp_scenario,
    "m17_bad": _ddp_scenario,
}


def main() -> int:
    import library.training.gradient_sync  # noqa: F401  (warm the heavy import first)

    if MODE == "manual":
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank)
        )
    else:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    _install_reduction_counter()

    failures = []
    for scenario in SCENARIOS:
        try:
            torch.cuda.synchronize()
            started = time.time()
            payload = _SCENARIO_FNS[scenario](scenario, rank, world)
            torch.cuda.synchronize()
            payload["seconds"] = round(time.time() - started, 3)
            _write_json(f"{scenario}.rank{rank}.json", payload)
            print(f"[{scenario} rank{rank}] ok {payload['seconds']}s", file=sys.stderr)
        except Exception as exc:  # keep going: the parent reports every scenario
            failures.append((scenario, f"{type(exc).__name__}: {exc}"))
            print(f"[{scenario} rank{rank}] FAILED {traceback.format_exc()}", file=sys.stderr)

    cleanup = {
        "rank": rank,
        "pid": os.getpid(),
        "device": str(torch.cuda.current_device()),
        "scenarios": SCENARIOS,
        "failures": [list(item) for item in failures],
        "process_group_initialized": bool(dist.is_initialized()),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if dist.is_initialized() and not failures:
        dist.barrier()
    cleanup["pending_after_barrier"] = 0
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:  # pragma: no cover - a failed rank may desync the group
            cleanup["destroy_error"] = f"{type(exc).__name__}: {exc}"
    cleanup["destroyed"] = not dist.is_initialized()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    cleanup["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
    cleanup["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
    _write_json(f"cleanup.rank{rank}.json", cleanup)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
