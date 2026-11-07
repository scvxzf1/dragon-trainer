"""M13–M18 CUDA contract: real devices, real NCCL, real DDP reducer, async lifecycle.

Gate (``contract/R3-IMPLEMENTATION-NOTES.md`` §5.1/§5.5): these cases are **not**
optional-skippable. Without the opt-in they are skipped with a visible reason so a CPU-only
run cannot silently look green; with the opt-in and a missing device / wrong UUID / missing
NCCL they **fail** instead of skipping. Run with::

    RUN_FP16_TRANSACTION_CUDA=1 \
    CUDA_VISIBLE_DEVICES=GPU-b4b014b2-953b-28e8-b8c1-425c4a401165,GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061 \
    <venv>/bin/python -m pytest tests/test_fp16_transaction_cuda.py -q -p no:cacheprovider

Rows (execution order per §C5, cheapest first):
  M13  single GPU fp16 + real scaler, healthy window → commits, effective == step_attempts,
       loss ledger finite, no R6 warning while a scaler exists
  M14  single GPU fp16 + real scaler, real *gradient* overflow (finite loss) → scaler skips,
       window fails by ``update_skipped``, L1 ``scaler_overflow`` and ``discard_causes`` stay apart
  M15  2 ranks, manual MAX all-reduce: identical verdict, identical bitwise weight hash,
       comparable discard causes
  M16  2 ranks, manual NCCL: apply/drop agree on both ranks (``fp16+scaler`` apply 档 and
       no-scaler drop 档), identical weight hash, **one** extra 1-element reduction per boundary
  M17  2 ranks, ``accelerate`` native DDP reducer (no manual reduction, no stub): no hang,
       identical weight hash, world-consistent verdict/apply-drop, comparable discard causes
  M18  2 ranks, real ``AsyncGradientSynchronizer.begin_step()/finish_step()``: healthy 档 **and**
       single-rank bad-window 档, both sides counted with auditable pairing, no divergence, and a
       process cleanup receipt

Driver discipline (all rows): every multi-process drive runs with a hard wall-clock timeout and
``PYTHONFAULTHANDLER=1``; on timeout the process group gets ``SIGABRT`` so the child dumps all
thread stacks, the dump is archived next to the receipts and the row **fails** (124/137 is never
retried into a pass). Any row that does not really execute must be reported as "未验" — never as
passed.

Evidence / provenance (one landing place, no manual harvesting)
--------------------------------------------------------------
Every row writes receipts into a **single resolved root**, so the formal t3-2 command already
decides where the evidence lands — nothing has to be copied out by hand afterwards:

1. ``EVIDENCE_DIR`` (the t3-2 variable) → ``<EVIDENCE_DIR>/cuda-gate-<run_id>/``
2. otherwise ``FP16_TRANSACTION_RECEIPT_DIR`` → used verbatim (ad-hoc / self-check runs)
3. otherwise the pytest ``tmp_path`` → last resort so a row can always produce a receipt

Each run gets one ``run_id`` (``<UTC-ish local stamp>-<pid>``, overridable with
``FP16_TRANSACTION_RUN_ID``). ``run.json`` in that root records the start, the resolved roots and
the row receipts; on interpreter exit it is rewritten with the end time, the per-launch return
codes and the receipt inventory. When ``BUDGET_DIR`` (or ``FP16_TRANSACTION_BUDGET_DIR``) is set,
the same record is also written as ``<BUDGET_DIR>/cuda-gate-<run_id>.run.json``, which makes the
shared GPU ledger and the receipt root the same authoritative pair instead of two ad-hoc
notebooks. Single-card rows (M13/M14) write ``m13.rank0.json`` / ``m14.rank0.json`` exactly like
the two-rank rows, and their assertions are evaluated **from those receipts** (the in-process
assertions stay as well, so a receipt that disagrees with memory fails the row).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

RUN_CUDA = os.environ.get("RUN_FP16_TRANSACTION_CUDA") == "1"

#: Per driver-launch hard timeout (the whole gate command has its own ``timeout 175s``).
#: ``library.training`` pulls transformers in, which costs ~14 s per rank, so the two-rank
#: launches need real headroom; a timeout still fails the row instead of skipping it.
DRIVER_TIMEOUT = int(os.environ.get("FP16_TRANSACTION_DRIVER_TIMEOUT", "45"))
#: Scenarios driven by the (single) manual-NCCL launch and the (single) accelerate launch.
MANUAL_SCENARIOS = (
    "m15_healthy",
    "m15_bad",
    "m16_fp16_scaler",
    "m16_no_scaler",
    "m18_paired",
    "m18_healthy",
    "m18_bad",
)
ACCELERATE_SCENARIOS = ("m17_healthy", "m17_bad")
#: One launch per family keeps the wall clock inside the 175 s gate command.
_LAUNCH_CACHE: dict[str, Path] = {}

#: Contract devices (§5.2/§5.3): GPU0 for every single-card row, GPU0+GPU1 for two ranks.
GPU0_UUID = "GPU-b4b014b2-953b-28e8-b8c1-425c4a401165"
GPU1_UUID = "GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061"

if not RUN_CUDA:
    pytestmark = pytest.mark.skip(
        reason="CUDA contract rows are opt-in: set RUN_FP16_TRANSACTION_CUDA=1 "
               "(M13–M18 are unverified otherwise, which fails the CUDA gate)"
    )


# ---------------------------------------------------------------------------
# Gate helpers — every one of these fails, never skips (§5.5 ①②)
# ---------------------------------------------------------------------------
def _visible_uuids() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [f"GPU-{torch.cuda.get_device_properties(i).uuid}" for i in range(torch.cuda.device_count())]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.fail(
            "RUN_FP16_TRANSACTION_CUDA=1 but torch.cuda.is_available() is False: "
            "every row stays unverified, the CUDA gate does not pass"
        )


def _require_gpus(count: int) -> None:
    _require_cuda()
    have = torch.cuda.device_count()
    if have < count:
        pytest.fail(
            f"RUN_FP16_TRANSACTION_CUDA=1 requires {count} visible CUDA device(s), found {have}: "
            "the row stays unverified, the gate does not pass"
        )


def _require_uuids(expected: list[str]) -> None:
    """§5.5②: a visible UUID that differs from the contract device must fail loudly."""
    _require_cuda()
    actual = _visible_uuids()
    if actual[: len(expected)] != expected:
        pytest.fail(
            "visible GPU UUIDs do not match the contract devices: "
            f"expected prefix {expected}, actual {actual}"
        )


def _require_nccl() -> None:
    """§5.5②: NCCL is mandatory for every two-rank row — never degrade to gloo."""
    import torch.distributed as dist

    if not dist.is_available():
        pytest.fail("NCCL unavailable: torch.distributed is not available — the gate does not pass")
    if not dist.is_nccl_available():
        pytest.fail("NCCL unavailable: this build has no NCCL backend — the gate does not pass")


def _weight_hash(module) -> str:
    """Bitwise hash of the parameters (test-side hashing is allowed, §5.5④)."""
    flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()]).to(torch.float32)
    return hashlib.sha256(flat.cpu().contiguous().numpy().tobytes()).hexdigest()


def _grad_hash(module) -> str:
    parts = []
    for parameter in module.parameters():
        grad = parameter.grad
        parts.append(
            torch.zeros(parameter.numel()) if grad is None else grad.detach().reshape(-1).to(torch.float32).cpu()
        )
    return hashlib.sha256(torch.cat(parts).contiguous().numpy().tobytes()).hexdigest()


#: One identity per pytest session — echoed into every receipt and the per-run record.
RUN_ID = os.environ.get("FP16_TRANSACTION_RUN_ID") or (
    time.strftime("%Y%m%dT%H%M%S", time.localtime()) + f"-pid{os.getpid()}"
)
#: Sub-directory created under ``EVIDENCE_DIR`` (one per run, never reused).
EVIDENCE_SUBDIR = f"cuda-gate-{RUN_ID}"

_RUN_STATE: dict[str, object] = {"receipt_dir": None, "budget_dir": None, "started_at": None}


def _budget_dir() -> Path | None:
    """The shared GPU ledger directory, when the caller configured one."""
    raw = os.environ.get("BUDGET_DIR") or os.environ.get("FP16_TRANSACTION_BUDGET_DIR")
    return Path(raw) if raw else None


def _receipt_dir(tmp_path: Path) -> Path:
    """Resolve the one receipt root for this run (see the module docstring).

    ``EVIDENCE_DIR`` (t3-2) → ``<EVIDENCE_DIR>/cuda-gate-<run_id>/``; else
    ``FP16_TRANSACTION_RECEIPT_DIR`` verbatim; else the pytest ``tmp_path``. The first call wins
    for the whole session, so M13/M14 receipts and the two-rank driver receipts always agree.
    """
    cached = _RUN_STATE.get("receipt_dir")
    if cached is not None:
        return cached  # type: ignore[return-value]
    evidence = os.environ.get("EVIDENCE_DIR")
    override = os.environ.get("FP16_TRANSACTION_RECEIPT_DIR")
    if evidence:
        target = Path(evidence) / EVIDENCE_SUBDIR
    elif override:
        target = Path(override)
    else:
        target = Path(tmp_path)
    target.mkdir(parents=True, exist_ok=True)
    _RUN_STATE["receipt_dir"] = target
    _RUN_STATE["started_at"] = time.time()
    _write_run_record()
    return target


def _run_record() -> dict:
    return {
        "schema": "fp16-transaction-cuda-run/1",
        "run_id": RUN_ID,
        "started_at": _RUN_STATE.get("started_at"),
        "started_at_text": (
            None
            if _RUN_STATE.get("started_at") is None
            else time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(float(_RUN_STATE["started_at"])))
        ),
        "receipt_dir": str(_RUN_STATE.get("receipt_dir")),
        "budget_dir": None if _budget_dir() is None else str(_budget_dir()),
        "evidence_dir": os.environ.get("EVIDENCE_DIR"),
        "receipt_dir_override": os.environ.get("FP16_TRANSACTION_RECEIPT_DIR"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "rows": ["m13", "m14", "m15", "m16", "m17", "m18"],
    }


def _write_run_record(**extra: object) -> None:
    """Persist ``run.json`` (and the ledger copy when ``BUDGET_DIR`` is set)."""
    record = _run_record()
    record.update(extra)
    receipt_dir = _RUN_STATE.get("receipt_dir")
    if receipt_dir is None:
        return
    try:
        _write_json(Path(str(receipt_dir)) / "run.json", record)
        budget = _budget_dir()
        if budget is not None:
            budget.mkdir(parents=True, exist_ok=True)
            _write_json(budget / f"cuda-gate-{RUN_ID}.run.json", record)
    except OSError:  # pragma: no cover - evidence must never break the row
        pass


def _finalize_run_record() -> None:
    """atexit hook: close the per-run record with the end time and what actually landed."""
    receipt_dir = _RUN_STATE.get("receipt_dir")
    if receipt_dir is None:
        return
    root = Path(str(receipt_dir))
    launches = {}
    for path in sorted(root.glob("*.launch.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):  # pragma: no cover - torn file
            continue
        launches[path.stem] = {
            "returncode": payload.get("returncode"),
            "seconds": payload.get("seconds"),
        }
    receipts = sorted(path.name for path in root.glob("*.json") if ".launch." not in path.name)
    _write_run_record(
        finished_at=time.time(),
        finished_at_text=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        wall_seconds=round(time.time() - float(_RUN_STATE.get("started_at") or time.time()), 1),
        launches=launches,
        receipts=receipts,
        receipt_count=len(receipts),
    )


def _write_json(path: Path, payload: dict) -> None:
    """Atomic write: a crash can never leave a half-written 0-byte receipt."""
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


atexit.register(_finalize_run_record)


def _worktree_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Two-rank drivers (M15–M18) — one script, three launch flavours
# ---------------------------------------------------------------------------


def _driver_script() -> Path:
    """The two-rank driver lives next to this file (not collected by pytest)."""
    return Path(__file__).resolve().parent / "fp16_cuda_driver.py"


def _base_env(receipt_dir: Path, scenarios: list[str], mode: str) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "FP16_DRIVER_MODE": mode,
            "FP16_SCENARIOS": ",".join(scenarios),
            "FP16_RECEIPT_DIR": str(receipt_dir),
            "PYTHONPATH": str(_worktree_root()) + os.pathsep + env.get("PYTHONPATH", ""),
            "PYTHONFAULTHANDLER": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _spawn(cmd: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=str(_worktree_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # one process group ⇒ a timeout can reap everything
    )


def _kill_group(process: subprocess.Popen, signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:  # pragma: no cover - already gone
        pass


def _run_group(
    specs: list[tuple[str, list[str], dict]], receipt_dir: Path, timeout: int = DRIVER_TIMEOUT
) -> None:
    """Run every rank process **concurrently** under one shared hard timeout.

    NCCL ranks rendezvous, so they must overlap. ``SIGABRT`` on timeout makes each child dump
    every thread stack (``PYTHONFAULTHANDLER=1``); the dump is archived and the row fails —
    a timeout is never retried into a pass.
    """
    started = time.time()
    deadline = started + timeout
    running = [(label, _spawn(cmd, env)) for label, cmd, env in specs]
    finished: dict[str, tuple[int, str, str]] = {}
    while running and time.time() < deadline:
        for label, process in list(running):
            if process.poll() is None:
                continue
            stdout, stderr = process.communicate()
            finished[label] = (process.returncode, stdout, stderr)
            running.remove((label, process))
        time.sleep(0.1)

    if running:
        for _label, process in running:
            _kill_group(process, signal.SIGABRT)
        parts = []
        for label, process in running:
            try:
                stdout, stderr = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
                _kill_group(process, signal.SIGKILL)
                stdout, stderr = process.communicate()
            parts.append(f"=== {label} ===\n--- stdout ---\n{stdout}\n--- stderr (stacks) ---\n{stderr}\n")
        archive = receipt_dir / "driver.timeout-stacks.txt"
        archive.write_text(
            f"hard timeout after {timeout}s (SIGABRT ⇒ faulthandler stack dump)\n"
            + "\n".join(parts)
        )
        pytest.fail(
            f"driver exceeded the {timeout}s hard timeout for {[label for label, _ in running]} "
            f"(124/137 ⇒ unverified, never a retried pass). Stack archive: {archive}"
        )

    elapsed = round(time.time() - started, 2)
    failures = []
    for label, (returncode, stdout, stderr) in finished.items():
        command = next(cmd for name, cmd, _env in specs if name == label)
        (receipt_dir / f"{label}.launch.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "command": command,
                    "returncode": returncode,
                    "seconds": elapsed,
                    "start_new_session": True,
                    "hard_timeout_seconds": timeout,
                    "faulthandler": True,
                    "stderr_tail": stderr[-4000:],
                },
                indent=1,
            )
        )
        if returncode != 0:
            failures.append(
                f"{label}: rc={returncode}\nstdout:\n{stdout[-2500:]}\nstderr:\n{stderr[-2500:]}"
            )
    if failures:
        pytest.fail("two-rank driver failed (row stays unverified):\n" + "\n".join(failures))


def _run_manual_driver(tmp_path: Path, scenarios: list[str]) -> Path:
    if "manual" in _LAUNCH_CACHE:
        return _LAUNCH_CACHE["manual"]
    receipt_dir = _receipt_dir(tmp_path)
    script = _driver_script()
    port = 29700 + (os.getpid() % 200)
    base = _base_env(receipt_dir, scenarios, "manual")
    base.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
        }
    )
    specs = []
    for rank in range(2):
        rank_env = dict(base)
        rank_env["RANK"] = str(rank)
        rank_env["LOCAL_RANK"] = str(rank)
        specs.append((f"manual.rank{rank}", [sys.executable, str(script)], rank_env))
    _run_group(specs, receipt_dir, timeout=110)
    _LAUNCH_CACHE["manual"] = receipt_dir
    return receipt_dir


def _run_accelerate_driver(tmp_path: Path, scenarios: list[str]) -> Path:
    if "accelerate" in _LAUNCH_CACHE:
        return _LAUNCH_CACHE["accelerate"]
    receipt_dir = _receipt_dir(tmp_path)
    script = _driver_script()
    env = _base_env(receipt_dir, scenarios, "accelerate")
    accelerate = Path(sys.executable).parent / "accelerate"
    if not accelerate.exists():
        pytest.fail("NCCL unavailable: the accelerate launcher is missing (cannot build native DDP)")
    command = [
        str(accelerate), "launch",
        "--multi_gpu",
        "--num_processes", "2",
        "--num_machines", "1",
        "--mixed_precision", "no",
        "--dynamo_backend", "no",
        "--main_process_port", str(29800 + (os.getpid() % 100)),
        str(script),
    ]
    _run_group([("accelerate", command, env)], receipt_dir, timeout=90)
    _LAUNCH_CACHE["accelerate"] = receipt_dir
    return receipt_dir


def _receipts(receipt_dir: Path, scenario: str) -> list[dict]:
    out = []
    for rank in range(2):
        path = receipt_dir / f"{scenario}.rank{rank}.json"
        assert path.exists(), f"{scenario}: rank {rank} produced no receipt ({path})"
        out.append(json.loads(path.read_text()))
    return out


def _cleanup_receipts(receipt_dir: Path) -> list[dict]:
    """M18④ / every two-rank row: process cleanup receipt (no leftover rank, no leak)."""
    out = []
    for rank in range(2):
        path = receipt_dir / f"cleanup.rank{rank}.json"
        assert path.exists(), f"rank {rank} left no cleanup receipt ({path})"
        payload = json.loads(path.read_text())
        assert payload["destroyed"] is True, f"rank {rank} did not destroy its process group"
        assert payload["failures"] == [], f"rank {rank} reported scenario failures: {payload['failures']}"
        assert payload["pending_after_barrier"] == 0
        # Per-card budget (§1.2: ≤2 GiB per device) plus "nothing model-sized survived cleanup":
        # the residual is NCCL's own communication buffers, well below any model/activation.
        assert payload["cuda_max_allocated_bytes"] <= 2 * 1024**3, (
            f"rank {rank} peaked at {payload['cuda_max_allocated_bytes']} device bytes "
            "(per-card budget is 2 GiB)"
        )
        assert payload["cuda_allocated_bytes"] <= 512 * 1024**2, (
            f"rank {rank} still holds {payload['cuda_allocated_bytes']} device bytes after cleanup"
        )
        out.append(payload)
    # No leftover processes: every rank pid must be gone once the launcher returns.
    for payload in out:
        with pytest.raises(ProcessLookupError):
            os.kill(payload["pid"], 0)
    return out


def _reduction_assertions(receipt: dict, windows: int) -> None:
    """M16③ — 1-element MAX reductions are boundary-scoped, never micro-batch-scoped.

    The window route issues exactly two of them per boundary: one for the update-drop gate
    (``agree_drop_update``) inside the micro-batch body and one for the window verdict
    (``agree_failure_reason``). ``micro_batches`` is deliberately 4, so a regression that
    moved a reduction to the micro-batch level would show up as 4+ instead of 2.

    Reading note — **captain 裁定 3**: the matrix wording "每边界只多 1 次" means "the count
    does not grow with micro-batches (never per micro-batch)"; the measured/asserted **2 per
    boundary** is the compliant implementation, because the drop gate and the window verdict
    are two decisions taken at two different moments. The assertions below therefore stay at
    2 per boundary — do not "fix" them to 1 to match the literal wording.
    """
    assert receipt["micro_batches"] == 4, "the scenario must keep 4 micro-batches per window"
    assert receipt["reduction_calls"] == 2 * windows, (
        f"M16③: expected {2 * windows} 1-element MAX reductions (2 per boundary, "
        f"{windows} boundary/ies), observed {receipt['reduction_calls']} for "
        f"{receipt['micro_batches']} micro-batches"
    )
    assert receipt["reduction_numels"] == [1] * (2 * windows), (
        "the counting stub must only ever see 1-element reductions"
    )


def _window_cause_totals(receipt: dict) -> int:
    return sum(int(value) for value in receipt["window_discard_causes"].values())


# ---------------------------------------------------------------------------
# M13 / M14 — single GPU, real fp16 + GradScaler
# ---------------------------------------------------------------------------
def _single_gpu_attempt(*, overflow: bool) -> SimpleNamespace:
    """One E3 window on the real accelerate fp16 stack (single process, GPU0)."""
    from accelerate import Accelerator

    from library.training import window_accounting, window_coordination
    from library.training.loss_recorder import LossRecorder

    accelerator = Accelerator(mixed_precision="fp16", cpu=False)
    if accelerator.scaler is None:
        pytest.fail("accelerate did not create a GradScaler for fp16: cannot exercise M13/M14")
    assert accelerator.device.type == "cuda", "the fp16 row must run on the CUDA device"

    torch.manual_seed(4321)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False)).to(accelerator.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    model, optimizer = accelerator.prepare(model, optimizer)

    state = SimpleNamespace(
        accelerator=accelerator,
        update_skipped=None,
        global_step=0,
        loss_recorder=LossRecorder(),
        args=SimpleNamespace(max_nonfinite_window_discards=3, adaptive_max_nonfinite_skips=0),
    )
    window_accounting.ensure_counters(state)
    window_coordination.ensure(state)

    step_calls = {"n": 0}
    original_step = optimizer.step

    def counting_step(*args, **kwargs):
        step_calls["n"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counting_step

    accelerator.gradient_state._set_sync_gradients(True)
    buffer = window_accounting.WindowBuffer()
    for micro in range(2):
        # Tiny magnitudes on purpose: the GradScaler's initial scale (65536) must not
        # overflow a *healthy* fp16 window, otherwise M13 would silently test M14.
        inputs = torch.randn(2, 8, device=accelerator.device) * 0.05 * (1.0 + 0.5 * micro)
        loss = model(inputs).float().pow(2).mean()
        if overflow and micro == 0:
            # A *finite* loss whose gradients overflow fp16: the scaler (not the loss
            # predicate) must be what skips the step, so the window fails on
            # ``update_skipped`` while the L1 bucket stays ``scaler_overflow``.
            loss = loss * 1e5
        assert math_isfinite(float(loss.detach())), "M14 overflow must keep the loss ledger finite"
        buffer.add(object(), float(loss.detach()), epoch=0, step=micro)
        accelerator.backward(loss)

    buffer.record_attempt()
    scale_before = accelerator.scaler.get_scale()
    state.update_skipped = None
    loss_is_finite = window_accounting.losses_are_finite(buffer.losses)
    drop_update = not loss_is_finite
    agreed_drop = window_coordination.agree_drop_update(state, drop_update)
    if agreed_drop:  # pragma: no cover - the single-rank rows never take this branch
        state.update_skipped = True
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
        state.update_skipped = bool(accelerator.optimizer_step_was_skipped)
        if state.update_skipped:
            optimizer.zero_grad(set_to_none=True)
    cause = window_coordination.note_attempt(
        state,
        route=window_coordination.ROUTE_WINDOW,
        skip_update=False,
        force_skip=False,
        loss_is_finite=loss_is_finite,
        step_attempted=not agreed_drop,
    )
    snapshot_after_attempt = window_coordination.snapshot(state)
    local_reason = window_coordination.local_window_reason(state, buffer)
    reason = window_coordination.agree_failure_reason(state, local_reason)
    if reason is None:
        window_accounting.commit_window(state, buffer)
        decision = "commit"
    else:
        window_accounting.discard_window(state, buffer, reason, local_reason=local_reason)
        decision = "discard"
    return SimpleNamespace(
        scaler_scale_before=scale_before,
        scaler_scale_after=accelerator.scaler.get_scale(),
        accelerator=accelerator,
        state=state,
        buffer=buffer,
        cause=cause,
        reason=reason,
        local_reason=local_reason,
        decision=decision,
        step_calls=step_calls,
        snapshot=window_coordination.snapshot(state),
        snapshot_after_attempt=snapshot_after_attempt,
        loss_is_finite=loss_is_finite,
    )


def math_isfinite(value: float) -> bool:
    import math

    return math.isfinite(value)


def _single_gpu_receipt(attempt: SimpleNamespace, row: str, receipt_dir: Path) -> dict:
    """Persist the M13/M14 readings as ``<row>.rank0.json`` and return the landed file.

    Same shape as the two-rank receipts (one file per row/rank), so t3/t4 read every CUDA row
    the same way instead of trusting a pytest traceback alone.
    """
    state = attempt.state
    snapshot = attempt.snapshot
    losses = [float(value) for value in (getattr(state.loss_recorder, "loss_list", None) or [])]
    payload = {
        "schema": "fp16-transaction-cuda-row/1",
        "run_id": RUN_ID,
        "row": row,
        "rank": 0,
        "world_size": int(getattr(attempt.accelerator, "num_processes", 1)),
        "device": str(attempt.accelerator.device),
        "scaler": attempt.accelerator.scaler is not None,
        "scaler_scale_before": float(attempt.scaler_scale_before),
        "scaler_scale_after": float(attempt.scaler_scale_after),
        "optimizer_step_was_skipped": bool(attempt.accelerator.optimizer_step_was_skipped),
        "step_calls": int(attempt.step_calls["n"]),
        "step_attempts": int(snapshot["step_attempts"]),
        "updates_effective": int(snapshot["updates_effective"]),
        "updates_skipped": int(snapshot["updates_skipped"]),
        "updates_skipped_by_scaler": int(snapshot["updates_skipped_by_scaler"]),
        "windows_committed": int(snapshot["windows_committed"]),
        "windows_discarded": int(snapshot["windows_discarded"]),
        "window_discard_causes": dict(snapshot["window_discard_causes"]),
        "window_discard_propagated": int(snapshot["window_discard_propagated"]),
        "originating_causes": dict(snapshot["originating_causes"]),
        "loss_list": losses,
        "loss_list_all_finite": all(math_isfinite(value) for value in losses),
        "loss_is_finite": bool(attempt.loss_is_finite),
        "update_skipped": state.update_skipped is True,
        "boundary_cause": attempt.cause,
        "local_reason": attempt.local_reason,
        "reason": attempt.reason,
        "decision": attempt.decision,
        "micro_batches": 2,
        "pid": os.getpid(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    path = Path(receipt_dir) / f"{row}.rank0.json"
    _write_json(path, payload)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.skipif(not RUN_CUDA, reason="opt-in CUDA row")
def test_m13_fp16_scaler_healthy_window_commits_without_warning(caplog, tmp_path):
    """M13 ①②③（真实单卡 fp16 + 真 scaler）。收据：``m13.rank0.json``。"""
    _require_gpus(1)
    _require_uuids([GPU0_UUID])

    from library.runtime.accelerator import _warn_fp16_without_scaler

    attempt = _single_gpu_attempt(overflow=False)

    # ① scaler 存在 ⇒ 不发 R6 warning
    with caplog.at_level(logging.WARNING):
        emitted = _warn_fp16_without_scaler(
            SimpleNamespace(mixed_precision="fp16"), attempt.accelerator
        )
    assert attempt.accelerator.scaler is not None, "M13 前提：真 scaler 存在"
    assert emitted is False, "M13①：有 scaler 时不得发 R6 warning"
    assert [r for r in caplog.records if r.name == "library.runtime.accelerator"] == [], (
        "M13①：library.runtime.accelerator 不得留下 warning 记录"
    )

    # ② updates.effective == updates.step_attempts
    assert attempt.step_calls["n"] == 1, "健康窗口只应发生一次优化器更新"
    assert attempt.scaler_scale_after == attempt.scaler_scale_before, (
        "M13 前提：健康窗口不得触发 scaler 降 scale（否则这一行会退化成 M14）"
    )
    assert attempt.snapshot["step_attempts"] == 1
    assert attempt.snapshot["updates_effective"] == attempt.snapshot["step_attempts"], (
        "M13②：effective 必须等于 step_attempts"
    )
    assert attempt.snapshot["updates_skipped_by_scaler"] == 0
    assert attempt.cause == "applied"
    assert attempt.decision == "commit"

    # ③ loss_list 有限（整窗提交式发布）
    published = attempt.state.loss_recorder.loss_list
    assert len(published) == 2, "M13③：提交窗口把两个微批 loss 都发布"
    assert all(math_isfinite(value) for value in published), "M13③：loss_list 必须全有限"

    # 落盘并对收据求值（进程内断言保留在上方，二者不一致同样判失败）
    receipt = _single_gpu_receipt(attempt, "m13", _receipt_dir(tmp_path))
    assert receipt["run_id"] == RUN_ID
    assert receipt["row"] == "m13"
    assert receipt["scaler"] is True, "M13 前提（收据）：真 scaler 存在"
    assert receipt["scaler_scale_after"] == receipt["scaler_scale_before"], (
        "M13 前提（收据）：健康窗口不得降 scale"
    )
    assert receipt["optimizer_step_was_skipped"] is False
    assert receipt["step_calls"] == 1
    assert receipt["step_attempts"] == 1
    assert receipt["updates_effective"] == receipt["step_attempts"], "M13②（收据）：effective == attempts"
    assert receipt["updates_skipped_by_scaler"] == 0
    assert receipt["windows_committed"] == 1
    assert receipt["window_discard_causes"] == {}
    assert receipt["boundary_cause"] == "applied"
    assert receipt["decision"] == "commit"
    assert len(receipt["loss_list"]) == 2, "M13③（收据）：两个微批都发布"
    assert receipt["loss_list_all_finite"] is True, "M13③（收据）：loss_list 全有限"


@pytest.mark.skipif(not RUN_CUDA, reason="opt-in CUDA row")
def test_m14_fp16_scaler_overflow_skips_update_and_accounts_both_layers(tmp_path):
    """M14 ①②③④（真实 scaler 溢出跳步；L1 与 E3 枚举互不替代）。收据：``m14.rank0.json``。"""
    _require_gpus(1)
    _require_uuids([GPU0_UUID])

    attempt = _single_gpu_attempt(overflow=True)

    # 前提：真的发生了 scaler 跳步（不是手工置位）
    assert attempt.step_calls["n"] == 1, "边界上 optimizer.step() 只调用一次"
    assert attempt.accelerator.optimizer_step_was_skipped is True, (
        "M14 前提：GradScaler 必须真的报告跳步"
    )
    assert attempt.scaler_scale_after < attempt.scaler_scale_before, (
        "M14 前提：溢出必须让 GradScaler 降 scale"
    )

    # ① 更新被跳：effective 不增、step_attempts 增
    assert attempt.snapshot_after_attempt["step_attempts"] == 1, "M14①：step_attempts 必须 +1"
    assert attempt.snapshot_after_attempt["updates_effective"] == 0, "M14①：effective 不得增加"
    assert attempt.snapshot_after_attempt["windows_committed"] == 0

    # ② 窗口按 update_skipped 判失败
    assert attempt.local_reason == "update_skipped", (
        "M14②：loss 有限、梯度溢出时窗口必须按 update_skipped 判失败"
    )
    assert attempt.reason == "update_skipped"
    assert attempt.decision == "discard"

    # ③ state.update_skipped is True
    assert attempt.state.update_skipped is True, "M14③：update_skipped 必须为 True"

    # ④ discard_causes.update_skipped +1，且与 L1 scaler_overflow 并行记录
    assert attempt.snapshot["window_discard_causes"] == {"update_skipped": 1}, (
        "M14④：E3 两值枚举的键名与计数都必须原样保留"
    )
    assert attempt.cause == "scaler_overflow", "M14④：L1 桶单独记 scaler_overflow"
    assert attempt.snapshot["updates_skipped_by_scaler"] == 1
    assert attempt.snapshot["originating_causes"]["scaler_overflow"] == 1

    # 落盘并对收据求值（进程内断言保留在上方）
    receipt = _single_gpu_receipt(attempt, "m14", _receipt_dir(tmp_path))
    assert receipt["run_id"] == RUN_ID
    assert receipt["row"] == "m14"
    assert receipt["scaler"] is True
    assert receipt["optimizer_step_was_skipped"] is True, "M14 前提（收据）：真实 scaler 跳步"
    assert receipt["scaler_scale_after"] < receipt["scaler_scale_before"], (
        "M14 前提（收据）：溢出必须降 scale"
    )
    assert receipt["step_calls"] == 1
    assert receipt["step_attempts"] == 1, "M14①（收据）：step_attempts 增"
    assert receipt["updates_effective"] == 0, "M14①（收据）：effective 不增"
    assert receipt["windows_committed"] == 0
    assert receipt["local_reason"] == "update_skipped", "M14②（收据）：按 update_skipped 判失败"
    assert receipt["reason"] == "update_skipped"
    assert receipt["decision"] == "discard"
    assert receipt["update_skipped"] is True, "M14③（收据）：update_skipped is True"
    assert receipt["window_discard_causes"] == {"update_skipped": 1}, "M14④（收据）：E3 枚举键不变"
    assert receipt["originating_causes"]["scaler_overflow"] == 1, "M14④（收据）：L1 桶并行记录"
    assert receipt["updates_skipped_by_scaler"] == 1
    assert receipt["loss_list"] == [], (
        "M14：discard 窗口不得发布 loss（commit-only 时间轴）"
    )
    assert receipt["loss_is_finite"] is True, "M14：溢出档的 loss 本身仍有限（溢出在梯度侧）"


# ---------------------------------------------------------------------------
# M15 / M16 — two ranks, manual NCCL
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RUN_CUDA, reason="opt-in CUDA row")
def test_m15_two_rank_manual_nccl_verdict_and_weight_hash(tmp_path):
    """M15 ①②③④：manual NCCL 双档（健康 + 单 rank 坏 loss）。"""
    _require_gpus(2)
    _require_uuids([GPU0_UUID, GPU1_UUID])
    _require_nccl()

    receipt_dir = _run_manual_driver(tmp_path, list(MANUAL_SCENARIOS))
    healthy = _receipts(receipt_dir, "m15_healthy")
    bad = _receipts(receipt_dir, "m15_bad")
    _cleanup_receipts(receipt_dir)

    # ① 两 rank 取出同一 commit/replay/discard 决定
    assert healthy[0]["decision"] == healthy[1]["decision"] == "commit"
    assert bad[0]["decision"] == bad[1]["decision"] == "discard"
    assert bad[0]["reason"] == bad[1]["reason"] == "update_skipped", (
        "M15①：世界判定必须一致（bad 档 rank1 记 nonfinite_loss，rank0 记 update_skipped，"
        "MAX 规约后两 rank 取 update_skipped）"
    )
    assert bad[0]["local_reason"] != bad[1]["local_reason"], (
        "本地观测本就不同：坏 rank 自证 nonfinite_loss，干净 rank 只能是 update_skipped"
    )
    assert bad[1]["local_reason"] == "nonfinite_loss", "注入坏 loss 的 rank 必须自证成因"
    assert bad[0]["local_reason"] == "update_skipped"

    # ② driver 有硬超时（launch 收据）+ 超时栈留档机制
    launch = json.loads((receipt_dir / "manual.rank0.launch.json").read_text())
    assert 45 <= launch["hard_timeout_seconds"] <= 180, (
        "硬超时必须存在且落在每命令 180 s 上限内"
    )
    assert launch["faulthandler"] is True
    assert launch["start_new_session"] is True, "超时必须能整组回收（含子进程）"
    assert launch["returncode"] == 0

    # ③ 两 rank 权重逐位相同（hash 相等）
    assert healthy[0]["post_weight_hash"] == healthy[1]["post_weight_hash"], (
        "M15③：健康窗口后两 rank 权重 hash 必须相等"
    )
    assert healthy[0]["weight_changed"] is True, "健康档必须真的更新了权重"
    assert bad[0]["post_weight_hash"] == bad[1]["post_weight_hash"], (
        "M15③：坏窗后两 rank 权重 hash 必须相等（都没更新）"
    )
    assert bad[0]["post_weight_hash"] == bad[0]["pre_weight_hash"], "丢窗不得改动权重"
    assert bad[0]["grads_hash"] == bad[1]["grads_hash"], "manual mean-reduce 后两 rank 梯度一致"
    assert bad[0]["grads_finite"] is True and bad[1]["grads_finite"] is True, (
        "loss-poison 档只污染 loss 账本，梯度保持有限（否则这一行会退化成梯度注入）"
    )

    # ④ 两 rank discard_causes 一致：每个 rank 各记 1 个丢弃，终态成因取自同一 E3 枚举，
    #    且干净 rank 只许跟随（不得把世界成因写成本 rank 原发）
    assert healthy[0]["window_discard_causes"] == healthy[1]["window_discard_causes"] == {}
    assert _window_cause_totals(bad[0]) == _window_cause_totals(bad[1]) == 1
    assert bad[1]["window_discard_causes"] == {"nonfinite_loss": 1}, "原发 rank 记真实成因"
    assert bad[0]["window_discard_causes"] == {"update_skipped": 1}, (
        "M15④：跟随 rank 只能记本 rank 的窗口终态（update_skipped），不得冒充 nonfinite_loss"
    )
    # 边界层把 L1 原发与 L2 跟随分开（契约 C4.3②）
    assert bad[1]["boundary_cause"] == "nonfinite_loss"
    assert bad[1]["propagated_drops"] == 0
    assert bad[0]["boundary_cause"] is None
    assert bad[0]["propagated_drops"] == 1, "干净 rank 必须记成 L2 传播 drop"
    assert bad[0]["boundary_causes"] == {"no_local_reason": 1}, (
        "M15④：干净 rank 不得把世界丢弃记成自己的原发原因"
    )

    # 世界一致规约是每边界 2 次 1 元素 MAX（drop 门 + 窗口判定），与微批数无关
    _reduction_assertions(healthy[0], windows=1)
    _reduction_assertions(healthy[1], windows=1)


@pytest.mark.skipif(not RUN_CUDA, reason="opt-in CUDA row")
def test_m16_two_rank_apply_drop_agrees_and_reduces_once_per_boundary(tmp_path):
    """M16 ①②③：两档（fp16+scaler apply / 无 scaler drop）+ 每边界 1 次规约（共 2 侧）。"""
    _require_gpus(2)
    _require_uuids([GPU0_UUID, GPU1_UUID])
    _require_nccl()

    receipt_dir = _run_manual_driver(tmp_path, list(MANUAL_SCENARIOS))
    applies = _receipts(receipt_dir, "m16_fp16_scaler")
    drops = _receipts(receipt_dir, "m16_no_scaler")
    _cleanup_receipts(receipt_dir)

    # fp16+scaler 档：两 rank 同时 apply
    assert applies[0]["scaler"] is True and applies[1]["scaler"] is True
    assert applies[0]["update_skipped"] is False and applies[1]["update_skipped"] is False, (
        "M16①：两 rank 必须同时 apply（state.update_skipped 相同）"
    )
    assert applies[0]["agreed_drop"] == applies[1]["agreed_drop"] is False
    assert applies[0]["manual_sync_calls"] == applies[1]["manual_sync_calls"] == 1
    assert applies[0]["post_weight_hash"] == applies[1]["post_weight_hash"], "M16②：权重 hash 相同"
    assert applies[0]["weight_changed"] is True, "apply 档必须真的更新了权重"
    assert applies[0]["decision"] == applies[1]["decision"] == "commit"

    # 无 scaler 档：一 rank 梯度非有限 ⇒ 两 rank 同时 drop（不是只坏一个）
    assert drops[0]["scaler"] is False and drops[1]["scaler"] is False
    assert drops[0]["update_skipped"] is True and drops[1]["update_skipped"] is True, (
        "M16①：两 rank 必须同时 drop（state.update_skipped 相同）"
    )
    assert drops[0]["agreed_drop"] is True and drops[1]["agreed_drop"] is True
    assert drops[0]["post_weight_hash"] == drops[1]["post_weight_hash"], "M16②：权重 hash 相同"
    assert drops[0]["post_weight_hash"] == drops[0]["pre_weight_hash"], "丢弃不得改动权重"
    assert drops[0]["decision"] == drops[1]["decision"] == "discard"
    assert drops[0]["boundary_causes"] == drops[1]["boundary_causes"] == {
        "guard_drop_nonfinite_grad": 1
    }
    assert drops[0]["window_discard_causes"] == drops[1]["window_discard_causes"] == {
        "update_skipped": 1
    }

    # ③ 每边界只多 1 次 1 元素规约（drop 门），窗口判定再 1 次；合计 2，且不按微批计
    for receipt in (*applies, *drops):
        _reduction_assertions(receipt, windows=1)


# ---------------------------------------------------------------------------
# M17 — two ranks, accelerate native DDP reducer
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RUN_CUDA, reason="opt-in CUDA row")
def test_m17_native_ddp_reducer_world_consistent(tmp_path):
    """M17 ①②③④：accelerate 原生 DDP reducer + 真实 NCCL（健康 + 单 rank 坏窗）。"""
    _require_gpus(2)
    _require_uuids([GPU0_UUID, GPU1_UUID])
    _require_nccl()

    receipt_dir = _run_accelerate_driver(tmp_path, list(ACCELERATE_SCENARIOS))
    healthy = _receipts(receipt_dir, "m17_healthy")
    bad = _receipts(receipt_dir, "m17_bad")
    _cleanup_receipts(receipt_dir)

    for receipt in (*healthy, *bad):
        # 原生 reducer + 真实 NCCL，且从未回退到手法规约
        assert receipt["native_reducer"]["is_ddp"] is True, "M17：必须是框架原生 DDP 包装"
        assert receipt["native_reducer"]["has_native_reducer"] is True, (
            "M17：DDP 必须带原生 reducer（不得用手动规约替代）"
        )
        assert receipt["native_reducer"]["backend"] == "nccl", "M17：backend 必须是真实 NCCL"
        assert receipt["native_reducer"]["num_processes"] == 2
        assert receipt["manual_sync_calls"] == 0, (
            "M17：一旦走手动规约就等于把 M17 缩窄成 M15，必须为 0"
        )
        assert receipt["backend"] == "nccl" and receipt["world_size"] == 2
        _reduction_assertions(receipt, windows=1)

    # ① 跑完无挂起 + 两 rank 权重 hash 相同
    assert healthy[0]["post_weight_hash"] == healthy[1]["post_weight_hash"], (
        "M17①：两 rank 权重 hash 必须相同"
    )
    assert healthy[0]["weight_changed"] is True, "健康档必须真的更新权重"
    assert bad[0]["post_weight_hash"] == bad[1]["post_weight_hash"]
    assert bad[0]["post_weight_hash"] == bad[0]["pre_weight_hash"], "坏窗不得改动权重"
    # 非对称 fault：坏 rank 自己带着非有限梯度结束窗口，干净 rank 的梯度是有限的。
    # 世界一致的目的正是让这种非对称也不会出现"一边更新、一边丢弃"。
    assert bad[0]["grads_finite"] is True
    assert bad[1]["grads_finite"] is False

    # ② 判定与 apply/drop 世界一致
    assert healthy[0]["decision"] == healthy[1]["decision"] == "commit"
    assert healthy[0]["agreed_drop"] == healthy[1]["agreed_drop"] is False
    assert bad[0]["decision"] == bad[1]["decision"] == "discard"
    assert bad[0]["agreed_drop"] == bad[1]["agreed_drop"] is True
    assert bad[0]["reason"] == bad[1]["reason"] == "update_skipped"

    # ③ discard_causes 两 rank 一致
    assert healthy[0]["window_discard_causes"] == healthy[1]["window_discard_causes"] == {}
    assert bad[0]["window_discard_causes"] == bad[1]["window_discard_causes"] == {
        "update_skipped": 1
    }
    # 边界层必须把 L1 原发与 L2 跟随分开（契约 C4.3②）：坏 rank 自证守卫丢弃，
    # 干净 rank 只记 no_local_reason + propagated_drops，不得冒充原发原因。
    assert bad[1]["boundary_causes"] == {"guard_drop_nonfinite_grad": 1}
    assert bad[1]["propagated_drops"] == 0
    assert bad[0]["boundary_causes"] == {"no_local_reason": 1}
    assert bad[0]["propagated_drops"] == 1

    # ④ 硬超时 + 超时栈留档
    launch = json.loads((receipt_dir / "accelerate.launch.json").read_text())
    assert 45 <= launch["hard_timeout_seconds"] <= 180, (
        "硬超时必须存在且落在每命令 180 s 上限内"
    )
    assert launch["faulthandler"] is True
    assert launch["start_new_session"] is True
    assert launch["returncode"] == 0


# ---------------------------------------------------------------------------
# M18 — two ranks, real async begin/finish lifecycle
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RUN_CUDA, reason="opt-in CUDA row")
def test_m18_async_begin_finish_pairing_healthy_and_bad_window(tmp_path):
    """M18 ①②③④：健康档 + 单 rank 坏窗档，双侧计数 + 配对 + 清理收据。

    配对口径（§6.2 禁止「用 finish 单侧计数推断配对」）：生产接线是 ``begin_step()`` 每个
    微批 +1、``finish_step()`` 每个 optimizer 边界 −1，因此两档都必须核到
      ``begin_calls == windows × micro_batches``、``finish_calls == windows``、
      ``steps_pending == windows × (micro_batches − 1)``（独立推导，不读实现里的中间量），
      并且在任何 finish 之前 begin 侧就已可见。
    ``steps_pending`` 是配对可读性计数，**不是泄漏检测**：只有每窗单微批（``m18_paired``）
    才应当为 0，累积窗口停在 ``windows × (micro_batches − 1)``。
    """
    _require_gpus(2)
    _require_uuids([GPU0_UUID, GPU1_UUID])
    _require_nccl()

    from library.training.gradient_sync import AsyncGradientSynchronizer

    receipt_dir = _run_manual_driver(tmp_path, list(MANUAL_SCENARIOS))
    paired = _receipts(receipt_dir, "m18_paired")
    healthy = _receipts(receipt_dir, "m18_healthy")
    bad = _receipts(receipt_dir, "m18_bad")
    _cleanup_receipts(receipt_dir)

    for receipt in (*paired, *healthy, *bad):
        final = receipt["lifecycle_final"]
        windows, micro = receipt["windows"], receipt["micro_batches"]
        # ① 双侧都被计数 + 配对关系可核（不是只看 finish）
        assert final["finish_calls"] == windows, (
            f"M18①：每个边界一次 finish（窗口 {windows}），实际 {final}"
        )
        assert final["begin_calls"] == windows * micro, (
            f"M18①：begin 每个微批一次（{windows}×{micro}），实际 {final}"
        )
        assert final["steps_pending"] == final["begin_calls"] - final["finish_calls"], (
            "M18①：pending 深度必须与该窗口的 begin/finish 差自洽"
        )
        # 独立推导的期望值（不依赖实现的中间量）：每窗留下 micro-1 个未配对 begin
        expected_pending = windows * (micro - 1)
        assert final["steps_pending"] == expected_pending, (
            f"M18①：静止 pending 必须等于 windows×({micro}-1)={expected_pending}"
            f"（windows={windows}, micro_batches={micro}），实际 {final['steps_pending']}"
        )
        assert receipt["cleanup"]["pending_matches_gap"] is True, (
            "pending 必须与 begin−finish 差一致（收据自洽）"
        )
        assert AsyncGradientSynchronizer.expected_pending(windows, micro) == expected_pending, (
            "只读 helper expected_pending 必须与测试独立推导值一致"
        )
        for index, snapshot in enumerate(receipt["lifecycle_after_begin"]):
            assert snapshot["begin_calls"] == index * micro + 1, (
                "M18①：begin 侧必须早于 finish 独立可见"
            )
            assert snapshot["finish_calls"] == index
        for index, snapshot in enumerate(receipt["lifecycle_after_finish"]):
            assert snapshot["finish_calls"] == index + 1
            assert snapshot["steps_pending"] == (index + 1) * (micro - 1), (
                f"M18①：第 {index + 1} 次 finish 之后 pending 必须等于 "
                f"({index + 1})×({micro}-1)={(index + 1) * (micro - 1)}"
            )
        assert receipt["bucket_result"] and receipt["bucket_result"][0][2] >= 1, (
            "M18：async 分桶必须真的存在"
        )
        assert receipt["cleanup"]["synchronizer_closed"] is True

    # 平衡形态（1 微批/窗口）：begin == finish == 2、pending 归零
    for receipt in paired:
        micro = receipt["micro_batches"]
        assert micro == 1, "m18_paired 必须保持每窗单微批（grad_accum==1）"
        assert receipt["lifecycle_final"] == {
            "begin_calls": 2 * micro,
            "finish_calls": 2,
            "steps_pending": 2 * (micro - 1),
        }, "配对平衡形态（每窗单微批）必须 pending 归零"
        assert receipt["cleanup"]["pairing_balanced"] is True
    assert paired[0]["decision"] == paired[1]["decision"] == "commit"
    assert paired[0]["post_weight_hash"] == paired[1]["post_weight_hash"]
    assert paired[0]["grads_hash_per_window"] == paired[1]["grads_hash_per_window"], (
        "M18：配对平衡形态下 async 规约后两 rank 梯度必须逐位一致"
    )

    # 健康档（4 微批/窗口）：apply + 梯度经 async 分桶规约后一致
    assert healthy[0]["windows"] == 2 and healthy[0]["micro_batches"] == 4
    assert healthy[0]["decision"] == healthy[1]["decision"] == "commit"
    assert healthy[0]["update_skipped"] is False and healthy[1]["update_skipped"] is False
    assert healthy[0]["grads_hash_per_window"] == healthy[1]["grads_hash_per_window"], (
        "M18：健康档 async 分桶规约后两 rank 梯度必须逐位一致"
    )
    assert healthy[0]["post_weight_hash"] == healthy[1]["post_weight_hash"]
    assert healthy[0]["weight_changed"] is True
    _reduction_assertions(healthy[0], windows=2)

    # 坏窗档：③ 世界一致决定生效，② 不挂起不发散
    assert bad[0]["windows"] == 1
    assert bad[0]["decision"] == bad[1]["decision"] == "discard"
    assert bad[0]["agreed_drop"] == bad[1]["agreed_drop"] is True
    assert bad[0]["reason"] == bad[1]["reason"] == "update_skipped"
    assert bad[0]["update_skipped"] is True and bad[1]["update_skipped"] is True
    assert bad[0]["weights_finite"] is True and bad[1]["weights_finite"] is True, (
        "M18②：坏窗路径不得把非有限权重写进模型（不发散）"
    )
    assert bad[0]["post_weight_hash"] == bad[1]["post_weight_hash"]
    assert bad[0]["post_weight_hash"] == bad[0]["pre_weight_hash"]
    assert bad[0]["window_discard_causes"] == bad[1]["window_discard_causes"] == {
        "update_skipped": 1
    }
    # 坏窗 fault 经真实 async 分桶规约传到两侧：两 rank 都自证守卫丢弃（对称，无 L2 跟随）
    assert bad[0]["boundary_causes"] == bad[1]["boundary_causes"] == {
        "guard_drop_nonfinite_grad": 1
    }
    assert bad[0]["propagated_drops"] == bad[1]["propagated_drops"] == 0
    assert bad[0]["lifecycle_after_finish"][0]["steps_pending"] == 3
    _reduction_assertions(bad[0], windows=1)

    # ④ 无残留进程：清理收据 + 每个 rank 进程确已退出
    for rank in range(2):
        cleanup = json.loads((receipt_dir / f"cleanup.rank{rank}.json").read_text())
        assert cleanup["destroyed"] is True
        assert cleanup["cuda_max_allocated_bytes"] <= 2 * 1024**3

# ---------------------------------------------------------------------------
# M19-R2 ⑥ / M26 — the two-rank combination gate
# ---------------------------------------------------------------------------
#: L1 (originating) causes of the matrix C4.1 layer. A rank that owns one of these is the
#: originator; every other rank may only be the follower.
L1_CAUSES = frozenset(
    {"nonfinite_loss", "guard_drop_nonfinite_grad", "scaler_overflow", "window_prefix_nonfinite"}
)


def _window_view(receipt: dict) -> dict:
    """M19-R2 ⑥ ``windows.*`` view as written by the driver."""
    ledger = receipt.get("window_ledger")
    assert isinstance(ledger, dict), (
        f"{receipt.get('scenario')} rank {receipt.get('rank')} receipt has no window_ledger"
    )
    return ledger


@pytest.mark.skipif(not RUN_CUDA, reason="opt-in CUDA row")
def test_m19_two_rank_combination_gate_windows_and_discard_causes(tmp_path):
    """M19-R2 ⑥（双卡组合门）/ M26 双卡部分：两 rank 的 ``windows.*`` 与 discard_causes 一致。

    真实 CUDA 双卡档：manual NCCL 收据（m15_*/m16_*/m18_*）与 accelerate 原生 DDP 收据
    （m17_*）都已经在别的行里产出，这里**复用同一批双 rank 收据**，不新增第三次双卡 launch。

    判定口径（captain 裁定 1「可对账一致」）：
      (i)   决策层终态一致 —— decision/reason、``windows.committed/discarded/replayed/
            attempts`` 逐键相等（世界一致的决定）；
      (ii)  原发 XOR 跟随归属唯一 —— 恰好一个 rank 在 L1 桶里自证成因；
      (iii) 跟随 rank 的 L1 桶为空，不得把 nonfinite_loss / scaler_overflow /
            guard_drop_nonfinite_grad 记成自己的原发原因（契约 C4.3②）；
      (iv)  L2 传播计数与边界 propagated_drops 自洽（跟随 rank 有 1 次传播、原发 rank 为 0，
            且窗口层不得二次计数）。
    对称档（两 rank 同因）要求 ``window_ledger`` 与 discard_causes **逐键相等**。
    """
    _require_gpus(2)
    _require_uuids([GPU0_UUID, GPU1_UUID])
    _require_nccl()

    manual_dir = _run_manual_driver(tmp_path, list(MANUAL_SCENARIOS))
    ddp_dir = _run_accelerate_driver(tmp_path, list(ACCELERATE_SCENARIOS))

    # ---- (a) 对称/健康档：windows.* 与 discard_causes 必须逐键相等 ---------------------
    for scenario, directory in (
        ("m15_healthy", manual_dir),
        ("m16_fp16_scaler", manual_dir),
        ("m16_no_scaler", manual_dir),
        ("m18_healthy", manual_dir),
        ("m18_paired", manual_dir),
        ("m18_bad", manual_dir),
        ("m17_healthy", ddp_dir),
    ):
        left, right = _receipts(directory, scenario)
        assert _window_view(left) == _window_view(right), (
            f"M19-R2 ⑥（{scenario}）：两 rank 的 windows.* 必须逐键相等"
        )
        assert left["window_discard_causes"] == right["window_discard_causes"], (
            f"M19-R2 ⑥（{scenario}）：对称档 discard_causes 必须逐键相等"
        )
        assert _window_view(left)["attempts"] == left["step_attempts"], (
            f"M19-R2 ⑥（{scenario}）：windows.attempts 必须等于 step_attempts（I2 交叉校验）"
        )
        assert _window_view(left)["committed"] == left["effective"], (
            f"M19-R2 ⑥（{scenario}）：E3 路由下 effective 必须等于 windows.committed"
        )

    # ---- (b) 非对称档：原发/跟随按裁定 1 的四条可对账 ---------------------------------
    for scenario, directory in (("m15_bad", manual_dir), ("m17_bad", ddp_dir)):
        left, right = _receipts(directory, scenario)
        pair = (left, right)

        # (i) 决策层终态一致：决定 + 计账逐键相等
        assert left["decision"] == right["decision"] == "discard", f"M19-R2 ⑥（{scenario}）(i)"
        assert left["reason"] == right["reason"]
        assert left["window_world_decisions"] == right["window_world_decisions"], (
            f"M19-R2 ⑥（{scenario}）(i)：L3 世界决定必须一致"
        )
        for key in (
            "committed",
            "discarded",
            "replayed",
            "attempts",
            "consecutive_discards",
            "discard_propagated",
        ):
            assert _window_view(left)[key] == _window_view(right)[key], (
                f"M19-R2 ⑥（{scenario}）(i)：windows.{key} 必须逐键相等"
            )
        assert _window_view(left)["attempts"] == left["step_attempts"]

        # (ii) 原发 XOR 跟随：L1 归属唯一
        originating = [
            index
            for index, receipt in enumerate(pair)
            if set(receipt["boundary_causes"]) & L1_CAUSES
        ]
        assert len(originating) == 1, (
            f"M19-R2 ⑥（{scenario}）(ii)：L1 原发归属必须唯一，实际 {originating}"
        )
        origin = originating[0]
        follower = 1 - origin

        # (iii) 跟随 rank 的 L1 桶为空，不冒充任何 L1 原因
        assert not (set(pair[follower]["boundary_causes"]) & L1_CAUSES), (
            f"M19-R2 ⑥（{scenario}）(iii)：跟随 rank 的 L1 桶必须为空"
        )
        assert pair[follower]["boundary_causes"] == {"no_local_reason": 1}
        assert not (set(pair[follower]["window_discard_causes"]) & L1_CAUSES), (
            f"M19-R2 ⑥（{scenario}）(iii)：跟随 rank 的窗口层也不得冒充原发原因"
        )
        assert pair[follower]["window_discard_causes"] == {"update_skipped": 1}

        # (iv) L2 传播计数与边界 propagated_drops 自洽，且窗口层不二次计数
        assert pair[follower]["propagated_drops"] == 1, (
            f"M19-R2 ⑥（{scenario}）(iv)：跟随 rank 必须记 1 次 L2 传播"
        )
        assert pair[origin]["propagated_drops"] == 0
        assert _window_view(pair[follower])["discard_propagated"] == 0
        assert _window_view(pair[origin])["discard_propagated"] == 0
        # 窗口层成因的 L1 键不是必需的：loss 毒化档（m15_bad）窗口层直接记 nonfinite_loss，
        # 梯度毒化档（m17_bad）窗口层只记 update_skipped、原发归属留在边界层。因此这里只要求
        # 「两 rank 窗口层成因总量可对账且等于 windows.discarded」，L1 归属由上面 (ii) 保证。
        allowed_window_causes = {"update_skipped"} | set(L1_CAUSES)
        for receipt in pair:
            assert set(receipt["window_discard_causes"]) <= allowed_window_causes, (
                f"M19-R2 ⑥（{scenario}）：窗口层出现未知成因键 {receipt['window_discard_causes']}"
            )

        # discard_causes 总量可对账：两 rank 的键值总和相等且等于 windows.discarded
        assert _window_cause_totals(left) == _window_cause_totals(right), (
            f"M19-R2 ⑥（{scenario}）：discard_causes 总量必须可对账"
        )
        assert _window_cause_totals(left) == _window_view(left)["discarded"]
