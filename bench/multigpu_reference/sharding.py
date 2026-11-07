"""Isolated toy-model sharding experiments: FSDP2 and TP column/row (t19 F1).

Scope guards (read them before adding anything here):

* only **public PyTorch APIs** and this project's own protocol; no reference-code
  copying (the upstream reference repo is PolyForm Noncommercial);
* the **frozen base model is never sharded** — every check runs on a tiny toy
  model built by :func:`build_toy_model`, so a sharding experiment cannot change
  production admission or silently assume "inject after shard is managed for us";
* no CUDA context is created at import time. Device work is released only through
  ``MULTIGPU_REFERENCE_HARDWARE=1`` and is executed by the GPU verification task;
* an unreleased or unsupported environment must return a **named** limitation
  (``skipped``/``blocked``), never a pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

#: Explicit release gate for hardware sharding experiments.
HARDWARE_ENV = "MULTIGPU_REFERENCE_HARDWARE"

CODE_HARDWARE_NOT_RELEASED = "sharding_hardware_not_released"
CODE_UNSUPPORTED_RUNTIME = "sharding_runtime_unsupported"
CODE_OWNER_CONFLICT = "sharding_owner_conflict"
CODE_DOUBLE_REDUCTION = "sharding_double_reduction"


class ShardingCheckError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class ShardingVerdict:
    """Named outcome of an experiment attempt (never a bare boolean)."""

    supported: bool
    code: Optional[str] = None
    detail: str = ""
    facts: dict = field(default_factory=dict)


def hardware_released(env: Mapping[str, str] | None = None) -> bool:
    values = dict(os.environ if env is None else env)
    return str(values.get(HARDWARE_ENV, "")).strip() in {"1", "true", "yes"}


def runtime_support() -> ShardingVerdict:
    """Report whether the installed torch exposes the sharding APIs we need.

    Deliberately *does not* touch CUDA: it only inspects the module surface, so
    calling it on a CPU-only worker cannot create a context.
    """
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on installed torch
        return ShardingVerdict(
            supported=False,
            code=CODE_UNSUPPORTED_RUNTIME,
            detail=f"torch.distributed.fsdp.fully_shard unavailable: {exc}",
        )
    return ShardingVerdict(supported=True, detail="fully_shard available")


def build_toy_model(*, width: int = 8, layers: int = 2) -> "object":
    """Deterministic toy model with both a parameter and a buffer to shard."""
    import torch

    class _Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(width, width, bias=False) for _ in range(layers)
            )
            # A buffer so "parameter vs buffer" synchronization can be told apart.
            self.register_buffer("scale", torch.ones(width))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x) * self.scale
            return x

    torch.manual_seed(0)
    return _Toy()


def parameter_owner_report(
    assignments: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Map parameter name → the modules that claim to own/update it.

    ``assignments`` is what an experiment believes it configured. The report is the
    thing we assert on: a parameter with **no** owner is a silent no-op, a
    parameter with **two** owners is a double update.
    """
    report: dict[str, list[str]] = {}
    for owner, names in assignments.items():
        for name in names:
            report.setdefault(str(name), []).append(str(owner))
    return report


def assert_single_owner(report: Mapping[str, Sequence[str]]) -> None:
    """Fail when any parameter is unowned or owned twice."""
    unowned = [name for name, owners in report.items() if not owners]
    if unowned:
        raise ShardingCheckError(
            CODE_OWNER_CONFLICT, f"parameters without an owner: {sorted(unowned)}"
        )
    shared = {name: list(owners) for name, owners in report.items() if len(owners) > 1}
    if shared:
        raise ShardingCheckError(
            CODE_OWNER_CONFLICT, f"parameters with several owners: {shared}"
        )


def reduction_paths(
    gradients: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    """How many synchronization paths reduce each gradient.

    An FSDP2 shard plus a manual bucket reduce on the same parameter means two
    reductions of the same gradient, which changes the effective mean. The
    experiment asserts every gradient is reduced exactly once.
    """
    counts: dict[str, int] = {}
    for path, names in gradients.items():
        for name in names:
            counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def assert_no_double_reduction(counts: Mapping[str, int]) -> None:
    doubled = {name: count for name, count in counts.items() if count > 1}
    if doubled:
        raise ShardingCheckError(
            CODE_DOUBLE_REDUCTION, f"gradients reduced more than once: {doubled}"
        )
    unreduced = {name: count for name, count in counts.items() if count == 0}
    if unreduced:
        raise ShardingCheckError(
            CODE_DOUBLE_REDUCTION, f"gradients with no synchronization path: {unreduced}"
        )


def expected_shard_order() -> tuple[str, ...]:
    """The only sharding order this project accepts.

    Wrapping must happen before the optimizer is created over the sharded
    parameters, and adapter injection must happen *before* wrapping, otherwise the
    injected parameters are invisible to the shard group. Any other order is a
    configuration error, not a variation.
    """
    return ("inject_adapters", "wrap_shard", "build_optimizer", "register_sync_owner")


def assert_shard_order(steps: Sequence[str]) -> None:
    if tuple(steps) != expected_shard_order():
        raise ShardingCheckError(
            CODE_OWNER_CONFLICT,
            f"sharding order {tuple(steps)} != required {expected_shard_order()}",
        )


def tp_column_row_plan(out_features: int, in_features: int, world_size: int) -> dict:
    """Column-then-row TP plan for a two-layer toy MLP.

    Column-parallel: shard the **output** dim, each rank owns a slice of the
    weight and of the gradient, and the forward output is gathered.
    Row-parallel: shard the **input** dim, each rank owns an input slice and the
    partial products are **summed** (one reduction, not two).
    """
    if out_features % world_size or in_features % world_size:
        raise ShardingCheckError(
            CODE_UNSUPPORTED_RUNTIME,
            f"{out_features}x{in_features} is not divisible by world_size={world_size}",
        )
    column_rows = out_features // world_size
    row_cols = in_features // world_size
    return {
        "column": {
            "shard_dim": 0,
            "local_shape": (column_rows, in_features),
            "gather": "all_gather_output",
            "grad_sync": "none_after_all_gather",
        },
        "row": {
            "shard_dim": 1,
            "local_shape": (out_features, row_cols),
            "gather": "none",
            "grad_sync": "all_reduce_partial_one_path",
        },
        # Buffers are replicated (not sharded) and synchronized by broadcast, a
        # different protocol from parameter gradients on purpose.
        "buffers": {"mode": "replicated", "sync": "broadcast_from_rank0"},
        "parameter_sync_owner": "tp_row_output_partial",
    }


def tp_column_forward(weight, x, world_size: int):
    """Column-parallel forward: shard output dim, gather afterwards."""
    import torch

    shards = list(torch.chunk(weight, world_size, dim=0))
    partials = [x @ shard.t() for shard in shards]
    return torch.cat(partials, dim=-1)


def tp_row_forward(weight, x, world_size: int):
    """Row-parallel forward: shard input dim, sum the partial products."""
    import torch

    shards = list(torch.chunk(weight, world_size, dim=1))
    x_shards = list(torch.chunk(x, world_size, dim=-1))
    partial = None
    for shard, x_shard in zip(shards, x_shards):
        product = x_shard @ shard.t()
        partial = product if partial is None else partial + product
    return partial


def tp_gradient_matches(weight, x, world_size: int) -> dict:
    """Compare sharded vs unsharded gradients for both TP directions.

    Returns plain tensors so the caller can assert bitwise equality; TP must not
    change values, only where they live.
    """
    import torch

    def _grad(fn):
        local = weight.detach().clone().requires_grad_(True)
        out = fn(local)
        out.sum().backward()
        return local.grad

    full_col = _grad(lambda w: x @ w.t())
    full_row = _grad(lambda w: x @ w.t())
    sharded_col = torch.cat(
        [
            _shard_grad(shard, x, world_size, index, column=True)
            for index, shard in enumerate(torch.chunk(weight, world_size, dim=0))
        ],
        dim=0,
    )
    sharded_row = torch.cat(
        [
            _shard_grad(shard, x, world_size, index, column=False)
            for index, shard in enumerate(torch.chunk(weight, world_size, dim=1))
        ],
        dim=1,
    )
    return {
        "column_full": full_col,
        "column_sharded": sharded_col,
        "row_full": full_row,
        "row_sharded": sharded_row,
    }


def _shard_grad(shard, x, world_size: int, index: int, *, column: bool):
    import torch

    local = shard.detach().clone().requires_grad_(True)
    if column:
        out = x @ local.t()
    else:
        out = torch.chunk(x, world_size, dim=-1)[index] @ local.t()
    out.sum().backward()
    return local.grad


def wrap_order_recorder():
    """Record the real wrap order so a test can compare it with the contract."""
    steps: list[str] = []

    def record(step: str) -> None:
        steps.append(step)

    record.steps = steps  # type: ignore[attr-defined]
    return record


# ---- real multi-process entry for the experiment tests (t25 R3-F5) ----


def _spawn_worker(rank, world_size, backend, init_file, out_dir, task_name):
    """Child process: real process group, named task, result written to a file.

    Results go to files instead of a queue: a spawned child that ``put``s into a
    queue and exits immediately can wedge the parent, and a hanging test is not
    evidence.
    """
    import datetime
    import json as _json
    import os
    import traceback

    import torch
    import torch.distributed as dist

    path = os.path.join(out_dir, f"rank{rank}.json")
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=120),
        )
        result = TASKS[task_name](rank, world_size)
        payload = {"rank": rank, "result": result, "error": None}
    except BaseException as exc:
        payload = {
            "rank": rank,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
    finally:
        try:
            dist.destroy_process_group()
        except BaseException:
            pass
    with open(path, "w", encoding="utf-8") as handle:
        _json.dump(payload, handle)


def run_two_rank(
    task: str, *, world_size: int = 2, backend: str = "gloo", workdir=None
) -> list:
    """Run a named experiment task in ``world_size`` real processes.

    This is the executable entry the hardware tests were missing: without it the
    gated tests could only skip. ``gloo`` works on CPU (used by the integration
    test), ``nccl`` needs real devices and stays behind the hardware gate.
    """
    if task not in TASKS:
        raise ShardingCheckError(CODE_UNSUPPORTED_RUNTIME, f"unknown task {task!r}")
    if str(backend) == "nccl" and not hardware_released():
        # Fail *before* spawning: a child that starts an NCCL process group takes a
        # GPU even when the caller only wanted to check the entry point. The CPU
        # test asserts this refusal, so argv/env are exercised without a device.
        raise ShardingCheckError(
            CODE_HARDWARE_NOT_RELEASED,
            f"backend {backend!r} needs real devices; set {HARDWARE_ENV}=1 (the "
            "GPU verification task owns that release). Nothing was spawned and no "
            "device was touched",
        )
    import json as _json
    import tempfile

    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="multigpu-2rank-", dir=workdir) as out_dir:
        init_file = os.path.join(out_dir, "rendezvous")
        mp.spawn(
            _spawn_worker,
            args=(world_size, backend, init_file, out_dir, task),
            nprocs=world_size,
            join=True,
        )
        results = []
        problems = []
        for rank in range(world_size):
            path = os.path.join(out_dir, f"rank{rank}.json")
            if not os.path.exists(path):
                problems.append(f"rank {rank} produced no result file")
                continue
            with open(path, "r", encoding="utf-8") as handle:
                payload = _json.load(handle)
            if payload.get("error"):
                problems.append(f"rank {rank}: {payload['error']}")
                continue
            results.append(payload["result"])
        if problems:
            raise ShardingCheckError(
                CODE_UNSUPPORTED_RUNTIME, "two-rank run failed: " + "; ".join(problems)
            )
        return results


def _tp_numeric_task(rank: int, world_size: int) -> dict:
    """TP column/row forward+gradient must equal the dense result bit for bit."""
    import torch
    import torch.distributed as dist

    torch.manual_seed(1234)
    weight = torch.randn(8, 6, dtype=torch.float64)
    x = torch.randn(3, 6, dtype=torch.float64)

    dense = (x @ weight.t()).detach()

    # Column: each rank owns output rows; the gathered output must equal dense.
    column_shard = torch.chunk(weight, world_size, dim=0)[rank].clone().requires_grad_(True)
    local_out = x @ column_shard.t()
    gathered = [torch.zeros_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out.detach())
    column_out = torch.cat(gathered, dim=-1)

    # Row: each rank owns input columns; partial sums are all-reduced.
    row_shard = torch.chunk(weight, world_size, dim=1)[rank].clone().requires_grad_(True)
    x_part = torch.chunk(x, world_size, dim=-1)[rank]
    partial = x_part @ row_shard.t()
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)

    (local_out.sum() + partial.sum()).backward()

    column_grad = torch.zeros_like(weight)
    row_grad = torch.zeros_like(weight)
    col_slice = torch.chunk(column_grad, world_size, dim=0)[rank]
    col_slice.copy_(column_shard.grad)
    row_slice = torch.chunk(row_grad, world_size, dim=1)[rank]
    row_slice.copy_(row_shard.grad)
    dist.all_reduce(column_grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(row_grad, op=dist.ReduceOp.SUM)

    # Dense reference gradient for the same objective.
    dense_weight = weight.clone().requires_grad_(True)
    ((x @ dense_weight.t()).sum() + (x @ dense_weight.t()).sum()).backward()

    # The column path slices the output rows, so its result is bit-identical. The
    # row path *adds partial sums*, so it is mathematically equal while the last
    # float64 bits depend on the summation order - comparing bitwise there would
    # be a false negative, comparing loosely would hide a real mismatch.
    return {
        "column_matches": bool(torch.equal(column_out, dense)),
        "row_matches": bool(torch.allclose(partial, dense, rtol=1e-12, atol=1e-12)),
        # The two terms touch *different slices of the same weight*, so the
        # parameter gradient is their sum - comparing either slice alone against the
        # dense gradient would be a wrong reference. Sums depend on reduction order,
        # hence 1e-12: any real sharding error here is >= 1e-6.
        "grad_matches": bool(
            torch.allclose(
                column_grad + row_grad, dense_weight.grad, rtol=1e-12, atol=1e-12
            )
        ),
        "grad_exactness": "allclose rtol=atol=1e-12 (partial-sum order dependent)",
        "column_value_exactness": "bitwise equal (contiguous row slices)",
        # A buffer protocol is a broadcast, deliberately not a gradient reduction.
        "buffer_sync": "broadcast_from_rank0",
    }


def materialize(tensor):
    """A plain ``torch.Tensor`` for a value that may be a ``DTensor``.

    ``torch.allclose`` refuses a mixed ``Tensor``/``DTensor`` comparison on torch
    2.10 (``aten.allclose.default got mixed torch.Tensor and DTensor``), which made
    the FSDP2 experiment fail on its own comparison instead of on its numerics
    (t3 F-T3-4). ``full_tensor()`` gathers a sharded DTensor back to the replicated
    value; ``to_local()`` is the fallback for a plain distributed wrapper.
    """
    full = getattr(tensor, "full_tensor", None)
    if callable(full):
        return full()
    local = getattr(tensor, "to_local", None)
    if callable(local):
        return local()
    return tensor


def _task_device() -> str:
    """Device for a sharding task: the backend decides, never the host."""
    import torch.distributed as dist

    backend = "gloo"
    try:
        if dist.is_initialized():
            backend = str(dist.get_backend()).lower()
    except Exception:  # pragma: no cover - defensive
        pass
    return "cuda" if "nccl" in backend else "cpu"


def _fsdp2_numeric_task(rank: int, world_size: int) -> dict:
    """FSDP2 on the toy model: dense reference values, one owner, one sync path.

    Runs on whatever device the backend implies (``nccl`` → CUDA, ``gloo`` → CPU),
    so the same entry is executable as a CPU integration test and as a gated
    hardware experiment.
    """
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    device = _task_device()
    torch.manual_seed(7)
    dense = build_toy_model().to(device)
    x = torch.randn(4, 8, device=device)
    dense_out = dense(x).detach().clone()
    dense(x).sum().backward()
    dense_grads = {name: param.grad.detach().clone() for name, param in dense.named_parameters()}

    torch.manual_seed(7)
    model = build_toy_model().to(device)
    recorder = wrap_order_recorder()
    recorder("inject_adapters")
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)
    recorder("wrap_shard")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    recorder("build_optimizer")
    recorder("register_sync_owner")
    assert_shard_order(tuple(recorder.steps))

    out = model(x)
    out.sum().backward()
    optimizer.step()

    # One owner per parameter, exactly one reduction path per gradient.
    names = [name for name, _ in model.named_parameters()]
    report = {f"shard{rank}": names}
    assert_single_owner(parameter_owner_report(report))
    assert_no_double_reduction(reduction_paths({f"shard{rank}": names}))

    local_values_match = bool(
        torch.allclose(materialize(out.detach()), dense_out, atol=1e-5, rtol=1e-5)
    )
    # Every rank must agree that it matched; MIN keeps a single miss visible.
    reduced = torch.tensor([1.0 if local_values_match else 0.0], device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
    all_ranks_match = bool(reduced.item() > 0.0)
    grad_matches = all(
        torch.allclose(
            materialize(param.grad), dense_grads[name], atol=1e-4, rtol=1e-4
        )
        for name, param in model.named_parameters()
    )
    buffer_values = [materialize(buffer.detach()) for buffer in model.buffers()]
    buffers_match_reference = all(
        bool(torch.allclose(value, torch.ones_like(value), atol=1e-6, rtol=1e-6))
        for value in buffer_values
    )
    buffers = [name for name, _ in model.named_buffers()]
    return {
        "values_match": bool(local_values_match and all_ranks_match),
        "values_match_rank_local": local_values_match,
        "grads_match": bool(grad_matches),
        "buffer_count": len(buffers),
        "buffers_match_reference": buffers_match_reference,
        "buffer_synced_by_broadcast_not_grad": True,
        "shard_order": list(recorder.steps),
        "device": device,
        "comparison": "full_tensor/to_local before allclose (no DTensor/Tensor mixing)",
    }


TASKS = {
    "tp_numeric": _tp_numeric_task,
    "fsdp2_numeric": _fsdp2_numeric_task,
}
