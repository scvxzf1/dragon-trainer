"""Manual distributed synchronization for monkey-patched adapter networks.

Adapter modules are owned by a network container, but their forwards run from
the frozen DiT/text-encoder modules they patch. Wrapping that container in DDP
does not give DDP a matching root forward, so multi-process training keeps the
container unwrapped and synchronizes its optimizer gradients explicitly.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from accelerate.utils import broadcast


@dataclass(frozen=True)
class GradientSyncResult:
    parameter_count: int = 0
    reduced_parameter_count: int = 0
    materialized_gradient_count: int = 0
    bucket_count: int = 0


@dataclass(frozen=True)
class StateSyncResult:
    parameter_count: int = 0
    buffer_count: int = 0
    bucket_count: int = 0


@dataclass
class _AsyncBucket:
    parameters: list[torch.nn.Parameter]
    payload: torch.Tensor | None = None
    work: Any = None
    event: Any = None
    launched: bool = False


class AsyncGradientSynchronizer:
    """Launch fixed-order NCCL gradient buckets while backward is running.

    The synchronizer is deliberately opt-in. Methods that add gradients in
    ``after_backward`` use the existing synchronous path so those contributions
    cannot race an already-launched collective.
    """

    def __init__(self, accelerator: Any, optimizer: Any, *, bucket_bytes: int = 16 << 20):
        self.accelerator = accelerator
        self.parameters = optimizer_parameters(optimizer)
        self.buckets = self._make_buckets(bucket_bytes)
        self._handles = []
        self._active: list[_AsyncBucket] | None = None
        self._prior_grads: dict[int, torch.Tensor] = {}
        self._next_launch = 0
        self._comm_streams: dict[torch.device, torch.cuda.Stream] = {}
        # Async lifecycle observability (M18): the baseline only ever counted the
        # "finish" side, so an unmatched ``begin_step`` (early return / exception
        # between the two) was invisible. Both sides are counted and the pending
        # depth is exposed, which is what makes the pairing auditable.
        self.begin_calls = 0
        self.finish_calls = 0
        self.steps_pending = 0
        self._register_hooks()

    def _make_buckets(self, bucket_bytes: int) -> list[list[torch.nn.Parameter]]:
        # Backward visits leaf gradients roughly in reverse registration order;
        # matching that order lets early buckets start communication sooner.
        buckets: list[list[torch.nn.Parameter]] = []
        current: list[torch.nn.Parameter] = []
        current_bytes = 0
        for parameter in reversed(self.parameters):
            size = max(1, parameter.numel() * parameter.element_size())
            if current and current_bytes + size > bucket_bytes:
                buckets.append(current)
                current, current_bytes = [], 0
            current.append(parameter)
            current_bytes += size
        if current:
            buckets.append(current)
        return buckets

    def _register_hooks(self) -> None:
        if not hasattr(torch.autograd.graph, "register_multi_grad_hook"):
            return
        for index, parameters in enumerate(self.buckets):
            tensors = tuple(parameter for parameter in parameters if parameter.requires_grad)
            if not tensors:
                continue

            def callback(grads, bucket_index=index, bucket_parameters=tensors):
                self._mark_ready(bucket_index, bucket_parameters, grads)

            self._handles.append(
                torch.autograd.graph.register_multi_grad_hook(tensors, callback, mode="all")
            )

    def _stream_for(self, device: torch.device) -> torch.cuda.Stream:
        stream = self._comm_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._comm_streams[device] = stream
        return stream

    @property
    def lifecycle(self) -> dict[str, int]:
        """Read-only async begin/finish accounting (pairing readability, not leak detection).

        Wiring semantics (``loop.py::_run_microbatch`` / the sync boundary):

        * ``begin_step()`` runs once per **micro-batch** → ``begin_calls`` +1;
        * ``finish_step()`` runs once per **optimizer boundary** → ``finish_calls`` +1;
        * ``steps_pending`` therefore settles at ``windows × (micro_batches − 1)`` when a run
          finishes, e.g. 2 accumulation windows with 4 micro-batches each leave 6 pending.

        It is ``0`` only when every window holds a single micro-batch (``grad_accum == 1``,
        such as a one-shot window). A non-zero value here is **expected** for accumulation,
        not a leak: the counter exists so the pairing between the two sides is readable
        instead of being inferred from ``finish_calls`` alone (REPORT v2 F4). Use
        :meth:`expected_pending` for the derived value rather than hard-coding it.
        """
        return {
            "begin_calls": self.begin_calls,
            "finish_calls": self.finish_calls,
            "steps_pending": self.steps_pending,
        }

    @staticmethod
    def expected_pending(windows: int, micro_batches: int) -> int:
        """Read-only helper: the settled ``steps_pending`` for that many windows/micro-batches.

        ``windows × (micro_batches − 1)`` under the production wiring (``begin_step`` per
        micro-batch, ``finish_step`` per boundary). Purely derived — it changes no state and
        is not used by the training path.
        """
        return max(0, int(windows)) * max(0, int(micro_batches) - 1)

    def begin_step(self) -> None:
        self.begin_calls += 1
        self.steps_pending += 1
        if int(getattr(self.accelerator, "num_processes", 1)) <= 1:
            self._active = None
            return
        self._active = [_AsyncBucket(parameters=bucket) for bucket in self.buckets]
        self._prior_grads = {
            id(parameter): parameter.grad.detach().clone()
            for parameter in self.parameters
            if parameter.grad is not None
        }
        self._next_launch = 0

    def _mark_ready(
        self,
        index: int,
        parameters: tuple[torch.nn.Parameter, ...],
        grads: Iterable[torch.Tensor | None],
    ) -> None:
        if self._active is None:
            return
        self._active[index].payload = self._pack(
            self._active[index].parameters,
            grads=grads,
            parameters_for_grads=parameters,
        )
        self._launch_ready_buckets()

    def _pack(
        self,
        parameters: list[torch.nn.Parameter],
        *,
        grads: Iterable[torch.Tensor | None] | None = None,
        parameters_for_grads: tuple[torch.nn.Parameter, ...] | None = None,
    ) -> torch.Tensor:
        gradient_map = (
            {id(parameter): grad for parameter, grad in zip(parameters_for_grads or (), grads or ())}
            if grads is not None
            else {}
        )
        parts: list[torch.Tensor] = []
        presence: list[float] = []
        for parameter in parameters:
            incoming = gradient_map.get(id(parameter))
            prior = self._prior_grads.get(id(parameter))
            if incoming is not None and prior is not None:
                grad = prior + incoming.detach()
            elif incoming is not None:
                grad = incoming
            else:
                grad = parameter.grad
            if grad is None:
                parts.append(parameter.detach().new_zeros(parameter.numel()))
                presence.append(0.0)
            else:
                if grad.layout != torch.strided or grad.device != parameter.device or grad.dtype != parameter.dtype:
                    raise RuntimeError("adapter gradient device/dtype/layout is not synchronizable")
                parts.append(grad.detach().reshape(-1))
                presence.append(1.0)
        return torch.cat([*parts, parameters[0].new_tensor(presence)])

    def _launch_ready_buckets(self) -> None:
        if self._active is None:
            return
        while self._next_launch < len(self._active):
            bucket = self._active[self._next_launch]
            if bucket.payload is None:
                break
            payload = bucket.payload
            if payload.is_cuda and dist.is_initialized() and dist.get_backend() == "nccl":
                current = torch.cuda.current_stream(payload.device)
                stream = self._stream_for(payload.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    bucket.work = dist.all_reduce(payload, op=dist.ReduceOp.SUM, async_op=True)
                    bucket.event = torch.cuda.Event()
                    bucket.event.record(stream)
            else:
                bucket.work = None
                bucket.event = None
            bucket.launched = True
            self._next_launch += 1

    def finish_step(self) -> GradientSyncResult:
        self.finish_calls += 1
        self.steps_pending = max(0, self.steps_pending - 1)
        active = self._active
        if active is None:
            return GradientSyncResult()
        for bucket in active:
            if bucket.payload is None:
                bucket.payload = self._pack(bucket.parameters)
        self._launch_ready_buckets()
        reduced_parameter_count = 0
        materialized = 0
        for bucket in active:
            if bucket.work is not None:
                bucket.work.wait()
                bucket.event.synchronize()
                bucket.payload.div_(int(self.accelerator.num_processes))
            else:
                bucket.payload = self.accelerator.reduce(bucket.payload, reduction="mean")
            payload = bucket.payload
            offset = 0
            numel = sum(parameter.numel() for parameter in bucket.parameters)
            global_presence = payload[numel:].ne(0).to(device="cpu").tolist()
            for index, parameter in enumerate(bucket.parameters):
                end = offset + parameter.numel()
                if global_presence[index]:
                    synced = payload[offset:end].view_as(parameter)
                    if parameter.grad is None:
                        parameter.grad = synced.clone()
                        materialized += 1
                    else:
                        parameter.grad.copy_(synced)
                    reduced_parameter_count += 1
                offset = end
        self._active = None
        self._prior_grads.clear()
        return GradientSyncResult(len(self.parameters), reduced_parameter_count, materialized, len(active))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def optimizer_parameters(optimizer: Any) -> list[torch.nn.Parameter]:
    """Return unique parameters in deterministic optimizer-group order."""
    out: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group.get("params", ()):
            if not isinstance(param, torch.nn.Parameter):
                raise TypeError(
                    "optimizer param groups must contain torch.nn.Parameter objects"
                )
            if id(param) in seen:
                continue
            seen.add(id(param))
            out.append(param)
    return out


def _tensor_buckets(
    tensors: Iterable[torch.Tensor],
) -> list[list[torch.Tensor]]:
    buckets: OrderedDict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = (
        OrderedDict()
    )
    for tensor in tensors:
        if tensor.layout != torch.strided:
            raise RuntimeError(
                "manual adapter synchronization only supports dense strided tensors"
            )
        if tensor.device.type == "meta":
            raise RuntimeError("cannot synchronize adapter tensors on the meta device")
        buckets.setdefault((tensor.device, tensor.dtype), []).append(tensor)
    return list(buckets.values())


@torch.no_grad()
def synchronize_optimizer_state(
    accelerator: Any,
    network: torch.nn.Module,
    optimizer: Any,
    *,
    broadcast_fn: Callable[..., torch.Tensor] = broadcast,
) -> StateSyncResult:
    """Broadcast optimizer parameters and network buffers from rank zero once."""
    if int(getattr(accelerator, "num_processes", 1)) <= 1:
        return StateSyncResult()

    parameters = optimizer_parameters(optimizer)
    seen = {id(param) for param in parameters}
    buffers = [buffer for buffer in network.buffers() if id(buffer) not in seen]
    tensors: list[torch.Tensor] = [*parameters, *buffers]
    buckets = _tensor_buckets(tensors)

    for bucket in buckets:
        nonempty = [tensor for tensor in bucket if tensor.numel()]
        if not nonempty:
            continue
        flat = torch.cat([tensor.detach().reshape(-1) for tensor in nonempty])
        synced = broadcast_fn(flat, from_process=0)
        offset = 0
        for tensor in nonempty:
            end = offset + tensor.numel()
            tensor.copy_(synced[offset:end].view_as(tensor))
            offset = end

    return StateSyncResult(
        parameter_count=len(parameters),
        buffer_count=len(buffers),
        bucket_count=len(buckets),
    )


def prepare_network_for_manual_gradient_sync(
    accelerator: Any,
    network: torch.nn.Module,
    optimizer: Any,
) -> torch.nn.Module:
    """Move/register an adapter network without applying a DDP wrapper."""
    network = accelerator.prepare_model(network, evaluation_mode=True)
    state = synchronize_optimizer_state(accelerator, network, optimizer)
    if all(parameter.requires_grad for parameter in optimizer_parameters(optimizer)):
        network._anima_async_gradient_sync = AsyncGradientSynchronizer(
            accelerator, optimizer
        )
    overlap = "cuda-stream-overlap" if hasattr(network, "_anima_async_gradient_sync") else "sync-after-backward"
    accelerator.print(
        "manual adapter gradient sync enabled: "
        f"world_size={accelerator.num_processes}, "
        f"parameters={state.parameter_count}, buffers={state.buffer_count}, "
        f"state_buckets={state.bucket_count}, mode={overlap}"
    )
    return network


@torch.no_grad()
def synchronize_optimizer_gradients(
    accelerator: Any,
    optimizer: Any,
) -> GradientSyncResult:
    """Mean-reduce optimizer gradients while preserving rank-local ``None``.

    Every rank contributes an identically-shaped payload. A missing local
    gradient contributes zeros; if every rank is missing a gradient it remains
    ``None`` so weight decay and optimizer state do not advance for an unused
    parameter. Gradients intentionally remain loss-scaled here: a non-finite
    value then reaches every rank before GradScaler unscales/checks it.
    """
    if int(getattr(accelerator, "num_processes", 1)) <= 1:
        return GradientSyncResult()

    parameters = optimizer_parameters(optimizer)
    buckets = _tensor_buckets(parameters)
    reduced_parameter_count = 0
    materialized_gradient_count = 0

    for bucket in buckets:
        grad_parts: list[torch.Tensor] = []
        presence: list[float] = []
        total_numel = 0
        for param in bucket:
            grad = param.grad
            total_numel += param.numel()
            if grad is None:
                grad_parts.append(param.detach().new_zeros(param.numel()))
                presence.append(0.0)
                continue
            if grad.layout != torch.strided:
                raise RuntimeError(
                    "manual adapter gradient sync does not support sparse gradients"
                )
            if grad.device != param.device or grad.dtype != param.dtype:
                raise RuntimeError(
                    "adapter gradient device/dtype must match its optimizer parameter"
                )
            grad_parts.append(grad.detach().reshape(-1))
            presence.append(1.0)

        if not bucket:
            continue
        presence_tensor = bucket[0].new_tensor(presence)
        payload = torch.cat([*grad_parts, presence_tensor])
        reduced = accelerator.reduce(payload, reduction="mean")
        global_presence = reduced[total_numel:]
        present_on_any_rank = global_presence.ne(0).to(device="cpu").tolist()

        offset = 0
        for index, param in enumerate(bucket):
            end = offset + param.numel()
            if present_on_any_rank[index]:
                synced_grad = reduced[offset:end].view_as(param)
                if param.grad is None:
                    param.grad = synced_grad.clone()
                    materialized_gradient_count += 1
                else:
                    param.grad.copy_(synced_grad)
                reduced_parameter_count += 1
            offset = end

    return GradientSyncResult(
        parameter_count=len(parameters),
        reduced_parameter_count=reduced_parameter_count,
        materialized_gradient_count=materialized_gradient_count,
        bucket_count=len(buckets),
    )
