from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import library.training.bootstrap as bootstrap_mod
from library.training.scaler_ownership import GradFiniteness
import library.training.loop as training_loop
from library.training.bootstrap import TrainingBootstrap
from library.training.gradient_sync import (
    AsyncGradientSynchronizer,
    prepare_network_for_manual_gradient_sync,
    synchronize_optimizer_gradients,
    synchronize_optimizer_state,
)


def test_async_gradient_synchronizer_starts_from_backward_hook_and_finishes_mean_reduce():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))

    class FakeAccelerator:
        num_processes = 2

        @staticmethod
        def reduce(tensor, reduction):
            assert reduction == "mean"
            return tensor * 0.5

    optimizer = _OptimizerView([first, second])
    synchronizer = AsyncGradientSynchronizer(FakeAccelerator(), optimizer, bucket_bytes=1)
    synchronizer.begin_step()
    (first.sum() * 2 + second.sum() * 4).backward()
    assert all(bucket.payload is not None for bucket in synchronizer._active or [])
    result = synchronizer.finish_step()

    assert first.grad.tolist() == pytest.approx([1.0, 1.0])
    assert second.grad.tolist() == pytest.approx([2.0])
    assert result.reduced_parameter_count == 2
    synchronizer.close()


def test_async_gradient_synchronizer_preserves_prior_accumulation():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([3.0])

    class FakeAccelerator:
        num_processes = 2

        @staticmethod
        def reduce(tensor, reduction):
            assert reduction == "mean"
            return tensor * 0.5

    synchronizer = AsyncGradientSynchronizer(
        FakeAccelerator(), _OptimizerView([parameter]), bucket_bytes=1
    )
    synchronizer.begin_step()
    (parameter.sum() * 2).backward()
    synchronizer.finish_step()

    assert parameter.grad.tolist() == pytest.approx([2.5])
    synchronizer.close()


class _OptimizerView:
    def __init__(self, *groups) -> None:
        self.param_groups = [{"params": list(group)} for group in groups]


class _ScriptedAccelerator:
    num_processes = 2

    def __init__(self, reduced: torch.Tensor) -> None:
        self.reduced = reduced
        self.payloads: list[torch.Tensor] = []

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        assert reduction == "mean"
        self.payloads.append(tensor.clone())
        return self.reduced.to(device=tensor.device, dtype=tensor.dtype)


def test_gradient_sync_reduces_in_optimizer_order_and_materializes_missing_grad():
    first = torch.nn.Parameter(torch.zeros(2))
    conditional = torch.nn.Parameter(torch.zeros(1))
    globally_unused = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([1.0, 2.0])

    # Layout: gradients (2 + 1 + 1), then one presence scalar per parameter.
    accelerator = _ScriptedAccelerator(
        torch.tensor([3.0, 5.0, 4.0, 0.0, 1.0, 0.5, 0.0])
    )
    optimizer = _OptimizerView(
        [first, conditional, globally_unused],
        [first],  # Duplicate references must not alter collective layout.
    )

    result = synchronize_optimizer_gradients(accelerator, optimizer)

    assert accelerator.payloads[0].tolist() == [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert first.grad.tolist() == [3.0, 5.0]
    assert conditional.grad.tolist() == [4.0]
    assert globally_unused.grad is None
    assert result.parameter_count == 3
    assert result.reduced_parameter_count == 2
    assert result.materialized_gradient_count == 1
    assert result.bucket_count == 1


def test_gradient_sync_buckets_mixed_dtypes_without_changing_existing_storage():
    fp32 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    fp64 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    fp32.grad = torch.tensor([2.0, 4.0])
    fp64.grad = torch.tensor([6.0], dtype=torch.float64)
    fp32_grad = fp32.grad
    fp64_grad = fp64.grad

    class IdentityAccelerator:
        num_processes = 2

        def __init__(self) -> None:
            self.dtypes: list[torch.dtype] = []

        def reduce(self, tensor, reduction):
            assert reduction == "mean"
            self.dtypes.append(tensor.dtype)
            return tensor

    accelerator = IdentityAccelerator()
    result = synchronize_optimizer_gradients(
        accelerator, _OptimizerView([fp32, fp64])
    )

    assert accelerator.dtypes == [torch.float32, torch.float64]
    assert fp32.grad is fp32_grad
    assert fp64.grad is fp64_grad
    assert result.bucket_count == 2


def test_single_process_gradient_sync_is_a_noop():
    param = torch.nn.Parameter(torch.ones(1))
    param.grad = torch.tensor([7.0])
    accelerator = SimpleNamespace(num_processes=1)

    result = synchronize_optimizer_gradients(accelerator, _OptimizerView([param]))

    assert param.grad.tolist() == [7.0]
    assert result.parameter_count == 0


def test_state_sync_broadcasts_only_optimizer_parameters_plus_network_buffers():
    network = torch.nn.Module()
    network.trainable = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    network.frozen = torch.nn.Parameter(torch.tensor([8.0]), requires_grad=False)
    network.register_buffer("basis", torch.tensor([3.0]))
    optimizer = _OptimizerView([network.trainable])
    # Some monkey-patched parameters are temporarily frozen when the base DiT
    # is frozen, then re-enabled by prepare_grad_etc after accelerator setup.
    network.trainable.requires_grad_(False)
    payloads: list[torch.Tensor] = []

    def fake_broadcast(tensor, from_process):
        assert from_process == 0
        payloads.append(tensor.clone())
        return torch.full_like(tensor, 11.0)

    result = synchronize_optimizer_state(
        SimpleNamespace(num_processes=2),
        network,
        optimizer,
        broadcast_fn=fake_broadcast,
    )

    assert len(payloads) == 1
    assert network.trainable.tolist() == [11.0, 11.0]
    assert network.basis.tolist() == [11.0]
    assert network.frozen.tolist() == [8.0]
    assert result.parameter_count == 1
    assert result.buffer_count == 1


def test_distributed_prepare_registers_network_without_ddp_wrapper(monkeypatch):
    network = torch.nn.Linear(2, 1, bias=False)
    optimizer = _OptimizerView([network.weight])
    calls: list[tuple[torch.nn.Module, bool]] = []

    class FakeAccelerator:
        num_processes = 2

        def prepare_model(self, model, evaluation_mode):
            calls.append((model, evaluation_mode))
            return model

        def print(self, *args):
            return None

    monkeypatch.setattr(
        "library.training.gradient_sync.synchronize_optimizer_state",
        lambda *_args, **_kwargs: SimpleNamespace(
            parameter_count=1, buffer_count=0, bucket_count=1
        ),
    )

    prepared = prepare_network_for_manual_gradient_sync(
        FakeAccelerator(), network, optimizer
    )

    assert prepared is network
    assert calls == [(network, True)]
    assert not isinstance(prepared, torch.nn.parallel.DistributedDataParallel)


def test_bootstrap_multi_process_prepares_optimizer_without_wrapping_network(
    monkeypatch,
):
    events: list[tuple[str, object]] = []

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def prepare_grad_etc(self, text_encoder, unet):
            del text_encoder, unet
            events.append(("prepare_grad_etc", self))

    class FakeAccelerator:
        num_processes = 2
        device = torch.device("cpu")

        def prepare(self, *objects):
            events.append(("prepare", objects))
            return objects

        @staticmethod
        def unwrap_model(model):
            return model

    class FakeTrainer:
        @staticmethod
        def cast_unet(args):
            del args
            return False

        @staticmethod
        def cast_text_encoder(args):
            del args
            return False

        @staticmethod
        def get_text_encoders_train_flags(args, text_encoders):
            del args
            return [False] * len(text_encoders)

    network = Network()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    train_loader = object()
    val_loader = object()
    scheduler = object()
    unet = torch.nn.Linear(1, 1)

    def fake_prepare_network(accelerator, candidate, candidate_optimizer):
        assert accelerator.num_processes == 2
        assert candidate is network
        assert candidate_optimizer is optimizer
        events.append(("manual_sync_prepare", candidate))
        return candidate

    monkeypatch.setattr(
        bootstrap_mod,
        "prepare_network_for_manual_gradient_sync",
        fake_prepare_network,
    )
    result = TrainingBootstrap().prepare_with_accelerator(
        FakeTrainer(),
        SimpleNamespace(
            full_fp16=False,
            full_bf16=False,
            gradient_checkpointing=False,
        ),
        FakeAccelerator(),
        network,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        unet,
        text_encoders=[],
        text_encoder=None,
        vae=None,
        vae_dtype=torch.float32,
        weight_dtype=torch.float32,
        train_unet=False,
        train_text_encoder=False,
        cache_latents=True,
    )

    prepared_objects = next(value for name, value in events if name == "prepare")
    assert prepared_objects == (optimizer, train_loader, val_loader, scheduler)
    assert network not in prepared_objects
    assert result.network is network
    assert result.training_model is network
    assert any(name == "manual_sync_prepare" for name, _value in events)


def test_training_step_syncs_after_adapter_hook_and_before_stats_clip_step(
    monkeypatch,
):
    events: list[str] = []
    parameter = torch.nn.Parameter(torch.ones(1))

    class Network(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = parameter

        def clear_step_caches(self):
            events.append("clear")

        def capture_up_grad_stats(self):
            events.append("capture")

        def get_trainable_params(self):
            return [self.weight]

    class Accelerator:
        sync_gradients = True
        mixed_precision = "no"
        device = torch.device("cpu")

        @staticmethod
        def accumulate(model):
            del model
            from contextlib import nullcontext

            return nullcontext()

        @staticmethod
        def unwrap_model(model):
            return model

        @staticmethod
        def backward(loss):
            del loss
            events.append("backward")

        @staticmethod
        def clip_grad_norm_(params, max_norm):
            assert list(params) == [parameter]
            assert max_norm == 1.0
            events.append("clip")

    class Optimizer:
        param_groups = [{"params": [parameter]}]

        @staticmethod
        def step():
            events.append("optimizer")

        @staticmethod
        def zero_grad(set_to_none):
            assert set_to_none is True
            events.append("zero")

    class Scheduler:
        @staticmethod
        def step():
            events.append("scheduler")

    trainer = SimpleNamespace(
        memory_probe=None,
        peak_probe=None,
        _cudagraph_mark_step=False,
        _state=SimpleNamespace(personalization_observer={}),
        on_step_start=lambda *_args, **_kwargs: events.append("step_start"),
        process_batch=lambda *_args, **_kwargs: (
            events.append("forward") or torch.tensor(1.0, requires_grad=True)
        ),
        run_after_backward=lambda *_args, **_kwargs: events.append("after_hook"),
    )
    network = Network()
    state = SimpleNamespace(
        args=SimpleNamespace(
            max_grad_norm=1.0,
            log_every_n_steps=1,
            max_train_steps=1,
        ),
        accelerator=Accelerator(),
        network=network,
        training_model=network,
        optimizer=Optimizer(),
        lr_scheduler=Scheduler(),
        text_encoder=None,
        unet=None,
        train_ctx=object(),
        on_step_start_for_network=lambda *_args: None,
        profile_started=False,
        global_step=0,
        is_tracking=True,
    )
    monkeypatch.setattr(
        training_loop,
        "synchronize_optimizer_gradients",
        lambda *_args: events.append("sync"),
    )
    monkeypatch.setattr(training_loop, "debug_finite_enabled", lambda _args: True)
    monkeypatch.setattr(
        training_loop,
        "check_loss_finite",
        lambda *_args, **_kwargs: events.append("loss_finite"),
    )
    monkeypatch.setattr(
        training_loop,
        "inspect_grad_ownership",
        lambda *_args, **_kwargs: (
            events.append("grads_finite") or GradFiniteness()
        ),
    )

    training_loop._run_step(trainer, state, batch=object())

    ordered = [
        "backward",
        "after_hook",
        "sync",
        "grads_finite",
        "capture",
        "clip",
        "optimizer",
        "scheduler",
        "zero",
    ]
    assert [event for event in events if event in ordered] == ordered


class _DistributedAccelerator:
    def __init__(self, world_size: int) -> None:
        self.num_processes = world_size

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        assert reduction == "mean"
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced.div_(self.num_processes)
        return reduced


def _gloo_gradient_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        shared = torch.nn.Parameter(torch.tensor([10.0, 20.0]))
        conditional = torch.nn.Parameter(torch.tensor([30.0]))
        globally_unused = torch.nn.Parameter(torch.tensor([40.0]))
        shared.grad = (
            torch.tensor([1.0, 3.0])
            if rank == 0
            else torch.tensor([5.0, 7.0])
        )
        if rank == 1:
            conditional.grad = torch.tensor([4.0])

        optimizer = torch.optim.SGD(
            [shared, conditional, globally_unused], lr=0.1, weight_decay=0.5
        )
        result = synchronize_optimizer_gradients(
            _DistributedAccelerator(world_size), optimizer
        )
        optimizer.step()

        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "shared_grad": shared.grad.tolist(),
                    "conditional_grad": conditional.grad.tolist(),
                    "unused_grad_is_none": globally_unused.grad is None,
                    "shared": shared.tolist(),
                    "conditional": conditional.tolist(),
                    "unused": globally_unused.tolist(),
                    "materialized": result.materialized_gradient_count,
                }
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.focused
def test_two_process_gloo_syncs_conditional_gradients_and_optimizer_steps(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    world_size = 2
    init_method = f"file://{tmp_path / 'gloo-init'}"
    mp.spawn(
        _gloo_gradient_worker,
        args=(world_size, init_method, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    results = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    for result in results:
        assert result["shared_grad"] == pytest.approx([3.0, 5.0])
        assert result["conditional_grad"] == pytest.approx([2.0])
        assert result["unused_grad_is_none"] is True
        # Weight decay applies only to parameters with a global gradient.
        assert result["shared"] == pytest.approx([9.2, 18.5])
        assert result["conditional"] == pytest.approx([28.3])
        assert result["unused"] == pytest.approx([40.0])
    assert [result["materialized"] for result in results] == [1, 0]


class _ProbeAdapter(torch.nn.Module):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor([float(rank + 1)]))
        self.conditional = torch.nn.Parameter(torch.tensor([float(rank + 3)]))
        self.unused = torch.nn.Parameter(torch.tensor([float(rank + 5)]))
        self.register_buffer("basis", torch.tensor([float(rank + 7)]))


def _accelerate_probe_main() -> None:
    """Tiny NCCL probe used manually; it never loads the Anima base model."""
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision="no")
    if accelerator.num_processes != 2:
        raise RuntimeError("probe requires exactly two accelerator processes")
    try:
        network = _ProbeAdapter(accelerator.process_index)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        network = prepare_network_for_manual_gradient_sync(
            accelerator, network, optimizer
        )
        optimizer = accelerator.prepare(optimizer)
        async_sync = getattr(network, "_anima_async_gradient_sync", None)
        if async_sync is None:
            raise RuntimeError("async gradient synchronizer was not installed")
        async_sync.begin_step()

        loss = network.shared.sum() * float(accelerator.process_index + 1)
        if accelerator.process_index == 1:
            loss = loss + network.conditional.sum() * 4.0
        accelerator.backward(loss)
        result = async_sync.finish_step()
        optimizer.step()

        local = torch.cat(
            [network.shared, network.conditional, network.unused, network.basis]
        )
        gathered = accelerator.gather(local).reshape(accelerator.num_processes, -1)
        expected = local.new_tensor([0.85, 2.8, 5.0, 7.0])
        if not torch.allclose(
            gathered, expected.expand_as(gathered), atol=1e-6, rtol=0
        ):
            raise AssertionError(
                f"distributed probe mismatch: {gathered.cpu().tolist()}"
            )
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "backend": str(accelerator.distributed_type),
                        "world_size": accelerator.num_processes,
                        "values": gathered.cpu().tolist(),
                        "reduced_parameters": result.reduced_parameter_count,
                    }
                )
            )
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    _accelerate_probe_main()
