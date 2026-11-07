"""多进程窗口/scaler 协议的共享夹具。

供 ``tests/test_multigpu_window_protocol.py``（本批实现）与后续
``tests/test_multigpu_reference_hot.py`` 复用。

设计边界（诚实说明，避免把 CPU/Gloo 证据当成 CUDA 证据）：

* **真实的部分**：Gloo 进程组与 ``all_reduce``（协议集体是真正的跨进程调用）、
  真 ``torch.amp.GradScaler("cpu")`` 的 scale / growth tracker / backoff 状态机、
  生产 ``library/training/loop.py::_run_epoch_steps`` 与 ``_run_microbatch`` 的
  决策链、真 ``nn.Linear`` + 真 ``torch.optim`` 的参数更新。
* **替身的部分**：accelerate 包装层（``prepare``/``AcceleratedOptimizer``）用一个
  行为等价的小 shim 代替，因为其内部只是一层 if 判定；Anima 基座与 CUDA 路径
  不在 CPU 层验证范围（属 V-CUDA-SMALL / V-REAL）。
* **梯度归约默认不验证**：`patch_loop_for_fixture` 默认把
  ``synchronize_optimizer_gradients`` 换成 no-op，因为默认用例只关心窗口/scaler 决策。
  需要证明"参数一致真的依赖归约"时，必须显式传 ``real_gradient_sync=True``
  （此时不替换该函数、并让两 rank 的局部 loss 缩放不同）；真实梯度归约面另有
  ``tests/test_distributed_gradient_sync.py`` 覆盖，本轮新增的 Gloo 用例只是补充
  端到端证据，不替代它。
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def gloo_world(world_size: int) -> Iterator[int]:
    """在本进程内建立一个 Gloo 进程组（端口固定到本 rank 才 init）。

    与 ``tests/test_distributed_gradient_sync.py`` 的 ``file://`` 初始化保持同一
    风格，避免引入第二套约定。
    """
    port = free_port()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group(backend="gloo", rank=0, world_size=world_size)
    try:
        yield world_size
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ScalerOptimizerShim:
    """``AcceleratedOptimizer`` 的最小行为替身（真 scaler、真内层 optimizer）。

    与 accelerate 1.13.0 ``accelerator/optimizer.py:161-180`` 同序：
    ``scaler.step(optimizer)`` → ``scaler.update()``，并在非边界处为 no-op。
    """

    def __init__(self, accelerator, optimizer) -> None:
        self.accelerator = accelerator
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.step_calls = 0
        self.applied_steps = 0

    def step(self, *_args, **_kwargs):
        if not self.accelerator.sync_gradients:
            return
        self.step_calls += 1
        inner = self.optimizer
        before = getattr(inner, "_step_counter", 0)
        self.accelerator.scaler.step(inner)
        self.accelerator.scaler.update()
        after = getattr(inner, "_step_counter", 0)
        self.accelerator.optimizer_step_was_skipped = after == before
        if after != before:
            self.applied_steps += 1

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)


class _GradientState:
    """与 ``tests/window_replay_harness.py::FakeGradientState`` 同形的梯度状态。

    ``_set_sync_gradients`` 是生产 loop 的 ``_forced_accumulate`` 用来钉住重放
    边界的接口；缺了它重放路径会退化并在尾部窗口多提交一次。
    """

    def __init__(self, num_steps: int = 1) -> None:
        self.num_steps = num_steps
        self.step = 0
        self.sync_gradients = False
        self.end_of_dataloader = True

    def _set_sync_gradients(self, value: bool) -> None:
        self.sync_gradients = bool(value)


class CountingSGD(torch.optim.SGD):
    """记录真正应用步数的 SGD（用来判断某 rank 是否真的改了权重）。"""

    def step(self, closure=None):
        result = super().step(closure)
        self._step_counter = getattr(self, "_step_counter", 0) + 1
        return result


class LoopAccelerator:
    """``accelerate.Accelerator`` 的行为等价 shim，集体走真 Gloo。"""

    def __init__(self, *, world_size: int, scaler: torch.amp.GradScaler | None) -> None:
        self.num_processes = world_size
        self.process_index = dist.get_rank()
        self.device = torch.device("cpu")
        self.mixed_precision = "fp16" if scaler is not None else "no"
        self.scaler = scaler
        self.use_distributed = world_size > 1
        self.optimizer_step_was_skipped = False
        self.events: list[str] = []
        self._accum_keys: list[int] = []
        self.gradient_state = _GradientState(num_steps=1)

    # --- accelerate surface used by the production loop -------------------
    @property
    def sync_gradients(self) -> bool:
        return bool(self.gradient_state.sync_gradients)

    def accumulate(self, _model):
        state = self.gradient_state

        @contextmanager
        def _cm():
            state.step = getattr(state, "step", 0) + 1
            state.sync_gradients = state.step % max(1, state.num_steps) == 0
            yield

        return _cm()

    def no_sync(self, _model):
        return _null()

    def unwrap_model(self, model):
        return model

    def backward(self, loss):
        # 与 accelerate ``Accelerator.backward`` 同序：有 scaler 时先 scale 再反传，
        # 否则 ``scaler.step`` 会因为 ``_scale is None`` 直接报错。
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(list(params), max_norm)

    def reduce(self, tensor, reduction="mean"):
        """真 Gloo 归约：``sum`` 精确实现，``mean`` 除 world size。

        刻意不放宽成"任意 reduction 都当 sum"——accelerate 的该行为是缺陷面，
        不在这里复刻。
        """
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        if reduction == "mean":
            out /= self.num_processes
        return out

    def wait_for_everyone(self):
        dist.barrier()

    def skip_first_batches(self, loader, count):
        raise AssertionError("initial_step must be 0 in these fixtures")


@contextmanager
def _null():
    yield


def build_loop_state(
    *,
    accelerator: LoopAccelerator,
    max_train_steps: int,
    batches: int,
    accum: int,
    loss_plan: Callable[[int], float],
    loss_scale: float = 1.0,
    model: torch.nn.Module,
    save_snapshots: bool = False,
    unused_parameter: bool = False,
) -> tuple[Any, Any, torch.optim.Optimizer]:
    """搭一个能驱动生产 loop 的最小 state/trainer。"""
    if unused_parameter:
        # 参与 optimizer 但从不参与 forward → 边界处 grad 恒为 None，
        # 用来覆盖"presence/None 不得被协议改写"的路径。
        model.register_parameter("unused_probe", torch.nn.Parameter(torch.zeros(2)))
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    accelerator.gradient_state.num_steps = accum
    optimizer = ScalerOptimizerShim(accelerator, optimizer)
    counter = {"calls": 0}

    def process_batch(*_args, **_kwargs):
        value = loss_plan(counter["calls"])
        counter["calls"] += 1
        base = model(torch.ones(1, 2)).square().mean()
        # 数值精确给定、计算图保留：`* 0.0` 让梯度保持有限，
        # 而 `+ value` 允许 value 为 nan → “坏 loss + 有限梯度”。
        # ``base * loss_scale`` keeps the graph real; 让两 rank 的局部梯度不同，
        # 只有真归约才能让参数重新一致。
        return base * float(loss_scale) + float(value)

    trainer = SimpleNamespace(
        memory_probe=None,
        peak_probe=None,
        _cudagraph_mark_step=False,
        _state=SimpleNamespace(personalization_observer={}),
        on_step_start=lambda *_a, **_k: None,
        process_batch=process_batch,
        run_after_backward=lambda *_a, **_k: None,
    )
    state = SimpleNamespace(
        args=SimpleNamespace(
            max_grad_norm=0.0,
            log_every_n_steps=1,
            max_train_steps=max_train_steps,
            stage_schedule=None,
            stage_schedule_enabled=False,
            _stage_num_update_steps_per_epoch=0,
            gradient_accumulation_steps=accum,
            max_nonfinite_window_discards=3,
        ),
        accelerator=accelerator,
        network=model,
        training_model=model,
        optimizer=optimizer,
        lr_scheduler=SimpleNamespace(step=lambda: None),
        text_encoder=None,
        unet=None,
        train_ctx=object(),
        on_step_start_for_network=lambda *_a: None,
        profile_started=False,
        profile_range=None,
        global_step=0,
        initial_step=0,
        is_tracking=False,
        stage_index=-1,
        current_step=SimpleNamespace(value=0),
        train_dataloader=[object() for _ in range(batches)],
        progress_bar=SimpleNamespace(update=lambda n: None),
        saver=SimpleNamespace(
            maybe_save_step=lambda *_a, **_k: None,
        ),
        optimizer_train_fn=lambda: None,
        nonfinite_grad_skips=0,
        train_dataset_group=None,
        _save_snapshots=save_snapshots,
    )
    return state, trainer, optimizer


def patch_loop_for_fixture(
    loop_module, *, snapshots: list | None = None, real_gradient_sync: bool = False
) -> dict:
    """静音生产 loop 里与协议无关的旁路（日志/采样/自适应），返回原值用于还原。

    ``real_gradient_sync=True`` 保留真实的 ``synchronize_optimizer_gradients``，
    让"两 rank 参数逐位一致"真正依赖 Gloo 归约而不是被 no-op 掩盖。
    """
    originals: dict[str, Any] = {}
    quiet = {
        "_maybe_apply_stage_schedule": lambda *_a, **_k: None,
        "_record_recent_step_seconds": lambda *_a, **_k: None,
        "_sample_at_step": lambda *_a, **_k: None,
        "_log_step": lambda *_a, **_k: None,
        "_maybe_run_step_validation": lambda *_a, **_k: None,
        "_maybe_auto_block_swap": lambda *_a, **_k: None,
        "_maybe_scale_norm": lambda _state: (None, None, None, {}),
        "precision_observe": lambda *_a, **_k: None,
        "adaptive_notify": lambda *_a, **_k: None,
    }
    if not real_gradient_sync:
        quiet["synchronize_optimizer_gradients"] = lambda *_a: None
    for name, replacement in quiet.items():
        originals[name] = getattr(loop_module, name, None)
        setattr(loop_module, name, replacement)
    return originals


def restore_loop(loop_module, originals: dict) -> None:
    for name, value in originals.items():
        if value is not None:
            setattr(loop_module, name, value)


# ---------------------------------------------------------------------------
# S-0：single/dual permutation 一致性（t3 F-T3-1a）
# ---------------------------------------------------------------------------


class _SlotDataset(torch.utils.data.Dataset):
    """Dataset whose samples carry their own index, so a stream is readable."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {"value": torch.tensor(int(index))}


def _slot_collate(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def _slot_loader(size: int, batch_size: int) -> torch.utils.data.DataLoader:
    """The production construction: shuffle, zero workers (``bootstrap.py``)."""
    return torch.utils.data.DataLoader(
        _SlotDataset(size),
        shuffle=True,
        batch_size=int(batch_size),
        collate_fn=_slot_collate,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))


def loader_stream_single(
    size: int, batch_size: int, seed: int, out_path: str, gate_args=None
) -> None:
    """One worker, one freshly prepared loader; the stream is written to ``out_path``.

    Every case runs in its own process on purpose: accelerate's multi-process branch
    consumes global RNG when it prepares a loader, so a second preparation inside
    the same process would compare two different permutations.
    """
    import json
    from pathlib import Path

    from accelerate import Accelerator

    from library.training.multigpu import loader_seed

    from types import SimpleNamespace

    _seed_everything(seed)
    # The production args surface the gate reads: ``--seed`` plus the flag.
    gate_args = gate_args or SimpleNamespace(
        seed=int(seed), multigpu_seedable_sampler=False
    )
    kwargs = loader_seed.dataloader_config_kwargs(gate_args, os.environ)
    from accelerate import DataLoaderConfiguration

    acc = Accelerator(
        cpu=True,
        dataloader_config=DataLoaderConfiguration(**kwargs) if kwargs else None,
    )
    loader = acc.prepare_data_loader(_slot_loader(size, batch_size))
    stream = [[int(value) for value in batch["value"].tolist()] for batch in loader]
    Path(out_path).write_text(
        json.dumps({"num_processes": int(acc.num_processes), "batches": stream}),
        encoding="utf-8",
    )


def loader_stream_rank(
    rank: int, world_size: int, size: int, batch_size: int, seed: int, port: int, out_dir: str
) -> None:
    """One rank of a real two-process Gloo run; writes ``rank<N>.json``."""
    import json
    from datetime import timedelta
    from pathlib import Path

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(int(port))

    from accelerate import Accelerator, DataLoaderConfiguration, InitProcessGroupKwargs

    from library.training.multigpu import loader_seed

    from types import SimpleNamespace

    _seed_everything(seed)
    gate_args = SimpleNamespace(seed=int(seed), multigpu_seedable_sampler=False)
    payload = {"rank": rank, "batches": [], "num_processes": 0, "error": None}
    try:
        kwargs = loader_seed.dataloader_config_kwargs(gate_args, os.environ)
        acc = Accelerator(
            cpu=True,
            kwargs_handlers=[
                InitProcessGroupKwargs(backend="gloo", timeout=timedelta(seconds=120))
            ],
            dataloader_config=DataLoaderConfiguration(**kwargs) if kwargs else None,
        )
        payload["num_processes"] = int(acc.num_processes)
        loader = acc.prepare_data_loader(_slot_loader(size, batch_size))
        for local_index, batch in enumerate(loader):
            payload["batches"].append(
                {
                    "local_index": local_index,
                    "values": [int(value) for value in batch["value"].tolist()],
                }
            )
    except BaseException as exc:  # recorded, not swallowed
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        Path(out_dir, f"rank{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except BaseException:
            pass
