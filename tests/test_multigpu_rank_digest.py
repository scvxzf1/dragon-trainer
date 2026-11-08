"""Per-rank commit-boundary digests: group R (R-a/R-b/R-c) becomes judgeable.

t13 could not claim any of group R, because a state directory only carries rank 0's
``model.safetensors`` and one ``scaler.pt``: there was nothing to compare pairwise.
These tests pin the observation that fixes that, plus the two properties that make
it safe to leave on in a real run: no collective, and commit-boundary cost only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training.multigpu import rank_digest


def _module(scale=1.0, seed=0):
    """A tiny adapter-shaped module: named parameters, deterministic values."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    if scale != 1.0:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(scale)
    return model


def _digest(module, *, rank=0, step=10, epoch=1):
    return rank_digest.build(
        rank=rank, global_step=step, epoch=epoch, adapter=module
    )


def _write_all(directory, digests):
    for digest in digests:
        rank_digest.write(digest, str(directory))


def test_identical_ranks_are_equal(tmp_path):
    _write_all(tmp_path, [_digest(_module(), rank=0), _digest(_module(), rank=1)])

    summary = rank_digest.summarize_state_dir(str(tmp_path), 2)

    assert summary["equal"] is True
    assert summary["complete"] is True
    assert summary["codes"] == []
    assert sorted(summary["ranks"]) == ["0", "1"]
    assert summary["ranks"]["0"]["adapter"]["aggregate"] == summary["ranks"]["1"][
        "adapter"
    ]["aggregate"]
    # The per-tensor map stays in the file, not in the bundle-sized summary.
    assert "tensors" not in summary["ranks"]["0"]["adapter"]
    assert (tmp_path / rank_digest.FILE.format(rank=0)).is_file()


def test_one_changed_element_is_a_named_mismatch(tmp_path):
    left = _module()
    right = _module()
    with torch.no_grad():
        right[0].weight[0, 0].add_(1e-3)  # a single element, on one rank only

    _write_all(tmp_path, [_digest(left, rank=0), _digest(right, rank=1)])
    summary = rank_digest.summarize_state_dir(str(tmp_path), 2)

    assert summary["equal"] is False
    assert rank_digest.CODE_DIGEST_MISMATCH in summary["codes"]
    assert rank_digest.CODE_DIGEST_MISSING not in summary["codes"]
    assert summary["differences"] == ["adapter"]
    assert summary["ranks"]["0"]["adapter"]["aggregate"] != summary["ranks"]["1"][
        "adapter"
    ]["aggregate"]


def test_a_missing_rank_is_named_and_not_silent(tmp_path):
    _write_all(tmp_path, [_digest(_module(), rank=0)])

    summary = rank_digest.summarize_state_dir(str(tmp_path), 2)

    assert summary["complete"] is False
    assert summary["equal"] is False
    assert rank_digest.CODE_DIGEST_MISSING in summary["codes"]
    assert summary["missing"] == [1]


def test_tensor_order_does_not_change_the_hash():
    """Same contents, different iteration order: the hash must not notice."""
    tensors = {
        "b": torch.arange(4, dtype=torch.float32),
        "a": torch.ones(3),
        "c": torch.zeros(2),
    }
    reordered = {
        "c": torch.zeros(2),
        "a": torch.ones(3),
        "b": torch.arange(4, dtype=torch.float32),
    }

    first = rank_digest.adapter_digest(tensors)
    second = rank_digest.adapter_digest(reordered)

    assert first["aggregate"] == second["aggregate"]
    assert first["tensors"] == second["tensors"]
    left = _digest_of_dicts(tensors)
    right = _digest_of_dicts(reordered)
    assert rank_digest.compare({"0": left, "1": right}, 2)["equal"] is True


def _digest_of_dicts(tensors):
    return rank_digest.build(rank=0, global_step=1, adapter=tensors)


def test_the_digest_takes_no_collective_and_is_millisecond_cheap(monkeypatch):
    """Observation only: no collective, and a commit boundary pays for it once."""
    import torch.distributed as dist

    def _fail(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("the rank digest must not call a collective")

    for name in ("all_reduce", "all_gather", "all_gather_object", "barrier",
                 "broadcast", "reduce", "gather", "all_to_all"):
        monkeypatch.setattr(dist, name, _fail, raising=False)

    torch.manual_seed(0)
    modules = [torch.nn.Linear(64, 64) for _ in range(120)]  # ~2 MB of weights
    model = torch.nn.ModuleList(modules)

    started = time.perf_counter()
    digest = rank_digest.build(rank=0, global_step=10, adapter=model)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert digest["adapter"]["count"] == 240  # 120 layers × (weight, bias)
    assert elapsed_ms < 200.0, f"digest took {elapsed_ms:.1f} ms"
    print(f"adapter digest of 240 tensors took {elapsed_ms:.2f} ms")


def test_scaler_optimizer_scheduler_and_window_are_recorded():
    # AdamW is the production-shaped optimizer: its state carries the step counter
    # (a plain momentum-free SGD has no ``state`` at all, so there is nothing to
    # report and the digest says ``None`` instead of inventing a number).
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=0.5)
    for _ in range(3):
        optimizer.zero_grad()
        optimizer.param_groups[0]["params"][0].grad = torch.ones(2)
        optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    class _Scaler:
        growth_tracker = 7

        def get_scale(self):
            return 4096.0

    window = SimpleNamespace(
        window_attempts=3,
        window_replays=1,
        window_discards=0,
        window_consecutive_discards=0,
        window_total_loss=1.25,
        window_num_steps=3,
        batches_consumed=12,
        batches_per_epoch=30,
        resume_skipped_batches=0,
    )

    digest = rank_digest.build(
        rank=1,
        global_step=10,
        epoch=2,
        adapter=_module(),
        scaler=_Scaler(),
        optimizer=optimizer,
        scheduler=scheduler,
        window=window,
    )

    assert digest["global_step"] == 10 and digest["epoch"] == 2 and digest["rank"] == 1
    assert digest["scaler"] == {
        "present": True,
        "type": "_Scaler",
        "scale": 4096.0,
        "growth_tracker": 7,
    }
    assert digest["optimizer"]["step"] == 3
    assert digest["scheduler"]["lrs"] == [0.5]
    assert digest["window"]["window_attempts"] == 3
    assert digest["window"]["window_discards"] == 0
    assert digest["window"]["batches_consumed"] == 12


def test_a_missing_scaler_is_marked_absent_not_guessed():
    digest = rank_digest.build(rank=0, global_step=1, adapter=_module())
    assert digest["scaler"] == {"present": False}
    assert digest["optimizer"]["step"] is None
    assert digest["scheduler"]["lrs"] is None
    assert digest["window"] == {}
    # An optimizer that keeps no counter is reported as None, never guessed.
    plain = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    assert rank_digest.optimizer_step(plain) is None


def test_the_gate_switches_the_digest_off():
    state = SimpleNamespace(
        accelerator=SimpleNamespace(process_index=0, scaler=None),
        network=_module(),
        optimizer=None,
        lr_scheduler=None,
        args=SimpleNamespace(),
    )
    assert rank_digest.build_from_loop_state(state, global_step=1) is not None
    assert rank_digest.enabled({}) is True  # default on
    assert (
        rank_digest.build_from_loop_state(
            state, global_step=1, env={rank_digest.ENV: "0"}
        )
        is None
    )
    assert rank_digest.enabled({}, {rank_digest.ENV: "off"}) is False
    assert rank_digest.enabled(SimpleNamespace(multigpu_rank_digest=False), {}) is False


def test_bfloat16_and_float16_tensors_hash_without_a_numpy_roundtrip():
    """bf16 has no numpy dtype: the byte view must still work and stay deterministic."""
    tensors = {
        "bf16": torch.arange(8, dtype=torch.bfloat16),
        "fp16": torch.arange(8, dtype=torch.float16),
    }
    first = rank_digest.adapter_digest(tensors)
    second = rank_digest.adapter_digest(dict(tensors))
    assert first == second


# ---------------------------------------------------------------------------
# 与 cursor artifact 同机制的端到端证据（真实双 Gloo 进程）
# ---------------------------------------------------------------------------


def _gloo_digest_worker(rank, world_size, init_file, out_dir):
    from types import SimpleNamespace as _NS

    import torch as _torch
    import torch.distributed as _dist

    from library.training.checkpoints import save_and_remove_state_stepwise

    _dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )

    class _Acc:
        num_processes = world_size
        process_index = rank
        is_main_process = rank == 0

        def __init__(self):
            self.device = _torch.device("cpu")
            self.state = _NS(distributed_type="MULTI_CPU")

        def reduce(self, tensor, reduction="sum"):
            out = tensor.clone()
            _dist.all_reduce(out, op=_dist.ReduceOp.SUM)
            return out / world_size if reduction == "mean" else out

        def wait_for_everyone(self):
            _dist.barrier()

        def save_state(self, output_dir):
            import random as _random

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / f"shard-{rank}.json").write_text(
                json.dumps({"rank": rank}), encoding="utf-8"
            )
            (Path(output_dir) / "train_state.json").write_text(
                json.dumps({"current_epoch": 1, "current_step": 10}), encoding="utf-8"
            )
            _torch.save(
                {
                    "step": 10,
                    "random_state": _random.getstate(),
                    "torch_manual_seed": _torch.get_rng_state(),
                },
                Path(output_dir) / f"random_states_{rank}.pkl",
            )

    args = SimpleNamespace(
        output_dir=out_dir,
        output_name="demo",
        save_every_n_steps=10,
        save_last_n_steps=None,
        save_last_n_steps_state=None,
    )
    outcome = {"rank": rank, "error": None}
    try:
        # Rank 1 holds one slightly different element: the case group R must catch.
        model = _module()
        if rank == 1:
            with _torch.no_grad():
                model[0].weight[0, 0].add_(1e-3)
        digest = rank_digest.build(
            rank=rank, global_step=10, epoch=1, adapter=model
        )
        save_and_remove_state_stepwise(
            args,
            _Acc(),
            10,
            set(),
            loader_cursor={
                "sampler_position": 10,
                "epoch": 1,
                "batches_per_epoch": 30,
            },
            rank_digest=digest,
        )
    except BaseException as exc:  # recorded, never swallowed
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        Path(out_dir, f"digest-{rank}.json").write_text(
            json.dumps(outcome), encoding="utf-8"
        )
        if _dist.is_initialized():
            _dist.destroy_process_group()


@pytest.mark.integration
def test_both_ranks_write_a_digest_and_rank0_summarizes_it(tmp_path):
    import torch.multiprocessing as mp

    from library.training.multigpu import resume_protocol

    mp.spawn(
        _gloo_digest_worker,
        args=(2, str(tmp_path / "gloo-init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    outcomes = [
        json.loads(Path(tmp_path, f"digest-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    # A rank difference is reported, not turned into a failed run.
    assert all(outcome["error"] is None for outcome in outcomes), outcomes

    published = tmp_path / "demo-step00000010-state"
    for rank in (0, 1):
        assert (published / rank_digest.FILE.format(rank=rank)).is_file()

    bundle = resume_protocol.read_bundle(str(published))
    section = bundle["extra"]["rank_digests"]
    assert section["schema"] == rank_digest.SCHEMA
    assert section["complete"] is True
    assert section["equal"] is False
    assert section["codes"] == [rank_digest.CODE_DIGEST_MISMATCH]
    assert section["differences"] == ["adapter"]
    assert sorted(section["ranks"]) == ["0", "1"]
    # The per-rank digest files are owned per rank, like the cursor artifacts.
    assert rank_digest.FILE.format(rank=1) in bundle["rank_local_keys"]["1"]


# ---------------------------------------------------------------------------
# F-T18: 真实 GradScaler 的 growth_tracker（R-b），以及四轴判定
# ---------------------------------------------------------------------------


def test_a_real_stepped_gradscaler_reports_a_growth_tracker(tmp_path):
    """The real ``torch.amp.GradScaler`` keeps it in ``_growth_tracker``.

    Reading only the old public attribute made the digest report ``None`` for a
    genuinely-stepped scaler, i.e. R-b had nothing to compare.
    """
    def _scaled_state(rank):
        scaler = torch.amp.GradScaler("cpu", enabled=True)
        parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            loss = scaler.scale((parameter * 2).sum())
            loss.backward()
            scaler.step(optimizer)
            scaler.update()
        assert not hasattr(scaler, "growth_tracker")  # gone in current torch
        return scaler, parameter

    left_scaler, left_parameter = _scaled_state(0)
    right_scaler, right_parameter = _scaled_state(1)

    left = rank_digest.scaler_state(left_scaler)
    right = rank_digest.scaler_state(right_scaler)

    assert isinstance(left["growth_tracker"], int) and left["growth_tracker"] is not None
    assert left["growth_tracker"] == right["growth_tracker"]
    assert left["scale"] == right["scale"] and left["present"] is True

    _write_all(
        tmp_path,
        [
            rank_digest.build(rank=0, global_step=1, adapter=left_parameter,
                              scaler=left_scaler),
            rank_digest.build(rank=1, global_step=1, adapter=right_parameter,
                              scaler=right_scaler),
        ],
    )
    summary = rank_digest.summarize_state_dir(str(tmp_path), 2)
    assert summary["scaler_equal"] is True
    assert summary["equal"] is True


def test_the_four_axes_are_reported_separately(tmp_path):
    left = _module()
    right = _module()
    with torch.no_grad():
        right[0].weight[0, 0].add_(1e-3)

    _write_all(tmp_path, [_digest(left, rank=0), _digest(right, rank=1)])
    summary = rank_digest.summarize_state_dir(str(tmp_path), 2)

    assert summary["axes"] == {
        "adapter": False,
        "scaler": True,
        "optimizer": True,
        "window": True,
    }
    assert summary["adapter_equal"] is False
    assert summary["scaler_equal"] is True
    assert summary["optimizer_equal"] is True
    assert summary["window_equal"] is True
    assert summary["position_equal"] is True
    assert summary["differences"] == ["adapter"]


def test_rank_local_window_losses_do_not_raise_a_mismatch(tmp_path):
    """``window_total_loss`` is a local expectation, not a cross-rank invariant."""
    left = SimpleNamespace(window_total_loss=0.71, window_num_steps=2,
                           window_attempts=2, window_replays=0, window_discards=0)
    right = SimpleNamespace(window_total_loss=0.73, window_num_steps=2,
                            window_attempts=2, window_replays=0, window_discards=0)

    _write_all(
        tmp_path,
        [
            rank_digest.build(rank=0, global_step=1, adapter=_module(), window=left),
            rank_digest.build(rank=1, global_step=1, adapter=_module(), window=right),
        ],
    )
    summary = rank_digest.summarize_state_dir(str(tmp_path), 2)

    assert summary["equal"] is True, summary["message"]
    assert summary["window_equal"] is True  # counters match; losses are observation
    assert summary["codes"] == []


def test_a_window_counter_gap_is_still_a_mismatch(tmp_path):
    left = SimpleNamespace(window_attempts=2, window_discards=0)
    right = SimpleNamespace(window_attempts=3, window_discards=1)

    _write_all(
        tmp_path,
        [
            rank_digest.build(rank=0, global_step=1, adapter=_module(), window=left),
            rank_digest.build(rank=1, global_step=1, adapter=_module(), window=right),
        ],
    )
    summary = rank_digest.summarize_state_dir(str(tmp_path), 2)

    assert summary["equal"] is False
    assert summary["window_equal"] is False
    assert rank_digest.CODE_DIGEST_MISMATCH in summary["codes"]
    assert summary["differences"] == ["window"]


def test_offline_rb_from_a_loaded_scaler_pt(tmp_path):
    """R-b can be judged offline: ``scaler.pt`` is a state_dict mapping.

    The real files carry ``_growth_tracker`` (not the public ``growth_tracker``),
    e.g. ``{"scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
    "growth_interval": 2000, "_growth_tracker": 10}`` after 10 committed steps at
    fp16 — reading the old key reported ``None`` and left R-b with nothing to
    compare.
    """
    state_dict = {
        "scale": 65536.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": 10,
    }
    path = tmp_path / "scaler.pt"
    torch.save(state_dict, path)

    offline = rank_digest.scaler_state(torch.load(path, map_location="cpu"))

    assert offline == {
        "present": True,
        "type": "scaler_state_dict",
        "scale": 65536.0,
        "growth_tracker": 10,
    }
    # The legacy public key is still honoured when that is all a file has.
    legacy = rank_digest.scaler_state({"scale": 8.0, "growth_tracker": 3})
    assert legacy["growth_tracker"] == 3
