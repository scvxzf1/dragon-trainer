"""Distributed checkpoint publish protocol (t5, contract §8 P1-1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from library.training.multigpu import checkpoint_protocol as ckpt


class _FakeAccelerator:
    """Minimal accelerate stand-in: only the surface the protocol uses."""

    def __init__(self, *, index: int = 0, world: int = 1, fail_save: bool = False):
        self.num_processes = world
        self.process_index = index
        self.is_main_process = index == 0
        self.device = "cpu"
        self.fail_save = fail_save
        self.barriers = 0
        self.reductions: list[float] = []

    def save_state(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"shard-{self.process_index}.json").write_text(
            json.dumps({"rank": self.process_index}), encoding="utf-8"
        )
        (Path(output_dir) / "train_state.json").write_text(
            json.dumps({"current_step": 5}), encoding="utf-8"
        )
        if self.fail_save:
            raise RuntimeError("boom")

    def wait_for_everyone(self):
        self.barriers += 1

    def reduce(self, tensor, reduction="sum"):
        # Emulates a *sum* all-reduce of the same value on every rank: keeps the
        # OR verdict identical on both ranks, which is what the real one does.
        self.reductions.append(float(tensor.sum().item()))
        return tensor * self.num_processes


def _dirs(tmp_path):
    state = tmp_path / "state"
    return state, tmp_path / "state.tmp", tmp_path / "state.backup"


def test_publish_creates_state_and_drops_backup(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    tmp_dir.mkdir()
    (tmp_dir / "train_state.json").write_text("{}", encoding="utf-8")
    outcome = ckpt.publish_state_dir(str(state), str(tmp_dir), str(backup))
    assert outcome.published
    assert state.is_dir() and not tmp_dir.exists() and not backup.exists()


def test_publish_failure_restores_previous_state(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    state.mkdir()
    (state / "train_state.json").write_text('{"current_step": 1}', encoding="utf-8")
    # tmp missing -> publish must refuse and leave the old checkpoint intact
    outcome = ckpt.publish_state_dir(str(state), str(tmp_dir), str(backup))
    assert not outcome.published and outcome.error
    assert json.loads((state / "train_state.json").read_text())["current_step"] == 1
    assert not backup.exists()


def test_single_process_save_publishes_without_collectives(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    acc = _FakeAccelerator(world=1)
    ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))
    assert (state / "train_state.json").exists()
    assert acc.reductions == [] and acc.barriers == 0


def test_single_process_save_failure_cleans_tmp_and_keeps_old_state(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    state.mkdir()
    (state / "train_state.json").write_text('{"current_step": 1}', encoding="utf-8")
    acc = _FakeAccelerator(world=1, fail_save=True)
    with pytest.raises(RuntimeError, match="boom"):
        ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))
    assert not tmp_dir.exists()
    assert json.loads((state / "train_state.json").read_text())["current_step"] == 1


def test_non_main_rank_never_publishes(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    acc = _FakeAccelerator(index=1, world=2)
    ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))
    assert not state.exists()  # only rank 0 publishes
    assert (tmp_dir / "shard-1.json").exists()  # but every rank writes its shard
    # Two agreements, in the frozen order: the capture verdict, then the publish
    # verdict. A non-main rank reports "no failure" for both.
    assert acc.reductions == [0.0, 0.0]
    assert acc.barriers >= 2  # recovery barrier + post-publish barrier


def test_main_rank_publishes_and_agrees_the_result(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    acc = _FakeAccelerator(index=0, world=2)
    ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))
    assert (state / "train_state.json").exists()
    assert acc.reductions, "publish outcome must be agreed, not assumed"


def test_transport_error_is_not_negotiated(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    acc = _FakeAccelerator(index=0, world=2)

    def boom(_output_dir):
        raise RuntimeError("NCCL error: unhandled system error")

    acc.save_state = boom
    with pytest.raises(RuntimeError, match="NCCL"):
        ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))
    assert acc.reductions == []  # class II: no new collective on a broken group


def test_local_rank_failure_is_negotiated_on_multiprocess(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    acc = _FakeAccelerator(index=0, world=2, fail_save=True)
    with pytest.raises(ckpt.CheckpointPublishError) as excinfo:
        ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))
    assert ckpt.CODE_PUBLISH_FAILED in str(excinfo.value)
    assert acc.reductions  # healthy group -> agreed, not silent


def test_both_agreement_paths_send_the_same_fixed_keys(tmp_path, monkeypatch):
    """两条路径（capture 失败 / 发布）必须用同名同序的固定键集合。

    这是协议的一部分：``agree_any`` 按 mapping 顺序把布尔值打包成一个张量，
    若一侧发 ``{"save_state_failed"}`` 而另一侧发 ``{"publish_failed"}``，
    集体调用位置仍然对得上但含义已经分叉。
    """
    from library.training.multigpu import agreement as agreement_mod

    captured: list[tuple[str, ...]] = []

    def _record(accelerator, flags):
        captured.append(tuple(flags))
        assert set(flags) == set(ckpt.AGREEMENT_KEYS)
        return {key: any(bool(value) for value in flags.values()) for key in flags}

    monkeypatch.setattr(agreement_mod, "agree_any", _record)

    # 路径二：发布成功
    state, tmp_dir, backup = _dirs(tmp_path / "ok")
    (tmp_path / "ok").mkdir(exist_ok=True)
    ckpt.save_state_published(
        _FakeAccelerator(index=0, world=2), str(tmp_dir), str(state), str(backup)
    )

    # 路径一：capture 失败
    state_b, tmp_b, backup_b = _dirs(tmp_path / "fail")
    (tmp_path / "fail").mkdir(exist_ok=True)
    with pytest.raises(ckpt.CheckpointPublishError):
        ckpt.save_state_published(
            _FakeAccelerator(index=0, world=2, fail_save=True),
            str(tmp_b),
            str(state_b),
            str(backup_b),
        )

    # 成功路径两次（capture + publish），失败路径一次（capture 即中止）。
    assert len(captured) == 3, captured
    assert all(entry == ckpt.AGREEMENT_KEYS for entry in captured)  # 同名、同序


def _recording_accelerator(tmp_path, **kwargs):
    """Fake accelerator that records its protocol events in order."""
    import functools

    acc = _FakeAccelerator(**kwargs)
    events: list[str] = []
    reduce_, barrier_, save_ = acc.reduce, acc.wait_for_everyone, acc.save_state

    def reduce(tensor, reduction="sum"):
        events.append("agree")
        return reduce_(tensor, reduction)

    def wait_for_everyone():
        events.append("barrier")
        return barrier_()

    def save_state(output_dir):
        events.append("save")
        return save_(output_dir)

    acc.reduce = reduce
    acc.wait_for_everyone = wait_for_everyone
    acc.save_state = save_state
    return acc, events


def test_capture_verdict_is_agreed_before_the_publish_barrier(tmp_path):
    """The collective order is the fix for the hang (review t25).

    The capture agreement must be entered straight after this rank's own
    ``save_state`` — *not* after the pre-publish barrier. Entering it from the
    ``except`` block paired a failing rank's collective with its sibling's barrier
    and blocked the save until the process-group timeout.
    """
    state, tmp_dir, backup = _dirs(tmp_path)
    acc, events = _recording_accelerator(tmp_path, index=0, world=2)

    ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))

    assert events.count("save") == 1
    assert events.count("agree") == 2
    first_agree = events.index("agree")
    assert events[first_agree - 1] == "save"  # capture verdict, no barrier between
    assert events.index("agree", first_agree + 1) > events.index("save")


def test_capture_failure_is_agreed_before_any_barrier(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    acc, events = _recording_accelerator(tmp_path, index=0, world=2, fail_save=True)

    with pytest.raises(ckpt.CheckpointPublishError):
        ckpt.save_state_published(acc, str(tmp_dir), str(state), str(backup))

    # save -> agree (capture) directly: a barrier in between is exactly what hung.
    assert events[-2:] == ["save", "agree"] or events[-3:] == ["save", "agree", "barrier"]
    assert events.count("agree") == 1
    assert events.index("agree") == events.index("save") + 1


def test_agreement_flags_has_a_stable_order():
    flags = ckpt.agreement_flags(publish_failed=True)
    assert tuple(flags) == ckpt.AGREEMENT_KEYS == ("publish_failed", "save_state_failed")
    assert ckpt.agreement_failed({"publish_failed": False, "save_state_failed": True})
    assert ckpt.agreement_failed({"publish_failed": True, "save_state_failed": False})
    assert not ckpt.agreement_failed(
        {"publish_failed": False, "save_state_failed": False}
    )


def test_checkpoint_records_refuses_mismatched_positions(tmp_path):
    acc = _FakeAccelerator(index=0, world=2)
    # Fake reduce returns the *sum* of a value that differs per rank by design in
    # this stub, so an equal world must pass; a mismatch is covered by the Gloo
    # integration test below.
    assert ckpt.checkpoint_records(acc, global_step=7, epoch=1)["global_step"] == 7


# ---------------------------------------------------------------------------
# 真实双进程 Gloo 回归（t19 F2）：同一共享协议、全 rank 同序、有界退出
# ---------------------------------------------------------------------------


def _gloo_worker(rank, world_size, init_file, out_dir, plan):
    import json as _json
    import os as _os

    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    class _Acc:
        num_processes = world_size
        process_index = rank
        is_main_process = rank == 0
        device = "cpu"

        def __init__(self):
            self.reductions = 0

        def reduce(self, tensor, reduction="sum"):
            self.reductions += 1
            out = tensor.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            if reduction == "mean":
                out = out / world_size
            return out

        def wait_for_everyone(self):
            dist.barrier()

        def save_state(self, output_dir):
            _os.makedirs(output_dir, exist_ok=True)
            Path(output_dir, f"shard-{rank}.json").write_text(
                _json.dumps({"rank": rank}), encoding="utf-8"
            )
            Path(output_dir, "train_state.json").write_text(
                _json.dumps({"current_step": 3}), encoding="utf-8"
            )
            if plan.get("fail_save_rank") == rank:
                raise RuntimeError("local capture failed on rank")
            if plan.get("transport_all") or plan.get("transport_rank") == rank:
                raise RuntimeError("NCCL error: unhandled system error")

    acc = _Acc()
    state = Path(out_dir, "state")
    result = {"rank": rank, "outcome": "ok", "reductions": 0}
    try:
        if plan.get("mode") == "records_mismatch":
            ckpt.checkpoint_records(acc, global_step=rank, epoch=1)
        else:
            ckpt.save_state_published(
                acc, str(Path(out_dir, "state.tmp")), str(state), str(Path(out_dir, "state.backup"))
            )
    except BaseException as exc:  # recorded, not swallowed: the test asserts on it
        result["outcome"] = type(exc).__name__
        result["code"] = getattr(exc, "code", None)
        result["message"] = str(exc)
    finally:
        result["reductions"] = acc.reductions
        Path(out_dir, f"result-{rank}.json").write_text(
            __import__("json").dumps(result), encoding="utf-8"
        )
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_gloo(tmp_path, plan):
    import json as _json

    import torch.multiprocessing as mp

    init_file = str(tmp_path / "gloo-init")
    mp.spawn(_gloo_worker, args=(2, init_file, str(tmp_path), plan), nprocs=2, join=True)
    return [
        _json.loads(Path(tmp_path, f"result-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]


@pytest.mark.integration
def test_two_rank_publish_agrees_and_only_rank0_publishes(tmp_path):
    results = _run_gloo(tmp_path, {})
    assert all(result["outcome"] == "ok" for result in results), results
    published = tmp_path / "state"
    assert (published / "shard-0.json").exists() and (published / "shard-1.json").exists()
    assert not (tmp_path / "state.backup").exists()
    # Each rank entered exactly one agreement collective over a shared protocol.
    assert all(result["reductions"] >= 1 for result in results)


@pytest.mark.integration
def test_two_rank_local_capture_failure_exits_both_ranks(tmp_path):
    results = _run_gloo(tmp_path, {"fail_save_rank": 1})
    assert all(
        result["outcome"] == "CheckpointPublishError" for result in results
    ), results
    assert all(result["code"] == ckpt.CODE_PUBLISH_FAILED for result in results)
    assert not (tmp_path / "state").exists()  # nothing was published


@pytest.mark.integration
def test_two_rank_capture_failure_on_peer_also_stops_the_publisher(tmp_path):
    """发布者本身成功，但 peer 捕获失败 → 两 rank 仍同码退出（统一键集合生效）。"""
    results = _run_gloo(tmp_path, {"fail_save_rank": 1})
    assert all(
        result["outcome"] == "CheckpointPublishError" for result in results
    ), results
    assert not (tmp_path / "state").exists()  # 已发布目录被回滚


@pytest.mark.integration
def test_two_rank_state_mismatch_is_refused_by_both(tmp_path):
    results = _run_gloo(tmp_path, {"mode": "records_mismatch"})
    assert all(
        result["outcome"] == "CheckpointPublishError" for result in results
    ), results
    assert all(result["code"] == ckpt.CODE_RECORD_FAILED for result in results)


@pytest.mark.integration
def test_two_rank_transport_failure_adds_no_collective(tmp_path):
    """Class II: a broken transport must not be "agreed" — no new collective.

    Both ranks hit the transport failure, which is exactly the class-II situation
    (the process group is unusable): each rank re-raises the original error on the
    spot, so the run terminates instead of hanging in an agreement collective.
    """
    results = _run_gloo(tmp_path, {"transport_all": True})
    assert all(
        result["outcome"] == "RuntimeError" for result in results
    ), results  # original transport error, not a negotiated one
    assert all(result["reductions"] == 0 for result in results)  # no new collective


# ---------------------------------------------------------------------------
# 单进程：遗留 tmp/backup 与残留 shard 必须在发布前清理（t23）
# ---------------------------------------------------------------------------


def test_single_process_save_clears_leftovers_before_publishing(tmp_path):
    state, tmp_dir, backup = _dirs(tmp_path)
    # Leftovers from a crashed earlier run: stale tmp content, a stale backup with
    # old shards, and an old shard inside the current state dir.
    state.mkdir()
    (state / "shard-old.json").write_text("{}", encoding="utf-8")
    (state / "train_state.json").write_text('{"current_step": 1}', encoding="utf-8")
    tmp_dir.mkdir()
    (tmp_dir / "train_state.json").write_text('{"current_step": 9}', encoding="utf-8")
    backup.mkdir()
    (backup / "shard-stale.json").write_text("{}", encoding="utf-8")

    ckpt.save_state_published(
        _FakeAccelerator(world=1), str(tmp_dir), str(state), str(backup)
    )
    published = sorted(path.name for path in state.iterdir())
    assert published == ["shard-0.json", "train_state.json"]  # only this run
    assert "shard-old.json" not in published
    assert not tmp_dir.exists() and not backup.exists()


# ---------------------------------------------------------------------------
# 通过 save_checkpoint_state 驱动（不直接调 helper）的双进程 Gloo 用例
# ---------------------------------------------------------------------------


def _gloo_save_state_worker(rank, world_size, init_file, tmp_dir):
    import json as _json
    from types import SimpleNamespace

    import torch.distributed as dist

    from library.training.checkpoints import save_checkpoint_state

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    class _Acc:
        num_processes = world_size
        process_index = rank
        is_main_process = rank == 0

        def __init__(self):
            import torch as _torch

            # torch.device (not a namespace): agree_any passes it to torch.tensor.
            self.device = _torch.device("cpu")
            self.state = SimpleNamespace(distributed_type="MULTI_CPU")
            self.reductions = 0

        def reduce(self, tensor, reduction="sum"):
            self.reductions += 1
            out = tensor.clone()
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            if reduction == "mean":
                out = out / world_size
            return out

        def wait_for_everyone(self):
            dist.barrier()

        def save_state(self, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir, f"shard-{rank}.json").write_text(
                _json.dumps({"rank": rank}), encoding="utf-8"
            )
            Path(output_dir, "train_state.json").write_text(
                _json.dumps({"current_step": 5}), encoding="utf-8"
            )

    args = SimpleNamespace(output_dir=tmp_dir, output_name=None)
    result = {"rank": rank, "outcome": "ok", "code": None}
    try:
        # Deliberately disagree about the step: rank 1 claims a different position.
        save_checkpoint_state(args, _Acc(), None, global_step=5 + rank)
    except BaseException as exc:
        result["outcome"] = type(exc).__name__
        result["code"] = getattr(exc, "code", None)
        result["message"] = str(exc)
    finally:
        Path(tmp_dir, f"save-{rank}.json").write_text(_json.dumps(result), encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.integration
def test_two_rank_save_checkpoint_state_refuses_mismatched_step(tmp_path):
    import json as _json

    import torch.multiprocessing as mp

    init_file = str(tmp_path / "gloo-init")
    mp.spawn(
        _gloo_save_state_worker,
        args=(2, init_file, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    results = [
        _json.loads(Path(tmp_path, f"save-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert all(
        result["outcome"] == "CheckpointPublishError" for result in results
    ), results
    assert all(result["code"] == ckpt.CODE_RECORD_FAILED for result in results)
    assert not (tmp_path / "demo-checkpoint-state").exists()  # nothing published
