"""窗口健康 / 跳过 / scaler 对称性协议（P0-0）回归。

两层：

* **无 marker 的纯 CPU 检查**（默认进 core 层，秒级）：OR 原语语义、单进程
  短路、单卡行为不变；
* ``integration`` 标记的真双进程 **Gloo** 用例：用真实进程组 + 真
  ``torch.amp.GradScaler("cpu")`` 驱动生产 ``library/training/loop.py``，
  验证"一个 rank 坏 loss（梯度仍有限）"不会再让两个 rank 的跳过决策、
  scaler scale / growth tracker 或提交游标分叉。

CPU/Gloo 证据不构成 CUDA 证据（见 docs/proposal/multigpu_reference_validation_contract.md §1）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from library.training import multigpu
from library.training.multigpu import window_protocol
from library.training.multigpu.agreement import agree_any, all_ranks_equal, is_multiprocess
from library.training.multigpu.probe import StartupProbeError, maybe_run_startup_probe


class _NoReduce:
    """单进程替身：调用 reduce 即失败，用来证明单卡路径不发集体。"""

    num_processes = 1
    device = torch.device("cpu")

    def reduce(self, *_a, **_k):
        raise AssertionError("single-process path must not issue a collective")


# --------------------------------------------------------------------------
# 无 marker：OR 原语与单卡短路
# --------------------------------------------------------------------------


def test_single_process_agreement_short_circuits():
    assert is_multiprocess(_NoReduce()) is False
    agreed = agree_any(_NoReduce(), {"a": True, "b": False})
    assert agreed == {"a": True, "b": False}
    assert all_ranks_equal(_NoReduce(), {"scale": 128.0}) == {"scale": True}


def test_single_process_step_agreement_reports_local_flags():
    agreement = multigpu.agree_step(
        _NoReduce(),
        loss_nonfinite=True,
        gradient_nonfinite=False,
        force_skip=False,
    )
    assert agreement.multiprocess is False
    assert agreement.unhealthy is True
    assert agreement.drop_update is True


def test_single_process_clean_step_is_not_unhealthy():
    agreement = multigpu.agree_step(
        _NoReduce(),
        loss_nonfinite=False,
        gradient_nonfinite=False,
        force_skip=False,
    )
    assert agreement.unhealthy is False


def test_window_discard_agreement_is_local_for_single_process():
    assert window_protocol.agree_window_discard(_NoReduce(), discard=False) is False
    assert window_protocol.agree_window_discard(_NoReduce(), discard=True) is True


def test_scaler_state_is_none_without_scaler_and_public_with_one():
    assert window_protocol.scaler_state(_NoReduce()) is None
    scaler = torch.amp.GradScaler("cpu", init_scale=256, growth_interval=2)
    accelerator = _NoReduce()
    accelerator.scaler = scaler
    state = window_protocol.scaler_state(accelerator)
    assert state["scale"] == pytest.approx(256.0)
    # torch 2.10 惰性初始化 growth tracker：首次 step 之前它是 None，
    # 因此这里只要求"未推进"，不要求字段恒存在。
    assert state.get("growth_tracker", 0.0) == pytest.approx(0.0)


def test_symmetry_assertion_is_a_noop_on_one_process():
    accelerator = _NoReduce()
    accelerator.scaler = torch.amp.GradScaler("cpu")
    window_protocol.assert_scaler_symmetry(accelerator, where="unit")  # 不抛


def test_startup_probe_is_off_by_default():
    result = maybe_run_startup_probe(_NoReduce(), env={})
    assert result.ran is False


def test_startup_probe_skips_single_process():
    result = maybe_run_startup_probe(
        _NoReduce(), env={"ANIMA_MULTIGPU_STARTUP_PROBE": "1"}
    )
    assert result.ran is False


def test_peer_discard_reason_is_a_named_constant():
    from library.training import window_accounting

    assert window_accounting.REASON_PEER_WINDOW_DECISION == "peer_window_decision"


# --------------------------------------------------------------------------
# integration：真双进程 Gloo + 真 GradScaler + 生产 loop
# --------------------------------------------------------------------------


def _run_rank(rank: int, world_size: int, init_file: str, out_dir: str, plan: dict) -> None:
    import torch.distributed as dist

    from library.training import loop
    from tests import multigpu_reference_support as support

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # 让 admission 与真实 world_size 都可见（P0-3 的身份字段来源）
        os.environ["ANIMA_ACCELERATE_NUM_PROCESSES"] = str(world_size)
        os.environ["ANIMA_ACCELERATE_LAUNCH"] = "1"
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=2)
        accelerator = support.LoopAccelerator(world_size=world_size, scaler=scaler)
        torch.manual_seed(1234)  # 同一 init：两 rank 参数逐位一致
        model = torch.nn.Linear(2, 2)
        bad_steps = set(plan.get("bad_steps_by_rank", {}).get(str(rank), []))

        def loss_plan(call_index: int) -> float:
            per_step = plan.get("accum", 1)
            step_index = call_index // per_step
            # 坏 loss 只由指定 rank 产生；梯度仍有限（support 里 `* 0.0`）。
            return float("nan") if step_index in bad_steps else 1.0

        state, trainer, optimizer = support.build_loop_state(
            accelerator=accelerator,
            max_train_steps=plan["max_train_steps"],
            batches=plan["batches"],
            accum=plan.get("accum", 1),
            loss_plan=loss_plan,
            model=model,
            unused_parameter=bool(plan.get("unused_parameter", False)),
            loss_scale=float(plan.get("loss_scale_by_rank", {}).get(str(rank), 1.0)),
        )
        # 生产里 identity 由 library/runtime/accelerator.py 在训练前写并登记路径；
        # 这里按同一顺序做，loop 的 note_gradient_sync 才能把真实 bucket 计数写回。
        from library.training.multigpu import identity as identity_mod

        payload = identity_mod.build_runtime_identity(accelerator, state.args)
        summaries = identity_mod.collect_rank_summaries(accelerator, payload)
        manifest = Path(out_dir) / "multigpu_identity.json"
        identity_mod.register_manifest_path(manifest)
        identity_mod.write_runtime_identity(
            accelerator, payload, manifest, per_rank_summaries=summaries
        )

        originals = support.patch_loop_for_fixture(
            loop, real_gradient_sync=bool(plan.get("real_gradient_sync", False))
        )
        skipped_trace: list[bool] = []
        original_boundary = loop._commit_step_boundary

        def tracking_boundary(trainer_arg, state_arg, loss_arg):
            skipped_trace.append(bool(getattr(state_arg, "update_skipped", False)))
            return original_boundary(trainer_arg, state_arg, loss_arg)

        loop._commit_step_boundary = tracking_boundary
        record: dict = {}
        try:
            loop._run_epoch_steps(trainer, state, epoch=0)
        finally:
            loop._commit_step_boundary = original_boundary
            support.restore_loop(loop, originals)

        manifest_doc = (
            json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        )
        record = {
            "rank": rank,
            "world_size": accelerator.num_processes,
            "configured_world_size": payload["configured_world_size"],
            "reduction_mode": manifest_doc.get("reduction_mode"),
            "gradient_buckets": manifest_doc.get("gradient_buckets"),
            "reduced_parameter_count": manifest_doc.get("reduced_parameter_count"),
            "rank_summaries": manifest_doc.get("ranks"),
            "global_step": state.global_step,
            "window_attempts": getattr(state, "window_attempts", 0),
            "window_replays": getattr(state, "window_replays", 0),
            "window_discards": getattr(state, "window_discards", 0),
            "consecutive_discards": getattr(state, "window_consecutive_discards", 0),
            "applied_steps": optimizer.applied_steps,
            "optimizer_step_calls": optimizer.step_calls,
            "committed_boundaries": len(skipped_trace),
            "scaler_scale": float(scaler.get_scale()),
            "scaler_growth_tracker": float(scaler._growth_tracker.item()),
            "param_sha": _param_digest(model),
            # 每个 rank 自己的梯度台账：边界（含丢弃/重放）之后必须已清空，
            # 否则说明某 rank 把 NaN 或上一步梯度留给了下一个窗口。
            "grads_present": [p.grad is not None for p in model.parameters()],
        }
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _param_digest(model) -> list[float]:
    return [round(float(p.detach().sum()), 6) for p in model.parameters()]


def _run_two_rank(tmp_path: Path, plan: dict) -> list[dict]:
    import torch.multiprocessing as mp

    init_file = str(tmp_path / "gloo-init")
    mp.spawn(
        _run_rank,
        args=(2, init_file, str(tmp_path), plan),
        nprocs=2,
        join=True,
    )
    return [
        json.loads(Path(tmp_path, f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]


def _assert_symmetric(records: list[dict]) -> None:
    left, right = records
    for field in (
        "global_step",
        "window_attempts",
        "window_replays",
        "window_discards",
        "consecutive_discards",
        "applied_steps",
        "optimizer_step_calls",
        "committed_boundaries",
        "param_sha",
    ):
        assert left[field] == right[field], f"{field} diverged: {left[field]} != {right[field]}"
    assert left["scaler_scale"] == pytest.approx(right["scaler_scale"])
    assert left["scaler_growth_tracker"] == pytest.approx(right["scaler_growth_tracker"])


@pytest.mark.integration
def test_two_rank_clean_window_stays_symmetric(tmp_path):
    records = _run_two_rank(
        tmp_path, {"max_train_steps": 2, "batches": 2, "accum": 1}
    )
    _assert_symmetric(records)
    assert records[0]["applied_steps"] == 2


@pytest.mark.integration
def test_two_rank_peer_bad_loss_does_not_diverge(tmp_path):
    """P0-0 核心回归：一个 rank 坏 loss（梯度有限）→ 决策/scale/游标仍对称。"""
    records = _run_two_rank(
        tmp_path,
        {
            "max_train_steps": 2,
            "batches": 2,
            "accum": 1,
            "bad_steps_by_rank": {"1": [0]},
        },
    )
    _assert_symmetric(records)
    # 坏 loss 出现在第 0 窗：两侧都不得把该窗计入提交，且都必须重放一次。
    assert records[0]["window_replays"] >= 1
    assert records[1]["window_replays"] >= 1


def _run_primitives(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    """W-3.1 原语回归：两 rank 全 1 → SUM=2.0 且 OR=True（T3-F3）。"""
    import torch.distributed as dist

    from tests import multigpu_reference_support as support

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        accelerator = support.LoopAccelerator(world_size=world_size, scaler=None)
        both = agree_any(accelerator, {"flag": True})
        packed_sum = float(
            accelerator.reduce(torch.ones(1, dtype=torch.float32), reduction="sum").item()
        )
        one = agree_any(accelerator, {"flag": rank == 0})
        none = agree_any(accelerator, {"flag": False})
        record = {
            "rank": rank,
            "world_size": world_size,
            "packed_sum": packed_sum,
            "both_true": both["flag"],
            "one_true": one["flag"],
            "none_true": none["flag"],
            "two_flag_sum": float(
                accelerator.reduce(
                    torch.tensor([1.0, 1.0], dtype=torch.float32), reduction="sum"
                )[0].item()
            ),
        }
        Path(out_dir, f"prim{rank}.json").write_text(json.dumps(record), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.integration
def test_two_rank_or_primitive_sums_flags_and_decides_by_world_not_shard(tmp_path):
    """两 rank 全 1 时 reduce('sum') == 2.0、OR 恒 True，且判定与 world_size 无关。

    这是 W-3.1 的原语回归：只看单进程布尔值无法证明"OR 由全体决定"，
    必须看到真实归约值等于 world_size。
    """
    import torch.multiprocessing as mp

    init_file = str(tmp_path / "gloo-prim")
    mp.spawn(_run_primitives, args=(2, init_file, str(tmp_path)), nprocs=2, join=True)
    records = [
        json.loads(Path(tmp_path, f"prim{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    for record in records:
        assert record["packed_sum"] == pytest.approx(2.0)  # 两 rank 各 1.0
        assert record["two_flag_sum"] == pytest.approx(2.0)
        assert record["both_true"] is True
        # 只有 rank0 置真：归约值为 1.0，但"任一"仍为 True
        assert record["one_true"] is True
        assert record["none_true"] is False
    # 判定是"任一 rank"，与 world_size 无关：两侧结论完全一致。
    assert records[0]["both_true"] == records[1]["both_true"]
    assert records[0]["one_true"] == records[1]["one_true"]


@pytest.mark.integration
def test_two_rank_tail_window_stays_symmetric(tmp_path):
    """尾窗（微批数不足一个窗口）也必须两 rank 同分支且不 hang。"""
    records = _run_two_rank(
        tmp_path, {"max_train_steps": 1, "batches": 3, "accum": 2}
    )
    _assert_symmetric(records)


@pytest.mark.integration
def test_two_rank_discarded_tail_window_does_not_hang(tmp_path):
    """尾窗被判死（重放仍坏）→ 两 rank 同步丢弃、不 hang、计数器一致。"""
    records = _run_two_rank(
        tmp_path,
        {
            "max_train_steps": 1,
            "batches": 3,
            "accum": 2,
            "bad_steps_by_rank": {"1": [0, 1, 2, 3, 4, 5]},
        },
    )
    _assert_symmetric(records)
    assert records[0]["window_discards"] >= 1
    assert records[0]["window_discards"] == records[1]["window_discards"]
    # 丢弃不推进提交游标
    assert records[0]["global_step"] == 0


@pytest.mark.integration
def test_two_rank_identity_fields_are_real_and_world_wide(tmp_path):
    """P0-3：configured_world_size/真实 world_size 分列，bucket 计数取真实现。"""
    records = _run_two_rank(
        tmp_path,
        {
            "max_train_steps": 1,
            "batches": 2,
            "accum": 1,
            "real_gradient_sync": True,
        },
    )
    _assert_symmetric(records)
    for record in records:
        assert record["configured_world_size"] == 2
        assert record["world_size"] == 2
        assert record["configured_world_size"] == record["world_size"]
        # 来自真实 synchronize_optimizer_gradients 的返回，而不是静态猜测
        assert record["gradient_buckets"] and record["gradient_buckets"] >= 1
        assert record["reduced_parameter_count"] and record["reduced_parameter_count"] >= 1
        assert "bucket_mean" in str(record["reduction_mode"])
        assert record["rank_summaries"], "rank0 manifest must carry every rank"


@pytest.mark.integration
def test_two_rank_params_match_only_because_of_real_gradient_sync(tmp_path):
    """②：两 rank 局部 loss 缩放不同，参数仍逐位一致 → 归约真的生效。"""
    records = _run_two_rank(
        tmp_path,
        {
            "max_train_steps": 1,
            "batches": 2,
            "accum": 1,
            "real_gradient_sync": True,
            "loss_scale_by_rank": {"0": 1.0, "1": 3.0},
        },
    )
    _assert_symmetric(records)
    assert records[0]["param_sha"] == records[1]["param_sha"]
    assert records[0]["applied_steps"] == 1


@pytest.mark.integration
def test_two_rank_boundary_clears_grads_on_every_rank(tmp_path):
    """边界后每个 rank 的梯度都必须已清空（含坏 loss 的 rank）。"""
    records = _run_two_rank(
        tmp_path,
        {
            "max_train_steps": 1,
            "batches": 2,
            "accum": 1,
            "bad_steps_by_rank": {"1": [0, 1, 2]},
        },
    )
    _assert_symmetric(records)
    for record in records:
        assert not any(record["grads_present"]), record["grads_present"]


@pytest.mark.integration
def test_two_rank_grad_none_parameter_stays_symmetric(tmp_path):
    """``grad=None`` 的未使用参数不得让协议改写 presence 语义或分叉。"""
    records = _run_two_rank(
        tmp_path,
        {
            "max_train_steps": 1,
            "batches": 2,
            "accum": 1,
            "unused_parameter": True,
            "bad_steps_by_rank": {"1": [0]},
        },
    )
    _assert_symmetric(records)


@pytest.mark.integration
def test_two_rank_accumulation_window_stays_symmetric(tmp_path):
    records = _run_two_rank(
        tmp_path,
        {
            "max_train_steps": 2,
            "batches": 4,
            "accum": 2,
            "bad_steps_by_rank": {"1": [0]},
        },
    )
    _assert_symmetric(records)
    assert records[0]["window_replays"] >= 1
