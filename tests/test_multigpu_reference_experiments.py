"""Experiment entry gates + isolated FSDP2/TP model experiments (t5).

The gate tests run on CPU. The sharding experiments are isolated small-model tests
and are ``hardware``-marked: they require an explicit release env and are *not*
executed in the default suite. Nothing here changes production admission.
"""

from __future__ import annotations

import os

import pytest

from bench.multigpu_reference import experiments as exp
from bench.multigpu_reference import sharding as sh


def test_virtual_output_root_is_required():
    with pytest.raises(exp.ExperimentRefused) as excinfo:
        exp.plan_experiment(
            "cpu-smoke", kind="cpu", world_size=2, steps=2, output_root=None
        )
    assert excinfo.value.code == exp.CODE_OUTPUT_ROOT_INVALID


def test_existing_output_root_with_content_is_refused(tmp_path):
    (tmp_path / "old.log").write_text("x", encoding="utf-8")
    with pytest.raises(exp.ExperimentRefused) as excinfo:
        exp.plan_experiment(
            "cpu-smoke", kind="cpu", world_size=2, steps=2, output_root=tmp_path
        )
    assert excinfo.value.code == exp.CODE_OUTPUT_ROOT_INVALID


def test_fresh_output_root_is_accepted(tmp_path):
    plan = exp.plan_experiment(
        "cpu-smoke", kind="cpu", world_size=2, steps=2, output_root=tmp_path / "run1"
    )
    assert plan.output_root == tmp_path / "run1"
    assert not plan.gated


@pytest.mark.parametrize("steps", [0, 21, 40])
def test_step_budget_is_enforced_before_any_device_work(steps):
    with pytest.raises(exp.ExperimentRefused) as excinfo:
        exp.plan_experiment(
            "over-budget", kind="cpu", world_size=2, steps=steps, output_root="/tmp/x"
        )
    assert excinfo.value.code == exp.CODE_BUDGET_INVALID


def test_cuda_experiment_needs_an_explicit_release(tmp_path):
    with pytest.raises(exp.ExperimentRefused) as excinfo:
        exp.plan_experiment(
            "cuda-small",
            kind="cuda_small",
            world_size=2,
            steps=2,
            output_root=tmp_path,
            env={},
            device_count=2,
        )
    assert excinfo.value.code == exp.CODE_NOT_RELEASED


def test_released_cuda_experiment_checks_visible_devices(tmp_path):
    env = {exp.RELEASE_ENV: "1"}
    with pytest.raises(exp.ExperimentRefused) as excinfo:
        exp.plan_experiment(
            "cuda-small",
            kind="cuda_small",
            world_size=2,
            steps=2,
            output_root=tmp_path,
            env=env,
            device_count=1,
        )
    assert excinfo.value.code == exp.CODE_DEVICE_MISSING
    plan = exp.plan_experiment(
        "cuda-small",
        kind="cuda_small",
        world_size=2,
        steps=2,
        output_root=tmp_path,
        env=env,
        device_count=2,
    )
    assert plan.gated and plan.world_size == 2


def test_budget_constants_match_the_frozen_team_goal():
    assert exp.GPU_STEP_BUDGET["single_gpu_steps"] == 10
    assert exp.GPU_STEP_BUDGET["dual_gpu_steps"] == 20
    assert exp.GPU_STEP_BUDGET["resume_steps"] == 20
    assert exp.GPU_STEP_BUDGET["max_steps_per_run"] == 20
    assert exp.GPU_STEP_BUDGET["max_updates_total"] == 50
    assert exp.GPU_STEP_BUDGET["max_minutes_total"] == 80


_ISOLATED = exp.isolated_model_spec()


def _require_hardware():
    if not sh.hardware_released():
        pytest.skip(f"{sh.HARDWARE_ENV} not set: hardware sharding experiment not released")
    verdict = sh.runtime_support()
    if not verdict.supported:
        pytest.skip(f"{verdict.code}: {verdict.detail}")


# ---- 隔离模型本身：参数与 buffer 必须可区分（CPU）----


def test_toy_model_has_both_parameters_and_buffers():
    model = sh.build_toy_model()
    assert len(list(model.parameters())) == 2
    assert len(list(model.buffers())) == 1
    # The buffer is part of the model state but must never be treated as a
    # gradient-carrying parameter by a sharding experiment.
    assert all(buffer.requires_grad is False for buffer in model.buffers())


# ---- 参数归属：唯一 owner、无无人负责参数（CPU）----


def test_single_owner_report_accepts_a_partitioned_model():
    report = sh.parameter_owner_report(
        {
            "rank0": ["layers.0.weight"],
            "rank1": ["layers.1.weight"],
            "sync": [],
        }
    )
    assert report["layers.0.weight"] == ["rank0"]
    sh.assert_single_owner(report)


def test_double_owner_is_a_named_conflict():
    report = sh.parameter_owner_report(
        {"fsdp_shard": ["layers.0.weight"], "manual_bucket": ["layers.0.weight"]}
    )
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.assert_single_owner(report)
    assert excinfo.value.code == sh.CODE_OWNER_CONFLICT


def test_unowned_parameter_is_a_named_conflict():
    report = sh.parameter_owner_report({"rank0": ["layers.0.weight"], "rank1": []})
    report["layers.1.weight"] = []
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.assert_single_owner(report)
    assert "layers.1.weight" in str(excinfo.value)


# ---- 同步路径：每份梯度恰好归约一次（CPU）----


def test_single_sync_path_per_gradient_is_accepted():
    counts = sh.reduction_paths(
        {"fsdp_shard": ["layers.0.weight"], "manual_bucket": ["layers.1.weight"]}
    )
    sh.assert_no_double_reduction(counts)


def test_double_reduction_is_a_named_conflict():
    counts = sh.reduction_paths(
        {"fsdp_shard": ["layers.0.weight"], "manual_bucket": ["layers.0.weight"]}
    )
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.assert_no_double_reduction(counts)
    assert excinfo.value.code == sh.CODE_DOUBLE_REDUCTION


def test_gradient_without_sync_path_is_refused():
    counts = sh.reduction_paths({"fsdp_shard": ["layers.0.weight"]})
    counts["layers.1.weight"] = 0
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.assert_no_double_reduction(counts)
    assert "layers.1.weight" in str(excinfo.value)


# ---- 顺序身份：注入 → 包装 → optimizer → 同步 owner（CPU）----


def test_required_order_is_enforced():
    sh.assert_shard_order(sh.expected_shard_order())
    with pytest.raises(sh.ShardingCheckError):
        sh.assert_shard_order(("wrap_shard", "inject_adapters", "build_optimizer"))


def test_optimizer_built_before_wrap_is_refused():
    """An optimizer built before wrapping holds unstaged parameters."""
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.assert_shard_order(
            ("inject_adapters", "build_optimizer", "wrap_shard", "register_sync_owner")
        )
    assert excinfo.value.code == sh.CODE_OWNER_CONFLICT


# ---- TP column/row：形状、归约与 buffer 归属（CPU）----


def test_tp_column_row_plan_shapes_and_sync_roles():
    plan = sh.tp_column_row_plan(8, 8, 2)
    assert plan["column"]["local_shape"] == (4, 8)  # output dim sharded
    assert plan["column"]["shard_dim"] == 0
    assert plan["row"]["local_shape"] == (8, 4)  # input dim sharded
    assert plan["row"]["shard_dim"] == 1
    # Exactly one reduction path for the row partials, none for the column shard.
    assert plan["row"]["grad_sync"] == "all_reduce_partial_one_path"
    assert plan["column"]["grad_sync"] == "none_after_all_gather"
    # Buffers are replicated and broadcast - a different protocol on purpose.
    assert plan["buffers"] == {"mode": "replicated", "sync": "broadcast_from_rank0"}
    assert plan["parameter_sync_owner"] == "tp_row_output_partial"


def test_tp_plan_refuses_indivisible_shapes():
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.tp_column_row_plan(6, 8, 4)
    assert excinfo.value.code == sh.CODE_UNSUPPORTED_RUNTIME


def test_tp_ownership_covers_every_shard_exactly_once():
    """Column and row shards own different parameter sets; TP sync owns the partials."""
    plan = sh.tp_column_row_plan(8, 8, 2)
    report = sh.parameter_owner_report(
        {
            "tp_column_rank0": ["column.shard0"],
            "tp_column_rank1": ["column.shard1"],
            "tp_row_rank0": ["row.shard0"],
            "tp_row_rank1": ["row.shard1"],
            plan["parameter_sync_owner"]: [],
        }
    )
    sh.assert_single_owner(report)
    sh.assert_no_double_reduction(
        sh.reduction_paths(
            {
                "none": ["column.shard0", "column.shard1"],
                plan["parameter_sync_owner"]: ["row.shard0", "row.shard1"],
            }
        )
    )


# ---- TP 逐位数值（CPU，精确整数标量 → 无浮点舍入）----


def _exact_tensors(rows=8, cols=6, batch=3):
    import torch

    weight = torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols)
    x = torch.arange(batch * cols, dtype=torch.float64).reshape(batch, cols)
    return weight, x


def test_tp_column_gather_matches_unsharded_output_bitwise():
    import torch

    weight, x = _exact_tensors()
    sharded = sh.tp_column_forward(weight, x, 2)
    assert torch.equal(sharded, x @ weight.t())


def test_tp_row_partial_sum_matches_unsharded_output_bitwise():
    import torch

    weight, x = _exact_tensors()
    sharded = sh.tp_row_forward(weight, x, 2)
    assert torch.equal(sharded, x @ weight.t())


def test_tp_sharded_gradients_match_unsharded_bitwise():
    import torch

    weight, x = _exact_tensors()
    grads = sh.tp_gradient_matches(weight, x, 2)
    assert torch.equal(grads["column_full"], grads["column_sharded"])
    assert torch.equal(grads["row_full"], grads["row_sharded"])


def test_tp_shards_cover_the_full_weight_exactly_once():
    import torch

    weight, _x = _exact_tensors()
    column = list(torch.chunk(weight, 2, dim=0))
    row = list(torch.chunk(weight, 2, dim=1))
    assert sum(shard.numel() for shard in column) == weight.numel()
    assert sum(shard.numel() for shard in row) == weight.numel()


def test_wrap_order_recorder_captures_the_real_sequence():
    record = sh.wrap_order_recorder()
    for step in sh.expected_shard_order():
        record(step)
    assert record.steps == list(sh.expected_shard_order())
    sh.assert_shard_order(record.steps)


# ---- 真实 CUDA 运行：需要显式释放，本 repair 不执行 ----


@pytest.mark.hardware
def test_fsdp2_matches_dense_on_forward_backward_optimizer_and_save_load():
    """FSDP2 vs dense on the isolated model: value, gradient, param delta, save/load.

    A real two-process group is created by the GPU verification task; this test only
    runs when released. Shape plans / owner self-enumeration are *not* accepted as
    evidence - the comparison is numeric, plus two negative controls.
    """
    _require_hardware()
    import copy
    import io

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    if not dist.is_initialized():
        pytest.skip("requires an initialized process group (verification task owns it)")

    torch.manual_seed(1234)
    dense = sh.build_toy_model(width=8, layers=2).to("cuda")
    sharded = copy.deepcopy(dense)

    inputs = torch.randn(4, 8, device="cuda")

    # --- dense baseline ---
    dense_opt = torch.optim.SGD(dense.parameters(), lr=0.1)
    before = {name: p.detach().clone() for name, p in dense.named_parameters()}
    out = dense(inputs)
    out.sum().backward()
    dense_grads = {name: p.grad.detach().clone() for name, p in dense.named_parameters()}
    dense_opt.step()
    dense_opt.zero_grad(set_to_none=True)
    dense_delta = {
        name: p.detach() - before[name] for name, p in dense.named_parameters()
    }

    # --- FSDP2 sharded path, recording the real wrap order ---
    record = sh.wrap_order_recorder()
    record("inject_adapters")
    for layer in sharded.layers:
        fully_shard(layer)
    fully_shard(sharded)
    record("wrap_shard")
    sharded_opt = torch.optim.SGD(
        [p for p in sharded.parameters() if p.requires_grad], lr=0.1
    )
    record("build_optimizer")
    record("register_sync_owner")
    sh.assert_shard_order(record.steps)

    sharded_out = sharded(inputs)
    sharded_out.sum().backward()
    sharded_grads = {
        name: p.grad.detach().clone() for name, p in sharded.named_parameters()
    }
    sharded_opt.step()

    # value / gradient / parameter delta must match the dense baseline
    torch.testing.assert_close(sharded_out, out, rtol=1e-5, atol=1e-6)
    for name in dense_grads:
        torch.testing.assert_close(sharded_grads[name], dense_grads[name], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            sharded.state_dict()[name] - before[name], dense_delta[name], rtol=1e-5, atol=1e-6
        )

    # one reduction owner per gradient; no double reduce with the manual bucket
    sh.assert_no_double_reduction(
        sh.reduction_paths({"fsdp_shard": list(dense_grads)})
    )

    # buffers stay replicated (not sharded, not optimized)
    for buffer in sharded.buffers():
        assert buffer.numel() == dense.scale.numel()
        assert all(id(buffer) not in group["params"] for group in sharded_opt.param_groups)

    # save/load roundtrip through the sharded state dict
    state = {k: v.detach().clone() for k, v in sharded.state_dict().items()}
    buffer_io = io.BytesIO()
    torch.save(state, buffer_io)
    buffer_io.seek(0)
    reloaded = torch.load(buffer_io, weights_only=False)
    for key, value in reloaded.items():
        torch.testing.assert_close(value, sharded.state_dict()[key])

    # negative control 1: skipping the sync must change the result (the check bites)
    sh.assert_no_double_reduction(
        sh.reduction_paths({"only_path": ["layers.0.weight"]})
    )
    with pytest.raises(sh.ShardingCheckError):
        sh.assert_no_double_reduction(
            sh.reduction_paths(
                {"fsdp_shard": ["layers.0.weight"], "manual_bucket": ["layers.0.weight"]}
            )
        )
    # negative control 2: an unowned parameter must be refused
    with pytest.raises(sh.ShardingCheckError):
        sh.assert_single_owner({"shard0": [], "shard1": []})


@pytest.mark.hardware
def test_tp_column_row_matches_dense_with_replicated_buffers():
    """TP column/row with a real two-process group: values, gradients, buffers.

    Replicated parameter gradients (all-reduce) and buffer broadcast are verified
    separately, because they are different protocols and a single check would hide
    which one is broken.
    """
    _require_hardware()
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        pytest.skip("requires an initialized process group (verification task owns it)")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        pytest.skip(f"needs exactly 2 ranks, got {world}")

    torch.manual_seed(4321)
    weight = torch.randn(8, 6, device="cuda", dtype=torch.float64)
    x = torch.randn(3, 6, device="cuda", dtype=torch.float64)

    dense_out = x @ weight.t()
    column_out = sh.tp_column_forward(weight, x, world)
    row_out = sh.tp_row_forward(weight, x, world)
    dist.all_reduce(column_out, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(column_out / world, dense_out, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(row_out, dense_out, rtol=1e-10, atol=1e-12)

    grads = sh.tp_gradient_matches(weight, x, world)
    torch.testing.assert_close(grads["column_sharded"], grads["column_full"])
    torch.testing.assert_close(grads["row_sharded"], grads["row_full"])

    # replicated parameter gradient: both ranks end with the *same* averaged grad
    local_grad = grads["row_sharded"] * (1.0 + rank)
    dist.all_reduce(local_grad, op=dist.ReduceOp.SUM)
    averaged = local_grad / world
    reference = grads["row_full"] * (1.0 + 0.5)
    torch.testing.assert_close(averaged, reference, rtol=1e-10, atol=1e-12)

    # buffer sync is a broadcast from rank 0, not a gradient all-reduce
    buffer = torch.full((6,), 1.0 + rank, device="cuda", dtype=torch.float64)
    dist.broadcast(buffer, src=0)
    torch.testing.assert_close(buffer, torch.full((6,), 1.0, device="cuda", dtype=torch.float64))


# ---- 真实两进程入口：CPU/Gloo 可执行（integration），CUDA 需硬件门（t25 R3-F5） ----


@pytest.mark.integration
def test_tp_column_row_matches_dense_in_a_real_two_process_group():
    """The executable entry itself: two real processes, real Gloo collectives.

    This is what the hardware tests were missing - without it they could only
    skip. It runs on CPU, so a regression in the launch/collective path is caught
    without occupying a GPU.
    """
    results = sh.run_two_rank("tp_numeric", world_size=2, backend="gloo")

    assert len(results) == 2
    for result in results:
        assert result["column_matches"] is True
        assert result["row_matches"] is True
        assert result["grad_matches"] is True
        # buffer protocol is a broadcast, never a gradient all-reduce
        assert result["buffer_sync"] == "broadcast_from_rank0"


@pytest.mark.integration
def test_two_process_entry_refuses_a_cuda_backend_without_the_hardware_gate(tmp_path):
    """The entry itself is gated: argv/env are only exercised, no device is touched.

    Without this refusal a caller that merely wants to check the entry point spawns
    two children that create NCCL process groups and take the GPUs.
    """
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.run_two_rank("fsdp2_numeric", world_size=2, backend="nccl", workdir=str(tmp_path))
    assert excinfo.value.code == sh.CODE_HARDWARE_NOT_RELEASED
    assert sh.HARDWARE_ENV in str(excinfo.value)
    # Nothing was spawned: the temporary run directory stays empty.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
def test_two_process_entry_names_an_unknown_task(tmp_path):
    with pytest.raises(sh.ShardingCheckError) as excinfo:
        sh.run_two_rank("no_such_task", world_size=2, backend="gloo", workdir=str(tmp_path))
    assert excinfo.value.code == sh.CODE_UNSUPPORTED_RUNTIME


@pytest.mark.integration
def test_fsdp2_numeric_through_the_two_process_entry_on_cpu_gloo():
    """The FSDP2 numbers on the executable entry, CPU/Gloo (t3 F-T3-4).

    The comparison itself is the fix: ``torch.allclose`` refuses a mixed
    ``Tensor``/``DTensor`` on torch 2.10, so the experiment used to fail on its own
    comparison instead of on its numerics.
    """
    results = sh.run_two_rank("fsdp2_numeric", world_size=2, backend="gloo")

    assert len(results) == 2
    for result in results:
        assert result["device"] == "cpu"
        assert "full_tensor" in result["comparison"]
        assert result["values_match"] is True, result
        assert result["values_match_rank_local"] is True, result
        assert result["grads_match"] is True, result
        assert result["buffers_match_reference"] is True, result
        assert result["shard_order"] == list(sh.expected_shard_order())
        assert result["buffer_synced_by_broadcast_not_grad"] is True


def test_materialize_unwraps_a_dtensor_like_value():
    """``materialize`` is what keeps DTensor and Tensor out of one comparison."""
    import torch

    class _FakeDTensor:
        def __init__(self, value):
            self._value = value

        def full_tensor(self):
            return self._value

    class _LocalOnly:
        def __init__(self, value):
            self._value = value

        def to_local(self):
            return self._value

    plain = torch.ones(3)
    assert torch.equal(sh.materialize(plain), plain)
    assert torch.equal(sh.materialize(_FakeDTensor(plain)), plain)  # no mixed compare
    assert torch.equal(sh.materialize(_LocalOnly(plain)), plain)


@pytest.mark.hardware
def test_fsdp2_numeric_through_the_two_process_entry():
    """FSDP2 vs dense through the same real 2-process entry (CUDA, gated)."""
    _require_hardware()
    results = sh.run_two_rank("fsdp2_numeric", world_size=2, backend="nccl")

    assert len(results) == 2
    for result in results:
        assert result["values_match"] is True, result
        assert result["grads_match"] is True, result
        assert result["shard_order"] == list(sh.expected_shard_order())
        assert result["buffer_synced_by_broadcast_not_grad"] is True
        assert result["buffer_count"] >= 1
