"""多进程准入/原因码/世界规模判定的单元测试（P0-1/P0-4）。

纯 CPU、无进程组：默认进 core 层（无 marker）。
"""

from __future__ import annotations

import pytest

from library.runtime.launch import (
    configure_accelerate_for_gpu_selection,
    resolve_training_world_size_for_gpu_selection,
)
from library.training.multigpu.admission import (
    CODE_INVALID_NUM_PROCESSES,
    CODE_LAUNCH_WITHOUT_WORKERS,
    CODE_SELECTION_MISMATCH,
    CODE_SINGLE_GPU_MULTIPROCESS,
    resolve_admission,
    resolve_world_size,
)

NUM_ENV = "ANIMA_ACCELERATE_NUM_PROCESSES"
LAUNCH_ENV = "ANIMA_ACCELERATE_LAUNCH"


def test_single_gpu_is_unchanged_by_default():
    decision = resolve_admission([0], {})
    assert decision.ok
    assert decision.world_size == 1
    assert decision.num_processes == 1


def test_two_gpu_selection_defaults_to_two_workers():
    decision = resolve_admission([0, 1], {})
    assert decision.ok
    assert decision.num_processes == 2
    assert decision.world_size == 2


def test_explicit_matching_worker_count_is_accepted():
    decision = resolve_admission([0, 1], {NUM_ENV: "2", LAUNCH_ENV: "1"})
    assert decision.ok
    assert decision.world_size == 2


def test_two_cards_with_one_worker_is_refused_with_a_reason_code():
    decision = resolve_admission([0, 1], {NUM_ENV: "1", LAUNCH_ENV: "1"})
    assert not decision.ok
    assert decision.code == CODE_SINGLE_GPU_MULTIPROCESS
    with pytest.raises(ValueError) as excinfo:
        decision.raise_if_rejected()
    assert CODE_SINGLE_GPU_MULTIPROCESS in str(excinfo.value)


def test_fewer_workers_than_cards_is_refused():
    decision = resolve_admission([0, 1, 2], {NUM_ENV: "2", LAUNCH_ENV: "1"})
    assert not decision.ok
    assert decision.code == CODE_SELECTION_MISMATCH


def test_more_workers_than_cards_is_refused():
    """方向 (>)：显式 worker 多于卡数同样拒绝，不再是"用户偏好"。

    repair round 2 / T3-F1：`num_processes > len(selection)` 以前被当作可保留的
    显式值放行，等于允许一个 worker 共用半张卡；合同 §7 P0-1 要求冲突即拒绝。
    """
    decision = resolve_admission([0, 1], {NUM_ENV: "4", LAUNCH_ENV: "1"})
    assert not decision.ok
    assert decision.code == CODE_SELECTION_MISMATCH
    with pytest.raises(ValueError) as excinfo:
        decision.raise_if_rejected()
    assert CODE_SELECTION_MISMATCH in str(excinfo.value)


def test_launcher_refuses_more_workers_than_cards_with_the_same_code():
    """launcher 与准入判定对 (>) 方向抛同一原因码，CLI/Web 口径一致。"""
    env = {NUM_ENV: "4", LAUNCH_ENV: "1"}
    with pytest.raises(ValueError) as excinfo:
        configure_accelerate_for_gpu_selection(env, [0, 1])
    assert CODE_SELECTION_MISMATCH in str(excinfo.value)


def test_fewer_workers_than_cards_launcher_uses_the_same_code():
    """launcher 与准入判定对 (<) 方向抛同一原因码。"""
    env = {NUM_ENV: "2", LAUNCH_ENV: "1"}
    with pytest.raises(ValueError) as excinfo:
        configure_accelerate_for_gpu_selection(env, [0, 1, 2])
    assert CODE_SELECTION_MISMATCH in str(excinfo.value)


def test_launch_without_workers_is_refused():
    decision = resolve_admission([], {LAUNCH_ENV: "1", NUM_ENV: "1"})
    assert not decision.ok
    assert decision.code == CODE_LAUNCH_WITHOUT_WORKERS


@pytest.mark.parametrize("raw", ["zero", "0", "-2"])
def test_invalid_worker_count_is_refused(raw):
    decision = resolve_admission([0, 1], {NUM_ENV: raw, LAUNCH_ENV: "1"})
    assert not decision.ok
    assert decision.code == CODE_INVALID_NUM_PROCESSES


def test_launch_disabled_keeps_dormant_worker_value():
    """LAUNCH=0 时进程数只是休眠值，保持历史行为（不改写、不拒绝）。"""
    decision = resolve_admission([0, 1], {NUM_ENV: "4", LAUNCH_ENV: "0"})
    assert decision.ok
    assert decision.num_processes == 4
    assert decision.world_size == 1


def test_world_size_prefers_launcher_world_size_env():
    """进程内判定必须用真实 world_size，而不是占位 1。"""
    assert resolve_world_size(env={"WORLD_SIZE": "2"}) == 2
    assert resolve_world_size(env={NUM_ENV: "2", LAUNCH_ENV: "1"}) == 2
    assert resolve_world_size(env={}) == 1


def test_world_size_ignores_garbage_world_size_env():
    assert resolve_world_size(env={"WORLD_SIZE": "not-a-number"}) == 1


def test_launcher_refuses_the_same_conflict_the_matrix_would():
    """CLI/Web 启动路径与准入判定共用同一原因码，不各写一套。"""
    env = {NUM_ENV: "1", LAUNCH_ENV: "1"}
    with pytest.raises(ValueError) as excinfo:
        configure_accelerate_for_gpu_selection(env, [0, 1])
    assert CODE_SINGLE_GPU_MULTIPROCESS in str(excinfo.value)


def test_launcher_still_defaults_two_cards_to_two_workers():
    env: dict[str, str] = {}
    configure_accelerate_for_gpu_selection(env, [0, 1])
    assert env[LAUNCH_ENV] == "1"
    assert env[NUM_ENV] == "2"
    assert resolve_training_world_size_for_gpu_selection([0, 1], env) == 2


def test_single_card_selection_does_not_enter_multiprocess():
    env: dict[str, str] = {}
    configure_accelerate_for_gpu_selection(env, [0])
    assert env == {}
    assert resolve_training_world_size_for_gpu_selection([0], env) == 1


def test_single_process_keeps_existing_rejections():
    """单卡默认不变：多进程专属拒绝仍在（native/islands/PP 等不被放宽）。"""
    from library.training.compat_matrix import check_training_compat

    config = {
        "model_family": "anima",
        "model_type": "anima",
        "adaptive_precision": "native",
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
    }
    single = check_training_compat(config, world_size=1)
    multi = check_training_compat(config, world_size=2)
    single_codes = {error.code for error in single.errors}
    multi_codes = {error.code for error in multi.errors}
    # 单卡侧已有拒绝不得因为多卡工作被放宽：两者都必须仍然报错。
    assert single_codes
    assert multi_codes
    # 多卡必须保留 PP 运行时未接线的拒绝（不被本次实现顺手打开）。
    assert "pipeline_parallel_runtime_unavailable" in multi_codes
    # native/islands 系列的多进程拒绝也必须仍在。
    assert "adaptive_precision_contract" in multi_codes


def test_world_size_env_wins_even_when_smaller_than_admission_value():
    """合法 WORLD_SIZE 直接以其为准（admission 值仅作回退，review t4）。"""
    env = {"WORLD_SIZE": "1", NUM_ENV: "2", LAUNCH_ENV: "1"}
    assert resolve_world_size(env=env) == 1
    assert resolve_admission([0, 1], env).num_processes == 2  # 意向仍是 2


@pytest.mark.parametrize("workers", ["2", "4"])
def test_single_selected_gpu_with_multiple_workers_is_refused(workers):
    """selection 长度为 1 而 num_processes=2/4 → 同码拒绝（launcher 与判定一致）。"""
    env = {NUM_ENV: workers, LAUNCH_ENV: "1"}
    decision = resolve_admission([0], env)
    assert not decision.ok
    assert decision.code == CODE_SELECTION_MISMATCH
    with pytest.raises(ValueError) as excinfo:
        configure_accelerate_for_gpu_selection(env, [0])
    assert CODE_SELECTION_MISMATCH in str(excinfo.value)


def test_identity_payload_records_configured_and_real_world_size_separately():
    """P0-3：configured_world_size 与真实 world_size 分列，bucket 计数初值为 None。"""
    from types import SimpleNamespace

    import torch

    from library.training.multigpu import identity

    accelerator = SimpleNamespace(
        num_processes=1,
        process_index=0,
        device=torch.device("cpu"),
        mixed_precision="no",
        scaler=None,
        state=SimpleNamespace(distributed_type="NO"),
    )
    payload = identity.build_runtime_identity(
        accelerator, SimpleNamespace(train_batch_size=1, gradient_accumulation_steps=1)
    )
    assert payload["world_size"] == 1
    assert payload["configured_world_size"] == 1
    assert payload["gradient_buckets"] is None  # 第一个边界之前不伪造计数
    assert payload["reduction_mode"] == "pending_first_boundary"
