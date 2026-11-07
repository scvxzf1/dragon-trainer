from __future__ import annotations

from pathlib import Path

import pytest

from library.runtime.launch import (
    ACCELERATE_LAUNCH_ENV,
    ACCELERATE_MIXED_PRECISION_ENV,
    ACCELERATE_NUM_PROCESSES_ENV,
    accelerate_training_command_prefix,
    configure_accelerate_for_gpu_selection,
    resolve_accelerate_mixed_precision,
    resolve_accelerate_num_processes,
    resolve_training_world_size_for_gpu_selection,
)


def test_training_command_defaults_to_direct_train_script(monkeypatch):
    monkeypatch.delenv(ACCELERATE_LAUNCH_ENV, raising=False)
    monkeypatch.delenv(ACCELERATE_NUM_PROCESSES_ENV, raising=False)
    cmd = accelerate_training_command_prefix("python", "train.py")
    assert cmd == ["python", "train.py"]


def test_explicit_env_mapping_isolated_from_process_env(monkeypatch):
    monkeypatch.setenv(ACCELERATE_LAUNCH_ENV, "1")
    monkeypatch.setenv(ACCELERATE_NUM_PROCESSES_ENV, "8")
    monkeypatch.setenv(ACCELERATE_MIXED_PRECISION_ENV, "fp16")

    cmd = accelerate_training_command_prefix("python", "train.py", {})

    assert cmd == ["python", "train.py"]


def test_direct_training_command_ignores_accelerate_detail_env_when_launch_disabled():
    env = {
        ACCELERATE_NUM_PROCESSES_ENV: "8",
        ACCELERATE_MIXED_PRECISION_ENV: "fp16",
    }
    cmd = accelerate_training_command_prefix("python", "train.py", env)
    assert cmd == ["python", "train.py"]


def test_direct_training_command_ignores_invalid_accelerate_detail_env_when_launch_disabled():
    env = {
        ACCELERATE_LAUNCH_ENV: "0",
        ACCELERATE_NUM_PROCESSES_ENV: "many",
        ACCELERATE_MIXED_PRECISION_ENV: "fp4",
    }
    cmd = accelerate_training_command_prefix("python", "train.py", env)
    assert cmd == ["python", "train.py"]


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", ""])
def test_falsey_accelerate_launch_env_keeps_direct_training_command(value: str):
    env = {
        ACCELERATE_LAUNCH_ENV: value,
        ACCELERATE_NUM_PROCESSES_ENV: "8",
        ACCELERATE_MIXED_PRECISION_ENV: "fp16",
    }
    cmd = accelerate_training_command_prefix("python", "train.py", env)
    assert cmd == ["python", "train.py"]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_accelerate_launch_env_enables_launch_command(value: str):
    env = {ACCELERATE_LAUNCH_ENV: value}
    cmd = accelerate_training_command_prefix("python", "train.py", env)
    assert cmd[:4] == ["python", "-m", "accelerate.commands.accelerate_cli", "launch"]
    assert cmd[-1] == "train.py"


def test_accelerate_launch_command_wraps_train_script_with_safe_defaults():
    env = {
        ACCELERATE_LAUNCH_ENV: "1",
        ACCELERATE_NUM_PROCESSES_ENV: "2",
        ACCELERATE_MIXED_PRECISION_ENV: "fp16",
    }

    cmd = accelerate_training_command_prefix("python", "train.py", env)

    assert cmd == [
        "python",
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--num_processes",
        "2",
        "--num_machines",
        "1",
        "--dynamo_backend",
        "no",
        "--num_cpu_threads_per_process",
        "3",
        "--mixed_precision",
        "fp16",
        "train.py",
    ]


def test_accelerate_command_stringifies_path_train_script():
    train_script = Path("train.py")

    assert accelerate_training_command_prefix("python", train_script, {}) == [
        "python",
        "train.py",
    ]

    cmd = accelerate_training_command_prefix(
        "python",
        train_script,
        {ACCELERATE_LAUNCH_ENV: "1"},
    )

    assert cmd[-1] == "train.py"


def test_accelerate_launch_env_values_are_stripped_before_parsing():
    env = {
        ACCELERATE_LAUNCH_ENV: " true ",
        ACCELERATE_NUM_PROCESSES_ENV: " 3 ",
        ACCELERATE_MIXED_PRECISION_ENV: " FP16 ",
    }
    cmd = accelerate_training_command_prefix("python", "train.py", env)
    assert cmd[cmd.index("--num_processes") + 1] == "3"
    assert cmd[cmd.index("--mixed_precision") + 1] == "fp16"


def test_accelerate_num_processes_defaults_to_single_process(monkeypatch):
    monkeypatch.setenv(ACCELERATE_LAUNCH_ENV, "1")
    monkeypatch.delenv(ACCELERATE_NUM_PROCESSES_ENV, raising=False)
    cmd = accelerate_training_command_prefix("python", "train.py")
    assert cmd[cmd.index("--num_processes") + 1] == "1"


def test_accelerate_num_processes_env_override():
    env = {ACCELERATE_NUM_PROCESSES_ENV: "2"}
    assert resolve_accelerate_num_processes(env) == "2"


def test_multi_gpu_selection_defaults_to_matching_accelerate_workers():
    env = {}

    configure_accelerate_for_gpu_selection(env, [2, 0])

    assert env[ACCELERATE_LAUNCH_ENV] == "1"
    assert env[ACCELERATE_NUM_PROCESSES_ENV] == "2"


def test_single_gpu_selection_keeps_direct_launch_defaults():
    env = {}

    configure_accelerate_for_gpu_selection(env, [0])

    assert env == {}


def test_multi_gpu_selection_preserves_explicit_launch_overrides():
    env = {
        ACCELERATE_LAUNCH_ENV: "0",
        ACCELERATE_NUM_PROCESSES_ENV: "4",
    }

    configure_accelerate_for_gpu_selection(env, [0, 1])

    assert env == {
        ACCELERATE_LAUNCH_ENV: "0",
        ACCELERATE_NUM_PROCESSES_ENV: "4",
    }


@pytest.mark.parametrize(
    ("gpu_selection", "world_size"),
    [
        ([], 1),
        ([0], 1),
        ([0, 1], 2),
        ([0, 1, 2], 3),
    ],
)
def test_gpu_selection_world_size_matches_webui_launch_policy(
    gpu_selection: list[int], world_size: int
) -> None:
    assert (
        resolve_training_world_size_for_gpu_selection(gpu_selection, {}) == world_size
    )


def test_gpu_selection_world_size_rejects_more_workers_than_cards() -> None:
    """repair round 2 / T3-F1：显式多 worker 共用少量卡不再被放行。

    旧行为（保留 4 并原样启动）会把"N 张卡、4 个 worker"当作用户偏好放过，
    与准入合同 §7 P0-1「冲突即拒绝启动」相反，因此改为同码拒绝。
    """
    env = {
        ACCELERATE_LAUNCH_ENV: "1",
        ACCELERATE_NUM_PROCESSES_ENV: "4",
    }

    with pytest.raises(ValueError) as excinfo:
        resolve_training_world_size_for_gpu_selection([0, 1], env)
    assert "multigpu_num_processes_selection_mismatch" in str(excinfo.value)


@pytest.mark.parametrize("value", ["no", "fp16", "bf16"])
def test_accelerate_mixed_precision_env_override(value: str):
    env = {
        ACCELERATE_LAUNCH_ENV: "1",
        ACCELERATE_MIXED_PRECISION_ENV: value,
    }
    cmd = accelerate_training_command_prefix("python", "train.py", env)
    assert resolve_accelerate_mixed_precision(env) == value
    assert cmd[cmd.index("--mixed_precision") + 1] == value


def test_accelerate_mixed_precision_defaults_to_bf16():
    assert resolve_accelerate_mixed_precision({}) == "bf16"


def test_accelerate_mixed_precision_rejects_invalid_value():
    with pytest.raises(ValueError):
        resolve_accelerate_mixed_precision({ACCELERATE_MIXED_PRECISION_ENV: "fp4"})


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_accelerate_num_processes_rejects_invalid_values(value: str):
    with pytest.raises(ValueError):
        resolve_accelerate_num_processes({ACCELERATE_NUM_PROCESSES_ENV: value})
