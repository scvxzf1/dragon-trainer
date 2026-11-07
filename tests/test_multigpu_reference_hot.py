"""Hot/real-training entry gate (t5).

These checks pin the *preconditions* of the bounded Anima run (budget, fresh
output root, explicit release) without starting it. The run itself belongs to the
later verification task and needs separate approval.
"""

from __future__ import annotations

import pytest

from bench.multigpu_reference import experiments as exp
from bench.multigpu_reference import hot_run as hot


def test_real_training_entry_refuses_without_release(tmp_path):
    with pytest.raises(exp.ExperimentRefused) as excinfo:
        exp.plan_experiment(
            "anima-short",
            kind="real_training",
            world_size=2,
            steps=20,
            output_root=tmp_path,
            env={},
        )
    assert excinfo.value.code == exp.CODE_NOT_RELEASED


def test_real_training_entry_accepts_only_a_fresh_gated_root(tmp_path):
    plan = exp.plan_experiment(
        "anima-short",
        kind="real_training",
        world_size=2,
        steps=20,
        output_root=tmp_path / "run-a",
        env={exp.RELEASE_ENV: "1"},
    )
    assert plan.gated and plan.steps == 20


@pytest.mark.hardware
def test_real_training_is_not_started_by_the_test_suite():
    pytest.skip("bounded Anima run is executed only by the verification task")


# ---- RUN-A/B/C 入口预检（mock，不启动训练）----


@pytest.fixture()
def _assets(tmp_path):
    train = tmp_path / "train.py"
    train.write_text("# fake entry\n", encoding="utf-8")
    config = tmp_path / "run.toml"
    config.write_text("[base]\n", encoding="utf-8")
    base = tmp_path / "anima-base.safetensors"
    base.write_bytes(b"0" * 16)
    return {"train_script": str(train), "config": str(config), "base": str(base)}


def test_multi_gpu_runs_use_an_accelerate_launch_prefix(_assets, tmp_path):
    fixture = tmp_path / "l3.jsonl"
    fixture.write_text(
        '{"sample_id": "s0", "noise": [0.1, 0.2], "timestep": 0.5}\n',
        encoding="utf-8",
    )
    for name in ("RUN-B", "RUN-C"):
        plan = hot.plan_hot_run(
            name,
            train_script=_assets["train_script"],
            config=_assets["config"],
            base_assets=[_assets["base"]],
            l3_fixture=str(fixture),
            project_root=tmp_path,
            env={hot.HOT_ENV: "1"},
            run_id=f"{name.lower()}-cmd",
        )
        command = " ".join(plan.command)
        assert "accelerate" in command and "launch" in command
        assert "--num_processes" in plan.command
        index = plan.command.index("--num_processes")
        assert plan.command[index + 1] == "2"
        assert plan.facts["launch_env"]["ANIMA_ACCELERATE_NUM_PROCESSES"] == "2"
        assert plan.facts["world_size"] == 2


def test_single_gpu_run_does_not_use_accelerate_launch(_assets, tmp_path):
    fixture = tmp_path / "l3.jsonl"
    fixture.write_text(
        '{"sample_id": "s0", "noise": [0.1, 0.2], "timestep": 0.5}\n',
        encoding="utf-8",
    )
    plan = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(fixture),
        project_root=tmp_path,
        env={hot.HOT_ENV: "1"},
        run_id="run-a-cmd",
    )
    assert "accelerate" not in " ".join(plan.command)
    assert plan.facts["launch_env"]["ANIMA_ACCELERATE_LAUNCH"] == "0"


def test_run_c_counts_resume_steps_against_the_budget():
    assert hot.RUN_SPECS["RUN-C"].steps == 10
    assert hot.RUN_SPECS["RUN-C"].resume_steps == 10
    assert hot.RUN_SPECS["RUN-C"].budgeted_updates == 20
    assert hot.RUN_SPECS["RUN-B"].budgeted_updates == 20
    assert hot.RUN_SPECS["RUN-A"].budgeted_updates == 10


def test_run_specs_cover_the_three_approved_shapes():
    assert set(hot.RUN_SPECS) == {"RUN-A", "RUN-B", "RUN-C"}
    assert hot.RUN_SPECS["RUN-A"].steps == 10 and hot.RUN_SPECS["RUN-A"].world_size == 1
    assert hot.RUN_SPECS["RUN-B"].steps == 20 and hot.RUN_SPECS["RUN-B"].world_size == 2
    assert hot.RUN_SPECS["RUN-C"].steps == 10 and hot.RUN_SPECS["RUN-C"].world_size == 2


def test_hot_entry_refuses_without_release(_assets, tmp_path):
    with pytest.raises(hot.HotRunRefused) as excinfo:
        hot.plan_hot_run(
            "RUN-A",
            train_script=_assets["train_script"],
            config=_assets["config"],
            base_assets=[_assets["base"]],
            project_root=tmp_path,
            env={},
        )
    assert excinfo.value.code == hot.CODE_HOT_NOT_RELEASED


def test_hot_entry_refuses_missing_base_assets(_assets, tmp_path):
    with pytest.raises(hot.HotRunRefused) as excinfo:
        hot.plan_hot_run(
            "RUN-B",
            train_script=_assets["train_script"],
            config=_assets["config"],
            base_assets=[_assets["base"], str(tmp_path / "absent.safetensors")],
            project_root=tmp_path,
            env={hot.HOT_ENV: "1"},
        )
    assert excinfo.value.code == hot.CODE_BASE_ASSET_MISSING


def test_hot_entry_refuses_user_output_tree_as_root(_assets, tmp_path):
    with pytest.raises(hot.HotRunRefused) as excinfo:
        hot.plan_hot_run(
            "RUN-A",
            train_script=_assets["train_script"],
            config=_assets["config"],
            base_assets=[_assets["base"]],
            output_root_parent="output/runs",
            project_root=tmp_path,
            env={hot.HOT_ENV: "1"},
        )
    assert excinfo.value.code == hot.CODE_OUTPUT_ROOT_INVALID


def test_total_update_budget_is_enforced(_assets, tmp_path):
    with pytest.raises(hot.HotRunRefused) as excinfo:
        hot.plan_hot_run(
            "RUN-B",
            train_script=_assets["train_script"],
            config=_assets["config"],
            base_assets=[_assets["base"]],
            project_root=tmp_path,
            env={hot.HOT_ENV: "1"},
            prior_updates=40,  # 40 + 20 > 50
        )
    assert excinfo.value.code == hot.CODE_BUDGET_EXCEEDED


def test_run_without_l3_fixture_is_blocked_not_green(_assets, tmp_path):
    plan = hot.plan_hot_run(
        "RUN-B",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        project_root=tmp_path,
        env={hot.HOT_ENV: "1"},
    )
    assert plan.status == "blocked"
    assert plan.code == hot.CODE_L3_FIXTURE_MISSING
    assert "L1/L2 candidate" in plan.detail
    with pytest.raises(hot.HotRunRefused):
        plan.raise_if_not_ready()


def test_run_with_l3_fixture_is_ready_with_fixed_output_root(_assets, tmp_path):
    fixture = tmp_path / "noise_timestep_fixture.jsonl"
    fixture.write_text(
        '{"sample_id": "s0", "noise": [0.1, 0.2], "timestep": 0.5}\n',
        encoding="utf-8",
    )
    plan = hot.plan_hot_run(
        "RUN-B",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(fixture),
        project_root=tmp_path,
        env={hot.HOT_ENV: "1"},
        run_id="run-b-test",
    )
    assert plan.status == "ready"
    assert plan.output_root == tmp_path / hot.OUTPUT_PARENT / "run-b-test"
    assert "--max_train_steps" in plan.command and "20" in plan.command
    assert plan.facts["no_download"] and plan.facts["no_preprocessing"]
    assert plan.facts["updates_after_this_run"] == 20


def test_run_ids_are_unique_per_run(_assets, tmp_path):
    fixture = tmp_path / "l3.jsonl"
    fixture.write_text(
        '{"sample_id": "s0", "noise": [0.1, 0.2], "timestep": 0.5}\n',
        encoding="utf-8",
    )
    first = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(fixture),
        project_root=tmp_path,
        env={hot.HOT_ENV: "1"},
    )
    second = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(fixture),
        project_root=tmp_path,
        env={hot.HOT_ENV: "1"},
    )
    assert first.run_id != second.run_id


def test_execute_dry_run_spawns_nothing(_assets, tmp_path):
    fixture = tmp_path / "l3.jsonl"
    fixture.write_text(
        '{"sample_id": "s0", "noise": [0.1, 0.2], "timestep": 0.5}\n',
        encoding="utf-8",
    )
    plan = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(fixture),
        project_root=tmp_path,
        env={hot.HOT_ENV: "1"},
        run_id="run-a-dry",
    )
    dry = hot.execute(plan)
    assert dry["phases"][0]["ran"] is False  # nothing spawned without a runner




# ---- RUN-C 两段必须是真实可执行的命令（t25 R3-F4） ----


@pytest.mark.parametrize("name", ["RUN-A", "RUN-B", "RUN-C"])
def test_command_uses_real_cli_flags_only(_assets, tmp_path, name):
    """Every flag must exist in ``cli_args``; the old command invented ``--resume_from_checkpoint``."""
    plan = hot.plan_hot_run(
        name,
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
    )
    # Only the trainer's own argv: tokens before the script belong to
    # ``accelerate launch``, which has its own option set.
    script_at = plan.command.index(str(_assets["train_script"]))
    flags = [
        part for part in plan.command[script_at + 1 :] if str(part).startswith("--")
    ]
    known = _cli_flags()
    unknown = sorted({flag for flag in flags if flag not in known})
    assert unknown == [], f"{name} uses flags the trainer does not have: {unknown}"


def test_run_c_commit_phase_writes_a_state_at_the_committed_step(_assets, tmp_path):
    plan = hot.plan_hot_run(
        "RUN-C",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
    )
    commit, resume = plan.facts["phases"]
    spec = hot.RUN_SPECS["RUN-C"]

    assert commit["command"].count("--save_state") == 1
    assert _value(commit["command"], "--save_every_n_steps") == str(spec.steps)
    assert commit["expected_committed_step"] == spec.steps
    # The interrupt point is the state directory the trainer really writes: derived
    # from the trainer's own STEP_STATE_NAME, not from a hand-written copy of the
    # pattern (review R3-F4).
    from library.training.checkpoints import DEFAULT_STEP_NAME, STEP_STATE_NAME, default_if_none

    expected_dir = STEP_STATE_NAME.format(
        default_if_none(hot.RUN_OUTPUT_NAME, DEFAULT_STEP_NAME), spec.steps
    )
    assert commit["stop_when_state_dir"] == str(plan.output_root / expected_dir)
    assert str(plan.output_root) in commit["stop_when_state_dir"]


def test_run_c_resume_phase_uses_the_real_resume_flag_on_that_state(_assets, tmp_path):
    plan = hot.plan_hot_run(
        "RUN-C",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
    )
    commit, resume = plan.facts["phases"]

    assert "--resume_from_checkpoint" not in resume["command"]
    resume_dir = _value(resume["command"], "--resume")
    assert resume_dir == commit["stop_when_state_dir"]
    assert resume["resume_from"] == resume_dir
    # Same scheduler horizon in both phases: the schedule must not change on resume.
    assert _value(commit["command"], "--max_train_steps") == _value(
        resume["command"], "--max_train_steps"
    )
    assert _value(resume["command"], "--max_train_steps") == str(
        hot.RUN_SPECS["RUN-C"].budgeted_updates
    )


def test_horizon_and_lr_are_pinned_across_runs(_assets, tmp_path):
    """A Δθ comparison must not measure a different LR schedule or seed."""
    for name in ("RUN-A", "RUN-B", "RUN-C"):
        plan = hot.plan_hot_run(
            name,
            train_script=_assets["train_script"],
            config=_assets["config"],
            base_assets=[_assets["base"]],
            l3_fixture=str(_write_fixture(tmp_path)),
            env={hot.HOT_ENV: "1"},
            project_root=tmp_path,
        )
        for phase in plan.facts["phases"]:
            assert _value(phase["command"], "--lr_scheduler") == "constant"
            assert _value(phase["command"], "--lr_warmup_steps") == "0"
            assert _value(phase["command"], "--seed") == str(hot.SEED)
            assert _value(phase["command"], "--max_data_loader_n_workers") == "0"
            assert _value(phase["command"], "--output_name") == hot.RUN_OUTPUT_NAME
            # S-0: the same seedable sampler in every run, otherwise the per-sample
            # slots differ across world sizes and the comparison is invalid.
            assert "--multigpu_seedable_sampler" in phase["command"]
        assert plan.facts["seedable_sampler"] is True


def test_l3_fixture_is_handed_to_the_training_path(_assets, tmp_path):
    fixture = _write_fixture(tmp_path)
    plan = hot.plan_hot_run(
        "RUN-B",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(fixture),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
    )
    from library.training.multigpu import noise_fixture

    # The gate and the trainer must read the same schema *and* the same file.
    assert plan.facts["launch_env"][noise_fixture.ENV] == str(fixture)
    assert noise_fixture.active({"ANIMA_MULTIGPU_NOISE_FIXTURE": str(fixture)}) is not None
    # A real training batch carries no ``image_keys`` (that key only exists in the
    # do-not-train debug mode), so the run entry must select the cursor-derived ids
    # or the fixture would fail closed on its very first batch.
    assert plan.facts["launch_env"][noise_fixture.ENV_IDS] == noise_fixture.IDS_CURSOR
    assert plan.facts["l3_ids_mode"] == noise_fixture.IDS_CURSOR


def test_execute_passes_the_stop_directory_to_the_runner(_assets, tmp_path):
    plan = hot.plan_hot_run(
        "RUN-C",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
    )
    calls = []

    class _Done:
        returncode = 0

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return _Done()

    result = hot.execute(plan, runner=runner)
    assert len(calls) == 2
    assert calls[0][1]["stop_when_state_dir"].endswith("-step00000010-state")
    assert "stop_when_state_dir" not in calls[1][1]
    assert result["phases"][0]["stop_when_state_dir"] is not None


def _write_fixture(tmp_path):
    fixture = tmp_path / "l3.jsonl"
    fixture.write_text(
        '{"sample_id": "s0", "noise": [0.1, 0.2], "timestep": 0.5}\n',
        encoding="utf-8",
    )
    return fixture


def _cli_flags() -> set[str]:
    """Every long option the trainer really parses."""
    from library.training import cli_args

    parser = cli_args.setup_parser()
    flags = set()
    for action in parser._actions:
        flags.update(action.option_strings)
    return flags


def _value(command, flag):
    return command[command.index(flag) + 1]


# ---------------------------------------------------------------------------
# S-0：single/dual permutation 一致性必须由显式 opt-in 门提供（t3 F-T3-1a）
# ---------------------------------------------------------------------------

_SLOT_SAMPLES = 12
_SLOT_SEED = 1234


def _seedable_env(tmp_path, *, gate: bool):
    import os

    env = dict(os.environ)
    env.pop("ANIMA_MULTIGPU_SEEDABLE_SAMPLER", None)
    env.pop("ANIMA_MULTIGPU_SEEDABLE_SAMPLER_SEED", None)
    if gate:
        env["ANIMA_MULTIGPU_SEEDABLE_SAMPLER"] = "1"
    env["PYTHONPATH"] = str(tmp_path.parent if False else os.getcwd())
    return env


def _single_stream(tmp_path, *, gate: bool):
    import json
    import os
    import subprocess
    import sys

    out = tmp_path / f"single-{int(gate)}.json"
    code = (
        "from tests.multigpu_reference_support import loader_stream_single as f;"
        f"f({_SLOT_SAMPLES}, 1, {_SLOT_SEED}, {str(out)!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env=_seedable_env(tmp_path, gate=gate),
        cwd=os.getcwd(),
        check=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    return [batch[0] for batch in payload["batches"]]


def _dual_stream(tmp_path, *, gate: bool):
    """Two real Gloo workers; the global order is the rank-interleaved stream."""
    import json
    import os

    import torch.multiprocessing as mp

    from tests import multigpu_reference_support as support

    out_dir = tmp_path / f"dual-{int(gate)}"
    out_dir.mkdir()
    for key in ("ANIMA_MULTIGPU_SEEDABLE_SAMPLER", "ANIMA_MULTIGPU_SEEDABLE_SAMPLER_SEED"):
        os.environ.pop(key, None)
    if gate:
        os.environ["ANIMA_MULTIGPU_SEEDABLE_SAMPLER"] = "1"
    try:
        mp.spawn(
            support.loader_stream_rank,
            args=(2, _SLOT_SAMPLES, 1, _SLOT_SEED, support.free_port(), str(out_dir)),
            nprocs=2,
            join=True,
        )
    finally:
        os.environ.pop("ANIMA_MULTIGPU_SEEDABLE_SAMPLER", None)
    ranks = [
        json.loads((out_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert all(rank["error"] is None for rank in ranks), ranks
    assert all(len(rank["batches"]) > 0 for rank in ranks), ranks
    stream: list[int] = []
    for index in range(len(ranks[0]["batches"])):
        stream.append(ranks[0]["batches"][index]["values"][0])
        stream.append(ranks[1]["batches"][index]["values"][0])
    return stream


def test_seedable_sampler_gate_is_off_by_default(monkeypatch):
    """Default: nothing is passed to Accelerator, so today's behaviour is unchanged."""
    import types

    import torch

    from library.runtime import accelerator as accel_mod
    from library.training.multigpu import loader_seed

    monkeypatch.delenv(loader_seed.ENV, raising=False)
    monkeypatch.delenv(loader_seed.SEED_ENV, raising=False)
    args = types.SimpleNamespace(seed=_SLOT_SEED, multigpu_seedable_sampler=False)
    assert loader_seed.enabled(args, {}) is False
    assert loader_seed.dataloader_config_kwargs(args, {}) == {}

    captured: dict = {}

    class _FakeAcc:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_processes = 1
            self.process_index = 0
            self.is_main_process = True
            self.device = torch.device("cpu")
            self.distributed_type = "NO"
            self.scaler = None
            self.state = types.SimpleNamespace(distributed_type="NO")

        def print(self, *a, **k):
            pass

    monkeypatch.setattr(accel_mod, "Accelerator", _FakeAcc)
    fake_args = types.SimpleNamespace(
        log_prefix=None,
        logging_dir=str(__import__("pathlib").Path("/tmp") / "seedable-gate-test"),
        log_with=None,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        adaptive_oom_retry_resume=False,
        multigpu_startup_probe=False,
        multigpu_seedable_sampler=False,
        seed=_SLOT_SEED,
        method=None,
        preset=None,
        output_dir=None,
    )
    accel_mod.prepare_accelerator(fake_args)
    assert "dataloader_config" not in captured  # unchanged default path


def test_seedable_sampler_gate_is_opt_in_and_merges(monkeypatch):
    import types

    import torch

    from library.runtime import accelerator as accel_mod
    from library.training.multigpu import loader_seed

    monkeypatch.setenv(loader_seed.ENV, "1")
    captured: dict = {}

    class _FakeAcc:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_processes = 1
            self.process_index = 0
            self.is_main_process = True
            self.device = torch.device("cpu")
            self.distributed_type = "NO"
            self.scaler = None
            self.state = types.SimpleNamespace(distributed_type="NO")

        def print(self, *a, **k):
            pass

    monkeypatch.setattr(accel_mod, "Accelerator", _FakeAcc)
    fake_args = types.SimpleNamespace(
        log_prefix=None,
        logging_dir=str(__import__("pathlib").Path("/tmp") / "seedable-gate-test"),
        log_with=None,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        # both switches must survive in the single DataLoaderConfiguration
        adaptive_oom_retry_resume=True,
        multigpu_startup_probe=False,
        multigpu_seedable_sampler=False,
        seed=_SLOT_SEED,
        method=None,
        preset=None,
        output_dir=None,
    )
    accel_mod.prepare_accelerator(fake_args)
    config = captured["dataloader_config"]
    assert config.use_seedable_sampler is True
    assert config.data_seed == _SLOT_SEED
    assert config.use_stateful_dataloader is True


def test_single_and_dual_permutations_match_only_with_the_gate(tmp_path):
    """The S-0 premise: with the gate on, both world sizes see the same order."""
    gated_single = _single_stream(tmp_path, gate=True)
    gated_dual = _dual_stream(tmp_path, gate=True)
    assert gated_single == gated_dual, (gated_single, gated_dual)
    assert sorted(gated_single) == list(range(_SLOT_SAMPLES))  # one full permutation

    # Negative control: without the gate the two branches shuffle differently, so
    # the equality above is the gate's doing and not a tautology.
    ungated_single = _single_stream(tmp_path, gate=False)
    ungated_dual = _dual_stream(tmp_path, gate=False)
    assert ungated_single != ungated_dual, (ungated_single, ungated_dual)


# ---------------------------------------------------------------------------
# 单卡 batch_size=2 必须可通过配置表达（t3 F-T3-1b/F-T3-7）
# ---------------------------------------------------------------------------


def _group(*batch_sizes):
    class _Ds:
        def __init__(self, size):
            self.batch_size = size

    class _Group:
        def __init__(self, sizes):
            self.datasets = [_Ds(size) for size in sizes]

    return _Group(batch_sizes)


def test_loader_batch_plan_keeps_the_wrapper_at_one_and_reads_the_blueprint():
    import types

    from library.training.dataloader_batch import (
        DatasetBatchError,
        resolve_loader_batch,
    )

    default = resolve_loader_batch(types.SimpleNamespace(), _group(1, 1))
    assert default.loader_batch_size == 1  # the dataset item already is a batch
    assert default.per_device_batch_size == 1

    batch2 = resolve_loader_batch(
        types.SimpleNamespace(train_batch_size=2), _group(2)
    )
    assert batch2.loader_batch_size == 1
    assert batch2.per_device_batch_size == 2

    with pytest.raises(DatasetBatchError) as excinfo:
        resolve_loader_batch(types.SimpleNamespace(train_batch_size=2), _group(1))
    assert excinfo.value.code == "dataset_batch_size_mismatch"

    with pytest.raises(DatasetBatchError):
        resolve_loader_batch(types.SimpleNamespace(), _group(1, 2))


def test_single_batch2_and_dual_batch1_cover_the_same_window_multiset():
    """The RUN-A premise: one worker at batch 2 == two workers at batch 1 per update."""
    from library.training.multigpu import noise_fixture as fx

    for update in range(4):
        single_window = fx.sample_ids_from_position(
            local_batch_index=update, rank=0, world_size=1, batch_size=2
        )
        dual_window = (
            fx.sample_ids_from_position(
                local_batch_index=update, rank=0, world_size=2, batch_size=1
            )
            + fx.sample_ids_from_position(
                local_batch_index=update, rank=1, world_size=2, batch_size=1
            )
        )
        assert single_window == dual_window, (update, single_window, dual_window)


# ---------------------------------------------------------------------------
# t13 F-T13-3：真实执行链的启动期形状（解释器前缀 + Popen-safe env）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["RUN-A", "RUN-B", "RUN-C"])
def test_run_command_starts_with_an_absolute_interpreter(_assets, tmp_path, name):
    """argv[0] must be a real interpreter path, never the bare name ``python``.

    t13's second attempt died with ``ModuleNotFoundError: torch`` because the child
    resolved ``python`` through its own PATH.
    """
    import os
    import sys

    plan = hot.plan_hot_run(
        name,
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
        run_id=f"{name.lower()}-interp",
    )
    assert plan.command[0] != "python"  # the old, broken prefix
    assert os.path.isabs(plan.command[0])
    assert plan.command[0] == sys.executable
    assert plan.facts["interpreter"] == sys.executable


def test_the_pinned_interpreter_can_import_torch(_assets, tmp_path):
    """The prefix must name a python that actually has the project's deps."""
    import subprocess

    plan = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
        run_id="run-a-torch",
    )
    completed = subprocess.run(
        [plan.command[0], "-c", "import torch, sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == plan.command[0]


def test_explicit_interpreter_is_used_and_a_bare_name_is_refused(_assets, tmp_path):
    import sys

    plan = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
        run_id="run-a-explicit",
        python=sys.executable,
    )
    assert plan.command[0] == sys.executable

    with pytest.raises(hot.HotRunRefused) as excinfo:
        hot.plan_hot_run(
            "RUN-A",
            train_script=_assets["train_script"],
            config=_assets["config"],
            base_assets=[_assets["base"]],
            l3_fixture=str(_write_fixture(tmp_path)),
            env={hot.HOT_ENV: "1"},
            project_root=tmp_path,
            run_id="run-a-bare",
            python="python",
        )
    assert excinfo.value.code == hot.CODE_INTERPRETER_INVALID


def test_single_run_omits_the_process_count_key(_assets, tmp_path):
    """A key that does not apply is absent, not ``None`` (Popen TypeError)."""
    plan = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
        run_id="run-a-env",
    )
    assert "ANIMA_ACCELERATE_NUM_PROCESSES" not in plan.facts["launch_env"]
    assert plan.facts["launch_env"]["ANIMA_ACCELERATE_LAUNCH"] == "0"
    assert all(
        isinstance(value, str) for value in plan.facts["launch_env"].values()
    )


def test_multi_run_keeps_the_process_count_as_a_string(_assets, tmp_path):
    plan = hot.plan_hot_run(
        "RUN-B",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
        run_id="run-b-env",
    )
    assert plan.facts["launch_env"]["ANIMA_ACCELERATE_NUM_PROCESSES"] == "2"
    assert plan.facts["launch_env"]["ANIMA_ACCELERATE_LAUNCH"] == "1"


def test_clean_launch_env_drops_unset_entries_and_stringifies():
    assert hot.clean_launch_env(
        {"A": None, "B": 2, "C": "x", "D": True}
    ) == {"B": "2", "C": "x", "D": "True"}


@pytest.mark.parametrize("name", ["RUN-A", "RUN-B"])
def test_execute_hands_a_popen_safe_env_to_a_real_runner(_assets, tmp_path, name):
    """No ``None``/non-str values reach ``Popen``; the base env is not lost."""
    plan = hot.plan_hot_run(
        name,
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
        run_id=f"{name.lower()}-envreal",
    )
    seen = []

    class _Done:
        returncode = 0

    def runner(command, **kwargs):
        seen.append(kwargs["env"])
        return _Done()

    hot.execute(plan, runner=runner)
    assert seen
    for env in seen:
        assert all(isinstance(key, str) for key in env)
        assert all(isinstance(value, str) for value in env.values())
        assert env["ANIMA_ACCELERATE_LAUNCH"] == plan.facts["launch_env"][
            "ANIMA_ACCELERATE_LAUNCH"
        ]
        assert "PATH" in env  # layered over this process's environment
    if name == "RUN-A":
        assert "ANIMA_ACCELERATE_NUM_PROCESSES" not in seen[0]
    else:
        assert seen[0]["ANIMA_ACCELERATE_NUM_PROCESSES"] == "2"


def test_mock_execute_reports_the_env_keys_without_spawning(_assets, tmp_path):
    plan = hot.plan_hot_run(
        "RUN-A",
        train_script=_assets["train_script"],
        config=_assets["config"],
        base_assets=[_assets["base"]],
        l3_fixture=str(_write_fixture(tmp_path)),
        env={hot.HOT_ENV: "1"},
        project_root=tmp_path,
        run_id="run-a-mockenv",
    )
    dry = hot.execute(plan)
    assert dry["phases"][0]["ran"] is False
    env_keys = dry["phases"][0]["env_keys"]
    assert "ANIMA_ACCELERATE_LAUNCH" in env_keys
    # A single worker run carries no process-count key at all.
    assert "ANIMA_ACCELERATE_NUM_PROCESSES" not in env_keys
