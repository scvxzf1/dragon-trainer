"""Per-rank PP checkpoint acceptance (task t5).

What this file proves
---------------------
1. **A save -> process restart -> load roundtrip is bit-exact.**  Two Gloo ranks
   on CPU take real optimizer steps, save into a new run root, and a *second,
   freshly spawned* pair of processes rebuilds the fixture and restores.  The
   restored adapter weights, optimizer state, fp16 state and step counter are
   compared with ``torch.equal`` (tensors) and ``==`` (scalars) against the state
   the first pair recorded.
2. **A resume replays exactly.**  The ``benchmark.verify_resume`` shape
   (save -> step -> load -> step) must yield an adapter ``max_abs`` of exactly
   zero and a loss delta of exactly zero.
3. **Every mismatch is a named refusal.**  A changed block range, stage count,
   rank count, run identity, artifact bytes, missing manifest/artifact, or a
   different step/skip decision raises ``pp_checkpoint_*`` -- never a silent
   accept.
4. **Saving cannot overwrite anything.**  The run root must be newly created, the
   artifacts stay inside it, and pre-existing training output is untouched.
5. **Only stage-local adapters are captured.**  The snapshot of a stage contains
   the adapters attached to the blocks it owns, and a stage that owns trainable
   state outside an adapter module is refused instead of silently dropping it.

The single-process cases run everywhere; the roundtrip case is a two-process
Gloo test (``integration``/``probe``), like the rest of the PP acceptance suite.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training.pp import (
    CONDITIONING_LOCAL,
    PipelineParallelError,
    PipelineRuntimeConfig,
    assert_snapshot_digests_match,
    assert_snapshot_roundtrip,
    build_anima_pipeline_runtime,
    build_anima_stage_module,
    capture_stage_snapshot,
    compare_snapshots,
    content_hash_file,
    content_hash_object,
    content_hash_tensor,
    content_hash_tensors,
    create_run_root,
    load_distributed_checkpoint,
    read_manifest,
    read_stage_checkpoint,
    save_distributed_checkpoint,
    save_stage_checkpoint,
    validate_identity,
)
from library.training.pp import checkpoint as checkpoint_module
from library.training.pp.errors import (
    CODE_CHECKPOINT_IDENTITY_MISMATCH,
    CODE_CHECKPOINT_INCOMPLETE,
    CODE_CHECKPOINT_RUN_ROOT_NOT_NEW,
    CODE_CHECKPOINT_STATE_MISMATCH,
    CODE_CHECKPOINT_STEP_MISMATCH,
    CODE_CHECKPOINT_TOPOLOGY_MISMATCH,
    CODE_SCALER_DECISION_ASYMMETRY,
)
from tests.pp_checkpoint_test_support import (
    SAVE_AFTER_STEPS,
    run_identity,
    run_pair,
    run_pair_expect_failure,
    stage_lora_parameters,
    tiny_lora,
    tiny_model,
    worker_restore,
    worker_save,
)

IDENTITY = {"base": "a" * 64}
FAKE_TOPOLOGY = {
    "family": "anima",
    "stage_index": 0,
    "stages": 2,
    "block_range": [0, 2],
    "num_blocks": 4,
    "is_first": True,
    "is_last": False,
    "microbatches": 2,
    "schedule": "1f1b",
    "conditioning": "local",
    "mixed_precision": "fp16",
    "latent_shape": [4, 4, 1, 4, 4],
    "token_family": "4200",
    "frozen_input_indices": [],
    "frozen_output_indices": [],
}


# ---------------------------------------------------------------------------
# fake runtime (a stage with exactly one adapter, one optimizer, one transaction)
# ---------------------------------------------------------------------------


class _FakeAdapter(torch.nn.Module):
    def __init__(self, name: str, *, seed: int = 0) -> None:
        super().__init__()
        self.lora_name = name
        generator = torch.Generator().manual_seed(seed)
        self.lora_down = torch.nn.Linear(2, 2, bias=False)
        self.lora_up = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    def forward(self, hidden):  # pragma: no cover - never called
        return hidden


class _FakeStage(torch.nn.Module):
    """Mirrors ``_AnimaStageBase``: adapters are held non-owningly."""

    def __init__(self, *, extra_trainable: torch.nn.Parameter | None = None) -> None:
        super().__init__()
        self.adapter_modules = (_FakeAdapter("lora_unet_blocks_0_mlp.0"),)
        self.extra_trainable = extra_trainable

    def trainable_parameters(self):
        for parameter in self.adapter_modules[0].parameters():
            yield parameter
        if self.extra_trainable is not None:
            yield self.extra_trainable


class _FakeTransaction:
    def __init__(self, *, scale: float = 8.0, growth_tracker: int = 3) -> None:
        self.scale = float(scale)
        self.growth_tracker = int(growth_tracker)
        self.last_skip: bool | None = False
        self.scaler = None

    def restore(self, *, scale, growth_tracker=0, last_skip=None, where=None):
        self.scale = float(scale)
        self.growth_tracker = int(growth_tracker)
        self.last_skip = last_skip


def _fake_runtime(*, extra_trainable: bool = False, topology: dict | None = None):
    stage = _FakeStage(
        extra_trainable=(
            torch.nn.Parameter(torch.ones(2)) if extra_trainable else None
        )
    )
    parameters = list(stage.adapter_modules[0].parameters())
    if stage.extra_trainable is not None:
        parameters.append(stage.extra_trainable)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    runtime = SimpleNamespace(
        optimizer=optimizer,
        stage_module=stage,
        transaction=_FakeTransaction(),
        topology=lambda: dict(topology or FAKE_TOPOLOGY),
    )
    return runtime


def _snapshot(runtime, *, step: int = 3, identity: dict | None = None):
    return capture_stage_snapshot(
        runtime, step=step, identity=IDENTITY if identity is None else identity
    )


def _save(runtime, root, *, step: int = 3, identity: dict | None = None):
    return save_distributed_checkpoint(
        root, runtime, step=step, identity=IDENTITY if identity is None else identity
    )


def _rewrite_manifest(root: Path, mutate) -> None:
    path = Path(root) / "pp_checkpoint.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# content hashing: content, never metadata
# ---------------------------------------------------------------------------


def test_content_hashes_track_content_not_metadata(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"one")
    first = content_hash_file(path)
    # A copy that keeps the bytes but changes mtime/size must hash the same: the
    # fingerprint this replaces hashed mtime_ns and the absolute path.
    path.touch()
    assert content_hash_file(path) == first
    shutil.copyfile(path, tmp_path / "copy.bin")
    assert content_hash_file(tmp_path / "copy.bin") == first
    path.write_bytes(b"two")
    assert content_hash_file(path) != first

    base = torch.arange(6.0).reshape(2, 3)
    assert content_hash_tensor(base) == content_hash_tensor(base.clone())
    assert content_hash_tensor(base) != content_hash_tensor(base + 1)
    assert content_hash_tensor(base) != content_hash_tensor(base.to(torch.float64))
    assert content_hash_tensor(base) != content_hash_tensor(base.reshape(3, 2))
    assert content_hash_tensors({"a": base}) != content_hash_tensors({"b": base})
    # float.hex keeps the digest exact: two scales that differ in the last bit
    # must not collapse to the same repr.
    assert content_hash_object({"scale": 1.0}) != content_hash_object(
        {"scale": float.fromhex("0x1.0000000000001p+0")}
    )


def test_identity_only_accepts_content_hashes():
    assert validate_identity(None) == {}
    assert validate_identity({"base": "b" * 64}) == {"base": "b" * 64}
    for identity in (
        {"base": "/mnt/models/anima.safetensors"},
        {"base": "1700000000.123"},
        {"base": {"path": "/mnt/models/anima.safetensors", "size": 12}},
        {"/mnt/models/anima.safetensors": "c" * 64},
        {"base": 12345},
    ):
        with pytest.raises(PipelineParallelError) as excinfo:
            validate_identity(identity)
        assert excinfo.value.code == CODE_CHECKPOINT_IDENTITY_MISMATCH
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_identity(["base"])
    assert excinfo.value.code == CODE_CHECKPOINT_IDENTITY_MISMATCH


# ---------------------------------------------------------------------------
# single-process roundtrip on a fake stage
# ---------------------------------------------------------------------------


def test_fake_runtime_roundtrip_is_bit_exact(tmp_path):
    runtime = _fake_runtime()
    root = tmp_path / "run"
    result = _save(runtime, root, step=7)
    assert result.step == 7
    assert sorted(path.name for path in root.iterdir()) == [
        "pp_checkpoint.json",
        "pp_stage-rank0.pt",
    ]
    manifest = read_manifest(root)
    assert manifest["world_size"] == 1
    assert manifest["rank_local_keys"] == {"0": ["pp_stage-rank0.pt"]}

    before = _snapshot(runtime, step=7)
    # Wreck the live state, then restore it from disk.
    with torch.no_grad():
        for parameter in runtime.stage_module.adapter_modules[0].parameters():
            parameter.zero_()
    runtime.transaction.scale = 1.0
    runtime.transaction.growth_tracker = 0

    state = load_distributed_checkpoint(root, runtime, identity=IDENTITY)
    assert state.step == 7
    after = capture_stage_snapshot(runtime, step=state.step, identity=IDENTITY)
    report = compare_snapshots(before, after)
    assert report["equal"], report
    assert report["adapter_max_abs"] == 0.0
    assert report["step_delta"] == 0
    assert before.digest() == after.digest()
    assert_snapshot_roundtrip(before, after)


def test_state_drift_is_reported_and_refused(tmp_path):
    runtime = _fake_runtime()
    root = tmp_path / "run"
    _save(runtime, root, step=4)
    before = _snapshot(runtime, step=4)

    # Adapter drift: one bit in one weight.
    drifted = _snapshot(runtime, step=4)
    name = next(iter(drifted.adapters))
    key = next(iter(drifted.adapters[name]))
    drifted.adapters[name][key] = drifted.adapters[name][key].clone()
    drifted.adapters[name][key].view(-1)[0] += 1e-7
    report = compare_snapshots(before, drifted)
    assert report["adapter_max_abs"] > 0
    assert not report["equal"]
    with pytest.raises(PipelineParallelError) as excinfo:
        assert_snapshot_roundtrip(before, drifted)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH

    # Optimizer drift: same weights, different AdamW moments.
    optimizer_drift = _snapshot(runtime, step=4)
    optimizer_drift.optimizer["state"][0]["exp_avg"] = torch.zeros_like(
        optimizer_drift.optimizer["state"][0]["exp_avg"]
    )
    assert not compare_snapshots(before, optimizer_drift)["optimizer_equal"]

    # Scalar drift: step counter and fp16 scale are values, not tensors.
    step_drift = _snapshot(runtime, step=5)
    assert compare_snapshots(before, step_drift)["step_delta"] == 1
    scale_drift = _snapshot(runtime, step=4)
    scale_drift = replace(scale_drift, scale=scale_drift.scale + 1.0)
    assert compare_snapshots(before, scale_drift)["scale_delta"] == 1.0


def test_snapshot_digest_mismatch_is_named():
    runtime = _fake_runtime()
    digest = _snapshot(runtime, step=3).digest()
    assert_snapshot_digests_match(digest, dict(digest))
    with pytest.raises(PipelineParallelError) as excinfo:
        assert_snapshot_digests_match(digest, {**digest, "step": 4})
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
    assert "step" in str(excinfo.value)


# ---------------------------------------------------------------------------
# run-root safety
# ---------------------------------------------------------------------------


def test_run_root_must_be_new(tmp_path):
    root = create_run_root(tmp_path / "run")
    assert root.is_dir()
    with pytest.raises(PipelineParallelError) as excinfo:
        create_run_root(tmp_path / "run")
    assert excinfo.value.code == CODE_CHECKPOINT_RUN_ROOT_NOT_NEW

    # A save never writes into an existing run root, even an empty one, and it
    # does not leave a marker there either: that directory is user data.
    runtime = _fake_runtime()
    with pytest.raises(PipelineParallelError) as excinfo:
        _save(runtime, tmp_path / "run")
    assert excinfo.value.code == CODE_CHECKPOINT_RUN_ROOT_NOT_NEW
    assert list((tmp_path / "run").iterdir()) == []

    # And an existing stage artifact is never overwritten.
    fresh = create_run_root(tmp_path / "fresh")
    snapshot = _snapshot(runtime, step=1)
    save_stage_checkpoint(fresh, snapshot, rank=0)
    with pytest.raises(PipelineParallelError) as excinfo:
        save_stage_checkpoint(fresh, snapshot, rank=0)
    assert excinfo.value.code == CODE_CHECKPOINT_RUN_ROOT_NOT_NEW


def test_save_keeps_pre_existing_output_untouched(tmp_path):
    sentinel = tmp_path / "existing-output.txt"
    sentinel.write_text("user training output")
    runtime = _fake_runtime()
    root = tmp_path / "run"
    _save(runtime, root, step=2)
    assert sentinel.read_text() == "user training output"
    assert sorted(path.name for path in root.iterdir()) == [
        "pp_checkpoint.json",
        "pp_stage-rank0.pt",
    ]
    assert not list(root.glob("*.tmp"))


# ---------------------------------------------------------------------------
# fail-closed capture
# ---------------------------------------------------------------------------


def test_trainable_state_outside_adapters_is_refused():
    runtime = _fake_runtime(extra_trainable=True)
    with pytest.raises(PipelineParallelError) as excinfo:
        _snapshot(runtime, step=1)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
    assert "outside its adapter modules" in str(excinfo.value)


def test_runtime_without_optimizer_has_nothing_to_checkpoint():
    runtime = _fake_runtime()
    runtime.optimizer = None
    with pytest.raises(PipelineParallelError) as excinfo:
        _snapshot(runtime, step=1)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH


@pytest.mark.parametrize("step", [-1, 1.0, "3", True])
def test_invalid_step_is_refused(step):
    runtime = _fake_runtime()
    with pytest.raises(PipelineParallelError) as excinfo:
        _snapshot(runtime, step=step)
    assert excinfo.value.code == CODE_CHECKPOINT_STEP_MISMATCH


def test_failed_save_publishes_no_manifest(tmp_path):
    runtime = _fake_runtime()
    runtime.optimizer = None  # capture fails on every rank
    root = tmp_path / "run"
    with pytest.raises(PipelineParallelError) as excinfo:
        _save(runtime, root, step=1)
    assert excinfo.value.code == CODE_CHECKPOINT_STATE_MISMATCH
    assert not (root / "pp_checkpoint.json").exists()
    assert (root / "pp_checkpoint.FAILED.json").exists()
    with pytest.raises(PipelineParallelError) as excinfo:
        read_manifest(root)
    assert excinfo.value.code == CODE_CHECKPOINT_INCOMPLETE


# ---------------------------------------------------------------------------
# named refusals for every mismatch
# ---------------------------------------------------------------------------


def _set_block_range(manifest):
    manifest["ranks"][0]["topology"]["block_range"] = [1, 3]


def _set_stage_count(manifest):
    manifest["stages"] = 3
    manifest["ranks"][0]["topology"]["stages"] = 3


def _set_world_size(manifest):
    manifest["world_size"] = 2


def _set_identity(manifest):
    manifest["identity"]["base"] = "b" * 64


def _set_plan_ranges(manifest):
    # The saved plan owns blocks 0..1 for stage 0; this run's plan owns 0..2.
    manifest["plan"]["ranges"][0] = [0, 1]


def _drop_rank_entry(manifest):
    manifest["ranks"] = []


def _bump_version(manifest):
    manifest["version"] = 99


@pytest.mark.parametrize(
    ("mutate", "code", "fragment"),
    [
        (_set_block_range, CODE_CHECKPOINT_TOPOLOGY_MISMATCH, "block_range"),
        (_set_stage_count, CODE_CHECKPOINT_TOPOLOGY_MISMATCH, "stages"),
        (_set_world_size, CODE_CHECKPOINT_TOPOLOGY_MISMATCH, "rank"),
        (_set_identity, CODE_CHECKPOINT_IDENTITY_MISMATCH, "identity"),
        (_set_plan_ranges, CODE_CHECKPOINT_TOPOLOGY_MISMATCH, "pipeline plan"),
        (_drop_rank_entry, CODE_CHECKPOINT_INCOMPLETE, "no artifact for rank 0"),
        (_bump_version, CODE_CHECKPOINT_INCOMPLETE, "version"),
    ],
)
def test_manifest_mismatches_are_named(tmp_path, mutate, code, fragment):
    runtime = _fake_runtime()
    root = tmp_path / "run"
    _save(runtime, root, step=5)
    _rewrite_manifest(root, mutate)
    with pytest.raises(PipelineParallelError) as excinfo:
        load_distributed_checkpoint(root, runtime, identity=IDENTITY)
    assert excinfo.value.code == code
    assert fragment in str(excinfo.value)


def test_modified_artifact_is_refused_by_content_hash(tmp_path):
    runtime = _fake_runtime()
    root = tmp_path / "run"
    _save(runtime, root, step=5)
    artifact = root / "pp_stage-rank0.pt"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(PipelineParallelError) as excinfo:
        load_distributed_checkpoint(root, runtime, identity=IDENTITY)
    assert excinfo.value.code == CODE_CHECKPOINT_IDENTITY_MISMATCH
    assert "content hash" in str(excinfo.value)


def test_missing_artifact_is_refused(tmp_path):
    runtime = _fake_runtime()
    root = tmp_path / "run"
    _save(runtime, root, step=5)
    (root / "pp_stage-rank0.pt").unlink()
    with pytest.raises(PipelineParallelError) as excinfo:
        read_stage_checkpoint(root, runtime, identity=IDENTITY)
    assert excinfo.value.code == CODE_CHECKPOINT_INCOMPLETE
    assert "missing" in str(excinfo.value)


def _peer_gather(**overrides):
    """Stand-in for ``all_gather_object`` that invents a disagreeing peer rank.

    The negotiation collective is the whole point of the resume protocol, and the
    disagreement it exists to catch cannot be produced by one process on its own.
    """

    def gather(value, group=None):
        peer = dict(value)
        peer["rank"] = int(value.get("rank", 0)) + 1
        peer.update(overrides)
        return [dict(value), peer]

    return gather


def test_step_disagreement_between_ranks_is_refused(tmp_path, monkeypatch):
    runtime = _fake_runtime()
    monkeypatch.setattr(
        checkpoint_module,
        "_all_gather_object",
        _peer_gather(step=6, sha256="0" * 64),
    )
    root = tmp_path / "run"
    with pytest.raises(PipelineParallelError) as excinfo:
        _save(runtime, root, step=5)
    assert excinfo.value.code == CODE_CHECKPOINT_STEP_MISMATCH
    # The refusal happens before the manifest is published.
    assert not (root / "pp_checkpoint.json").exists()


def test_skip_decision_disagreement_on_load_is_refused(tmp_path, monkeypatch):
    runtime = _fake_runtime()
    root = tmp_path / "run"
    _save(runtime, root, step=5)
    monkeypatch.setattr(
        checkpoint_module, "_all_gather_object", _peer_gather(last_skip=True)
    )
    with pytest.raises(PipelineParallelError) as excinfo:
        load_distributed_checkpoint(root, runtime, identity=IDENTITY)
    assert excinfo.value.code == CODE_SCALER_DECISION_ASYMMETRY


def test_a_peer_failure_aborts_the_load_before_any_rank_applies(tmp_path, monkeypatch):
    runtime = _fake_runtime()
    root = tmp_path / "run"
    _save(runtime, root, step=5)
    before = _snapshot(runtime, step=5)
    with torch.no_grad():
        for parameter in runtime.stage_module.adapter_modules[0].parameters():
            parameter.zero_()
    monkeypatch.setattr(
        checkpoint_module,
        "_all_gather_object",
        _peer_gather(error="rank 1: RuntimeError: disk went away", code=CODE_CHECKPOINT_INCOMPLETE),
    )
    with pytest.raises(PipelineParallelError) as excinfo:
        load_distributed_checkpoint(root, runtime, identity=IDENTITY)
    assert excinfo.value.code == CODE_CHECKPOINT_INCOMPLETE
    assert "disk went away" in str(excinfo.value)
    # The healthy rank must not have restored anything: it stays zeroed.
    assert all(
        bool(torch.equal(parameter, torch.zeros_like(parameter)))
        for parameter in runtime.stage_module.adapter_modules[0].parameters()
    )
    assert before.step == 5


# ---------------------------------------------------------------------------
# a real runtime: stage-local capture and a real roundtrip (single process)
# ---------------------------------------------------------------------------


def _real_runtime(rank: int = 0):
    model = tiny_model()
    network = tiny_lora(model)
    layout, _ = build_anima_stage_module(
        model,
        stage_index=rank,
        stages=2,
        conditioning=CONDITIONING_LOCAL,
        network=network,
    )
    optimizer = torch.optim.AdamW(
        stage_lora_parameters(network, layout.block_range), lr=1e-3
    )
    runtime = build_anima_pipeline_runtime(
        model,
        config=PipelineRuntimeConfig(
            stages=2, microbatches=2, conditioning=CONDITIONING_LOCAL
        ),
        world_size=2,
        rank=rank,
        device=torch.device("cpu"),
        latent_shape=(4, 4, 1, 4, 4),
        dtype=torch.float32,
        timesteps_shape=(4,),
        context_shape=(4, 3, 24),
        scaler=None,
        optimizer=optimizer,
        network=network,
        group=torch.distributed.group.WORLD,
    )
    return model, network, layout, runtime


@pytest.fixture
def single_rank_group(tmp_path):
    """A real 2-stage *rank 0* runtime needs a live group; one rank is enough.

    ``torch.distributed.pipelining.PipelineStage`` calls ``dist.get_rank(group)``
    in its constructor, so a rank-local build is impossible without an
    initialized process group.  This fixture provides a 1-rank Gloo group, which
    makes the artifact's ``world_size`` 1 while its topology still records the
    2-stage plan -- that is exactly what these single-process cases test.
    """

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'single-init'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_real_stage_captures_only_its_own_adapters(tmp_path, single_rank_group):
    from library.training.pp import adapter_block_index

    model, network, layout, runtime = _real_runtime(0)
    snapshot = capture_stage_snapshot(runtime, step=2, identity=run_identity(model))
    owned = set(layout.block_indices)
    assert snapshot.adapters, "the stage must own at least one adapter"
    for name in snapshot.adapters:
        assert adapter_block_index(name) in owned
    all_blocks = {
        adapter_block_index(lora.lora_name) for lora in network.unet_loras
    }
    assert all_blocks - owned, "the fixture must have adapters on both stages"
    assert snapshot.scaler is None  # this stage does not own the loss scale
    assert snapshot.scale is not None

    root = tmp_path / "run"
    saved = _save(runtime, root, step=2, identity=run_identity(model))
    assert saved.manifest["stages"] == 2
    state = load_distributed_checkpoint(root, runtime, identity=run_identity(model))
    assert state.step == 2
    restored = capture_stage_snapshot(runtime, step=2, identity=run_identity(model))
    assert compare_snapshots(snapshot, restored)["equal"]

    _rewrite_manifest(root, _set_block_range)
    with pytest.raises(PipelineParallelError) as excinfo:
        load_distributed_checkpoint(root, runtime, identity=run_identity(model))
    assert excinfo.value.code == CODE_CHECKPOINT_TOPOLOGY_MISMATCH


def test_identity_mismatch_between_runs_is_refused(tmp_path, single_rank_group):
    model, _network, _layout, runtime = _real_runtime(0)
    root = tmp_path / "run"
    _save(runtime, root, step=2, identity=run_identity(model))
    other_identity = {"base": content_hash_tensors({"other": torch.ones(3)})}
    with pytest.raises(PipelineParallelError) as excinfo:
        load_distributed_checkpoint(root, runtime, identity=other_identity)
    assert excinfo.value.code == CODE_CHECKPOINT_IDENTITY_MISMATCH


# ---------------------------------------------------------------------------
# the acceptance case: save, restart the processes, load, compare
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.probe
def test_two_rank_checkpoint_roundtrip_survives_a_process_restart(tmp_path):
    root = tmp_path / "run"
    save_dir = tmp_path / "phase1"
    load_dir = tmp_path / "phase2"
    sentinel = tmp_path / "existing-output.txt"
    sentinel.write_text("user training output")

    run_pair(save_dir, worker_save, str(root))
    saved = [json.loads((save_dir / f"save-rank{rank}.json").read_text()) for rank in range(2)]
    for report in saved:
        assert report["step"] == SAVE_AFTER_STEPS
        assert report["world_size"] == 2
        # Exactly the manifest plus one artifact per rank -- nothing else.
        assert report["files"] == [
            "pp_checkpoint.json",
            "pp_stage-rank0.pt",
            "pp_stage-rank1.pt",
        ]
        assert report["adapters"], "each stage must save its own adapters"
        assert report["digest"]["adapter_hash"]

    # A *fresh* pair of processes: this is the "restart" in the acceptance claim.
    run_pair(load_dir, worker_restore, str(root), str(save_dir))
    restored = [
        json.loads((load_dir / f"restore-rank{rank}.json").read_text())
        for rank in range(2)
    ]
    for report in restored:
        assert report["step"] == SAVE_AFTER_STEPS
        assert report["restored_step"] == SAVE_AFTER_STEPS
        assert report["tensors_equal"] is True
        assert report["optimizer_tensors_equal"] is True
        assert report["digest_equal"] is True
        comparison = report["comparison"]
        assert comparison["adapter_max_abs"] == 0.0
        assert comparison["optimizer_equal"] is True
        assert comparison["scaler_equal"] is True
        assert comparison["scale_equal"] is True
        assert comparison["step_delta"] == 0
        assert comparison["last_skip_equal"] is True
        assert comparison["topology_equal"] is True
        assert comparison["equal"] is True, comparison
        # benchmark.verify_resume shape: save -> step -> load -> step.
        assert report["replay_adapter_max_abs"] == 0.0
        assert report["replay_loss_delta"] == 0.0
        assert report["replay_scale_delta"] == 0.0
        assert report["replay_skipped"] == [False, False]
        assert report["root_files"] == [
            "pp_checkpoint.json",
            "pp_stage-rank0.pt",
            "pp_stage-rank1.pt",
        ]

    assert sentinel.read_text() == "user training output"
    assert not list(root.glob("*.tmp"))
    assert sorted(path.name for path in root.iterdir()) == [
        "pp_checkpoint.json",
        "pp_stage-rank0.pt",
        "pp_stage-rank1.pt",
    ]


@pytest.mark.integration
@pytest.mark.probe
def test_two_rank_save_refuses_an_existing_run_root_without_deadlock(tmp_path):
    # The run root already exists (a previous checkpoint, or training output):
    # *both* ranks must refuse, nothing inside may be touched, and no rank may be
    # left waiting in a collective -- the refusal is negotiated, not raised
    # rank-locally.
    root = tmp_path / "existing-run"
    root.mkdir()
    sentinel = root / "old-checkpoint.txt"
    sentinel.write_text("previous checkpoint")

    errors = run_pair_expect_failure(tmp_path / "phase1", worker_save, str(root))
    # Both ranks report the same named refusal (each traceback mentions it more
    # than once, so the count is a lower bound, not an equality).
    assert errors.count("pp_checkpoint_run_root_not_new") >= 2, errors
    assert sorted(path.name for path in (tmp_path / "phase1").glob("*.error")) == [
        "save-rank0.error",
        "save-rank1.error",
    ]
    assert sentinel.read_text() == "previous checkpoint"
    assert sorted(path.name for path in root.iterdir()) == ["old-checkpoint.txt"]
