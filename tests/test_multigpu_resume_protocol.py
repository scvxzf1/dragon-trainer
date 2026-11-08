"""Versioned resume metadata and pre-resume validation (t5, contract §8 P1-2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from library.training.multigpu import resume_protocol as rp


class _Acc:
    def __init__(self, *, index=0, world=2):
        self.num_processes = world
        self.process_index = index
        self.device = SimpleNamespace(type="cpu", index=None)


def _write_rng_state(state_dir, rank: int) -> None:
    import random as _random

    import numpy as _np
    import torch as _torch

    state_dir.mkdir(parents=True, exist_ok=True)
    _torch.save(
        {
            "step": 1,
            "random_state": _random.getstate(),
            "numpy_random_seed": _np.random.get_state(),
            "torch_manual_seed": _torch.get_rng_state(),
        },
        state_dir / f"random_states_{rank}.pkl",
    )


def _rank_bundle(tmp_path=None, *, ranks=2, sampler_position=4, index=0, **kwargs):
    """One worker's own bundle: it only knows *its* device."""
    state_dir = (tmp_path or Path(".")) / "evidence"
    for rank in range(ranks):
        _write_rng_state(state_dir, rank)
    base = dict(
        backend="gloo",
        loader_cursor={"epoch": 1, "sampler_position": sampler_position},
        evidence_dir=str(state_dir),
        global_step=10,
    )
    base.update(kwargs)
    return rp.build_bundle(_Acc(index=index, world=ranks), **base)


def _bundle(tmp_path=None, *, ranks=2, sampler_position=4, **kwargs):
    """The bundle production really publishes: every rank's report merged.

    ``publish_state_with_bundle`` builds ``rank_devices`` from the per-rank cursor
    artifacts of *all* workers before rank 0 writes the file, so a world_size=2
    bundle always describes both ranks. A single ``build_bundle`` call only knows
    the calling worker's device, which is exactly the inconsistency the validator
    must refuse (see the rank-device tests below).
    """
    per_rank = [
        _rank_bundle(tmp_path, ranks=ranks, sampler_position=sampler_position, index=rank, **kwargs)
        for rank in range(ranks)
    ]
    return rp.merge_rank_bundles(per_rank)


def test_bundle_records_topology_rng_and_loader_cursor(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle["version"] == rp.BUNDLE_VERSION
    assert bundle["world_size"] == 2
    assert bundle["backend"] == "gloo"
    assert bundle["loader_cursor"] == {"epoch": 1, "sampler_position": 4}
    assert bundle["rng"]["source"] == "loaded_random_states"
    assert bundle["rng"]["python"] and bundle["rng"]["numpy"] and bundle["rng"]["torch"]
    assert bundle["rng"]["cuda"] is None  # cpu topology: not applicable, not faked
    assert bundle["rank_devices"]["0"]["type"] == "cpu"
    assert bundle["global_step"] == 10


def test_merge_rank_bundles_builds_the_full_rank_mapping(tmp_path):
    left = _rank_bundle(tmp_path, index=0)
    right = _rank_bundle(tmp_path, index=1)
    assert sorted(left["rank_devices"]) == ["0"]  # one worker knows only itself
    merged = rp.merge_rank_bundles([left, right])
    assert sorted(merged["rank_devices"], key=int) == ["0", "1"]
    assert merged["world_size"] == 2


def test_a_bundle_that_describes_only_some_ranks_is_refused_not_degraded(tmp_path):
    """Missing rank_devices is an inconsistent record, not a topology change.

    ``rank_devices`` is invalid *and* it is not "degraded": degrading is reserved
    for a known topology change (different world_size/backend). The publish path
    always writes one entry per rank, so a partial mapping cannot have been
    produced by a save of this size.
    """
    bundle = _rank_bundle(tmp_path)
    decision = rp.validate_bundle(bundle, world_size=2)
    assert not decision.exact
    assert decision.code == rp.CODE_BUNDLE_INVALID
    assert "missing rank(s) ['1']" in decision.message
    with pytest.raises(rp.ResumeBundleError):
        decision.raise_if_rejected()


def test_a_bundle_describing_more_ranks_than_it_declares_is_refused(tmp_path):
    bundle = _bundle(tmp_path)  # describes ranks 0 and 1
    bundle["world_size"] = 1  # ... but claims a single-worker run
    decision = rp.validate_bundle(bundle, world_size=1)
    assert not decision.exact
    assert decision.code == rp.CODE_BUNDLE_INVALID
    assert "unexpected rank(s)" in decision.message


def test_a_bundle_without_any_rank_devices_is_refused(tmp_path):
    bundle = _bundle(tmp_path)
    bundle.pop("rank_devices")
    decision = rp.validate_bundle(bundle, world_size=2)
    assert decision.code == rp.CODE_BUNDLE_INVALID


def test_write_and_read_roundtrip_is_atomic(tmp_path):
    bundle = _bundle(tmp_path)
    written = rp.write_bundle(bundle, str(tmp_path))
    assert written and Path(written).exists()
    assert not (tmp_path / (rp.BUNDLE_FILENAME + ".tmp")).exists()
    assert rp.read_bundle(str(tmp_path))["world_size"] == 2


def test_non_main_rank_does_not_write(tmp_path):
    assert rp.write_bundle(_bundle(tmp_path), str(tmp_path), is_main=False) is None
    assert not (tmp_path / rp.BUNDLE_FILENAME).exists()


def test_same_world_size_with_streams_is_an_exact_resume(tmp_path):
    decision = rp.validate_bundle(_bundle(tmp_path), world_size=2, backend="gloo")
    assert decision.exact and decision.code is None


def test_changed_world_size_is_rejected_not_guessed(tmp_path):
    decision = rp.validate_bundle(_bundle(tmp_path), world_size=4)
    assert not decision.exact
    assert decision.code == rp.CODE_TOPOLOGY_MISMATCH
    with pytest.raises(rp.ResumeBundleError):
        decision.raise_if_rejected()


def test_changed_world_size_can_only_degrade_with_an_explicit_opt_in(tmp_path):
    decision = rp.validate_bundle(_bundle(tmp_path), world_size=4, allow_degraded=True)
    assert not decision.exact
    assert decision.code == rp.CODE_TOPOLOGY_DEGRADED


def test_legacy_state_without_bundle_is_never_called_exact(tmp_path):
    decision = rp.guard_resume(str(tmp_path), world_size=2)
    assert not decision.exact
    assert decision.code == rp.CODE_LEGACY_NOT_EXACT
    decision.raise_if_rejected()  # legacy is allowed, just not exact


def test_version_mismatch_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["version"] = rp.BUNDLE_VERSION + 1
    rp.write_bundle(bundle, str(tmp_path))
    decision = rp.guard_resume(str(tmp_path), world_size=2)
    assert decision.code == rp.CODE_BUNDLE_INVALID
    with pytest.raises(rp.ResumeBundleError):
        decision.raise_if_rejected()


def test_unreadable_bundle_is_a_named_error(tmp_path):
    (tmp_path / rp.BUNDLE_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(rp.ResumeBundleError) as excinfo:
        rp.guard_resume(str(tmp_path), world_size=2)
    assert rp.CODE_BUNDLE_INVALID in str(excinfo.value)


def test_missing_rng_streams_downgrade_to_named_degradation(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["rng"] = dict(bundle["rng"], torch=False)
    decision = rp.validate_bundle(bundle, world_size=2)
    assert not decision.exact
    assert decision.code == rp.CODE_TOPOLOGY_DEGRADED
    assert "torch" in decision.message


def test_backend_change_is_not_an_exact_resume(tmp_path):
    decision = rp.validate_bundle(_bundle(tmp_path), world_size=2, backend="nccl")
    assert not decision.exact
    assert decision.code == rp.CODE_TOPOLOGY_MISMATCH


# ---------------------------------------------------------------------------
# 真实双进程 Gloo 回归（t19 F2）：逐 rank RNG / loader 游标 / topology 映射
# ---------------------------------------------------------------------------


def _gloo_resume_worker(rank, world_size, init_file, out_dir):
    import json as _json

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
        device = SimpleNamespace(type="cpu", index=None)

    try:
        import random as _random

        import numpy as _np
        import torch as _torch

        target = Path(out_dir, "state")
        target.mkdir(parents=True, exist_ok=True)
        _torch.save(
            {
                "step": 1,
            "random_state": _random.getstate(),
            "numpy_random_seed": _np.random.get_state(),
            "torch_manual_seed": _torch.get_rng_state(),
            },
            target / f"random_states_{rank}.pkl",
        )
        dist.barrier()
        local = rp.build_bundle(
            _Acc(),
            backend="gloo",
            loader_cursor={"epoch": 1, "sampler_position": 4 + rank},  # rank-local
            evidence_dir=str(target),
            global_step=10,
        )
        gathered: list[dict] = [None] * world_size  # type: ignore[list-item]
        dist.all_gather_object(gathered, local)
        merged = rp.merge_rank_bundles(gathered)
        if rank == 0:
            rp.write_bundle(merged, str(target), is_main=True)
        dist.barrier()
        decision = rp.guard_resume(
            str(target),
            world_size=world_size,
            backend="gloo",
            resume_cursor=merged["loader_cursor"],
        )
        mismatch = rp.validate_bundle(rp.read_bundle(str(target)), world_size=world_size + 2)
        record = {
            "rank": rank,
            "exact": decision.exact,
            "code": decision.code,
            "rank_devices": sorted(merged["rank_devices"], key=int),
            "world_size": merged["world_size"],
            "mismatch_code": mismatch.code,
            "mismatch_exact": mismatch.exact,
        }
        Path(out_dir, f"resume-{rank}.json").write_text(_json.dumps(record), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.integration
def test_two_rank_resume_bundle_covers_both_ranks_and_rejects_other_topology(tmp_path):
    import json as _json

    import torch.multiprocessing as mp

    init_file = str(tmp_path / "gloo-init")
    mp.spawn(_gloo_resume_worker, args=(2, init_file, str(tmp_path)), nprocs=2, join=True)
    records = [
        _json.loads(Path(tmp_path, f"resume-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    for record in records:
        assert record["exact"] is True and record["code"] is None
        assert record["world_size"] == 2
        assert record["rank_devices"] == ["0", "1"]  # topology mapping is world-wide
        # Same bundle, different world size -> refused, never "probably fine".
        assert record["mismatch_exact"] is False
        assert record["mismatch_code"] == rp.CODE_TOPOLOGY_MISMATCH


# ---------------------------------------------------------------------------
# t23：rank 独占键、落盘 RNG 证据、cursor 比对
# ---------------------------------------------------------------------------


def test_real_accelerator_save_produces_readable_rng_evidence(tmp_path):
    """F-T3-3: the reader must parse what ``accelerator.save_state`` really writes.

    A real ``Accelerator`` (no distributed group, CPU) writes
    ``random_states_0.pkl`` with accelerate's own keys, so this is the only honest
    fixture: a hand-written pickle in the *bundle's* vocabulary would test the
    reader against itself and keep the bug green.
    """
    from accelerate import Accelerator

    from library.training.multigpu import rng_evidence

    Accelerator(cpu=True).save_state(str(tmp_path))

    evidence = rng_evidence.inspect_state_dir(
        str(tmp_path), world_size=1, uses_cuda=False
    )
    assert evidence["schema"] == "accelerate"
    assert evidence["python"] and evidence["numpy"] and evidence["torch"]
    assert evidence["cuda"] is None  # CPU topology: not applicable, never faked
    assert evidence["ranks"]["0"]["schema"] == "accelerate"

    bundle = rp.build_bundle(
        _Acc(index=0, world=1),
        backend="gloo",
        evidence_dir=str(tmp_path),
        loader_cursor={"epoch": 1, "sampler_position": 4},
    )
    decision = rp.validate_bundle(bundle, world_size=1)
    assert decision.exact, decision.message  # exact is reachable on a real save


def test_cursor_drift_is_not_swallowed_by_rng_degradation(tmp_path):
    """The cursor check must run before the generic RNG verdict (t3 F-T3-3).

    Otherwise a resume that would replay the wrong samples is reported as a mere
    ``..._topology_degraded`` and the drift disappears.
    """
    from library.training.multigpu import cursor as cursor_mod

    bundle = _with_rank_cursors(_bundle(tmp_path), {0: 4, 1: 5})
    bundle["rng"] = dict(bundle["rng"], torch=False)  # degraded RNG
    decision = rp.validate_bundle(
        bundle,
        world_size=2,
        rank=1,
        resume_cursor=cursor_mod.make_cursor(
            position=9, epoch=1, rank=1, batches_per_epoch=30
        ),
    )
    assert decision.code == rp.CODE_CURSOR_MISMATCH
    assert not decision.exact
    assert "torch" in decision.message  # the RNG problem is still reported


def test_rank_local_keys_must_be_exclusive():
    mapping = rp.rank_local_keys(
        {
            "0": ["random_states_0.pkl", "loader_cursor_0.json"],
            "1": ["random_states_1.pkl", "loader_cursor_1.json"],
        }
    )
    assert mapping["0"] == ["loader_cursor_0.json", "random_states_0.pkl"]


def test_shared_rank_key_is_refused():
    with pytest.raises(rp.ResumeBundleError) as excinfo:
        rp.rank_local_keys({"0": ["random_states.pkl"], "1": ["random_states.pkl"]})
    assert excinfo.value.code == rp.CODE_SHARED_RANK_KEY


def test_rng_flags_come_from_files_that_really_exist(tmp_path):
    empty = rp.rng_evidence_from_disk(str(tmp_path), world_size=1, uses_cuda=False)
    assert empty["python"] is False and empty["torch"] is False
    (tmp_path / "random_states_0.pkl").write_bytes(b"x" * 8)
    found = rp.rng_evidence_from_disk(str(tmp_path), world_size=1, uses_cuda=False)
    assert found["python"] is True and found["torch"] is True
    assert found["cuda"] is False  # cpu topology: no cuda stream was ever saved
    assert found["files"] == ["random_states_0.pkl"]


def test_rng_evidence_requires_every_rank(tmp_path):
    (tmp_path / "random_states_0.pkl").write_bytes(b"x" * 8)
    partial = rp.rng_evidence_from_disk(str(tmp_path), world_size=2)
    assert partial["python"] is False and partial["ranks_covered"] == 1


def test_bundle_uses_disk_evidence_instead_of_asserted_flags(tmp_path):
    state = tmp_path / "state"
    _write_rng_state(state, 0)
    _write_rng_state(state, 1)
    bundle = rp.build_bundle(
        _Acc(),
        backend="gloo",
        evidence_dir=str(state),
        loader_cursor={"sampler_position": 1234},
        rank_local={"0": ["random_states_0.pkl"], "1": ["random_states_1.pkl"]},
    )
    assert bundle["rng"]["source"] == "loaded_random_states"
    assert bundle["rng"]["ranks_covered"] == 2
    assert bundle["rank_local_keys"]["1"] == ["random_states_1.pkl"]
    assert bundle["loader_cursor"] == {"sampler_position": 1234}


def test_loader_cursor_mismatch_is_a_named_non_exact_resume(tmp_path):
    bundle = _bundle(tmp_path, sampler_position=1234)
    same = rp.validate_bundle(
        bundle, world_size=2, resume_cursor={"epoch": 1, "sampler_position": 1234}
    )
    assert same.exact
    drifted = rp.validate_bundle(
        bundle, world_size=2, resume_cursor={"epoch": 1, "sampler_position": 999}
    )
    assert not drifted.exact
    assert drifted.code == rp.CODE_CURSOR_MISMATCH
    drifted.raise_if_rejected()  # non-exact, but not a hard rejection


def _with_rank_cursors(bundle, positions):
    from library.training.multigpu import cursor as cursor_mod

    bundle = dict(bundle)
    bundle["extra"] = {
        "rank_cursors": {
            str(rank): cursor_mod.make_cursor(
                position=position, epoch=1, rank=rank, batches_per_epoch=30
            )
            for rank, position in positions.items()
        }
    }
    return bundle


def test_rank_cursor_comparison_uses_that_ranks_own_position(tmp_path):
    """``guard_resume(resume_cursor=..., rank=r)`` is the production comparison."""
    from library.training.multigpu import cursor as cursor_mod

    bundle = _with_rank_cursors(_bundle(tmp_path), {0: 4, 1: 5})

    expected_rank1 = cursor_mod.make_cursor(
        position=5, epoch=1, rank=1, batches_per_epoch=30
    )
    exact = rp.validate_bundle(
        bundle, world_size=2, rank=1, resume_cursor=expected_rank1
    )
    assert exact.exact and exact.code is None

    drifted = rp.validate_bundle(
        bundle,
        world_size=2,
        rank=1,
        resume_cursor=cursor_mod.make_cursor(
            position=4, epoch=1, rank=1, batches_per_epoch=30
        ),
    )
    assert not drifted.exact
    assert drifted.code == rp.CODE_CURSOR_MISMATCH
    assert "per-rank" in drifted.message


def test_rank_cursor_comparison_ignores_the_device_report(tmp_path):
    """The cursor artifact also carries the device; that is not position drift."""
    from library.training.multigpu import cursor as cursor_mod

    bundle = _with_rank_cursors(_bundle(tmp_path), {0: 4, 1: 5})
    bundle["extra"]["rank_cursors"]["1"].update(
        {"device_type": "cpu", "device_index": None}
    )
    decision = rp.validate_bundle(
        bundle,
        world_size=2,
        rank=1,
        resume_cursor=cursor_mod.make_cursor(
            position=5, epoch=1, rank=1, batches_per_epoch=30
        ),
    )
    assert decision.exact, decision.message


def test_rank_without_a_real_cursor_is_unknown_not_exact(tmp_path):
    from library.training.multigpu import cursor as cursor_mod

    bundle = _bundle(tmp_path)
    bundle["extra"] = {
        "rank_cursors": {
            "0": cursor_mod.make_cursor(position=4, epoch=1, rank=0, batches_per_epoch=30),
            "1": {"schema": "multigpu_loader_cursor_v1", "sampler_position": "not_reported", "epoch": 1, "rank": 1},
        }
    }
    decision = rp.validate_bundle(
        bundle,
        world_size=2,
        rank=1,
        resume_cursor=cursor_mod.make_cursor(
            position=5, epoch=1, rank=1, batches_per_epoch=30
        ),
    )
    assert not decision.exact
    assert decision.code == rp.CODE_CURSOR_UNKNOWN


def test_python_numpy_torch_rng_roundtrip_reproduces_the_next_draw(tmp_path):
    """恢复后随机序列必须与保存时“下一步本应产生”的序列逐位一致（CPU 实测）。"""
    import random as _random

    import numpy as _np
    import torch as _torch

    from library.training.multigpu import rng_evidence

    state = tmp_path / "rt"
    _random.seed(7)
    _np.random.seed(7)
    _torch.manual_seed(7)
    _write_rng_state(state, 0)
    expected = (
        _random.random(),
        float(_np.random.rand()),
        float(_torch.rand(1)),
    )
    # Drift every stream, then restore from the evidence file and compare.
    _random.random()
    _np.random.rand()
    _torch.rand(3)
    payload = _torch.load(state / "random_states_0.pkl", weights_only=False)
    # accelerate's real keys: a restore through them is what makes the evidence
    # meaningful (the bundle vocabulary never appears in the pickle).
    _random.setstate(payload["random_state"])
    _np.random.set_state(payload["numpy_random_seed"])
    _torch.set_rng_state(payload["torch_manual_seed"])
    restored = (
        _random.random(),
        float(_np.random.rand()),
        float(_torch.rand(1)),
    )
    assert restored == expected
    assert rng_evidence.state_files(str(state)) == {0: str(state / "random_states_0.pkl")}


def test_unknown_cursor_is_never_an_exact_resume(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["loader_cursor"] = {"epoch": 1, "sampler_position": "not_reported"}
    decision = rp.validate_bundle(bundle, world_size=2)
    assert not decision.exact
    assert decision.code == rp.CODE_CURSOR_UNKNOWN


def test_unreadable_rng_evidence_is_named_and_not_exact(tmp_path):
    state = tmp_path / "broken"
    state.mkdir()
    (state / "random_states_0.pkl").write_bytes(b"not a pickle")
    (state / "random_states_1.pkl").write_bytes(b"not a pickle")
    bundle = rp.build_bundle(
        _Acc(),
        backend="gloo",
        evidence_dir=str(state),
        loader_cursor={"sampler_position": 4},
    )
    # Production publishes a complete device mapping (one entry per rank, taken
    # from each rank's own report), so pin it here: this test is about the RNG
    # evidence, not about an incomplete topology record.
    bundle["rank_devices"] = {
        "0": {"type": "cpu", "index": None},
        "1": {"type": "cpu", "index": None},
    }
    assert bundle["rng"]["source"] == "unreadable"
    decision = rp.validate_bundle(bundle, world_size=2)
    assert not decision.exact
    assert decision.code == rp.CODE_TOPOLOGY_DEGRADED


# ---------------------------------------------------------------------------
# F-T10-1：恢复时刻必须对真实目录**再核验**一次 RNG 流
# ---------------------------------------------------------------------------


def _real_two_rank_state(tmp_path, *, name="state"):
    """A state directory produced by a real ``Accelerator.save_state``.

    Two ranks are simulated by copying rank 0's pickle: accelerate writes exactly
    one ``random_states_<rank>.pkl`` per process, and the schema is what matters
    here, not the bytes.
    """
    import shutil

    from accelerate import Accelerator

    state = tmp_path / name
    state.mkdir()
    Accelerator(cpu=True).save_state(str(state))
    shutil.copyfile(
        state / "random_states_0.pkl", state / "random_states_1.pkl"
    )
    bundle = rp.build_bundle(
        _Acc(index=0, world=2),
        backend="gloo",
        evidence_dir=str(state),
        loader_cursor={"epoch": 1, "sampler_position": 4},
    )
    bundle["rank_devices"] = {
        "0": {"type": "cpu", "index": None},
        "1": {"type": "cpu", "index": None},
    }
    rp.write_bundle(bundle, str(state))
    return state


def test_a_healthy_real_state_is_exact_and_its_rng_is_rechecked(tmp_path):
    state = _real_two_rank_state(tmp_path)
    decision = rp.guard_resume(str(state), world_size=2)
    assert decision.exact, decision.message
    assert decision.codes == ()


def test_rng_files_that_vanished_are_refused_at_resume_time(tmp_path):
    """F-T10-1: the bundle's save-time flags cannot see a deleted pickle.

    accelerate swallows a failed RNG restore (``checkpointing.py``), so without this
    second observation a half-broken state would resume on an unrelated stream while
    the bundle still claims ``exact``.
    """
    state = _real_two_rank_state(tmp_path)
    assert rp.guard_resume(str(state), world_size=2).exact

    (state / "random_states_1.pkl").unlink()

    # The recorded flags still look fine - which is exactly why the recheck exists.
    assert rp.validate_bundle(rp.read_bundle(str(state)), world_size=2).exact

    broken = rp.guard_resume(str(state), world_size=2)
    assert not broken.exact
    assert broken.code == rp.CODE_RNG_EVIDENCE_MISSING_AT_RESUME
    # The message names the rank that is gone, not just "something broke".
    assert "rank(s) [1]" in broken.message


def test_a_corrupt_rng_pickle_is_refused_at_resume_time(tmp_path):
    state = _real_two_rank_state(tmp_path)
    (state / "random_states_1.pkl").write_bytes(b"not a pickle")
    decision = rp.guard_resume(str(state), world_size=2)
    assert not decision.exact
    assert decision.code == rp.CODE_RNG_EVIDENCE_MISSING_AT_RESUME


def test_a_pickle_missing_a_stream_is_degraded_at_resume_time(tmp_path):
    import random as _random

    import torch as _torch

    state = _real_two_rank_state(tmp_path)
    # A readable pickle that no longer carries the torch stream.
    _torch.save(
        {"step": 1, "random_state": _random.getstate()},
        state / "random_states_1.pkl",
    )
    decision = rp.guard_resume(str(state), world_size=2)
    assert not decision.exact
    assert decision.code == rp.CODE_RNG_STREAMS_DEGRADED_AT_RESUME
    assert "torch" in decision.message


def test_rng_and_cursor_problems_are_reported_together(tmp_path):
    """Independence: neither problem may hide the other (t3 F-T3-3 extended)."""
    from library.training.multigpu import cursor as cursor_mod

    state = _real_two_rank_state(tmp_path)
    bundle = rp.read_bundle(str(state))
    bundle["extra"] = {
        "rank_cursors": {
            "0": cursor_mod.make_cursor(position=4, epoch=1, rank=0, batches_per_epoch=30),
            "1": cursor_mod.make_cursor(position=5, epoch=1, rank=1, batches_per_epoch=30),
        }
    }
    rp.write_bundle(bundle, str(state))
    (state / "random_states_1.pkl").unlink()

    decision = rp.guard_resume(
        str(state),
        world_size=2,
        rank=1,
        resume_cursor=cursor_mod.make_cursor(
            position=9, epoch=1, rank=1, batches_per_epoch=30
        ),
    )
    assert not decision.exact
    assert rp.CODE_CURSOR_MISMATCH in decision.codes
    assert rp.CODE_RNG_EVIDENCE_MISSING_AT_RESUME in decision.codes
    assert decision.code == rp.CODE_CURSOR_MISMATCH  # cursor stays the primary
    assert "data order" in decision.message and "RNG" in decision.message


def test_legacy_pickle_vocabulary_is_still_readable_and_marked(tmp_path):
    """One schema, one reader: legacy keys stay readable, and say so."""
    import random as _random

    import numpy as _np
    import torch as _torch

    from library.training.multigpu import rng_evidence

    state = tmp_path / "legacy"
    state.mkdir()
    _torch.save(
        {
            "python": _random.getstate(),
            "numpy": _np.random.get_state(),
            "torch": _torch.get_rng_state(),
        },
        state / "random_states_0.pkl",
    )
    evidence = rng_evidence.observe_state_dir(
        str(state), world_size=1, uses_cuda=False
    )
    assert evidence["schema"] == "legacy"
    assert evidence["python"] and evidence["numpy"] and evidence["torch"]
    assert evidence["source"] == "loaded_random_states"


# ---------------------------------------------------------------------------
# F-T18：exact 的措辞必须说清"是否真的在恢复时刻重读过"
# ---------------------------------------------------------------------------


def test_exact_says_whether_the_rng_was_reread(tmp_path):
    from library.training.multigpu import cursor as cursor_mod

    state = _real_two_rank_state(tmp_path)
    reread = rp.guard_resume(str(state), world_size=2)
    assert reread.exact
    assert reread.rng_recheck == rp.RNG_RECHECK_DONE
    assert "re-read at resume time" in reread.message

    # ``verify_rng_files=False`` is a read-only inspection: the decision then rests
    # on the save-time evidence and must say so instead of borrowing the stronger
    # wording.
    save_time_only = rp.guard_resume(
        str(state), world_size=2, verify_rng_files=False
    )
    assert save_time_only.exact
    assert save_time_only.rng_recheck == rp.RNG_RECHECK_SKIPPED
    assert "no resume-time re-read requested" in save_time_only.message

    # A direct ``validate_bundle`` call has no state dir to re-read either.
    decision = rp.validate_bundle(rp.read_bundle(str(state)), world_size=2)
    assert decision.rng_recheck == rp.RNG_RECHECK_SKIPPED
    assert "no resume-time re-read requested" in decision.message


def test_a_rejected_resume_keeps_the_reread_flag(tmp_path):
    state = _real_two_rank_state(tmp_path)
    (state / "random_states_1.pkl").unlink()
    decision = rp.guard_resume(str(state), world_size=2)
    assert not decision.exact
    assert decision.rng_recheck == rp.RNG_RECHECK_DONE
    assert decision.code == rp.CODE_RNG_EVIDENCE_MISSING_AT_RESUME


# ---------------------------------------------------------------------------
# F-T18：恢复判决必须落盘进 multigpu_identity.json（可机检）
# ---------------------------------------------------------------------------


def test_the_resume_verdict_is_written_into_the_identity_manifest(tmp_path):
    import json

    from library.training.multigpu import identity

    state = _real_two_rank_state(tmp_path)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "multigpu_identity.json").write_text(
        json.dumps({"schema": "multigpu-identity", "accuracy_source": "fp16"}),
        encoding="utf-8",
    )

    decision = rp.guard_resume(str(state), world_size=2)
    target = identity.record_resume_outcome(out_dir, decision=decision)

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["resume_exact"] is True
    assert document["resume_codes"] == []
    assert document["resume_bundle_version"] == decision.bundle["version"]
    assert document["resume"]["rng_recheck"] == rp.RNG_RECHECK_DONE
    # The manifest keeps what the run-identity writer put there.
    assert document["schema"] == "multigpu-identity"
    assert document["accuracy_source"] == "fp16"

    # A non-main rank writes nothing (rank 0 owns the manifest).
    (state / "random_states_1.pkl").unlink()
    broken = rp.guard_resume(str(state), world_size=2)
    assert (
        identity.record_resume_outcome(
            out_dir, decision=broken, accelerator=_Acc(index=1, world=2)
        )
        is None
    )
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["resume_exact"] is True  # untouched by rank 1

    identity.record_resume_outcome(
        out_dir, decision=broken, accelerator=_Acc(index=0, world=2)
    )
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["resume_exact"] is False
    assert rp.CODE_RNG_EVIDENCE_MISSING_AT_RESUME in document["resume_codes"]
