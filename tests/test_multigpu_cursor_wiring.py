"""Production wiring of the resume cursor and the bounded-run command (t25).

The reviewers' finding was that the *helpers* were tested while the **production
call chain** was not: the save entry point never passed a real step/cursor, the
load path never compared a cursor, and the RUN command used flags the trainer does
not have. These tests therefore drive the real entry points with minimal fakes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from library.training import checkpoints as ckpt
from library.training import loop
from library.training.multigpu import cursor as cursor_mod
from library.training.multigpu import resume_protocol as rp


class _SaveAccelerator:
    """Accelerate stand-in whose ``save_state`` writes what accelerate really writes.

    It emits ``random_states_<rank>.pkl`` (the RNG evidence the bundle derives its
    flags from) and a ``shard-<rank>`` file, so ``exact`` is decided from disk
    facts instead of from an asserted boolean.
    """

    def __init__(self, *, index=0, world=1, fail_save=False, accel_device=None):
        import torch

        self.num_processes = world
        self.process_index = index
        self.is_main_process = index == 0
        self.device = accel_device or torch.device("cpu")
        self.fail_save = fail_save
        self.state = type("S", (), {"distributed_type": "MULTI_CPU" if world > 1 else "NO"})()
        self.barriers = 0

    def save_state(self, output_dir):
        """Write what ``accelerate.checkpointing.save_accelerator_state`` writes.

        The keys are accelerate's real ones (``random_state`` /
        ``numpy_random_seed`` / ``torch_manual_seed``), never the bundle's own
        vocabulary: a fake that writes whatever the reader wants is how the RNG
        evidence stayed broken on a real save (t3 F-T3-3).
        """
        import random

        import numpy
        import torch

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / f"shard-{self.process_index}.json").write_text(
            json.dumps({"rank": self.process_index}), encoding="utf-8"
        )
        (Path(output_dir) / "train_state.json").write_text(
            json.dumps({"current_epoch": 1, "current_step": 10}), encoding="utf-8"
        )
        torch.save(
            {
                "step": 10,
                "random_state": random.getstate(),
                "numpy_random_seed": numpy.random.get_state(),
                "torch_manual_seed": torch.get_rng_state(),
            },
            Path(output_dir) / f"random_states_{self.process_index}.pkl",
        )
        if self.fail_save:
            raise RuntimeError("boom")

    def wait_for_everyone(self):
        self.barriers += 1

    def reduce(self, tensor, reduction="sum"):
        return tensor * self.num_processes


def _args(tmp_path, **over):
    args = type(
        "A",
        (),
        {
            "output_dir": str(tmp_path),
            "output_name": "demo",
            "save_every_n_steps": 10,
            "save_every_n_epochs": None,
            "checkpointing_epochs": None,
            "save_state": True,
            "save_model_as": "safetensors",
            "no_metadata": True,
            "save_last_n_steps": None,
            "save_last_n_steps_state": None,
            "save_last_n_epochs": None,
            "save_last_n_epochs_state": None,
        },
    )()
    for key, value in over.items():
        setattr(args, key, value)
    return args


def _accept_all_equal(monkeypatch):
    monkeypatch.setattr(
        "library.training.multigpu.window_protocol.all_ranks_equal",
        lambda accelerator, values: {key: True for key in values},
    )


# ---- R3-F1: the production save call carries the real step and cursor ----


def test_save_checkpoint_state_records_real_step_epoch_and_rank_cursors(tmp_path, monkeypatch):
    _accept_all_equal(monkeypatch)
    args = _args(tmp_path)
    accelerator = _SaveAccelerator()

    ckpt.save_checkpoint_state(
        args,
        accelerator,
        epoch_no=2,
        global_step=10,
        loader_cursor={"sampler_position": 10, "epoch": 2, "batches_per_epoch": 30},
    )

    bundle = rp.read_bundle(str(tmp_path / "demo-checkpoint-000002-state"))
    assert bundle is not None
    # The real step/epoch reached the bundle instead of None/placeholder.
    assert bundle["global_step"] == 10
    assert bundle["extra"]["epoch"] == 2
    assert bundle["extra"]["rank_identity"]["local_rank"] == 0
    cursors = bundle["extra"]["rank_cursors"]
    assert cursors["0"]["sampler_position"] == 10
    assert cursors["0"]["epoch"] == 2
    # RNG flags come from the pickles that really landed on disk.
    assert bundle["rng"]["source"] == "loaded_random_states"
    assert bundle["rng"]["python"] is True
    # Per-rank cursor artifacts are declared as rank-local keys.
    assert bundle["rank_local_keys"] == {"0": [ckpt.RANK_CURSOR_FILE.format(rank=0)]}
    # rank_devices now describes the worker that wrote it, from its own report.
    assert bundle["rank_devices"]["0"]["type"] == "cpu"


def test_state_without_a_reported_cursor_cannot_be_exact(tmp_path, monkeypatch):
    _accept_all_equal(monkeypatch)
    args = _args(tmp_path)
    ckpt.save_checkpoint_state(args, _SaveAccelerator(), epoch_no=1, global_step=10)

    decision = rp.guard_resume(
        str(tmp_path / "demo-checkpoint-000001-state"), world_size=1
    )
    assert decision.exact is False
    assert decision.code == rp.CODE_CURSOR_UNKNOWN


# ---- R3-F2: the resume path compares the saved position ----


def test_resume_cursor_matches_only_the_position_the_plan_skips():
    saved = cursor_mod.make_cursor(position=30, epoch=1, rank=0, batches_per_epoch=30)
    expected = cursor_mod.make_cursor(
        position=cursor_mod.expected_position(
            epoch_to_start=1, initial_step=0, batches_per_epoch=30
        ),
        epoch=1,
        rank=0,
        batches_per_epoch=30,
    )
    cursor_mod.check_cursor(saved, expected=expected)

    drifted = dict(saved, sampler_position=29)
    with pytest.raises(rp.ResumeBundleError) as excinfo:
        cursor_mod.check_cursor(drifted, expected=expected)
    assert excinfo.value.code == rp.CODE_CURSOR_MISMATCH


def test_placeholder_cursor_is_unknown_not_exact():
    saved = {"sampler_position": "not_reported", "epoch": 1, "rank": 0}
    with pytest.raises(rp.ResumeBundleError) as excinfo:
        cursor_mod.check_cursor(saved, expected={"sampler_position": 0, "epoch": 1, "rank": 0})
    assert excinfo.value.code == rp.CODE_CURSOR_UNKNOWN


def test_verify_resume_cursor_raises_on_a_mismatched_state(tmp_path, monkeypatch):
    """The production consumer: saved position vs what this resume really skips."""
    from library.training import train_session

    state_dir = tmp_path / "demo-checkpoint-000001-state"
    state_dir.mkdir()
    bundle = {
        "version": rp.BUNDLE_VERSION,
        "world_size": 1,
        "backend": "none",
        "rank_devices": {"0": {"type": "cpu", "index": None}},
        "rng": {"source": "loaded_random_states", "python": True, "numpy": True, "torch": True, "cuda": None},
        "loader_cursor": cursor_mod.make_cursor(position=5, epoch=1, rank=0, batches_per_epoch=5),
        "global_step": 5,
        "extra": {
            "rank_cursors": {"0": cursor_mod.make_cursor(position=5, epoch=1, rank=0, batches_per_epoch=5)},
            "epoch": 1,
        },
    }
    rp.write_bundle(bundle, str(state_dir))

    accelerator = _SaveAccelerator()
    args = type("A", (), {"resume": str(state_dir)})()

    train_session.verify_resume_cursor(
        args, accelerator, list(range(5)), epoch_to_start=1, initial_step=0
    )  # position 5 == skipped whole epoch of 5

    with pytest.raises(rp.ResumeBundleError) as excinfo:
        train_session.verify_resume_cursor(
            args, accelerator, list(range(7)), epoch_to_start=1, initial_step=0
        )
    assert excinfo.value.code == rp.CODE_CURSOR_MISMATCH


# ---- the loop is the real producer of the cursor ----


def test_loop_cursor_payload_reports_the_consumed_position():
    from library.training.multigpu import cursor as cm

    state = type(
        "S",
        (),
        {
            "batches_consumed": 12,
            "batches_per_epoch": 30,
            "current_epoch": type("V", (), {"value": 1})(),
            "accelerator": _SaveAccelerator(index=1, world=2),
        },
    )()
    payload = loop.loader_cursor_payload(state)
    assert payload["schema"] == cm.CURSOR_SCHEMA
    assert payload["sampler_position"] == 12
    assert payload["rank"] == 1

    state.batches_per_epoch = 0
    assert loop.loader_cursor_payload(state) is None


# ---- R3-F3: the epoch/step state paths use the same protocol ----


def test_epoch_state_save_routes_through_the_publish_protocol(tmp_path, monkeypatch):
    _accept_all_equal(monkeypatch)
    args = _args(tmp_path, save_every_n_epochs=1)

    ckpt.save_and_remove_state_on_epoch_end(
        args,
        _SaveAccelerator(),
        epoch_no=1,
        global_step=10,
        loader_cursor={"sampler_position": 10, "epoch": 1, "batches_per_epoch": 30},
    )

    published = tmp_path / "demo-000001-state"
    assert published.is_dir()
    bundle = rp.read_bundle(str(published))
    assert bundle["global_step"] == 10
    assert bundle["extra"]["rank_cursors"]["0"]["sampler_position"] == 10
    # No stale tmp/backup left behind by the publish.
    assert not (tmp_path / "demo-000001-state.tmp").exists()
    assert not (tmp_path / "demo-000001-state.backup").exists()


def test_single_process_save_recovers_stale_tmp_and_backup(tmp_path, monkeypatch):
    _accept_all_equal(monkeypatch)
    args = _args(tmp_path)
    state_dir = tmp_path / "demo-checkpoint-000001-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 1}), encoding="utf-8"
    )
    # A crashed run left a backup and a half-written tmp directory behind.
    backup = tmp_path / "demo-checkpoint-000001-state.backup"
    backup.mkdir()
    (backup / "leftover.txt").write_text("old", encoding="utf-8")
    stale_tmp = tmp_path / "demo-checkpoint-000001-state.tmp"
    stale_tmp.mkdir()
    (stale_tmp / "leftover.txt").write_text("stale", encoding="utf-8")

    ckpt.save_checkpoint_state(args, _SaveAccelerator(), epoch_no=1, global_step=10)

    assert not stale_tmp.exists()
    assert not backup.exists()
    assert not (state_dir / "leftover.txt").exists()
    assert (state_dir / "shard-0.json").exists()


# ---- R3-F1: the step/epoch cadence paths must save state on *every* rank ----


def _saver(args, accelerator):
    return ckpt.CheckpointSaver(
        args=args,
        accelerator=accelerator,
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=type("V", (), {"value": 1})(),
        current_step=type("V", (), {"value": 10})(),
    )


def test_step_cadence_state_save_runs_on_a_non_main_rank(tmp_path, monkeypatch):
    """A rank-0-only state save hangs the group: the publish protocol's first
    collective is a barrier every rank must reach (contract §8 P1-1)."""
    events: list[str] = []
    monkeypatch.setattr(
        ckpt,
        "save_and_remove_state_stepwise",
        lambda *a, **k: events.append("state"),
    )
    args = _args(tmp_path, save_every_n_steps=10)
    accelerator = _SaveAccelerator(index=1, world=2)
    saver = _saver(args, accelerator)
    saver.save = lambda *a, **k: events.append("model")  # model file is rank-0 only

    saver.maybe_save_step(
        network=None,
        global_step=10,
        epoch=0,
        loader_cursor={"sampler_position": 10},
    )

    assert events == ["state"], events  # no model write here, but the state is saved


def test_epoch_cadence_state_save_runs_on_a_non_main_rank(tmp_path, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        ckpt,
        "save_and_remove_state_on_epoch_end",
        lambda *a, **k: events.append("state"),
    )
    args = _args(tmp_path, save_every_n_epochs=1)
    accelerator = _SaveAccelerator(index=1, world=2)
    saver = _saver(args, accelerator)
    saver.save = lambda *a, **k: events.append("model")

    saver.maybe_save_epoch(
        network=None,
        global_step=10,
        epoch=0,
        num_train_epochs=1,
        loader_cursor={"sampler_position": 10},
    )

    assert events == ["state"], events


def test_resumable_cadence_passes_the_real_step_and_cursor(tmp_path, monkeypatch):
    """R3-F1: the bundle must record the training step and the committed cursor."""
    _accept_all_equal(monkeypatch)
    args = _args(tmp_path, checkpointing_epochs=1)
    saver = _saver(args, _SaveAccelerator())
    saver.save = lambda *a, **k: None  # the model file is not what this pins

    saver.maybe_save_resumable(
        network=None,
        global_step=10,
        epoch=0,
        num_train_epochs=1,
        loader_cursor={"sampler_position": 10, "epoch": 1, "batches_per_epoch": 30},
    )

    bundle = rp.read_bundle(str(tmp_path / "demo-checkpoint-000001-state"))
    assert bundle is not None
    assert bundle["global_step"] == 10  # not None, not a placeholder
    assert bundle["extra"]["rank_cursors"]["0"]["sampler_position"] == 10


# ---- R3-F1 (integration): both ranks really produce the step state ----


def _gloo_stepwise_worker(rank, world_size, init_file, out_dir):
    import json as _json
    from types import SimpleNamespace

    import torch as _torch
    import torch.distributed as dist

    from library.training.checkpoints import save_and_remove_state_stepwise

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
            import random as _random

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / f"shard-{rank}.json").write_text(
                _json.dumps({"rank": rank}), encoding="utf-8"
            )
            (Path(output_dir) / "train_state.json").write_text(
                _json.dumps({"current_epoch": 1, "current_step": 10}), encoding="utf-8"
            )
            _torch.save(
                {
                    "python": _random.getstate(),
                    "numpy": _random.getstate(),
                    "torch": _torch.get_rng_state(),
                    "cuda": None,
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
        save_and_remove_state_stepwise(
            args,
            _Acc(),
            10,
            set(),
            loader_cursor=cursor_mod.make_cursor(
                position=10, epoch=1, rank=rank, batches_per_epoch=30
            ),
        )
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        Path(out_dir, f"stepwise-{rank}.json").write_text(
            json.dumps(outcome), encoding="utf-8"
        )
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.integration
def test_two_ranks_produce_one_step_state_with_both_cursors(tmp_path):
    import torch.multiprocessing as mp

    init_file = str(tmp_path / "gloo-init")
    mp.spawn(
        _gloo_stepwise_worker,
        args=(2, init_file, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    outcomes = [
        json.loads(Path(tmp_path, f"stepwise-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert all(outcome["error"] is None for outcome in outcomes), outcomes

    published = tmp_path / "demo-step00000010-state"
    assert published.is_dir()
    # Both ranks' shards, RNG evidence and cursors are inside the published state.
    assert (published / "shard-0.json").exists() and (published / "shard-1.json").exists()
    bundle = rp.read_bundle(str(published))
    assert sorted(bundle["rank_devices"]) == ["0", "1"]
    assert bundle["extra"]["rank_cursors"]["1"]["sampler_position"] == 10
    assert bundle["global_step"] == 10
    decision = rp.guard_resume(str(published), world_size=2)
    assert decision.exact, decision.message


# ---- F-T3-2 端到端：cursor 位置 + RNG 证据 + fixture slot 必须是同一条链 ----


def _real_loader(n=12):
    """The production loader shape: shuffle + zero workers, wrapped by accelerate."""
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, Dataset

    class _Ds(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return {"value": torch.tensor(int(index))}

    def _collate(batch):
        return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

    loader = DataLoader(
        _Ds(n), shuffle=False, batch_size=1, collate_fn=_collate, num_workers=0
    )
    accelerator = Accelerator(cpu=True)
    return accelerator, accelerator.prepare_data_loader(loader)


def test_resume_counter_equals_the_real_first_batch_index(tmp_path):
    """Driven by the real skip semantics of ``loop._run_epoch_steps``.

    12 samples, ``initial_step=4``: the loop calls
    ``skip_first_batches(initial_step - 1)`` and then consumes one more batch through
    ``if state.initial_step > 0: ... continue``, so the first batch that really runs
    carries local index 4 — which is what the loop's counter must start at
    (t3 F-T3-2 measured counter 3 vs real index 4).
    """
    import torch
    from library.training.multigpu import noise_fixture as fx

    accelerator, loader = _real_loader(12)
    initial_step = 4
    counter = cursor_mod.resume_skipped_batches(
        epoch_to_start=0, initial_step=initial_step, batches_per_epoch=len(loader)
    )
    assert counter == 4
    assert (
        cursor_mod.expected_position(
            epoch_to_start=0,
            initial_step=initial_step,
            batches_per_epoch=len(loader),
        )
        == counter
    )

    stream = accelerator.skip_first_batches(loader, initial_step - 1)
    iterator = iter(stream)
    next(iterator)  # the batch the loop consumes through `continue`
    first = next(iterator)
    real_index = int(first["value"].item())
    assert real_index == counter  # counter is the real local batch index

    # ... so the fixture pins the same slot the continuous run has at that position.
    assert fx.sample_ids_from_position(
        local_batch_index=counter, rank=0, world_size=1, batch_size=1
    ) == fx.sample_ids_from_position(
        local_batch_index=real_index, rank=0, world_size=1, batch_size=1
    ) == ["s00000004"]


def test_resumed_state_is_exact_at_the_real_first_batch_position(tmp_path, monkeypatch):
    """Saved cursor + real RNG evidence + the resume plan, in one assertion."""
    _accept_all_equal(monkeypatch)
    from library.training.multigpu import noise_fixture as fx

    accelerator, loader = _real_loader(12)
    counter = cursor_mod.resume_skipped_batches(
        epoch_to_start=0, initial_step=4, batches_per_epoch=len(loader)
    )
    args = _args(tmp_path)
    ckpt.save_checkpoint_state(
        args,
        _SaveAccelerator(),
        epoch_no=1,
        global_step=counter,
        loader_cursor=cursor_mod.make_cursor(
            position=counter, epoch=1, rank=0, batches_per_epoch=len(loader)
        ),
    )

    expected = cursor_mod.make_cursor(
        position=counter, epoch=1, rank=0, batches_per_epoch=len(loader)
    )
    decision = rp.guard_resume(
        str(tmp_path / "demo-checkpoint-000001-state"),
        world_size=1,
        rank=0,
        resume_cursor=expected,
    )
    assert decision.exact, decision.message
    assert fx.sample_ids_from_position(
        local_batch_index=expected["sampler_position"], rank=0, world_size=1, batch_size=1
    ) == ["s00000004"]
