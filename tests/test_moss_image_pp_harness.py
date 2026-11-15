"""Regression checks for call-local PP rendezvous and failure evidence (F4)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import moss_pp_test_support as harness


class _InlineProcess:
    """Exercise the real worker wrapper without spawning a model per test."""

    def __init__(self, *, target, args):
        self.target, self.args, self.exitcode = target, args, None

    def start(self):
        try:
            self.target(*self.args)
        except BaseException:
            self.exitcode = 1
        else:
            self.exitcode = 0

    def join(self, timeout):
        pass

    def is_alive(self):
        return False


def _succeed(rank, init_path, output_dir):
    pass


def _fail_one(rank, init_path, output_dir):
    if rank == 0:
        raise RuntimeError("fresh rank 0")


def _fail_both(rank, init_path, output_dir):
    raise RuntimeError(f"fresh rank {rank}")


def _record_rendezvous(rank, init_path, output_dir):
    if rank == 0:
        # Leave the store file behind: the next call must not reuse it even if
        # a process group did not clean it up.
        assert not Path(init_path).exists()
        Path(init_path).write_text("used rendezvous")
    Path(output_dir, f"rank{rank}.init_path").write_text(init_path)


@pytest.fixture
def inline_context(monkeypatch):
    monkeypatch.setattr(
        harness, "get_context", lambda _: SimpleNamespace(Process=_InlineProcess)
    )


def test_real_spawn_preserves_outputs_and_never_reuses_rendezvous(tmp_path):
    harness.run_pair(tmp_path, _record_rendezvous)
    first = Path(tmp_path, "rank0.init_path").read_text()
    harness.run_pair(tmp_path, _record_rendezvous)
    second = Path(tmp_path, "rank0.init_path").read_text()
    assert first != second
    assert Path(first).exists() and Path(second).exists()
    assert Path(tmp_path, "rank1.init_path").read_text() == second


@pytest.mark.parametrize("worker", [_succeed, _fail_one])
def test_stale_errors_cannot_turn_success_or_partial_failure_into_pass(
    tmp_path, inline_context, worker
):
    Path(tmp_path, "rank0.error").write_text("stale previous failure")
    with pytest.raises(AssertionError, match="expected both ranks to fail"):
        harness.run_pair_expect_failure(tmp_path, worker)


def test_failure_receipts_are_fresh_and_rank_complete(tmp_path, inline_context):
    Path(tmp_path, "rank0.error").write_text("stale previous failure")
    errors = harness.run_pair_expect_failure(tmp_path, _fail_both)
    assert len(errors) == 2
    assert all(f"fresh rank {rank}" in errors[rank] for rank in range(2))
    assert all("stale previous failure" not in error for error in errors)
    # A successful subsequent call is not affected by those failure receipts.
    harness.run_pair(tmp_path, _succeed)


def test_timeout_cleanup_is_not_an_expected_worker_failure(tmp_path, monkeypatch):
    class TimedOutProcess(_InlineProcess):
        def start(self):
            self.exitcode = None

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.exitcode = -15

    monkeypatch.setattr(
        harness, "get_context", lambda _: SimpleNamespace(Process=TimedOutProcess)
    )
    Path(tmp_path, "rank0.error").write_text("stale previous failure")
    with pytest.raises(AssertionError, match="exit codes \\[None, None\\]"):
        harness.run_pair_expect_failure(tmp_path, _succeed)


def test_join_uses_one_shared_deadline(tmp_path, monkeypatch):
    clock = [0.0]
    budgets = []

    class ClockedProcess(_InlineProcess):
        def join(self, timeout):
            budgets.append(timeout)
            clock[0] += 7.0

    monkeypatch.setattr(harness, "monotonic", lambda: clock[0])
    monkeypatch.setattr(harness, "get_context", lambda _: SimpleNamespace(Process=ClockedProcess))
    harness.run_pair(tmp_path, _succeed, deadline=10.0)
    assert budgets == [10.0, 3.0]


def test_expired_deadline_does_not_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "monotonic", lambda: 10.0)

    def forbidden_context(*args):
        raise AssertionError("expired calls must not spawn")

    monkeypatch.setattr(harness, "get_context", forbidden_context)
    with pytest.raises(TimeoutError, match="before spawning"):
        harness.run_pair(tmp_path, _succeed, deadline=9.0)


@pytest.mark.parametrize("worker", [_succeed, _fail_both])
def test_worker_finished_marker_is_not_process_exit(tmp_path, monkeypatch, worker):
    class PostWorkerAlive(_InlineProcess):
        def start(self):
            super().start()
            self.exitcode = None  # body finished, process shutdown has not

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            self.exitcode = -15

    monkeypatch.setattr(harness, "get_context", lambda _: SimpleNamespace(Process=PostWorkerAlive))
    with pytest.raises(AssertionError, match=r"exit codes:? \[None, None\]") as excinfo:
        if worker is _succeed:
            harness.run_pair(tmp_path, worker)
        else:
            harness.run_pair_expect_failure(tmp_path, worker)
    assert "worker_returned" in str(excinfo.value) or "worker_raised" in str(excinfo.value)
    assert all(
        "worker_" in path.read_text()
        for path in tmp_path.glob("pp-pair-*/rank*.lifecycle")
    )
