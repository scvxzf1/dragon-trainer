"""Opt-in real-loop progress boundaries for the isolated training worker."""

import logging
from pathlib import Path
import shutil

from library.training.auto_block_swap.process import write_result
from .training_state import save_training_state


class NonfiniteSkip(RuntimeError):
    """The trial produced more non-finite optimizer skips than its budget allows.

    Reported as its own ``nonfinite_skip`` result so the supervisor can tell an
    fp16 overflow pattern (retryable, configuration-related) from a real error or
    a CUDA OOM.
    """


class TrainingRuntime:
    def __init__(self, directory, *, precision_contract_id=None, args=None):
        self.directory = Path(directory)
        self.precision_contract_id = precision_contract_id
        self.args = args
        self.precision_manifest_id = None
        self.stage = "setup"
        self.optimizer_started = False
        self.completed_steps = 0
        # A fraction of fp16 overflow is normal scaler warm-up, so a trial may
        # tolerate a small consecutive skip budget; 0 keeps the strict contract.
        self.max_nonfinite_skips = max(
            0, int(getattr(args, "adaptive_max_nonfinite_skips", 0) or 0)
        ) if args is not None else 0
        self.attempted_steps = 0
        self.effective_steps = 0
        self.skipped_steps = 0
        self.consecutive_skips = 0
        self._pending_skipped = None
        self.saved_state = None
        self.finished = False
        self.capacity = getattr(args, "_adaptive_swap_capacity", None) if args is not None else None
        self.host_limit = None
        self.host_budget_reason = "not_measured"

    def validate_swap_capacity(self, model, family):
        if self.capacity is None:
            return
        from library.training.auto_block_swap.capacity import (
            HostLimitReached, loaded_host_capacity,
        )

        if family != self.capacity["model_family"]:
            raise ValueError("Adaptive swap model family drifted after preload")
        measured = loaded_host_capacity(self.args, model, self.capacity)
        self.host_limit = measured["host_limit"]
        self.host_budget_reason = measured["host_budget_reason"]
        if self.host_limit is not None and self.args.blocks_to_swap > self.host_limit:
            raise HostLimitReached(
                f"Requested {self.args.blocks_to_swap} swapped blocks exceeds measured "
                f"host limit {self.host_limit}"
            )

    def phase(self, stage):
        self.stage = stage
        if stage == "optimizer":
            # Sticky before step(): an exception can follow a partial update.
            self.optimizer_started = True

    def validate_precision(self, model=None):
        if self.precision_contract_id is not None:
            from .training_precision import validate_local_precision_contract

            validate_local_precision_contract(self.args, self.precision_contract_id)
        if model is None:
            return
        names = getattr(model, "_adaptive_precision_manifest_names", None)
        expected = getattr(model, "_adaptive_precision_manifest_id", None)
        if names is None or expected is None:
            raise ValueError("precision_contract_mismatch: installed dtype manifest is missing")
        from .training_precision import realized_precision_manifest_id

        actual = realized_precision_manifest_id(model, names)
        if actual != expected:
            raise ValueError("precision_contract_mismatch: installed dtype manifest drifted")
        if self.precision_manifest_id is None:
            self.precision_manifest_id = actual
        elif self.precision_manifest_id != actual:
            raise ValueError("precision_contract_mismatch: realized dtype manifest drifted")

    def after_optimizer(self, state):
        if not self.optimizer_started or self._pending_skipped is not None:
            raise RuntimeError("Adaptive worker observed an invalid optimizer event")
        # Unified flag: a guard skip (no scaler) never shows up in
        # ``optimizer_step_was_skipped`` and must not be counted as effective.
        from library.training.scaler_ownership import update_skipped as _update_skipped

        skipped = _update_skipped(state)
        self._pending_skipped = skipped
        self.attempted_steps += 1
        if not skipped:
            self.effective_steps += 1
            self.consecutive_skips = 0
            return
        # A skipped step is not an effective step: weights did not move, so it can
        # never count toward completed_steps or the measured step time.
        self.skipped_steps += 1
        self.consecutive_skips += 1
        if self.consecutive_skips > self.max_nonfinite_skips:
            raise NonfiniteSkip(
                "Adaptive worker stopped: loss scaler skipped optimizer update "
                f"({self.consecutive_skips} consecutive skips > budget "
                f"{self.max_nonfinite_skips}; effective={self.effective_steps}, "
                f"skipped={self.skipped_steps})"
            )
        logging.getLogger(__name__).warning(
            "Adaptive worker tolerated a skipped optimizer update "
            "(%d/%d consecutive); it does not count as an effective step",
            self.consecutive_skips,
            self.max_nonfinite_skips,
        )

    def commit(self, state):
        # ``_pending_skipped`` is only cleared after a successful commit, so a
        # retried commit (e.g. atomic save failure) legitimately arrives with no
        # pending event. Accept either identity: the measurement counters advanced
        # by ``after_optimizer``, or the legacy cursor relation used by save-retry
        # tests that drive ``commit`` directly.
        counters_match = state.global_step == self.effective_steps + self.skipped_steps
        cursor_match = state.global_step == self.completed_steps + 1
        if not self.optimizer_started or not (counters_match or cursor_match):
            raise RuntimeError("Adaptive worker observed an invalid optimizer boundary")
        self.stage = "checkpoint"
        from .resume_runtime import enabled, commit
        saved = commit(self, state) if enabled(self.args) else save_training_state(state, self.directory)
        previous = self.saved_state
        self.saved_state = str(saved)
        # ``completed_steps`` is the resume cursor identity (resume_checkpoint
        # requires global_step == completed_steps == step), so it keeps tracking
        # global_step; effective/skipped stay as the measurement counters.
        self.completed_steps = state.global_step
        self._pending_skipped = None
        write_result(self.directory / "latest-state.json", self.result("running"))
        if previous is not None and not enabled(self.args):
            try:
                shutil.rmtree(previous)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove previous owned adaptive snapshot")
        self.stage = "between_steps"

    def finish(self, state):
        if self._pending_skipped is not None:
            raise RuntimeError("Adaptive worker returned mid-step")
        target = int(getattr(state.args, "max_train_steps", 0) or 0)
        if self.effective_steps < target:
            # Ran out of step budget before collecting enough real updates.
            raise NonfiniteSkip(
                f"Adaptive worker collected {self.effective_steps} effective steps "
                f"< target {target} ({self.skipped_steps} skipped)"
            )
        if (
            self.completed_steps <= 0
            or self.effective_steps + self.skipped_steps != state.global_step
        ):
            raise RuntimeError("Adaptive worker returned without completing requested optimizer steps")
        self.finished = True
        self.stage = "finished"

    def result(self, status):
        capacity = self.capacity or {}
        effective_cap = capacity.get("effective_cap")
        if effective_cap is not None and self.host_limit is not None:
            effective_cap = min(effective_cap, self.host_limit)
        from .resume_runtime import enabled
        recovery = enabled(self.args)
        return {"status": status, "stage": self.stage,
                "model_limit": capacity.get("model_limit"),
                "requested_cap": capacity.get("requested_cap"),
                "effective_cap": effective_cap,
                "host_limit": self.host_limit,
                "host_budget_reason": self.host_budget_reason,
                "optimizer_started": self.optimizer_started,
                "completed_steps": self.completed_steps, "saved_state": self.saved_state,
                "attempted_steps": self.attempted_steps,
                "effective_steps": self.effective_steps,
                "skipped_steps": self.skipped_steps,
                "max_nonfinite_skips": getattr(self, "max_nonfinite_skips", 0),
                "precision_contract_id": self.precision_contract_id,
                "precision_manifest_id": self.precision_manifest_id,
                "committed_checkpoint": self.saved_state if recovery else None,
                "data_cursor_resume_supported": recovery,
                "resume_limitation": None if recovery else "dataset_order_and_prefetch_cursor_not_restored"}


def notify(trainer, event, state=None):
    from library.training.precision_report import notify as precision_notify

    precision_notify(event, state)
    runtime = getattr(trainer, "_adaptive_training_runtime", None)
    if runtime is None:
        return
    from .native_runtime import NativeStartupRuntime
    if event == "setup" or (event == "commit" and type(runtime) is not NativeStartupRuntime):
        # The commit path also writes a progress result; validate before it
        # can publish an obsolete dtype identity.
        runtime.validate_precision(getattr(trainer, "_native_precision_model",
                                           getattr(trainer, "_adaptive_precision_manifest_model", None)))
    if event in {"after_optimizer", "commit", "finish"}:
        getattr(runtime, event)(state)
    else:
        runtime.phase(event)


def validate_entry(trainer, args):
    if getattr(args, "adaptive_oom_retry", False):
        raise ValueError("adaptive_oom_retry must use the supervised train.py CLI entry")
    private = getattr(args, "_adaptive_training_worker", None)
    runtime = getattr(trainer, "_adaptive_training_runtime", None)
    if private or runtime is not None:
        from .native_runtime import NativeStartupRuntime
        from .native_precision import native_enabled
        expected_type = NativeStartupRuntime if native_enabled(args) else TrainingRuntime
        if not private or type(runtime) is not expected_type:
            raise ValueError("Adaptive training worker requires an isolated runtime")
