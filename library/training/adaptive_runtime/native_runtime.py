"""Startup-only native AMP supervision, with scalar loop boundaries only."""

from pathlib import Path

from library.training.auto_block_swap.process import write_result
from .native_manifest import SCHEMA, base_metadata, execution_metadata, fingerprint


class NativeStartupRuntime:
    def __init__(self, directory, *, precision_contract_id=None, args=None):
        self.directory = Path(directory)
        self.precision_contract_id = precision_contract_id
        self.args = args
        from .native_family import native_family
        self.family = native_family(getattr(args, "model_family", "anima"))
        self.precision_manifest_id = None
        self.stage = "setup"
        self.optimizer_started = False
        self.finished = False
        self.completed_steps = 0
        self.attempted_steps = 0
        self.effective_steps = 0
        self.skipped_steps = 0
        # Native startup trials never run the skip budget, but ``result()`` is
        # shared with TrainingRuntime and reports it, so the field must exist.
        self.max_nonfinite_skips = 0
        self.saved_state = None
        self.precision_report_path = None
        self.capacity = getattr(args, "_adaptive_swap_capacity", None)
        self.host_limit = None
        self.host_budget_reason = "not_measured"
        self._base = None
        self._model = None
        self._network = None
        self._accelerator = None
        self._execution = None
        self._pending_skipped = None

    def capture_model(self, model):
        self._validate_source_stats()
        if self._base is not None:
            raise ValueError("precision_contract_mismatch: native base captured twice")
        self._base = base_metadata(model, family=self.family)
        self._model = model
        self.precision_manifest_id = fingerprint({"schema": SCHEMA, "base": self._base})

    def validate_swap_capacity(self, model, family):
        from .training_runtime import TrainingRuntime

        # Share metadata-only host budget accounting, not snapshot behavior.
        TrainingRuntime.validate_swap_capacity(self, model, family)

    def validate_precision(self, model=None):
        if self.precision_contract_id is not None:
            from .training_precision import validate_local_precision_contract
            validate_local_precision_contract(self.args, self.precision_contract_id)
        model = model if model is not None else self._model
        if model is not None:
            if self._base is None or base_metadata(model, family=self.family) != self._base:
                raise ValueError("precision_contract_mismatch: native base metadata drifted")
        if self._execution is not None:
            if execution_metadata(self._model, self._network, self._accelerator, family=self.family) != self._execution:
                raise ValueError("precision_contract_mismatch: native execution policy drifted")

    def prepared(self, model, network, accelerator):
        self._validate_source_stats()
        if getattr(self.args, "_native_source_identities", None):
            # Coarse filesystem timestamps can miss an equal-size rewrite.
            # Recheck content once after base/adapter loading, never per step.
            from .contract import validate_native_sources
            validate_native_sources(self.args)
        if getattr(self.args, "gradient_accumulation_steps", 1) != 1:
            raise ValueError("Native startup runtime requires accumulation=1")
        unwrap = getattr(accelerator, "unwrap_model", lambda value: value)
        model = unwrap(model)
        self.validate_precision(model)
        if self._base is None:
            raise ValueError("precision_contract_mismatch: native base was not captured")
        self._model = model
        self._network = unwrap(network)
        self._accelerator = accelerator
        self._execution = execution_metadata(self._model, self._network, accelerator, family=self.family)

    def _validate_source_stats(self):
        if getattr(self.args, "_native_source_identities", None):
            from .contract import validate_native_source_stats
            validate_native_source_stats(self.args)

    def phase(self, stage):
        if stage in {"forward", "backward", "optimizer"}:
            if self._execution is None:
                raise RuntimeError("Native worker cannot train before prepared validation")
        self.stage = stage
        if stage == "optimizer":
            # Reject drift before optimizer.step can modify parameters.
            self.validate_precision()
            self.optimizer_started = True
            self.attempted_steps += 1

    def after_optimizer(self, state):
        if not self.optimizer_started or self._pending_skipped is not None:
            raise RuntimeError("Native worker observed an invalid optimizer event")
        self._pending_skipped = bool(state.accelerator.optimizer_step_was_skipped)
        self.skipped_steps += int(self._pending_skipped)
        self.effective_steps += int(not self._pending_skipped)

    def commit(self, state):
        self.validate_precision()
        if (self._execution is None or not self.optimizer_started or self._pending_skipped is None
                or state.global_step != self.completed_steps + 1):
            raise RuntimeError("Native worker observed an invalid optimizer boundary")
        from .resume_runtime import enabled, commit
        if enabled(self.args):
            self.stage = "checkpoint"
            commit(self, state)
        self.completed_steps = state.global_step
        self._pending_skipped = None
        self.stage = "between_steps"
        self.directory.mkdir(parents=True, exist_ok=True)
        write_result(self.directory / "latest-state.json", self.result("running"))

    def finish(self, state):
        self.validate_precision()
        if (self._execution is None or self._pending_skipped is not None or self.completed_steps <= 0
                or self.completed_steps != state.global_step
                or state.global_step != state.args.max_train_steps):
            raise RuntimeError("Native worker returned without completing the requested training loop")
        self.finished = True
        self.stage = "finished"
        from library.training.precision_report import current_report_path
        self.precision_report_path = current_report_path()

    def result(self, status):
        from .training_runtime import TrainingRuntime

        if status == "ok" and not self.finished:
            raise RuntimeError("Native worker cannot report success before loop completion")
        result = TrainingRuntime.result(self, status)
        from .resume_runtime import enabled
        result.update({"schema": SCHEMA, "precision_schema": SCHEMA, "attempted_steps": self.attempted_steps,
                       "effective_steps": self.effective_steps,
                       "skipped_steps": self.skipped_steps, "finished": self.finished,
                       "precision_report_path": self.precision_report_path,
                       "resume_limitation": None if enabled(self.args) else "native_startup_retry_only_no_training_state"})
        return result
