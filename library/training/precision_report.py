"""Lightweight, main-rank training precision observations; never a quality gate."""
from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid

logger = logging.getLogger(__name__)
_current = ContextVar("training_precision_report", default=None)
_CONTRACT_KEYS = (
    "model_family", "network_module", "network_args", "network_dim", "network_alpha",
    "network_train_unet_only", "network_train_text_encoder_only", "attn_mode",
    "gradient_checkpointing", "selective_checkpoint", "blocks_to_swap",
    "cpu_offload_checkpointing", "unsloth_offload_checkpointing", "torch_compile",
    "compile_inductor_mode", "compile_dynamic_seq", "base_compute", "nf4_prequantized_path",
    "full_fp16", "full_bf16", "adaptive_loss_scale", "adaptive_fp32_modules",
    "adaptive_oom_retry", "gradient_accumulation_steps",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _scalar(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_scalar(item) for item in value]
    return str(value)


def _main_rank():
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    # RANK is global; LOCAL_RANK alone cannot distinguish multi-node rank zero.
    for key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "LOCAL_RANK"):
        if key in os.environ:
            try:
                return int(os.environ[key]) == 0
            except ValueError:
                return False
    return True


def _dtype_histogram(model, *, trainable_only=False):
    counts = defaultdict(lambda: {"parameter_tensors": 0, "elements": 0})
    if model is None:
        return {}
    for parameter in model.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        bucket = counts[str(parameter.dtype)]
        bucket["parameter_tensors"] += 1
        bucket["elements"] += parameter.numel()
    return dict(counts)


def _scale(accelerator):
    scaler = getattr(accelerator, "scaler", None)
    return None if scaler is None else float(scaler.get_scale())


class PrecisionReport:
    """Own one output artifact, observing only scalars and parameter metadata."""

    def __init__(self, args):
        self.args = args
        self.path = None
        self.accelerator = None
        self.state = None
        self.completion_observed = False
        self.enabled = _main_rank()
        self.data = {
            "schema_version": 1, "run_id": uuid.uuid4().hex,
            "status": "running", "stage": "initialization", "start_time": _now(), "end_time": None,
            "precision": {"requested": getattr(args, "adaptive_precision", "off") or "off"},
            "contract": {}, "gpu": None, "actual_parameters": {}, "amp": {},
            "resume": {"requested": bool(getattr(args, "resume", None) or getattr(args, "auto_resume", False)),
                       "inherited_global_step": 0},
            "updates": {"attempted": 0, "effective": 0, "skipped": 0, "total_global_step": 0,
                        "scope": "current invocation only; total_global_step includes inherited and skipped boundaries"},
            "validation_scope": {
                "hardware_selection": "hardware decision is not quality certification",
                "current_run_quality_certified": False,
                "historical_evidence": "external validation reports have their own model and run scopes; not certification of this run",
                "native_oom_retry_supported": "implemented_startup_only",
                "native_oom_retry_validation": "cpu_static_not_quality_certification",
                "update_recovery_supported": False,
            },
        }

    def configuration(self):
        args = self.args
        requested = self.data["precision"]["requested"]
        resolved = getattr(args, "adaptive_resolved_mode", None) or (None if requested == "auto" else requested)
        self.data["precision"].update({
            "resolved": resolved, "mixed_precision": getattr(args, "mixed_precision", None),
            "selection_reason": getattr(args, "adaptive_resolution_reason", None) or (
                "hardware policy or configured fallback; not numerical certification" if requested == "auto"
                else "explicit configuration"),
        })
        self.data["contract"] = {key: _scalar(getattr(args, key, None)) for key in _CONTRACT_KEYS}
        self.data["resume"]["requested"] |= bool(getattr(args, "resume", None))
        if getattr(args, "model_family", "anima") in {"krea2", "krea2_raw"}:
            self.data["validation_scope"].update(native_oom_retry_supported="unsupported_family",
                                                   native_oom_retry_validation="not_applicable")

    def write(self):
        if not self.enabled:
            return
        self.configuration()
        output = getattr(self.args, "output_dir", None)
        if not output:
            logger.warning("precision report unavailable: actual output_dir is unknown")
            return
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".precision_report-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            if self.path is None:
                candidate = directory / "precision_report.json"
                try:
                    # Publish a complete file atomically without clobbering any prior run.
                    os.link(temporary, candidate)
                except FileExistsError:
                    candidate = directory / f"precision_report.{self.data['run_id']}.json"
                    os.link(temporary, candidate)
                self.path = candidate
            else:
                os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def accelerator_ready(self, accelerator):
        import torch

        self.accelerator = accelerator
        self.enabled = bool(accelerator.is_main_process)
        self.data["amp"].update({
            "native_amp": bool(getattr(accelerator, "native_amp", False)),
            "autocast_mixed_precision": getattr(accelerator, "mixed_precision", None),
            "scaler_present": getattr(accelerator, "scaler", None) is not None,
            "scaler_initial": _scale(accelerator), "scaler_resumed": None,
        })
        device = torch.device(accelerator.device)
        self.data["gpu"] = {"device": str(device), "cuda": device.type == "cuda"}
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            self.data["gpu"].update({"name": properties.name,
                                     "compute_capability": [properties.major, properties.minor],
                                     "total_memory_bytes": properties.total_memory})
        self.phase("accelerator_ready")

    def models_prepared(self, unet, network):
        accelerator = self.accelerator
        model = accelerator.unwrap_model(unet)
        adapter = accelerator.unwrap_model(network)
        self.data["actual_parameters"] = {
            "model": _dtype_histogram(model), "adapter": _dtype_histogram(adapter),
            "trainable_adapter": _dtype_histogram(adapter, trainable_only=True),
        }
        from .adaptive_runtime.native_family import report_guards
        self.data["fp32_guards"] = report_guards(model, getattr(self.args, "model_family", "anima"))
        self.phase("models_prepared")

    def resumed(self, step):
        self.data["resume"]["inherited_global_step"] = int(step)
        self.data["updates"]["total_global_step"] = int(step)
        self.data["amp"]["scaler_resumed"] = _scale(self.accelerator)
        self.phase("resume_ready")

    def phase(self, event, state=None):
        self.data["stage"] = event
        if event == "finish":
            self.completion_observed = True
        if state is not None:
            self.state = state
        # Writes are lifecycle-boundary only, never per-microstep resource probes.
        if event in {"model_load", "models_prepared", "resume_ready", "loop_ready", "final_save", "finish", "accelerator_ready"}:
            self.write()

    def optimizer(self, state, *, after=False, skipped=None):
        self.state = state
        if not state.accelerator.sync_gradients:
            return
        updates = self.data["updates"]
        if after:
            # ``skipped`` lets the no-scaler gradient guard report a dropped update
            # the same way a GradScaler skip is reported. Known limitation: this
            # bucket merges both causes (scaler overflow vs guard skip), so the
            # report cannot tell them apart. Use ``update_accounting`` below for the
            # authoritative per-attempt view.
            if skipped is None:
                skipped = bool(state.accelerator.optimizer_step_was_skipped)
            updates["skipped" if skipped else "effective"] += 1
        else:
            updates["attempted"] += 1

    def transaction_accounting(self, state):
        """Publish the window-transaction counters (R5) next to the legacy counters.

        Legacy keys are **kept exactly as the baseline produced them** (contract C3):
        ``attempted`` / ``effective`` / ``skipped`` still count the *commit* path only,
        so on the epoch-loop route ``attempted == effective == windows.committed`` and
        ``skipped`` is structurally 0 (a rejected update never reaches the commit path).
        New keys (``step_attempts`` and the attribution layers) carry the authoritative
        per-boundary view; the ``scope`` string says which is which.

        Scopes (contract R5.4 / C8.2): these counters are **local** to this process, the
        artifact is written by the **main rank** only, ``step_attempts`` / ``windows.*``
        cover the **window route** only (``_run_step`` direct callers are counted in
        ``direct_route_boundaries``), and the verdict / apply-drop decisions are
        **world-wide** only while the rank consensus is active.
        """
        from library.training import numerical_observation, window_coordination

        self.state = state
        snapshot = window_coordination.snapshot(state)
        updates = self.data["updates"]
        updates["scope"] = (
            "local process counters; LEGACY attempted/effective/skipped count the commit "
            "path only (epoch-loop route: attempted == effective == windows.committed, "
            "skipped structurally 0); see step_attempts and attribution.* for every "
            "optimizer boundary attempt including replays and discarded windows"
        )
        updates["legacy_keys_note"] = (
            "attempted/effective/skipped keep baseline semantics (commit path only); "
            "they must not be read as the attempt total"
        )
        updates["step_attempts"] = snapshot["step_attempts"]
        updates["dropped_by_scaler_or_guard"] = snapshot["updates_skipped"]
        updates["skipped_by_scaler"] = snapshot["updates_skipped_by_scaler"]
        updates["dropped_without_scaler"] = snapshot["updates_dropped_without_scaler"]
        updates["dropped_with_scaler"] = snapshot["updates_dropped_with_scaler"]
        updates["direct_route_boundaries"] = snapshot["direct_route_boundaries"]
        updates["windows"] = {
            "committed": snapshot["windows_committed"],
            "replayed": snapshot["window_replays"],
            "discarded": snapshot["windows_discarded"],
            "consecutive_discards": snapshot["window_consecutive_discards"],
            "attempts": snapshot["window_attempts"],
        }
        updates["discard_causes"] = snapshot["window_discard_causes"]
        # Three attribution layers, kept apart on purpose (contract C4): an originating
        # cause is never inferred from the reduced world label, and a propagated drop is
        # never folded into an originating bucket.
        updates["attribution"] = {
            "L1_originating_causes": snapshot["originating_causes"],
            "L2_propagated_drops": snapshot["propagated_drops"],
            "L2_propagated_window_discards": snapshot["window_discard_propagated"],
            "L3_boundary_decisions": snapshot["world_decisions"],
            "L3_window_decisions": snapshot["window_world_decisions"],
            "note": "wire codes used for the reduction are a format convention, not a "
                    "severity ranking; L1 is built from local evidence only",
        }
        updates["scope_detail"] = {
            "counters": "local (per process)",
            "artifact": "main rank only",
            "window_counters": "window route only; _run_step direct callers counted in "
                               "direct_route_boundaries",
            "world_decision": "world-wide while the multi-rank consensus is active; local "
                              "for a single process",
            # Only genuinely uncovered paths stay here. "async bad window" and "native DDP
            # reducer" moved to the covered side: both are exercised on real devices by
            # tests/test_fp16_transaction_cuda.py (M18 bad-window 档, M17 accelerate DDP).
            # The two-rank combination gate (M19-R2 ⑥) is asserted there as well, but only
            # under the CUDA opt-in — a default CPU run still cannot claim it.
            "unverified_paths": ">2 ranks, mid-window zero_grad world consistency, "
                                "two-rank combination gate (M19-R2 ⑥) outside the CUDA "
                                "opt-in run",
            "write_cadence": "published on transaction boundaries only; no per-step "
                             "artifact writes",
        }
        self.data["nonfinite_observations"] = numerical_observation.snapshot()

    def finish(self, status, exc=None):
        self.data.update(status=status, end_time=_now())
        if exc is not None:
            self.data["exception"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        if self.state is not None:
            self.data["updates"]["total_global_step"] = int(self.state.global_step)
        if self.accelerator is not None:
            try:
                self.data["amp"]["scaler_final"] = _scale(self.accelerator)
            except Exception as error:
                logger.warning("precision report scaler observation failed: %s", error)
        try:
            self._resources()
        except Exception as error:
            logger.warning("precision report resource observation failed: %s", error)
        self.write()

    def _resources(self):
        # ru_maxrss is KiB on Linux, bytes on macOS. CUDA peaks are shared process
        # counters since their last reset by any caller, not isolated run peaks.
        try:
            import resource

            maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_max = int(maximum * (1 if sys.platform == "darwin" else 1024))
        except ImportError:
            import psutil

            rss_max = psutil.Process().memory_info().peak_wset
        self.data["resources"] = {"process_rss_max_bytes": rss_max,
                                  "rss_scope": "process lifetime",
                                  "cuda_peak_scope": "device process counters since last external reset; reporter does not reset"}
        if self.accelerator is not None:
            import torch

            device = torch.device(self.accelerator.device)
            if device.type == "cuda":
                self.data["resources"].update({
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                })


def _safe(report, method, *args, **kwargs):
    if report is None or not report.enabled:
        return
    try:
        getattr(report, method)(*args, **kwargs)
    except Exception as exc:
        logger.warning("precision report %s failed: %s", method, exc)


def observe(method, *args, **kwargs):
    _safe(_current.get(), method, *args, **kwargs)


def notify(event, state=None):
    observe("phase", event, state)


def current_report_path():
    report = _current.get()
    return str(report.path) if report is not None and report.path is not None else None


def reported_session(trainer, args, callback):
    """Cover entry rejection through final save without adding objects to args."""
    report = PrecisionReport(args)
    token = _current.set(report)
    _safe(report, "write")
    try:
        result = callback(trainer, args)
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else (
            "rejected" if report.data["stage"] in {"initialization", "contract_validation"} else "failed")
        _safe(report, "finish", status, exc)
        raise
    else:
        if report.completion_observed:
            status = "completed"
        elif report.data["stage"] == "debug_dataset":
            status = "debug_completed"
        elif report.data["stage"] in {"no_data", "network_not_created"}:
            status = "rejected"
        else:
            status = "incomplete"
        if status != "completed":
            report.data["termination_reason"] = report.data["stage"]
        _safe(report, "finish", status)
        return result
    finally:
        runtime = getattr(trainer, "_adaptive_training_runtime", None)
        if runtime is not None and hasattr(runtime, "precision_report_path"):
            runtime.precision_report_path = str(report.path) if report.path is not None else None
        _current.reset(token)


def report_entry_rejection(args, exc):
    """CLI hardware resolution can reject before a trainer/session exists."""
    report = PrecisionReport(args)
    _safe(report, "finish", "cancelled" if isinstance(exc, KeyboardInterrupt) else "rejected", exc)


def report_supervisor(report, summary, summary_path):
    """Observe child scalar evidence; the parent never claims loaded dtypes."""
    try:
        attempts = summary.get("attempts", [])
        status = summary["status"]
        results = [item.get("result", {}) for item in attempts]
        child = results[-1] if results else {}
        recovery = bool(getattr(report.args, "adaptive_oom_retry_resume", False))
        report.data.update(status={"ok": "completed", "failed": "failed", "running": "running"}.get(status, status),
                           stage="native_committed_resume_supervisor" if recovery else "native_startup_supervisor",
                           end_time=None if status == "running" else _now())
        report.data["supervisor"] = {
            "native_scope": "committed_resume" if recovery else "startup_only", "attempts": attempts,
            "selected": summary.get("selected"), "summary_path": str(summary_path),
            "failure_reason": attempts[-1].get("stop_reason", child.get("reason", child.get("status"))) if attempts else None,
            "child_reports": [result.get("precision_report_path") for result in results],
        }
        report.data["updates"].update(attempted=child.get("attempted_steps", 0),
                                     effective=child.get("effective_steps", 0),
                                     skipped=child.get("skipped_steps", 0),
                                     total_global_step=child.get("completed_steps", 0),
                                     scope="last child scalar observations; parent does not execute optimizer")
        report.write()
    except Exception as exc:
        logger.warning("precision supervisor report failed: %s", exc)
