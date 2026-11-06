"""Bounded numerical-profile selection plus OOM planning for disposable probes."""

from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
import math

from .retry import next_plan


@dataclass(frozen=True)
class PrecisionProfile:
    name: str
    precision: str
    fp32_patterns: tuple[str, ...] = ()
    loss_scale: float = 1.0

    def __post_init__(self):
        if (not self.name or self.precision not in {"bf16", "fp16-islands", "fp32-reference"}
                or not math.isfinite(self.loss_scale) or self.loss_scale < 1
                or any(not isinstance(p, str) or not p for p in self.fp32_patterns)
                or self.precision != "fp16-islands" and (self.fp32_patterns or self.loss_scale != 1)):
            raise ValueError("Invalid experimental precision profile")


def applied_profile(result, profile, plan):
    resolved = result.get("fp32_modules")
    return (result.get("precision_profile_resolved") is True
            and result.get("precision") == profile.precision
            and result.get("loss_scale", 1.0) == profile.loss_scale
            and result.get("swap") == plan.blocks_to_swap
            and isinstance(resolved, list) and all(isinstance(n, str) for n in resolved)
            and all(any(fnmatchcase(n, p) for n in resolved) for p in profile.fp32_patterns))


def comparison_gate(comparison):
    if (not isinstance(comparison, dict) or comparison.get("status") != "compared"
            or comparison.get("scope") != "fixed_pre_step_state_and_inputs_not_quality_certificate"):
        return None
    cases = comparison.get("cases")
    gate = comparison.get("within_experimental_tolerances")
    if (not isinstance(cases, list) or not cases or type(gate) is not bool
            or any(not isinstance(c, dict) or type(c.get("within_experimental_tolerances")) is not bool
                   for c in cases)
            or gate != all(c["within_experimental_tolerances"] for c in cases)):
        return None
    return gate


def search_profiles(initial, limits, profiles, runner, comparator, *, record):
    """Callbacks must run fresh workers and compare their complete captures.

    Candidate order is explicit, not a claim that extra FP32 monotonically helps.
    No user progress or checkpoint is carried between disposable attempts.
    Numerical acceptance is scoped to the supplied fixed-state reference cases.
    """
    if (not profiles or len({p.name for p in profiles}) != len(profiles)
            or initial.blocks_to_swap > limits.max_blocks or not initial.gradient_checkpointing
            or initial.micro_batch != 1 or initial.accumulation != 1 or limits.allow_batch_change):
        raise ValueError("Unique profiles, valid swap bounds and fixed checkpointed batch=1 required")
    history = []
    report = {"status": "searching", "production_ready": False, "precision_calibrated": False,
              "full_model_calibrated": False, "calibration_only": True, "attempts": history}
    plan, index = initial, 0
    for attempt in range(limits.max_attempts):
        profile = profiles[index]
        item = {"attempt": attempt, "profile": asdict(profile), "plan": asdict(plan)}
        history.append(item)
        record(report)
        try:
            result = runner(profile, plan, attempt=attempt)
        except Exception as exc:
            item.update(stop_reason="worker_exception", error=f"{type(exc).__name__}: {exc}")
            break
        item["result"] = result
        if not isinstance(result, dict) or result.get("disposable_probe") is not True:
            item["stop_reason"] = "worker_failed_or_not_disposable"
            break
        status = result.get("status")
        if status == "ok":
            if not applied_profile(result, profile, plan):
                item["stop_reason"] = "precision_profile_not_applied"
                break
            try:
                comparison = comparator(attempt)
            except Exception as exc:
                item.update(stop_reason="comparison_failed", error=f"{type(exc).__name__}: {exc}")
                break
            item["comparison"] = comparison
            gate = comparison_gate(comparison)
            if gate is None:
                item["stop_reason"] = "invalid_fixed_state_comparison"
                break
            if gate:
                report.update(status="probe_validated", selected_profile=asdict(profile),
                              selected_plan=asdict(plan), resolved_fp32_modules=result["fp32_modules"])
                record(report)
                return report
            item["next_reason"] = "numerical_tolerance_failed"
        elif status == "cuda_oom":
            change = next_plan(plan, limits, stage=result.get("stage", "unknown"))
            if change is not None:
                plan, item["next_reason"] = change
                record(report)
                continue
            item["next_reason"] = "profile_memory_budget_exhausted"
        elif status == "nonfinite_skip":
            item["next_reason"] = "nonfinite_skip"
            item["stop_reason"] = "nonfinite_skip"
            break
        elif status == "error" and result.get("failure_kind") == "nonfinite":
            item["next_reason"] = "nonfinite_profile"
        else:
            item["stop_reason"] = "nonrecoverable_worker_failure"
            break
        index += 1
        if index == len(profiles):
            item["stop_reason"] = "profiles_exhausted"
            break
        record(report)
    report["status"] = "failed"
    if history and "stop_reason" not in history[-1]:
        history[-1]["stop_reason"] = "attempt_limit"
    record(report)
    return report
