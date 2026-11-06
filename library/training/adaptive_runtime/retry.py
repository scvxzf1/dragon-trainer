"""Pure, bounded OOM recovery decisions; never alter numerical precision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class MemoryPlan:
    blocks_to_swap: int = 0
    gradient_checkpointing: bool = False
    micro_batch: int = 1
    accumulation: int = 1

    def __post_init__(self):
        if self.blocks_to_swap < 0 or min(self.micro_batch, self.accumulation) < 1:
            raise ValueError("Invalid memory plan")


@dataclass(frozen=True)
class RetryLimits:
    max_blocks: int
    max_attempts: int = 4
    swap_increment: int = 2
    allow_batch_change: bool = False
    allow_checkpoint_change: bool = False
    swap_strategy: str = "linear"

    def __post_init__(self):
        if (self.max_blocks < 0 or min(self.max_attempts, self.swap_increment) < 1
                or self.swap_strategy not in {"linear", "exponential"}):
            raise ValueError("Invalid recovery limits")


def next_plan(current: MemoryPlan, limits: RetryLimits, *, stage: str,
              attempt: int = 0, remaining_attempts: int | None = None):
    """Return (plan, reason), or None when no authorized adjustment remains."""
    if current.blocks_to_swap > limits.max_blocks:
        raise ValueError("Current swap exceeds the model/host-memory limit")
    if stage in ("model_load", "forward", "backward"):
        if current.blocks_to_swap < limits.max_blocks:
            increment = (limits.swap_increment * 2 ** attempt
                         if limits.swap_strategy == "exponential" else limits.swap_increment)
            count = min(limits.max_blocks, current.blocks_to_swap + increment)
            if limits.swap_strategy == "exponential" and remaining_attempts == 1:
                count = limits.max_blocks
            return replace(current, blocks_to_swap=count), "increase_block_swap"
    if stage in ("forward", "backward"):
        if limits.allow_checkpoint_change and not current.gradient_checkpointing:
            return replace(current, gradient_checkpointing=True), "enable_checkpointing"
        if limits.allow_batch_change:
            # Only exact divisors preserve the configured effective batch.
            divisor = next((n for n in range(2, current.micro_batch + 1)
                            if current.micro_batch % n == 0), None)
            if divisor:
                return replace(current, micro_batch=current.micro_batch // divisor,
                               accumulation=current.accumulation * divisor), "reduce_micro_batch"
    return None


def run_recovery(initial, limits, runner, *, record, expected_precision_contract=None,
                 capacity=None, startup_only=False):
    """Runner must start a FRESH process and return a structured worker result.

    Only explicit CUDA OOM results are retried. A started optimizer requires a
    committed checkpoint path supplied by the worker; its validation/loading is
    the training adapter's responsibility. No checkpoint means fail closed.
    When supplied, ``expected_precision_contract`` must match every structured
    ``ok`` or ``cuda_oom`` result before any next plan is considered.
    """
    if startup_only and (limits.allow_batch_change or limits.allow_checkpoint_change):
        raise ValueError("Native startup retry may only increase block swap")
    plan, resume = initial, None
    seen = set()
    history = []
    realized_manifest = None
    effective_cap = limits.max_blocks
    cap_reason = "model_structure_or_user_cap"

    def report(status, **extra):
        value = {"status": status, "attempts": history, **extra}
        if capacity is not None:
            value.update(model_limit=capacity["model_limit"],
                         requested_cap=capacity["requested_cap"],
                         effective_cap=effective_cap, cap_reason=cap_reason)
        record(value)
        return value

    for attempt in range(limits.max_attempts):
        if plan in seen:
            history.append({"attempt": attempt, "plan": asdict(plan),
                            "stop_reason": "repeated_plan"})
            return report("failed")
        seen.add(plan)
        result = runner(plan, attempt=attempt, resume=resume)
        if not isinstance(result, dict):
            result = {"status": "error", "error": "invalid structured worker result"}
        history.append({"attempt": attempt, "plan": asdict(plan), "result": result})
        report("running")
        status = result.get("status")
        worker_capacity = capacity is not None and (
            status in {"ok", "cuda_oom"}
            or (status == "host_limit" and "model_limit" in result)
        )
        if worker_capacity:
            host_limit = result.get("host_limit")
            if (plan.blocks_to_swap > 0 and host_limit is None
                    and (status == "ok" or (status == "cuda_oom"
                         and result.get("stage") != "model_load"))):
                history[-1]["stop_reason"] = "missing_host_budget"
                break
            if (result.get("model_limit") != capacity["model_limit"]
                    or result.get("requested_cap") != capacity["requested_cap"]
                    or result.get("effective_cap") != min(
                        capacity["effective_cap"],
                        host_limit if type(host_limit) is int else capacity["effective_cap"],
                    )):
                history[-1]["stop_reason"] = "model_capacity_mismatch"
                break
            if host_limit is not None:
                if type(host_limit) is not int or not 0 <= host_limit <= capacity["model_limit"]:
                    history[-1]["stop_reason"] = "invalid_host_limit"
                    break
                if host_limit < effective_cap:
                    effective_cap = host_limit
                    cap_reason = "measured_host_budget"
                    history[-1]["cap_adjustment"] = cap_reason
                if plan.blocks_to_swap > effective_cap:
                    history[-1]["stop_reason"] = "host_limit"
                    break
        if expected_precision_contract is not None:
            actual = result.get("precision_contract_id")
            if status in {"ok", "cuda_oom"} and actual != expected_precision_contract:
                history[-1]["stop_reason"] = "precision_contract_mismatch"
                break
        manifest = result.get("precision_manifest_id")
        if startup_only and status in {"ok", "cuda_oom"} and manifest is not None:
            if (not isinstance(manifest, str) or len(manifest) != 64
                    or any(c not in "0123456789abcdef" for c in manifest)):
                history[-1]["stop_reason"] = "invalid_precision_manifest"
                break
        if expected_precision_contract is not None and status in {"ok", "cuda_oom"}:
            if manifest is None:
                # No DiT exists yet when loading itself OOMs; its dtype
                # partition is checked as soon as a later attempt installs it.
                if not (status == "cuda_oom" and result.get("stage") == "model_load"
                        and result.get("optimizer_started") is False):
                    history[-1]["stop_reason"] = "missing_precision_manifest"
                    break
            elif realized_manifest is None:
                realized_manifest = manifest
            elif manifest != realized_manifest:
                history[-1]["stop_reason"] = "realized_precision_manifest_mismatch"
                break
        if startup_only and status == "cancelled":
            return report("cancelled")
        if startup_only and status in {"ok", "cuda_oom"}:
            progress = result.get("optimizer_started")
            if type(progress) is not bool:
                history[-1]["stop_reason"] = "unknown_optimizer_progress"
                break
            if status == "ok":
                count = result.get("completed_steps")
                if (result.get("finished") is not True or result.get("stage") != "finished"
                        or progress is not True or type(count) is not int or count <= 0):
                    history[-1]["stop_reason"] = "incomplete_native_training"
                    break
            elif progress:
                history[-1]["stop_reason"] = "native_startup_only_optimizer_started"
                break
            elif result.get("stage") not in {"model_load", "forward", "backward"}:
                history[-1]["stop_reason"] = "unsupported_startup_stage"
                break
        if status == "ok":
            extra = {"selected": asdict(plan)}
            if expected_precision_contract is not None:
                extra["precision_contract_id"] = expected_precision_contract
            return report("ok", **extra)
        if status == "nonfinite_skip":
            # Distinct from a generic error: the trial produced more scaler skips
            # than its budget allows, which is a precision/config signal.
            history[-1]["stop_reason"] = "nonfinite_skip"
            break
        if status != "cuda_oom":
            break
        progress = result.get("optimizer_started")
        if not isinstance(progress, bool):
            history[-1]["stop_reason"] = "unknown_optimizer_progress"
            break
        if progress:
            resume = result.get("committed_checkpoint")
            if not resume:
                history[-1]["stop_reason"] = "no_committed_checkpoint"
                break
        if attempt + 1 == limits.max_attempts:
            history[-1]["stop_reason"] = "attempts_exhausted"
            break
        current_limits = replace(limits, max_blocks=effective_cap)
        change = next_plan(plan, current_limits, stage=result.get("stage", "unknown"),
                           attempt=attempt, remaining_attempts=limits.max_attempts - attempt - 1)
        if change is None:
            history[-1]["stop_reason"] = "no_authorized_adjustment"
            break
        plan, reason = change
        history[-1]["next_reason"] = reason
    return report("failed")
