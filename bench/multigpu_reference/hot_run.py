"""Bounded real-Anima run entry: RUN-A/B/C preflight (t19 F2).

This module **builds and validates** the command for a bounded run against the
project's own ``train.py``. It never executes anything by itself: execution is
gated behind ``MULTIGPU_REFERENCE_HOT=1`` *and* a caller that passes
``execute=True``, and the verification task owns that call.

Honesty rules baked into the type system:

* an unreadable/missing asset is ``base_asset_missing`` — never "assume the cache
  is fine";
* the 4096-token fixed-shape asset set is only an **L1/L2 candidate**; RUN-A/B/C
  additionally require the per-sample noise/timestep L3 fixture, and without it
  the entry returns ``blocked`` instead of a toy green;
* every run gets a unique output root under
  ``output/tests/multigpu_reference_validation/<run-id>/`` — the user's output
  tree and model cache are read-only inputs;
* no download and no preprocessing step is ever part of the command.
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

HOT_ENV = "MULTIGPU_REFERENCE_HOT"
OUTPUT_PARENT = "output/tests/multigpu_reference_validation"

#: One fixed output stem + seed for all three runs, so their状态/weight names and
#: random draws line up (contract §13 V-2.1/V-2.3).
RUN_OUTPUT_NAME = "anima-multigpu-ref"
SEED = 20261107


def _state_dir_name(output_name: str, step: int) -> str:
    """The state directory the trainer really writes (``STEP_STATE_NAME``).

    Taken from the trainer's own constant instead of re-spelling the pattern: the
    RUN-C stop directory has to be the one ``save_and_remove_state_stepwise``
    creates, otherwise the verifier waits for a directory that never appears
    (review R3-F4).
    """
    from library.training.checkpoints import (
        DEFAULT_STEP_NAME,
        STEP_STATE_NAME,
        default_if_none,
    )

    return STEP_STATE_NAME.format(
        default_if_none(output_name, DEFAULT_STEP_NAME), int(step)
    )

CODE_HOT_NOT_RELEASED = "hot_run_not_released"
CODE_BASE_ASSET_MISSING = "hot_run_base_asset_missing"
CODE_OUTPUT_ROOT_INVALID = "hot_run_output_root_invalid"
CODE_BUDGET_EXCEEDED = "hot_run_budget_exceeded"
CODE_L3_FIXTURE_MISSING = "hot_run_l3_fixture_missing"
CODE_L3_FIXTURE_INVALID = "hot_run_l3_fixture_invalid"
CODE_L2_ONLY_CANDIDATE = "hot_run_l2_candidate_only"

#: Frozen budgets from the team goal. A/B/C together must stay inside these.
#:
#: ``TOTAL_UPDATE_BUDGET`` is the cap for **one batch of runs** (the single DAG that
#: A/B/C form together). The cumulative budget the team is allowed to spend is
#: *not* a constant here: the caller expresses it as ``prior_updates`` (updates
#: already charged by earlier, user-approved runs) plus this run's
#: ``spec.budgeted_updates``, and ``plan_hot_run`` refuses when that sum crosses the
#: cap. A user-approved increase is therefore expressed by passing the real
#: ``prior_updates``, never by editing this number.
TOTAL_UPDATE_BUDGET = 50
STEP_CAP_PER_RUN = 20
MINUTES_PER_RUN = 20
MINUTES_TOTAL = 80


@dataclass(frozen=True)
class RunSpec:
    name: str
    world_size: int
    steps: int
    description: str
    needs_l3_fixture: bool = True
    #: continuation steps counted separately in the budget (RUN-C = 10 + 10)
    resume_steps: int = 0
    #: Explicit stop point for a *short probe* (F-T22-3). Without it the stop point
    #: is ``steps``, so a probe that only wants the first state directory has to
    #: set ``--save_every_n_steps 1`` and would still run to ``steps`` because the
    #: verifier waits for the state dir of step ``steps``. With ``stop_at_step`` the
    #: save cadence, the expected committed step and ``stop_when_state_dir`` all
    #: follow the probe horizon, so the run spends exactly that many updates.
    stop_at_step: Optional[int] = None

    @property
    def stop_step(self) -> int:
        """The step whose state directory ends the commit phase."""
        return int(self.stop_at_step or self.steps)

    @property
    def budgeted_updates(self) -> int:
        """What this run costs against the 50-update total (RUN-C counts 20)."""
        return int(self.steps) + int(self.resume_steps)


#: RUN-A/B/C are the three already-approved shapes of the bounded real run.
RUN_SPECS: dict[str, RunSpec] = {
    "RUN-A": RunSpec("RUN-A", 1, 10, "single GPU, 10 updates"),
    "RUN-B": RunSpec("RUN-B", 2, 20, "two GPUs, 20 updates"),
    "RUN-C": RunSpec(
        "RUN-C", 2, 10, "two GPUs, 10 + 10 resume updates", resume_steps=10
    ),
}


def pinned_order_env() -> str:
    """The env var name of the pinned-order gate (single source: the bench module)."""
    from bench.multigpu_reference import loader_order

    return loader_order.ENV


def verify_pinned_gate(plan: "HotRunPlan", env: Mapping[str, str]) -> dict:
    """Read the gate back out of a plan and the env that was actually handed over.

    Returns the recorded facts and raises when the artifact would be unable to prove
    which mode ran — the failure t35 hit, where the gate was exported by hand outside
    the plan and could not be audited afterwards (V40-F3).
    """
    name = pinned_order_env()
    recorded = plan.facts.get("pinned_order")
    in_launch_env = (plan.facts.get("launch_env") or {}).get(name)
    in_handed_env = dict(env).get(name)
    if recorded is None:
        raise HotRunRefused(
            CODE_HOT_NOT_RELEASED,
            f"plan facts do not record the pinned gate ({name}); the artifact could "
            "not prove whether the measurement mode was on",
        )
    expected = "1" if recorded else None
    if in_launch_env != expected:
        raise HotRunRefused(
            CODE_HOT_NOT_RELEASED,
            f"facts['pinned_order']={recorded!r} but launch_env[{name}]={in_launch_env!r}",
        )
    if in_handed_env != expected:
        raise HotRunRefused(
            CODE_HOT_NOT_RELEASED,
            f"the runner env carries {name}={in_handed_env!r}, plan expects {expected!r}",
        )
    return {
        "pinned_order": bool(recorded),
        "env_name": name,
        "launch_env": in_launch_env,
        "handed_env": in_handed_env,
    }


def hot_released(env: Mapping[str, str] | None = None) -> bool:
    values = dict(os.environ if env is None else env)
    return str(values.get(HOT_ENV, "")).strip() in {"1", "true", "yes"}


@dataclass
class HotRunPlan:
    spec: RunSpec
    run_id: str
    output_root: Path
    command: list[str]
    status: str  # "ready" | "blocked"
    code: Optional[str] = None
    detail: str = ""
    facts: dict = field(default_factory=dict)

    def raise_if_not_ready(self) -> None:
        if self.status != "ready":
            raise HotRunRefused(self.code or "hot_run_blocked", self.detail)


class HotRunRefused(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _readable(path: Optional[str]) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.R_OK)


def check_base_assets(paths: Sequence[Optional[str]]) -> tuple[bool, list[str]]:
    """Read-only check: every base asset must already be on disk (never fetched)."""
    missing = [str(path) for path in paths if not _readable(path)]
    return (not missing, missing)


def inspect_l3_fixture(path: Optional[str]) -> dict:
    """Validate the L3 fixture **content**, not just that a file exists.

    L3 requires a per-``sample_id`` noise tensor and timestep so the numeric gate is
    reproducible sample by sample. A file that merely exists (e.g. ``{}``) is not a
    fixture, and treating it as one would produce exactly the "toy green" this
    entry exists to prevent.
    """
    if not _readable(path):
        return {"ok": False, "code": CODE_L3_FIXTURE_MISSING, "samples": 0, "detail": "no readable L3 fixture file"}
    # Same parser the *training* path uses (``library/training/multigpu/
    # noise_fixture.py``), so "the gate accepts it" and "the trainer can consume
    # it" cannot drift apart. A schema-only copy would be a second truth source.
    from library.training.multigpu import noise_fixture

    try:
        records = noise_fixture.load_fixture(str(path))
    except noise_fixture.NoiseFixtureError as exc:
        return {
            "ok": False,
            "code": exc.code,
            "samples": 0,
            "detail": str(exc),
        }
    return {
        "ok": True,
        "code": None,
        "samples": len(records),
        "detail": f"{len(records)} per-sample noise/timestep entries",
    }


def _output_root(run_id: str, project_root: Path) -> Path:
    return project_root / OUTPUT_PARENT / run_id


#: The command prefix named a bare ``python``; a real child then resolved it
#: through its own PATH and died with ``ModuleNotFoundError: torch`` (t13 F-T13-3a).
CODE_INTERPRETER_INVALID = "hot_run_interpreter_invalid"


def resolve_interpreter(explicit: Optional[str] = None) -> str:
    """The interpreter that will run the trainer: an **absolute** path.

    Never a bare name. The previous prefix was ``"python"``, so the real child
    resolved it through ``PATH`` and started a python without this project's
    dependencies (``ModuleNotFoundError: torch`` nine seconds into t13's second
    attempt). ``sys.executable`` is the interpreter that is already running the
    harness, i.e. the project venv when the harness is started from it.
    """
    candidate = explicit if explicit is not None else sys.executable
    if not candidate:
        raise HotRunRefused(
            CODE_INTERPRETER_INVALID,
            "no interpreter available (sys.executable is empty); pass python=<absolute path>",
        )
    candidate = str(candidate)
    if not os.path.isabs(candidate):
        raise HotRunRefused(
            CODE_INTERPRETER_INVALID,
            f"the run interpreter must be an absolute path, got {candidate!r}: a bare "
            "name is resolved through the child's PATH and can select a python "
            "without torch",
        )
    return candidate


def clean_launch_env(env: Optional[Mapping[str, object]]) -> dict[str, str]:
    """The launch env with unset entries dropped and values stringified.

    ``ANIMA_ACCELERATE_NUM_PROCESSES`` used to be written as ``None`` for a single
    worker run; ``subprocess.Popen(env=...)`` rejects a ``None`` value outright
    (t13 F-T13-3b). A key that does not apply is *omitted*, never set to ``None``.
    """
    cleaned: dict[str, str] = {}
    for key, value in dict(env or {}).items():
        if value is None:
            continue
        cleaned[str(key)] = value if isinstance(value, str) else str(value)
    return cleaned


def plan_hot_run(
    name: str,
    *,
    train_script: str,
    config: str,
    base_assets: Sequence[Optional[str]],
    output_root_parent: str | os.PathLike[str] | None = None,
    l3_fixture: Optional[str] = None,
    project_root: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    prior_updates: int = 0,
    run_id: Optional[str] = None,
    python: Optional[str] = None,
    pinned_order: bool = True,
) -> HotRunPlan:
    """Validate every gate for one bounded run and return the would-be command.

    ``python`` overrides the interpreter used for the run; it must be an absolute
    path and defaults to ``sys.executable`` (see :func:`resolve_interpreter`).

    ``pinned_order`` requests the pinned data-order measurement gate. It defaults to
    on because every single-vs-dual comparison needs it: without it the two
    topologies consume unrelated descriptor permutations and the numbers compare
    different data. The request is recorded twice — as ``facts["pinned_order"]`` and
    as the real ``ANIMA_MULTIGPU_PINNED_ORDER=1`` in ``facts["launch_env"]`` — so the
    artifact can prove the gate was open instead of relying on whoever exported it
    by hand (t35 did exactly that, V40-F3). Pass ``pinned_order=False`` only for a
    run that must reproduce the production shuffle.
    """
    spec = RUN_SPECS.get(name)
    if spec is None:
        raise HotRunRefused(CODE_OUTPUT_ROOT_INVALID, f"unknown run {name!r}")
    values = dict(os.environ if env is None else env)
    if not hot_released(values):
        raise HotRunRefused(
            CODE_HOT_NOT_RELEASED,
            f"set {HOT_ENV}=1 to release the bounded real run; the verification "
            "task owns that call",
        )
    if spec.budgeted_updates > STEP_CAP_PER_RUN:
        raise HotRunRefused(
            CODE_BUDGET_EXCEEDED,
            f"{spec.budgeted_updates} budgeted updates (steps {spec.steps} + resume "
            f"{spec.resume_steps}) > cap {STEP_CAP_PER_RUN}",
        )
    if int(prior_updates) + spec.budgeted_updates > TOTAL_UPDATE_BUDGET:
        raise HotRunRefused(
            CODE_BUDGET_EXCEEDED,
            f"{prior_updates}+{spec.budgeted_updates} updates > total budget "
            f"{TOTAL_UPDATE_BUDGET}",
        )
    if not _readable(train_script) or not _readable(config):
        raise HotRunRefused(
            CODE_BASE_ASSET_MISSING,
            f"train script/config not readable: {train_script!r}, {config!r}",
        )
    ok, missing = check_base_assets(base_assets)
    if not ok:
        raise HotRunRefused(
            CODE_BASE_ASSET_MISSING,
            "base assets missing (this entry never downloads or preprocesses): "
            + ", ".join(missing),
        )
    if output_root_parent is not None:
        raise HotRunRefused(
            CODE_OUTPUT_ROOT_INVALID,
            "output root is fixed to "
            f"{OUTPUT_PARENT}/<run-id>/; do not point it at the user's output tree",
        )

    root = Path(project_root or os.getcwd())
    resolved_id = run_id or f"{name.lower()}-{uuid.uuid4().hex[:8]}"
    out = _output_root(resolved_id, root)
    if out.exists() and any(out.iterdir()):
        raise HotRunRefused(
            CODE_OUTPUT_ROOT_INVALID, f"run id {resolved_id} already has output"
        )

    # Build the launch prefix through the project's own launcher helper so the
    # command has the same shape the real entry point produces (``accelerate
    # launch --num_processes N`` for a multi-worker run).
    from library.runtime.launch import accelerate_training_command_prefix

    launch_env = dict(values)
    if spec.world_size > 1:
        launch_env["ANIMA_ACCELERATE_LAUNCH"] = "1"
        launch_env["ANIMA_ACCELERATE_NUM_PROCESSES"] = str(spec.world_size)
    else:
        launch_env["ANIMA_ACCELERATE_LAUNCH"] = "0"
    interpreter = resolve_interpreter(python)
    prefix = accelerate_training_command_prefix(
        interpreter, str(train_script), launch_env
    )
    horizon = spec.budgeted_updates  # RUN-C: one 20-step scheduler horizon
    # A constant LR and a fixed seed at a *shared* horizon: the scheduler horizon
    # is ``max_train_steps * num_processes``, so RUN-C must keep 20 in both phases
    # and the LR must not depend on the horizon at all, otherwise a Δθ comparison
    # measures the schedule instead of DDP.
    base = list(prefix) + [
        "--config_file",
        str(config),
        "--output_dir",
        str(out),
        "--output_name",
        RUN_OUTPUT_NAME,
        "--seed",
        str(SEED),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        # Exact replay of the sample order requires workers=0 (contract §13 V-2.2).
        "--max_data_loader_n_workers",
        "0",
        # S-0 (contract §13 V-2.4): without the seedable sampler the single-worker
        # and the two-worker run shuffle into different permutations, so the same
        # sample never lands on the same slot and a Δθ comparison measures the
        # data instead of DDP (t3 F-T3-1a). Opt-in flag; the run that produces the
        # reference numbers must carry the same flag.
        "--multigpu_seedable_sampler",
        "--max_train_steps",
        str(horizon),
    ]
    command = list(base)
    commit_command = list(base)
    resume_state = out / _state_dir_name(RUN_OUTPUT_NAME, spec.stop_step)
    if spec.resume_steps:
        # The interrupt point is a *real* state directory this run writes; the
        # verifier stops the process once it appears.
        commit_command += [
            "--save_state",
            "--save_every_n_steps",
            str(spec.stop_step),
        ]
    phases = [
        {
            "phase": "commit",
            "command": commit_command,
            # A short probe commits exactly ``stop_at_step``; the full run commits
            # ``steps`` (they are equal unless the probe asked for less).
            "expected_committed_step": int(spec.stop_step if spec.resume_steps else spec.steps),
            "stop_when_state_dir": (
                str(resume_state) if spec.resume_steps else None
            ),
            "saves_state": bool(spec.resume_steps),
        }
    ]
    if spec.resume_steps:
        phases.append(
            {
                "phase": "resume",
                "resume_from": str(resume_state),
                "expected_committed_step": spec.steps,
                # Real CLI: ``--resume`` on a real state dir, same horizon.
                "command": list(base) + ["--resume", str(resume_state)],
                "saves_state": False,
            }
        )
    facts = {
        "scheduler_horizon_steps": horizon,
        "phases": phases,
        "seed": SEED,
        "lr_scheduler": "constant",
        "output_name": RUN_OUTPUT_NAME,
        "world_size": spec.world_size,
        "steps": spec.steps,
        "resume_steps": spec.resume_steps,
        "budgeted_updates": spec.budgeted_updates,
        # Only the keys that apply: a single worker run has no
        # ANIMA_ACCELERATE_NUM_PROCESSES at all (``None`` in a Popen env is a
        # TypeError, not "unset").
        "launch_env": clean_launch_env(
            {
                "ANIMA_ACCELERATE_LAUNCH": launch_env.get("ANIMA_ACCELERATE_LAUNCH"),
                "ANIMA_ACCELERATE_NUM_PROCESSES": launch_env.get(
                    "ANIMA_ACCELERATE_NUM_PROCESSES"
                ),

            }
        ),
        "interpreter": interpreter,
        # Cumulative accounting is the caller's ``prior_updates`` + this run's
        # charge; the constant above only caps a single batch of runs.
        "prior_updates": int(prior_updates),
        "updates_after_this_run": int(prior_updates) + spec.budgeted_updates,
        "stop_at_step": int(spec.stop_at_step or spec.steps),
        "minutes_per_run_cap": MINUTES_PER_RUN,
        "minutes_total_cap": MINUTES_TOTAL,
        "pinned_order": bool(pinned_order),
        "seedable_sampler": True,
        "s0_permutation_shared_across_world_sizes": True,
        "reads_only_model_and_cache": True,
        "no_download": True,
        "no_preprocessing": True,
    }
    if pinned_order:
        # The measurement gate travels with the plan, exactly like the noise fixture
        # below: a run whose facts say "pinned" but whose env does not carry it would
        # silently compare different data (V40-F3).
        facts["launch_env"][pinned_order_env()] = "1"
    fixture = inspect_l3_fixture(l3_fixture) if spec.needs_l3_fixture else {"ok": True}
    if spec.needs_l3_fixture and not fixture["ok"]:
        # Path 3's fixed 4096-token cache is an L1/L2 candidate only; the numeric
        # gate needs per-sample noise/timestep fixtures. Report the block instead
        # of producing a toy green.
        return HotRunPlan(
            spec=spec,
            run_id=resolved_id,
            output_root=out,
            command=command,
            status="blocked",
            code=fixture["code"],
            detail=(
                "L3 per-sample noise/timestep fixture unusable ("
                + str(fixture["detail"])
                + "): RUN-A/B/C numeric gates are not satisfiable, and a fixed "
                "4096-token asset set is only an L1/L2 candidate (not a RUN unlock)"
            ),
            facts=facts,
        )
    if spec.needs_l3_fixture:
        facts["l3_fixture"] = str(l3_fixture)
        facts["l3_samples"] = fixture.get("samples")
        # The fixture only matters if training consumes it: pass it through the env
        # var that ``library/training/multigpu/noise_fixture.py`` reads, so the gate
        # and the run can never diverge.
        from library.training.multigpu import noise_fixture

        facts["launch_env"][noise_fixture.ENV] = str(l3_fixture)
        # A *real* training batch carries no per-sample id: ``image_keys`` is only
        # populated in the do-not-train ``--debug_dataset`` mode. Deriving the ids
        # from the committed sampler position is the only mode that actually
        # engages the fixture in a bounded real run, so the run entry pins it.
        facts["launch_env"][noise_fixture.ENV_IDS] = noise_fixture.IDS_CURSOR
        facts["l3_ids_mode"] = noise_fixture.IDS_CURSOR
    return HotRunPlan(
        spec=spec,
        run_id=resolved_id,
        output_root=out,
        command=command,
        status="ready",
        facts=facts,
    )


def execute(plan: HotRunPlan, *, runner=None, base_env=None) -> dict:
    """Run the planned command through ``runner`` (a subprocess-compatible callable).

    Real execution is still gated: without a ``runner`` (mock), nothing is spawned.
    A verifier passes e.g. ``runner=lambda cmd, **kw: subprocess.run(cmd, **kw)``
    after the GPU task released the run; the mock path proves argv/env shape on CPU
    without launching the trainer.

    The ``env`` handed to a real runner is this process's environment (or
    ``base_env``) with the launch keys layered on top, because the launch keys are
    additions and a child started with *only* them would lose ``PATH`` and friends.
    Entries are cleaned first: ``Popen`` rejects a ``None`` value, and a key that
    does not apply is absent rather than ``None`` (t13 F-T13-3b).
    """
    if plan.status != "ready":
        plan.raise_if_not_ready()
    launch_env = clean_launch_env(plan.facts.get("launch_env") or {})
    # Refuse to hand over an env whose gate state disagrees with the plan (a hand
    # export outside the plan is exactly the t35 hole, V40-F3 / t44).
    handed_env = {**dict(os.environ if base_env is None else base_env), **launch_env}
    verify_pinned_gate(plan, handed_env)
    results: list[dict] = []
    for phase in plan.facts.get("phases") or [
        {"phase": "single", "command": plan.command}
    ]:
        command = phase["command"]
        stop_when = phase.get("stop_when_state_dir")
        if runner is None:
            results.append(
                {
                    "phase": phase["phase"],
                    "command": command,
                    "ran": False,
                    # What a real runner would have been given, so a mock run can
                    # assert the env shape without spawning anything.
                    "env_keys": sorted(launch_env),
                    "pinned_order": launch_env.get(pinned_order_env()),
                }
            )
            continue
        kwargs = {"env": dict(handed_env)}
        if stop_when:
            # A real runner polls for this directory and terminates the run, which is
            # how RUN-C stops after its committed 10th update while keeping the same
            # 20-step scheduler horizon.
            kwargs["stop_when_state_dir"] = stop_when
        completed = runner(command, **kwargs)
        results.append(
            {
                "phase": phase["phase"],
                "command": command,
                "ran": True,
                "stop_when_state_dir": kwargs.get("stop_when_state_dir"),
                "pinned_order": launch_env.get(pinned_order_env()),
                "returncode": getattr(completed, "returncode", None),
            }
        )
    return {"run_id": plan.run_id, "output_root": str(plan.output_root), "phases": results}
