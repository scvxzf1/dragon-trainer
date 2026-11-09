"""PP checkpoint artifacts: run root, manifest schema, payload read/write.

The protocol (who saves when, and how the ranks agree) is in
:mod:`library.training.pp.checkpoint`; the state itself, its content hashing and
its comparison are in :mod:`library.training.pp.checkpoint_state`.  This module is
the file layer between them:

* the **run root** is created once and never reused (:func:`create_run_root`), so
  a save can never land on an existing checkpoint or training output;
* one **artifact per rank** (:func:`save_stage_checkpoint` /
  :func:`read_stage_checkpoint`), written atomically through a ``.tmp`` file;
* one **manifest** (:func:`read_manifest`) that binds the artifacts together:
  world size, stage count, step, run identity, the full pipeline plan, each rank's
  topology/scale/skip decision, and the sha256 of every artifact;
* every **named refusal**: a changed topology or plan, a changed rank count, a
  content hash that no longer matches, a missing artifact, a state that does not
  fit this stage.  Nothing here is applied to a runtime -- that is
  :func:`library.training.pp.checkpoint.apply_stage_checkpoint`, called only after
  the group agreed.

Why the plan is part of the manifest
------------------------------------
``runtime.topology()`` only carries *this* rank's block range, so a plan change
that happens to leave one rank's slice untouched would otherwise pass.  The
manifest therefore records the whole range list, recomputed from
family/stage/block counts, and the load path recomputes it again and compares.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from .checkpoint_state import (
    PP_CHECKPOINT_FORMAT,
    PP_CHECKPOINT_VERSION,
    PpStageSnapshot,
    _validate_step,
    content_hash_file,
    stage_adapter_modules,
    validate_identity,
)
from .errors import (
    CODE_CHECKPOINT_IDENTITY_MISMATCH,
    CODE_CHECKPOINT_INCOMPLETE,
    CODE_CHECKPOINT_RUN_ROOT_NOT_NEW,
    CODE_CHECKPOINT_STATE_MISMATCH,
    CODE_CHECKPOINT_TOPOLOGY_MISMATCH,
    PipelineParallelError,
)

#: The single manifest that binds the per-rank artifacts of one run root.
MANIFEST_FILENAME = "pp_checkpoint.json"
#: One stage artifact per rank, inside the run root.
STAGE_FILENAME_TEMPLATE = "pp_stage-rank{rank}.pt"
#: Written instead of a manifest when the group refused a save.
FAILED_MARKER_FILENAME = "pp_checkpoint.FAILED.json"


def manifest_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_FILENAME

def stage_filename(rank: int) -> str:
    return STAGE_FILENAME_TEMPLATE.format(rank=int(rank))

def stage_path(root: str | Path, rank: int) -> Path:
    return Path(root) / stage_filename(rank)

def _distributed(group: Any = None) -> bool:
    return bool(dist.is_available() and dist.is_initialized())

def _resolve_rank(rank: int | None, group: Any = None) -> int:
    if rank is not None:
        return int(rank)
    return int(dist.get_rank(group)) if _distributed(group) else 0

def _resolve_world_size(world_size: int | None, group: Any = None) -> int:
    if world_size is not None:
        return int(world_size)
    return int(dist.get_world_size(group)) if _distributed(group) else 1

def create_run_root(root: str | Path) -> Path:
    """Create the run root for this checkpoint, refusing any existing path.

    A PP checkpoint must never land on top of an existing checkpoint or training
    output, so "the directory already exists" is a refusal even when it is empty:
    reusing a run root is how a half-written save silently becomes a resume.
    """

    path = Path(root)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PipelineParallelError(
            CODE_CHECKPOINT_RUN_ROOT_NOT_NEW,
            f"{path} already exists; a PP checkpoint is only written into a newly "
            "created run root so an existing checkpoint or training output is "
            "never overwritten",
        ) from exc
    except OSError as exc:
        raise PipelineParallelError(
            CODE_CHECKPOINT_RUN_ROOT_NOT_NEW,
            f"cannot create the PP checkpoint run root {path}: {exc}",
        ) from exc
    return path

def save_stage_checkpoint(
    root: str | Path,
    snapshot: PpStageSnapshot,
    *,
    rank: int | None = None,
) -> Path:
    """Write this rank's payload into an already created run root."""

    path_root = Path(root)
    if not path_root.is_dir():
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"run root {path_root} does not exist; create it with create_run_root() "
            "before any rank writes its stage file",
        )
    path = stage_path(path_root, _resolve_rank(rank))
    if path.exists():
        raise PipelineParallelError(
            CODE_CHECKPOINT_RUN_ROOT_NOT_NEW,
            f"{path} already exists; refusing to overwrite an existing stage "
            "artifact",
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(snapshot.as_payload(), temporary)
    os.replace(temporary, path)
    return path

def _write_manifest(root: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = manifest_path(root)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path

def _mark_failed(root: str | Path, detail: str) -> None:
    """Best-effort marker so a refused run root is never mistaken for a save."""

    try:
        path = Path(root) / FAILED_MARKER_FILENAME
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"error": detail}, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError:  # pragma: no cover - disk level, the raise already happened
        pass

def _plan_record(
    first_topology: Mapping[str, Any], ranks: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """The authoritative plan, cross-checked against every rank's own report.

    The whole range list is recorded, not just the slice of one stage: a plan
    change that happens to leave this rank's range untouched must still refuse.
    The ranges are recomputed from the family/stage/block counts rather than
    assembled from the reports, so a manifest cannot describe a plan that no
    runtime would build -- and every rank's reported range is verified against it.
    """

    from library.models.pipeline_parallel import make_pipeline_plan

    plan = make_pipeline_plan(
        family=str(first_topology["family"]),
        stages=int(first_topology["stages"]),
        num_blocks=int(first_topology["num_blocks"]),
    )
    ranges = [list(range_pair) for range_pair in plan.ranges]
    for entry in ranks:
        topology = dict(entry["topology"])
        stage_index = int(topology["stage_index"])
        block_range = list(topology["block_range"])
        if stage_index >= len(ranges) or block_range != ranges[stage_index]:
            raise PipelineParallelError(
                CODE_CHECKPOINT_TOPOLOGY_MISMATCH,
                f"rank {entry['rank']} reports block_range {block_range} for stage "
                f"{stage_index}, but the plan for {plan.family} with "
                f"{plan.stages} stage(s) over {plan.num_blocks} blocks owns "
                f"{ranges[stage_index] if stage_index < len(ranges) else None}",
            )
    return {
        "family": plan.family,
        "stages": int(plan.stages),
        "num_blocks": int(plan.num_blocks),
        "ranges": ranges,
    }

def read_manifest(root: str | Path) -> dict[str, Any]:
    """Read and schema-check the manifest of a PP run root."""

    path = manifest_path(root)
    if not path.is_file():
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"{Path(root)} has no {MANIFEST_FILENAME}; the run was never completed "
            "(a save publishes its manifest only after every rank succeeded)",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE, f"unreadable PP manifest {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE, f"{path} is not a JSON object"
        )
    if data.get("format") != PP_CHECKPOINT_FORMAT:
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"{path} is not an Anima PP stage checkpoint (format={data.get('format')!r})",
        )
    if data.get("version") != PP_CHECKPOINT_VERSION:
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"PP checkpoint version {data.get('version')!r} != supported "
            f"{PP_CHECKPOINT_VERSION}",
        )
    return data

def _rank_entry(manifest: Mapping[str, Any], rank: int) -> dict[str, Any]:
    for entry in manifest.get("ranks") or ():
        if int(entry.get("rank", -1)) == int(rank):
            return dict(entry)
    raise PipelineParallelError(
        CODE_CHECKPOINT_INCOMPLETE,
        f"the manifest has no artifact for rank {rank} (it lists "
        f"{[entry.get('rank') for entry in manifest.get('ranks') or ()]}); this rank "
        "was not part of the saved topology",
    )

def _topology_differences(
    saved: Mapping[str, Any], live: Mapping[str, Any]
) -> list[str]:
    return [
        f"{key}: saved={saved.get(key)!r} live={live.get(key)!r}"
        for key in sorted(set(saved) | set(live))
        if saved.get(key) != live.get(key)
    ]

def _validate_topology(
    saved: Mapping[str, Any], live: Mapping[str, Any], *, where: str
) -> None:
    differences = _topology_differences(saved, live)
    if not differences:
        return
    raise PipelineParallelError(
        CODE_CHECKPOINT_TOPOLOGY_MISMATCH,
        f"{where} does not match this run: " + "; ".join(differences),
    )

def _validate_plan(
    manifest: Mapping[str, Any], live_topology: Mapping[str, Any], *, where: str
) -> None:
    """Recompute the plan this run would produce and compare it with the saved one.

    ``runtime.topology()`` only carries *this* rank's range, so a plan change that
    happens to leave this rank's slice untouched would otherwise slip through.
    """

    saved = manifest.get("plan")
    if not isinstance(saved, Mapping):
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"{where} manifest carries no pipeline plan (the block ownership of "
            "every stage is part of the checkpoint contract)",
        )
    from library.models.pipeline_parallel import make_pipeline_plan

    try:
        plan = make_pipeline_plan(
            family=str(live_topology["family"]),
            stages=int(live_topology["stages"]),
            num_blocks=int(live_topology["num_blocks"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineParallelError(
            CODE_CHECKPOINT_TOPOLOGY_MISMATCH,
            f"{where}: this run's topology cannot produce a pipeline plan: {exc}",
        ) from exc
    live = {
        "family": plan.family,
        "stages": int(plan.stages),
        "num_blocks": int(plan.num_blocks),
        "ranges": [list(range_pair) for range_pair in plan.ranges],
    }
    if dict(saved) != live:
        raise PipelineParallelError(
            CODE_CHECKPOINT_TOPOLOGY_MISMATCH,
            f"{where} pipeline plan does not match this run: saved={dict(saved)}, "
            f"live={live}; the block ownership of at least one stage changed",
        )

def _validate_adapters(
    saved: Mapping[str, Mapping[str, Tensor]], stage_module: Any
) -> dict[str, Any]:
    owned = stage_adapter_modules(stage_module)
    missing = sorted(set(owned) - set(saved))
    unexpected = sorted(set(saved) - set(owned))
    if missing or unexpected:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            "the checkpoint's adapter set does not match this stage's block range "
            f"(missing={missing}, unexpected={unexpected}); a stage may only resume "
            "the adapters attached to the blocks it owns",
        )
    for name, module in owned.items():
        expected = module.state_dict()
        actual = saved[name]
        if set(expected) != set(actual):
            raise PipelineParallelError(
                CODE_CHECKPOINT_STATE_MISMATCH,
                f"adapter {name!r} state keys differ (missing="
                f"{sorted(set(expected) - set(actual))}, unexpected="
                f"{sorted(set(actual) - set(expected))})",
            )
        for key, tensor in expected.items():
            value = actual[key]
            if not isinstance(value, Tensor):
                raise PipelineParallelError(
                    CODE_CHECKPOINT_STATE_MISMATCH,
                    f"adapter {name!r}[{key!r}] is not a tensor ({type(value).__name__})",
                )
            if tuple(value.shape) != tuple(tensor.shape) or value.dtype != tensor.dtype:
                raise PipelineParallelError(
                    CODE_CHECKPOINT_STATE_MISMATCH,
                    f"adapter {name!r}[{key!r}] is {tuple(value.shape)}/{value.dtype}, "
                    f"this stage needs {tuple(tensor.shape)}/{tensor.dtype}",
                )
    return owned


@dataclass(frozen=True)

class PpResumeState:
    """A validated per-rank checkpoint, not yet applied to the runtime."""

    root: Path
    rank: int
    step: int
    topology: dict[str, Any]
    identity: dict[str, str]
    adapters: dict[str, dict[str, Tensor]]
    optimizer: dict[str, Any]
    scaler: dict[str, Any] | None
    scale: float | None
    growth_tracker: int
    last_skip: bool | None
    manifest: dict[str, Any]
    path: Path

    def snapshot(self) -> PpStageSnapshot:
        return PpStageSnapshot(
            step=self.step,
            topology=dict(self.topology),
            identity=dict(self.identity),
            adapters={
                name: dict(state) for name, state in self.adapters.items()
            },
            optimizer=self.optimizer,
            scaler=None if self.scaler is None else dict(self.scaler),
            scale=self.scale,
            growth_tracker=self.growth_tracker,
            last_skip=self.last_skip,
        )

    def digest(self) -> dict[str, Any]:
        return self.snapshot().digest()

def _resolve_artifact(
    root_path: Path,
    manifest: Mapping[str, Any],
    *,
    rank: int | None,
    world_size: int | None,
    group: Any,
) -> tuple[int, dict[str, Any], Path]:
    """This rank's manifest entry plus its artifact, verified by content hash."""

    saved_world_size = int(manifest.get("world_size", 0) or 0)
    live_world_size = _resolve_world_size(world_size, group)
    if saved_world_size != live_world_size:
        raise PipelineParallelError(
            CODE_CHECKPOINT_TOPOLOGY_MISMATCH,
            f"the checkpoint was written by {saved_world_size} rank(s) but this run "
            f"has {live_world_size}; a per-rank artifact cannot be re-placed across "
            "a different rank count",
        )
    live_rank = _resolve_rank(rank, group)
    entry = _rank_entry(manifest, live_rank)
    path = root_path / str(entry["file"])
    if not path.is_file():
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"rank {live_rank} artifact {path} is missing from the run root",
        )
    actual_sha = content_hash_file(path)
    if actual_sha != entry.get("sha256"):
        raise PipelineParallelError(
            CODE_CHECKPOINT_IDENTITY_MISMATCH,
            f"rank {live_rank} artifact {path.name} has content hash {actual_sha} but "
            f"the manifest recorded {entry.get('sha256')}; the file was modified or "
            "replaced after the save",
        )
    return live_rank, entry, path

def _read_payload(path: Path) -> dict[str, Any]:
    """``torch.load`` plus the payload schema checks (no state is applied)."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE, f"unreadable PP stage artifact {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("format") != PP_CHECKPOINT_FORMAT:
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"{path} is not an Anima PP stage artifact (format="
            f"{payload.get('format') if isinstance(payload, dict) else None!r})",
        )
    if payload.get("version") != PP_CHECKPOINT_VERSION:
        raise PipelineParallelError(
            CODE_CHECKPOINT_INCOMPLETE,
            f"{path} has version {payload.get('version')!r} != supported "
            f"{PP_CHECKPOINT_VERSION}",
        )
    return payload

def read_stage_checkpoint(
    root: str | Path,
    runtime: Any,
    *,
    identity: Mapping[str, str] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    group: Any = None,
) -> PpResumeState:
    """Validate this rank's artifact against the live runtime.  Pure read.

    Every rejection is a named ``pp_checkpoint_*`` error: a topology that changed
    (block range, stage count, rank count, schedule, microbatch count, latent
    shape), a content hash that no longer matches the manifest, a step or state
    that does not fit this stage.  Nothing is applied here, so the caller can
    negotiate across ranks before any rank mutates its model.
    """

    root_path = Path(root)
    manifest = read_manifest(root_path)
    live_rank, entry, path = _resolve_artifact(
        root_path, manifest, rank=rank, world_size=world_size, group=group
    )
    live_topology = dict(runtime.topology())
    _validate_topology(
        dict(entry.get("topology") or {}),
        live_topology,
        where=f"rank {live_rank} saved topology",
    )
    _validate_plan(manifest, live_topology, where=f"rank {live_rank} saved topology")
    saved_identity = dict(manifest.get("identity") or {})
    requested_identity = validate_identity(identity)
    if requested_identity and requested_identity != saved_identity:
        raise PipelineParallelError(
            CODE_CHECKPOINT_IDENTITY_MISMATCH,
            "this run's identity does not match the checkpoint's "
            f"(saved={saved_identity}, live={requested_identity}); the checkpoint "
            "belongs to different content (base weights / dataset / config)",
        )
    payload = _read_payload(path)
    payload_identity = dict(payload.get("identity") or {})
    if payload_identity != saved_identity:
        raise PipelineParallelError(
            CODE_CHECKPOINT_IDENTITY_MISMATCH,
            f"rank {live_rank} artifact identity {payload_identity} disagrees with the "
            f"manifest's {saved_identity}",
        )
    _validate_topology(
        dict(payload.get("topology") or {}),
        live_topology,
        where=f"rank {live_rank} artifact topology",
    )
    adapters = {
        str(name): dict(state) for name, state in (payload.get("adapters") or {}).items()
    }
    _validate_adapters(adapters, runtime.stage_module)
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or not optimizer.get("param_groups"):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"rank {live_rank} artifact carries no optimizer state",
        )
    scaler = payload.get("scaler")
    if scaler is not None and not isinstance(scaler, Mapping):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"rank {live_rank} artifact carries a non-mapping scaler state",
        )
    scale = payload.get("scale")
    return PpResumeState(
        root=root_path,
        rank=live_rank,
        step=_validate_step(payload.get("step")),
        topology=live_topology,
        identity=saved_identity,
        adapters=adapters,
        optimizer=dict(optimizer),
        scaler=None if scaler is None else dict(scaler),
        scale=None if scale is None else float(scale),
        growth_tracker=int(payload.get("growth_tracker", 0) or 0),
        last_skip=payload.get("last_skip"),
        manifest=manifest,
        path=path,
    )

__all__ = [
    "FAILED_MARKER_FILENAME",
    "MANIFEST_FILENAME",
    "STAGE_FILENAME_TEMPLATE",
    "PpResumeState",
    "create_run_root",
    "manifest_path",
    "read_manifest",
    "read_stage_checkpoint",
    "save_stage_checkpoint",
    "stage_filename",
    "stage_path",
]
