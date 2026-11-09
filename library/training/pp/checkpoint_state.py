"""Per-rank PP training state: content hashing, snapshots, comparison.

One PP rank's training state is a small, closed set of facts:

* the **stage-local LoRA adapter weights** -- only the adapters attached to the
  blocks this rank owns (``stage_module.adapter_modules`` is the ownership truth,
  ``network.unet_loras`` is the whole model's adapter set);
* this rank's **optimizer state**;
* the **fp16 state**: the ``GradScaler`` on the last stage, plus the
  pipeline-wide scale / growth tracker / last skip decision on every stage;
* the **step counter**;
* the **PP topology** the artifact belongs to (stages, block range, plan
  identity, schedule, conditioning, microbatch count, latent shape);
* the **run identity**: content hashes of the artifacts this run is made of.

Content hashes, never paths
---------------------------
The experiment fingerprint this replaces hashed ``mtime_ns`` plus
``str(path.resolve())``: unstable (a copied file "changed") and unusable in a
library.  Every identity value here is a **sha256 of content** --
:func:`content_hash_file` for an artifact on disk, :func:`content_hash_tensors`
for in-memory weights, :func:`content_hash_object` for structured state -- and
:func:`validate_identity` refuses anything that is not a sha256 hex digest, so an
mtime/absolute-path fingerprint cannot be reintroduced by accident.

Comparison is the acceptance claim
----------------------------------
``scripts/experiments/anima_parallel/benchmark.py::verify_resume`` proves a
resume with "save -> step -> load -> step" and then demands
``adapter max_abs == 0`` and ``loss_delta == 0``.
:func:`compare_snapshots` / :func:`assert_snapshot_roundtrip` are that assertion
expressed on the checkpointed state (``torch.equal`` for tensors, ``==`` for
scalars), so the same claim can be made without re-running a step.

The save/load protocol (run root, manifest, cross-rank negotiation) lives in
:mod:`library.training.pp.checkpoint`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .errors import (
    CODE_CHECKPOINT_IDENTITY_MISMATCH,
    CODE_CHECKPOINT_STATE_MISMATCH,
    CODE_CHECKPOINT_STEP_MISMATCH,
    PipelineParallelError,
)

#: Payload/manifest schema name; a foreign artifact is refused, not read.
PP_CHECKPOINT_FORMAT = "anima-pp-stage-checkpoint"
#: Bumped whenever the payload/manifest meaning changes (append-only history).
PP_CHECKPOINT_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEY_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$")


# ---------------------------------------------------------------------------
# content hashing (the only accepted fingerprint vocabulary)
# ---------------------------------------------------------------------------


def _tensor_bytes(tensor: Tensor) -> bytes:
    """Raw little-endian bytes of a tensor, independent of storage layout."""

    contiguous = tensor.detach().to("cpu").contiguous()
    if contiguous.numel() == 0:
        return b""
    return contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()


def content_hash_tensor(tensor: Tensor) -> str:
    """sha256 over ``shape``, ``dtype`` and the tensor's bytes.

    The header is hashed too: two tensors with identical bytes but different
    shapes or dtypes are different state, and a bare bytes-hash would call them
    equal.
    """

    if not isinstance(tensor, Tensor):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"content_hash_tensor expects a tensor, got {type(tensor).__name__}",
        )
    digest = hashlib.sha256()
    digest.update(f"{tuple(tensor.shape)}|{tensor.dtype}|".encode())
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def content_hash_tensors(tensors: Mapping[str, Tensor]) -> str:
    """sha256 over a name -> tensor mapping, in sorted-name order."""

    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(str(name).encode())
        digest.update(b"\x00")
        digest.update(content_hash_tensor(tensors[name]).encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def content_hash_file(path: str | Path) -> str:
    """sha256 of a file's bytes.  The path itself is never recorded."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_state(value: Any) -> Any:
    """JSON-friendly, exact rendering of a state dict (tensors become hashes)."""

    if isinstance(value, Tensor):
        return {"tensor": content_hash_tensor(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_state(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_state(item) for item in value]
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": value}
    if isinstance(value, float):
        # ``float.hex`` is exact: 0.1 stays 0x1.999999999999ap-4, so a digest can
        # never call two different scales equal through repr rounding.
        return {"float": float.hex(value)}
    if value is None:
        return None
    if isinstance(value, str):
        return {"str": value}
    return {"repr": repr(value)}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_state(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def content_hash_object(value: Any) -> str:
    """sha256 of a structured state (optimizer state dict, scaler state, ...)."""

    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def topology_digest(topology: Mapping[str, Any]) -> str:
    """sha256 of the PP topology record (ranges/stages/plan identity)."""

    return content_hash_object(dict(topology))


def validate_identity(identity: Mapping[str, str] | None) -> dict[str, str]:
    """Normalize a run identity: logical name -> sha256 of content.

    Rejects anything that is not a content hash, which is what keeps an
    ``mtime``/absolute-path fingerprint (the defect this replaces) from coming
    back: a path is neither a 64-hex digest nor a valid logical name.
    """

    if identity is None:
        return {}
    if not isinstance(identity, Mapping):
        raise PipelineParallelError(
            CODE_CHECKPOINT_IDENTITY_MISMATCH,
            "identity must be a mapping of logical name -> sha256 content hash, got "
            f"{type(identity).__name__}",
        )
    normalized: dict[str, str] = {}
    for raw_name, raw_value in identity.items():
        name = str(raw_name)
        if _IDENTITY_KEY_RE.fullmatch(name) is None:
            raise PipelineParallelError(
                CODE_CHECKPOINT_IDENTITY_MISMATCH,
                f"identity key {raw_name!r} is not a logical name; PP checkpoints "
                "record content hashes under short logical names, never paths",
            )
        if not isinstance(raw_value, str) or _SHA256_RE.fullmatch(raw_value) is None:
            raise PipelineParallelError(
                CODE_CHECKPOINT_IDENTITY_MISMATCH,
                f"identity[{name!r}] must be a sha256 hex digest of content, got "
                f"{raw_value!r}; mtime/size/absolute-path fingerprints are not "
                "accepted (use content_hash_file / content_hash_tensors)",
            )
        normalized[name] = raw_value
    return dict(sorted(normalized.items()))


# ---------------------------------------------------------------------------
# stage-local state
# ---------------------------------------------------------------------------


def _adapter_name(module: Any) -> str:
    name = getattr(module, "lora_name", None)
    if not isinstance(name, str) or not name:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"adapter module {type(module).__name__} has no lora_name; a stage-local "
            "checkpoint keys its adapter state by that name",
        )
    return name


def stage_adapter_modules(stage_module: Any) -> dict[str, Any]:
    """This stage's adapter modules, keyed by their global ``lora_name``."""

    modules: dict[str, Any] = {}
    for module in getattr(stage_module, "adapter_modules", ()):
        name = _adapter_name(module)
        if name in modules:
            raise PipelineParallelError(
                CODE_CHECKPOINT_STATE_MISMATCH,
                f"two adapter modules of this stage share the name {name!r}; the "
                "stage-local checkpoint could not tell them apart",
            )
        modules[name] = module
    return modules


def stage_adapter_state(stage_module: Any) -> dict[str, dict[str, Tensor]]:
    """Stage-local adapter weights: ``{lora_name: state_dict}``.

    Nested on purpose.  A flat ``"<lora_name>.<key>"`` namespace (as
    ``scripts/experiments/anima_parallel/common.py::lora_state_for_blocks``
    writes) is ambiguous whenever one adapter name is a prefix of another, and
    LoRA names such as ``lora_unet_blocks_1_mlp.0`` / ``..._mlp.1`` make that the
    normal case.
    """

    state: dict[str, dict[str, Tensor]] = {}
    for name, module in sorted(stage_adapter_modules(stage_module).items()):
        state[name] = {
            key: value.detach().to("cpu").clone()
            for key, value in module.state_dict().items()
        }
    return state


def _assert_stage_state_is_complete(stage_module: Any) -> None:
    """Refuse a stage whose trainable state the adapter state dict does not cover.

    Method adapters (soft tokens and friends) register trainable parameters inside
    the block they patch, so they are *not* part of an adapter module.  A
    checkpoint that silently dropped them would resume into a different model, so
    PP v0 fails closed instead.
    """

    covered = {
        id(parameter)
        for module in getattr(stage_module, "adapter_modules", ())
        for parameter in module.parameters()
    }
    uncovered = [
        parameter
        for parameter in stage_module.trainable_parameters()
        if id(parameter) not in covered
    ]
    if uncovered:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"this stage owns {len(uncovered)} trainable parameter(s) outside its "
            "adapter modules (method adapters register inside blocks); PP v0 "
            "checkpoints stage-local LoRA adapters only, so these weights would be "
            "lost on resume",
        )


def _cpu_clone_state(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone_state(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------


def _validate_step(step: Any) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise PipelineParallelError(
            CODE_CHECKPOINT_STEP_MISMATCH,
            f"step must be an int, got {type(step).__name__} ({step!r})",
        )
    if step < 0:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STEP_MISMATCH, f"step must be non-negative, got {step}"
        )
    return int(step)


@dataclass(frozen=True)
class PpStageSnapshot:
    """One rank's complete PP training state, ready to be written or compared."""

    step: int
    topology: dict[str, Any]
    identity: dict[str, str]
    adapters: dict[str, dict[str, Tensor]]
    optimizer: dict[str, Any]
    scaler: dict[str, Any] | None
    scale: float | None
    growth_tracker: int
    last_skip: bool | None

    def as_payload(self) -> dict[str, Any]:
        """The on-disk payload for this rank (``torch.save``-able)."""

        return {
            "format": PP_CHECKPOINT_FORMAT,
            "version": PP_CHECKPOINT_VERSION,
            "step": int(self.step),
            "topology": dict(self.topology),
            "identity": dict(self.identity),
            "adapters": {
                name: {key: value for key, value in state.items()}
                for name, state in self.adapters.items()
            },
            "optimizer": self.optimizer,
            "scaler": None if self.scaler is None else dict(self.scaler),
            "scale": None if self.scale is None else float(self.scale),
            "growth_tracker": int(self.growth_tracker),
            "last_skip": self.last_skip,
        }

    def digest(self) -> dict[str, Any]:
        return _state_digest(
            step=self.step,
            topology=self.topology,
            identity=self.identity,
            adapters=self.adapters,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scale=self.scale,
            growth_tracker=self.growth_tracker,
            last_skip=self.last_skip,
        )


def _state_digest(
    *,
    step: int,
    topology: Mapping[str, Any],
    identity: Mapping[str, str],
    adapters: Mapping[str, Mapping[str, Tensor]],
    optimizer: Any,
    scaler: Any,
    scale: float | None,
    growth_tracker: int,
    last_skip: bool | None,
) -> dict[str, Any]:
    """Comparable, JSON-friendly summary of a state (no tensors, no paths)."""

    adapter_tensors: dict[str, Tensor] = {}
    for name, state in adapters.items():
        for key, value in state.items():
            adapter_tensors[f"{name}.{key}"] = value
    return {
        "step": int(step),
        "topology_digest": topology_digest(topology),
        "identity": dict(sorted(identity.items())),
        "adapter_hash": content_hash_tensors(adapter_tensors),
        "optimizer_hash": content_hash_object(optimizer),
        "scaler_hash": None if scaler is None else content_hash_object(scaler),
        "scale": None if scale is None else float(scale),
        "growth_tracker": int(growth_tracker),
        "last_skip": last_skip,
    }


def capture_stage_snapshot(
    runtime: Any,
    *,
    step: int,
    identity: Mapping[str, str] | None = None,
) -> PpStageSnapshot:
    """Read this rank's complete PP training state off a live runtime."""

    optimizer = getattr(runtime, "optimizer", None)
    if optimizer is None:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            "this runtime owns no optimizer, so it has no per-stage training state "
            "to checkpoint",
        )
    stage_module = getattr(runtime, "stage_module", None)
    if stage_module is None:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            "this runtime exposes no stage_module, so the stage-local adapter state "
            "cannot be identified",
        )
    _assert_stage_state_is_complete(stage_module)
    transaction = getattr(runtime, "transaction", None)
    if transaction is None:
        scaler_state: dict[str, Any] | None = None
        scale: float | None = None
        growth_tracker = 0
        last_skip: bool | None = None
    else:
        scaler = getattr(transaction, "scaler", None)
        scaler_state = None if scaler is None else _cpu_clone_state(scaler.state_dict())
        scale = float(transaction.scale)
        growth_tracker = int(transaction.growth_tracker)
        last_skip = transaction.last_skip
    return PpStageSnapshot(
        step=_validate_step(step),
        topology=dict(runtime.topology()),
        identity=validate_identity(identity),
        adapters=stage_adapter_state(stage_module),
        optimizer=_cpu_clone_state(optimizer.state_dict()),
        scaler=scaler_state,
        scale=scale,
        growth_tracker=growth_tracker,
        last_skip=last_skip,
    )


# ---------------------------------------------------------------------------
# comparison (the ``benchmark.verify_resume`` assertion shape)
# ---------------------------------------------------------------------------


def _states_equal(left: Any, right: Any) -> bool:
    """``torch.equal`` for tensors, ``==`` for scalars, structural otherwise."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        if not (isinstance(left, Tensor) and isinstance(right, Tensor)):
            return False
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return False
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_states_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_states_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    return type(left) is type(right) and left == right


def _adapter_max_abs(
    left: Mapping[str, Mapping[str, Tensor]],
    right: Mapping[str, Mapping[str, Tensor]],
) -> float:
    if set(left) != set(right):
        return float("inf")
    worst = 0.0
    for name in left:
        if set(left[name]) != set(right[name]):
            return float("inf")
        for key, tensor in left[name].items():
            other = right[name][key]
            if tuple(tensor.shape) != tuple(other.shape):
                return float("inf")
            if tensor.numel() == 0:
                continue
            worst = max(
                worst, float((tensor.float() - other.float()).abs().max().item())
            )
    return worst


def compare_snapshots(
    expected: PpStageSnapshot, actual: PpStageSnapshot
) -> dict[str, Any]:
    """Compare two states and report *why* they differ, never just ``False``.

    ``adapter_max_abs`` / ``loss``-style zero-delta reporting is the shape
    ``scripts/experiments/anima_parallel/benchmark.py::verify_resume`` asserts on:
    a resume is only proven when the adapter difference and the step/loss deltas
    are exactly zero.
    """

    adapter_max_abs = _adapter_max_abs(expected.adapters, actual.adapters)
    scale_delta: float | None
    if expected.scale is None or actual.scale is None:
        scale_delta = None
    else:
        scale_delta = abs(float(expected.scale) - float(actual.scale))
    report = {
        "adapter_max_abs": adapter_max_abs,
        "adapter_equal": adapter_max_abs == 0.0,
        "optimizer_equal": _states_equal(expected.optimizer, actual.optimizer),
        "scaler_equal": _states_equal(expected.scaler, actual.scaler),
        "scale_delta": scale_delta,
        "scale_equal": expected.scale == actual.scale,
        "growth_tracker_delta": int(actual.growth_tracker)
        - int(expected.growth_tracker),
        "step_delta": int(actual.step) - int(expected.step),
        "last_skip_equal": expected.last_skip == actual.last_skip,
        "topology_equal": dict(expected.topology) == dict(actual.topology),
        "identity_equal": dict(expected.identity) == dict(actual.identity),
    }
    report["equal"] = bool(
        report["adapter_equal"]
        and report["optimizer_equal"]
        and report["scaler_equal"]
        and report["scale_equal"]
        and report["growth_tracker_delta"] == 0
        and report["step_delta"] == 0
        and report["last_skip_equal"]
        and report["topology_equal"]
        and report["identity_equal"]
    )
    return report


def assert_snapshot_roundtrip(
    expected: PpStageSnapshot,
    actual: PpStageSnapshot,
    *,
    where: str = "pp_checkpoint_roundtrip",
) -> dict[str, Any]:
    """Raise a named error unless the two states are bit-identical."""

    report = compare_snapshots(expected, actual)
    if report["equal"]:
        return report
    problems = [
        f"adapter_max_abs={report['adapter_max_abs']!r}",
        f"optimizer_equal={report['optimizer_equal']}",
        f"scaler_equal={report['scaler_equal']}",
        f"scale_delta={report['scale_delta']!r}",
        f"growth_tracker_delta={report['growth_tracker_delta']}",
        f"step_delta={report['step_delta']}",
        f"last_skip_equal={report['last_skip_equal']}",
        f"topology_equal={report['topology_equal']}",
        f"identity_equal={report['identity_equal']}",
    ]
    raise PipelineParallelError(
        CODE_CHECKPOINT_STATE_MISMATCH,
        f"PP checkpoint roundtrip is not bit-exact at {where}: " + ", ".join(problems),
    )


def assert_snapshot_digests_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    where: str = "pp_checkpoint_roundtrip",
) -> None:
    """Compare two :meth:`PpStageSnapshot.digest` results (e.g. across processes)."""

    differences = [
        f"{key}: expected={expected.get(key)!r} actual={actual.get(key)!r}"
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]
    if differences:
        raise PipelineParallelError(
            CODE_CHECKPOINT_STATE_MISMATCH,
            f"PP checkpoint digests differ at {where}: " + "; ".join(differences),
        )


__all__ = [
    "PP_CHECKPOINT_FORMAT",
    "PP_CHECKPOINT_VERSION",
    "PpStageSnapshot",
    "assert_snapshot_digests_match",
    "assert_snapshot_roundtrip",
    "capture_stage_snapshot",
    "compare_snapshots",
    "content_hash_file",
    "content_hash_object",
    "content_hash_tensor",
    "content_hash_tensors",
    "stage_adapter_modules",
    "stage_adapter_state",
    "topology_digest",
    "validate_identity",
]
