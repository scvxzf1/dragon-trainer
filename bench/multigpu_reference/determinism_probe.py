"""Per-micro-batch diagnostics for the cross-shape determinism question (t24).

The field comparison S-d (single GPU, per-device batch 2) vs dual GPU (per-device
batch 1) diverges by ~1.2e-2 with the same magnitude in fp32 and fp16, so it is not
a precision story. t23 ruled out the whole data side with hard evidence (slot
mapping via the real ``accelerate`` shard batcher, the two frozen configs differing
only in ``batch_size``, the fixture really reaching the forward through
``sigma_override``, the resume cursor not replaying, and AdamW being scale
invariant). What is left is the **model forward / gradient layer**, and that can
only be judged by measuring it.

This module is that measurement. It is a pure diagnostic:

* **Off by default.** Nothing is installed unless ``ANIMA_MULTIGPU_DETERMINISM_PROBE``
  names an output path; with the variable unset the module is never imported by the
  trainer and every hook below is inert.
* **Zero production edits.** Everything is installed from the outside by
  monkeypatching three things that are looked up at call time —
  ``library.training.loop._run_microbatch`` (to know which micro-batch is running),
  ``LossComposer.compose`` + the stage-1 loss functions (to capture the per-sample
  and batch losses) and ``torch.optim.AdamW.step`` (to read gradients, AdamW
  moments and the adapter weights right where the optimizer consumes them).
  Nothing in ``library/`` changes, no math changes, no threshold or bucket table is
  touched, and :func:`uninstall` restores the originals.
* **Mechanically judged.** :func:`compare` returns one of four mutually exclusive
  verdicts with the first differing field and both values, so a reviewer does not
  read logs:

  ==========================  =====================================================
  verdict                     meaning
  ==========================  =====================================================
  ``init_divergence``         step-0 adapter hash differs: different initialisation
                              or seed, nothing about the data or the forward.
  ``forward_shape_sensitive`` same init, step-1 loss differs: the forward depends on
                              the batch *shape* (SDPA kernel choice, compile graph,
                              attention backend).
  ``backward_or_sync``        same loss, different gradient norm: the backward pass or
                              the cross-rank gradient sync diverges.
  ``input_divergence``        the pinned per-sample inputs themselves differ (slot id,
                              sigma, latent/text-cache fingerprint) — the comparison
                              is invalid before any training question is asked.
  ==========================  =====================================================

  ``compare(..., mode="resume")`` answers the other question: the continuous run and
  the resumed run at the same global step, field by field, first difference first.

Usage (CPU, no GPU needed)::

    python -m bench.multigpu_reference.determinism_probe segments
    python -m bench.multigpu_reference.determinism_probe compare --a run-a.jsonl --b run-b.jsonl
    python -m bench.multigpu_reference.determinism_probe compare --a cont.jsonl --b resumed.jsonl --mode resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

ENV = "ANIMA_MULTIGPU_DETERMINISM_PROBE"
#: Optional: directory of Anima latent caches (``*_anima.npz``). When set, every
#: per-sample latent is resolved back to the file it came from, so a record states
#: the *stem* instead of asking a reviewer to trust the slot formula.
ENV_LATENT_CACHE = "ANIMA_MULTIGPU_PROBE_LATENT_CACHE"

#: Where each recorded quantity is read, and whether that read can be disturbed by
#: an in-place write. Printed into every record (F-T27-1): a hash that cannot be
#: traced to a capture point cannot separate a real divergence from a probe
#: artefact.
CAPTURE_POINTS = {
    "noise": (
        "library/training/multigpu/noise_fixture.py:NoiseFixture.apply return value, "
        "hashed immediately at the injection point, i.e. BEFORE the sampler consumes "
        "it and before any noise overlay"
    ),
    "latent": (
        "library/training/loop.py:_run_microbatch batch['latents'], hashed at "
        "optimizer.step time, i.e. AFTER the forward/backward of that micro-batch"
    ),
    "text": (
        "same batch dict, encoder-output keys, hashed at optimizer.step time"
    ),
    "in_place_noise_injection": (
        "none: noise_target.py draws `noise = torch.randn_like(latents)` (out of "
        "place), noise_fixture.apply writes into `noise.clone()`, and the sampler "
        "returns a fresh `noisy_input`; no mul_/add_/copy_ touches latents or noise"
    ),
}
SCHEMA = "multigpu-determinism-probe-v1"

CODE_INIT = "init_divergence"
CODE_FORWARD = "forward_shape_sensitive"
CODE_BACKWARD = "backward_or_sync"
CODE_INPUT = "input_divergence"
CODE_AGREE = "no_divergence"

#: Order in which the resume comparison reports the first differing field: the
#: causal order of one training step (inputs -> forward -> backward -> optimizer).
RESUME_FIELD_ORDER = (
    "slots",
    "sigmas",
    "noise_source",
    "noise_value",
    "timestep_injected",
    "latent_value",
    "text_value",
    "noise_per_sample_hash",
    "latent_per_sample_hash",
    "text_per_sample_hash",
    "latent_fingerprint",
    "text_fingerprint",
    "per_sample_loss",
    "loss",
    "grad_norm_total",
    "exp_avg_norm",
    "exp_avg_sq_norm",
    "adapter_hash",
)

#: Records whose comparison needs a tolerance instead of exact equality.
FLOAT_FIELDS = (
    "per_sample_loss",
    "loss",
    "grad_norm_total",
    "exp_avg_norm",
    "exp_avg_sq_norm",
)


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def _to_cpu(value: Any) -> Any:
    """Host copy of a tensor, whatever device it lives on (F-T25-1)."""
    import torch

    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        return value.detach().to("cpu")
    return value


#: Batch keys that carry the text-encoder conditioning.
_TEXT_MARKERS = ("te", "text", "crossattn", "prompt", "t5", "caption", "encoder")
#: ...but a key containing any of these is *not* text, even when it matches a
#: marker as a substring. ``"latents"`` contains ``"te"``, and an earlier version
#: therefore folded the latent into the text fingerprint — which made "the text
#: differs" unfalsifiable in the t27 record.
_NOT_TEXT_MARKERS = ("latent", "noise", "mask", "image", "cond_", "ip_", "vae")


def is_text_key(key: str) -> bool:
    """Whether a batch key holds text-encoder conditioning."""
    lowered = str(key).lower()
    if any(marker in lowered for marker in _NOT_TEXT_MARKERS):
        return False
    return any(marker in lowered for marker in _TEXT_MARKERS)


def _raw_summary(value: Any) -> Optional[dict[str, Any]]:
    """Identity of one tensor a reviewer can read: head, sum, dtype, shape.

    F-T27-1: the t27 run recorded only hashes, so when a hash matched nothing on
    disk there was no way to ask "then what tensor is it?". These fields answer it
    without a debugger.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        return None
    tensor = _to_cpu(value).detach()
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return {"head8": [], "sum": 0.0, "dtype": str(tensor.dtype),
                "shape": list(tensor.shape), "numel": 0}
    head = flat[:8].float().tolist()
    return {
        "head8": [round(float(item), 12) for item in head],
        "sum": round(float(flat.float().sum()), 6),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": int(flat.numel()),
    }


def tensor_fingerprint(value: Any) -> Optional[str]:
    """A stable, cheap content fingerprint for one tensor.

    Halves the element count (worst case 2× under-sampling) and folds three
    independent reductions into one blake2b digest, so *any* content difference,
    including a single element, changes the value while the cost stays bounded.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        return None
    # F-T25-1: every tensor reaching this function may live on a CUDA device, and
    # ``.numpy()`` refuses a device tensor. The shim is the one
    # validation-engineer had to patch in at their run root, taken in here so a GPU
    # run cannot die on its first optimizer step. The hashed bytes are unchanged:
    # the same tensor hashed on host and on device produces the same digest.
    flat = _to_cpu(value).detach().reshape(-1)
    if flat.numel() == 0:
        return "empty"
    if flat.is_floating_point():
        flat = flat.float()
    flat = flat.contiguous()
    digest = hashlib.blake2b(digest_size=12)
    digest.update(str(flat.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    # Every other element keeps the digest sensitive to any content or length
    # change while bounding the cost on a large cached batch.
    sample = flat[::2].contiguous()
    digest.update(sample.numpy().tobytes())
    digest.update(str(float(flat.sum().item())).encode())
    digest.update(str(float((flat * flat).sum().item())).encode())
    return digest.hexdigest()


def adapters_digest(named_tensors: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """Aggregate hash over ``(name, tensor)`` pairs, order insensitive."""
    per_tensor: dict[str, str] = {}
    for name, tensor in named_tensors:
        fingerprint = tensor_fingerprint(tensor)
        if fingerprint is not None:
            per_tensor[name] = fingerprint
    aggregate = hashlib.blake2b(digest_size=12)
    for name in sorted(per_tensor):
        aggregate.update(f"{name}={per_tensor[name]};".encode())
    return {"aggregate": aggregate.hexdigest(), "count": len(per_tensor)}


def _l2(tensors: Iterable[Any]) -> float:
    import torch

    total = 0.0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.numel():
            total += float(torch.linalg.vector_norm(tensor.detach().float()) ** 2)
    return total ** 0.5


# ---------------------------------------------------------------------------
# latent provenance (hash -> the npz the sample really came from)
# ---------------------------------------------------------------------------


class LatentIndex:
    """Resolve a per-sample latent back to its cache file.

    F-T27-1/ground truth: a record must state which ``*_anima.npz`` (and which
    caption) the sample came from, not merely a slot id derived from a formula.
    The index is built from the cache directory itself, so a **match is proof** and
    a **miss is evidence**: a latent that matches nothing on disk was not read from
    that cache, whatever the slot formula says.
    """

    def __init__(self, cache_dir: str | os.PathLike[str]):
        self.cache_dir = Path(cache_dir)
        self._by_hash: Optional[dict[str, dict[str, Any]]] = None
        self.entries = 0

    def _build(self) -> dict[str, dict[str, Any]]:
        by_hash: dict[str, dict[str, Any]] = {}
        if not self.cache_dir.is_dir():
            return by_hash
        import numpy as np

        for path in sorted(self.cache_dir.glob("*_anima.npz")):
            try:
                with np.load(path) as archive:
                    key = next(
                        (name for name in archive.files if name.startswith("latents")),
                        None,
                    )
                    if key is None:
                        continue
                    rows = archive[key]
            except Exception:
                continue
            stem = path.name[: -len("_anima.npz")]
            caption = self.cache_dir / f"{stem}.txt"
            for row in rows if rows.ndim == 4 else [rows]:
                fingerprint = tensor_fingerprint(_to_cpu_tensor(row))
                if fingerprint is None:
                    continue
                by_hash[fingerprint] = {
                    "stem": stem,
                    "npz_path": str(self.cache_dir / path.name),
                    "caption_path": str(caption) if caption.exists() else None,
                }
        self.entries = len(by_hash)
        return by_hash

    def build(self) -> "LatentIndex":
        if self._by_hash is None:
            self._by_hash = self._build()
        return self

    def resolve(self, fingerprint: Optional[str]) -> Optional[dict[str, Any]]:
        if not fingerprint:
            return None
        if self._by_hash is None:
            self._by_hash = self._build()
        return self._by_hash.get(fingerprint)


def _to_cpu_tensor(row: Any) -> Any:
    """numpy row -> the tensor the trainer would have loaded."""
    import torch

    return torch.from_numpy(row) if not isinstance(row, torch.Tensor) else row


def latent_index(env: Mapping[str, str] | None = None) -> Optional[LatentIndex]:
    values = dict(os.environ if env is None else env)
    raw = str(values.get(ENV_LATENT_CACHE, "") or "").strip()
    if not raw:
        return None
    index = LatentIndex(raw)
    index.build()
    return index


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


@dataclass
class Recorder:
    """Writes one JSONL record per micro-batch per rank."""

    path: Path
    rtol: float = 1e-6
    _handle: Any = None
    _current: Optional[dict[str, Any]] = None
    _per_sample_loss: Any = None
    _batch_loss: Any = None
    _injected: Optional[dict[str, Any]] = None
    _index: Optional["LatentIndex"] = None
    _records: int = 0

    def open(self) -> "Recorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One file per (path, rank) so two ranks never interleave mid-line.
        self._handle = open(self.path, "a", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    # -- capture points -----------------------------------------------------
    def note_loss_call(self, per_sample: Any, batch_loss: Any) -> None:
        self._per_sample_loss = (
            None if per_sample is None else per_sample.detach().clone()
        )
        self._batch_loss = None if batch_loss is None else float(batch_loss)

    def begin(self, trainer: Any, state: Any, batch: Any) -> None:
        self._current = {"trainer": trainer, "state": state, "batch": batch}
        self._per_sample_loss = None
        self._batch_loss = None
        self._injected = None

    def note_injected(self, payload: dict[str, Any]) -> None:
        """What the noise fixture *actually* injected for this micro-batch.

        The record used to carry only the slot ids and the sigma the fixture file
        *declares*. If the injected noise does not match the declared slot, every
        existing criterion is blind to it — which is the most suspicious remaining
        explanation for the batch2/batch1 loss gap (t25). This captures the bytes.
        """
        self._injected = payload

    def on_optimizer_step(
        self,
        optimizer: Any,
        *,
        adapter_hash_before: Optional[str] = None,
        grad_raw: Optional[dict[str, Any]] = None,
    ) -> None:
        """Called *after* ``AdamW.step``: gradients and moments are still there.

        ``adapter_hash_before`` is the weight digest captured by the patched
        ``AdamW.step`` *before* the update. The init-divergence criterion needs the
        pre-update value: comparing post-update hashes at step 0 would blame a
        gradient difference on the initialisation.
        """
        current = self._current
        if current is None:
            return
        state = current["state"]
        try:
            record = self._build(optimizer, state, current["batch"])
            record["adapter_hash_before"] = adapter_hash_before
            # Gradients as the optimizer consumed them (t44): the tensor-level
            # evidence a float-order argument needs, next to the scalar norms.
            record["grad_raw"] = grad_raw
        except Exception as exc:  # pragma: no cover - diagnostics never break a run
            record = {
                "schema": SCHEMA,
                "probe_error": f"{type(exc).__name__}: {exc}",
                "rank": int(getattr(state.accelerator, "process_index", 0) or 0),
                "global_step": int(getattr(state, "global_step", -1)),
            }
        self._write(record)

    def end(self) -> None:
        self._current = None

    # -- record construction ------------------------------------------------
    def _build(self, optimizer: Any, state: Any, batch: Any) -> dict[str, Any]:
        import torch

        from library.training.multigpu import noise_fixture

        accelerator = state.accelerator
        rank = int(getattr(accelerator, "process_index", 0) or 0)
        world = int(getattr(accelerator, "num_processes", 1) or 1)
        step = int(getattr(state, "global_step", -1))
        consumed = int(getattr(state, "batches_consumed", 0) or 0)

        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        # ``--network_train_unet_only`` means the optimizer's parameters *are* the
        # adapter, so this is the adapter without touching the network.
        adapter = adapters_digest(
            (f"p{index}", parameter) for index, parameter in enumerate(parameters)
        )

        inputs = _inputs_fingerprint(
            batch,
            consumed,
            rank,
            world,
            noise_fixture,
            injected=self._injected,
            cache_index=self._index,
        )
        gradients = {
            "grad_norm_total": _l2(p.grad for p in parameters if p.grad is not None),
            "grad_norm_by_param": {
                f"p{index}": round(float(torch.linalg.vector_norm(p.grad.detach().float())), 9)
                for index, p in enumerate(parameters)
                if p.grad is not None
            },
            "grad_norm_by_group": [
                round(_l2(p.grad for p in group["params"] if p.grad is not None), 9)
                for group in optimizer.param_groups
            ],
            "grad_norm_by_prefix": _adapter_prefix_norms(
                parameters, _parameter_names(state)
            ),
            "grad_tensor_count": sum(1 for p in parameters if p.grad is not None),
        }
        moments = {
            "exp_avg_norm": _l2(
                optimizer.state[p].get("exp_avg") for p in parameters if p in optimizer.state
            ),
            "exp_avg_sq_norm": _l2(
                optimizer.state[p].get("exp_avg_sq")
                for p in parameters
                if p in optimizer.state
            ),
            "state_entries": sum(1 for p in parameters if p in optimizer.state),
        }

        per_sample = None
        if self._per_sample_loss is not None:
            per_sample = [round(float(v), 12) for v in self._per_sample_loss.reshape(-1)]

        return {
            "schema": SCHEMA,
            "rank": rank,
            "world_size": world,
            "global_step": step,
            "batches_consumed": consumed,
            "per_device_batch": len(inputs["slots"]),
            "slots": inputs["slots"],
            "sigmas": inputs["sigmas"],
            "latent_fingerprint": inputs["latent_fingerprint"],
            "text_fingerprint": inputs["text_fingerprint"],
            "latent_per_sample": inputs["latent_per_sample"],
            "text_per_sample": inputs["text_per_sample"],
            "noise_per_sample_hash": inputs["noise_per_sample_hash"],
            "timestep_injected": inputs["timestep_injected"],
            "noise_source": inputs["noise_source"],
            "noise_raw": inputs["noise_raw"],
            "latent_raw": inputs["latent_raw"],
            "text_raw": inputs["text_raw"],
            "text_key": inputs["text_key"],
            "latent_provenance": inputs["latent_provenance"],
            "capture_points": CAPTURE_POINTS,
            "fixture_provenance": _fixture_provenance(),
            # Sampler audit (t44): tells "descriptor reorder" and "loader re-iterated"
            # apart from the record itself instead of leaving it to inference.
            "stream_audit": _stream_audit(),
            "batch_keys": inputs["batch_keys"],
            "per_sample_loss": per_sample,
            "loss": self._batch_loss,
            "adapter_hash": adapter["aggregate"],
            "adapter_tensors": adapter["count"],
            **gradients,
            **moments,
        }

    def _write(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()
        self._records += 1

    @property
    def records(self) -> int:
        return self._records


def optimizer_adapter_digest(optimizer: Any) -> str:
    """Aggregate digest of the optimizer's own parameters (the adapter)."""
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    return adapters_digest(
        (f"p{index}", parameter) for index, parameter in enumerate(parameters)
    )["aggregate"]


#: Opt-in exact gradient dump. The per-tensor summaries below cannot yield a
#: tensor-level ``rel L2`` between two runs (that needs both vectors element-wise);
#: setting this to a path prefix writes the tensors once, when a gate needs it.
GRAD_DUMP_ENV = "ANIMA_MULTIGPU_PROBE_GRAD_DUMP"


def _grad_raw(optimizer: Any) -> dict[str, Any]:
    """Per-tensor gradient identity, captured **before** ``optimizer.step`` (t44).

    The record used to carry only norms: ``grad_norm_total``, 392 per-parameter L2
    norms and per-prefix norms. Those prove "the gradient magnitude matches", not
    "the gradient tensor matches a float-order-different reduction". These entries
    are captured in the ``AdamW.step`` patch *before* the original call (AdamW never
    writes ``.grad``, but capturing first keeps the point unambiguous) and follow the
    ``noise_raw``/``latent_raw`` convention: dtype, shape, numel, head8, sum.

    ``l2``/``abs_sum``/``max_abs`` are added so two runs can be compared per tensor
    without shipping every element, and ``l2_total`` is the global vector norm. A
    global ``rel L2`` between *two* runs still needs the elements, so an explicit
    ``ANIMA_MULTIGPU_PROBE_GRAD_DUMP=<prefix>`` writes an ``.npz`` next to the probe
    output and records its path + sha256 — that is the artifact a verification gate
    should compare, rather than re-deriving it from hashes.
    """
    import torch

    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    tensors: list[dict[str, Any]] = []
    flat_values: dict[str, Any] = {}
    total_sq = 0.0
    max_abs = 0.0
    for index, parameter in enumerate(parameters):
        grad = parameter.grad
        if grad is None:
            continue
        cpu = _to_cpu(grad).detach()
        flat = cpu.float().reshape(-1)
        numel = int(flat.numel())
        head = [round(float(item), 12) for item in flat[:8].tolist()]
        l2 = float(torch.linalg.vector_norm(flat)) if numel else 0.0
        peak = float(flat.abs().max()) if numel else 0.0
        total_sq += l2 * l2
        max_abs = max(max_abs, peak)
        tensors.append(
            {
                "index": index,
                "dtype": str(grad.dtype),
                "shape": list(grad.shape),
                "numel": numel,
                "head8": head,
                "sum": round(float(flat.sum()), 6),
                "abs_sum": round(float(flat.abs().sum()), 6),
                "l2": round(l2, 9),
                "max_abs": round(peak, 9),
            }
        )
        flat_values[f"p{index}"] = cpu.numpy()

    dump = None
    target = str(os.environ.get(GRAD_DUMP_ENV, "") or "").strip()
    if target and flat_values:
        try:
            import hashlib
            import numpy as np

            path = f"{target}-grads.npz"
            np.savez_compressed(path, **flat_values)
            with open(path, "rb") as handle:
                dump = {
                    "path": path,
                    "sha256": hashlib.sha256(handle.read()).hexdigest(),
                    "tensors": len(flat_values),
                }
        except Exception as exc:  # pragma: no cover - IO guard only
            dump = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "tensor_count": len(tensors),
        "l2_total": round(total_sq ** 0.5, 9),
        "max_abs_total": round(max_abs, 9),
        "tensors": tensors,
        "dump": dump,
    }


def _stream_audit() -> dict[str, Any]:
    """What the pinned sampler actually emitted in this process, per loader.

    ``t35``/``t43`` failures were diagnosed by reading item sequences out of the
    capture and reasoning backwards; the two causes — a mid-run *descriptor*
    reorder and a *re-iteration* of the loader — produce different audit shapes, so
    recording the audit makes the next anomaly readable directly:

    * ``passes`` counts ``__iter__`` starts on the pinned sampler (grows between
      records ⇒ the loader was re-iterated, i.e. the stream replayed its prefix);
    * ``duplicates`` counts repeated item indices in what was emitted;
    * ``prefix_ok`` is true while the emission is still the ascending identity
      stream, which is what a single continuous pass must look like.

    Always present. ``loaders`` is empty when the gate is closed (no audited loader
    exists), and ``available=False`` only if the gate module itself cannot be
    imported, so an empty list is never ambiguous about *why* it is empty.
    """

    try:
        from library.training.adaptive_runtime import pinned_order

        entries = pinned_order.audit()
        gate_open = bool(pinned_order.enabled())
        gate_name = pinned_order.PINNED_ORDER_ENV
    except Exception as exc:  # pragma: no cover - import guard only
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    loaders = []
    for entry in entries:
        emitted = list(entry.get("emitted") or [])
        loaders.append(
            {
                "item_count": entry.get("item_count"),
                "passes": entry.get("passes"),
                "emitted": len(emitted),
                "emitted_total": entry.get("emitted_total"),
                "emitted_truncated": entry.get("emitted_truncated"),
                "duplicates": entry.get("duplicates"),
                "prefix_ok": emitted == list(range(len(emitted))),
            }
        )
    return {
        "available": True,
        "gate": gate_name,
        "gate_open": gate_open,
        "loaders": loaders,
    }


def _fixture_provenance() -> Optional[dict[str, Any]]:
    """Which fixture file (path + sha256) and id mode the process is using.

    Two runs can both report ``noise_source=fixture`` while reading *different*
    files; only the digest tells them apart.
    """
    from library.training.multigpu import noise_fixture as fixture_mod

    path = str(os.environ.get(fixture_mod.ENV, "") or "").strip()
    if not path:
        return None
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        digest = None
    try:
        mode = fixture_mod.ids_mode()
    except Exception:
        mode = "invalid"
    return {"path": path, "sha256": digest, "ids_mode": mode}


def _parameter_names(state: Any) -> dict[int, str]:
    """``id(parameter) -> name`` for the network, so roles stop being ``other``.

    F-T25-3: the optimizer's parameters carry no names, and an earlier version
    reported every role as ``other``. The network is reachable from the loop state,
    so the names are read here instead of being guessed from ordering.
    """
    names: dict[int, str] = {}
    network = getattr(state, "network", None)
    for accessor in ("named_parameters", "parameters"):
        method = getattr(network, accessor, None)
        if method is None:
            continue
        try:
            items = method()
        except Exception:
            continue
        if accessor == "named_parameters":
            for name, parameter in items:
                names[id(parameter)] = str(name)
        else:
            for index, parameter in enumerate(items):
                names.setdefault(id(parameter), f"p{index}")
        break
    return names


def _adapter_prefix_norms(
    parameters: Sequence[Any], names: Optional[Mapping[int, str]] = None
) -> dict[str, float]:
    """Gradient norm per adapter role (``lora_down``/``lora_up``/``alpha``)."""
    import torch

    buckets: dict[str, list[Any]] = {}
    for index, parameter in enumerate(parameters):
        if parameter.grad is None:
            continue
        name = str((names or {}).get(id(parameter)) or f"p{index}")
        role = "other"
        for candidate in ("lora_down", "lora_up", "alpha", "lora_mid"):
            if candidate in name:
                role = candidate
                break
        buckets.setdefault(role, []).append(parameter.grad)
    return {
        role: round(_l2(grads), 9) for role, grads in sorted(buckets.items())
    }


def _inputs_fingerprint(
    batch: Any,
    consumed: int,
    rank: int,
    world: int,
    noise_fixture: Any,
    *,
    injected: Optional[Mapping[str, Any]] = None,
    cache_index: Optional["LatentIndex"] = None,
) -> dict[str, Any]:
    import torch

    slots: list[str] = []
    sigmas: list[Optional[float]] = []
    latent_fp = None
    text_fp = None
    keys: list[str] = []
    if isinstance(batch, Mapping):
        keys = sorted(str(key) for key in batch)
        latents = None
        for key in ("latents", "latent", "cached_latents"):
            if isinstance(batch.get(key), torch.Tensor):
                latents = batch[key]
                break
        if latents is not None:
            latent_fp = tensor_fingerprint(latents)
            batch_size = int(latents.shape[0])
            fixture = noise_fixture.active()
            try:
                slots = noise_fixture.sample_ids_from_position(
                    local_batch_index=int(consumed),
                    rank=int(rank),
                    world_size=int(world),
                    batch_size=batch_size,
                )
            except Exception:
                slots = []
            if fixture is not None and slots:
                records = fixture.records
                sigmas = [
                    (None if slot not in records else float(records[slot]["timestep"]))
                    for slot in slots
                ]
        text_parts = {}
        for key in keys:
            if not is_text_key(key):
                continue
            value = batch.get(key)
            fingerprint = tensor_fingerprint(value)
            if fingerprint is not None:
                text_parts[key] = fingerprint
        if text_parts:
            digest = hashlib.blake2b(digest_size=12)
            for key in sorted(text_parts):
                digest.update(f"{key}={text_parts[key]};".encode())
            text_fp = digest.hexdigest()
    latent_per_sample = None
    if latents is not None:
        latent_per_sample = [
            tensor_fingerprint(latents[index]) for index in range(int(latents.shape[0]))
        ]
    text_per_sample = None
    if isinstance(batch, Mapping):
        per_sample_text: list[str] = []
        size = None if latents is None else int(latents.shape[0])
        for index in range(size or 0):
            digest = hashlib.blake2b(digest_size=8)
            for key in keys:
                if not is_text_key(key):
                    continue
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and value.shape[0] > index:
                    digest.update(f"{key}={tensor_fingerprint(value[index])};".encode())
            per_sample_text.append(digest.hexdigest())
        text_per_sample = per_sample_text or None
    injected = injected or {}
    latent_raw = None
    text_raw = None
    if latents is not None:
        latent_raw = [_raw_summary(latents[index]) for index in range(int(latents.shape[0]))]
    first_text_key = next(
        (
            key
            for key in keys
            if isinstance(batch.get(key), torch.Tensor) and is_text_key(key)
        ),
        None,
    )
    if first_text_key is not None:
        value = batch[first_text_key]
        text_raw = [_raw_summary(value[index]) for index in range(int(value.shape[0]))]
    resolved = None
    if cache_index is not None and latent_per_sample is not None:
        resolved = [cache_index.resolve(item) for item in latent_per_sample]
    return {
        "slots": slots,
        "sigmas": sigmas,
        "latent_fingerprint": latent_fp,
        "text_fingerprint": text_fp,
        "latent_per_sample": latent_per_sample,
        "text_per_sample": text_per_sample,
        # F-29-2: this is a *hash*; the raw values live in ``noise_raw``. The old
        # shared name made a hash difference look like a value difference.
        "noise_per_sample_hash": injected.get("noise"),
        "timestep_injected": injected.get("sigmas"),
        "noise_source": "fixture" if injected else "unpinned",
        "noise_raw": injected.get("noise_raw"),
        "latent_raw": latent_raw,
        "text_raw": text_raw,
        "text_key": first_text_key,
        "latent_provenance": resolved,
        "batch_keys": keys,
    }


# ---------------------------------------------------------------------------
# installation (pure bench-side monkeypatch, reversible)
# ---------------------------------------------------------------------------


@dataclass
class _Patch:
    target: Any
    name: str
    original: Any
    wrapper: Any

    def restore(self) -> None:
        setattr(self.target, self.name, self.original)


_PATCHES: list[_Patch] = []
_ACTIVE: Optional[Recorder] = None


def enabled(env: Mapping[str, str] | None = None) -> Optional[Path]:
    values = dict(os.environ if env is None else env)
    raw = str(values.get(ENV, "") or "").strip()
    return Path(raw) if raw else None


def install(recorder: Recorder) -> Recorder:
    """Install the hooks. Idempotent; :func:`uninstall` reverses it."""
    global _ACTIVE
    if _PATCHES:
        return recorder
    import torch

    import library.training.loop as loop_mod
    from library.training import losses as losses_mod

    # (1) Micro-batch identity: the only place that knows trainer/state/batch.
    original_run = loop_mod._run_microbatch

    def patched_run_microbatch(trainer, state, batch, **kwargs):
        recorder.begin(trainer, state, batch)
        try:
            return original_run(trainer, state, batch, **kwargs)
        finally:
            recorder.end()

    loop_mod._run_microbatch = patched_run_microbatch
    _PATCHES.append(_Patch(loop_mod, "_run_microbatch", original_run, patched_run_microbatch))

    # (2) Loss: capture the per-sample vector and the scalar the loop consumes.
    original_compose = losses_mod.LossComposer.compose
    original_stage1 = {
        name: losses_mod.LOSS_REGISTRY[name]
        for name in losses_mod._STAGE_PER_SAMPLE
        if name in losses_mod.LOSS_REGISTRY
    }
    captured: dict[str, Any] = {}

    def patched_stage(name, function):
        def wrapper(ctx):
            out = function(ctx)
            if torch.is_tensor(out):
                captured[name] = out.detach().clone()
            return out

        return wrapper

    for name, function in original_stage1.items():
        losses_mod.LOSS_REGISTRY[name] = patched_stage(name, function)

    def patched_compose(self, ctx):
        captured.clear()
        out = original_compose(self, ctx)
        recorder.note_loss_call(_combined_per_sample(captured), out.detach().item())
        return out

    losses_mod.LossComposer.compose = patched_compose
    _PATCHES.append(
        _Patch(losses_mod.LossComposer, "compose", original_compose, patched_compose)
    )
    _PATCHES.append(
        _Patch(
            losses_mod,
            "LOSS_REGISTRY",
            dict(original_stage1),
            losses_mod.LOSS_REGISTRY,
        )
    )
    # NOTE: the registry stays patched until ``uninstall()``. An earlier version
    # restored it right here, which silently disabled the capture — that is why the
    # t25 run recorded ``per_sample_loss: null`` (F-T25-3).

    # (3) Injected noise/timestep: observe what the fixture really hands the model.
    from library.training.multigpu import noise_fixture as fixture_mod

    original_apply = fixture_mod.NoiseFixture.apply

    def patched_apply(self, noise, latents, batch, **kwargs):
        pinned_noise, sigmas = original_apply(self, noise, latents, batch, **kwargs)
        if _ACTIVE is not None:
            ids = fixture_mod.sample_ids_from_position(
                local_batch_index=int(kwargs.get("local_batch_index") or 0),
                rank=int(kwargs.get("rank", 0) or 0),
                world_size=int(kwargs.get("world_size", 1) or 1),
                batch_size=int(pinned_noise.shape[0]),
            )
            _ACTIVE.note_injected(
                {
                    "slots": ids,
                    "noise": [tensor_fingerprint(row) for row in pinned_noise],
                    "noise_raw": [_raw_summary(row) for row in pinned_noise],
                    "sigmas": [round(float(value), 9) for value in sigmas.reshape(-1)],
                }
            )
        return pinned_noise, sigmas

    fixture_mod.NoiseFixture.apply = patched_apply
    _PATCHES.append(
        _Patch(fixture_mod.NoiseFixture, "apply", original_apply, patched_apply)
    )

    # (4) Optimizer step: gradients and AdamW moments right where they are used.
    original_step = torch.optim.AdamW.step

    def patched_step(self, closure=None, **kwargs):
        before = optimizer_adapter_digest(self)
        # Gradients as the optimizer is about to consume them (t44): a tensor-level
        # comparison needs this, and ``AdamW.step`` never touches ``.grad``, so
        # capturing here keeps "before the update" explicit.
        grad_raw = _grad_raw(self)
        out = original_step(self, closure=closure, **kwargs)
        recorder.on_optimizer_step(
            self, adapter_hash_before=before, grad_raw=grad_raw
        )
        return out

    torch.optim.AdamW.step = patched_step
    _PATCHES.append(_Patch(torch.optim.AdamW, "step", original_step, patched_step))

    label_parameters()
    _ACTIVE = recorder
    return recorder


def _combined_per_sample(captured: Mapping[str, Any]) -> Any:
    """The stage-1 per-sample vector (the same `first/else +` the composer uses)."""
    combined = None
    for name in sorted(captured):
        contribution = captured[name]
        combined = contribution if combined is None else combined + contribution
    return combined


def label_parameters() -> None:
    """Name every optimizer parameter once so role norms are readable.

    Reads the trainer's network lazily (the probe is installed before the model is
    wrapped), so it is a no-op until a parameter set is visible.
    """
    # Naming happens on first use inside the recorder through the network's
    # ``named_parameters`` when available; this function only marks the module as
    # initialised so repeated installs stay cheap.
    return None


def uninstall() -> None:
    global _ACTIVE
    import torch

    from library.training import losses as losses_mod

    for patch in reversed(_PATCHES):
        patch.restore()
    _PATCHES.clear()
    # The loss registry dict was replaced by identity in the patch list; make sure
    # the *original* callables are back exactly once.
    if isinstance(_PATCHES, list):
        losses_mod.LOSS_REGISTRY.update(
            {name: fn for name, fn in list(losses_mod.LOSS_REGISTRY.items())}
        )
    _ACTIVE = None


def active() -> Optional[Recorder]:
    return _ACTIVE


def rank_path(path: Path, rank: int) -> Path:
    """Per-rank output file so two ranks never interleave inside one line."""
    if rank <= 0:
        return path
    return path.with_name(f"{path.stem}-rank{rank}{path.suffix}")


def maybe_install(env: Mapping[str, str] | None = None) -> Optional[Recorder]:
    """Install from the environment. Returns ``None`` when the probe is off.

    Rank 0 writes ``$ANIMA_MULTIGPU_DETERMINISM_PROBE``; every other rank writes
    the same name with a ``-rank<r>`` suffix, so a two-worker run produces one
    self-contained file per rank and :func:`compare` can pair them.
    """
    values = dict(os.environ if env is None else env)
    path = enabled(values)
    if path is None:
        return None
    try:
        rank = int(values.get("RANK") or values.get("LOCAL_RANK") or 0)
    except (TypeError, ValueError):
        rank = 0
    recorder = Recorder(path=rank_path(path, rank)).open()
    recorder._index = latent_index(values)
    return install(recorder)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def load(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _by_step(records: Iterable[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group by global step; every rank's micro-batch of that step is one entry."""
    out: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        out.setdefault(int(record.get("global_step", -1)), []).append(dict(record))
    for step in out:
        out[step].sort(key=lambda item: (int(item.get("rank", 0)), item.get("slots") or []))
    return out


def _union_field(entries: Sequence[Mapping[str, Any]], name: str) -> Any:
    """A step-level view of a per-rank field: sorted union over ranks.

    F-T25-2: the union is what makes a cross-world comparison meaningful. A
    ``world_size=2`` step holds two records, each with one sample; a
    ``world_size=1`` step holds one record with both samples. Comparing the records
    rank-by-rank made the cross-world case report ``init_divergence`` on every run,
    because rank 1 has no counterpart.
    """
    values: list[Any] = []
    for entry in entries:
        value = entry.get(name)
        if isinstance(value, list):
            values.extend(value)
        elif value is not None:
            values.append(value)
    if not values:
        return None
    try:
        return sorted(values)
    except TypeError:
        return values


def _hash_set(entries: Sequence[Mapping[str, Any]], name: str) -> Any:
    """The distinct values of a per-rank hash field (``None`` when unrecorded)."""
    values = {
        str(entry.get(name))
        for entry in entries
        if entry.get(name) is not None
    }
    return sorted(values) if values else None


def _slots_of(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(_union_field(entries, "slots") or [])


def _sample_map(
    entries: Sequence[Mapping[str, Any]],
    *,
    slots: Sequence[str],
    name: str,
    legacy: Optional[str] = None,
) -> dict[str, Any]:
    """``slot -> value`` for one per-sample field, across every rank of the step.

    This is the alignment F-T25-2 asks for: a sample is identified by its slot, not
    by the rank that happened to hold it.
    """
    mapping: dict[str, Any] = {}
    for entry in entries:
        entry_slots = entry.get("slots") or []
        value = entry.get(name)
        if value is None and legacy:
            value = entry.get(legacy)
        if not isinstance(value, list):
            continue
        for slot, item in zip(entry_slots, value):
            mapping[str(slot)] = item
    return {slot: mapping.get(slot) for slot in slots}


def _step_scalar(entries: Sequence[Mapping[str, Any]], name: str) -> Optional[float]:
    """Mean over ranks — the quantity DDP makes global."""
    values = [entry.get(name) for entry in entries if entry.get(name) is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _equal(left: Any, right: Any, rtol: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(float(left)), abs(float(right)), 1e-12)
        return abs(float(left) - float(right)) <= rtol * scale
    return left == right


def _step_view(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "slots": _union_field(entries, "slots"),
        "sigmas": _union_field(entries, "sigmas"),
        "latent_fingerprint": _union_field(entries, "latent_fingerprint"),
        "text_fingerprint": _union_field(entries, "text_fingerprint"),
        "noise_value": _sample_map(entries, slots=_slots_of(entries), name="noise_raw"),
        "noise_per_sample_hash": _sample_map(
            entries,
            slots=_slots_of(entries),
            name="noise_per_sample_hash",
            legacy="noise_per_sample",
        ),
        "timestep_injected": _sample_map(
            entries, slots=_slots_of(entries), name="timestep_injected"
        ),
        "latent_value": _sample_map(entries, slots=_slots_of(entries), name="latent_raw"),
        "text_value": _sample_map(entries, slots=_slots_of(entries), name="text_raw"),
        "latent_per_sample_hash": _sample_map(
            entries, slots=_slots_of(entries), name="latent_per_sample"
        ),
        "text_per_sample_hash": _sample_map(
            entries, slots=_slots_of(entries), name="text_per_sample"
        ),
        "noise_source": _hash_set(entries, "noise_source"),
        "per_sample_loss": _union_field(entries, "per_sample_loss"),
        "loss": _step_scalar(entries, "loss"),
        "grad_norm_total": _step_scalar(entries, "grad_norm_total"),
        "exp_avg_norm": _step_scalar(entries, "exp_avg_norm"),
        "exp_avg_sq_norm": _step_scalar(entries, "exp_avg_sq_norm"),
        "adapter_hash": _hash_set(entries, "adapter_hash"),
        "adapter_hash_before": _hash_set(entries, "adapter_hash_before"),
    }


def _first_difference(
    left: Mapping[str, Any], right: Mapping[str, Any], order: Sequence[str], rtol: float
) -> Optional[dict[str, Any]]:
    for name in order:
        if not _equal(left.get(name), right.get(name), rtol):
            return {"field": name, "a": left.get(name), "b": right.get(name)}
    return None


def compare(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    *,
    rtol: float = 1e-6,
    mode: str = "shape",
) -> dict[str, Any]:
    """The mechanical verdict.

    ``mode="shape"`` answers "single batch2 vs dual batch1" with the three
    mutually exclusive criteria; ``mode="resume"`` answers "continuous vs resumed"
    by reporting the first differing field in causal order.
    """
    steps_a = _by_step(records_a)
    steps_b = _by_step(records_b)
    common = sorted(set(steps_a) & set(steps_b))
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": mode,
        "rtol": rtol,
        "steps_a": sorted(steps_a),
        "steps_b": sorted(steps_b),
        "common_steps": common,
        "verdict": CODE_AGREE,
        "first_difference": None,
        "detail": "",
    }
    if not common:
        result["verdict"] = "no_common_step"
        result["detail"] = (
            "the two recordings share no global step; nothing can be compared "
            "(a probe run that stopped before its first commit produces this)"
        )
        return result

    views = {step: (_step_view(steps_a[step]), _step_view(steps_b[step])) for step in common}

    if mode == "resume":
        for step in common:
            left, right = views[step]
            difference = _first_difference(left, right, RESUME_FIELD_ORDER, rtol)
            if difference is not None:
                result["verdict"] = "resume_divergence"
                result["first_difference"] = {"global_step": step, **difference}
                result["detail"] = (
                    f"the resumed run first differs at global step {step} in "
                    f"{difference['field']}: continuous={difference['a']!r} "
                    f"resumed={difference['b']!r}"
                )
                return result
        result["detail"] = (
            "every compared field matches within rtol across "
            f"{len(common)} global step(s): the resumed segment reproduces the "
            "continuous run up to this point"
        )
        return result

    # ---- shape mode: three mutually exclusive criteria, in causal order -------
    init_step = common[0]
    left, right = views[init_step]
    # The pre-update digest is the initialisation witness; fall back to the
    # post-update one when a recording predates that field.
    init_left = left.get("adapter_hash_before") or left["adapter_hash"]
    init_right = right.get("adapter_hash_before") or right["adapter_hash"]
    if not _equal(init_left, init_right, 0.0):
        result["verdict"] = CODE_INIT
        result["first_difference"] = {
            "global_step": init_step,
            "field": "adapter_hash",
            "a": init_left,
            "b": init_right,
        }
        result["detail"] = (
            "the step-0 adapter hash differs before any update: the two shapes do "
            "not start from the same weights (initialisation or seed divergence). "
            "No data/forward/backward question is answerable yet"
        )
        return result

    input_difference = _first_difference(
        left,
        right,
        (
            "slots",
            "sigmas",
            "noise_source",
            # F-29-3: judge on what was actually fed to the model, not on a second
            # hash of the same tensor; the hashes stay last as a tie-breaker.
            "noise_value",
            "timestep_injected",
            "latent_value",
            "text_value",
            "noise_per_sample_hash",
            "latent_per_sample_hash",
            "text_per_sample_hash",
            "latent_fingerprint",
            "text_fingerprint",
        ),
        rtol,
    )
    if input_difference is not None:
        result["verdict"] = CODE_INPUT
        result["first_difference"] = {"global_step": init_step, **input_difference}
        result["detail"] = (
            "same initial weights but the pinned per-sample inputs differ before the "
            "first update, so this pair measures the data, not the protocol"
        )
        return result

    loss_difference = _first_difference(left, right, ("loss",), rtol)
    if loss_difference is not None:
        result["first_difference"] = {"global_step": init_step, **loss_difference}
        result["verdict"] = CODE_FORWARD
        result["detail"] = (
            "same initial weights and same pinned inputs, yet the loss differs on "
            "the first step: the forward is sensitive to the batch shape (SDPA "
            "kernel choice, compile graph, attention backend)"
        )
        return result

    grad_difference = _first_difference(left, right, ("grad_norm_total",), rtol)
    if grad_difference is not None:
        result["first_difference"] = {"global_step": init_step, **grad_difference}
        result["verdict"] = CODE_BACKWARD
        result["detail"] = (
            "same initial weights, same inputs and the same loss, but the gradient "
            "norm differs after the first backward: the backward pass or the "
            "cross-rank gradient sync diverges"
        )
        return result

    for step in common[1:]:
        left, right = views[step]
        difference = _first_difference(left, right, RESUME_FIELD_ORDER, rtol)
        if difference is not None:
            result["verdict"] = CODE_BACKWARD
            result["first_difference"] = {"global_step": step, **difference}
            result["detail"] = (
                f"step {init_step} agrees but global step {step} first differs in "
                f"{difference['field']}: the divergence is accumulated, not "
                "introduced at step 1"
            )
            return result

    result["detail"] = (
        "every compared field of every common step matches within rtol: the two "
        "shapes are equivalent up to floating-point reduction order"
    )
    return result


def provenance_report(
    records: Sequence[Mapping[str, Any]], index: "LatentIndex"
) -> dict[str, Any]:
    """Trace every per-sample latent in a recording back to a cache file.

    Works on records written before the raw fields existed: the per-sample latent
    hash is enough to prove "this is that npz" or to prove "this is nothing on
    disk". A miss is the interesting answer — it means the tensor a run fed to the
    forward was not read from the cache.
    """
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "cache_dir": str(index.cache_dir),
        "indexed_latents": index.entries,
        "steps": {},
        "summary": {"matched": 0, "unmatched": 0},
    }
    for record in records:
        step = int(record.get("global_step", -1))
        slots = record.get("slots") or []
        hashes = record.get("latent_per_sample") or []
        rows = []
        for slot, fingerprint in zip(slots, hashes):
            match = index.resolve(fingerprint)
            if match is None:
                out["summary"]["unmatched"] += 1
                rows.append(
                    {
                        "slot": slot,
                        "latent_hash": fingerprint,
                        "matcher": None,
                        "verdict": "not_from_this_cache",
                    }
                )
            else:
                out["summary"]["matched"] += 1
                rows.append(
                    {
                        "slot": slot,
                        "latent_hash": fingerprint,
                        **match,
                        "verdict": "disk_cache",
                    }
                )
        out["steps"].setdefault(str(step), []).append(
            {"rank": record.get("rank"), "rows": rows}
        )
    return out


# ---------------------------------------------------------------------------
# the four ready-made segments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeSegment:
    """One short probe segment, driven like a ``hot_run`` phase."""

    name: str
    world_size: int
    per_device_batch: int
    steps: int
    stop_at_step: int
    precision: str
    config: str
    #: continuation steps for the resume segment (0 = continuous)
    resume_steps: int = 0
    notes: str = ""

    def args(self, fixture: str, output_dir: str, probe_path: str) -> list[str]:
        argv = [
            "--config_file", self.config,
            "--output_dir", output_dir,
            "--seed", "20261107",
            "--max_data_loader_n_workers", "0",
            "--multigpu_seedable_sampler",
            "--max_train_steps", str(self.steps),
            "--train_batch_size", str(self.per_device_batch),
            "--mixed_precision", self.precision,
        ]
        if self.resume_steps:
            argv += ["--save_state", "--save_every_n_steps", str(self.stop_at_step)]
        return argv

    def command(self, fixture: str, output_dir: str, probe_path: str) -> str:
        """The exact shell line, including the probe wrapper.

        The probe is installed *inside* the training process by the wrapper, so a
        real run needs no change to ``train.py`` or to the launch plumbing.
        """
        argv = " ".join(self.args(fixture, output_dir, probe_path))
        wrapped = (
            "-m bench.multigpu_reference.determinism_probe wrap -- " + argv
        )
        if self.world_size > 1:
            return (
                "accelerate launch --num_processes "
                f"{self.world_size} --mixed_precision {self.precision} " + wrapped
            )
        return "python " + wrapped

    def env(self, fixture: str, probe_path: str) -> dict[str, str]:
        return {
            ENV: probe_path,
            "ANIMA_MULTIGPU_NOISE_FIXTURE": fixture,
            "ANIMA_MULTIGPU_NOISE_FIXTURE_IDS": "cursor",
        }


#: Frozen config names from t13/t22. Only ``batch_size`` (and precision in the
#: R-b probe) differ, exactly as in the t22 reference path.
FROZEN_CONFIG_DIR = (
    "output/tests/multigpu_reference_validation/t13-anima-hot-20261108/runtime-config"
)

SEGMENTS: dict[str, ProbeSegment] = {
    "single-batch2": ProbeSegment(
        name="single-batch2",
        world_size=1,
        per_device_batch=2,
        steps=3,
        stop_at_step=3,
        precision="no",
        config=f"{FROZEN_CONFIG_DIR}/run_batch2.toml",
        notes="S-d side 1: one worker, two samples per step",
    ),
    "dual-batch1": ProbeSegment(
        name="dual-batch1",
        world_size=2,
        per_device_batch=1,
        steps=3,
        stop_at_step=3,
        precision="no",
        config=f"{FROZEN_CONFIG_DIR}/run_batch1.toml",
        notes="S-d side 2: two workers, one sample each per step",
    ),
    "dual-continuous": ProbeSegment(
        name="dual-continuous",
        world_size=2,
        per_device_batch=1,
        steps=2,
        stop_at_step=2,
        precision="no",
        config=f"{FROZEN_CONFIG_DIR}/run_batch1.toml",
        notes="S-g side 1: two workers, both steps in one process pair",
    ),
    "dual-resume": ProbeSegment(
        name="dual-resume",
        world_size=2,
        per_device_batch=1,
        steps=1,
        stop_at_step=1,
        precision="no",
        config=f"{FROZEN_CONFIG_DIR}/run_batch1.toml",
        resume_steps=1,
        notes="S-g side 2: one step, save, then one resumed step (stop_at_step=1)",
    ),
}


def segment_report() -> dict[str, Any]:
    """The four segments plus the argv diff against the dual-batch1 baseline."""
    baseline = SEGMENTS["dual-batch1"]
    fixture = "<l3_fixture.jsonl>"
    out: dict[str, Any] = {"schema": SCHEMA, "frozen_config_dir": FROZEN_CONFIG_DIR, "segments": {}}
    base_argv = baseline.args(fixture, "<out>/dual-batch1", "<probe>/dual-batch1.jsonl")
    for name, segment in SEGMENTS.items():
        argv = segment.args(fixture, f"<out>/{name}", f"<probe>/{name}.jsonl")
        added = [token for token in argv if token not in base_argv]
        removed = [token for token in base_argv if token not in argv]
        out["segments"][name] = {
            "world_size": segment.world_size,
            "per_device_batch": segment.per_device_batch,
            "steps": segment.steps,
            "stop_at_step": segment.stop_at_step,
            "resume_steps": segment.resume_steps,
            "precision": segment.precision,
            "config": segment.config,
            "argv": argv,
            "env": segment.env(fixture, f"<probe>/{name}.jsonl"),
            "argv_diff_vs_dual-batch1": {"added": added, "removed": removed},
            "launch": segment.command(
                fixture, f"<out>/{name}", f"<probe>/{name}.jsonl"
            ),
            "notes": segment.notes,
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main_train(argv: Sequence[str]) -> int:
    """Install the probe from the environment, then run ``train.py`` unchanged.

    This is what the segments call: ``python -m ...determinism_probe wrap --
    <train.py argv>``. With the env var unset it is a transparent pass-through, so
    the same command line works for a probed and an unprobed run.
    """
    import runpy

    recorder = maybe_install()
    script = PROJECT_ROOT / "train.py"
    sys.argv = [str(script), *[str(item) for item in argv]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        # F-T25-4: a segment that saves state can leave accelerate/``torch.distributed``
        # mid-teardown when the process is stopped externally. Restoring the hooks
        # and closing our file must therefore never raise, and must never touch the
        # process group — the probe owns no collective.
        try:
            if recorder is not None:
                recorder.close()
        finally:
            try:
                uninstall()
            except Exception:  # pragma: no cover - teardown must be silent
                pass
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="determinism_probe",
        description=(
            "Per-micro-batch determinism diagnostics (off by default; enable with "
            f"{ENV}=<path>)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("segments", help="print the four ready-made segment invocations")
    prov_parser = sub.add_parser(
        "provenance", help="trace per-sample latents of a recording back to cache files"
    )
    prov_parser.add_argument("--records", required=True, nargs="+")
    prov_parser.add_argument("--latent-cache", required=True)
    prov_parser.add_argument("--out", default=None)

    sub.add_parser(
        "wrap",
        help="install the probe from the environment and run train.py (pass its argv after --)",
    )

    cmp_parser = sub.add_parser("compare", help="apply the mechanical verdicts")
    cmp_parser.add_argument("--a", required=True)
    cmp_parser.add_argument("--b", required=True)
    cmp_parser.add_argument("--rtol", type=float, default=1e-6)
    cmp_parser.add_argument("--mode", choices=("shape", "resume"), default="shape")
    cmp_parser.add_argument("--out", default=None, help="write the verdict JSON here")

    if argv is None:
        argv = sys.argv[1:]
    if "wrap" in argv:
        separator = list(argv).index("wrap")
        if "--" in argv[separator:]:
            cut = list(argv).index("--", separator)
            return main_train(list(argv)[cut + 1 :])
        return main_train(list(argv)[separator + 1 :])

    args = parser.parse_args(argv)
    if args.command == "segments":
        print(json.dumps(segment_report(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "provenance":
        records: list[dict[str, Any]] = []
        for path in args.records:
            records.extend(load(path))
        report = provenance_report(records, LatentIndex(args.latent_cache).build())
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if report["summary"]["unmatched"] == 0 else 1

    verdict = compare(load(args.a), load(args.b), rtol=args.rtol, mode=args.mode)
    text = json.dumps(verdict, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if verdict["verdict"] in (CODE_AGREE,) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
