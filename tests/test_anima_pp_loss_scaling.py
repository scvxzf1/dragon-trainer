"""The PP loss divisor is load-bearing — negative regression + dual-track metric.

Why this file exists (task t6, verification side)
------------------------------------------------
``library/training/pp/schedule.py`` builds the pipeline schedule with
``scale_grads=False`` and puts ``1/n_microbatches`` inside the last-stage loss
(:func:`library.training.pp.schedule.make_last_stage_loss_fn`).  The reason is a
torch 2.10 trap: ``PipelineStage.scale_grads`` divides ``self.submod.parameters()``
(``torch/distributed/pipelining/stage.py:570-582``), and a monkey-patched LoRA
adapter is **not** in the base model's parameter tree, so ``scale_grads=True``
would silently leave every adapter gradient ``n`` times too large.

``tests/test_training_pp_runtime.py::test_loss_carries_the_one_over_microbatches_factor``
only asserts ``loss == raw_mse / n`` on synthetic tensors.  That does **not**
prove the factor reaches the adapter gradients: a future change could flip
``scale_grads`` back to ``True`` (which divides the *registered* adapters) or
drop the divisor, and the loss-only test would keep passing while the trained
parameters are off by ``n``.

This file closes that hole the non-invasive way: the divisor is *removed through
the existing public ``loss_fn`` parameter of* ``build_anima_schedule`` — no
production code is modified, no internals are monkeypatched.  ``scaled=False``
builds the exact same production factory with ``microbatches=1`` (i.e.
``divisor = 1``), so baseline and negative run differ in nothing but the
``1/n`` factor.

It also implements the user's dual-track acceptance metric (see
``docs/experimental/anima_pipeline_parallel_v0.md`` §「双轨验收判据（track ① /
track ②）」):

* track ① structural equivalence — bit-exact vs the *same micro-batch split*;
  covered by ``tests/test_anima_pp_equivalence.py`` (``torch.equal``), not here;
* track ② algorithmic equivalence — vs the single-device **full batch**, where
  float addition is non-associative, scored by **gradient cosine similarity >
  0.9999** instead of an allclose.  The equivalence suite asserts the same
  metric (``tests/test_anima_pp_equivalence.py::_assert_gradient_cosine``); this
  file adds the *negative regression* that cosine cannot see — dropping the
  ``1/n`` must scale the adapter gradients by exactly ``n``.

Scope: CPU + Gloo, fp32, two stages, tiny Anima fixture (4 blocks, 8 MLP-LoRA
adapters, de-zeroed AdaLN).  No CUDA is touched and nothing here is a GPU
numerics or performance claim.
"""

from __future__ import annotations

import json
import traceback
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pytest
import torch

from tests import pp_equivalence_test_support as h

DIST_TIMEOUT = timedelta(seconds=90)
JOIN_TIMEOUT_SECONDS = 100

#: Track ② threshold (user ruling): gradient cosine similarity must exceed this.
COSINE_THRESHOLD = 0.9999

#: ``n`` times the baseline gradient, recomputed in floating point, can differ
#: from the un-scaled run by one ulp (dividing by 3 is not exact).  1e-6 is far
#: tighter than any real defect: dropping the factor changes the gradient by a
#: factor of 2 or 3, i.e. a relative error of 0.5 or 0.67.
RATIO_RTOL = 1e-6

#: The two cases the negative regression runs on.  ``microbatches=3`` uses
#: ``Schedule1F1B`` (the production schedule); ``microbatches=2`` is the smallest
#: count torch 2.10 accepts for a 2-stage pipeline and is GPipe-equivalent.
CASES: tuple[tuple[int, str], ...] = ((2, "gpipe"), (3, "1f1b"))


# --- one two-rank run --------------------------------------------------------


def _run_rank(rank: int, microbatches: int, schedule_name: str, scaled: bool) -> dict:
    """Run one PP step on this rank and return its gradients as plain lists."""

    from library.training.pp.schedule import (
        build_anima_schedule,
        make_last_stage_loss_fn,
    )

    # Single-device references, built from the same seeds as the PP fixture.
    reference_model, reference_network, inputs, target = h.build_fixture(
        microbatches=microbatches
    )
    full_batch = h.reference_step(
        reference_model, reference_network, inputs, target, microbatched=False
    )
    microbatched = h.reference_step(
        reference_model, reference_network, inputs, target, microbatched=True
    )
    del reference_model, reference_network

    model, network = h.build_micro_anima(seed=h.MODEL_SEED)
    examples = h.stage_examples(model, inputs)
    module = h.AnimaPPStageModule(
        model,
        network,
        rank=rank,
        split=h.BLOCK_SPLIT,
        rope=examples.rope,
        batch_inputs=inputs,
    )
    if rank == 0:
        input_args: tuple[torch.Tensor, ...] = tuple(value[:1] for value in inputs)
        output_args = (examples.boundary,)
    else:
        input_args = (examples.boundary,)
        output_args = (examples.prediction,)
    stage = h.build_pipeline_stage(
        module, input_args=input_args, output_args=output_args
    )

    module.reset_microbatch_cursor()
    # The only difference between the two runs: divisor n versus divisor 1.
    # Both go through the production factory, so the guard behaviour (loss only
    # on the last stage, mean reduction) is identical.
    loss_fn = make_last_stage_loss_fn(
        is_last=rank == 1, microbatches=microbatches if scaled else 1
    )
    schedule = build_anima_schedule(
        stage,
        microbatches=microbatches,
        schedule=schedule_name,
        loss_fn=loss_fn,
    )
    losses, prediction = h.step_pipeline(
        schedule, rank=rank, inputs=inputs, target=target
    )
    if rank == 1:
        assert prediction is not None, "the last stage must return a prediction"
    else:
        assert prediction is None, "only the last stage returns a prediction"

    owned = h.owned_blocks(rank)
    grads = h.adapter_grads(network, blocks=owned)
    reference_grads = {
        key: value for key, value in full_batch.grads.items() if key in grads
    }
    microbatched_grads = {
        key: value for key, value in microbatched.grads.items() if key in grads
    }
    return {
        "rank": rank,
        "microbatches": microbatches,
        "schedule": schedule_name,
        "scaled": scaled,
        "losses": [float(value) for value in losses],
        "reference_full_batch_loss": full_batch.loss,
        "reference_microbatched_loss": microbatched.loss,
        "grads": {key: value.reshape(-1).tolist() for key, value in grads.items()},
        "reference_full_batch_grads": {
            key: value.reshape(-1).tolist() for key, value in reference_grads.items()
        },
        "reference_microbatched_grads": {
            key: value.reshape(-1).tolist() for key, value in microbatched_grads.items()
        },
    }


def _worker(
    rank: int, init_path: str, workdir: Path, microbatches: int, schedule: str, scaled: bool
) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_path}",
            rank=rank,
            world_size=2,
            timeout=DIST_TIMEOUT,
        )
        try:
            report = _run_rank(rank, microbatches, schedule, scaled)
        finally:
            dist.destroy_process_group()
    except Exception:
        (workdir / f"rank{rank}.error").write_text(traceback.format_exc())
        raise
    (workdir / f"rank{rank}.json").write_text(json.dumps(report))


def run_case(
    workdir: Path, *, microbatches: int, schedule: str, scaled: bool
) -> dict[str, Any]:
    """Spawn two Gloo ranks for one (microbatches, schedule, scaled) combination."""

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    context = get_context("spawn")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, str(workdir / "init"), workdir, microbatches, schedule, scaled),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
        errors = sorted(workdir.glob("rank*.error"))
        error_text = "\n".join(path.read_text() for path in errors)
        assert all(process.exitcode == 0 for process in processes), (
            f"PP case failed (microbatches={microbatches} schedule={schedule} "
            f"scaled={scaled}):\n{error_text}"
        )
        reports = {
            rank: json.loads((workdir / f"rank{rank}.json").read_text())
            for rank in range(2)
        }
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    return reports


@pytest.fixture(scope="session")
def case_reports(tmp_path_factory):
    """Run every case once and reuse it across the two tests below (4 spawns)."""

    cache: dict[tuple[int, str, bool], dict[int, dict]] = {}

    def get(microbatches: int, schedule: str, scaled: bool) -> dict[int, dict]:
        key = (microbatches, schedule, scaled)
        if key not in cache:
            workdir = tmp_path_factory.mktemp(
                f"pp_scale_{microbatches}_{schedule}_{'scaled' if scaled else 'unscaled'}"
            )
            cache[key] = run_case(
                workdir, microbatches=microbatches, schedule=schedule, scaled=scaled
            )
        return cache[key]

    return get


# --- helpers -----------------------------------------------------------------


def _tensor_map(payload: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value) for key, value in payload.items()}


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    dot = torch.dot(actual.double(), expected.double())
    norm = actual.double().norm() * expected.double().norm()
    return float(dot / norm)


def _flatten(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([grads[key] for key in sorted(grads)])


# --- track ② : algorithmic equivalence by gradient cosine --------------------


@pytest.mark.integration
@pytest.mark.probe
@pytest.mark.parametrize("microbatches,schedule", CASES)
def test_gradient_cosine_vs_full_batch_exceeds_threshold(
    case_reports, microbatches: int, schedule: str
):
    """Track ②: PP gradients must point the same way as the full-batch step.

    The comparison target is the single-device **full batch** forward/backward.
    Splitting the batch changes GEMM shapes and the order in which
    per-microbatch contributions are summed, so bit-equality is not claimed
    (and must not be used as an acceptance condition); the metric is the cosine
    similarity of the gradient vectors, which is insensitive to the scalar
    accumulation noise that float non-associativity introduces.
    """

    reports = case_reports(microbatches, schedule, True)
    for rank in (0, 1):
        grads = _tensor_map(reports[rank]["grads"])
        reference = _tensor_map(reports[rank]["reference_full_batch_grads"])
        assert grads and set(grads) == set(reference)

        per_tensor = {
            key: _cosine(grads[key], reference[key]) for key in sorted(grads)
        }
        aggregate = _cosine(_flatten(grads), _flatten(reference))
        detail = ", ".join(
            f"{key}={value:.9f}" for key, value in sorted(per_tensor.items())
        )
        assert aggregate > COSINE_THRESHOLD, (
            f"rank {rank} (microbatches={microbatches}, schedule={schedule}): "
            f"gradient cosine vs the full batch is {aggregate:.9f}, below "
            f"{COSINE_THRESHOLD}; per-tensor: {detail}"
        )
        assert min(per_tensor.values()) > COSINE_THRESHOLD, (
            f"rank {rank}: at least one adapter gradient points away from the "
            f"full-batch reference; per-tensor: {detail}"
        )
        # The scale is right too: cosine is scale-invariant, so pin the
        # magnitude against the correctly-scaled microbatched reference.
        microbatched = _tensor_map(reports[rank]["reference_microbatched_grads"])
        comparison = h.compare_grads(grads, microbatched, name="grads_vs_microbatched")
        assert comparison["rel_to_scale"] < 1e-3, (
            f"rank {rank}: PP adapter gradients drifted from the microbatched "
            f"reference (rel_to_scale={comparison['rel_to_scale']:.3e}, "
            f"worst={comparison['worst_entry']})"
        )


# --- negative regression: the 1/n factor is load-bearing --------------------


@pytest.mark.integration
@pytest.mark.probe
@pytest.mark.parametrize("microbatches,schedule", CASES)
def test_adapter_grads_are_exactly_n_times_larger_without_the_loss_divisor(
    case_reports, microbatches: int, schedule: str
):
    """Removing the ``1/n`` from the loss must scale adapter grads by exactly ``n``.

    Regression guard for the torch 2.10 ``scale_grads`` trap: that post-pass only
    walks ``self.submod.parameters()`` and therefore never sees a monkey-patched
    LoRA adapter.  PP v0 keeps ``scale_grads=False`` and divides inside the loss;
    if someone drops the divisor (or "restores" ``scale_grads=True`` without the
    matching loss change) the adapter gradients become ``n`` times too large
    while a loss-only assertion would still pass.

    Both runs use the production ``make_last_stage_loss_fn``; the only change is
    ``microbatches=n`` (divisor ``n``) versus ``microbatches=1`` (divisor ``1``).
    """

    baseline = case_reports(microbatches, schedule, True)
    unscaled = case_reports(microbatches, schedule, False)
    n = float(microbatches)

    for rank in (0, 1):
        base_grads = _tensor_map(baseline[rank]["grads"])
        neg_grads = _tensor_map(unscaled[rank]["grads"])
        assert base_grads and set(base_grads) == set(neg_grads)

        # The baseline must itself be at the right scale, otherwise "n times the
        # baseline" would be measuring the wrong thing.
        reference = _tensor_map(baseline[rank]["reference_microbatched_grads"])
        base_vs_reference = h.compare_grads(
            base_grads, reference, name="baseline_vs_reference"
        )
        assert base_vs_reference["rel_to_scale"] < 1e-3, (
            f"rank {rank}: the baseline run is not at the reference scale "
            f"(rel_to_scale={base_vs_reference['rel_to_scale']:.3e})"
        )

        # Guard against a vacuous pass: without the divisor every adapter
        # gradient must be n times larger, measured on the gradient norm.
        ratio_report = {}
        for key in sorted(base_grads):
            ratio_report[key] = float(
                neg_grads[key].norm() / base_grads[key].norm()
            )
        assert all(
            abs(value - n) <= RATIO_RTOL * n for value in ratio_report.values()
        ), (
            f"rank {rank}: the un-scaled run did not scale the adapter gradients "
            f"by exactly {n:g}: {ratio_report}"
        )

        for key in sorted(base_grads):
            scaled_baseline = base_grads[key] * n
            if microbatches == 2:
                # Division by 2 is exact in binary floating point, and so is the
                # multiplication back, so this case is bit-exact.
                assert torch.equal(neg_grads[key], scaled_baseline), (
                    f"rank {rank}: {key} is not exactly {n:g}x the baseline; "
                    f"max_abs_diff="
                    f"{float((neg_grads[key] - scaled_baseline).abs().max()):.3e}"
                )
            # For n=3 the division is not exact and the per-microbatch
            # contributions are accumulated after scaling, so `n * baseline` and
            # the un-scaled run differ by a few ulps of the *intermediate*
            # magnitudes -- measured as a deviation relative to the tensor scale.
            # A missing factor is a 2x/3x error, i.e. six orders of magnitude
            # larger than this bound.
            scale = float(scaled_baseline.abs().max())
            deviation = float((neg_grads[key] - scaled_baseline).abs().max())
            assert deviation <= RATIO_RTOL * max(scale, 1e-12), (
                f"rank {rank}: {key} is not {n:g}x the baseline gradient; "
                f"deviation/scale={deviation / max(scale, 1e-12):.3e} "
                f"(norm ratio={ratio_report[key]:.9f})"
            )

        # The loss moved by the same factor, which pins *where* the difference
        # comes from (the loss, not the schedule or the adapter registration).
        base_loss = sum(baseline[rank]["losses"]) / max(len(baseline[rank]["losses"]), 1)
        neg_loss = sum(unscaled[rank]["losses"]) / max(len(unscaled[rank]["losses"]), 1)
        if rank == 1:
            assert base_loss > 0.0
            torch.testing.assert_close(
                torch.tensor(neg_loss / base_loss),
                torch.tensor(n),
                rtol=RATIO_RTOL,
                atol=0.0,
            )
        else:
            assert baseline[rank]["losses"] == [] and unscaled[rank]["losses"] == []


@pytest.mark.integration
@pytest.mark.probe
def test_structural_equivalence_is_claimed_only_against_the_same_split():
    """Track ① is bit-exact only for the *same* micro-batch split.

    This is a documentation test: it fails if the equality policy ever gets
    widened to "bit-exact against the full batch", which float non-associativity
    makes impossible for ``microbatches >= 2``.
    """

    from tests import pp_equivalence_dist_test_support as dist

    assert dist.CaseConfig(microbatches=1).bit_exact_claimed is True
    for microbatches in (2, 3):
        assert dist.CaseConfig(microbatches=microbatches).bit_exact_claimed is False
