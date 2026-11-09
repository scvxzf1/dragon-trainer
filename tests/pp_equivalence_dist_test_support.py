"""Two-rank Gloo case driver for the Anima PP equivalence harness (task t3).

This is the distributed half of ``tests/pp_equivalence_test_support.py``: it
spawns two Gloo ranks on CPU, runs one equivalence case through a real
``torch.distributed.pipelining`` stage split and returns an aggregated,
JSON-serialisable report that the pytest cases assert on.

Split from the fixture/comparison module only to keep both files readable; the
equality policy, the torch 2.10 constraints and the fixture rationale are
documented in ``tests/pp_equivalence_test_support.py``.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import torch

from tests.pp_equivalence_test_support import (
    BLOCK_SPLIT,
    DIST_TIMEOUT,
    JOIN_TIMEOUT_SECONDS,
    MODEL_SEED,
    AnimaPPStageModule,
    adapter_grads,
    boundary_spec,
    build_fixture,
    build_micro_anima,
    build_pipeline_stage,
    build_schedule,
    compare_grads,
    compare_losses,
    compare_tensor_sequences,
    compare_tensors,
    grad_exchange_plan,
    gradient_cosine,
    owned_blocks,
    reference_boundary,
    reference_boundary_full_batch,
    reference_step,
    skipped_entries,
    stage_examples,
    step_pipeline,
    step_single_microbatch,
)


@dataclass(frozen=True)
class CaseConfig:
    """One equivalence case: microbatch count, schedule and boundary mode."""

    microbatches: int
    schedule: str = "gpipe"
    wire_conditioning: bool = False
    split: int = BLOCK_SPLIT
    dezero_adaln: bool = True
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "microbatches": self.microbatches,
            "schedule": self.schedule,
            "wire_conditioning": self.wire_conditioning,
            "split": self.split,
            "dezero_adaln": self.dezero_adaln,
            "label": self.label,
        }

    @property
    def bit_exact_claimed(self) -> bool:
        return self.microbatches == 1

    def describe(self) -> str:
        boundary = (
            "hidden+conditioning" if self.wire_conditioning else "hidden-only"
        )
        equality = "bit-exact" if self.bit_exact_claimed else "numeric-only"
        return (
            f"microbatches={self.microbatches} schedule={self.schedule} "
            f"boundary={boundary} split={self.split} equality={equality}"
        )


def _tensor_list(tensors: Sequence[torch.Tensor]) -> list[list[float]]:
    return [tensor.detach().reshape(-1).tolist() for tensor in tensors]


def _run_worker(rank: int, init_path: str, workdir: Path, case: CaseConfig) -> None:
    """Entry point of one spawned rank."""

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
            report = _compare_on_rank(rank, case)
        finally:
            dist.destroy_process_group()
    except Exception:
        (workdir / f"rank{rank}.error").write_text(traceback.format_exc())
        raise
    (workdir / f"rank{rank}.json").write_text(json.dumps(report))


def _compare_on_rank(rank: int, case: CaseConfig) -> dict[str, Any]:
    """Build the fixtures, run the PP step and collect this rank's comparisons."""

    spec = boundary_spec(wire_conditioning=case.wire_conditioning)

    reference_model, reference_network, inputs, target = build_fixture(
        microbatches=case.microbatches, dezero_adaln=case.dezero_adaln
    )
    full_batch = reference_step(
        reference_model, reference_network, inputs, target, microbatched=False
    )
    microbatched = reference_step(
        reference_model, reference_network, inputs, target, microbatched=True
    )
    boundary_reference = reference_boundary(
        reference_model, inputs, split=case.split
    )
    boundary_full_batch = reference_boundary_full_batch(
        reference_model, inputs, split=case.split
    )
    del reference_model, reference_network

    model, network = build_micro_anima(seed=MODEL_SEED, dezero_adaln=case.dezero_adaln)
    examples = stage_examples(model, inputs, split=case.split)
    module = AnimaPPStageModule(
        model,
        network,
        rank=rank,
        split=case.split,
        rope=examples.rope,
        batch_inputs=inputs,
        wire_conditioning=case.wire_conditioning,
    )
    if rank == 0:
        input_args: tuple[torch.Tensor, ...] = tuple(value[:1] for value in inputs)
        output_args = (
            (examples.boundary, examples.t_embedding, examples.adaln)
            if case.wire_conditioning
            else (examples.boundary,)
        )
    else:
        input_args = (
            (examples.boundary, examples.t_embedding, examples.adaln)
            if case.wire_conditioning
            else (examples.boundary,)
        )
        output_args = (examples.prediction,)
    stage = build_pipeline_stage(
        module, input_args=input_args, output_args=output_args
    )

    module.reset_microbatch_cursor()
    if case.schedule == "single":
        losses, prediction = step_single_microbatch(
            stage, rank=rank, inputs=inputs, target=target
        )
    else:
        schedule = build_schedule(
            stage, microbatches=case.microbatches, schedule=case.schedule
        )
        losses, prediction = step_pipeline(
            schedule, rank=rank, inputs=inputs, target=target
        )

    grads = adapter_grads(network, blocks=owned_blocks(rank, split=case.split))
    reference_grads = {
        key: value for key, value in full_batch.grads.items() if key in grads
    }
    microbatched_grads = {
        key: value for key, value in microbatched.grads.items() if key in grads
    }

    report: dict[str, Any] = {
        "rank": rank,
        "case": case.as_dict(),
        "losses": losses,
        "loss_count": len(losses),
        "grads": {key: value.reshape(-1).tolist() for key, value in grads.items()},
        "reference_full_batch_loss": full_batch.loss,
        "reference_microbatched_loss": microbatched.loss,
        "reference_full_batch_losses": list(full_batch.losses),
        "reference_microbatched_losses": list(microbatched.losses),
        "boundary": {
            "spec": [entry.name for entry in spec],
            "on_wire": [entry.name for entry in spec if entry.on_wire],
            "skipped": list(skipped_entries(spec)),
            "plan": list(grad_exchange_plan(spec)),
            "stage_output_tensors": len(stage.get_outputs_meta()),
            "stage_input_tensors": len(getattr(stage, "inputs_meta", ()) or ()),
            "grad_send_slots": (
                len(stage.grad_send_info) if stage.grad_send_info is not None else None
            ),
            "grad_recv_slots": sorted(
                {len(value) for value in stage.grad_recv_info.values()}
            ),
        },
        "communication": dict(getattr(stage, "communication", {})),
        "structure": {
            "stage_modules": sorted(name for name, _ in module.model.named_children()),
            "adapter_modules": len(module.adapters),
            "stage_parameter_count": sum(1 for _ in module.parameters()),
            "stage_trainable_parameter_count": sum(
                1 for parameter in module.parameters() if parameter.requires_grad
            ),
        },
        "comparisons": {
            "grads_vs_microbatched": compare_grads(
                grads, microbatched_grads, name="grads_vs_microbatched"
            ),
            "grads_vs_full_batch": compare_grads(
                grads, reference_grads, name="grads_vs_full_batch"
            ),
        },
        # Track ② of the dual-track acceptance (user ruling): the gradient
        # *direction* must agree with the single-device reference.  Reported
        # next to the element-wise comparisons so the equivalence suite can
        # assert both; the element-wise numbers stay the magnitude evidence.
        "cosine": {
            "grads_vs_full_batch": gradient_cosine(
                grads, reference_grads, name="grads_vs_full_batch"
            ),
            "grads_vs_microbatched": gradient_cosine(
                grads, microbatched_grads, name="grads_vs_microbatched"
            ),
        },
    }

    if rank == 0:
        report["boundary_sent"] = _tensor_list(module.boundary_out)
        report["reference_boundary"] = _tensor_list(boundary_reference)
        report["reference_boundary_full_batch"] = _tensor_list(boundary_full_batch)
        report["comparisons"].update(
            {
                "boundary_vs_microbatched": compare_tensor_sequences(
                    "boundary", module.boundary_out, boundary_reference
                ).as_dict(),
                "boundary_vs_full_batch": compare_tensor_sequences(
                    "boundary", module.boundary_out, boundary_full_batch
                ).as_dict(),
            }
        )
    else:
        if prediction is None:
            raise AssertionError("last stage did not return a prediction")
        report["prediction"] = prediction.reshape(prediction.shape[0], -1).tolist()
        report["boundary_received"] = _tensor_list(module.boundary_in)
        report["reference_prediction"] = full_batch.prediction.reshape(
            full_batch.prediction.shape[0], -1
        ).tolist()
        report["reference_microbatched_prediction"] = microbatched.prediction.reshape(
            microbatched.prediction.shape[0], -1
        ).tolist()
        report["comparisons"].update(
            {
                "prediction_vs_microbatched": compare_tensors(
                    "prediction", prediction, microbatched.prediction
                ).as_dict(),
                "prediction_vs_full_batch": compare_tensors(
                    "prediction", prediction, full_batch.prediction
                ).as_dict(),
                "boundary_vs_microbatched": compare_tensor_sequences(
                    "boundary_received", module.boundary_in, boundary_reference
                ).as_dict(),
                "boundary_vs_full_batch": compare_tensor_sequences(
                    "boundary_received", module.boundary_in, boundary_full_batch
                ).as_dict(),
                "losses_vs_microbatched": compare_losses(
                    "losses_vs_microbatched", losses, microbatched.losses
                ),
                "mean_loss_vs_microbatched": compare_losses(
                    "mean_loss_vs_microbatched",
                    [sum(losses) / len(losses)],
                    [microbatched.loss],
                ),
                "mean_loss_vs_full_batch": compare_losses(
                    "mean_loss_vs_full_batch",
                    [sum(losses) / len(losses)],
                    [full_batch.loss],
                ),
            }
        )
    return report


def run_case(workdir: Path, case: CaseConfig) -> dict[str, Any]:
    """Spawn the two Gloo ranks, run the case and return the aggregated report."""

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    init_path = workdir / "init"
    context = get_context("spawn")
    processes = [
        context.Process(
            target=_run_worker, args=(rank, str(init_path), workdir, case)
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
        assert all(
            process.exitcode == 0 for process in processes
        ), f"PP case failed ({case.describe()}):\n{error_text}"
        reports = [
            json.loads((workdir / f"rank{rank}.json").read_text())
            for rank in range(2)
        ]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    return {"case": case.as_dict(), "ranks": reports}


def rank_report(report: dict[str, Any], rank: int) -> dict[str, Any]:
    for entry in report["ranks"]:
        if entry["rank"] == rank:
            return entry
    raise AssertionError(f"rank {rank} missing from report")


def comparison(report: dict[str, Any], rank: int, name: str) -> dict[str, Any]:
    entry = rank_report(report, rank)["comparisons"]
    if name not in entry:
        raise AssertionError(f"comparison {name!r} missing on rank {rank}")
    return entry[name]


def cosine(report: dict[str, Any], rank: int, name: str) -> dict[str, Any]:
    """Track ② metric recorded for ``name`` on ``rank``."""

    entry = rank_report(report, rank).get("cosine", {})
    if name not in entry:
        raise AssertionError(f"cosine {name!r} missing on rank {rank}")
    return entry[name]


def format_comparison(name: str, data: dict[str, Any]) -> str:
    return (
        f"{name}: equal={data['equal']} max_abs_diff={data['max_abs_diff']:.3e} "
        f"max_rel_diff={data['max_rel_diff']:.3e} "
        f"rel_to_scale={data['rel_to_scale']:.3e} numel={data['numel']}"
    )


def report_summary(report: dict[str, Any]) -> str:
    """Multi-line summary used in test failure messages."""

    lines = [f"case: {report['case']}"]
    for entry in report["ranks"]:
        lines.append(f" rank {entry['rank']}: losses={entry['losses']}")
        for name, data in sorted(entry["comparisons"].items()):
            lines.append("  " + format_comparison(name, data))
        lines.append(f"  communication={entry['communication']}")
        lines.append(f"  boundary={entry['boundary']}")
    return "\n".join(lines)
