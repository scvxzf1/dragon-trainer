"""Inventory and full optimizer-update probes using the production step body."""

from __future__ import annotations

import statistics
import math
import time
from pathlib import Path

import torch

from library.env import resolve_model_family
from .capacity import loaded_blocks, loaded_model_limit
from .policy import host_swap_limit
from .process import ProbeComplete, write_result
from .resources import host_memory
from .preferences import gpu_reserve_bytes, preference


def _quant_bytes(state) -> int:
    if state is None:
        return 0
    total = sum(
        value.numel() * value.element_size()
        for key in ("absmax", "code", "offset")
        if isinstance(value := getattr(state, key, None), torch.Tensor)
    )
    return total + _quant_bytes(getattr(state, "state2", None))


def model_block_bytes(model, family: str) -> list[int]:
    blocks = loaded_blocks(model, family)
    sizes = []
    for block in blocks:
        # Before adapter application, count all base weights conservatively.
        size = 0
        for module in block.modules():
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor):
                if weight.device.type == "meta":
                    raise ValueError("AUTO cannot inventory meta block weights")
                size += weight.numel() * weight.element_size()
                size += _quant_bytes(getattr(weight, "quant_state", None))
        sizes.append(int(size))
    return sizes


def after_model_load(args, model) -> None:
    request = getattr(args, "_auto_swap_probe", None)
    resolved = getattr(args, "_auto_swap_resolved", False)
    if not request and not resolved:
        return
    if not request and not args.blocks_to_swap:
        return
    sizes = model_block_bytes(model, resolve_model_family(args))
    memory = host_memory()
    sparse = preference(args) == "ram"
    maximum = host_swap_limit(sizes, memory.available, memory.reserve, sparse=sparse)
    if sparse and getattr(model, "offloader", None) is not None:
        model.offloader.master_scope = "participating"
    inventory = {
        "status": "inventory",
        "block_count": len(sizes),
        "block_bytes": sizes,
        "model_swap_limit": loaded_model_limit(model, resolve_model_family(args)),
        "host_swap_limit": maximum,
        "host_after_model_load": memory.to_dict(),
    }
    if request and request["inventory"]:
        raise ProbeComplete(inventory)
    if args.blocks_to_swap > maximum:
        if resolved:
            raise RuntimeError(
                "AUTO block-swap host budget changed before training; recalibration required"
            )
        raise ProbeComplete({**inventory, "status": "host_limit"})


def check_dataset(args, group, validation_group) -> None:
    if not getattr(args, "_auto_swap_probe", None) and not getattr(
        args, "_auto_swap_resolved", False
    ):
        return
    if validation_group is not None and len(validation_group):
        raise ValueError(
            "AUTO v1 does not calibrate validation; remove validation splits first"
        )
    if not hasattr(group, "datasets"):
        raise ValueError("AUTO v1 requires the built-in bucketed dataset")
    if resolve_model_family(args) == "anima" and any(
        getattr(subset, "caption_dropout_rate", 0) > 0
        for dataset in group.datasets
        for subset in getattr(dataset, "subsets", [])
    ):
        from library.inference.uncond import default_uncond_path

        if not default_uncond_path().is_file():
            raise ValueError(
                "AUTO requires the existing Anima uncond text cache for caption "
                "dropout; run text preprocessing before calibration"
            )
    representative_batches(group)


def representative_batches(group) -> list[dict]:
    """One fullest real batch per active resolution and dataset, largest first."""
    cases = []
    offset = 0
    for dataset_id, dataset in enumerate(group.datasets):
        by_bucket = {}
        for index, batch in enumerate(dataset.buckets_indices):
            bucket = dataset.bucket_manager.buckets[batch.bucket_index]
            width, height = dataset.bucket_manager.resos[batch.bucket_index]
            resolution = (int(width), int(height))
            size = min(
                batch.bucket_batch_size,
                len(bucket) - batch.batch_index * batch.bucket_batch_size,
            )
            if size <= 0:
                continue
            previous = by_bucket.get(resolution)
            if previous is None or size > previous["batch_size"]:
                by_bucket[resolution] = {
                    "index": offset + index,
                    "dataset": dataset_id,
                    "width": int(width),
                    "height": int(height),
                    "batch_size": size,
                    "weight": 0 if previous is None else previous["weight"],
                }
            by_bucket[resolution]["weight"] += 1
        cases.extend(by_bucket.values())
        offset += len(dataset)
    if not cases:
        raise ValueError("AUTO requires a nonempty bucketed training dataset")
    return sorted(
        cases, key=lambda c: (-c["width"] * c["height"] * c["batch_size"], c["index"])
    )


def weighted_case_seconds(cases: list[dict], per_case: list[float]) -> float:
    """Expected update time from per-case hot medians and dataset batch counts."""
    if not cases or len(cases) != len(per_case):
        raise ValueError("AUTO requires matching nonempty case timings and weights")
    weights = [case.get("weight") for case in cases]
    if any(type(weight) is not int or weight <= 0 for weight in weights):
        raise ValueError("AUTO requires positive integer case weights")
    if any(not math.isfinite(seconds) or seconds <= 0 for seconds in per_case):
        raise ValueError("AUTO requires finite positive case timing")
    return sum(weight * seconds for weight, seconds in zip(weights, per_case)) / sum(weights)


def _cuda_budget(device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return int(free + torch.cuda.memory_reserved(device))


def _probe_updates(trainer, state, cases: list[dict], group) -> list[dict]:
    from library.training.loop import _run_step

    accumulation = state.args.gradient_accumulation_steps
    updates = int(
        getattr(state.args, "_auto_swap_probe", {}).get("updates_per_case", 3)
    )
    if not 3 <= updates <= 64:
        raise ValueError("AUTO probe updates_per_case must be between 3 and 64")
    indices = [case["index"] for case in cases for _ in range(updates * accumulation)]
    loader = torch.utils.data.DataLoader(
        group,
        batch_size=1,
        sampler=indices,
        collate_fn=state.train_dataloader.collate_fn,
        num_workers=0,
        pin_memory=state.args.dataloader_pin_memory,
    )
    loader = state.accelerator.prepare_data_loader(loader)
    device = state.accelerator.device
    samples = []
    started = None
    budget = _cuda_budget(device)
    microsteps = 0
    loss_sum = 0.0
    skipped = 0
    consecutive_skips = 0
    for batch in loader:
        if started is None:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        state.current_step.value = state.global_step
        loss = _run_step(trainer, state, batch)
        microsteps += 1
        if not torch.isfinite(loss.detach()).all().item():
            raise RuntimeError("AUTO probe produced a nonfinite loss")
        loss_sum += loss.detach().float().item()
        del loss, batch
        if not state.accelerator.sync_gradients:
            continue
        torch.cuda.synchronize(device)
        if microsteps != accumulation:
            raise RuntimeError("AUTO probe did not complete a full optimizer update")
        from library.training.scaler_ownership import update_skipped as _update_skipped

        if _update_skipped(state):
            # No weights moved, so this step is not a valid timing/memory sample.
            from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

            skipped += 1
            consecutive_skips += 1
            # ``skip_budget`` is deliberately a separate name: reusing ``budget``
            # here overwrote the CUDA memory budget used for ``headroom`` below.
            skip_budget = max(
                0, int(getattr(state.args, "adaptive_max_nonfinite_skips", 0) or 0)
            )
            # Same semantics as TrainingRuntime/online: the budget bounds
            # *consecutive* skips so scaler warm-up is tolerated, while a config
            # that overflows every N steps still trips it.
            if consecutive_skips > skip_budget:
                raise NonfiniteSkip(
                    "AUTO probe stopped: loss scaler skipped optimizer update "
                    f"({consecutive_skips} consecutive skips > budget {skip_budget})"
                )
            microsteps = 0
            loss_sum = 0.0
            state.global_step += 1
            started = None
            continue
        consecutive_skips = 0
        budget = min(budget, _cuda_budget(device))
        samples.append(
            {
                "case": state.global_step // updates,
                "update": state.global_step % updates,
                "loss": loss_sum / accumulation,
                "seconds": time.perf_counter() - started,
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
                "headroom": budget - torch.cuda.max_memory_allocated(device),
            }
        )
        directory = getattr(state.args, "_auto_swap_probe", {}).get("directory")
        if directory:
            write_result(Path(directory) / "updates.json", {"updates": samples})
        state.global_step += 1
        started = None
        microsteps = 0
        loss_sum = 0.0
    if skipped:
        counts = {}
        for sample in samples:
            counts[sample["case"]] = counts.get(sample["case"], 0) + 1
        short = {case: count for case, count in counts.items() if count < updates}
        if short or len(counts) < len(cases):
            from library.training.adaptive_runtime.training_runtime import NonfiniteSkip

            raise NonfiniteSkip(
                "AUTO probe did not collect enough effective updates after "
                f"{skipped} skipped steps: {short or 'missing cases'}"
            )
    if len(samples) != updates * len(cases) or microsteps:
        raise RuntimeError("AUTO probe ended before all bucket updates completed")
    return samples


def run_probe(trainer, state) -> None:
    if not getattr(state.args, "_auto_swap_probe", None):
        return
    if state.accelerator.device.type != "cuda" or state.accelerator.num_processes != 1:
        raise ValueError("AUTO probes require single-process CUDA")
    if state.train_ctx.train_text_encoder:
        raise ValueError("AUTO v1 does not support text encoder training")
    group = state.train_dataloader.dataset
    state.current_epoch.value = 1
    group.set_current_epoch(1)
    cases = representative_batches(group)
    state.accelerator.unwrap_model(state.network).on_epoch_start(
        state.text_encoder, state.unet
    )
    state.optimizer_train_fn()
    samples = _probe_updates(trainer, state, cases, group)
    total = torch.cuda.get_device_properties(state.accelerator.device).total_memory
    reserve = gpu_reserve_bytes(state.args, total)
    headroom = min(sample["headroom"] for sample in samples)
    per_case = [
        statistics.median(
            s["seconds"] for s in samples if s["case"] == i and s["update"] > 0
        )
        for i in range(len(cases))
    ]
    offloader = getattr(state.accelerator.unwrap_model(state.unet), "offloader", None)
    raise ProbeComplete(
        {
            "status": "ok",
            "blocks": state.args.blocks_to_swap,
            "cases": cases,
            "updates": samples,
            "seconds": weighted_case_seconds(cases, per_case),
            "unweighted_seconds": sum(per_case) / len(per_case),
            "headroom": headroom,
            "gpu_reserve": reserve,
            "safe": headroom >= reserve,
            "cpu_master_bytes": int(
                getattr(offloader, "_frozen_weight_master_bytes", 0)
            ),
            "cpu_master_bytes_by_block": list(
                getattr(offloader, "_frozen_weight_bytes_by_block", [])
            ),
        }
    )
