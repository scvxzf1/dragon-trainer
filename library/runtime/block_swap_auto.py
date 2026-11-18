"""In-process block-swap water-level auto tuner.

The existing AUTO block swap is a separate-process startup calibration and its docs
state it is deliberately not the same as blocks_to_swap = "auto". This module adds the
lighter in-process variant: after the first optimizer step the run measures its own
free VRAM, converts the remaining headroom into "how many more swapped blocks can stay
resident", and promotes them at the optimizer boundary through the existing
ModelOffloader.reconfigure() transaction.

Promotion only ever reduces the swap count, so it cannot OOM a run that started with a
conservative count; the boundary transaction rolls itself back if migration allocation
fails.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_MIB = 400
_ENV_MARGIN_MIB = "ANIMA_BLOCK_SWAP_AUTO_MARGIN_MIB"


def resolve_margin_bytes(margin_mib: Optional[int] = None) -> int:
    """Free-VRAM safety margin used by the promotion planning."""
    if margin_mib is None:
        raw = os.getenv(_ENV_MARGIN_MIB, "").strip()
        try:
            margin_mib = int(raw) if raw else DEFAULT_MARGIN_MIB
        except ValueError:
            margin_mib = DEFAULT_MARGIN_MIB
    return max(0, int(margin_mib)) * 1024 * 1024


def resolve_offloader_and_blocks(model: Any) -> Optional[tuple[Any, Any]]:
    """Best-effort (offloader, blocks) lookup across the model families.

    Anima and Krea-2 keep both attributes on the DiT itself; Z-Image and
    Qwen-Image 2.1 wrap the offloader in an adapter. Returns None when the model is
    not block-swapping, so callers can warn instead of failing a run.
    """
    if model is None:
        return None
    candidates = [model]
    for attr in (
        "_block_swap_controller",
        "_z_image_block_swap_adapter",
        "_qwen_image_2_1_block_swap_adapter",
        "offloader_holder",
    ):
        holder = getattr(model, attr, None)
        if holder is not None:
            candidates.append(holder)
    for holder in candidates:
        offloader = getattr(holder, "offloader", None)
        blocks = getattr(holder, "blocks", None)
        if blocks is None:
            blocks = getattr(holder, "layers", None)
        if offloader is not None and blocks is not None:
            return offloader, blocks
    return None


def per_block_bytes(offloader: Any, block_index: int) -> int:
    """Resident cost of one block: its frozen CPU masters stored bytes."""
    masters = getattr(offloader, "_cpu_weight_masters", None)
    if not masters or block_index >= len(masters) or block_index < 0:
        return 0
    total = 0
    for master in masters[block_index].values():
        stored = getattr(master, "stored_nbytes", None)
        if callable(stored):
            try:
                total += int(stored())
                continue
            except Exception:
                pass
        try:
            total += int(getattr(master, "nbytes", 0) or 0)
        except Exception:
            pass
    return total


def cuda_free_bytes(device: Any = None) -> Optional[int]:
    """Free device memory, or None when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        return None


_ENV_DMA_RATIO_FLOOR = "ANIMA_BLOCK_SWAP_AUTO_DMA_RATIO_FLOOR"
# Off by default: unconditional promotion is speed-optimal (round 3 measured
# swap=2 and swap=5 within noise). A positive value trades the last few percent
# of speed for keeping blocks swapped, which matters when VRAM is scarce.
DEFAULT_DMA_RATIO_FLOOR = 0.0


def resolve_dma_ratio_floor(value: Optional[float] = None) -> float:
    """Stop promoting once the projected DMA time drops below this step fraction.

    Round-3 measurements showed that below swap=5 on the 50HX the step is no
    longer transfer bound, so promoting further only costs residency.
    """
    if value is None:
        raw = os.getenv(_ENV_DMA_RATIO_FLOOR, "").strip()
        try:
            value = float(raw) if raw else DEFAULT_DMA_RATIO_FLOOR
        except ValueError:
            value = DEFAULT_DMA_RATIO_FLOOR
    return min(0.95, max(0.0, float(value)))


def measure_link_bytes_per_s(
    device: Any = None, *, probe_bytes: int = 64 * 1024 * 1024, repeats: int = 3
) -> Optional[float]:
    """Pinned H2D throughput in bytes/s, measured in-process (one block-sized copy)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        import torch as _torch

        if isinstance(device, _torch.device):
            index = device.index if device.index is not None else torch.cuda.current_device()
        elif device is None:
            index = torch.cuda.current_device()
        else:
            index = int(device)
        target = _torch.device("cuda", index)
        host = torch.empty(probe_bytes // 2, dtype=torch.bfloat16, device="cpu").pin_memory()
        gpu = torch.empty_like(host, device=target)
        stream = torch.cuda.Stream(device=target)
        best = None
        for _ in range(max(1, repeats)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                gpu.copy_(host, non_blocking=True)
                end.record(stream)
            stream.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            if elapsed_ms > 0:
                rate = probe_bytes / (elapsed_ms / 1000.0)
                best = rate if best is None else max(best, rate)
        return best
    except Exception:
        return None


def plan_promotion(
    offloader: Any,
    blocks: Any,
    *,
    margin_bytes: int,
    free_bytes: Optional[int] = None,
    step_ms: Optional[float] = None,
    link_bytes_per_s: Optional[float] = None,
    dma_ratio_floor: Optional[float] = None,
) -> dict:
    """How many swapped blocks can become resident, and the resulting count."""
    current = int(getattr(offloader, "blocks_to_swap", 0) or 0)
    count = len(blocks)
    if free_bytes is None:
        free_bytes = cuda_free_bytes(getattr(offloader, "device", None))
    block_bytes = per_block_bytes(offloader, count - current) if current else 0
    plan = {
        "current_swap": current,
        "num_blocks": count,
        "free_bytes": free_bytes,
        "margin_bytes": int(margin_bytes),
        "block_bytes": block_bytes,
        "promotable": 0,
        "new_swap": current,
        "reason": "",
    }
    if current == 0:
        plan["reason"] = "block swap disabled"
        return plan
    if free_bytes is None:
        plan["reason"] = "CUDA unavailable"
        return plan
    if block_bytes <= 0:
        plan["reason"] = "no CPU masters measured yet"
        return plan
    headroom = int(free_bytes) - int(margin_bytes)
    if headroom <= 0:
        plan["reason"] = "no headroom above margin"
        return plan
    promotable = headroom // block_bytes
    # Keep the offloader contract: at least two blocks stay swapped.
    promotable = min(promotable, max(0, current - 2))

    # Stop condition: each swapped block is restored twice per step (forward tail
    # plus backward early blocks), so the projected DMA time after promotion is
    # 2 * swap * block_bytes / link. Below the configured step fraction the step
    # is no longer transfer bound and promoting further buys nothing.
    floor = resolve_dma_ratio_floor(dma_ratio_floor)
    plan["step_ms"] = None if step_ms is None else float(step_ms)
    plan["link_bytes_per_s"] = None if link_bytes_per_s is None else float(link_bytes_per_s)
    plan["dma_ratio_floor"] = floor
    plan["stopped_by_dma"] = False
    if step_ms and link_bytes_per_s and link_bytes_per_s > 0:
        def projected(swap: int) -> float:
            return (2.0 * swap * block_bytes) / link_bytes_per_s * 1000.0

        plan["projected_dma_ms_before"] = projected(current)
        plan["projected_dma_ms_after"] = projected(current - promotable)
        budget_ms = floor * float(step_ms)
        # Promote until the first level whose projected DMA is no longer hidden;
        # stopping earlier would leave traffic that still has to be covered.
        while promotable > 0 and projected(current - promotable) < budget_ms:
            promotable -= 1
        if promotable < min(headroom // block_bytes, max(0, current - 2)):
            plan["stopped_by_dma"] = True
        plan["projected_dma_ms_after"] = projected(current - promotable)
    plan["promotable"] = int(promotable)
    plan["new_swap"] = int(current - promotable)
    plan["reason"] = (
        "stopped: projected DMA already below the step fraction"
        if plan["stopped_by_dma"]
        else "headroom converted into resident blocks"
    )
    return plan


def maybe_promote(
    model: Any,
    *,
    margin_mib: Optional[int] = None,
    free_bytes: Optional[int] = None,
    step_ms: Optional[float] = None,
    dma_ratio_floor: Optional[float] = None,
) -> Optional[dict]:
    """Plan and apply one promotion at an optimizer boundary.

    Returns the plan with an applied flag, or None when the model is not
    block-swapping.
    """
    found = resolve_offloader_and_blocks(model)
    if found is None:
        logger.warning(
            "block_swap_auto requested but no block-swap offloader was found on the model; "
            "leaving the manual blocks_to_swap unchanged."
        )
        return None
    offloader, blocks = found
    plan = plan_promotion(
        offloader,
        blocks,
        margin_bytes=resolve_margin_bytes(margin_mib),
        free_bytes=free_bytes,
        step_ms=step_ms,
        dma_ratio_floor=dma_ratio_floor,
        link_bytes_per_s=measure_link_bytes_per_s(getattr(offloader, "device", None))
        if step_ms and (dma_ratio_floor or resolve_dma_ratio_floor() > 0)
        else None,
    )
    plan["applied"] = False
    if plan["promotable"] <= 0 or plan["new_swap"] == plan["current_swap"]:
        logger.info(
            "block_swap_auto: keeping blocks_to_swap=%s (free=%.2f GiB, margin=%.2f GiB, "
            "block=%.1f MiB, %s)",
            plan["current_swap"],
            (plan["free_bytes"] or 0) / 2**30,
            plan["margin_bytes"] / 2**30,
            plan["block_bytes"] / 2**20,
            plan["reason"],
        )
        return plan
    logger.info(
        "block_swap_auto: promoting %s block(s) to resident, blocks_to_swap %s -> %s "
        "(free=%.2f GiB, margin=%.2f GiB, block=%.1f MiB)",
        plan["promotable"],
        plan["current_swap"],
        plan["new_swap"],
        (plan["free_bytes"] or 0) / 2**30,
        plan["margin_bytes"] / 2**30,
        plan["block_bytes"] / 2**20,
    )
    if plan.get("projected_dma_ms_before") is not None:
        logger.info(
            "block_swap_auto: projected DMA %.0f -> %.0f ms (step %.0f ms, floor %.2f, "
            "link %.2f GB/s)%s",
            plan["projected_dma_ms_before"],
            plan["projected_dma_ms_after"],
            plan["step_ms"] or 0.0,
            plan["dma_ratio_floor"],
            (plan["link_bytes_per_s"] or 0.0) / 1e9,
            " [stopped by the DMA floor]" if plan.get("stopped_by_dma") else "",
        )
    plan["applied"] = bool(offloader.reconfigure(blocks, plan["new_swap"]))
    return plan
