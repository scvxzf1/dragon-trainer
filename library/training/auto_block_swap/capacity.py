"""Local model structure and loaded swap capacity without model downloads."""

from __future__ import annotations

import json
from itertools import chain
from pathlib import Path

from library.env import resolve_model_family
from library.models.family_registry import MODEL_FAMILY_REGISTRY, get_model_family_spec


class HostLimitReached(RuntimeError):
    """The measured host budget cannot support the requested swap count."""


def model_swap_limit(block_count: int) -> int:
    if type(block_count) is not int or block_count < 3:
        raise ValueError(f"Invalid swappable model block count: {block_count!r}")
    return block_count - 2


def preload_model_capacity(args) -> dict:
    """Read local checkpoint metadata only; fail closed on unknown structure."""
    family = resolve_model_family(args)
    spec = get_model_family_spec(family)
    if not spec.topology.supports_adaptive_swap_capacity:
        raise ValueError(f"Adaptive swap capacity does not support model_family={family!r}")
    path = getattr(args, "pretrained_model_name_or_path", None)
    if family == "anima":
        from library.anima.checkpoint import inspect_anima_checkpoint

        if not path:
            raise ValueError("Anima checkpoint path is required for adaptive retry")
        count = inspect_anima_checkpoint(path).num_blocks
    elif family == "krea2_raw":
        from library.models.krea2_raw.dit import SingleMMDiTConfig

        if not path or not Path(path).expanduser().is_file():
            raise ValueError("Local Krea-2 checkpoint is required for adaptive retry")
        count = SingleMMDiTConfig.krea2_raw().layers
    elif family == "z_image":
        from library.models.z_image.weights import validate_z_image_single_file

        if not path:
            raise ValueError("Z-Image transformer path is required for adaptive retry")
        root = Path(path).expanduser()
        if root.is_file():
            validate_z_image_single_file(str(root), "transformer")
            count = spec.topology.default_num_blocks
        else:
            config = root / "transformer" / "config.json"
            if not config.is_file():
                config = root / "config.json"
            try:
                data = json.loads(config.read_text(encoding="utf-8"))
                count = data["n_layers"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid local Z-Image transformer config: {config}") from exc
    else:
        raise ValueError(f"Adaptive swap capacity does not support model_family={family!r}")
    return {"model_family": family, "block_count": count,
            "model_limit": model_swap_limit(count)}


def loaded_blocks(model, family: str):
    spec = MODEL_FAMILY_REGISTRY.get(family)
    if spec is None or not spec.topology.supports_adaptive_swap_capacity:
        raise ValueError(f"Adaptive swap capacity does not support model_family={family!r}")
    return getattr(model, spec.topology.block_container)


def loaded_model_limit(model, family: str) -> int:
    return model_swap_limit(len(loaded_blocks(model, family)))


def loaded_host_capacity(args, model, expected: dict) -> dict:
    """Only a CPU-resident model permits the incremental host-master budget."""
    from .probe import model_block_bytes
    from .policy import host_swap_limit
    from .resources import host_memory

    family = expected["model_family"]
    count = len(loaded_blocks(model, family))
    if count != expected["block_count"] or model_swap_limit(count) != expected["model_limit"]:
        raise ValueError("Adaptive swap model structure drifted after preload")
    blocks = loaded_blocks(model, family)
    if any(tensor.device.type != "cpu" for block in blocks
           for tensor in chain(block.parameters(), block.buffers())):
        return {"host_limit": None, "host_budget_reason": "model_not_cpu_resident"}
    sizes = model_block_bytes(model, family)
    if len(sizes) != count:
        raise ValueError("Adaptive swap loaded block inventory drifted")
    memory = host_memory()
    limit = host_swap_limit(sizes, memory.available, memory.reserve)
    return {"host_limit": limit, "host_budget_reason": "measured_cpu_model",
            "host_after_model_load": memory.to_dict()}
