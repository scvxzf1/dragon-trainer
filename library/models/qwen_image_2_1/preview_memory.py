"""Qwen preview configuration and temporary, lossless block residency."""

from contextlib import contextmanager
from collections.abc import Mapping
import logging
from library.training.preview_decode_config import (
    add_arguments as add_decode_arguments, configuration_errors as decode_errors,
)

logger = logging.getLogger(__name__)


DEFAULTS = {"qwen_sample_swap_mode": "inherit", "qwen_sample_blocks_to_swap": 0}


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def configuration_errors(config, *, family=None):
    """Return key/message pairs shared by CLI and Web/TOML preflight."""
    errors = decode_errors(config)
    mode = _get(config, "qwen_sample_swap_mode", "inherit")
    count = _get(config, "qwen_sample_blocks_to_swap", 0)
    if mode not in ("inherit", "custom"):
        errors.append(("qwen_sample_swap_mode", "Expected inherit or custom"))
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 30:
        errors.append(("qwen_sample_blocks_to_swap", "Expected an integer between 0 and 30"))
    family = family or _get(config, "model_family", "anima") or "anima"
    if family != "qwen_image_2_1":
        for key, default in DEFAULTS.items():
            value = _get(config, key, default)
            if value != default:
                errors.append((key, "Qwen preview memory overrides require qwen_image_2_1"))
    elif mode == "custom" and count != (_get(config, "blocks_to_swap", 0) or 0):
        from library.runtime.block_swap_config import normalize_block_swap_transfer_dtype
        try:
            transfer = normalize_block_swap_transfer_dtype(_get(config, "block_swap_transfer_dtype"))
        except ValueError as exc:
            errors.append(("block_swap_transfer_dtype", str(exc)))
        else:
            if transfer != "bf16":
                errors.append(("qwen_sample_swap_mode", "Custom preview swap changes require lossless BF16 transfers"))
    return errors


def validate_config(config, *, family=None):
    errors = configuration_errors(config, family=family)
    if errors:
        raise ValueError("; ".join(f"{key}: {message}" for key, message in errors))


def add_arguments(parser):
    parser.add_argument("--qwen_sample_swap_mode", choices=("inherit", "custom"), default="inherit",
                        help="Qwen preview block residency: inherit training or use a custom count.")
    parser.add_argument("--qwen_sample_blocks_to_swap", type=int, default=0,
                        help="Qwen custom preview swap count, 0..30; zero keeps all blocks resident.")
    add_decode_arguments(parser)


def needs_preview_offloader(config):
    return (_get(config, "qwen_sample_swap_mode", "inherit") == "custom"
            and _get(config, "qwen_sample_blocks_to_swap", 0) > 0)


@contextmanager
def preview_swap_scope(model, config):
    """Restore the training topology even when entry or generation fails."""
    validate_config(config, family="qwen_image_2_1")
    old_count = getattr(model, "blocks_to_swap", 0)
    count = (old_count if _get(config, "qwen_sample_swap_mode", "inherit") == "inherit"
             else _get(config, "qwen_sample_blocks_to_swap", 0))
    changed = count != old_count
    offloader = getattr(model, "offloader", None)
    blocks = getattr(model, "transformer_blocks", None)
    if changed:
        if offloader is None or blocks is None:
            raise RuntimeError("Custom Qwen preview swap requires an installed training offloader")
        if getattr(offloader, "forward_only", False) or not getattr(offloader, "supports_backward", True):
            raise RuntimeError("Custom Qwen preview swap requires training forward_only=False")
        if getattr(offloader, "transfer_dtype", "bf16") != "bf16":
            raise ValueError("Custom Qwen preview swap requires lossless BF16 transfers")
        if any(p.grad is not None for p in model.parameters() if p.requires_grad):
            raise RuntimeError("Custom Qwen preview swap requires cleared optimizer gradients")
    error = None
    try:
        logger.info("Qwen preview block swap: mode=%s, training=%s, preview=%s",
                    _get(config, "qwen_sample_swap_mode", "inherit"), old_count, count)
        if changed:
            offloader.reconfigure(blocks, count)
            model.blocks_to_swap = count
        switch = getattr(model, "switch_block_swap_for_inference", None)
        if switch:
            switch()
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if changed and offloader is not None and blocks is not None:
                offloader.set_forward_only(False)
                offloader.reconfigure(blocks, old_count)
                model.blocks_to_swap = old_count
            switch = getattr(model, "switch_block_swap_for_training", None)
            if switch:
                switch()
        except BaseException as restore_error:
            fatal = RuntimeError("Qwen preview failed to restore training block swap")
            if error is not None:
                fatal.add_note(f"Original preview error: {error!r}")
            raise fatal from restore_error


@contextmanager
def preview_state_scope(model, network, device):
    """Preserve RNG around all entry/exit hooks, including failed mode changes."""
    import torch

    modules = [(module, module.training) for module in (model, network) if module is not None]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        error = None
        try:
            for module, _ in modules:
                module.eval()
            if network is not None:
                from library.training.preview_lifecycle import reset_network_preview_routing
                reset_network_preview_routing(network, include_timestep_mask=True)
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            failures = []
            for module, training in modules:
                try:
                    module.train(training)
                except Exception as exc:
                    failures.append(exc)
            if failures:
                fatal = RuntimeError("Qwen preview failed to restore model/adapter mode")
                if error is not None:
                    fatal.add_note(f"Original preview error: {error!r}")
                for failure in failures[1:]:
                    fatal.add_note(f"Additional mode restore error: {failure!r}")
                raise fatal from failures[0]
