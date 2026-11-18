"""Shared block-swap controller, explicit swap mixin, and CLI config adapter.

Provides two residency integration modes around ``ModelOffloader``:
1. ``BlockSwapController`` (``wrap_forward`` mode): wraps stock Diffusers /
   third-party block ``forward`` methods and ``_gradient_checkpointing_func``
   without modifying upstream model classes (used by Z-Image and Qwen-Image 2.1).
2. ``ExplicitBlockSwapMixin`` (``explicit`` mode): provides the unified
   residency lifecycle methods (``move_to_device_except_swap_blocks``,
   ``prepare_block_swap_before_forward``, ``switch_block_swap_for_inference``,
   ``switch_block_swap_for_training``, ``pause_block_swap``,
   ``resume_block_swap``, ``flush_block_swap_profile``) for DiTs whose native
   block loop already invokes ``wait_for_block`` / ``submit_move_blocks``
   (used by Anima, Krea-2, and MOSS-Image).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn

from library.runtime.offloading import ModelOffloader

_UNSET = object()


@dataclass(frozen=True)
class BlockSwapConfig:
    """Normalized ``ModelOffloader`` keyword configuration parsed from CLI/TOML args."""

    profile_jsonl: str | None = None
    transfer_dtype: str | None = None
    restore_mode: str | None = None
    int8_restore_mode: str | None = None
    int8_scope: str | None = None
    int8_staging_ring: int | None = None
    int8_dual_stream: bool | None = None
    int8_restore_chunk_rows: int | None = None

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        profile_jsonl: str | None | object = _UNSET,
    ) -> BlockSwapConfig:
        if profile_jsonl is _UNSET:
            from library.training.train_bootstrap import resolve_block_swap_profile_jsonl

            resolved_profile = resolve_block_swap_profile_jsonl(args)
        else:
            resolved_profile = profile_jsonl  # type: ignore[assignment]
        return cls(
            profile_jsonl=resolved_profile,
            transfer_dtype=getattr(args, "block_swap_transfer_dtype", None),
            restore_mode=getattr(args, "block_swap_restore_mode", None),
            int8_restore_mode=getattr(args, "block_swap_int8_restore_mode", None),
            int8_scope=getattr(args, "block_swap_int8_scope", None),
            int8_staging_ring=getattr(args, "block_swap_int8_staging_ring", None),
            int8_dual_stream=getattr(args, "block_swap_int8_dual_stream", None),
            int8_restore_chunk_rows=getattr(
                args, "block_swap_int8_restore_chunk_rows", None
            ),
        )

    def to_offloader_kwargs(self) -> dict[str, Any]:
        return {
            "profile_jsonl": self.profile_jsonl,
            "transfer_dtype": self.transfer_dtype,
            "restore_mode": self.restore_mode,
            "int8_restore_mode": self.int8_restore_mode,
            "int8_scope": self.int8_scope,
            "int8_staging_ring": self.int8_staging_ring,
            "int8_dual_stream": self.int8_dual_stream,
            "int8_restore_chunk_rows": self.int8_restore_chunk_rows,
        }

    def probe_fields(self, blocks_to_swap: int) -> dict[str, Any]:
        return {
            "blocks_to_swap": blocks_to_swap,
            "block_swap_profile_jsonl": self.profile_jsonl or "off",
            "block_swap_transfer_dtype": self.transfer_dtype,
            "block_swap_restore_mode": self.restore_mode,
        }


class ExplicitBlockSwapMixin:
    """Shared training/eval residency lifecycle for explicit-hook DiT models."""

    _block_swap_container_attr: str = "blocks"
    _block_swap_switch_banner: str | None = None

    def _init_explicit_block_swap(
        self,
        num_blocks: int,
        device: torch.device | str,
        *,
        offloader_cls: Callable[..., Any] = ModelOffloader,
        **offloader_kwargs: Any,
    ) -> Any:
        container = getattr(self, self._block_swap_container_attr)
        max_blocks = len(container) - 2
        assert num_blocks <= max_blocks, (
            f"Cannot swap more than {max_blocks} blocks. Requested: {num_blocks} blocks."
        )
        self.blocks_to_swap = num_blocks
        self.offloader = offloader_cls(
            container,
            self.blocks_to_swap,
            torch.device(device),
            **offloader_kwargs,
        )
        return self.offloader

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if not getattr(self, "blocks_to_swap", 0):
            self.to(device)  # type: ignore[attr-defined]
            return
        attr = self._block_swap_container_attr
        saved = getattr(self, attr)
        setattr(self, attr, None)
        try:
            self.to(device)  # type: ignore[attr-defined]
        finally:
            setattr(self, attr, saved)

    def prepare_block_swap_before_forward(self, free_cache: bool = True) -> None:
        if not getattr(self, "blocks_to_swap", 0) or getattr(self, "offloader", None) is None:
            return
        self.offloader.prepare_block_devices_before_forward(
            getattr(self, self._block_swap_container_attr),
            free_cache=free_cache,
        )

    def switch_block_swap_for_inference(self) -> None:
        if not getattr(self, "blocks_to_swap", 0) or getattr(self, "offloader", None) is None:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()
        if self._block_swap_switch_banner:
            print(f"{self._block_swap_switch_banner}: Block swap set to forward only.")

    def switch_block_swap_for_training(self) -> None:
        if not getattr(self, "blocks_to_swap", 0) or getattr(self, "offloader", None) is None:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()
        if self._block_swap_switch_banner:
            print(
                f"{self._block_swap_switch_banner}: Block swap set to forward and backward."
            )

    def pause_block_swap(self) -> bool:
        if not getattr(self, "blocks_to_swap", 0) or getattr(self, "offloader", None) is None:
            return False
        if getattr(self, "_paused_blocks_to_swap", None) is not None:
            return False
        for block_idx in list(self.offloader.futures):
            self.offloader._wait_blocks_move(block_idx, phase="pause")
        self.offloader.restore_blocks_to_device(
            getattr(self, self._block_swap_container_attr),
            self.offloader.device,
        )
        if self.offloader.cuda_available:
            torch.cuda.synchronize(self.offloader.device)
        self._paused_blocks_to_swap = self.blocks_to_swap
        self.blocks_to_swap = 0
        return True

    def resume_block_swap(self) -> bool:
        paused = getattr(self, "_paused_blocks_to_swap", None)
        if paused is None:
            return False
        self.blocks_to_swap = paused
        self._paused_blocks_to_swap = None
        self.prepare_block_swap_before_forward()
        return True

    def flush_block_swap_profile(self, blocking: bool = False) -> None:
        if getattr(self, "offloader", None) is not None:
            self.offloader.flush_profile_events(blocking=blocking)


class BlockSwapController:
    """Wrap-forward block-swap adapter around stock ``nn.ModuleList`` containers."""

    _default_container_attr: str = "blocks"
    _default_min_blocks: int = 0
    _default_family_label: str = "Model"
    _default_adapter_attr: str | None = None
    _default_direct_checkpoint_attr: str | None = None

    def __init__(
        self,
        model: nn.Module,
        blocks_to_swap: int,
        device: torch.device,
        *,
        container_attr: str | None = None,
        min_blocks: int | None = None,
        family_label: str | None = None,
        adapter_attr: str | None = None,
        direct_checkpoint_attr: str | None = None,
        offloader_cls: Callable[..., Any] = ModelOffloader,
        profile_jsonl: str | None = None,
        transfer_dtype: str | None = None,
        restore_mode: str | None = None,
        int8_restore_mode: str | None = None,
        int8_staging_ring: int | None = None,
        int8_dual_stream: bool | None = None,
        int8_restore_chunk_rows: int | None = None,
        int8_scope: str | None = None,
    ) -> None:
        resolved_container = container_attr or self._default_container_attr
        resolved_min = (
            self._default_min_blocks if min_blocks is None else int(min_blocks)
        )
        resolved_label = family_label or self._default_family_label
        resolved_adapter_attr = (
            self._default_adapter_attr if adapter_attr is None else adapter_attr
        )
        resolved_direct_attr = (
            self._default_direct_checkpoint_attr
            if direct_checkpoint_attr is None
            else direct_checkpoint_attr
        )

        blocks = getattr(model, resolved_container, None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(
                f"{resolved_label} block swap requires model.{resolved_container} "
                "to be a ModuleList"
            )
        max_blocks = len(blocks) - 2
        if blocks_to_swap < resolved_min or blocks_to_swap > max_blocks:
            raise ValueError(
                f"{resolved_label} blocks_to_swap must be between {resolved_min} and "
                f"{max_blocks}; got {blocks_to_swap}"
            )

        self.model = model
        self._container_attr = resolved_container
        self._direct_checkpoint_attr = resolved_direct_attr
        self.blocks = blocks
        if resolved_container != "blocks":
            setattr(self, resolved_container, blocks)
        self.block_indices = {id(block): index for index, block in enumerate(blocks)}
        self.layer_indices = self.block_indices
        self.offloader = offloader_cls(
            blocks,
            blocks_to_swap,
            device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=transfer_dtype,
            restore_mode=restore_mode,
            int8_restore_mode=int8_restore_mode,
            int8_staging_ring=int8_staging_ring,
            int8_dual_stream=int8_dual_stream,
            int8_restore_chunk_rows=int8_restore_chunk_rows,
            int8_scope=int8_scope,
        )
        self._original_checkpoint_func = getattr(
            model, "_gradient_checkpointing_func", None
        )
        self._original_enable_gradient_checkpointing = (
            model.enable_gradient_checkpointing
        )
        self.inner_forwards: list[Callable[..., Any]] = []
        self._original_forwards = self.inner_forwards

        model.blocks_to_swap = blocks_to_swap
        model.offloader = self.offloader
        model._block_swap_controller = self
        if resolved_adapter_attr:
            setattr(model, resolved_adapter_attr, self)
        self._bind_training_protocol()
        self._install_forward_dispatch()

    @property
    def _swap_blocks(self) -> Any:
        blocks = getattr(self, "blocks", None)
        if blocks is not None:
            return blocks
        return getattr(self, "layers", None)

    def _install_forward_dispatch(self) -> None:
        if bool(getattr(self.model, "gradient_checkpointing", False)):
            if self._original_checkpoint_func is None:
                raise RuntimeError(
                    f"{self._default_family_label} gradient checkpointing is enabled "
                    "without a checkpoint function"
                )
            self.model._gradient_checkpointing_func = self._checkpoint_with_swap

        for index, block in enumerate(self.blocks):
            original_forward = block.forward
            self.inner_forwards.append(original_forward)

            def forward_with_swap(
                _block: nn.Module,
                *args: Any,
                _index: int = index,
                **kwargs: Any,
            ) -> Any:
                # The outer checkpoint wrapper drives the initial forward;
                # backward hooks drive its non-reentrant recomputation.
                checkpointed = bool(
                    torch.is_grad_enabled()
                    and getattr(self.model, "gradient_checkpointing", False)
                )
                forward = self.inner_forwards[_index]
                if checkpointed or not getattr(self.model, "blocks_to_swap", 0):
                    return forward(*args, **kwargs)
                return self._run_forward(_index, forward, *args, **kwargs)

            block.forward = MethodType(forward_with_swap, block)

    def _checkpoint_with_swap(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        index = self.block_indices.get(id(function))
        direct_attr = getattr(self, "_direct_checkpoint_attr", None)
        is_direct = bool(direct_attr and hasattr(function, direct_attr))
        if index is None or not getattr(self.model, "blocks_to_swap", 0):
            if index is not None and is_direct:
                return function(*args, **kwargs)
            return self._original_checkpoint_func(function, *args, **kwargs)
        self.offloader.wait_for_block(index)
        if is_direct:
            output = function(*args, **kwargs)
        else:
            output = self._original_checkpoint_func(function, *args, **kwargs)
        self.offloader.submit_move_blocks(self._swap_blocks, index)
        return output

    def _run_forward(
        self,
        index: int,
        forward: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.offloader.wait_for_block(index)
        output = forward(*args, **kwargs)
        self.offloader.submit_move_blocks(self._swap_blocks, index)
        return output

    def _bind_training_protocol(self) -> None:
        for name in (
            "enable_gradient_checkpointing",
            "move_to_device_except_swap_blocks",
            "prepare_block_swap_before_forward",
            "switch_block_swap_for_inference",
            "switch_block_swap_for_training",
            "pause_block_swap",
            "resume_block_swap",
            "flush_block_swap_profile",
        ):
            setattr(self.model, name, getattr(self, name))

    def enable_gradient_checkpointing(self, *args: Any, **kwargs: Any) -> None:
        self._original_enable_gradient_checkpointing(*args, **kwargs)
        self._original_checkpoint_func = self.model._gradient_checkpointing_func
        self.model._gradient_checkpointing_func = self._checkpoint_with_swap

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if not getattr(self.model, "blocks_to_swap", 0):
            self.model.to(device)
            return
        container_attr = getattr(self, "_container_attr", self._default_container_attr)
        saved = getattr(self.model, container_attr)
        setattr(self.model, container_attr, None)
        try:
            self.model.to(device)
        finally:
            setattr(self.model, container_attr, saved)

    def prepare_block_swap_before_forward(self, free_cache: bool = True) -> None:
        if getattr(self.model, "blocks_to_swap", 0):
            self.offloader.prepare_block_devices_before_forward(
                self._swap_blocks, free_cache=free_cache
            )

    def switch_block_swap_for_inference(self) -> None:
        if getattr(self.model, "blocks_to_swap", 0):
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if getattr(self.model, "blocks_to_swap", 0):
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def pause_block_swap(self) -> bool:
        if not getattr(self.model, "blocks_to_swap", 0):
            return False
        if getattr(self.model, "_paused_blocks_to_swap", None) is not None:
            return False
        for index in list(self.offloader.futures):
            self.offloader._wait_blocks_move(index, phase="pause")
        self.offloader.restore_blocks_to_device(self._swap_blocks, self.offloader.device)
        if self.offloader.cuda_available:
            torch.cuda.synchronize(self.offloader.device)
        self.model._paused_blocks_to_swap = self.model.blocks_to_swap
        self.model.blocks_to_swap = 0
        return True

    def resume_block_swap(self) -> bool:
        blocks_to_swap = getattr(self.model, "_paused_blocks_to_swap", None)
        if blocks_to_swap is None:
            return False
        self.model.blocks_to_swap = blocks_to_swap
        self.model._paused_blocks_to_swap = None
        self.prepare_block_swap_before_forward()
        return True

    def flush_block_swap_profile(self, blocking: bool = False) -> None:
        self.offloader.flush_profile_events(blocking=blocking)
