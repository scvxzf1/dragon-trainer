"""Shared low-memory orchestration for standalone family inference.

Family runners own model-specific loading, conditioning, sampling, and decode
functions.  This module only owns the lifetime boundary between those stages:
text encoding finishes before DiT loading, and DiT sampling finishes before
VAE decoding.  Keeping that boundary here prevents each family from growing a
slightly different cleanup and OOM policy.
"""

from __future__ import annotations

import gc
import logging
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

import torch

from library.runtime.device import clean_memory_on_device

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_OOM_RETRIES = 1


def is_cuda_oom(error: BaseException) -> bool:
    """Return whether *error* is a retryable CUDA allocator failure."""
    if not isinstance(error, torch.OutOfMemoryError):
        message = str(error).lower()
        if "out of memory" not in message or "cuda" not in message:
            return False
    message = str(error).lower()
    host_markers = (
        "cpu",
        "host",
        "defaultcpuallocator",
        "can't allocate memory",
        "illegal memory access",
        "timeout",
        "nan",
    )
    return not any(marker in message for marker in host_markers)


def _release_resource(resource: Any) -> None:
    """Move stage-owned modules off accelerator before dropping references."""
    if resource is None:
        return
    if isinstance(resource, (tuple, list, set)):
        for item in resource:
            _release_resource(item)
        return
    if isinstance(resource, dict):
        for item in resource.values():
            _release_resource(item)
        return
    if isinstance(resource, torch.nn.Module):
        resource.to(device="cpu")


class FamilyInferenceRunner:
    """Run family stages with one deterministic cleanup and retry policy."""

    def __init__(
        self,
        family: str,
        device: torch.device,
        *,
        cleanup: Callable[[torch.device], None] = clean_memory_on_device,
    ) -> None:
        self.family = str(family)
        self.device = torch.device(device)
        self._cleanup = cleanup

    def release(
        self,
        resource: Any = None,
        *,
        error: BaseException | None = None,
        clear: bool = True,
    ) -> None:
        """Release a stage resource and clear allocator state."""
        primary = error
        try:
            _release_resource(resource)
        except BaseException as cleanup_error:
            if primary is None:
                primary = cleanup_error
            else:
                primary.add_note(
                    "inference stage resource cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if clear:
            try:
                gc.collect()
                self._cleanup(self.device)
            except BaseException as cleanup_error:
                if primary is None:
                    primary = cleanup_error
                else:
                    primary.add_note(
                        "inference allocator cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        if error is None and primary is not None:
            raise primary

    def run_with_oom_retry(
        self,
        operation: Callable[[int], T],
        *,
        max_retries: int = 0,
        on_oom: Callable[[int, BaseException], None] | None = None,
        release_on_oom: bool = True,
    ) -> T:
        """Run an operation, retrying only genuine CUDA OOM failures."""
        for attempt in range(max_retries + 1):
            try:
                return operation(attempt)
            except BaseException as error:
                if not is_cuda_oom(error) or attempt >= max_retries:
                    raise
                logger.warning(
                    "%s inference CUDA OOM on attempt %s; retrying",
                    self.family,
                    attempt + 1,
                )
                if release_on_oom:
                    self.release(error=error)
                if on_oom is not None:
                    on_oom(attempt, error)
        raise AssertionError("unreachable inference retry loop")

    def run_stage(
        self,
        name: str,
        loader: Callable[[], R],
        operation: Callable[[R], T],
        *,
        max_oom_retries: int = DEFAULT_OOM_RETRIES,
        on_oom: Callable[[int, BaseException], None] | None = None,
        clear_after: bool = True,
    ) -> T:
        """Load, execute, and release one TE/DiT/VAE stage."""

        def attempt(_: int) -> T:
            resource = None
            stage_error: BaseException | None = None
            try:
                resource = loader()
                return operation(resource)
            except BaseException as error:
                stage_error = error
                raise
            finally:
                try:
                    self.release(resource, error=stage_error, clear=clear_after)
                finally:
                    resource = None
                    stage_error = None

        return self.run_with_oom_retry(
            attempt,
            max_retries=max_oom_retries,
            on_oom=on_oom,
            release_on_oom=False,
        )

    def run_text(
        self,
        loader: Callable[[], R],
        operation: Callable[[R], T] | None = None,
        *,
        max_oom_retries: int = DEFAULT_OOM_RETRIES,
        on_oom: Callable[[int, BaseException], None] | None = None,
    ) -> T:
        """Execute a text stage and release its TE on success or failure.

        ``operation=None`` is retained for callers whose loader already owns
        local cleanup; allocator cleanup still runs on both paths.
        """
        if operation is not None:
            return self.run_stage(
                "text",
                loader,
                operation,
                max_oom_retries=max_oom_retries,
                on_oom=on_oom,
            )
        stage_error: BaseException | None = None
        try:
            return loader()
        except BaseException as error:
            stage_error = error
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            self.release(error=stage_error)
