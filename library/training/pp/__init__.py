"""Anima pipeline-parallel v0 runtime.

Public surface, in the order a caller needs it:

``contract``      -- normalized runtime config + fail-closed validation.
``layout``        -- the 4D/5D latent contract and the single-token-bucket rule.
``placement``     -- non-destructive PP=2 block ownership for one rank.
``schedule``      -- ``Schedule1F1B`` wiring, boundary metadata, last-stage loss.
``transactions``  -- one fp16 scale and one skip decision for the whole pipeline.
``runtime``       -- :func:`build_anima_pipeline_runtime` / ``AnimaPipelineRuntime``.

Nothing in this package falls back to DDP: every "PP is unavailable" case raises
:class:`PipelineParallelError` with a stable ``pp_*`` code from
:mod:`library.training.pp.errors`.

Lazy public surface (PEP 562)
-----------------------------
The re-exports below are resolved on first access instead of at package import
time.  Two reasons:

1. **Import graph.**  An eager ``__init__`` made every entry point pay for all
   eleven submodules: ``import library.training.pp.gate`` (a two-function module
   used by the CLI and the compat matrix) dragged in ``checkpoint``, ``runtime``,
   ``schedule`` and, transitively, ``torch.distributed.pipelining``.  It also
   *amplified* dependency mistakes -- one bad import anywhere in the package broke
   every entry point.  That is exactly how the ``library`` -> ``scripts`` reverse
   dependency in ``numerics`` used to surface as four ``scripts.*`` modules in
   ``sys.modules`` for a single ``import library.training.pp``.
2. **Blast radius.**  With lazy resolution each entry point's import graph is its
   own, so a future submodule cannot break the others at import time.

``from library.training.pp import X`` and ``pp.X`` keep working unchanged, as do
``pp.<submodule>`` and ``dir()``.  ``from library.training.pp import *`` resolves
every name in ``__all__`` (and therefore imports every submodule) -- that is the
one case that still pays the full cost, and no production call site does it.

Guard: ``tests/test_pp_import_boundary.py`` asserts that no module under
``library/`` imports from ``scripts/``, and that each public entry point can be
imported in a fresh interpreter without pulling a single ``scripts.*`` module.
"""

from __future__ import annotations

from importlib import import_module

#: ``public name -> defining submodule``.  Grouped by module, then sorted, so the
#: file stays reviewable; ``__all__`` is derived from it (one source of truth).
_EXPORTS: dict[str, str] = {
    # .checkpoint
    "PpCheckpointResult": "checkpoint",
    "apply_stage_checkpoint": "checkpoint",
    "load_distributed_checkpoint": "checkpoint",
    "save_distributed_checkpoint": "checkpoint",
    # .checkpoint_artifacts
    "PpResumeState": "checkpoint_artifacts",
    "create_run_root": "checkpoint_artifacts",
    "read_manifest": "checkpoint_artifacts",
    "read_stage_checkpoint": "checkpoint_artifacts",
    "save_stage_checkpoint": "checkpoint_artifacts",
    # .checkpoint_state
    "PP_CHECKPOINT_FORMAT": "checkpoint_state",
    "PP_CHECKPOINT_VERSION": "checkpoint_state",
    "PpStageSnapshot": "checkpoint_state",
    "assert_snapshot_digests_match": "checkpoint_state",
    "assert_snapshot_roundtrip": "checkpoint_state",
    "capture_stage_snapshot": "checkpoint_state",
    "compare_snapshots": "checkpoint_state",
    "content_hash_file": "checkpoint_state",
    "content_hash_object": "checkpoint_state",
    "content_hash_tensor": "checkpoint_state",
    "content_hash_tensors": "checkpoint_state",
    "stage_adapter_state": "checkpoint_state",
    "topology_digest": "checkpoint_state",
    "validate_identity": "checkpoint_state",
    # .contract
    "PP_CONDITIONING_MODES": "contract",
    "PP_SUPPORTED_MIXED_PRECISION": "contract",
    "PP_SUPPORTED_SCHEDULES": "contract",
    "PipelineRuntimeConfig": "contract",
    "PpComputePolicy": "contract",
    "assert_compute_policy_agreement": "contract",
    "quantization_markers": "contract",
    "require_enabled": "contract",
    "require_pipelining_api": "contract",
    "validate_distributed_backend": "contract",
    "validate_family_config": "contract",
    "validate_master_weight_dtype": "contract",
    "validate_mixed_precision": "contract",
    "validate_no_quantization": "contract",
    "validate_runtime_config": "contract",
    # .errors
    "PipelineParallelError": "errors",
    # .layout
    "ADALN_CHUNKS": "layout",
    "TIME_AXIS": "layout",
    "adaln_lora_shape": "layout",
    "assert_uniform_token_bucket": "layout",
    "boundary_hidden_shape": "layout",
    "latent_to_model_5d": "layout",
    "model_pred_to_loss_4d": "layout",
    "prediction_shape": "layout",
    "split_batch_shape": "layout",
    "timestep_embedding_shape": "layout",
    "token_count": "layout",
    "token_family": "layout",
    "validate_latent_5d_shape": "layout",
    # .placement
    "AnimaStageLayout": "placement",
    "adapter_block_index": "placement",
    "assert_stage_capabilities": "placement",
    "build_anima_stage_layout": "placement",
    "build_anima_stage_module": "placement",
    "layout_from_plan": "placement",
    "release_foreign_blocks": "placement",
    "stage_adapter_modules": "placement",
    # .runtime
    "AnimaPipelineRuntime": "runtime",
    "PpStepResult": "runtime",
    "assert_runtime_capabilities": "runtime",
    "build_anima_pipeline_runtime": "runtime",
    # .schedule
    "boundary_metas": "schedule",
    "build_anima_schedule": "schedule",
    "build_pipeline_stage": "schedule",
    "make_last_stage_loss_fn": "schedule",
    "run_pipeline_step": "schedule",
    # .stages
    "AnimaFirstStage": "stages",
    "AnimaLastStage": "stages",
    "CONDITIONING_LOCAL": "stages",
    "CONDITIONING_WIRE": "stages",
    "FrozenBoundaryPipelineStage": "stages",
    "WIRE_FROZEN_INDICES": "stages",
    "assert_frozen_boundary_agreement": "stages",
    "frozen_boundary_indices": "stages",
    # .transactions
    "PpFp16Transaction": "transactions",
    "PpStepOutcome": "transactions",
    "agree_overflow": "transactions",
    "broadcast_scale": "transactions",
    "divide_gradients": "transactions",
    "gradients_have_nonfinite": "transactions",
    "make_clip_gradients": "transactions",
    "verify_skip_unanimity": "transactions",
}

#: Submodules reachable as ``library.training.pp.<name>``.  The eager ``__init__``
#: got these for free from ``from .<module> import ...``; the lazy version has to
#: import them on demand to keep ``pp.errors`` / ``pp.gate`` working after a bare
#: ``import library.training.pp``.
_SUBMODULES: frozenset[str] = frozenset(
    {
        "checkpoint",
        "checkpoint_artifacts",
        "checkpoint_state",
        "conditioning",
        "contract",
        "errors",
        "gate",
        "layout",
        "numerics",
        "placement",
        "runtime",
        "schedule",
        "stages",
        "transactions",
    }
)

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resolve a public name (or a submodule) on first access, then cache it."""

    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(import_module(f".{module}", __name__), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
