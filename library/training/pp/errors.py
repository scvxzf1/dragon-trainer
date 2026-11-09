"""Named, fail-closed errors for the Anima pipeline-parallel runtime.

Why a dedicated module
----------------------
``torch.distributed.pipelining`` reports most misconfiguration as a bare
``RuntimeError``/``ValueError`` from deep inside the schedule, and the trainer
must never answer "pipeline parallel is unavailable" with a silent DDP
fallback.  Every rejection raised by :mod:`library.training.pp` therefore
carries a stable ``pp_*`` code, so bootstrap / WebUI preflight / daemon can fail
closed on the code instead of matching on message text.

Codes are append-only: never rename one, because they surface in preflight
output, checkpoint contracts and test assertions.
"""

from __future__ import annotations

#: ``torch.distributed.pipelining`` is missing or too old for the runtime API.
CODE_RUNTIME_UNAVAILABLE = "pp_runtime_unavailable"
#: ``pipeline_parallel`` was requested but the runtime was built with it off.
CODE_NOT_ENABLED = "pp_not_enabled"
#: The family-level PP contract rejected the requested configuration.
CODE_CONFIG_INVALID = "pp_config_invalid"
#: ``world_size != stages``: the PP topology must cover every rank exactly once.
CODE_WORLD_SIZE_MISMATCH = "pp_world_size_mismatch"
#: The whole ``PipelineScheduleSingle`` family needs it too, GPipe included.
CODE_MICROBATCHES_BELOW_STAGES = "pp_microbatches_below_stages"
#: Microbatch count is outside the family contract's bounds.
CODE_MICROBATCHES_INVALID = "pp_microbatches_invalid"
#: Schedule name is not one the PP v0 runtime implements.
CODE_SCHEDULE_UNSUPPORTED = "pp_schedule_unsupported"
#: Conditioning transport mode is not one the PP v0 runtime implements.
CODE_CONDITIONING_UNSUPPORTED = "pp_conditioning_unsupported"
#: PP v0 is plain fp16 + ``GradScaler`` only.
CODE_MIXED_PRECISION_UNSUPPORTED = "pp_mixed_precision_unsupported"
#: fp16 without a scaler would silently corrupt fp16 gradients.
CODE_GRAD_SCALER_MISSING = "pp_grad_scaler_missing"
#: Quantized bases (NF4 / fp8 / ConvRot) are out of scope for PP v0.
CODE_QUANTIZATION_UNSUPPORTED = "pp_quantization_unsupported"
#: The model cannot be placed on a stage as-is (missing submodules, swap, ...).
CODE_STAGE_CONTRACT = "pp_stage_contract_violation"
#: Loss must be computed on the last stage only.
CODE_LOSS_STAGE = "pp_loss_stage_violation"
#: 4D/5D latent contract violation at the PP boundary.
CODE_SHAPE_CONTRACT = "pp_shape_contract_violation"
#: One PP step may only carry a single token bucket (fixed-shape wire buffers).
CODE_TOKEN_BUCKET_MISMATCH = "pp_token_bucket_mismatch"
#: The two stages disagree about which boundary tensors are frozen.
CODE_FROZEN_BOUNDARY_MISMATCH = "pp_frozen_boundary_mismatch"
#: The two ranks reached different GradScaler skip decisions for the same step.
CODE_SCALER_DECISION_ASYMMETRY = "pp_scaler_decision_asymmetry"
#: A stage derived a different compute dtype than the pipeline decided once.
CODE_COMPUTE_POLICY_MISMATCH = "pp_compute_policy_mismatch"
#: CUDA PP needs a multi-backend process group (``cpu:gloo,cuda:nccl``).
CODE_BACKEND_MULTIBACKEND_REQUIRED = "pp_backend_multibackend_required"
#: fp16 PP needs fp32 master weights: ``GradScaler`` refuses fp16 gradients.
CODE_FP16_MASTER_WEIGHTS_REQUIRED = "pp_fp16_master_weights_required"
#: The accelerate data-parallel trainer path cannot serve ``pipeline_parallel``.
CODE_TRAINER_PATH = "pp_trainer_path_unavailable"
#: ``accelerator.save_state`` would write every rank's state into one directory.
CODE_ACCELERATE_CHECKPOINT = "pp_accelerate_checkpoint_unsupported"
#: A PP checkpoint artifact is missing, unreadable or was never published.
CODE_CHECKPOINT_INCOMPLETE = "pp_checkpoint_incomplete"
#: The saved topology (ranges/stages/rank count/schedule) is not this one.
CODE_CHECKPOINT_TOPOLOGY_MISMATCH = "pp_checkpoint_topology_mismatch"
#: A content hash (run identity or artifact bytes) does not match the manifest.
CODE_CHECKPOINT_IDENTITY_MISMATCH = "pp_checkpoint_identity_mismatch"
#: The stages saved different step counters; the run cannot resume as one.
CODE_CHECKPOINT_STEP_MISMATCH = "pp_checkpoint_step_mismatch"
#: The checkpoint's adapter/optimizer state does not fit this stage.
CODE_CHECKPOINT_STATE_MISMATCH = "pp_checkpoint_state_mismatch"
#: The target run root already exists; PP checkpoints never overwrite a run.
CODE_CHECKPOINT_RUN_ROOT_NOT_NEW = "pp_checkpoint_run_root_not_new"


class PipelineParallelError(RuntimeError):
    """Fail-closed PP rejection carrying a stable machine-readable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.detail = message


__all__ = [
    "CODE_ACCELERATE_CHECKPOINT",
    "CODE_BACKEND_MULTIBACKEND_REQUIRED",
    "CODE_CHECKPOINT_IDENTITY_MISMATCH",
    "CODE_CHECKPOINT_INCOMPLETE",
    "CODE_CHECKPOINT_RUN_ROOT_NOT_NEW",
    "CODE_CHECKPOINT_STATE_MISMATCH",
    "CODE_CHECKPOINT_STEP_MISMATCH",
    "CODE_CHECKPOINT_TOPOLOGY_MISMATCH",
    "CODE_COMPUTE_POLICY_MISMATCH",
    "CODE_CONDITIONING_UNSUPPORTED",
    "CODE_CONFIG_INVALID",
    "CODE_FP16_MASTER_WEIGHTS_REQUIRED",
    "CODE_FROZEN_BOUNDARY_MISMATCH",
    "CODE_GRAD_SCALER_MISSING",
    "CODE_LOSS_STAGE",
    "CODE_MICROBATCHES_BELOW_STAGES",
    "CODE_MICROBATCHES_INVALID",
    "CODE_MIXED_PRECISION_UNSUPPORTED",
    "CODE_NOT_ENABLED",
    "CODE_QUANTIZATION_UNSUPPORTED",
    "CODE_RUNTIME_UNAVAILABLE",
    "CODE_SCALER_DECISION_ASYMMETRY",
    "CODE_SCHEDULE_UNSUPPORTED",
    "CODE_SHAPE_CONTRACT",
    "CODE_STAGE_CONTRACT",
    "CODE_TOKEN_BUCKET_MISMATCH",
    "CODE_TRAINER_PATH",
    "CODE_WORLD_SIZE_MISMATCH",
    "PipelineParallelError",
]
