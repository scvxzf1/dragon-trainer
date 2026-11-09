"""Shared training compatibility checks for optimization flags.

The helpers here are intentionally pure: they inspect a mapping/object and
return errors, warnings, and suggested mutations. CLI and Web preflight can
consume the same rules without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from library.models.family_registry import (
    get_model_family_spec,
    normalize_registered_family,
)
from library.models.pipeline_parallel import (
    PipelineParallelConfig,
    validate_pipeline_parallel_config,
)


VALID_SELECTIVE_CHECKPOINTS = {
    "off",
    "adapter_aware",
    "every_other",
    "mlp_only",
    "mlp_layer1_only",
    "peak_blocks_adapter_aware",
    "peak_blocks_mlp",
    "peak_blocks_mlp_layer1",
}
# Keep this whitelist in sync with
# library/models/krea2_raw/attention_backend.py::normalize_krea2_attention_mode:
# sdpa/cudnn/cudnn_sdpa are normalized to torch there.
KREA2_ATTN_MODES = {"", "torch", "flash", "sdpa", "cudnn", "cudnn_sdpa"}
KREA2_SELECTIVE_CHECKPOINTS = {"off", "every_other"}
KREA2_UNSUPPORTED_ADAPTER_FLAGS = (
    "use_ip_adapter",
    "use_easycontrol",
    "use_byg",
    "use_lokr",
    "use_loha",
    "use_glora",
    "use_vera",
    "use_ortho",
    "use_chimera_hydra",
    "use_timestep_mask",
    "add_reft",
    "train_llm_adapter",
)


# ---------------------------------------------------------------------------
# Unverified-combination policy
#
# Classification source: docs/findings/compat_unverified_audit_20261109.md
# (52 ``out.error`` sites -> 77 codes; the two always-false sites are deleted).
#
# * ``strict``: every audited site is an error again (origin/dev behavior).
# * ``warn``  : only the audited C1 codes (implementation exists and takes
#               effect, real-machine acceptance missing) degrade to warnings.
# * ``allow`` : C2/C3 codes (no code path in the family, or a recipe/VRAM
#               policy) degrade too; A/B/mixed input-validation codes never do.
# ---------------------------------------------------------------------------

COMPAT_UNVERIFIED_POLICY_KEY = "compat_unverified_policy"
COMPAT_UNVERIFIED_POLICIES = ("strict", "warn", "allow")
DEFAULT_COMPAT_UNVERIFIED_POLICY = "warn"
UNVERIFIED_MARKER = "[unverified 未验证]"
UNWIRED_NOTE = (
    "该键在本族未接线/会被忽略，或偏离官方配方/显存预算；"
    "allow 档放行后可能在下游 raise 或被静默忽略。"
)

# C1: the implementation path exists and consumes the flag; only the
# real-machine/numeric acceptance is missing.
UNVERIFIED_C1_CODES = frozenset({
    "qwen_image_2_1_mixed_precision",
    "qwen_image_2_1_masked_loss",
    "qwen_image_2_1_loss_type",
    "qwen_image_2_1_weighting_scheme",
    "z_image_loss_type",
    "z_image_flow_shift",
    "z_image_weighting_scheme",
})

# Per-code C1 evidence: why relaxing to a warning is safe (the flag is read).
UNVERIFIED_C1_REASONS = {
    "qwen_image_2_1_mixed_precision": (
        "the qwen torch/sdpa path has no fp16 guard (only flash requires bf16); "
        "qwen fp16 has no accepted real-machine run"
    ),
    "qwen_image_2_1_masked_loss": (
        "the shared masked flow-matching loss applies to qwen's [B,64,H,W] "
        "target; only the data/quality acceptance is missing"
    ),
    "qwen_image_2_1_loss_type": (
        "library/training/losses.py applies l1/huber on the qwen path; only the "
        "official L2 recipe has been accepted"
    ),
    "qwen_image_2_1_weighting_scheme": (
        "library/training/noise_target.py consumes args.weighting_scheme on the "
        "qwen path"
    ),
    "z_image_loss_type": (
        "library/training/losses.py applies l1/huber on the z_image path; only "
        "the official L2 recipe has been accepted"
    ),
    "z_image_flow_shift": (
        "library/models/z_image/family.py feeds args.discrete_flow_shift into "
        "shifted_uniform_sigmas, so non-6.0 values take effect"
    ),
    "z_image_weighting_scheme": (
        "library/training/noise_target.py consumes args.weighting_scheme on the "
        "z_image path"
    ),
}

# C2: no code path in the family; relaxing either raises downstream or makes
# the flag a silent no-op.
UNVERIFIED_C2_CODES = frozenset({
    "qwen_image_2_1_base_compute",
    "qwen_image_2_1_adaptive_precision",
    "qwen_image_2_1_compile_backend",
    "qwen_image_2_1_compile_mode",
    "qwen_image_2_1_auto_block_swap",
    "qwen_image_2_1_selective_checkpoint",
    "qwen_image_2_1_cpu_offload_checkpointing",
    "qwen_image_2_1_network_train_unet_only",
    "qwen_image_2_1_weighted_captions",
    "qwen_image_2_1_caption_dropout_rate",
    "qwen_image_2_1_timestep_sampling",
    "qwen_image_2_1_preview_text_cache",
    "qwen_image_2_1_layer_start",
    "qwen_image_2_1_t_min",
    "qwen_fused_projections_family",
    "qwen_fused_projections_plain_lora",
    "qwen_fused_projections_mixed_precision",
    "qwen_fused_projections_base_compute",
    "qwen_fused_projections_block_swap_transfer_dtype",
    "qwen_fused_projections_gradient_checkpointing",
    "qwen_fused_projections_selective_checkpoint",
    "qwen_fused_projections_cpu_offload_checkpointing",
    "pipeline_parallel_runtime_unavailable",
    "krea2_compile_inductor_mode",
    "krea2_selective_checkpoint",
    "krea2_v100_flash_stability",
    "z_image_base_compute",
    "z_image_torch_compile",
    "z_image_selective_checkpoint",
    "z_image_checkpoint_offload",
    "z_image_layer_range",
    "z_image_weighted_captions",
    "z_image_timestep_range",
    "z_image_training_sampler",
    "z_image_v100_flash_stability",
    "z_image_timestep_sampling",
    "z_image_caption_dropout",
    "family_plain_lora_only",
})

# C3: runnable, but departs from the official recipe or the VRAM policy.
UNVERIFIED_C3_CODES = frozenset({
    "qwen_image_2_1_block_swap_checkpoint",
    "qwen_nf4_compile_experiment",
    "z_image_bf16_only",
    "block_swap_cpu_offload",
    "block_swap_unsloth_offload",
})

# A (semantic conflict / structural), B (input validation) and B+C2 mixed.
# These stay hard errors under every policy; ``allow`` must not turn a typo or
# a silently overridden switch into a warning.
HARD_BLOCK_CODES = frozenset({
    "selective_full_gradient_checkpointing",
    "selective_cpu_offload",
    "selective_unsloth_offload",
    "unsloth_cpu_offload",
    "convrot_block_swap_int8_mutex",
    "block_swap_soft_tokens",
    "block_swap_functional_loss",
    "qwen_image_2_1_attn_mode",
    "qwen_image_2_1_blocks_to_swap",
    "invalid_qwen_saved_projection_blocks",
    "invalid_qwen_projection_budget_mib",
    "invalid_qwen_fused_projections",
    "invalid_selective_checkpoint",
    "invalid_model_family",
    "pipeline_parallel_config",
    "negative_blocks_to_swap",
    "invalid_base_compute",
    "krea2_invalid_attn_mode",
    "z_image_attention_mode",
    "z_image_block_swap_range",
    "auto_block_swap_contract",
    "adaptive_precision_contract",
    "sample_vae_decode_contract",
    "qwen_preview_memory_contract",
    "qwen_saved_projection_contract",
})

# Codes that ``allow`` may relax (C2 + C3). Unknown/new codes are never relaxed.
RELAXABLE_UNWIRED_CODES = UNVERIFIED_C2_CODES | UNVERIFIED_C3_CODES

_POLICY_ALIASES = {
    "strict": "strict",
    "error": "strict",
    "off": "strict",
    "none": "strict",
    "warn": "warn",
    "warning": "warn",
    "default": "warn",
    "allow": "allow",
    "permissive": "allow",
}


def resolve_unverified_policy(
    config: Mapping[str, Any] | object | None = None,
    override: str | None = None,
) -> tuple[str, str | None]:
    """Resolve the unverified policy from an explicit override or the config.

    Returns ``(policy, invalid_message)``. An explicit keyword wins over the
    ``compat_unverified_policy`` config key; an unknown value falls back to the
    default and reports a message the caller turns into a hard error.
    """

    raw = override if override is not None else _get(config, COMPAT_UNVERIFIED_POLICY_KEY, None)
    if raw is None:
        return DEFAULT_COMPAT_UNVERIFIED_POLICY, None
    text = str(raw).strip().lower()
    if not text:
        return DEFAULT_COMPAT_UNVERIFIED_POLICY, None
    policy = _POLICY_ALIASES.get(text)
    if policy is None:
        return DEFAULT_COMPAT_UNVERIFIED_POLICY, (
            f"unknown {COMPAT_UNVERIFIED_POLICY_KEY}={raw!r}; expected "
            + " | ".join(COMPAT_UNVERIFIED_POLICIES)
            + f" (using {DEFAULT_COMPAT_UNVERIFIED_POLICY!r})"
        )
    return policy, None


@dataclass(frozen=True)
class TrainingCompatIssue:
    code: str
    key: str
    message: str


@dataclass(frozen=True)
class TrainingCompatMutation:
    code: str
    key: str
    value: Any
    message: str


@dataclass(frozen=True)
class TrainingCompatResult:
    errors: tuple[TrainingCompatIssue, ...]
    warnings: tuple[TrainingCompatIssue, ...]
    mutations: tuple[TrainingCompatMutation, ...]
    unverified_policy: str = DEFAULT_COMPAT_UNVERIFIED_POLICY

    @property
    def ok(self) -> bool:
        return not self.errors


class _CompatBuilder:
    def __init__(self, unverified_policy: str = DEFAULT_COMPAT_UNVERIFIED_POLICY) -> None:
        self.unverified_policy = unverified_policy
        self.errors: list[TrainingCompatIssue] = []
        self.warnings: list[TrainingCompatIssue] = []
        self.mutations: list[TrainingCompatMutation] = []

    def error(self, code: str, key: str, message: str) -> None:
        # ``allow`` is the only policy that relaxes audited C2/C3 sites; A/B
        # codes and any code outside the audited tables stay hard errors.
        if self.unverified_policy == "allow" and code in RELAXABLE_UNWIRED_CODES:
            self.warnings.append(
                TrainingCompatIssue(
                    code,
                    key,
                    f"{UNVERIFIED_MARKER}[allow] {message} {UNWIRED_NOTE}",
                )
            )
            return
        self.errors.append(TrainingCompatIssue(code, key, message))

    def unverified(self, code: str, key: str, message: str) -> None:
        """Report an audited C1 site: unvalidated but effective implementation.

        ``strict`` keeps the historical error and message verbatim; ``warn``
        (default) and ``allow`` report a warning carrying the 未验证 marker and
        the per-code reason.
        """

        reason = UNVERIFIED_C1_REASONS[code]
        if self.unverified_policy == "strict":
            self.errors.append(TrainingCompatIssue(code, key, message))
            return
        self.warnings.append(
            TrainingCompatIssue(
                code,
                key,
                f"{UNVERIFIED_MARKER} {message} Reason: {reason}. "
                "Downgraded to a warning by "
                f"{COMPAT_UNVERIFIED_POLICY_KEY}={self.unverified_policy}; set "
                f"{COMPAT_UNVERIFIED_POLICY_KEY}=strict to reject this combination.",
            )
        )

    def warning(self, code: str, key: str, message: str) -> None:
        self.warnings.append(TrainingCompatIssue(code, key, message))

    def mutate(self, code: str, key: str, value: Any, message: str) -> None:
        self.mutations.append(TrainingCompatMutation(code, key, value, message))

    def build(self) -> TrainingCompatResult:
        return TrainingCompatResult(
            errors=tuple(self.errors),
            warnings=tuple(self.warnings),
            mutations=tuple(self.mutations),
            unverified_policy=self.unverified_policy,
        )


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _with_network_args(config: Mapping[str, Any] | object) -> Mapping[str, Any] | object:
    """Overlay network_args using the same precedence as training bootstrap."""
    if isinstance(config, Mapping):
        values = dict(config)
    else:
        try:
            values = vars(config).copy()
        except TypeError:
            return config
    arguments = values.get("network_args")
    if not isinstance(arguments, (list, tuple)):
        return values
    for argument in arguments:
        if not isinstance(argument, str) or "=" not in argument:
            continue
        key, value = argument.split("=", 1)
        if key not in {"model_family", "network_module"}:
            values[key] = value
    return values


def _bool_value(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "none", ""}:
        return False
    return fallback


def _int_value(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_value(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _nested_caption_dropout_enabled(config: Mapping[str, Any] | object) -> bool:
    if not isinstance(config, Mapping):
        return False
    datasets = config.get("datasets")
    if not isinstance(datasets, (list, tuple)):
        return False
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        if _float_value(dataset.get("caption_dropout_rate"), 0.0) > 0:
            return True
        subsets = dataset.get("subsets")
        if not isinstance(subsets, (list, tuple)):
            continue
        if any(
            isinstance(subset, Mapping)
            and _float_value(subset.get("caption_dropout_rate"), 0.0) > 0
            for subset in subsets
        ):
            return True
    return False


def _check_qwen_image_2_1_contract(out: _CompatBuilder, config, *,
                                   selective_checkpoint: str, blocks_to_swap: int,
                                   torch_compile: bool, cpu_offload_checkpointing: bool,
                                   unsloth_offload_checkpointing: bool) -> None:
    def forbid(flag: str, active: bool, message: str) -> None:
        if active:
            out.error(f"qwen_image_2_1_{flag}", flag, message)

    def unverified(flag: str, active: bool, message: str) -> None:
        # Audited C1: the qwen path consumes the flag; only real-machine
        # acceptance is missing. Relaxed by compat_unverified_policy.
        if active:
            out.unverified(f"qwen_image_2_1_{flag}", flag, message)

    task = str(_get(config, "qwen_image_2_1_task", "t2i") or "t2i").strip().lower()
    family = "Qwen Image 2.1 Edit adapter" if task == "edit" else "Qwen Image 2.1 T2I adapter"
    unverified("mixed_precision", str(_get(config, "mixed_precision", "bf16") or "bf16").lower() != "bf16",
               f"{family} requires mixed_precision=bf16")
    forbid("base_compute", str(_get(config, "base_compute", "bf16") or "bf16").lower() != "bf16",
           f"{family} has no runtime NF4/ConvRot base quantization; keep "
           "base_compute=bf16 (pre-quantized mixed-NF4 weights load under bf16)")
    forbid("adaptive_precision", str(_get(config, "adaptive_precision", "off") or "off").lower() != "off",
           f"{family} does not support adaptive precision")
    mode = str(_get(config, "attn_mode", "torch") or "torch").lower()
    forbid("attn_mode", mode not in {"torch", "sdpa", "native", "flash"} or _bool_value(_get(config, "xformers")),
           f"{family} supports torch, sdpa and native (sdpa/native are torch "
           "aliases) plus bf16 FlashAttention varlen")
    forbid("blocks_to_swap", blocks_to_swap > 30,
           f"{family} blocks_to_swap must be between 0 and 30")
    forbid("block_swap_checkpoint", blocks_to_swap > 0 and not _bool_value(_get(config, "gradient_checkpointing")),
           f"{family} block swap requires full gradient_checkpointing")
    forbid("compile_backend", torch_compile and str(_get(config, "dynamo_backend", "inductor") or "inductor") == "cudagraphs",
           f"{family} block compile does not support CUDA Graphs")
    forbid("compile_mode", torch_compile and _get(config, "compile_inductor_mode") not in (None, "default"),
           f"{family} block compile supports only default Inductor mode")
    forbid("auto_block_swap", _bool_value(_get(config, "auto_block_swap")),
           f"{family} AUTO block swap is not implemented")
    forbid("selective_checkpoint", selective_checkpoint != "off",
           f"{family} supports only full gradient checkpointing")
    forbid("cpu_offload_checkpointing", cpu_offload_checkpointing or unsloth_offload_checkpointing,
           f"{family} checkpoint offload is not implemented")
    forbid("network_train_unet_only", not _bool_value(_get(config, "network_train_unet_only"), True),
           f"{family} cannot train the text encoder")
    forbid("weighted_captions", _bool_value(_get(config, "weighted_captions")),
           f"{family} weighted captions are not implemented")
    forbid("caption_dropout_rate", _float_value(_get(config, "caption_dropout_rate")) > 0
           or _nested_caption_dropout_enabled(config),
           f"{family} requires caption_dropout_rate=0")
    unverified("masked_loss", _bool_value(_get(config, "masked_loss")),
               f"{family} masked loss has no accepted real-machine run")
    unverified("loss_type", str(_get(config, "loss_type", "l2") or "l2").lower() != "l2",
               f"{family} official recipe uses the L2 flow-matching loss only")
    forbid("timestep_sampling", str(_get(config, "timestep_sampling", "uniform") or "uniform").lower() != "uniform",
           f"{family} uses uniformly sampled dynamic-shift sigmas")
    unverified("weighting_scheme", str(_get(config, "weighting_scheme", "none") or "none").lower() != "none",
               f"{family} official recipe uses weighting_scheme=none")
    sampling = any(_bool_value(_get(config, key)) if key == "sample_at_first"
                   else _int_value(_get(config, key)) > 0
                   for key in ("sample_at_first", "sample_every_n_steps", "sample_every_n_epochs"))
    forbid("preview_text_cache", sampling and not _bool_value(_get(config, "cache_text_encoder_outputs")),
           f"{family} preview requires cached text encoder outputs")
    forbid("layer_start", _get(config, "layer_start") is not None or _get(config, "layer_end") is not None,
           f"{family} layer-range targeting is not implemented")
    forbid("t_min", _get(config, "t_min") is not None or _get(config, "t_max") is not None,
           f"{family} uses the full sigma range")


def check_training_compat(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    unverified_policy: str | None = None,
) -> TrainingCompatResult:
    """Validate optimization-flag combinations used by training and Web preflight.

    ``unverified_policy`` (or the ``compat_unverified_policy`` config key)
    selects how audited unverified combinations are reported: ``strict``
    (origin/dev behavior), ``warn`` (default; the seven audited C1 codes become
    warnings) or ``allow`` (audited C2/C3 codes become warnings too). Input
    validation and semantic-conflict codes stay errors under every policy.
    """

    policy, policy_error = resolve_unverified_policy(config, unverified_policy)
    out = _CompatBuilder(policy)
    if policy_error is not None:
        out.error(
            "invalid_compat_unverified_policy",
            COMPAT_UNVERIFIED_POLICY_KEY,
            policy_error,
        )
    from library.quantization.qwen_nf4_compile_policy import qwen_nf4_compile_experiment_errors

    for message in qwen_nf4_compile_experiment_errors(_get(config, "model_family"), config):
        out.error("qwen_nf4_compile_experiment", "qwen_nf4_compile_experiment", message)
    if _get(config, "qwen_nf4_compile_experiment") is True and world_size not in (None, 1):
        out.error("qwen_nf4_compile_experiment", "qwen_nf4_compile_experiment", "single-process training required")
    config = _with_network_args(config)

    from library.training.auto_block_swap.config import configuration_errors

    for message in configuration_errors(config, world_size=world_size or 1):
        out.error("auto_block_swap_contract", "auto_block_swap", message)

    from library.training.adaptive_runtime.training_config import configuration_errors as precision_errors
    for message in precision_errors(config, world_size=world_size or 1):
        out.error("adaptive_precision_contract", "adaptive_precision", message)

    selective_checkpoint = (
        str(_get(config, "selective_checkpoint", "off") or "off").strip().lower()
    )
    gradient_checkpointing = _bool_value(_get(config, "gradient_checkpointing"), False)
    cpu_offload_checkpointing = _bool_value(
        _get(config, "cpu_offload_checkpointing"), False
    )
    unsloth_offload_checkpointing = _bool_value(
        _get(config, "unsloth_offload_checkpointing"), False
    )
    blocks_to_swap = _int_value(_get(config, "blocks_to_swap"), 0)
    if _bool_value(_get(config, "auto_block_swap"), False):
        blocks_to_swap = 1  # Check enabled-swap contracts before calibration.
    torch_compile = _bool_value(_get(config, "torch_compile"), False)
    use_lokr = _bool_value(_get(config, "use_lokr"), False)
    network_module = str(_get(config, "network_module", "") or "")
    functional_loss_weight = _float_value(_get(config, "functional_loss_weight"), 0.0)
    dynamo_backend = str(_get(config, "dynamo_backend", "") or "")
    compile_inductor_mode = _get(config, "compile_inductor_mode", None)
    model_family = str(_get(config, "model_family", "") or "anima").strip().lower()
    try:
        canonical_family = normalize_registered_family(
            model_family,
            source="training compatibility model_family",
            allow_aliases=True,
        )
        family_spec = get_model_family_spec(canonical_family)
    except ValueError as exc:
        out.error("invalid_model_family", "model_family", str(exc))
        return out.build()
    from library.training.task_contracts import check_training_task

    check_training_task(out, config, family_spec)
    krea2_family = family_spec.name == "krea2_raw"
    z_image_family = family_spec.name == "z_image"
    qwen_image_2_1_family = family_spec.name == "qwen_image_2_1"
    from library.models.qwen_image_2_1.preview_memory import configuration_errors as preview_memory_errors

    for key, message in preview_memory_errors(config, family=family_spec.name):
        code = "sample_vae_decode_contract" if key == "sample_vae_decode" else "qwen_preview_memory_contract"
        out.error(code, key, message)
    fused_projections = str(_get(config, "qwen_fused_projections", "off") or "off").strip().lower()
    saved_blocks = _get(config, "qwen_saved_projection_blocks", 0) or 0
    projection_budget = _get(config, "qwen_projection_budget_mib", 0) or 0
    if not isinstance(saved_blocks, int) or isinstance(saved_blocks, bool) or saved_blocks < 0:
        out.error("invalid_qwen_saved_projection_blocks", "qwen_saved_projection_blocks", "Expected a nonnegative integer")
    if not isinstance(projection_budget, int) or isinstance(projection_budget, bool) or projection_budget < 0:
        out.error("invalid_qwen_projection_budget_mib", "qwen_projection_budget_mib", "Expected a nonnegative integer")
    if isinstance(saved_blocks, int) and isinstance(projection_budget, int) and (
        bool(saved_blocks) != bool(projection_budget) or (saved_blocks > 0 and fused_projections != "all")
    ):
        out.error("qwen_saved_projection_contract", "qwen_saved_projection_blocks", "Saved projections require mode=all, a positive block count and MiB budget")
    if fused_projections not in {"off", "mlp", "qkv", "all"}:
        out.error("invalid_qwen_fused_projections", "qwen_fused_projections", "Expected off, mlp, qkv, or all")
    elif fused_projections != "off":
        if not qwen_image_2_1_family:
            out.error("qwen_fused_projections_family", "qwen_fused_projections", "Projection packing requires qwen_image_2_1")
        elif _bool_value(_get(config, "use_lokr")) or _bool_value(_get(config, "use_loha")):
            out.error("qwen_fused_projections_plain_lora", "qwen_fused_projections", "Projection packing requires plain LoRA; LoKr and LoHa are unsupported")
        for key, active, detail in (
            ("mixed_precision", str(_get(config, "mixed_precision", "bf16") or "bf16").lower() != "bf16", "BF16 mixed precision"),
            ("base_compute", str(_get(config, "base_compute", "bf16") or "bf16").lower() != "bf16", "BF16 base compute"),
            ("block_swap_transfer_dtype", str(_get(config, "block_swap_transfer_dtype", "bf16") or "bf16").lower() != "bf16", "BF16 block-swap transfer"),
            ("gradient_checkpointing", not gradient_checkpointing, "full gradient checkpointing"),
            ("selective_checkpoint", selective_checkpoint != "off", "selective checkpoint off"),
            ("cpu_offload_checkpointing", cpu_offload_checkpointing or unsloth_offload_checkpointing, "checkpoint offload disabled"),
        ):
            if active:
                out.error("qwen_fused_projections_" + key, key, "Experimental projection packing requires " + detail)
    pipeline_parallel_error: ValueError | None = None
    try:
        pipeline_parallel = PipelineParallelConfig.from_config(config).enabled
    except ValueError as exc:
        pipeline_parallel = False
        pipeline_parallel_error = exc
    compile_dynamic_seq = _bool_value(_get(config, "compile_dynamic_seq"), False)
    compile_seq_bands = _bool_value(_get(config, "compile_seq_bands"), False)
    v100_flash_stability = (
        str(_get(config, "v100_flash_stability", "off") or "off").strip().lower()
    )

    if pipeline_parallel_error is not None:
        out.error(
            "pipeline_parallel_config",
            "pipeline_parallel",
            str(pipeline_parallel_error),
        )
    elif pipeline_parallel:
        try:
            validate_pipeline_parallel_config(config, world_size=world_size)
        except ValueError as exc:
            out.error(
                "pipeline_parallel_config",
                "pipeline_parallel",
                str(exc),
            )
        else:
            # The family planner above is static metadata. This is the runtime
            # gate: it opens only for a family that publishes a stage-local PP
            # backend *and* whose runtime contract validates in this process
            # (schedule, microbatches >= stages, world size, no quantized base,
            # torch.distributed.pipelining importable). Every other family keeps
            # failing closed with a named code.
            from library.training.pp.gate import pipeline_parallel_runtime_issue

            issue = pipeline_parallel_runtime_issue(config, world_size=world_size)
            if issue is not None:
                out.error(issue.code, "pipeline_parallel", issue.message)
            else:
                out.warning(
                    "pipeline_parallel_experimental_runtime",
                    "pipeline_parallel",
                    f"{family_spec.display_name} pipeline_parallel resolves to the "
                    "experimental stage-local PP v0 runtime (2 stages, 1F1B, fp16 "
                    "transaction). It is served by library.training.pp, not by the "
                    "Accelerate trainer: the trainer entry refuses a "
                    "pipeline_parallel run with pp_trainer_path_unavailable instead "
                    "of silently falling back to DDP.",
                )

    # Krea-2 gets a family-specific message/mutation below. Avoid emitting a
    # duplicate generic mutation when its fixed-padded gate is the real cause.
    if (
        compile_seq_bands
        and not compile_dynamic_seq
        and family_spec.name == "anima"
    ):
        message = (
            "compile_seq_bands only applies with compile_dynamic_seq; "
            "compile_seq_bands will be disabled."
        )
        out.warning(
            "compile_seq_bands_requires_dynamic_seq",
            "compile_seq_bands",
            message,
        )
        out.mutate(
            "compile_seq_bands_requires_dynamic_seq",
            "compile_seq_bands",
            False,
            message,
        )

    if selective_checkpoint not in VALID_SELECTIVE_CHECKPOINTS:
        out.error(
            "invalid_selective_checkpoint",
            "selective_checkpoint",
            "--selective_checkpoint must be one of: "
            + ", ".join(sorted(VALID_SELECTIVE_CHECKPOINTS)),
        )

    if family_spec.plain_lora_only:
        adapter_conflicts = []
        network_spec_name = None
        if network_module not in {"", "networks.lora_anima"}:
            adapter_conflicts.append(f"network_module={network_module!r}")
        if family_spec.supported_network_specs is not None:
            from networks.registry import resolve_network_spec

            try:
                network_spec = resolve_network_spec(config)
            except ValueError as exc:
                adapter_conflicts.append(f"network_args ({exc})")
            else:
                network_spec_name = network_spec.name
                if network_spec.name not in family_spec.supported_network_specs:
                    adapter_conflicts.append(f"network_spec={network_spec.name!r}")
        allowed_flags = {"use_lokr", "use_loha"} if qwen_image_2_1_family else set()
        adapter_conflicts.extend(
            flag
            for flag in KREA2_UNSUPPORTED_ADAPTER_FLAGS
            if flag not in allowed_flags
            if _bool_value(_get(config, flag), False)
        )
        qwen_task = str(_get(config, "qwen_image_2_1_task", "t2i") or "t2i").strip().lower()
        if qwen_image_2_1_family and qwen_task == "edit":
            if network_spec_name in {"lokr", "loha"} or any(
                _bool_value(_get(config, flag), False) for flag in ("use_lokr", "use_loha")
            ):
                adapter_conflicts.append("qwen_image_2_1_task=edit requires plain LoRA")
        moe_style = (
            str(_get(config, "use_moe_style", "false") or "false").strip().lower()
        )
        if moe_style not in {"", "0", "false", "none", "off"}:
            adapter_conflicts.append(f"use_moe_style={moe_style!r}")
        if _bool_value(_get(config, "route_per_layer"), False):
            adapter_conflicts.append("route_per_layer")
        router_source = (
            str(_get(config, "router_source", "none") or "none").strip().lower()
        )
        if router_source != "none":
            adapter_conflicts.append(f"router_source={router_source!r}")
        if _bool_value(_get(config, "dora_wd"), False):
            adapter_conflicts.append("dora_wd")
        if _int_value(_get(config, "step_expert_K"), 0) > 1:
            adapter_conflicts.append("step_expert_K")
        if _bool_value(_get(config, "train_adaln"), False):
            adapter_conflicts.append("train_adaln")
        if _int_value(_get(config, "num_registers"), 0) > 0:
            adapter_conflicts.append("num_registers")
        if functional_loss_weight > 0:
            adapter_conflicts.append("functional_loss_weight")
        if adapter_conflicts:
            out.error(
                ("krea2_plain_lora_only" if krea2_family else "family_plain_lora_only"),
                "network_module",
                f"{family_spec.display_name} training supports only "
                + ("plain LoRA, LoKr, or LoHa" if qwen_image_2_1_family else "plain LoRA")
                + "; unsupported: " + ", ".join(adapter_conflicts),
            )

    if family_spec.name == "moss_image":
        from library.training.moss_compat import check_moss_training_contract

        check_moss_training_contract(out, config)

    if qwen_image_2_1_family:
        _check_qwen_image_2_1_contract(
            out, config,
            selective_checkpoint=selective_checkpoint,
            blocks_to_swap=blocks_to_swap,
            torch_compile=torch_compile,
            cpu_offload_checkpointing=cpu_offload_checkpointing,
            unsloth_offload_checkpointing=unsloth_offload_checkpointing,
        )

    if krea2_family:
        krea2_attn_mode = str(_get(config, "attn_mode", "") or "").strip().lower()
        if krea2_attn_mode not in KREA2_ATTN_MODES:
            out.error(
                "krea2_invalid_attn_mode",
                "attn_mode",
                "Krea-2 attn_mode supports torch and flash; sdpa/cudnn/cudnn_sdpa "
                f"are torch aliases; got {krea2_attn_mode!r}.",
            )
        if compile_dynamic_seq:
            message = (
                "Krea-2 compile uses two fixed padded token-family graphs; "
                "compile_dynamic_seq will be disabled."
            )
            out.warning(
                "krea2_compile_dynamic_seq",
                "compile_dynamic_seq",
                message,
            )
            out.mutate(
                "krea2_compile_dynamic_seq",
                "compile_dynamic_seq",
                False,
                message,
            )
        if compile_seq_bands:
            message = (
                "Krea-2 compile uses fixed padded token-family graphs; "
                "compile_seq_bands will be disabled."
            )
            out.warning(
                "krea2_compile_seq_bands",
                "compile_seq_bands",
                message,
            )
            out.mutate(
                "krea2_compile_seq_bands",
                "compile_seq_bands",
                False,
                message,
            )
        normalized_inductor_mode = (
            str(compile_inductor_mode or "default").strip().lower()
        )
        if normalized_inductor_mode != "default":
            out.error(
                "krea2_compile_inductor_mode",
                "compile_inductor_mode",
                "Krea-2 compile supports only compile_inductor_mode=default; "
                f"got {compile_inductor_mode!r}.",
            )
        if selective_checkpoint not in KREA2_SELECTIVE_CHECKPOINTS:
            out.error(
                "krea2_selective_checkpoint",
                "selective_checkpoint",
                "Krea-2 selective_checkpoint supports only off or every_other; "
                f"got {selective_checkpoint!r}.",
            )
        if v100_flash_stability != "off":
            out.error(
                "krea2_v100_flash_stability",
                "v100_flash_stability",
                "v100_flash_stability is Anima-only and must be off for Krea-2.",
            )

    if z_image_family:
        if compile_seq_bands:
            message = (
                "compile_seq_bands is Anima-only; Z-Image does not use "
                "native-flatten dynamic sequence bands, so it will be disabled."
            )
            out.warning(
                "z_image_compile_seq_bands",
                "compile_seq_bands",
                message,
            )
            out.mutate(
                "z_image_compile_seq_bands",
                "compile_seq_bands",
                False,
                message,
            )
        attn_mode = str(_get(config, "attn_mode", "flash") or "flash").strip().lower()
        if attn_mode not in {"torch", "flash", "sdpa", "native"} or _bool_value(
            _get(config, "xformers"), False
        ):
            out.error(
                "z_image_attention_mode",
                "attn_mode",
                "Z-Image supports attn_mode=torch, sdpa and native (sdpa/native "
                "are torch aliases) plus flash; xformers must be disabled.",
            )
        mixed_precision = (
            str(_get(config, "mixed_precision", "bf16") or "bf16").strip().lower()
        )
        from library.training.adaptive_runtime.training_config import islands_enabled
        adaptive_islands = islands_enabled(config)
        if mixed_precision != "bf16" and not (
            (adaptive_islands and mixed_precision == "fp16" or mixed_precision == "no")
            and attn_mode in {"torch", "sdpa", "native"}
        ):
            out.error(
                "z_image_bf16_only",
                "mixed_precision",
                "Z-Image v1 is validated with mixed_precision=bf16; adaptive "
                "FP16/FP32 or pure FP32 paths require torch/sdpa attention.",
            )
        base_compute = (
            str(_get(config, "base_compute", "bf16") or "bf16").strip().lower()
        )
        if base_compute != "bf16":
            out.error(
                "z_image_base_compute",
                "base_compute",
                "Z-Image v1 has no runtime NF4/ConvRot base quantization; "
                "pre-quantized mixed-NF4 weights are supported but must load "
                "with base_compute=bf16.",
            )
        if torch_compile:
            out.error(
                "z_image_torch_compile",
                "torch_compile",
                "Z-Image torch.compile has not been validated; keep torch_compile=false.",
            )
        if blocks_to_swap > 28:
            out.error(
                "z_image_block_swap_range",
                "blocks_to_swap",
                "Z-Image has 30 main layers and requires at least two resident layers; "
                "use blocks_to_swap between 0 and 28.",
            )
        if selective_checkpoint != "off":
            out.error(
                "z_image_selective_checkpoint",
                "selective_checkpoint",
                "Z-Image v1 supports full gradient checkpointing only.",
            )
        if cpu_offload_checkpointing or unsloth_offload_checkpointing:
            out.error(
                "z_image_checkpoint_offload",
                "gradient_checkpointing",
                "Z-Image v1 does not support CPU/Unsloth activation offload.",
            )
        if (
            _get(config, "layer_start") is not None
            or _get(config, "layer_end") is not None
        ):
            out.error(
                "z_image_layer_range",
                "layer_start",
                "Z-Image layer_start/layer_end is not implemented for layers.N and "
                "refiner blocks; remove the layer range.",
            )
        if _bool_value(_get(config, "weighted_captions"), False):
            out.error(
                "z_image_weighted_captions",
                "weighted_captions",
                "Z-Image weighted captions are not implemented in v1.",
            )
        loss_type = str(_get(config, "loss_type", "l2") or "l2").strip().lower()
        if loss_type != "l2":
            out.unverified(
                "z_image_loss_type",
                "loss_type",
                "Z-Image official recipe uses the L2 flow-matching loss only.",
            )
        if _get(config, "t_min") is not None or _get(config, "t_max") is not None:
            out.error(
                "z_image_timestep_range",
                "t_min",
                "Z-Image v1 samples the complete official scheduler grid; "
                "t_min/t_max are not supported.",
            )
        sampler = str(_get(config, "sampler", "default") or "default").strip().lower()
        if sampler != "default":
            out.error(
                "z_image_training_sampler",
                "sampler",
                "Z-Image v1 supports only the official default training sampler.",
            )
        if v100_flash_stability != "off":
            out.error(
                "z_image_v100_flash_stability",
                "v100_flash_stability",
                "v100_flash_stability is Anima-only and must be off for Z-Image.",
            )
        flow_shift = _float_value(_get(config, "discrete_flow_shift"), 6.0)
        if abs(flow_shift - 6.0) > 1e-7:
            out.unverified(
                "z_image_flow_shift",
                "discrete_flow_shift",
                "Z-Image official recipe uses discrete_flow_shift=6.0.",
            )
        timestep_sampling = (
            str(_get(config, "timestep_sampling", "uniform") or "uniform")
            .strip()
            .lower()
        )
        if timestep_sampling != "uniform":
            out.error(
                "z_image_timestep_sampling",
                "timestep_sampling",
                "Z-Image v1 uses the official shifted-uniform sigma schedule.",
            )
        weighting_scheme = (
            str(_get(config, "weighting_scheme", "none") or "none").strip().lower()
        )
        if weighting_scheme != "none":
            out.unverified(
                "z_image_weighting_scheme",
                "weighting_scheme",
                "Z-Image official recipe uses weighting_scheme=none.",
            )
        caption_dropout_rate = _float_value(_get(config, "caption_dropout_rate"), 0.0)
        if caption_dropout_rate > 0 or _nested_caption_dropout_enabled(config):
            out.error(
                "z_image_caption_dropout",
                "caption_dropout_rate",
                "Z-Image caption dropout requires cached empty-prompt embeddings and "
                "is not supported in v1.",
            )

    if blocks_to_swap < 0:
        out.error(
            "negative_blocks_to_swap",
            "blocks_to_swap",
            "blocks_to_swap must be greater than or equal to 0.",
        )

    block_swap_enabled = blocks_to_swap > 0
    selective_enabled = selective_checkpoint != "off"

    if selective_enabled and gradient_checkpointing:
        out.error(
            "selective_full_gradient_checkpointing",
            "gradient_checkpointing",
            "--selective_checkpoint is a selective DiT checkpoint mode; "
            "do not combine it with full --gradient_checkpointing.",
        )
    if selective_enabled and cpu_offload_checkpointing:
        out.error(
            "selective_cpu_offload",
            "cpu_offload_checkpointing",
            "--selective_checkpoint does not support CPU activation offload.",
        )
    if selective_enabled and unsloth_offload_checkpointing:
        out.error(
            "selective_unsloth_offload",
            "unsloth_offload_checkpointing",
            "--selective_checkpoint cannot be combined with "
            "--unsloth_offload_checkpointing.",
        )

    if block_swap_enabled and cpu_offload_checkpointing:
        out.error(
            "block_swap_cpu_offload",
            "cpu_offload_checkpointing",
            "blocks_to_swap supports standard gradient_checkpointing, but is "
            "not supported with cpu_offload_checkpointing",
        )

    if unsloth_offload_checkpointing:
        if not gradient_checkpointing and not selective_enabled:
            out.warning(
                "unsloth_enables_gradient_checkpointing",
                "gradient_checkpointing",
                "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled",
            )
            out.mutate(
                "unsloth_enables_gradient_checkpointing",
                "gradient_checkpointing",
                True,
                "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled",
            )
        if cpu_offload_checkpointing:
            out.error(
                "unsloth_cpu_offload",
                "cpu_offload_checkpointing",
                "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing",
            )
        if block_swap_enabled:
            out.error(
                "block_swap_unsloth_offload",
                "unsloth_offload_checkpointing",
                "blocks_to_swap supports standard gradient_checkpointing, but is "
                "not supported with unsloth_offload_checkpointing",
            )

    if cpu_offload_checkpointing and not gradient_checkpointing:
        out.warning(
            "cpu_offload_without_gradient_checkpointing",
            "cpu_offload_checkpointing",
            "cpu_offload_checkpointing only affects activation checkpointing; "
            "enable gradient_checkpointing or disable cpu_offload_checkpointing.",
        )

    if use_lokr and gradient_checkpointing and torch_compile:
        out.warning(
            "lokr_full_checkpoint_compile",
            "torch_compile",
            "LoKr with full gradient_checkpointing under torch_compile is "
            "experimental. Training keeps torch_compile enabled, pins larger "
            "Dynamo graph budgets, and traces the factorized LoKr forward "
            "in-graph; blocks_to_swap remains independent.",
        )

    base_compute = str(_get(config, "base_compute", "bf16") or "bf16").strip().lower()
    block_swap_transfer_dtype = (
        str(_get(config, "block_swap_transfer_dtype", "bf16") or "bf16").strip().lower()
    )
    convrot_active = base_compute in {
        "w8a16_convrot",
        "w8a8_convrot",
        "w8a16",
        "w8a8",
    }
    nf4_active = base_compute == "nf4"
    if convrot_active and block_swap_transfer_dtype in {"int8", "int8_linear", "i8"}:
        out.error(
            "convrot_block_swap_int8_mutex",
            "base_compute",
            "base_compute ConvRot paths are mutually exclusive with "
            "block_swap_transfer_dtype=int8 (double dequant / confused semantics). "
            "Use block_swap_transfer_dtype=bf16 when enabling ConvRot.",
        )
    # NF4 (Krea-2 QLoRA) 与 ConvRot 互斥: ConvRot 是 anima cross-attn/AdaLN 专属
    # 量化路径, NF4 是 Krea-2 single-stream 通用 bnb 4-bit, 两者架构假设不同.
    if nf4_active and convrot_active:
        out.error(
            "nf4_convrot_mutex",
            "base_compute",
            "base_compute=nf4 is mutually exclusive with ConvRot paths "
            "(w8a16_convrot/w8a8_convrot). NF4 is the Krea-2 QLoRA path; "
            "ConvRot is anima-specific. Pick one.",
        )
    # NF4 × block_swap 已验证通过 (方向 A: deepcopy master + Params4bit.to() 整体
    # 搬运, offloading.py isinstance 分流不碰 bf16/int8/fp8 路径). 端到端探针
    # (probe_nf4_blockswap.py, PG199, 1024, swap=4, 30 步): host RAM 18.18GB
    # (bf16 路径 22.64GB master 单项超它曾在 62GB 机宕机), GPU 10.26GB, loss
    # 0.0084->0.0017 单调下降, LoRA grad 非零, DiT frozen 不变. 保留 warning:
    # NF4 block swap 主战场是 host RAM (master 5.66GB) 而非 GPU, 提醒用户关注
    # pinned master 内存预算.
    if nf4_active and block_swap_enabled:
        out.warning(
            "nf4_block_swap_host_ram",
            "base_compute",
            "base_compute=nf4 + blocks_to_swap verified (offloader NF4 path "
            "via Params4bit integral transport, end-to-end probe green). "
            "Main constraint is host RAM: pinned NF4 masters ~5.7GB (bf16 "
            "path 22.64GB), watch pinned-master memory budget on low-RAM hosts.",
        )
    # nf4_prequantized_path: 只在 base_compute=nf4 时生效. 给了路径但 base_compute
    # 不是 nf4 → 路径被静默忽略, 提醒用户 (常见于从 NF4 配置切回 bf16 忘删路径).
    nf4_path = _get(config, "nf4_prequantized_path", None)
    if nf4_path and not nf4_active:
        out.warning(
            "nf4_path_ignored",
            "nf4_prequantized_path",
            "--nf4_prequantized_path only takes effect with --base_compute nf4; "
            "ignored under current base_compute. Drop the path or switch to nf4.",
        )
    if base_compute not in {
        "bf16",
        "fp16",
        "none",
        "off",
        "w8a16_convrot",
        "w8a8_convrot",
        "w8a16",
        "w8a8",
        "nf4",
    }:
        out.error(
            "invalid_base_compute",
            "base_compute",
            f"unknown base_compute={base_compute!r}; expected bf16 | fp16 | "
            "none | off | w8a16_convrot | w8a8_convrot | w8a16 | w8a8 | nf4",
        )

    # ConvRot on attention linears + flash + torch.compile can materialize
    # float32 Q/K/V into FlashAttention (FA only accepts fp16/bf16). The
    # dispatcher now hard-casts at the flash entry, so this is informational
    # rather than a hard block — still surface so operators know the combo is
    # on the known-risk path.
    attn_mode = str(_get(config, "attn_mode", "") or "").strip().lower()
    convrot_scope = str(_get(config, "convrot_scope", "mlp") or "mlp").strip().lower()
    convrot_touches_attn = False
    if convrot_active:
        scope_tokens = {
            item.strip() for item in convrot_scope.split(",") if item.strip()
        }
        attn_scope_tokens = {
            "all",
            "attention",
            "attn",
            "self",
            "self_attn",
            "self_attn_qkv",
            "self_qkv",
            "self_attn_out",
            "self_out",
            "cross",
            "cross_attn",
            "cross_attn_q",
            "cross_q",
            "cross_attn_kv",
            "cross_kv",
            "cross_attn_out",
            "cross_out",
            "attention_out",
            "attn_out",
        }
        convrot_touches_attn = bool(scope_tokens & attn_scope_tokens)
    if (
        convrot_active
        and convrot_touches_attn
        and attn_mode in {"flash", "flash4"}
        and torch_compile
    ):
        out.warning(
            "convrot_attn_flash_compile_dtype",
            "base_compute",
            "base_compute ConvRot with convrot_scope covering attention "
            f"({convrot_scope!r}) + attn_mode={attn_mode!r} + torch_compile can "
            "materialize float32 Q/K/V; FlashAttention only accepts fp16/bf16. "
            "The flash dispatcher hard-casts at entry as a safety net. Prefer "
            "convrot_scope=mlp to avoid the cast tax if attention quant is not required.",
        )

    if block_swap_enabled:
        if torch_compile and dynamo_backend == "cudagraphs":
            out.warning(
                "block_swap_cudagraphs_disable_compile",
                "torch_compile",
                "blocks_to_swap moves DiT block weights between CPU/GPU, "
                "so dynamo_backend='cudagraphs' is unsafe. Disabling torch_compile.",
            )
            out.mutate(
                "block_swap_cudagraphs_disable_compile",
                "torch_compile",
                False,
                "blocks_to_swap moves DiT block weights between CPU/GPU, "
                "so dynamo_backend='cudagraphs' is unsafe. Disabling torch_compile.",
            )
            torch_compile = False

        if torch_compile and compile_inductor_mode in {
            "reduce-overhead",
            "max-autotune",
        }:
            safe_mode = (
                "max-autotune-no-cudagraphs"
                if compile_inductor_mode == "max-autotune"
                else None
            )
            out.warning(
                "block_swap_compile_mode_cudagraphs",
                "compile_inductor_mode",
                "blocks_to_swap is incompatible with Inductor CUDAGraph modes "
                f"({compile_inductor_mode!r}); using {safe_mode or 'default'} mode instead.",
            )
            out.mutate(
                "block_swap_compile_mode_cudagraphs",
                "compile_inductor_mode",
                safe_mode,
                "blocks_to_swap is incompatible with Inductor CUDAGraph modes "
                f"({compile_inductor_mode!r}); using {safe_mode or 'default'} mode instead.",
            )

        if network_module == "networks.methods.soft_tokens":
            out.error(
                "block_swap_soft_tokens",
                "blocks_to_swap",
                "blocks_to_swap is not supported with soft_tokens. "
                "Keep blocks_to_swap=0 for this multi-forward method.",
            )
        if functional_loss_weight > 0.0:
            out.error(
                "block_swap_functional_loss",
                "blocks_to_swap",
                "blocks_to_swap is not supported with functional_loss_weight > 0. "
                "Disable block swap for postfix/functional multi-forward training.",
            )

    return out.build()


def apply_training_compat_mutations(
    target: Mapping[str, Any] | object,
    result: TrainingCompatResult,
) -> None:
    """Apply suggested mutations to a mutable mapping or argparse namespace."""

    for mutation in result.mutations:
        if isinstance(target, dict):
            target[mutation.key] = mutation.value
        else:
            setattr(target, mutation.key, mutation.value)
