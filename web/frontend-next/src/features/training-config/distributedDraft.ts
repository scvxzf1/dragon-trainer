import type { DistributedSettings } from "./distributedApi";

type Values = Record<string, unknown>;
const PP_KEYS = [
  "pipeline_parallel_stages", "pipeline_parallel_microbatches",
  "pipeline_parallel_schedule", "pipeline_parallel_split",
  "pipeline_parallel_blocks_to_swap",
] as const;

export function effectiveDistributedSettings(
  draft: Values, merged: Values | undefined, defaults: DistributedSettings | undefined,
) {
  const values = { ...merged, ...draft };
  const number = (key: (typeof PP_KEYS)[number], fallback: number) =>
    Number(values[key] ?? defaults?.[key] ?? fallback);
  return {
    pipeline_parallel_stages: number("pipeline_parallel_stages", 2),
    pipeline_parallel_microbatches: number("pipeline_parallel_microbatches", 4),
    pipeline_parallel_schedule: String(values.pipeline_parallel_schedule ?? defaults?.pipeline_parallel_schedule ?? "1f1b"),
    pipeline_parallel_split: String(values.pipeline_parallel_split ?? defaults?.pipeline_parallel_split ?? "balanced"),
    pipeline_parallel_blocks_to_swap: number("pipeline_parallel_blocks_to_swap", 0),
  };
}

export function pipelineSettingsOnEnable(
  draft: Values, baseline: Values, merged: Values | undefined,
  defaults: DistributedSettings | undefined,
) {
  const explicit = { ...merged };
  for (const key of PP_KEYS) {
    if (draft[key] !== baseline[key]) explicit[key] = draft[key];
  }
  // Hydration fills missing numeric fields with UI defaults. Those values
  // are not explicit overrides of the user's saved machine preferences.
  return effectiveDistributedSettings({}, explicit, defaults);
}

const PP_DIAGNOSIS_SCALAR_KEYS = [
  "model_family", "mixed_precision", "base_compute", "train_batch_size",
  "gradient_accumulation_steps", "gradient_checkpointing", "selective_checkpoint",
  "cpu_offload_checkpointing", "unsloth_offload_checkpointing",
  "blocks_to_swap", "auto_block_swap", "adaptive_precision", "adaptive_oom_retry",
  "torch_compile", "compile_dynamic_seq", "compile_seq_bands",
  "sample_at_first", "sample_every_n_steps", "sample_every_n_epochs",
  "use_vae_cache", "use_text_cache", "skip_cache_check",
  "loss_type", "weighting_scheme", "stage_schedule_enabled",
  "save_n_epoch_ratio", "save_every_n_epochs", "persistent_data_loader_workers",
  "router_source", "use_moe_style", "use_ip_adapter", "use_easycontrol",
  "use_byg", "caption_dropout_rate", "ip_noise_gamma", "network_train_unet_only",
  "pipeline_parallel", ...PP_KEYS,
] as const;

export function distributedDiagnosisOverrides(draft: Values, merged: Values | undefined) {
  const values = { ...merged, ...draft };
  return Object.fromEntries(PP_DIAGNOSIS_SCALAR_KEYS.flatMap((key) => {
    const value = values[key];
    return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
      ? [[key, value]] : [];
  }));
}
