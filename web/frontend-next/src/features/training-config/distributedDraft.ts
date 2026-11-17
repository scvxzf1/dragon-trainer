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

export function distributedDiagnosisOverrides(draft: Values, merged: Values | undefined) {
  const values = { ...merged, ...draft };
  return Object.fromEntries([
    "model_family", "mixed_precision", "train_batch_size", "blocks_to_swap",
    "torch_compile", "pipeline_parallel", ...PP_KEYS,
  ].flatMap((key) => {
    const value = values[key];
    return typeof value === "string" || typeof value === "number" || typeof value === "boolean"
      ? [[key, value]] : [];
  }));
}
