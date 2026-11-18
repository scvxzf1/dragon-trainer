import { describe, expect, it } from "vitest";
import { distributedDiagnosisOverrides, effectiveDistributedSettings, pipelineSettingsOnEnable } from "./distributedDraft";
import { ppWiredForFamily } from "./distributedDiagnosis";
import type { DistributedSettings } from "./distributedApi";
import { fieldsForConfig } from "./fieldCatalog";
import { draftFromMerged, trainingPatchValues, sameTrainingValue } from "./trainingForm";

const defaults: DistributedSettings = {
  mode: "pp", gpu_whitelist: [1, 2], pipeline_parallel_stages: 2,
  pipeline_parallel_microbatches: 4, pipeline_parallel_schedule: "1f1b",
  pipeline_parallel_split: "balanced", pipeline_parallel_blocks_to_swap: 4,
};

describe("distributed card values", () => {
  it("uses machine defaults on enable despite implicit hydration defaults", () => {
    const config = { model_family: "krea2_raw", pipeline_parallel: false };
    const baseline = draftFromMerged(config, fieldsForConfig(config));
    expect(pipelineSettingsOnEnable(baseline, baseline, config, defaults)
      .pipeline_parallel_blocks_to_swap).toBe(4);
    expect(pipelineSettingsOnEnable(
      { ...baseline, pipeline_parallel_blocks_to_swap: 2 }, baseline, config, defaults,
    ).pipeline_parallel_blocks_to_swap).toBe(2);
    expect(pipelineSettingsOnEnable(
      baseline, baseline, { ...config, pipeline_parallel_blocks_to_swap: 0 }, defaults,
    ).pipeline_parallel_blocks_to_swap).toBe(0);
  });
  it("tracks and saves swap edits in an imported PP config without that key", () => {
    const config = { model_family: "krea2_raw", pipeline_parallel: true };
    const fields = fieldsForConfig(config);
    const swap = fields.find((field) => field.key === "pipeline_parallel_blocks_to_swap")!;
    expect(swap).toMatchObject({ kind: "number", group: "resources" });
    const baseline = draftFromMerged(config, fields);
    const draft: typeof baseline = { ...baseline, pipeline_parallel_blocks_to_swap: 4 };
    expect(sameTrainingValue(draft[swap.key], baseline[swap.key], swap.kind)).toBe(false);
    expect(trainingPatchValues(draft, baseline, fields, config)).toEqual({
      pipeline_parallel_blocks_to_swap: 4,
    });
    expect(trainingPatchValues(
      { ...draft, pipeline_parallel_blocks_to_swap: 0 }, draft, fields, config,
    )).toEqual({ pipeline_parallel_blocks_to_swap: 0 });
  });
  it("shows the actual configuration including zero before machine defaults", () => {
    expect(effectiveDistributedSettings(
      { pipeline_parallel_blocks_to_swap: 0 },
      { pipeline_parallel_blocks_to_swap: 2 }, defaults,
    ).pipeline_parallel_blocks_to_swap).toBe(0);
  });

  it("uses machine defaults only when the configuration has no value", () => {
    expect(effectiveDistributedSettings({}, {}, defaults).pipeline_parallel_blocks_to_swap).toBe(4);
  });

  it("diagnoses draft family, PP, and admission scalar edits without exposing paths", () => {
    expect(distributedDiagnosisOverrides(
      {
        model_family: "anima",
        pipeline_parallel_blocks_to_swap: 0,
        gradient_accumulation_steps: 2,
        sample_at_first: true,
        output_dir: "/private",
        sample_prompts: "prompts.txt",
      },
      { model_family: "krea2_raw", pipeline_parallel_blocks_to_swap: 4 },
    )).toEqual({
      model_family: "anima",
      pipeline_parallel_blocks_to_swap: 0,
      gradient_accumulation_steps: 2,
      sample_at_first: true,
    });
  });

  it("allows Krea-2 aliases before a diagnosis and respects an explicit refusal", () => {
    expect(ppWiredForFamily("krea2")).toBe(true);
    expect(ppWiredForFamily("krea2_raw")).toBe(true);
    expect(ppWiredForFamily("krea2", [{
      family: "krea2_raw", display_name: "Krea-2", wired: false,
      runtime_backend: null, stages: [], schedules: [],
    }])).toBe(false);
  });
});
