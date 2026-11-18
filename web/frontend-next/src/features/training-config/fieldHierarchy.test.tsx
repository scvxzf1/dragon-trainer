import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { fieldsForConfig, filterTrainingFields } from "./fieldCatalog";
import { stageSummary } from "./stageGroups";
import { resourceSummary } from "./resourceGroups";
import { TrainingStageFields } from "./TrainingResourceGroups";
import type { TrainingDraft } from "./trainingForm";

afterEach(cleanup);

it("hides cross-family and inactive topology noise in applicable view while surfacing invalid legacy values", () => {
  const cleanKrea2: TrainingDraft = {
    model_family: "krea2_raw",
    base_compute: "bf16",
    cache_llm_adapter_outputs: false,
    v100_flash_stability: "off",
    compile_dynamic_seq: false,
    compile_seq_bands: false,
    nf4_dir1: false,
    krea2_attn_fp16: false,
    pipeline_parallel: false,
    precision_preference: "bf16",
  };
  const allFields = fieldsForConfig(cleanKrea2);
  const applicable = filterTrainingFields(allFields, cleanKrea2, cleanKrea2, "", "applicable", "lora")
    .map((field) => field.key);

  expect(applicable).not.toContain("cache_llm_adapter_outputs");
  expect(applicable).not.toContain("v100_flash_stability");
  expect(applicable).not.toContain("compile_dynamic_seq");
  expect(applicable).not.toContain("nf4_dir1");
  expect(applicable).not.toContain("pipeline_parallel_stages");
  expect(applicable).not.toContain("precision_preference");
  expect(applicable).toContain("krea2_attn_fp16");

  // Krea-2 ignores cache_llm_adapter_outputs even if copied as true from Anima templates,
  // while compile_dynamic_seq=true on Krea-2 and cache_llm_adapter_outputs=true on MOSS stay visible for fixing.
  const dirtyLegacy: TrainingDraft = { ...cleanKrea2, compile_dynamic_seq: true, cache_llm_adapter_outputs: true };
  const legacyVisible = filterTrainingFields(allFields, dirtyLegacy, dirtyLegacy, "", "applicable", "lora")
    .map((field) => field.key);
  expect(legacyVisible).toContain("compile_dynamic_seq");
  expect(legacyVisible).not.toContain("cache_llm_adapter_outputs");

  const mossConflict: TrainingDraft = { ...cleanKrea2, model_family: "moss_image", cache_llm_adapter_outputs: true };
  expect(
    filterTrainingFields(allFields, mossConflict, mossConflict, "", "applicable", "lora").map((field) => field.key),
  ).toContain("cache_llm_adapter_outputs");
});

it("collapses inactive dependent children into the secondary drawer and expands them inline when parent activates", async () => {
  const user = userEvent.setup();
  const inactiveDraft: TrainingDraft = {
    model_family: "krea2_raw",
    auto_block_swap: false,
    blocks_to_swap: 0,
    block_swap_transfer_dtype: "bf16",
    block_swap_restore_mode: "foreach",
    disable_block_swap_for_eval: false,
  };
  const swapFields = filterTrainingFields(
    fieldsForConfig(inactiveDraft),
    inactiveDraft,
    inactiveDraft,
    "",
    "applicable",
    "lora",
  ).filter((field) => field.group === "resources");

  const view = render(
    <TrainingStageFields
      stage="resources"
      search=""
      view="applicable"
      expandedGroups={{ "resources\0\0applicable": { residency: true } }}
      fields={swapFields}
      draft={inactiveDraft}
      baseline={inactiveDraft}
      ownKeys={new Set()}
      disabled={false}
      onChange={vi.fn()}
    />,
  );

  const residency = document.querySelector('.training-resource-group[data-group="residency"]') as HTMLElement;
  expect(within(residency).getByLabelText("Block swap 数量")).toBeInTheDocument();
  expect(within(residency).queryByLabelText("块交换传输精度")).not.toBeInTheDocument();
  const drawerBtn = within(residency).getByRole("button", { name: /按需与高级选项/ });
  expect(drawerBtn).toHaveAttribute("aria-expanded", "false");

  await user.click(drawerBtn);
  expect(within(residency).getByLabelText("块交换传输精度")).toBeDisabled();

  // Once blocks_to_swap > 0, dependent children promote into an inline subpanel under the parent.
  const activeDraft: TrainingDraft = { ...inactiveDraft, blocks_to_swap: 16 };
  view.rerender(
    <TrainingStageFields
      stage="resources"
      search=""
      view="applicable"
      expandedGroups={{ "resources\0\0applicable": { residency: true } }}
      fields={swapFields}
      draft={activeDraft}
      baseline={inactiveDraft}
      ownKeys={new Set(["blocks_to_swap"])}
      disabled={false}
      onChange={vi.fn()}
    />,
  );
  const subpanel = residency.querySelector('.training-dependent-panel[data-parent-key="blocks_to_swap"]') as HTMLElement;
  expect(subpanel).not.toBeNull();
  expect(within(subpanel).getByLabelText("块交换传输精度")).toBeEnabled();
});

it("formats model-family and effective training summaries without cross-family bleed", () => {
  expect(
    stageSummary("input", "models", {
      model_family: "krea2_raw",
      qwen_image_2_1_task: "t2i",
      pretrained_model_name_or_path: "models/diffusion_models/krea2_raw_bf16.safetensors",
    }),
  ).toBe("Krea-2 Raw · krea2_raw_bf16");

  expect(
    stageSummary("input", "models", {
      model_family: "qwen_image_2_1",
      qwen_image_2_1_task: "edit",
    }),
  ).toBe("Qwen-Image 2.1 · Edit");

  expect(
    stageSummary("training", "volume", {
      max_train_epochs: 43,
      max_train_steps: 0,
      train_batch_size: 1,
      gradient_accumulation_steps: 2,
    }),
  ).toBe("43 轮 · batch 1×2");

  expect(
    resourceSummary("compute", {
      mixed_precision: "bf16",
      base_compute: "nf4",
      attn_mode: "flash",
    }),
  ).toBe("训练 bf16 · 底模 nf4 · flash");
});
