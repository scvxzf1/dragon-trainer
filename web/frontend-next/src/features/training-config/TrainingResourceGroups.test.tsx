import { cleanup, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { fieldAvailability, fieldsForConfig, filterTrainingFields } from "./fieldCatalog";
import { groupResourceFields, resourceSummary } from "./resourceGroups";
import { TrainingResourceGroups } from "./TrainingResourceGroups";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { draftFromMerged } from "./trainingForm";
import type { TrainingDraft } from "./trainingForm";

afterEach(cleanup);
const draft = { mixed_precision: "bf16", auto_block_swap: false, blocks_to_swap: 20, torch_compile: false };
const fields = fieldsForConfig(draft).filter((field) => field.group === "resources");

it("groups every resource exactly once, with AUTO before manual swap", () => {
  const groups = groupResourceFields(fields);
  expect(groups.flatMap((group) => group.fields).map((field) => field.key).sort())
    .toEqual(fields.map((field) => field.key).sort());
  const swap = groups.find((group) => group.id === "residency")!.fields.map((field) => field.key);
  expect(swap[0]).toBe("auto_block_swap");
  expect(swap.indexOf("auto_block_swap_mode")).toBeLessThan(swap.indexOf("blocks_to_swap"));
  expect(groups.find((group) => group.id === "compute")!.fields[0].key).toBe("mixed_precision");
  expect(groups.find((group) => group.id === "diagnostics")!.fields.map((field) => field.key)).toContain("debug_finite_checks");
  expect(groupResourceFields([{ key: "future", label: "Future", kind: "text", group: "resources" }])[0].id).toBe("other");
});

it("keeps adaptive precision and OOM retry in separate resource groups", () => {
  const keys = [
    "adaptive_precision", "adaptive_fp32_modules", "adaptive_loss_scale",
    "adaptive_oom_retry", "adaptive_oom_retry_max_attempts",
    "adaptive_oom_retry_swap_increment", "adaptive_oom_retry_max_swap",
    "adaptive_oom_retry_timeout",
  ];
  const groups = groupResourceFields(fields);
  const adaptive = groups.find((group) => group.id === "adaptive_precision")!;
  const retry = groups.find((group) => group.id === "oom_retry")!;
  expect(adaptive.fields.map((field) => field.key)).toEqual([
    "adaptive_precision", "adaptive_fp32_modules", "adaptive_loss_scale",
  ]);
  expect(retry.fields.map((field) => field.key)).toEqual([
    "adaptive_oom_retry", "adaptive_oom_retry_max_attempts",
    "adaptive_oom_retry_swap_increment", "adaptive_oom_retry_max_swap",
    "adaptive_oom_retry_timeout",
  ]);
  expect(groups.find((group) => group.id === "compute")!.fields.map((field) => field.key))
    .not.toEqual(expect.arrayContaining(keys));
  expect(groups.map((group) => group.id).slice(0, 6)).toEqual([
    "compute", "residency", "activation", "compile", "adaptive_precision", "oom_retry",
  ]);
  for (const model_family of ["anima", "krea2_raw", "z_image"]) {
    for (const key of ["adaptive_precision", "adaptive_fp32_modules", "adaptive_loss_scale"]) {
      expect(fieldAvailability(key, { model_family }, "lora").enabled).toBe(true);
    }
  }
  expect(resourceSummary("adaptive_precision", { adaptive_precision: "auto" })).toBe("auto");
  expect(resourceSummary("oom_retry", { adaptive_oom_retry: false })).toBe("关闭");
});

it("shows the Qwen NF4 compile experiment as an Edit-only boolean compile control", async () => {
  const user = userEvent.setup();
  const source = {
    model_family: "qwen_image_2_1",
    qwen_image_2_1_task: "edit",
    qwen_nf4_compile_experiment: false,
  };
  const field = fieldsForConfig(source).find((candidate) => candidate.key === "qwen_nf4_compile_experiment")!;
  expect(field.group).toBe("resources");
  expect(field.kind).toBe("boolean");
  expect(field.label).toBe("Qwen NF4 编译实验");
  expect(groupResourceFields(fieldsForConfig(source)).find((group) => group.id === "compile")?.fields)
    .toContainEqual(expect.objectContaining({ key: "qwen_nf4_compile_experiment" }));
  expect(filterTrainingFields(fieldsForConfig(source), source, source, "", "applicable", "lora")
    .map((candidate) => candidate.key)).toContain("qwen_nf4_compile_experiment");

  const onChange = vi.fn();
  render(<TrainingFieldEditor fields={[field]} draft={source} ownKeys={new Set(Object.keys(source))} disabled={false} onChange={onChange} method="lora" />);
  expect(screen.getByLabelText("Qwen NF4 编译实验")).not.toBeChecked();
  await user.click(screen.getByRole("button", { name: "查看Qwen NF4 编译实验帮助" }));
  expect(await screen.findByText(/默认关闭/)).toBeInTheDocument();
  await user.click(screen.getByLabelText("Qwen NF4 编译实验"));
  expect(onChange).toHaveBeenCalledWith("qwen_nf4_compile_experiment", true);

  for (const wrongScope of [
    { model_family: "qwen_image_2_1", qwen_image_2_1_task: "t2i" },
    { model_family: "anima", qwen_image_2_1_task: "edit" },
  ]) {
    expect(filterTrainingFields(fieldsForConfig(wrongScope), wrongScope, wrongScope, "", "applicable", "lora")
      .map((candidate) => candidate.key)).not.toContain("qwen_nf4_compile_experiment");
    const active = { ...wrongScope, qwen_nf4_compile_experiment: true };
    expect(filterTrainingFields(fieldsForConfig(active), active, active, "", "applicable", "lora")
      .map((candidate) => candidate.key)).toContain("qwen_nf4_compile_experiment");
  }
});

it("labels Qwen task options and explains the Edit dataset contract", () => {
  const field = fieldsForConfig({ model_family: "qwen_image_2_1" }).find(
    (candidate) => candidate.key === "qwen_image_2_1_task",
  )!;
  render(
    <TrainingFieldEditor
      fields={[field]}
      draft={{ model_family: "qwen_image_2_1", qwen_image_2_1_task: "t2i" }}
      ownKeys={new Set()}
      disabled={false}
      onChange={vi.fn()}
      method="lora"
    />,
  );
  expect(screen.getByLabelText("Qwen 任务")).toHaveValue("t2i");
  expect(screen.getByRole("option", { name: "普通文生图 (t2i)" })).toBeInTheDocument();
  expect(screen.getByRole("option", { name: "编辑数据集 (edit)" })).toBeInTheDocument();
  // The Edit dataset contract is documented in the field help dialog
  // (domain/help/input.js), not as an inline hint next to this select.
});

it("undoes individual numeric and boolean drafts without affecting other fields, and clears undo after save", async () => {
  const user = userEvent.setup();
  const testFields = fieldsForConfig({ network_dim: 16, network_train_unet_only: false })
    .filter((field) => ["network_dim", "network_train_unet_only"].includes(field.key));
  function Harness() {
    const [baseline, setBaseline] = useState<TrainingDraft>({ network_dim: 16, network_train_unet_only: false });
    const [draft, setDraft] = useState<TrainingDraft>(baseline);
    return <>
      <TrainingFieldEditor fields={testFields} baseline={baseline} draft={draft} ownKeys={new Set()} disabled={false} onChange={(key, value) => setDraft((current) => ({ ...current, [key]: value }))} />
      <button type="button" onClick={() => setBaseline({ ...draft })}>保存配置</button>
    </>;
  }
  render(<Harness />);

  await user.clear(screen.getByLabelText("LoRA rank"));
  await user.type(screen.getByLabelText("LoRA rank"), "32");
  await user.click(screen.getByLabelText("仅训练 DiT"));
  expect(screen.getAllByText("已修改")).toHaveLength(2);
  const undoNumber = screen.getByRole("button", { name: "撤销LoRA rank修改" });
  const undoBoolean = screen.getByRole("button", { name: "撤销仅训练 DiT修改" });
  await user.click(undoNumber);
  expect(screen.getByLabelText("LoRA rank")).toHaveValue(16);
  expect(screen.queryByRole("button", { name: "撤销LoRA rank修改" })).not.toBeInTheDocument();
  expect(undoBoolean).toBeInTheDocument();
  expect(screen.getByLabelText("仅训练 DiT")).toBeChecked();

  await user.click(screen.getByRole("button", { name: "保存配置" }));
  expect(screen.queryByRole("button", { name: "撤销仅训练 DiT修改" })).not.toBeInTheDocument();
  expect(screen.queryByText("已修改")).not.toBeInTheDocument();
});

it("can undo a dirty field after another setting disables its input", async () => {
  const user = userEvent.setup();
  const field = fieldsForConfig({ blocks_to_swap: 20 }).find((item) => item.key === "blocks_to_swap")!;
  const onChange = vi.fn();
  render(
    <TrainingFieldEditor
      fields={[field]}
      baseline={{ blocks_to_swap: 20 }}
      draft={{ model_family: "anima", blocks_to_swap: 24, auto_block_swap: true }}
      ownKeys={new Set(["blocks_to_swap"])}
      disabled={false}
      onChange={onChange}
    />,
  );
  expect(screen.getByLabelText("Block swap 数量")).toBeDisabled();
  await user.click(screen.getByRole("button", { name: "撤销Block swap 数量修改" }));
  expect(onChange).toHaveBeenCalledWith("blocks_to_swap", 20);
});

it("restores the dataset config field without changing another draft", async () => {
  const user = userEvent.setup();
  const onChange = vi.fn();
  const field = fieldsForConfig({ dataset_config: "configs/datasets/base.toml" })
    .find((item) => item.key === "dataset_config")!;
  render(
    <TrainingFieldEditor
      fields={[field]}
      baseline={{ dataset_config: "configs/datasets/base.toml" }}
      draft={{ dataset_config: "configs/datasets/edited.toml", network_dim: 32 }}
      ownKeys={new Set(["dataset_config"])}
      disabled={false}
      onChange={onChange}
    />,
  );
  await user.click(screen.getByRole("button", { name: "撤销数据集配置修改" }));
  expect(onChange).toHaveBeenCalledWith("dataset_config", "configs/datasets/base.toml");
  expect(onChange).toHaveBeenCalledTimes(1);
});

it("keeps disabled conflicts visible in the default field view", () => {
  const availability = fieldAvailability(
    "max_train_steps",
    { model_family: "krea2_raw", max_train_epochs: 2 },
    "lora",
  );
  expect(availability.visible).toBe(true);
  expect(availability.enabled).toBe(false);
});

it.each([
  { key: "max_token_length", label: "文本编码上限（token）", value: 1024, initial: 4096, code: "moss-token-cap-family" },
  { key: "moss_te_quantization", label: "MOSS 文本编码器精度", value: "nf4", initial: "bf16", code: "moss-te-family" },
])("only renders the MOSS-specific $key for MOSS in every field view", ({ key, label, value, initial, code }) => {
  const source = { model_family: "moss_image", [key]: value, network_dim: 16 };
  const baseline = { ...source, [key]: initial };
  const fields = fieldsForConfig(source).filter((field) => [key, "network_dim"].includes(field.key));
  const onChange = vi.fn();
  const editor = (draft: TrainingDraft) => (
    <TrainingFieldEditor
      fields={filterTrainingFields(fields, draft, baseline, "", "all", "lora")}
      draft={draft}
      baseline={baseline}
      ownKeys={new Set(["network_dim"])}
      disabled={false}
      onChange={onChange}
    />
  );
  const view = render(editor(source));
  const card = screen.getByLabelText(label).closest(".training-field-card")!;
  expect(screen.getByLabelText(label)).toHaveValue(value);
  expect(screen.getByLabelText(label)).toBeEnabled();
  expect(card).toHaveAttribute("data-available", "true");
  expect(card).toHaveAttribute("data-dirty", "true");
  expect(card.querySelector(".field-availability-reason")).toBeNull();

  for (const model_family of ["anima", "krea2_raw", "z_image", "qwen_image_2_1", "unknown"]) {
    const other = { ...source, model_family };
    expect(fieldAvailability(key, other, "lora")).toMatchObject({
      visible: false, enabled: false, code,
    });
    for (const fieldView of ["applicable", "all", "changed"]) {
      expect(filterTrainingFields(fields, other, baseline, "", fieldView, "lora").map((field) => field.key))
        .not.toContain(key);
      expect(filterTrainingFields(fields, other, baseline, key, fieldView, "lora"))
        .toEqual([]);
      expect(filterTrainingFields(fields, source, baseline, key, fieldView, "lora"))
        .toHaveLength(1);
    }
    view.rerender(editor(other));
    expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
    expect(document.getElementById(`training-field-${key}`)).toBeNull();
    expect(screen.getByLabelText("LoRA rank")).toBeEnabled();
  }
  view.rerender(editor(source));
  expect(screen.getByLabelText(label)).toHaveValue(value);
  expect(onChange).not.toHaveBeenCalled();
});

it("disables a field when its value is controlled by another field", () => {
  const field = fieldsForConfig({ max_train_steps: 1600 }).find(
    (candidate) => candidate.key === "max_train_steps",
  )!;
  render(
    <TrainingFieldEditor
      fields={[field]}
      draft={{ model_family: "krea2_raw", max_train_epochs: 2, max_train_steps: 1600 }}
      ownKeys={new Set(["max_train_steps"])}
      disabled={false}
      onChange={vi.fn()}
      method="lora"
    />,
  );
  expect(screen.getByLabelText("最大训练步数")).toBeDisabled();
  expect(screen.getByText(/max_train_epochs 已设置/)).toBeInTheDocument();
});

it("disables undisclosed and unknown fields even when shown for audit", () => {
  const source = { model_family: "anima", use_ip_adapter: false, resampler_heads: 8, __future_runtime_field__: 7 };
  const fields = fieldsForConfig(source).filter((field) =>
    ["use_ip_adapter", "resampler_heads", "__future_runtime_field__"].includes(field.key),
  );
  render(
    <TrainingFieldEditor
      fields={fields}
      draft={draftFromMerged(source, fields)}
      ownKeys={new Set(Object.keys(source))}
      disabled={false}
      onChange={vi.fn()}
      method="ip_adapter"
    />,
  );
  expect(screen.getByLabelText("启用 IP-Adapter")).toBeEnabled();
  expect(screen.getByLabelText("IP Resampler 头数")).toBeDisabled();
  expect(screen.getByLabelText("__future_runtime_field__")).toBeDisabled();
  expect(screen.getByText("该参数不进入常规训练流，仅在候选审计中显示。")).toBeInTheDocument();
});

it("keeps swap and compile child values while their controlling fields change", () => {
  const source = {
    model_family: "krea2_raw",
    auto_block_swap: false,
    auto_block_swap_mode: "dynamic",
    blocks_to_swap: 20,
    torch_compile: false,
    compile_block_scope: "resident",
    compile_dynamic_seq: false,
    compile_seq_bands: true,
  };
  const fields = fieldsForConfig(source).filter((field) =>
    [
      "auto_block_swap_mode",
      "blocks_to_swap",
      "torch_compile",
      "compile_block_scope",
      "compile_dynamic_seq",
      "compile_seq_bands",
    ].includes(field.key),
  );
  const baseline = draftFromMerged(source, fields);
  const onChange = vi.fn();
  const view = render(
    <TrainingFieldEditor
      fields={fields}
      draft={baseline}
      ownKeys={new Set(Object.keys(source))}
      disabled={false}
      onChange={onChange}
      method="lora"
    />,
  );

  const autoModeRow = document.getElementById("training-field-auto_block_swap_mode")!;
  const manualSwapRow = document.getElementById("training-field-blocks_to_swap")!;
  const compileScopeRow = document.getElementById("training-field-compile_block_scope")!;
  const compileBandsRow = document.getElementById("training-field-compile_seq_bands")!;
  expect(within(autoModeRow).getByLabelText("AUTO 调整模式")).toBeDisabled();
  expect(within(autoModeRow).getByText("请先启用 AUTO 块交换。")).toBeInTheDocument();
  expect(within(manualSwapRow).getByLabelText("Block swap 数量")).toBeEnabled();
  expect(within(manualSwapRow).getByLabelText("Block swap 数量")).toHaveValue(20);
  expect(within(compileScopeRow).getByLabelText("编译块范围")).toBeDisabled();
  expect(within(compileScopeRow).getByText("请先启用 torch.compile。")).toBeInTheDocument();

  const parentsEnabled = {
    ...baseline,
    auto_block_swap: true,
    torch_compile: true,
  };
  view.rerender(
    <TrainingFieldEditor
      fields={fields}
      draft={parentsEnabled}
      ownKeys={new Set(Object.keys(source))}
      disabled={false}
      onChange={onChange}
      method="lora"
    />,
  );
  expect(within(autoModeRow).getByLabelText("AUTO 调整模式")).toBeEnabled();
  expect(within(manualSwapRow).getByLabelText("Block swap 数量")).toBeDisabled();
  expect(within(manualSwapRow).getByLabelText("Block swap 数量")).toHaveValue(20);
  expect(within(compileScopeRow).getByLabelText("编译块范围")).toBeEnabled();
  expect(within(compileBandsRow).getByLabelText("分带动态序列编译")).toBeDisabled();
  expect(within(compileBandsRow).getByText("请先启用动态序列编译。")).toBeInTheDocument();

  view.rerender(
    <TrainingFieldEditor
      fields={fields}
      draft={{ ...parentsEnabled, compile_dynamic_seq: true }}
      ownKeys={new Set(Object.keys(source))}
      disabled={false}
      onChange={onChange}
      method="lora"
    />,
  );
  expect(within(compileBandsRow).getByLabelText("分带动态序列编译")).toBeEnabled();
  expect(onChange).not.toHaveBeenCalled();
});

it("omits AUTO swap children until AUTO is enabled in every field view", () => {
  const source = {
    model_family: "krea2_raw",
    auto_block_swap: false,
    auto_block_swap_mode: "startup",
    auto_block_swap_interval: 8,
    auto_block_swap_max_trials: 6,
    auto_block_swap_timeout: 120,
    auto_block_swap_swap_io_limit_mb: 512,
    auto_block_swap_vram_reserve_percent: 10,
    auto_block_swap_preference: "balanced",
    blocks_to_swap: 20,
  };
  const allFields = fieldsForConfig(source);
  const childKeys = allFields.map((field) => field.key).filter((key) =>
    key.startsWith("auto_block_swap_")
  );
  expect(childKeys).toHaveLength(7);
  const baseline = {
    ...source,
    auto_block_swap: true,
    auto_block_swap_mode: "dynamic",
    auto_block_swap_interval: 4,
  };

  for (const view of ["applicable", "all", "changed"]) {
    const hidden = filterTrainingFields(allFields, source, baseline, "", view, "lora");
    expect(hidden.map((field) => field.key)).not.toEqual(expect.arrayContaining(childKeys));
    expect(hidden.map((field) => field.key)).toContain("auto_block_swap");
    if (view !== "changed") expect(hidden.map((field) => field.key)).toContain("blocks_to_swap");

    const enabled = filterTrainingFields(allFields, baseline, source, "", view, "lora");
    expect(enabled.map((field) => field.key)).toContain("auto_block_swap_mode");
    expect(enabled.map((field) => field.key)).toContain("auto_block_swap_interval");
  }

  const hidden = filterTrainingFields(allFields, source, baseline, "", "all", "lora");
  render(
    <TrainingFieldEditor
      fields={hidden.filter((field) => ["auto_block_swap", "blocks_to_swap", ...childKeys].includes(field.key))}
      draft={source}
      ownKeys={new Set(Object.keys(source))}
      disabled={false}
      onChange={vi.fn()}
      method="lora"
    />,
  );
  expect(screen.getByLabelText("AUTO 块交换（实验）")).toBeInTheDocument();
  expect(screen.getByLabelText("Block swap 数量")).toBeEnabled();
  expect(screen.queryByText("请先启用 AUTO 块交换。")).not.toBeInTheDocument();
  expect(document.getElementById("training-field-auto_block_swap_mode")).not.toBeInTheDocument();
});

it("shows summaries, expands groups without edits, and reveals search matches", async () => {
  const onChange = vi.fn();
  const props = { fields, draft, onChange, ownKeys: new Set<string>(), disabled: false, view: "all", search: "" };
  const view = render(<TrainingResourceGroups {...props} />);
  const user = userEvent.setup();
  const swap = screen.getByRole("button", { name: /块交换与内存卸载/ });
  expect(swap).toHaveAttribute("aria-expanded", "false");
  expect(swap).toHaveTextContent("手动 · 20 块");
  await user.click(swap);
  expect(swap).toHaveAttribute("aria-expanded", "true");
  expect(within(screen.getByRole("region", { name: /块交换/ })).getByLabelText("AUTO 块交换（实验）")).toBeVisible();
  expect(onChange).not.toHaveBeenCalled();
  await user.click(swap);
  view.rerender(<TrainingResourceGroups {...props} search="auto_block_swap" fields={fields.filter((field) => field.key === "auto_block_swap")} />);
  expect(screen.getByRole("button", { name: /块交换与内存卸载/ })).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByLabelText("AUTO 块交换（实验）")).toBeVisible();
  expect(screen.queryByRole("button", { name: /精度与计算/ })).not.toBeInTheDocument();
  expect(resourceSummary("residency", { auto_block_swap: true, auto_block_swap_mode: "dynamic" })).toBe("AUTO · dynamic");
});
