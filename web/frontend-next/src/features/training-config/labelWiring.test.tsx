import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";

import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { TrainingFieldHelpContent } from "./TrainingFieldHelpContent";
import { fieldsForConfig } from "./fieldCatalog";
import { resolveFieldHelp } from "./domain/resolveFieldHelp";
import { draftFromMerged } from "./trainingForm";

const KREA2_EXPERIMENT_KEYS = ["nf4_dir1", "krea2_attn_fp16"];

afterEach(cleanup);

it("renders readable labels and help triggers for the two Krea-2 experimental switches", () => {
  const config = { model_family: "krea2", nf4_dir1: false, krea2_attn_fp16: false };
  const fields = fieldsForConfig(config).filter((field) =>
    KREA2_EXPERIMENT_KEYS.includes(field.key),
  );
  expect(fields).toHaveLength(KREA2_EXPERIMENT_KEYS.length);
  render(
    <TrainingFieldEditor
      fields={fields}
      draft={draftFromMerged(config, fields)}
      ownKeys={new Set(Object.keys(config))}
      disabled={false}
      onChange={() => {}}
      method="lora"
    />,
  );

  for (const field of fields) {
    expect(field.label, field.key).not.toBe(field.key);
    expect(screen.getByText(field.label)).toBeTruthy();
    expect(screen.getByRole("button", { name: `查看${field.label}帮助` })).toBeTruthy();
  }
  expect(screen.getByText("NF4 内层 FP16 GEMM（实验，sm<80）")).toBeTruthy();
  expect(screen.getByText("Krea-2 注意力 FP16 内层（实验，sm<80）")).toBeTruthy();
});

it("explains the experimental sm<80 restriction in both help texts", () => {
  for (const key of KREA2_EXPERIMENT_KEYS) {
    const help = resolveFieldHelp(key);
    expect(help, key).toBeDefined();
    const { unmount } = render(<TrainingFieldHelpContent help={help!} value={false} />);
    expect(screen.getByText(help!.summary)).toBeTruthy();
    expect(screen.getByText(/sm80 及以上/)).toBeTruthy();
    expect(screen.getAllByText(/实验/).length).toBeGreaterThan(0);
    unmount();
  }
});

it("states the krea2_attn_fp16 prerequisites and that flash ignores it", () => {
  const help = resolveFieldHelp("krea2_attn_fp16");
  expect(help).toBeDefined();
  const text = JSON.stringify(help);
  // 前置条件：attn_mode=torch/sdpa + BF16 compute
  expect(text).toMatch(/torch/);
  expect(text).toMatch(/sdpa/);
  expect(text).toMatch(/BF16/);
  // flash 下不生效，避免用户开启死开关
  expect(help!.cautions?.some((line) => /flash/.test(line) && /不生效/.test(line))).toBe(true);
  const { unmount } = render(<TrainingFieldHelpContent help={help!} value={false} />);
  expect(screen.getByText(/生效前置条件/)).toBeTruthy();
  expect(screen.getByText(/attn_mode=flash 下不起作用/)).toBeTruthy();
  unmount();
});
