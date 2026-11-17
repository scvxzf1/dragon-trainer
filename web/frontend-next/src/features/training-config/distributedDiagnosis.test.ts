import { describe, expect, it } from "vitest";
import {
  DISTRIBUTED_MODES,
  checkBadge,
  diagnosisSummary,
  distributedSummaryLine,
  normalizeDistributedMode,
  ppWiredForFamily,
} from "./distributedDiagnosis";

describe("normalizeDistributedMode", () => {
  it("缺省回落 DDP（旧队列兼容）", () => {
    expect(normalizeDistributedMode(undefined)).toBe("ddp");
    expect(normalizeDistributedMode("")).toBe("ddp");
    expect(normalizeDistributedMode("bogus")).toBe("ddp");
  });

  it("接受三态合法值", () => {
    expect(normalizeDistributedMode("single")).toBe("single");
    expect(normalizeDistributedMode("PP")).toBe("pp");
    expect(normalizeDistributedMode("ddp")).toBe("ddp");
  });

  it("三态清单完整", () => {
    expect(DISTRIBUTED_MODES.map((item) => item.value)).toEqual(["single", "ddp", "pp"]);
  });
});

describe("checkBadge", () => {
  it("三档映射 🔴🟡🔵", () => {
    expect(checkBadge({ level: "error", key: "a", message: "" })).toEqual({ icon: "🔴", tone: "danger" });
    expect(checkBadge({ level: "warning", key: "b", message: "" })).toEqual({ icon: "🟡", tone: "warning" });
    expect(checkBadge({ level: "info", key: "c", message: "" })).toEqual({ icon: "🔵", tone: "info" });
  });
});

describe("ppWiredForFamily", () => {
  it("Anima、MOSS 和 Krea-2 已接线，其他族置灰", () => {
    expect(ppWiredForFamily("anima")).toBe(true);
    expect(ppWiredForFamily("moss_image")).toBe(true);
    expect(ppWiredForFamily("krea2_raw")).toBe(true);
    expect(ppWiredForFamily("krea2")).toBe(true);
    expect(ppWiredForFamily("z_image")).toBe(false);
  });

  it("诊断结果优先于兜底表", () => {
    const wired = [{ family: "anima", display_name: "Anima", wired: true, runtime_backend: null, stages: [2], schedules: ["1f1b"] }];
    expect(ppWiredForFamily("anima", wired)).toBe(true);
    expect(ppWiredForFamily("moss_image", wired)).toBe(false);
  });
});

describe("diagnosisSummary / distributedSummaryLine", () => {
  const checks = [
    { level: "info", key: "a", message: "信息" },
    { level: "warning", key: "b", message: "风险" },
    { level: "error", key: "c", message: "必败" },
  ];
  it("计数与首要文案", () => {
    const summary = diagnosisSummary(checks);
    expect(summary.errors).toBe(1);
    expect(summary.warnings).toBe(1);
    expect(summary.headline?.key).toBe("c");
  });

  it("折叠行摘要带档位后缀", () => {
    expect(distributedSummaryLine("ddp", "GPU 1 · V100", { ok: true, mode: "ddp", modes: [], world_size: 2, wired_families: [], checks, hardware: { gpus: [], risks: [] } }))
      .toBe("数据并行 · GPU 1 · V100 🔴1");
    expect(distributedSummaryLine("single", "GPU 1 · V100", undefined)).toBe("单卡 · GPU 1 · V100");
  });
});
