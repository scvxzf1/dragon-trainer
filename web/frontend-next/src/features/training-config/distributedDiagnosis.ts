import type {
  DistributedCheck,
  DistributedDiagnosis,
  WiredPpFamily,
} from "./distributedApi";
export const DISTRIBUTED_MODES = [
  { value: "single", label: "单卡", hint: "只在一张 GPU 上训练" },
  { value: "ddp", label: "多卡 · 数据并行", hint: "每张卡一份完整模型副本" },
  { value: "pp", label: "流水线并行（实验）", hint: "模型按阶段切分到多张卡" },
] as const;

export type DistributedMode = (typeof DISTRIBUTED_MODES)[number]["value"];

export function normalizeDistributedMode(value: unknown): DistributedMode {
  const text = String(value ?? "").trim().toLowerCase();
  if (text === "single" || text === "ddp" || text === "pp") return text;
  return "ddp";
}

/** level → 三档徽标（🔴必败 / 🟡风险 / 🔵信息），与 t1 诊断端点口径一致。 */
export function checkBadge(check: DistributedCheck): { icon: string; tone: string } {
  if (check.level === "error") return { icon: "🔴", tone: "danger" };
  if (check.level === "warning") return { icon: "🟡", tone: "warning" };
  return { icon: "🔵", tone: "info" };
}

/** 与后端 TRAINER_PP_WIRED_FAMILIES 一致的兜底（诊断结果优先）。 */
export const FALLBACK_WIRED_PP_FAMILIES = ["anima", "moss_image"];

export function ppWiredForFamily(
  family: string | undefined | null,
  wiredFamilies?: WiredPpFamily[],
): boolean {
  const name = String(family || "anima").trim();
  if (wiredFamilies?.length) {
    return wiredFamilies.some((item) => item.family === name && item.wired);
  }
  return FALLBACK_WIRED_PP_FAMILIES.includes(name);
}

/** 汇总诊断 checks：错误/警告计数与首要文案。 */
export function diagnosisSummary(checks: DistributedCheck[]): {
  errors: number;
  warnings: number;
  headline: DistributedCheck | null;
} {
  const errors = checks.filter((check) => check.level === "error");
  const warnings = checks.filter((check) => check.level === "warning");
  const headline = errors[0] || warnings[0] || checks[0] || null;
  return { errors: errors.length, warnings: warnings.length, headline };
}

/** 卡片折叠行的摘要文案。 */
export function distributedSummaryLine(
  mode: DistributedMode,
  gpuLabel: string,
  diagnosis: DistributedDiagnosis | undefined,
): string {
  const base = mode === "single"
    ? `单卡 · ${gpuLabel}`
    : mode === "ddp"
      ? `数据并行 · ${gpuLabel}`
      : `流水线并行 · ${gpuLabel}`;
  if (!diagnosis) return base;
  const { errors, warnings } = diagnosisSummary(diagnosis.checks);
  const suffix = errors ? ` 🔴${errors}` : warnings ? ` 🟡${warnings}` : "";
  return `${base}${suffix}`;
}
