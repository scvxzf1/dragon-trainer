import { CONFIG_FIELD_CATALOG } from "./domain/config-field-catalog.js";
import { normalizeModelFamily } from "./domain/model-family.js";
import type { TrainingDraft, TrainingFieldSpec } from "./trainingForm";

export const RESOURCE_GROUPS = [
  ["compute", "精度与计算后端"],
  ["residency", "块交换与内存卸载"],
  ["activation", "梯度检查点"],
  ["compile", "编译加速"],
  ["adaptive_precision", "自适应精度"],
  ["oom_retry", "OOM 重试"],
  ["diagnostics", "诊断与预检"],
  ["topology", "流水线参数（实验审计）"],
  ["other", "其他资源参数"],
] as const;

export function resourceGroup(key: string) {
  if (["mixed_precision"].includes(key)) return "compute";
  if (["adaptive_precision", "adaptive_fp32_modules", "adaptive_loss_scale"].includes(key))
    return "adaptive_precision";
  if ([
    "adaptive_oom_retry",
    "adaptive_oom_retry_max_attempts",
    "adaptive_oom_retry_swap_increment",
    "adaptive_oom_retry_max_swap",
    "adaptive_oom_retry_timeout",
  ].includes(key)) return "oom_retry";
  if (key === "debug_finite_checks" || key === "v100_flash_stability") return "diagnostics";
  const cluster = CONFIG_FIELD_CATALOG[key]?.cluster;
  return RESOURCE_GROUPS.some(([id]) => id === cluster) ? cluster : "other";
}

export function groupResourceFields(fields: TrainingFieldSpec[]) {
  return RESOURCE_GROUPS.map(([id, title]) => ({
    id,
    title,
    fields: fields.filter((field) => resourceGroup(field.key) === id).sort((a, b) =>
      (CONFIG_FIELD_CATALOG[a.key]?.siblingOrder ?? 0) -
      (CONFIG_FIELD_CATALOG[b.key]?.siblingOrder ?? 0)),
  })).filter((group) => group.fields.length > 0);
}

export function resourceSummary(id: string, draft: TrainingDraft) {
  const value = (key: string, fallback = "未设置") => String(draft[key] ?? fallback);
  const enabled = (key: string) => draft[key] === true || draft[key] === "true";
  const probeActive = (key: string) => !["", "off", "false", "0"].includes(String(draft[key] ?? "off").toLowerCase());
  const family = normalizeModelFamily(draft.model_family || "anima");
  switch (id) {
    case "compute":
      return `训练 ${value("mixed_precision", "bf16")} · 底模 ${value("base_compute", "bf16")} · ${value("attn_mode", "flash")}`;
    case "adaptive_precision": {
      const mode = value("adaptive_precision", "off");
      return mode === "off" ? "关闭" : mode;
    }
    case "oom_retry":
      return enabled("adaptive_oom_retry")
        ? `开启 · ${value("adaptive_oom_retry_max_attempts")} 次 · +${value("adaptive_oom_retry_swap_increment")} 块`
        : "关闭";
    case "residency": {
      if (enabled("auto_block_swap")) return `AUTO · ${value("auto_block_swap_mode", "startup")}`;
      const blocks = Number(draft.blocks_to_swap ?? 0);
      return blocks > 0 ? `手动 · ${blocks} 块` : "手动 · 0 块（未启用）";
    }
    case "activation": {
      const sel = value("selective_checkpoint", "off");
      return `梯度检查点${enabled("gradient_checkpointing") ? "开启" : "关闭"} · ${sel === "off" ? "全量重算" : sel}`;
    }
    case "compile":
      if (!enabled("torch_compile") && (family === "z_image" || family === "moss_image")) {
        return "当前模型族不支持";
      }
      return enabled("torch_compile") ? `开启 · ${value("compile_block_scope", "resident")}` : "关闭";
    case "preprocess":
      return `显存 ${value("preprocess_memory_profile", "auto")} · workers ${value("max_data_loader_n_workers", "2")}`;
    case "diagnostics": {
      const active: string[] = [];
      if (enabled("debug_finite_checks")) active.push("有限值检查");
      if (probeActive("memory_probe_jsonl")) active.push("显存探针");
      if (probeActive("peak_probe_jsonl")) active.push("峰值探针");
      if (probeActive("gradient_flow_probe_jsonl")) active.push("梯度探针");
      return active.length ? active.join(" · ") : "默认关闭";
    }
    case "topology":
      return `流水线并行${enabled("pipeline_parallel") ? "开启" : "关闭"}`;
    default:
      return "";
  }
}
