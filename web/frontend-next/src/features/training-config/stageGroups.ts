import { CONFIG_FIELD_CATALOG, CONFIG_STAGE_META } from "./domain/config-field-catalog.js";
import { FIELD_OPTION_LABELS } from "./domain/labels-options.js";
import { normalizeModelFamily } from "./domain/model-family.js";
import type { TrainingDraft, TrainingFieldSpec } from "./trainingForm";
import { groupResourceFields, resourceSummary } from "./resourceGroups";

const OVERRIDES: Record<string, string> = {
  caption_extension: "captions",
  output_dir: "output",
  save_every_n_steps: "output",
};

const TITLES: Record<string, string> = {
  models: "模型与编码器",
  dataset: "数据集配置",
  captions: "标注与遮罩",
  filters: "输入筛选",
  cache: "缓存策略",
  contract: "基础方法",
  capacity: "容量与训练目标",
  "warm-start": "权重热启动",
  adapter: "适配器变体",
  orthogonal: "正交与时间步适配",
  raw: "网络额外参数",
  identity: "任务与输出名称",
  volume: "训练量与批次",
};

const CLUSTER_ORDER_OVERRIDES: Record<string, string[]> = {
  // Keep the training capacity controls immediately below the method contract.
  method: [
    "contract",
    "capacity",
    "adapter",
    "warm-start",
    "orthogonal",
    "reft",
    "routing",
    "fei",
    "chimera",
    "ip-adapter",
    "easycontrol",
    "soft-tokens",
    "raw",
    "spd-audit",
    "other",
  ],
};

function shortStem(pathValue: unknown): string {
  const raw = String(pathValue ?? "").trim();
  if (!raw) return "";
  const base = raw.split(/[\\/]/).pop() || raw;
  return base.replace(/\.(safetensors|ckpt|pt)$/i, "");
}

function isOn(value: unknown): boolean {
  return value === true || value === "true" || value === 1;
}

function formatLearningRate(value: unknown): string {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return String(value ?? "未设置");
  return num < 0.001 ? num.toExponential(1).replace(/\.0e/, "e") : String(num);
}

export function groupStageFields(stage: string, fields: TrainingFieldSpec[]) {
  if (stage === "resources") return groupResourceFields(fields);
  const declaredClusters = [
    ...(CONFIG_STAGE_META.find((item) => item.id === stage)?.clusters || []),
    ...(stage === "input" ? [{ id: "preprocess", label: "预处理与数据加载" }] : []),
  ];
  const order = CLUSTER_ORDER_OVERRIDES[stage];
  const clusters = order
    ? [...declaredClusters].sort((left, right) => {
        const leftIndex = order.indexOf(left.id);
        const rightIndex = order.indexOf(right.id);
        return (
          (leftIndex < 0 ? order.length : leftIndex) -
          (rightIndex < 0 ? order.length : rightIndex)
        );
      })
    : declaredClusters;
  const owner = (key: string) => {
    const id = OVERRIDES[key] || CONFIG_FIELD_CATALOG[key]?.cluster;
    return clusters.some((cluster) => cluster.id === id) ? id : "other";
  };
  return [...clusters, { id: "other", label: "其他配置与审计" }]
    .map(({ id, label }) => ({
      id,
      title: TITLES[id] || label,
      fields: fields
        .filter((field) => owner(field.key) === id)
        .sort(
          (a, b) =>
            (CONFIG_FIELD_CATALOG[a.key]?.siblingOrder ?? 0) -
            (CONFIG_FIELD_CATALOG[b.key]?.siblingOrder ?? 0),
        ),
    }))
    .filter((group) => group.fields.length > 0);
}

export function stageSummary(stage: string, id: string, draft: TrainingDraft): string {
  if (stage === "resources" || id === "preprocess") return resourceSummary(id, draft);
  const family = normalizeModelFamily(draft.model_family || "anima");
  switch (id) {
    case "models": {
      const familyLabel = FIELD_OPTION_LABELS.model_family?.[family] || String(draft.model_family || "Anima");
      const taskPart = family === "qwen_image_2_1"
        ? ` · ${String(draft.qwen_image_2_1_task || "t2i").toLowerCase() === "edit" ? "Edit" : "T2I"}`
        : "";
      const modelStem = shortStem(draft.pretrained_model_name_or_path);
      return `${familyLabel}${taskPart}${modelStem ? ` · ${modelStem}` : ""}`;
    }
    case "dataset":
      return shortStem(draft.dataset_config) ? String(draft.dataset_config).split(/[\\/]/).pop()! : "未选择数据集";
    case "captions": {
      const parts = [String(draft.caption_extension || ".txt")];
      if (isOn(draft.masked_loss)) parts.push("遮罩损失");
      if (Number(draft.caption_dropout_rate) > 0) parts.push(`丢弃 ${draft.caption_dropout_rate}`);
      return parts.join(" · ");
    }
    case "filters": {
      const parts: string[] = [];
      if (String(draft.path_pattern ?? "").trim()) parts.push(`匹配 ${draft.path_pattern}`);
      if (isOn(draft.drop_lowres_images)) parts.push(`≥ ${draft.min_pixels || 0} px`);
      return parts.length ? parts.join(" · ") : "默认不过滤";
    }
    case "cache": {
      const parts = [
        `VAE缓存${isOn(draft.use_vae_cache) ? "开启" : "关闭"}`,
        `文本缓存${isOn(draft.use_text_cache) ? "开启" : "关闭"}`,
      ];
      if (isOn(draft.force_rebuild_preprocess_cache)) parts.push("强制重建");
      return parts.join(" · ");
    }
    case "contract": {
      const mod = String(draft.network_module || "networks.lora_anima");
      const adapter = String(draft.lora_adapter_kind || "lora").toUpperCase();
      return mod === "networks.lora_anima" ? `${adapter} 适配器` : mod.split(".").pop() || mod;
    }
    case "capacity": {
      if (isOn(draft.dim_from_weights) && String(draft.network_weights ?? "").trim()) {
        return "从热启动权重读取维度";
      }
      const parts = [`rank ${draft.network_dim ?? 16}`, `alpha ${draft.network_alpha ?? 16}`];
      if (isOn(draft.train_adaln)) parts.push("AdaLN");
      return parts.join(" · ");
    }
    case "warm-start": {
      const stem = shortStem(draft.network_weights);
      return stem ? `继续自 ${stem}` : "从零初始化";
    }
    case "raw": {
      const raw = String(draft.network_args ?? "").trim();
      if (!raw || raw === "[]") return "无额外参数";
      return "含自定义参数";
    }
    case "identity":
      return String(draft.output_name || "未命名输出");
    case "volume": {
      const epochs = Number(draft.max_train_epochs || 0);
      const steps = Number(draft.max_train_steps || 0);
      const batch = Number(draft.train_batch_size || 1);
      const accum = Number(draft.gradient_accumulation_steps || 1);
      const lengthPart = epochs > 0 ? `${epochs} 轮` : `${steps} 步`;
      const batchPart = accum > 1 ? `batch ${batch}×${accum}` : `batch ${batch}`;
      return `${lengthPart} · ${batchPart}`;
    }
    case "optimizer": {
      const opt = String(draft.optimizer_type || "AdamW");
      const lr = formatLearningRate(draft.learning_rate);
      const sched = String(draft.lr_scheduler || "");
      return [opt, `lr ${lr}`, sched].filter(Boolean).join(" · ");
    }
    case "loss": {
      const sampling = String(draft.timestep_sampling || "sigmoid");
      const scheme = String(draft.weighting_scheme || "none");
      return scheme && scheme !== "none" && scheme !== "uniform"
        ? `${sampling} · 加权 ${scheme}`
        : `${sampling} · 均匀加权`;
    }
    case "regularization": {
      const active =
        Number(draft.prior_preservation_weight || 0) > 0 ||
        isOn(draft.blank_prompt_preservation) ||
        String(draft.diff_output_preservation_trigger ?? "").trim() !== "" ||
        Number(draft.inverted_mask_prior_weight || 0) > 0;
      return active ? "已启用先验保护" : "未启用先验正则";
    }
    case "preview": {
      const steps = Number(draft.sample_every_n_steps || 0);
      const epochs = Number(draft.sample_every_n_epochs || 0);
      const parts: string[] = [];
      if (steps > 0) parts.push(`每 ${steps} 步`);
      if (epochs > 0) parts.push(`每 ${epochs} 轮`);
      if (isOn(draft.sample_at_first)) parts.push("首步采样");
      return parts.length ? parts.join(" · ") : "未开启采样预览";
    }
    case "output": {
      const fmt = String(draft.save_model_as || "safetensors");
      const prec = String(draft.save_precision || "bf16");
      const everyEpochs = Number(draft.save_every_n_epochs || 0);
      const everySteps = Number(draft.save_every_n_steps || 0);
      const cadence = everySteps > 0 ? `每 ${everySteps} 步` : everyEpochs > 0 ? `每 ${everyEpochs} 轮` : "";
      return [fmt, prec, cadence].filter(Boolean).join(" · ");
    }
    case "observability": {
      const logger = String(draft.log_with || "本地日志");
      const interval = Number(draft.log_every_n_steps || 0);
      const parts = [logger];
      if (interval > 0) parts.push(`每 ${interval} 步`);
      if (isOn(draft.use_cmmd)) parts.push("CMMD");
      return parts.join(" · ");
    }
    default:
      return "";
  }
}

export function stageGroupDefaultOpen(stage: string, id: string) {
  return (
    (
      {
        input: ["models", "dataset"],
        method: ["contract", "capacity"],
        training: ["identity", "volume"],
        resources: ["compute"],
      } as Record<string, string[]>
    )[stage]?.includes(id) || false
  );
}
