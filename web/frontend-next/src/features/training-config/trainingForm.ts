import { parse } from "smol-toml";
import {
  displayConfigValue,
  prepareConfigPatch,
} from "./domain/config-values.js";
import { normalizeBooleanConfigValue } from "./domain/config-field-types.js";
import { FORM_UI_DEFAULTS } from "./domain/defaults.js";

export type TrainingFieldKind =
  "text" | "number" | "boolean" | "select" | "json";

export type TrainingFieldSpec = {
  key: string;
  label: string;
  group: "input" | "method" | "training" | "resources";
  kind: TrainingFieldKind;
  options?: string[];
  optionLabels?: Record<string, string>;
  help?: string;
  defaultValue?: string | number | boolean;
  min?: number;
  max?: number;
  step?: number | "any";
};

export const TRAINING_FIELDS: TrainingFieldSpec[] = [
  {
    key: "model_family",
    label: "模型族",
    group: "input",
    kind: "select",
    options: ["moss_image", "anima", "krea2_raw", "z_image", "qwen_image_2_1"],
    optionLabels: {
      moss_image: "MOSS-Image",
      anima: "Anima",
      krea2_raw: "Krea-2 Raw",
      z_image: "Z-Image",
      qwen_image_2_1: "Qwen-Image 2.1",
    },
  },
  {
    key: "qwen_image_2_1_task",
    label: "Qwen 任务",
    group: "input",
    kind: "select",
    options: ["t2i", "edit"],
    optionLabels: {
      t2i: "普通文生图 (t2i)",
      edit: "编辑数据集 (edit)",
    },
    defaultValue: "t2i",
  },
  {
    key: "qwen_text_encoder_cache_policy",
    label: "文本编码器缓存策略",
    group: "input",
    kind: "select",
    options: ["auto", "cpu_offload", "gpu", "cpu"],
    optionLabels: {
      auto: "自动（推荐）",
      cpu_offload: "GPU 计算＋CPU 卸载",
      gpu: "全部驻留 GPU",
      cpu: "仅 CPU",
    },
    defaultValue: "auto",
  },
  {
    key: "qwen_sample_swap_mode",
    label: "Qwen 预览块交换",
    group: "training",
    kind: "select",
    options: ["inherit", "custom"],
    optionLabels: {
      inherit: "继承训练配置",
      custom: "自定义数量",
    },
    defaultValue: "inherit",
  },
  {
    key: "qwen_sample_blocks_to_swap",
    label: "Qwen 预览交换块数",
    group: "training",
    kind: "number",
    min: 0,
    max: 30,
    step: 1,
    defaultValue: 0,
  },
  {
    key: "sample_vae_decode",
    label: "预览 VAE 解码",
    group: "training",
    kind: "select",
    options: ["auto", "tiled", "cpu"],
    optionLabels: {
      auto: "自动（OOM 回退）",
      tiled: "分块 GPU 解码（推荐）",
      cpu: "CPU 解码",
    },
    defaultValue: "tiled",
  },
  {
    key: "pretrained_model_name_or_path",
    label: "DiT 模型路径",
    group: "input",
    kind: "text",
  },
  { key: "qwen3", label: "Qwen3 文本编码器", group: "input", kind: "text" },
  { key: "vocab_pack_path", label: "Anima vocab pack 路径", group: "input", kind: "text" },
  { key: "vae", label: "VAE 路径", group: "input", kind: "text" },
  { key: "dataset_config", label: "数据集配置", group: "input", kind: "text" },
  {
    key: "caption_extension",
    label: "标注扩展名",
    group: "input",
    kind: "text",
  },
  {
    key: "use_vae_cache",
    label: "使用 VAE 缓存",
    group: "input",
    kind: "boolean",
  },
  {
    key: "use_text_cache",
    label: "使用文本缓存",
    group: "input",
    kind: "boolean",
  },
  { key: "network_module", label: "网络模块", group: "method", kind: "text" },
  {
    key: "network_dim",
    label: "LoRA rank",
    group: "method",
    kind: "number",
    min: 1,
    step: 1,
  },
  {
    key: "network_alpha",
    label: "LoRA alpha",
    group: "method",
    kind: "number",
    min: 0,
    step: 1,
  },
  {
    key: "network_train_unet_only",
    label: "仅训练 DiT",
    group: "method",
    kind: "boolean",
  },
  {
    key: "network_weights",
    label: "热启动权重",
    group: "method",
    kind: "text",
  },
  { key: "output_name", label: "输出名称", group: "training", kind: "text" },
  {
    key: "max_train_steps",
    label: "最大训练步数",
    group: "training",
    kind: "number",
    min: 1,
    step: 1,
  },
  {
    key: "max_train_epochs",
    label: "最大训练轮数",
    group: "training",
    kind: "number",
    min: 1,
    step: 1,
  },
  {
    key: "train_batch_size",
    label: "训练批大小",
    group: "training",
    kind: "number",
    min: 1,
    step: 1,
  },
  {
    key: "gradient_accumulation_steps",
    label: "梯度累积",
    group: "training",
    kind: "number",
    min: 1,
    step: 1,
  },
  {
    key: "learning_rate",
    label: "学习率",
    group: "training",
    kind: "number",
    min: 0,
    step: 0.000001,
  },
  {
    key: "optimizer_type",
    label: "优化器",
    group: "training",
    kind: "select",
    options: ["AdamW", "AdamW8bit", "Prodigy"],
  },
  {
    key: "lr_scheduler",
    label: "学习率调度器",
    group: "training",
    kind: "select",
    options: ["constant", "cosine", "linear"],
  },
  {
    key: "timestep_sampling",
    label: "时间步采样",
    group: "training",
    kind: "select",
    options: ["uniform", "sigmoid", "shift"],
  },
  {
    key: "seed",
    label: "随机种子",
    group: "training",
    kind: "number",
    min: 0,
    step: 1,
  },
  { key: "output_dir", label: "输出目录", group: "training", kind: "text" },
  {
    key: "save_every_n_steps",
    label: "权重保存间隔",
    group: "training",
    kind: "number",
    min: 0,
    step: 1,
  },
  {
    key: "mixed_precision",
    label: "混合精度",
    group: "resources",
    kind: "select",
    options: ["bf16", "fp16", "no"],
  },
  {
    key: "adaptive_precision",
    label: "自适应精度策略",
    group: "resources",
    kind: "select",
    options: ["off", "auto", "fp16_fp32"],
  },
  {
    key: "adaptive_fp32_modules",
    label: "自适应 FP32 模块",
    group: "resources",
    kind: "json",
  },
  {
    key: "adaptive_loss_scale",
    label: "自适应初始 Loss Scale",
    group: "resources",
    kind: "number",
    min: 1,
    step: "any",
  },
  {
    key: "adaptive_oom_retry",
    label: "OOM 自动重试",
    group: "resources",
    kind: "boolean",
  },
  {
    key: "adaptive_oom_retry_max_attempts",
    label: "OOM 最大尝试次数",
    group: "resources",
    kind: "number",
    min: 1,
    step: 1,
  },
  {
    key: "adaptive_oom_retry_swap_increment",
    label: "OOM 起始交换块增量",
    group: "resources",
    kind: "number",
    min: 1,
    step: 1,
  },
  {
    key: "adaptive_oom_retry_max_swap",
    label: "OOM 交换块预算（0=自动）",
    group: "resources",
    kind: "number",
    min: 0,
    step: 1,
  },
  {
    key: "adaptive_oom_retry_timeout",
    label: "OOM 单次尝试超时（秒）",
    group: "resources",
    kind: "number",
    min: 1,
    step: "any",
  },
  {
    key: "base_compute",
    label: "底模计算路径",
    group: "resources",
    kind: "select",
    options: ["bf16", "nf4", "fp8"],
  },
  {
    key: "attn_mode",
    label: "Attention 后端",
    group: "resources",
    kind: "select",
    options: ["flash", "torch", "sdpa"],
  },
  {
    key: "gradient_checkpointing",
    label: "梯度检查点",
    group: "resources",
    kind: "boolean",
  },
  {
    key: "selective_checkpoint",
    label: "选择性检查点",
    group: "resources",
    kind: "select",
    options: ["off", "every_other"],
  },
  {
    key: "blocks_to_swap",
    label: "Block swap 数量",
    group: "resources",
    kind: "number",
    min: 0,
    step: 1,
  },
  {
    key: "auto_block_swap_vram_reserve_percent",
    label: "保留显存（总容量 %）",
    group: "resources",
    kind: "number",
    min: 0,
    max: 90,
    step: 0.1,
  },
  {
    key: "auto_block_swap_preference",
    label: "显存 / 内存倾向",
    group: "resources",
    kind: "select",
    options: ["balanced", "vram", "ram"],
    optionLabels: {
      balanced: "均衡",
      vram: "优先节省显存",
      ram: "优先节省内存",
    },
  },
  {
    key: "torch_compile",
    label: "Torch compile",
    group: "resources",
    kind: "boolean",
  },
  {
    key: "max_data_loader_n_workers",
    label: "DataLoader workers",
    group: "resources",
    kind: "number",
    min: 0,
    step: 1,
  },
];

export type TrainingDraft = Record<string, string | number | boolean>;

export function sameTrainingValue(
  a: TrainingDraft[string],
  b: TrainingDraft[string],
  kind?: TrainingFieldKind,
) {
  if (Object.is(a, b)) return true;
  if (kind === "number" && a !== "" && b !== "") return Number(a) === Number(b);
  if (kind === "json") {
    try {
      return (
        JSON.stringify(JSON.parse(String(a))) ===
        JSON.stringify(JSON.parse(String(b)))
      );
    } catch {
      return false;
    }
  }
  return false;
}

export function draftFromMerged(
  config: Record<string, unknown>,
  fields = TRAINING_FIELDS,
): TrainingDraft {
  const draft = Object.fromEntries(
    fields.map((field) => [
      field.key,
      formValue(field, displayConfigValue(field.key, config)),
    ]),
  ) as TrainingDraft;
  // Resume is not a form control, but hotstart must be able to clear a saved resume.
  if (typeof config.resume === "string") draft.resume = config.resume;
  return draft;
}

export function restoreKnownFormDefaults(
  draft: TrainingDraft,
  fields: TrainingFieldSpec[],
  editableKeys: ReadonlySet<string>,
): TrainingDraft {
  const defaults = FORM_UI_DEFAULTS as Record<string, unknown>;
  const changes = Object.fromEntries(
    fields
      .filter((field) => editableKeys.has(field.key) && Object.hasOwn(defaults, field.key))
      .map((field) => {
        const value = defaults[field.key];
        return [
          field.key,
          field.kind === "json" ? JSON.stringify(value, null, 2) : value,
        ];
      }),
  ) as TrainingDraft;
  return { ...draft, ...changes };
}

export function trainingPatchValues(
  draft: TrainingDraft,
  baseline: TrainingDraft,
  fields = TRAINING_FIELDS,
  original: Record<string, unknown> = {},
) {
  const kinds = new Map(fields.map((field) => [field.key, field.kind]));
  const changes = Object.fromEntries(
    Object.entries(draft)
      .filter(
        ([key, value]) =>
          !sameTrainingValue(value, baseline[key], kinds.get(key)),
      )
      .map(([key, value]) => [
        key,
        kinds.get(key) === "json" ? JSON.parse(String(value)) : value,
      ]),
  );
  // Start from explicitly edited raw arguments when both editors change the same payload.
  return prepareConfigPatch(changes, {
    ...original,
    ...(changes.network_args ? { network_args: changes.network_args } : {}),
  });
}

export function rawConfigOwnKeys(content: string) {
  try {
    return new Set(
      Object.entries(parse(content))
        .filter(
          ([, value]) => typeof value !== "object" || Array.isArray(value),
        )
        .map(([key]) => key),
    );
  } catch {
    return new Set<string>();
  }
}

export function importedTrainingPath(name: string) {
  const stem = name
    .trim()
    .replace(/\.toml$/i, "")
    .replace(/[^\w\u3400-\u9fff.-]+/g, "_")
    .replace(/^[._-]+|[._-]+$/g, "");
  return stem ? `configs/imported/${stem}.toml` : "";
}

function formValue(
  field: TrainingFieldSpec,
  value: unknown,
): string | number | boolean {
  const resolved = value === undefined || value === null ||
    (value === "" && field.defaultValue !== undefined)
    ? field.defaultValue
    : value;
  if (field.kind === "json") return JSON.stringify(resolved ?? [], null, 2);
  if (field.kind === "boolean") return normalizeBooleanConfigValue(field.key, resolved);
  if (field.kind === "number")
    return resolved === undefined || resolved === null || resolved === ""
      ? ""
      : Number(resolved);
  if (resolved === undefined || resolved === null) return "";
  return String(resolved);
}
