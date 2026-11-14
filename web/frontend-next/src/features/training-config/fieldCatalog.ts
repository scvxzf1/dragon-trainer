import {
  FIELD_LABEL_ZH,
  FIELD_OPTIONS,
} from "./domain/labels-options.js";
import { configFieldInputKind, normalizeBooleanConfigValue } from "./domain/config-field-types.js";
import { CONFIG_FIELD_CATALOG } from "./domain/config-field-catalog.js";
import { configFieldDisclosure } from "./domain/config-field-disclosure-rules.js";
import { displayConfigValue } from "./domain/config-values.js";
import {
  configFieldAvailability,
  resolveConfigAdapterKind,
} from "./domain/config-field-availability.js";
import {
  configureModelFamilyCapabilities,
  modelFamilyOptionSupported,
  normalizeModelFamily,
} from "./domain/model-family.js";
import { apiRequest } from "../../api/client";
import {
  sameTrainingValue,
  TRAINING_FIELDS,
  type TrainingDraft,
  type TrainingFieldSpec,
} from "./trainingForm";

export type FieldAvailability = {
  visible: boolean;
  enabled: boolean;
  reason: string;
  code: string | null;
};

// These branches are unrelated to the current method/adapter and belong in
// the audit view. Conflicts that a user can resolve stay visible but disabled.
const HIDDEN_AVAILABILITY_CODES = new Set([
  "method-context",
  "method-scope",
  "spd-method",
  "chimera-method",
  "ip-adapter-method",
  "adapter-family",
  "lokr-adapter",
  "vera-adapter",
  "dora-adapter",
]);
const EXPLICIT_TRAINING_FIELD_KEYS = new Set(TRAINING_FIELDS.map((field) => field.key));

// Reuse metadata only; none of the legacy DOM runtime is mounted in React.
export function fieldsForConfig(
  config: Record<string, unknown>,
): TrainingFieldSpec[] {
  const fields = new Map(TRAINING_FIELDS.map((field) => [field.key, field]));
  for (const key of new Set([
    ...Object.keys(CONFIG_FIELD_CATALOG),
    ...Object.keys(config),
  ])) {
    if (fields.has(key)) continue;
    if (key === "general" || key === "datasets" || key.includes(".")) continue;
    const value = displayConfigValue(key, config);
    const options = FIELD_OPTIONS[key];
    const kind = configFieldInputKind(key, value, options);
    fields.set(key, {
      key,
      label: FIELD_LABEL_ZH[key] || key,
      group: CONFIG_FIELD_CATALOG[key]?.stage || "training",
      kind,
      options: options?.map(String),
      step: kind === "number" ? "any" : undefined,
    });
  }
  return [...fields.values()].map((field) => {
    if (CONFIG_FIELD_CATALOG[field.key]?.cluster === "preprocess") field = { ...field, group: "input" };
    const options = FIELD_OPTIONS[field.key];
    return field.kind === "select" &&
      options?.every((value) => typeof value === "string")
      ? { ...field, options: options as string[] }
      : field;
  });
}

export async function fetchFieldCapabilities(signal?: AbortSignal) {
  const payload = await apiRequest<{ items: unknown[] }>(
    "/api/config/model-families",
    { signal },
  );
  configureModelFamilyCapabilities(payload);
  return payload;
}

export function fieldAvailability(
  key: string,
  values: Record<string, unknown>,
  method: string,
): FieldAvailability {
  if (key === "qwen_image_2_1_task" && normalizeModelFamily(values.model_family) !== "qwen_image_2_1") {
    return {
      visible: false,
      enabled: false,
      reason: "Qwen 任务仅适用于 qwen_image_2_1；先选择该模型族。",
      code: "qwen-task-family",
    };
  }
  const context = {
    values,
    method,
    adapter: resolveConfigAdapterKind(values),
    modelFamily: values.model_family || "anima",
    baseCompute: values.base_compute || "bf16",
    maxTrainEpochsConfigured: Number(values.max_train_epochs) > 0,
    dimFromWeights: normalizeBooleanConfigValue("dim_from_weights", values.dim_from_weights),
    networkWeights: String(values.network_weights || ""),
  };
  const result = configFieldAvailability(key, context) as {
    visible?: boolean;
    enabled?: boolean;
    reason?: string;
    code?: string | null;
  };
  const disclosure = CONFIG_FIELD_CATALOG[key] || !EXPLICIT_TRAINING_FIELD_KEYS.has(key)
    ? configFieldDisclosure(key, context)
    : { visible: true, reason: "", code: null };
  const hidden = disclosure.visible === false;
  const code = hidden ? disclosure.code : result.code || null;
  const visible = !hidden && (result.visible ?? !HIDDEN_AVAILABILITY_CODES.has(code || ""));
  return {
    visible,
    enabled: visible && result.enabled !== false,
    reason: hidden ? disclosure.reason : result.reason || "",
    code,
  };
}

export function availableFieldOptions(
  field: TrainingFieldSpec,
  family: string,
  values?: Record<string, unknown>,
) {
  return field.options?.filter((option) =>
    modelFamilyOptionSupported(field.key, family, option, { values }),
  );
}

export function filterTrainingFields(
  fields: TrainingFieldSpec[],
  draft: TrainingDraft,
  baseline: TrainingDraft,
  search: string,
  view: string,
  method: string,
) {
  const query = search.trim().toLowerCase();
  return fields.filter((field) => {
    const changed = !sameTrainingValue(
      draft[field.key],
      baseline[field.key],
      field.kind,
    );
    const availability = fieldAvailability(field.key, draft, method);
    const matches = !query || `${field.label} ${field.key}`.toLowerCase().includes(query);
    if (!matches) return false;
    // MOSS-only controls must not become disabled cards in audit views.
    if (availability.code === "moss-token-cap-family" || availability.code === "moss-te-family") return false;
    if (availability.code === "auto-block-swap-disabled") return false;
    if (view === "all") return true;
    if (view === "changed") return changed;
    return availability.visible;
  });
}
