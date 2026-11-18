import { normalizeBooleanConfigValue } from "./domain/config-field-types.js";
import { normalizeModelFamily } from "./domain/model-family.js";
import type { FieldAvailability } from "./fieldCatalog";
import {
  sameTrainingValue,
  type TrainingDraft,
  type TrainingFieldSpec,
} from "./trainingForm";

export type FieldHierarchyItem = {
  field: TrainingFieldSpec;
  children: TrainingFieldSpec[];
};

export type ClusterFieldPartition = {
  items: FieldHierarchyItem[];
  drawerFields: TrainingFieldSpec[];
  drawerOwnCount: number;
};

const PIPELINE_TOPOLOGY_KEYS = new Set([
  "pipeline_parallel",
  "pipeline_parallel_stages",
  "pipeline_parallel_microbatches",
  "pipeline_parallel_blocks_to_swap",
  "pipeline_parallel_schedule",
  "pipeline_parallel_split",
]);

const LOW_FREQ_ADVANCED_KEYS = new Set([
  "skip_cache_check",
  "reuse_dataset_cache_copy",
  "reuse_vae_latents",
  "reuse_text_encoder_cache",
  "cache_fingerprint_mode",
  "force_rebuild_preprocess_cache",
  "preprocess_precision_preference",
  "dataloader_pin_memory",
  "persistent_data_loader_workers",
  "vae_chunk_size",
  "vae_disable_cache",
  "cpu_offload_checkpointing",
  "unsloth_offload_checkpointing",
  "activation_memory_budget",
  "checkpointing_epochs",
  "checkpointing_last_n_epochs",
  "logging_dir",
  "use_cmmd",
  "block_swap_profile_jsonl",
]);

const PARENT_BY_CHILD: Record<string, string> = {
  min_pixels: "drop_lowres_images",
  dim_from_weights: "network_weights",
  discrete_flow_shift: "timestep_sampling",
  min_snr_gamma: "weighting_scheme",
  p2_gamma: "weighting_scheme",
  p2_k: "weighting_scheme",
  sigmoid_scale: "weighting_scheme",
  sigmoid_bias: "weighting_scheme",
  diff_output_preservation_class: "diff_output_preservation_trigger",
  qwen_sample_blocks_to_swap: "qwen_sample_swap_mode",
  nf4_dir1: "base_compute",
  krea2_attn_fp16: "attn_mode",
  auto_block_swap_mode: "auto_block_swap",
  auto_block_swap_interval: "auto_block_swap",
  auto_block_swap_vram_reserve_percent: "auto_block_swap",
  auto_block_swap_preference: "auto_block_swap",
  auto_block_swap_max_trials: "auto_block_swap",
  auto_block_swap_timeout: "auto_block_swap",
  auto_block_swap_swap_io_limit_mb: "auto_block_swap",
  block_swap_transfer_dtype: "blocks_to_swap",
  block_swap_restore_mode: "blocks_to_swap",
  disable_block_swap_for_eval: "blocks_to_swap",
  block_swap_int8_restore_mode: "block_swap_transfer_dtype",
  block_swap_int8_staging_ring: "block_swap_transfer_dtype",
  block_swap_int8_dual_stream: "block_swap_transfer_dtype",
  block_swap_int8_restore_chunk_rows: "block_swap_transfer_dtype",
  selective_checkpoint_blocks: "selective_checkpoint",
  compile_block_scope: "torch_compile",
  compile_inductor_mode: "torch_compile",
  compile_dynamic_seq: "torch_compile",
  compile_seq_bands: "compile_dynamic_seq",
  adaptive_fp32_modules: "adaptive_precision",
  adaptive_loss_scale: "adaptive_precision",
  adaptive_oom_retry_max_attempts: "adaptive_oom_retry",
  adaptive_oom_retry_swap_increment: "adaptive_oom_retry",
  adaptive_oom_retry_max_swap: "adaptive_oom_retry",
  adaptive_oom_retry_timeout: "adaptive_oom_retry",
  memory_probe_max_steps: "memory_probe_jsonl",
  peak_probe_max_steps: "peak_probe_jsonl",
  peak_probe_level: "peak_probe_jsonl",
  gradient_flow_probe_every_n_steps: "gradient_flow_probe_jsonl",
  gradient_flow_probe_dense_steps: "gradient_flow_probe_jsonl",
};

const PREVIEW_DEPENDENT_KEYS = new Set([
  "sample_sampler",
  "sample_vae_decode",
  "qwen_sample_swap_mode",
  "qwen_sample_blocks_to_swap",
]);

function boolVal(key: string, draft: Record<string, unknown>, fallback = false) {
  return normalizeBooleanConfigValue(key, draft[key], fallback);
}

function strVal(value: unknown, fallback = "") {
  return String(value ?? fallback).trim().toLowerCase();
}

export function isScopeHiddenInApplicable(
  field: TrainingFieldSpec,
  draft: TrainingDraft,
  baseline: TrainingDraft,
): boolean {
  const { key, kind } = field;
  if (!sameTrainingValue(draft[key], baseline[key], kind)) return false;
  const family = normalizeModelFamily(draft.model_family || "anima");
  const baseCompute = strVal(draft.base_compute, "bf16");

  if (key === "precision_preference") return true;
  if (key === "use_custom_down_autograd" && !boolVal(key, draft)) return true;
  if (PIPELINE_TOPOLOGY_KEYS.has(key) && !boolVal("pipeline_parallel", draft)) return true;
  if (key === "cache_llm_adapter_outputs" && family !== "anima" && (family !== "moss_image" || !boolVal(key, draft))) {
    return true;
  }
  if (key === "v100_flash_stability" && family !== "anima" && strVal(draft[key], "off") === "off") return true;
  if ((key === "compile_dynamic_seq" || key === "compile_seq_bands") && family !== "anima" && !boolVal(key, draft)) {
    return true;
  }
  if (key === "krea2_attn_fp16" && family !== "krea2_raw" && !boolVal(key, draft)) return true;
  if (key === "nf4_dir1" && baseCompute !== "nf4" && !boolVal(key, draft)) return true;
  return false;
}

function isChildConditionActive(
  key: string,
  draft: TrainingDraft,
  availability: FieldAvailability,
): boolean {
  if (!availability.enabled) return false;
  const sampling = strVal(draft.timestep_sampling, "sigmoid");
  const weighting = strVal(draft.weighting_scheme, "none");
  const previewActive =
    Number(draft.sample_every_n_epochs || 0) > 0 ||
    Number(draft.sample_every_n_steps || 0) > 0 ||
    boolVal("sample_at_first", draft);

  if (PREVIEW_DEPENDENT_KEYS.has(key) && !previewActive) return false;
  if (key === "min_pixels") return boolVal("drop_lowres_images", draft);
  if (key === "dim_from_weights") return Boolean(strVal(draft.network_weights));
  if (key === "discrete_flow_shift") return sampling === "shift" || sampling === "flux_shift";
  if (key === "min_snr_gamma") return weighting === "min_snr";
  if (key === "p2_gamma" || key === "p2_k") return weighting === "p2";
  if (key === "sigmoid_scale" || key === "sigmoid_bias") return weighting === "logit_normal";
  if (key === "diff_output_preservation_class") return Boolean(strVal(draft.diff_output_preservation_trigger));
  if (key === "nf4_dir1") return strVal(draft.base_compute, "bf16") === "nf4" || boolVal(key, draft);
  if (key === "krea2_attn_fp16") {
    const attn = strVal(draft.attn_mode, "flash");
    return attn === "torch" || attn === "sdpa" || boolVal(key, draft);
  }
  if (key === "adaptive_fp32_modules" || key === "adaptive_loss_scale") {
    return strVal(draft.adaptive_precision, "off") !== "off";
  }
  if (key.startsWith("adaptive_oom_retry_")) return boolVal("adaptive_oom_retry", draft);
  return true;
}

export function partitionClusterFields(
  fields: TrainingFieldSpec[],
  options: {
    draft: TrainingDraft;
    baseline: TrainingDraft;
    defaults?: TrainingDraft;
    ownKeys: Set<string>;
    getAvailability: (key: string) => FieldAvailability;
    flattenAll?: boolean;
  },
): ClusterFieldPartition {
  const { draft, baseline, defaults, ownKeys, getAvailability, flattenAll } = options;
  if (flattenAll) {
    return { items: fields.map((field) => ({ field, children: [] })), drawerFields: [], drawerOwnCount: 0 };
  }
  const byKey = new Map(fields.map((field) => [field.key, field]));
  const childBuckets = new Map<string, TrainingFieldSpec[]>();
  const drawerFields: TrainingFieldSpec[] = [];
  const consumed = new Set<string>();

  for (const field of fields) {
    const { key, kind } = field;
    const dirty = !sameTrainingValue(draft[key], baseline[key], kind);
    const overridden = defaults !== undefined && Object.hasOwn(defaults, key) && !sameTrainingValue(draft[key], defaults[key], kind);
    const availability = getAvailability(key);
    const parentKey = PARENT_BY_CHILD[key];

    if (parentKey && byKey.has(parentKey)) {
      if (isChildConditionActive(key, draft, availability) || dirty) {
        const targetParent = parentKey === "compile_dynamic_seq" && byKey.has("torch_compile") ? "torch_compile"
          : parentKey === "block_swap_transfer_dtype" && byKey.has("blocks_to_swap") ? "blocks_to_swap"
          : parentKey;
        const list = childBuckets.get(targetParent) || [];
        list.push(field);
        childBuckets.set(targetParent, list);
      } else {
        drawerFields.push(field);
      }
      consumed.add(key);
      continue;
    }

    if (PREVIEW_DEPENDENT_KEYS.has(key) && !isChildConditionActive(key, draft, availability) && !dirty && !overridden) {
      drawerFields.push(field);
      consumed.add(key);
      continue;
    }

    if (LOW_FREQ_ADVANCED_KEYS.has(key) && !dirty && !overridden && !ownKeys.has(key)) {
      drawerFields.push(field);
      consumed.add(key);
    }
  }

  const items = fields
    .filter((field) => !consumed.has(field.key))
    .map((field) => ({ field, children: childBuckets.get(field.key) || [] }));

  return {
    items,
    drawerFields,
    drawerOwnCount: drawerFields.filter((field) => ownKeys.has(field.key)).length,
  };
}
