import { apiRequest } from "../../api/client";

/** t1 诊断端点的 payload 契约（web/services/training/distributed.py）。 */
export type DistributedCheck = {
  level: "error" | "warning" | "info" | "ok" | string;
  key: string;
  code?: string;
  /** Human wording shown as the primary line. */
  message: string;
  /** Raw error text for this finding, shown behind a folded disclosure. */
  detail?: string;
};

export type DistributedGpuSnapshot = {
  index: number;
  uuid: string;
  name: string;
  memory_total_mb: number | null;
  capability: [number, number] | null;
};

export type WiredPpFamily = {
  family: string;
  display_name: string;
  wired: boolean;
  runtime_backend: string | null;
  stages: number[];
  schedules: string[];
};

export type DistributedDiagnosis = {
  ok: boolean;
  mode: string;
  modes: string[];
  world_size: number | null;
  wired_families: WiredPpFamily[];
  checks: DistributedCheck[];
  hardware: { gpus: DistributedGpuSnapshot[]; risks: DistributedCheck[] };
};

export type DistributedSettings = {
  mode: "single" | "ddp" | "pp";
  gpu_whitelist: number[];
  pipeline_parallel_stages: number;
  pipeline_parallel_microbatches: number;
  pipeline_parallel_schedule: string;
  pipeline_parallel_split: string;
  pipeline_parallel_blocks_to_swap: number;
};

export type DistributedSettingsPayload = Partial<DistributedSettings>;

export async function fetchDistributedDiagnosis(
  input: {
    variant: string;
    preset: string;
    methods_subdir?: string;
    config_file?: string;
    gpu_whitelist?: string[];
    mode?: string;
  },
  signal?: AbortSignal,
): Promise<DistributedDiagnosis> {
  return apiRequest<DistributedDiagnosis>(
    "/api/training/distributed/diagnosis",
    {
      method: "POST",
      body: JSON.stringify({
        methods_subdir: "gui-methods",
        ...input,
      }),
      signal,
    },
  );
}

export async function fetchDistributedSettings(
  signal?: AbortSignal,
): Promise<DistributedSettings> {
  return apiRequest<DistributedSettings>("/api/settings/distributed", { signal });
}

export async function saveDistributedSettings(
  values: DistributedSettingsPayload,
): Promise<DistributedSettings> {
  return apiRequest<DistributedSettings>("/api/settings/distributed", {
    method: "PUT",
    body: JSON.stringify(values),
  });
}
