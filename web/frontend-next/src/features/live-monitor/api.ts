import { apiRequest } from "../../api/client";
import { fetchHistoryImages, type HistoryLogRecord, type HistoryTaskSummary } from "../training-history/api";

export type TrainingStatus = {
  status?: string;
  variant?: string;
  preset?: string;
  job?: string;
  output_dir?: string;
  task_id?: string;
  gpu_whitelist?: number[];
  last_output_at?: string | number;
  last_log_line?: string;
  last_log_id?: number;
  log_count?: number;
  metric_count?: number;
  latest_progress?: Record<string, unknown> & {
    current?: number;
    total?: number;
    loss?: number;
    lr?: number;
    rate?: string;
  };
  latest_metric?: Record<string, unknown>;
  latest_system?: Record<string, unknown> & {
    vram_used_gb?: number;
    vram_total_gb?: number;
    gpu_temp?: number;
    gpu_util?: number;
  };
  error_hint?: string;
  anomaly_message?: string;
  // Present only on the final "status" broadcast after the process exits.
  state?: string;
  message?: string;
};

export type LogRecord = {
  id?: number;
  ts?: number;
  line?: string;
  type?: string;
  level?: string;
};

export type GpuInfo = Record<string, unknown> & {
  index?: number;
  uuid?: string;
  name?: string;
  memory_used_gb?: number;
  memory_total_gb?: number;
  gpu_util?: number;
  gpu_temp?: number;
};
export type GpuInventory = { ok?: boolean; gpus?: GpuInfo[]; sampled_at?: number | null; stale?: boolean };

export const liveMonitorKeys = {
  status: ["live-monitor", "status"] as const,
  metrics: ["live-monitor", "metrics"] as const,
  logs: ["live-monitor", "logs"] as const,
  gpus: ["live-monitor", "gpus"] as const,
};

export function fetchTrainingStatus(signal?: AbortSignal) {
  return apiRequest<TrainingStatus>("/api/training/status", { signal });
}

export async function fetchTrainingMetrics(taskId?: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ all: "1" });
  if (taskId) query.set("task_id", taskId);
  const points = await apiRequest<Record<string, unknown>[]>(
    `/api/training/metrics?${query.toString()}`,
    { signal },
  );
  return points;
}

export async function fetchTrainingLogs(limit = 300, taskId?: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ limit: String(limit) });
  if (taskId) query.set("task_id", taskId);
  const data = await apiRequest<{ records: LogRecord[] }>(
    `/api/training/logs?${query.toString()}`,
    { signal },
  );
  return { ...data, records: data.records.slice(-limit) };
}

export function fetchGpus(signal?: AbortSignal, force = false) {
  return apiRequest<GpuInventory>(`/api/training/gpus${force ? "?refresh=1" : ""}`, {
    signal,
  });
}

export function stopTraining(taskId: string) {
  if (!taskId.trim()) throw new Error("任务身份未确认，不能停止训练");
  return apiRequest<{ ok?: boolean; message?: string }>("/api/training/stop", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId }),
    headers: { "Content-Type": "application/json" },
  });
}

// History endpoints reuse the training-history feature's typed clients; the
// monitor keeps a dedicated query key family so caches never mix by task.
export const monitorResultKeys = {
  meta: (taskId: string) => ["live-monitor", "result-meta", taskId] as const,
  images: (taskId: string) => ["live-monitor", "result-images", taskId] as const,
  logs: (taskId: string) => ["live-monitor", "result-logs", taskId] as const,
};

export function fetchHistoryResultMeta(taskId: string, signal?: AbortSignal) {
  return apiRequest<{ task?: HistoryTaskSummary }>(
    `/api/training/history/${encodeURIComponent(taskId)}`,
    { signal },
  );
}

export function fetchResultSampleImages(taskId: string, signal?: AbortSignal) {
  return fetchHistoryImages(taskId, signal, 60);
}

export function fetchResultHistoryLogs(taskId: string, after = 0, limit = 1000, signal?: AbortSignal) {
  const query = new URLSearchParams({ offset: String(after), limit: String(limit) });
  return apiRequest<{ ok?: boolean; total?: number; records: HistoryLogRecord[] }>(
    `/api/training/history/${encodeURIComponent(taskId)}/logs?${query.toString()}`,
    { signal },
  );
}
