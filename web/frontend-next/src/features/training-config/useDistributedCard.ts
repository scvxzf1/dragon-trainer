import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchDistributedDiagnosis,
  fetchDistributedSettings,
  saveDistributedSettings,
} from "./distributedApi";
import type { DistributedSettingsPayload } from "./distributedApi";
import {
  normalizeDistributedMode,
  ppWiredForFamily,
  type DistributedMode,
} from "./distributedDiagnosis";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";

type Props = {
  state: ReturnType<typeof useTrainingWorkspace>;
  /** Popover 展开时才拉取诊断（只读端点）。 */
  open: boolean;
};

export function useDistributedCard({ state, open }: Props) {
  const queryClient = useQueryClient();
  const device = state.deviceState;
  const file = state.selectedFile;
  const draft = state.draft;
  const pipeline = draft.pipeline_parallel === true || draft.pipeline_parallel === "true";
  const mode: DistributedMode = pipeline ? "pp" : normalizeDistributedMode(device.selection.mode);
  const family = String(draft.model_family || state.mergedConfig?.model_family || "anima");

  const settingsQuery = useQuery({
    queryKey: ["distributed-settings"],
    queryFn: ({ signal }) => fetchDistributedSettings(signal),
    staleTime: 30_000,
  });

  const diagnosisQuery = useQuery({
    queryKey: [
      "distributed-diagnosis",
      file?.path || "",
      state.context.selectedPreset,
      mode,
      device.gpuIds.join(","),
    ],
    queryFn: ({ signal }) => fetchDistributedDiagnosis({
      variant: file?.method || "lora",
      preset: state.context.selectedPreset,
      methods_subdir: file?.methods_subdir || "gui-methods",
      config_file: file?.path,
      gpu_whitelist: device.gpuIds,
      mode,
    }, signal),
    enabled: open && Boolean(file),
  });

  const settingsSave = useMutation({
    mutationFn: (values: DistributedSettingsPayload) => saveDistributedSettings(values),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["distributed-settings"] });
    },
  });

  const wiredFamilies = diagnosisQuery.data?.wired_families;
  const ppAvailable = useMemo(
    () => ppWiredForFamily(family, wiredFamilies) && device.devices.length >= 2,
    [family, wiredFamilies, device.devices.length],
  );
  const ppBlockedReason = !ppWiredForFamily(family, wiredFamilies)
    ? `模型族 ${family} 尚未接入流水线并行（仅 anima / moss_image 支持）`
    : device.devices.length < 2
      ? "流水线并行至少需要两张 GPU"
      : "";

  function changeMode(next: DistributedMode) {
    if (next === "pp") {
      if (!ppAvailable) return;
      device.setMode("ddp");
      const settings = settingsQuery.data;
      state.setDraft((current) => ({
        ...current,
        pipeline_parallel: true,
        pipeline_parallel_stages: settings?.pipeline_parallel_stages ?? 2,
        pipeline_parallel_microbatches: settings?.pipeline_parallel_microbatches ?? 4,
        pipeline_parallel_schedule: settings?.pipeline_parallel_schedule ?? "1f1b",
        pipeline_parallel_split: settings?.pipeline_parallel_split ?? "balanced",
      }));
      settingsSave.mutate({
        mode: "pp",
        gpu_whitelist: device.gpuIds.map((id) => Number(id)),
      });
    } else {
      device.setMode(next);
      if (pipeline) {
        state.setDraft((current) => ({ ...current, pipeline_parallel: false }));
      }
      settingsSave.mutate({
        mode: next,
        gpu_whitelist: next === "single"
          ? device.gpuIds.slice(0, 1).map((id) => Number(id))
          : device.gpuIds.map((id) => Number(id)),
      });
    }
    state.preflight.reset();
  }

  function updatePpNumber(key: "pipeline_parallel_stages" | "pipeline_parallel_microbatches", raw: string) {
    const value = Number.parseInt(raw, 10);
    if (!Number.isFinite(value) || value < 1) return;
    state.setDraft((current) => ({ ...current, [key]: value }));
    settingsSave.mutate({ [key]: value });
  }

  function selectDevice(gpu: Parameters<typeof device.selectDevice>[0]) {
    device.selectDevice(gpu);
    state.preflight.reset();
  }

  return {
    mode,
    family,
    pipeline,
    devices: device.devices,
    selection: device.selection,
    selectDevice,
    summary: device.summary,
    issue: device.issue,
    refreshDevices: device.refreshDevices,
    busy: state.busy,
    settings: settingsQuery.data,
    diagnosis: diagnosisQuery.data,
    diagnosisPending: diagnosisQuery.isPending || diagnosisQuery.isFetching,
    diagnosisError: diagnosisQuery.error instanceof Error ? diagnosisQuery.error.message : "",
    ppAvailable,
    ppBlockedReason,
    wiredFamilies,
    changeMode,
    updatePpNumber,
    settingsSaving: settingsSave.isPending,
    settingsError: settingsSave.error instanceof Error ? settingsSave.error.message : "",
  };
}
