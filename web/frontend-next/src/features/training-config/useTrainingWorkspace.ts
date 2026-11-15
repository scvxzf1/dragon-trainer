import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { trainingContextKeys } from "../../api/trainingContext";
import type { TrainingConfigGroup } from "../../api/trainingContext";
import { useTrainingContext } from "../../app/useTrainingContext";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import { datasetKeys } from "../dataset-editor/api";
import { useTrainingDevices } from "./useTrainingDevices";
import { ppWiredForFamily } from "./distributedDiagnosis";
import { useUnsavedChangesGuard } from "../dataset-editor/useUnsavedChangesGuard";
import { ApiError } from "../../api/client";
import {
  fetchRawTrainingConfig,
  previewTrainingConfigPatch,
  runTrainingPreflight,
  saveTrainingConfigAs,
  saveTrainingConfigPatch,
  trainingConfigKeys,
} from "./api";
import {
  fieldsForConfig,
  filterTrainingFields,
  fetchFieldCapabilities,
  fieldAvailability,
} from "./fieldCatalog";
import {
  draftFromMerged,
  importedTrainingPath,
  rawConfigOwnKeys,
  trainingPatchValues,
  sameTrainingValue,
  restoreKnownFormDefaults,
  type TrainingDraft,
} from "./trainingForm";

const UNSAVED_MESSAGE =
  "当前训练配置有未保存修改，离开会丢失这些修改。是否继续？";
type ContextSwitchTarget =
  | {
      kind: "config";
      value: string;
    }
  | {
      kind: "preset";
      value: string;
    };
type PendingContextSwitch = ContextSwitchTarget & {
  sourceFile: string | undefined;
  sourcePreset: string;
};
type ContextRefresh = {
  key: string;
  rawUpdateCount: number;
  mergedUpdateCount: number;
};
export const GROUPS = [
  ["input", "输入准备"],
  ["method", "方法配置"],
  ["training", "训练计划"],
  ["resources", "设备与性能"],
] as const;

export function useTrainingWorkspace() {
  const queryClient = useQueryClient();
  const context = useTrainingContext({ deferFallback: true });
  const deviceState = useTrainingDevices();
  const capabilities = useQuery({
    queryKey: ["training-config", "capabilities"],
    queryFn: ({ signal }) => fetchFieldCapabilities(signal),
    staleTime: Infinity,
  });
  const selectedFile = context.selectedFile;
  const rawQuery = useQuery({
    queryKey: trainingConfigKeys.raw(selectedFile?.path || ""),
    queryFn: ({ signal }) => fetchRawTrainingConfig(selectedFile!.path, signal),
    enabled: Boolean(selectedFile),
  });
  const [baseline, setBaseline] = useState<TrainingDraft>({});
  const [draft, setDraft] = useState<TrainingDraft>({});
  const pipelineRequested = draft.pipeline_parallel === true || draft.pipeline_parallel === "true";
  // PP 入口已按族接线（t1 后端 admission 兜底）：仅未接线族仍拦截启动。
  const pipelineUnavailable = pipelineRequested
    && !ppWiredForFamily(String(draft.model_family || context.mergedConfig?.model_family || "anima"));
  const [hydratedKey, setHydratedKey] = useState("");
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [activeStage, setActiveStage] = useState("input");
  const [launchMode, setLaunchMode] = useState<"start" | "queue" | null>(null);
  const [rawMode, setRawMode] = useState<"edit" | "new" | null>(null);
  const [fieldSearch, setFieldSearch] = useState("");
  const [fieldView, setFieldView] = useState("applicable");
  const [promptsOpen, setPromptsOpen] = useState(false);
  const [pendingContextSwitch, setPendingContextSwitch] =
    useState<PendingContextSwitch | null>(null);
  const pendingContextSwitchRef = useRef<PendingContextSwitch | null>(null);
  const fallbackProtectionRef = useRef<{ config?: string; preset?: string }>({});
  const validatingContextSwitchRef = useRef(false);
  const [validatingContextSwitch, setValidatingContextSwitch] = useState(false);
  const [contextRefresh, setContextRefresh] = useState<ContextRefresh | null>(null);
  const mergedConfig = context.mergedConfig || {};
  const fields = useMemo(
    () => fieldsForConfig(mergedConfig),
    [context.mergedConfig],
  );
  const hydrationKey = `${selectedFile?.path || ""}\0${context.selectedPreset}`;
  const rawState = queryClient.getQueryState(trainingConfigKeys.raw(selectedFile?.path || ""));
  const mergedState = queryClient.getQueryState(
    trainingContextKeys.merged(selectedFile?.path || "", context.selectedPreset),
  );
  const contextRefreshPending = Boolean(
    contextRefresh?.key === hydrationKey &&
      (!rawState || rawState.status !== "success" || rawState.fetchStatus !== "idle" ||
        rawState.dataUpdateCount <= contextRefresh.rawUpdateCount ||
        !mergedState || mergedState.status !== "success" || mergedState.fetchStatus !== "idle" ||
        mergedState.dataUpdateCount <= contextRefresh.mergedUpdateCount),
  );
  const patch = useMemo(() => {
    try {
      return {
        values: trainingPatchValues(draft, baseline, fields, mergedConfig),
        error: "",
      };
    } catch (error) {
      return {
        values: {},
        error: error instanceof Error ? error.message : "结构化字段格式错误",
      };
    }
  }, [draft, baseline, fields, mergedConfig]);
  const changes = patch.values;
  const dirty = !sameTrainingValue(draft.resume, baseline.resume, "text") || fields.some(
    (field) =>
      !sameTrainingValue(draft[field.key], baseline[field.key], field.kind),
  );
  const sourceUnavailable =
    Boolean(context.selectedFileId && !context.selectedFileAvailable) ||
    (!context.listsPending &&
      !context.presets.includes(context.selectedPreset));
  const ownKeys = useMemo(
    () => rawConfigOwnKeys(rawQuery.data?.content || ""),
    [rawQuery.data?.content],
  );
  useUnsavedChangesGuard(
    dirty || Boolean(rawMode) || promptsOpen || saveAsOpen,
    UNSAVED_MESSAGE,
  );

  useEffect(() => {
    if (
      !selectedFile ||
      !context.selectedFileAvailable ||
      !rawQuery.data ||
      context.isPending ||
      context.mergedConfigIsFetching ||
      rawQuery.isFetching ||
      contextRefreshPending ||
      context.listsRefetching ||
      hydratedKey === hydrationKey
    )
      return;
    const next = draftFromMerged(mergedConfig, fields);
    setBaseline(next);
    setDraft(next);
    setHydratedKey(hydrationKey);
    setNotice("");
  }, [
    context.isPending,
    context.mergedConfigIsFetching,
    contextRefreshPending,
    context.listsRefetching,
    context.selectedFileAvailable,
    hydratedKey,
    hydrationKey,
    mergedConfig,
    rawQuery.data,
    rawQuery.isFetching,
    selectedFile,
  ]);

  useEffect(() => {
    if (
      contextRefresh?.key === hydrationKey &&
      !contextRefreshPending &&
      hydratedKey === hydrationKey
    ) {
      setContextRefresh(null);
    }
  }, [contextRefresh, contextRefreshPending, hydratedKey, hydrationKey]);

  useEffect(() => {
    if (context.listsPending || context.listsRefetching || dirty || pendingContextSwitch) return;
    if (!context.selectedFileAvailable && !context.files.length) return;
    if (
      !context.selectedFileAvailable &&
      fallbackProtectionRef.current.config !== context.selectedFileId
    ) {
      const fallback = context.files.find(
        (file) => file.path === "configs/imported/lora.toml" && !file.locked,
      ) || context.files.find((file) => !file.locked) ||
        context.files.find((file) => file.path === "configs/gui-methods/lora.toml") ||
        context.files[0];
      if (fallback) context.selectConfigFile(fallback.path);
    } else if (context.selectedFileAvailable) {
      fallbackProtectionRef.current.config = undefined;
    }
    if (!context.presets.includes(context.selectedPreset)) {
      if (fallbackProtectionRef.current.preset !== context.selectedPreset) {
        context.selectPreset(
          context.presets.includes("default") ? "default" : context.presets[0] || "default",
        );
      }
    } else {
      fallbackProtectionRef.current.preset = undefined;
    }
  }, [
    context.files,
    context.isPending,
    context.listsRefetching,
    context.listsPending,
    context.presets,
    context.selectConfigFile,
    context.selectPreset,
    context.selectedFile?.path,
    context.selectedFileId,
    context.selectedFileAvailable,
    context.selectedPreset,
    dirty,
    pendingContextSwitch,
  ]);

  const preview = useMutation({
    mutationFn: () => previewTrainingConfigPatch(selectedFile!.path, changes),
  });
  const save = useMutation({
    mutationFn: (submitted: TrainingDraft) =>
      saveTrainingConfigPatch(
        selectedFile!.path,
        trainingPatchValues(submitted, baseline, fields, mergedConfig),
        rawQuery.data?.revision,
      ),
    onSuccess: async (result, submitted) => {
      setBaseline({ ...submitted });
      setNotice(result.message || "训练配置已保存");
      preview.reset();
      preflight.reset();
      queryClient.setQueryData(trainingConfigKeys.raw(selectedFile!.path), {
        file: selectedFile!.path,
        content: result.content,
        revision: result.revision,
        meta: rawQuery.data!.meta,
      });
      await invalidateTrainingQueries(
        queryClient,
        selectedFile!.path,
        context.selectedPreset,
      );
      const normalized = queryClient.getQueryData<Record<string, unknown>>(
        trainingContextKeys.merged(selectedFile!.path, context.selectedPreset),
      );
      if (normalized) {
        const next = draftFromMerged(normalized, fieldsForConfig(normalized));
        setBaseline(next);
        setDraft((current) =>
          JSON.stringify(current) === JSON.stringify(submitted)
            ? next
            : current,
        );
      }
    },
    retry: false,
  });
  const saveConflict = save.error instanceof ApiError && save.error.status === 409;
  async function reloadConflictedConfig() {
    if (!window.confirm("重新加载会放弃当前未保存的配置修改。是否继续？")) return;
    await Promise.all([
      queryClient.refetchQueries({ queryKey: trainingConfigKeys.raw(selectedFile!.path) }),
      queryClient.refetchQueries({ queryKey: trainingContextKeys.merged(selectedFile!.path, context.selectedPreset) }),
    ]);
    setHydratedKey("");
    save.reset();
  }
  const saveAs = useMutation({
    mutationFn: async (name: string) => {
      const target = importedTrainingPath(name);
      if (!target) throw new Error("请输入有效配置名称");
      const patched = await previewTrainingConfigPatch(
        selectedFile!.path,
        changes,
      );
      const saved = await saveTrainingConfigAs(target, patched.content);
      return { ...saved, file: target };
    },
    onSuccess: async (result) => {
      setSaveAsOpen(false);
      setNotice(result.message || "训练配置已另存");
      await queryClient.invalidateQueries({
        queryKey: trainingContextKeys.files(),
      });
      context.selectConfigFile(result.file);
      setHydratedKey("");
    },
  });
  const preflight = useMutation({
    mutationFn: () =>
      runTrainingPreflight(selectedFile!, context.selectedPreset, deviceState.gpuIds),
  });

  useEffect(() => { preflight.reset(); }, [JSON.stringify(deviceState.gpuIds), deviceState.selection.mode]);

  useEffect(() => {
    save.reset();
    saveAs.reset();
    preview.reset();
    preflight.reset();
  }, [hydrationKey]);

  function confirmDiscard(label: string) {
    return (
      !dirty ||
      window.confirm(
        `当前训练配置有未保存修改。${label}会丢失这些修改，是否继续？`,
      )
    );
  }

  function restorePageDefaults() {
    if (restoreDefaultsBlocked || !selectedFile) return;
    if (!window.confirm("恢复当前可编辑字段的页面默认值？这只会修改未保存草稿，恢复后仍需保存才生效。")) return;
    const editableKeys = new Set(
      fields
        .filter((field) =>
          fieldAvailability(field.key, draft, selectedFile.method || "lora").enabled,
        )
        .map((field) => field.key),
    );
    setDraft((current) => restoreKnownFormDefaults(current, fields, editableKeys));
    preview.reset();
    preflight.reset();
  }

  async function beforeAction(
    action: "preflight" | "start" | "queue" | "prompts",
  ) {
    if (busy || saveConflict || patch.error || (dirty && locked)) return;
    if (action !== "prompts" && deviceState.issue) return;
    if ((action === "start" || action === "queue") && pipelineUnavailable) return;
    try {
      if (dirty) await save.mutateAsync({ ...draft });
      if (action === "preflight") preflight.mutate();
      else if (action === "prompts") setPromptsOpen(true);
      else setLaunchMode(action);
    } catch {
      /* Save errors remain visible; never continue to execution. */
    }
  }

  function commitContextSwitch(target: ContextSwitchTarget) {
    setHydratedKey("");
    const file = target.kind === "config" ? target.value : selectedFile?.path || "";
    const preset = target.kind === "preset" ? target.value : context.selectedPreset;
    const rawKey = trainingConfigKeys.raw(file);
    const mergedKey = trainingContextKeys.merged(file, preset);
    setContextRefresh({
      key: `${file}\0${preset}`,
      rawUpdateCount: queryClient.getQueryState(rawKey)?.dataUpdateCount || 0,
      mergedUpdateCount: queryClient.getQueryState(mergedKey)?.dataUpdateCount || 0,
    });
    if (target.kind === "config") {
      fallbackProtectionRef.current.config = target.value;
      context.selectConfigFile(target.value);
    } else {
      fallbackProtectionRef.current.preset = target.value;
      context.selectPreset(target.value);
    }
    void Promise.all([
      queryClient.invalidateQueries({ queryKey: rawKey, exact: true, refetchType: "all" }),
      queryClient.invalidateQueries({ queryKey: mergedKey, exact: true, refetchType: "all" }),
    ]);
  }

  function requestContextSwitch(target: ContextSwitchTarget) {
    if (contextSelectionBusy || pendingContextSwitchRef.current) return;
    const currentValue =
      target.kind === "config"
        ? selectedFile?.path
        : context.selectedPreset;
    if (target.value === currentValue) return;
    if (!dirty) {
      commitContextSwitch(target);
      return;
    }
    const pending = {
      ...target,
      sourceFile: selectedFile?.path,
      sourcePreset: context.selectedPreset,
    };
    pendingContextSwitchRef.current = pending;
    setPendingContextSwitch(pending);
  }

  function cancelContextSwitch() {
    pendingContextSwitchRef.current = null;
    setPendingContextSwitch(null);
  }

  function confirmContextSwitch() {
    const target = pendingContextSwitchRef.current;
    if (!target || context.listsRefetching || validatingContextSwitchRef.current) return;
    validatingContextSwitchRef.current = true;
    setValidatingContextSwitch(true);
    const listKey = target.kind === "config"
      ? trainingContextKeys.files()
      : trainingContextKeys.presets();
    void (async () => {
      try {
        await queryClient.invalidateQueries({ queryKey: listKey, exact: true });
        if (pendingContextSwitchRef.current !== target) return;
        const state = queryClient.getQueryState(listKey);
        const refreshed = Boolean(
          state && state.status === "success" && !state.isInvalidated && state.fetchStatus === "idle",
        );
        const groups = target.kind === "config"
          ? queryClient.getQueryData<TrainingConfigGroup[]>(listKey)
          : undefined;
        const files = groups?.flatMap((group) =>
          (group.files || [])
            .filter((file) => file.trainable !== false)
            .map((file) => ({ ...file, methods_subdir: file.methods_subdir || group.methods_subdir })),
        ) || [];
        const presets = target.kind === "preset"
          ? queryClient.getQueryData<string[]>(listKey) || []
          : [];
        const targetStillAvailable = target.kind === "config"
          ? files.some((file) => file.path === target.value)
          : presets.includes(target.value);
        const sourceUnchanged =
          useTrainingContextStore.getState().configFile === target.sourceFile &&
          useTrainingContextStore.getState().preset === target.sourcePreset;
        pendingContextSwitchRef.current = null;
        setPendingContextSwitch(null);
        if (refreshed && sourceUnchanged && targetStillAvailable) commitContextSwitch(target);
      } finally {
        validatingContextSwitchRef.current = false;
        setValidatingContextSwitch(false);
      }
    })();
  }

  const guardedContext = {
    ...context,
    selectConfigFile: (file: string) =>
      requestContextSwitch({ kind: "config", value: file }),
    selectPreset: (preset: string) =>
      requestContextSwitch({ kind: "preset", value: preset }),
  };
  const locked = Boolean(selectedFile?.locked || selectedFile?.readonly);
  const actionBusy =
    preview.isPending ||
    save.isPending ||
    saveAs.isPending ||
    saveAsOpen ||
    preflight.isPending ||
    Boolean(pendingContextSwitch) ||
    Boolean(launchMode) ||
    Boolean(rawMode) ||
    promptsOpen;
  const contextRefreshFailed =
    Boolean(rawQuery.error || context.mergedConfigError) &&
    !rawQuery.isFetching &&
    !context.mergedConfigIsFetching;
  const contextSelectionBusy =
    actionBusy || (!contextRefreshFailed &&
      (contextRefreshPending || (hydratedKey !== hydrationKey && !sourceUnavailable)));
  const busy =
    actionBusy ||
    sourceUnavailable ||
    contextRefreshPending ||
    hydratedKey !== hydrationKey;
  const restoreDefaultsBlocked =
    !selectedFile ||
    sourceUnavailable ||
    locked ||
    busy ||
    context.isPending ||
    rawQuery.isPending ||
    capabilities.isPending ||
    hydratedKey !== hydrationKey;
  const visibleFields = filterTrainingFields(
    fields,
    draft,
    baseline,
    fieldSearch,
    fieldView,
    selectedFile?.method || "lora",
  );
  const commandBlocked =
    !selectedFile ||
    sourceUnavailable ||
    busy ||
    context.isPending ||
    rawQuery.isPending ||
    capabilities.isPending ||
    Boolean(capabilities.error) ||
    Boolean(patch.error) ||
    saveConflict ||
    (dirty && locked);

  return {
    queryClient,
    deviceState,
    pipelineUnavailable,
    context,
    capabilities,
    selectedFile,
    rawQuery,
    baseline,
    draft,
    setDraft,
    hydratedKey,
    setHydratedKey,
    saveAsOpen,
    setSaveAsOpen,
    notice,
    activeStage,
    setActiveStage,
    launchMode,
    setLaunchMode,
    rawMode,
    setRawMode,
    fieldSearch,
    setFieldSearch,
    fieldView,
    setFieldView,
    promptsOpen,
    setPromptsOpen,
    mergedConfig,
    fields,
    patch,
    changes,
    dirty,
    ownKeys,
    preview,
    save,
    saveConflict,
    reloadConflictedConfig,
    saveAs,
    preflight,
    confirmDiscard,
    pendingContextSwitch,
    validatingContextSwitch,
    contextSwitchConfirmDisabled: context.listsRefetching || validatingContextSwitch,
    confirmContextSwitch,
    cancelContextSwitch,
    restorePageDefaults,
    restoreDefaultsBlocked,
    beforeAction,
    guardedContext,
    contextSelectionBusy,
    locked,
    busy,
    visibleFields,
    commandBlocked,
  };
}

export async function invalidateTrainingQueries(
  queryClient: ReturnType<typeof useQueryClient>,
  file: string,
  preset: string,
) {
  await Promise.all([
    queryClient.invalidateQueries({
      queryKey: trainingContextKeys.merged(file, preset),
    }),
    queryClient.invalidateQueries({ queryKey: trainingConfigKeys.raw(file) }),
    queryClient.invalidateQueries({ queryKey: ["training-config", "inherited"] }),
    queryClient.invalidateQueries({ queryKey: datasetKeys.library() }),
    queryClient.invalidateQueries({
      queryKey: ["training-config", "estimate", file, preset],
    }),
  ]);
}
