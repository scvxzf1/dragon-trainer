import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowDown,
  ArrowUp,
  FolderPlus,
  Plus,
  Pencil,
  Save,
  Trash2,
  RotateCcw,
} from "lucide-react";
import { useRef, useState } from "react";
import { ApiError } from "../../api/client";
import { CommandDialog } from "../../components/CommandDialog";
import { capabilityLabels, findModelCapability, previewCapabilityLabels, useModelCapabilities } from "../../api/modelCapabilities";
import { useServerDraft } from "../../components/useServerDraft";
import { QueryFeedback } from "../../components/QueryFeedback";
import {
  fetchModelConfigs,
  saveModelConfigs,
  settingsKeys,
  type ModelConfigItem,
  type ModelConfigResponse,
  fetchModelPathStatus,
  type ModelPathStatus,
} from "./api";
import {
  addModel,
  deleteModel,
  modelGroups,
  MODEL_PATHS,
  moveModel,
  reorderModel,
} from "./modelLibrary";
import "./settings.css";
import { FALLBACK_MODEL_FAMILIES } from "../training-config/domain/model-family.js";
import { TrainingModelPathControl } from "../training-config/TrainingModelPathControl";

function formatModelSize(bytes: number | undefined) {
  if (bytes === undefined || !Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function ModelPathStatusBadge({
  value,
  disabled,
}: {
  value: string;
  disabled: boolean;
}) {
  const query = useQuery({
    queryKey: ["settings", "model-path-status", value],
    queryFn: ({ signal }) => fetchModelPathStatus(value.trim(), signal),
    enabled: !disabled && Boolean(value.trim()),
    retry: false,
    staleTime: 5000,
  });
  let label = "未填写";
  let tone = "empty";
  const status: ModelPathStatus | undefined = query.data;
  if (query.isFetching) {
    label = "正在检查";
    tone = "pending";
  } else if (query.error) {
    label = "无法校验";
    tone = "missing";
  } else if (status?.kind === "file") {
    label = `文件已就绪 · ${formatModelSize(status.size_bytes)}`;
    tone = "ready";
  } else if (status?.kind === "directory") {
    label = "目录已就绪";
    tone = "ready";
  } else if (value.trim()) {
    label = "路径未找到";
    tone = "missing";
  }
  return <span className={`model-path-status model-path-status-${tone}`}>{label}</span>;
}

function ModelPathField({
  fieldKey,
  label,
  value,
  disabled,
  onChange,
}: {
  fieldKey: string;
  label: string;
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <div className="model-path-card">
      <div className="model-path-heading">
        <label htmlFor={`training-control-model-${fieldKey}`}>{label}</label>
        <ModelPathStatusBadge value={value} disabled={disabled} />
      </div>
      <div className="model-path-control">
        <TrainingModelPathControl
          fieldKey={`model-${fieldKey}`}
          label={label}
          value={value}
          disabled={disabled}
          required
          onChange={onChange}
        />
      </div>
    </div>
  );
}

export function ModelConfigPage() {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: settingsKeys.models,
    queryFn: ({ signal }) => fetchModelConfigs(signal),
    retry: false,
  });
  const editor = useServerDraft(query.data);
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [taskFilter, setTaskFilter] = useState("");
  const [renameGroupId, setRenameGroupId] = useState<string | null>(null);
  const [renameLabel, setRenameLabel] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);
  const capabilities = useModelCapabilities();
  const familyOptions = capabilities.data?.items?.length
    ? capabilities.data.items
    : FALLBACK_MODEL_FAMILIES;
  const modelCapability = (family: string) => findModelCapability(capabilities.data?.items, family);
  const [notice, setNotice] = useState("");
  const save = useMutation({
    mutationFn: (submitted: ModelConfigResponse) => saveModelConfigs(submitted),
    retry: false,
    onSuccess: async (data, submitted) => {
      qc.setQueryData(settingsKeys.models, data);
      editor.accept(data, submitted);
      setNotice("模型配置已保存");
      await qc.invalidateQueries({ queryKey: settingsKeys.global });
    },
  });
  const recoveryError = save.error instanceof ApiError ? save.error : null;
  const requiresServerReload =
    recoveryError?.status === 0 || recoveryError?.status === 409;
  const draft = editor.draft;
  const item =
    draft?.items.find((entry) => entry.id === selectedId) || draft?.items[0];
  const groups = draft ? modelGroups(draft) : [];
  const groupId = groups.find((group) =>
    group.item_ids.includes(item?.id || ""),
  )?.id;
  function update(patch: Partial<ModelConfigItem>) {
    editor.setDraft(
      (d) =>
        d && {
          ...d,
          items: d.items.map((entry) =>
            entry.id === item?.id ? { ...entry, ...patch } : entry,
          ),
        },
    );
    setNotice("");
  }
  function addGroup() {
    const label = window.prompt("分组名称")?.trim();
    if (!label || !draft) return;
    if (
      groups.some((group) => group.label.toLowerCase() === label.toLowerCase())
    ) {
      setNotice("分组名称已存在");
      return;
    }
    editor.setDraft({
      ...draft,
      groups: [
        ...groups,
        {
          id: `group-${crypto.randomUUID().slice(0, 12)}`,
          label,
          item_ids: [],
        },
      ],
    });
  }
  function deleteGroup(id: string) {
    if (
      !draft ||
      groups.length < 2 ||
      !window.confirm(
        "删除分组后，其中的模型配置会移入其他分组。不会删除模型文件。",
      )
    )
      return;
    const removed = groups.find((group) => group.id === id)!;
    const remaining = groups.filter((group) => group.id !== id);
    remaining[0] = {
      ...remaining[0],
      item_ids: [...remaining[0].item_ids, ...removed.item_ids],
    };
    editor.setDraft({ ...draft, groups: remaining });
  }
  function openRenameGroup(groupId: string) {
    const group = groups.find((entry) => entry.id === groupId);
    if (!group) return;
    setRenameGroupId(groupId);
    setRenameLabel(group.label);
    setNotice("");
  }
  function submitRenameGroup() {
    const label = renameLabel.trim();
    if (!draft || !renameGroupId || !label) return;
    if (
      groups.some(
        (group) =>
          group.id !== renameGroupId &&
          group.label.toLowerCase() === label.toLowerCase(),
      )
    ) {
      setNotice("分组名称已存在");
      return;
    }
    editor.setDraft({
      ...draft,
      groups: groups.map((group) =>
        group.id === renameGroupId ? { ...group, label } : group,
      ),
    });
    setRenameGroupId(null);
    setNotice("");
  }
  return (
    <div className="settings-shell">
      <main className="settings-page">
        <header className="page-heading">
          <div>
            <p className="eyebrow">MODEL LIBRARY</p>
            <h1>模型配置</h1>
          </div>
          <span className="state-label">{editor.dirty ? "未保存" : query.error ? "读取失败" : !query.data ? "读取中" : "已同步"}</span>
        </header>
        <QueryFeedback query={query} label="模型配置" hasData={Boolean(query.data)} />
        {!draft ? null : !item ? (
          <button
            type="button"
            onClick={() => editor.setDraft(addModel(draft))}
          >
            <Plus size={16} />
            新建模型配置
          </button>
        ) : (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (!save.isPending && !query.error && !requiresServerReload)
                save.mutate(draft);
            }}
          >
            <fieldset
              disabled={save.isPending}
              className="unframed-fieldset model-workspace"
            >
              <aside className="object-library model-config-library">
                <header>
                  <h2>模型组合</h2>
                  <button
                    type="button"
                    title="新建模型配置"
                    aria-label="新建模型配置"
                    onClick={() => {
                      const next = addModel(draft, groupId);
                      editor.setDraft(next);
                      setSelectedId(next.items.at(-1)!.id);
                    }}
                  >
                    <Plus size={17} />
                  </button>
                  <button
                    type="button"
                    title="新建模型分组"
                    aria-label="新建模型分组"
                    onClick={addGroup}
                  >
                    <FolderPlus size={17} />
                  </button>
                </header>
                <input
                  aria-label="搜索模型配置"
                  placeholder="搜索模型配置"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <select aria-label="筛选训练能力" value={taskFilter} onChange={(e) => setTaskFilter(e.target.value)}>
                  <option value="">全部训练能力</option>
                  <option value="edit">编辑训练</option>
                  <option value="t2i">普通文生图</option>
                </select>
                {taskFilter && capabilities.error && (
                  <small role="alert">模型能力读取失败，暂不应用能力筛选，已保留全部模型配置。</small>
                )}
                {taskFilter && capabilities.isPending && (
                  <small role="status">模型能力读取中，暂时显示全部模型配置。</small>
                )}
                {groups.map((group) => (
                  <section key={group.id}>
                    <div className="library-group-heading">
                      <button
                        type="button"
                        title="重命名分组"
                        aria-label={group.label}
                        onClick={() => openRenameGroup(group.id)}
                      >
                        {group.label}
                        <Pencil size={13} aria-hidden="true" />
                      </button>
                      <button
                        type="button"
                        title="删除分组"
                        aria-label={`删除分组 ${group.label}`}
                        disabled={groups.length < 2}
                        onClick={() => deleteGroup(group.id)}
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                    {group.item_ids
                      .map((id) => draft.items.find((entry) => entry.id === id))
                      .filter(
                        (entry) => {
                          if (!entry) return false;
                          const capability = capabilities.error ? undefined : modelCapability(entry.model_family);
                          const supportsTask = capability?.supported_tasks?.includes(taskFilter);
                          return (
                            (!taskFilter || !capability?.supported_tasks || supportsTask) &&
                            `${entry.name} ${entry.model_family} ${capabilityLabels(capability).join(" ")}`
                              .toLowerCase()
                              .includes(search.toLowerCase())
                          );
                        },
                      )
                      .map((entry) => (
                        <button
                          className="object-row"
                          type="button"
                          key={entry!.id}
                          data-selected={entry!.id === item.id}
                          onClick={() => setSelectedId(entry!.id)}
                        >
                          <span>{entry!.name}</span>
                          <small>
                            {entry!.model_family}
                            {draft.default_id === entry!.id ? " · 默认" : ""}
                          </small>
                          <small>{capabilityLabels(modelCapability(entry!.model_family)).join(" · ")}</small>
                        </button>
                      ))}
                  </section>
                ))}
              </aside>
              <section className="model-editor">
                <header>
                  <div className="model-editor-title">
                    <h2>{item.name || "未命名配置"}</h2>
                    <div className="model-capability-row">
                      <div className="model-capability-group" aria-label="模型训练能力" aria-live="polite">
                        <span>训练</span>
                        <div className="model-capability-tags">
                          {capabilityLabels(modelCapability(item.model_family)).map((label) => <span key={label}>{label}</span>)}
                        </div>
                        {capabilities.error && <small role="alert">能力读取失败：{capabilities.error.message}<button type="button" onClick={() => void capabilities.refetch()} title="重新读取模型能力" aria-label="重新读取模型能力"><RotateCcw size={14} /></button></small>}
                      </div>
                      <div className="model-capability-group" aria-label="模型采样能力" aria-live="polite">
                        <span>采样</span>
                        <div className="model-capability-tags">
                          {previewCapabilityLabels(modelCapability(item.model_family)).map((label) => <span key={label}>{label}</span>)}
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="toolbar">
                    <button
                      type="button"
                      aria-label="上移模型配置"
                      title="上移"
                      onClick={() =>
                        editor.setDraft(reorderModel(draft, item.id, -1))
                      }
                    >
                      <ArrowUp size={16} />
                    </button>
                    <button
                      type="button"
                      aria-label="下移模型配置"
                      title="下移"
                      onClick={() =>
                        editor.setDraft(reorderModel(draft, item.id, 1))
                      }
                    >
                      <ArrowDown size={16} />
                    </button>
                    <button
                      type="button"
                      className="danger"
                      disabled={item.id === draft.default_id}
                      onClick={() => {
                        if (
                          window.confirm(
                            `删除模型配置“${item.name}”？不会删除模型文件。`,
                          )
                        )
                          editor.setDraft(deleteModel(draft, item.id));
                      }}
                    >
                      <Trash2 size={16} />
                      删除配置
                    </button>
                  </div>
                </header>
                <section className="model-properties" aria-label="基础属性">
                  <h3>基础属性</h3>
                  <div className="settings-grid model-properties-grid">
                  <label>
                    <span>名称</span>
                    <input
                      required
                      maxLength={80}
                      value={item.name}
                      onChange={(e) => update({ name: e.target.value })}
                    />
                  </label>
                  <label>
                    <span>模型族</span>
                    <select
                      aria-label="模型族"
                      value={item.model_family}
                      onChange={(e) => update({ model_family: e.target.value })}
                    >
                      {familyOptions.map((family) => (
                        <option key={family.name} value={family.name}>
                          {family.display_name}
                        </option>
                      ))}
                      {!familyOptions.some((family) => family.name === item.model_family) && (
                        <option value={item.model_family}>{item.model_family}</option>
                      )}
                    </select>
                  </label>
                  <label>
                    <span>所属分组</span>
                    <select
                      value={groupId}
                      onChange={(e) =>
                        editor.setDraft(
                          moveModel(draft, item.id, e.target.value),
                        )
                      }
                    >
                      {groups.map((group) => (
                        <option key={group.id} value={group.id}>
                          {group.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={draft.default_id === item.id}
                      disabled={draft.default_id === item.id}
                      onChange={() =>
                        editor.setDraft({ ...draft, default_id: item.id })
                      }
                    />
                    <span>默认模型组合</span>
                  </label>
                  </div>
                </section>
                <section className="model-path-section" aria-label="模型路径">
                  <h3>模型路径</h3>
                  <div className="model-path-grid">
                    {MODEL_PATHS.map(([key, label]) => (
                      <ModelPathField
                        key={key}
                        fieldKey={key}
                        label={label}
                        value={item[key]}
                        disabled={save.isPending}
                        onChange={(value) => update({ [key]: value })}
                      />
                    ))}
                  </div>
                </section>
                <footer className="settings-actions model-config-actions">
                  <button
                    className="primary-command"
                    type="submit"
                    disabled={
                      !editor.dirty ||
                      save.isPending ||
                      Boolean(query.error) ||
                      requiresServerReload
                    }
                  >
                    <Save size={16} />
                    保存模型配置
                  </button>
                  <button
                    type="button"
                    disabled={!editor.dirty || save.isPending}
                    onClick={editor.reset}
                  >
                    <RotateCcw size={16} />
                    还原修改
                  </button>
                  {notice && !editor.dirty && !query.error && <span role="status">{notice}</span>}
                </footer>
              </section>
            </fieldset>
          </form>
        )}
        {save.error && (
          <p className="form-error" role="alert">
            {save.error.message}
            {requiresServerReload && (
              <button
                type="button"
                disabled={query.isFetching}
                onClick={async () => {
                  if (
                    !window.confirm(
                      "重新载入服务器模型库会丢弃当前未保存修改，确认继续吗？",
                    )
                  )
                    return;
                  const result = await query.refetch();
                  if (result.data && !result.error) {
                    editor.accept(result.data, editor.draft!);
                    save.reset();
                    setNotice("已重新载入服务器模型库");
                  }
                }}
              >
                {recoveryError?.status === 0
                  ? "核对服务器版本"
                  : "重新载入服务器版本"}
              </button>
            )}
          </p>
        )}
        {renameGroupId && (
          <CommandDialog
            title="重命名分组"
            onClose={() => setRenameGroupId(null)}
            initialFocusRef={renameInputRef}
          >
            <form
              className="command-form"
              onSubmit={(event) => {
                event.preventDefault();
                submitRenameGroup();
              }}
            >
              <label>
                分组名称
                <input
                  ref={renameInputRef}
                  value={renameLabel}
                  maxLength={80}
                  onChange={(event) => setRenameLabel(event.target.value)}
                />
              </label>
              {renameLabel.trim() &&
                groups.some(
                  (group) =>
                    group.id !== renameGroupId &&
                    group.label.toLowerCase() === renameLabel.trim().toLowerCase(),
                ) && (
                  <p className="form-error" role="alert">
                    分组名称已存在
                  </p>
                )}
              <div className="toolbar">
                <button type="button" onClick={() => setRenameGroupId(null)}>
                  取消
                </button>
                <button
                  type="submit"
                  className="primary"
                  disabled={
                    !renameLabel.trim() ||
                    groups.some(
                      (group) =>
                        group.id !== renameGroupId &&
                        group.label.toLowerCase() === renameLabel.trim().toLowerCase(),
                    )
                  }
                >
                  确定
                </button>
              </div>
            </form>
          </CommandDialog>
        )}
      </main>
    </div>
  );
}
