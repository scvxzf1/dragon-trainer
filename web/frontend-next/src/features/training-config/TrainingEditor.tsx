import { useState } from "react";
import { TrainingStageNav } from "./TrainingStageNav";
import { useTrainingStageNavigation } from "./useTrainingStageNavigation";
import type { TrainingEditorView } from "./useTrainingEditorView";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import { TrainingStageFields } from "./TrainingResourceGroups";
import { GROUPS } from "./useTrainingWorkspace";
import { TrainingDevices } from "./TrainingDevices";
import { TrainingDistributedCard } from "./TrainingDistributedCard";
import { TrainingEditorScrollArea } from "./TrainingEditorScrollArea";
import { useTrainingInheritedDefaults } from "./useTrainingInheritedDefaults";
import { useTrainingBlockInventory } from "./useTrainingBlockInventory";

export function TrainingEditor({
  state,
  view = "groups",
}: {
  state: ReturnType<typeof useTrainingWorkspace>;
  view?: TrainingEditorView;
}) {
  const {
    context,
    selectedFile,
    rawQuery,
    draft,
    baseline,
    setDraft,
    activeStage,
    setActiveStage,
    fieldSearch,
    setFieldSearch,
    fieldView,
    setFieldView,
    fields,
    ownKeys,
    preview,
    preflight,
    busy,
    visibleFields,
  } = state;
  const [expandedGroups, setExpandedGroups] = useState<Record<string, Record<string, boolean>>>({});
  const defaults = useTrainingInheritedDefaults(selectedFile, context.selectedPreset, fields);
  const selectStage = useTrainingStageNavigation(view, activeStage, setActiveStage);
  // Keep the shared model header query alive when its field is outside the current stage.
  useTrainingBlockInventory(String(draft.model_family || "anima"), String(draft.pretrained_model_name_or_path || ""));
  return (
    <>
      <div className="training-editor-column" data-editor-view={view}>
        <div className="training-field-search">
          <input
            aria-label="搜索参数"
            placeholder="搜索参数名称或配置键"
            value={fieldSearch}
            onChange={(e) => setFieldSearch(e.target.value)}
          />
          <select
            aria-label="参数视图"
            value={fieldView}
            onChange={(e) => setFieldView(e.target.value)}
          >
            <option value="applicable">当前适用</option>
            <option value="changed">已修改</option>
            <option value="all">全部与审计</option>
          </select>
          <span>
            {visibleFields.length} / {fields.length}
          </span>
        </div>
        <TrainingStageNav view={view} activeStage={activeStage} onSelect={selectStage}
          filtered={Boolean(fieldSearch || fieldView !== "applicable")} fields={visibleFields} />
        <TrainingEditorScrollArea>
          {GROUPS.filter(([group]) => view === "stream" || group === activeStage).map(([group, title]) => (
            <section
              role={view === "stream" ? "region" : "tabpanel"}
              data-training-stage={group}
              hidden={view === "groups" && group !== activeStage}
              aria-labelledby={`training-tab-${group}`}
              tabIndex={0}
              className="training-edit-card"
              id={`training-stage-${group}`}
              key={group}
            >
              <header>
                <div>
                  <p className="eyebrow">{String(group).toUpperCase()}</p>
                  <h2>{title}</h2>
                </div>
                <span>
                  {
                    fields.filter(
                      (field) =>
                        field.group === group && ownKeys.has(field.key),
                    ).length
                  }{" "}
                  当前文件
                </span>
              </header>
              {group === "resources" && !fieldSearch && <TrainingDevices state={state} details />}
              {group === "resources" && !fieldSearch && <TrainingDistributedCard state={state} />}
              {!visibleFields.some((field) => field.group === group) && (
                <p role="status">当前分类没有符合筛选条件的配置项。</p>
              )}
              <TrainingStageFields
                onEditPrompts={() => state.beforeAction("prompts")}
                expandedGroups={expandedGroups}
                onExpandedGroupsChange={setExpandedGroups}
                stage={group}
                search={fieldSearch}
                view={fieldView}
                fields={visibleFields.filter((field) => field.group === group)}
                draft={draft}
                baseline={baseline}
                defaults={defaults}
                ownKeys={ownKeys}
                method={selectedFile?.method}
                disabled={
                  !selectedFile ||
                  context.isPending ||
                  rawQuery.isPending ||
                  busy
                }
                onChange={(key, value) => {
                  setDraft((current) => ({ ...current, [key]: value }));
                  preview.reset();
                  preflight.reset();
                }}
              />
            </section>
          ))}
        </TrainingEditorScrollArea>
      </div>
    </>
  );
}
