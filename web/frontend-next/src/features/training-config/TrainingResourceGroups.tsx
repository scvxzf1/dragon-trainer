import { useState, type ComponentProps } from "react";
import { ChevronDown, ChevronRight, CornerDownRight, SlidersHorizontal } from "lucide-react";
import { fieldAvailability } from "./fieldCatalog";
import { partitionClusterFields } from "./fieldHierarchy";
import { TrainingFieldEditor } from "./TrainingFieldEditor";
import { groupStageFields, stageSummary, stageGroupDefaultOpen } from "./stageGroups";
import "./TrainingResourceGroups.css";

type Expansion = Record<string, Record<string, boolean>>;
type Props = ComponentProps<typeof TrainingFieldEditor> & {
  search: string;
  view: string;
  expandedGroups?: Expansion;
  onExpandedGroupsChange?: (value: Expansion) => void;
};

export function TrainingResourceGroups(props: Props) {
  return <TrainingStageFields {...props} stage="resources" />;
}

export function TrainingStageFields({
  stage,
  search,
  view,
  expandedGroups,
  onExpandedGroupsChange,
  ...editor
}: Props & { stage: string }) {
  const [localExpanded, setLocalExpanded] = useState<Expansion>({});
  const [drawerOpen, setDrawerOpen] = useState<Record<string, boolean>>({});
  const expanded = expandedGroups ?? localExpanded;
  const setExpanded = onExpandedGroupsChange ?? setLocalExpanded;
  const hasSearch = Boolean(search.trim());
  const filtered = hasSearch || view === "changed";
  const flattenAll = hasSearch || view !== "applicable";
  const filter = `${stage}\0${search}\0${view}`;
  const values = expanded[filter] ?? {};
  const baseline = editor.baseline ?? editor.draft;
  const method = editor.method ?? "lora";

  return (
    <div className="training-resource-groups">
      {groupStageFields(stage, editor.fields).map(({ id, title, fields }) => {
        const defaultOpen = stageGroupDefaultOpen(stage, id);
        const open = values[id] ?? (filtered || defaultOpen);
        const ownFieldCount = fields.filter((field) => editor.ownKeys.has(field.key)).length;
        const summary = stageSummary(stage, id, editor.draft);
        const partition = partitionClusterFields(fields, {
          draft: editor.draft,
          baseline,
          defaults: editor.defaults,
          ownKeys: editor.ownKeys,
          getAvailability: (key) => fieldAvailability(key, editor.draft, method),
          flattenAll,
        });
        const promoteDrawer = partition.items.length === 0 && partition.drawerFields.length > 0;
        const mainFields = promoteDrawer ? partition.drawerFields : partition.items.map((item) => item.field);
        const childMap = new Map(partition.items.filter((item) => item.children.length > 0).map((item) => [item.field.key, item.children]));
        const drawerFields = promoteDrawer ? [] : partition.drawerFields;
        const drawerKey = `${stage}:${id}`;
        const isDrawerOpen = Boolean(drawerOpen[drawerKey]);
        const compactGroup = !hasSearch && defaultOpen && fields.length <= 2 && drawerFields.length === 0;
        const Icon = open ? ChevronDown : ChevronRight;
        const DrawerIcon = isDrawerOpen ? ChevronDown : ChevronRight;

        return (
          <section
            className="training-resource-group"
            data-stage={stage}
            data-group={id}
            data-compact-group={compactGroup || undefined}
            key={id}
          >
            <h3>
              <button
                type="button"
                id={`resource-heading-${stage}-${id}`}
                aria-expanded={open}
                aria-controls={`resource-fields-${stage}-${id}`}
                onClick={() => setExpanded({ ...expanded, [filter]: { ...values, [id]: !open } })}
              >
                <Icon size={16} aria-hidden="true" />
                <span className="resource-title">
                  <SlidersHorizontal size={14} aria-hidden="true" />
                  <span>{title}</span>
                </span>
                <span className="resource-meta">
                  {summary && <span className="resource-summary-badge">{summary}</span>}
                  {(!compactGroup || !open || !summary) && (
                    <span className="resource-field-count resource-chip">{fields.length} 项</span>
                  )}
                  {ownFieldCount > 0 && (
                    <span className="resource-own-count resource-chip">当前文件 {ownFieldCount} 项</span>
                  )}
                </span>
              </button>
            </h3>
            <div
              id={`resource-fields-${stage}-${id}`}
              hidden={!open}
              role="region"
              aria-labelledby={`resource-heading-${stage}-${id}`}
            >
              <TrainingFieldEditor
                {...editor}
                fields={mainFields}
                renderAfterField={(parentField) => {
                  const children = childMap.get(parentField.key);
                  if (!children?.length) return null;
                  return (
                    <div className="training-dependent-panel" data-parent-key={parentField.key}>
                      <div className="training-dependent-heading">
                        <CornerDownRight size={13} aria-hidden="true" />
                        <span>{parentField.label} · 关联配置</span>
                      </div>
                      <TrainingFieldEditor {...editor} fields={children} subgrid />
                    </div>
                  );
                }}
              />
              {drawerFields.length > 0 && (
                <div className="training-advanced-drawer" data-open={isDrawerOpen || undefined}>
                  <button
                    type="button"
                    className="training-advanced-drawer-toggle"
                    aria-expanded={isDrawerOpen}
                    onClick={() => setDrawerOpen((prev) => ({ ...prev, [drawerKey]: !isDrawerOpen }))}
                  >
                    <DrawerIcon size={14} aria-hidden="true" />
                    <span>按需与高级选项（{drawerFields.length} 项未激活或保持默认）</span>
                    {partition.drawerOwnCount > 0 && (
                      <span className="resource-own-count resource-chip">
                        当前文件 {partition.drawerOwnCount} 项
                      </span>
                    )}
                  </button>
                  {isDrawerOpen && (
                    <div className="training-advanced-drawer-body">
                      <TrainingFieldEditor {...editor} fields={drawerFields} subgrid />
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        );
      })}
    </div>
  );
}
