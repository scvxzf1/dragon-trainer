import { sameTrainingValue, type TrainingDraft, type TrainingFieldSpec } from "./trainingForm";
import { FilePenLine } from "lucide-react";
import { Fragment, lazy, Suspense, useState, type ReactNode } from "react";
import { SettingSwitch } from "../../components/SettingSwitch";
import { TrainingFieldHeader } from "./TrainingFieldHeader";
import { useTrainingFieldHelpDisplay } from "./TrainingFieldHelpPreference";
import { isTrainingModelPath, TrainingModelPathControl } from "./TrainingModelPathControl";
import { TrainingDatasetField } from "./TrainingDatasetField";
import { TrainingBlockSwapControl } from "./TrainingBlockSwapControl";
import { availableFieldOptions, fieldAvailability } from "./fieldCatalog";
import { FIELD_HELP_SUMMARY_ZH } from "./domain/field-help-summary.js";
import { FORM_UI_DEFAULTS } from "./domain/defaults.js";
import { getTrainingFieldLayout, getTrainingGridColumns } from "./trainingFieldLayout";

const TrainingFieldHelpDialog = lazy(() => import("./TrainingFieldHelpDialog").then((module) => ({ default: module.TrainingFieldHelpDialog })));

type Props = {
  fields: TrainingFieldSpec[];
  draft: TrainingDraft;
  baseline?: TrainingDraft;
  defaults?: TrainingDraft;
  ownKeys: Set<string>;
  disabled: boolean;
  onChange: (key: string, value: string | number | boolean) => void;
  method?: string;
  onEditPrompts?: () => void;
  renderAfterField?: (field: TrainingFieldSpec, onHelp: (target: TrainingFieldSpec) => void) => ReactNode;
  subgrid?: boolean;
};

export function TrainingFieldEditor({
  fields,
  draft,
  baseline = draft,
  defaults,
  ownKeys,
  disabled,
  onChange,
  method = "lora",
  onEditPrompts,
  renderAfterField,
  subgrid = false,
}: Props) {
  const [helpField, setHelpField] = useState<TrainingFieldSpec | null>(null);
  const { display: helpDisplay } = useTrainingFieldHelpDisplay();
  const modelFamily = String(draft.model_family || "anima");
  return (
    <>
    <div
      className={subgrid ? "training-edit-fields training-dependent-fields" : "training-edit-fields"}
      data-field-count={fields.length}
      data-auto-columns={getTrainingGridColumns(fields)}
    >
      {fields.map((field) => {
        const dirty = !sameTrainingValue(draft[field.key], baseline[field.key], field.kind);
        const layout = getTrainingFieldLayout(field, fields.length);
        const hasInherited = defaults !== undefined && Object.hasOwn(defaults, field.key);
        const overridden = hasInherited && !sameTrainingValue(draft[field.key], defaults[field.key], field.kind);
        const afterContent = renderAfterField?.(field, setHelpField);
        if (field.key === "dataset_config") {
          return (
            <Fragment key={field.key}>
              <TrainingDatasetField value={String(draft.dataset_config ?? "")} dirty={dirty} overridden={overridden} onReset={hasInherited ? () => onChange(field.key, defaults![field.key]) : undefined} disabled={disabled} own={ownKeys.has(field.key)} onHelp={() => setHelpField(field)} onChange={(value) => onChange(field.key, value)} onUndo={() => onChange(field.key, baseline.dataset_config ?? "")} />
              {afterContent}
            </Fragment>
          );
        }
        const availability = fieldAvailability(field.key, draft, method);
        const options = availableFieldOptions(
          field,
          String(draft.model_family || "anima"),
          draft,
        );
        const controlDisabled = disabled || !availability.enabled;
        return (
          <Fragment key={field.key}>
          <div
            id={`training-field-${field.key}`}
            tabIndex={-1}
            className="training-field-card"
            data-field-kind={field.kind}
            data-own={ownKeys.has(field.key) || undefined}
            data-dirty={dirty || undefined}
            data-overridden={overridden || undefined}
            data-model-path={isTrainingModelPath(field.key) || undefined}
            data-brick-width={field.kind === "text" || field.kind === "json" ? "wide" : "narrow"}
            data-auto-span={layout.span}
            data-auto-width={layout.width}
            data-available={availability.enabled}
            data-visible={availability.visible}
            data-availability-code={availability.code || undefined}
          >
            <TrainingFieldHeader label={field.label} fieldKey={field.key} own={ownKeys.has(field.key)} dirty={dirty} overridden={overridden} disabled={disabled}
              onHelp={() => setHelpField(field)} onUndo={() => onChange(field.key, baseline[field.key] ?? "")}
              onReset={hasInherited && availability.enabled ? () => onChange(field.key, defaults![field.key]) : undefined}>
              {field.kind === "boolean" && <SettingSwitch id={`training-control-${field.key}`} aria-label={field.label}
                checked={Boolean(draft[field.key])} disabled={controlDisabled} onCheckedChange={(checked) => onChange(field.key, checked)} />}
              {field.key === "sample_prompts" && onEditPrompts && <button type="button" title="编辑样张提示词" aria-label="编辑样张提示词" disabled={controlDisabled} onClick={onEditPrompts}><FilePenLine size={16} /></button>}
            </TrainingFieldHeader>
            {field.key === "blocks_to_swap" ? (
              <TrainingBlockSwapControl draft={draft} field={field} disabled={controlDisabled} onChange={onChange} />
            ) : isTrainingModelPath(field.key) ? (
              <TrainingModelPathControl fieldKey={field.key} label={field.label} value={String(draft[field.key] ?? "")} disabled={controlDisabled} onChange={(value) => onChange(field.key, value)} />
            ) : field.kind === "json" ? (
              <textarea
                id={`training-control-${field.key}`}
                aria-label={field.label}
                rows={5}
                value={String(draft[field.key] ?? "")}
                disabled={controlDisabled}
                onChange={(event) => onChange(field.key, event.target.value)}
              />
            ) : field.kind === "boolean" ? null : field.kind === "select" ? (
              <select
                id={`training-control-${field.key}`}
                aria-label={field.label}
                value={String(draft[field.key] ?? "")}
                disabled={controlDisabled}
                onChange={(event) => onChange(field.key, event.target.value)}
              >
                {!options?.includes(String(draft[field.key] ?? "")) && (
                  <option value={String(draft[field.key] ?? "")}>
                    {String(draft[field.key] ?? "") || "未设置"}
                  </option>
                )}
                {options?.map((option) => (
                  <option key={option} value={option}>
                    {field.optionLabels?.[option] || option}
                  </option>
                ))}
              </select>
            ) : (
              <input
                id={`training-control-${field.key}`}
                type={field.kind}
                aria-label={field.label}
                value={String(draft[field.key] ?? "")}
                min={field.min}
                max={field.max}
                step={field.step}
                disabled={controlDisabled}
                onChange={(event) =>
                  onChange(
                    field.key,
                    field.kind === "number" && event.target.value !== ""
                      ? Number(event.target.value)
                      : event.target.value,
                  )
                }
              />
            )}
            {helpDisplay !== "hidden" && (field.help || FIELD_HELP_SUMMARY_ZH[field.key]) && (
              <small className="training-field-help" data-help-display={helpDisplay}>
                {field.help || FIELD_HELP_SUMMARY_ZH[field.key]}
              </small>
            )}
            {!availability.enabled && (
              <small className="field-availability-reason">
                {availability.reason}
              </small>
            )}
          </div>
          {afterContent}
          </Fragment>
        );
      })}
    </div>
    {helpField && <Suspense fallback={<p role="status">正在加载字段说明</p>}>
      <TrainingFieldHelpDialog label={helpField.label} fieldKey={helpField.key} value={draft[helpField.key]} defaultValue={helpField.defaultValue ?? (FORM_UI_DEFAULTS as Record<string, unknown>)[helpField.key]} kind={helpField.kind} options={availableFieldOptions(helpField, modelFamily, draft)} optionLabels={helpField.optionLabels} modelFamily={modelFamily} draft={draft} unavailableReason={fieldAvailability(helpField.key, draft, method).enabled ? undefined : fieldAvailability(helpField.key, draft, method).reason} onClose={() => setHelpField(null)} />
    </Suspense>}
    </>
  );
}
