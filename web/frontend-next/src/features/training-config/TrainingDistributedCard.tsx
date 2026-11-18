import { ChevronDown, ChevronRight, Cpu, RefreshCw } from "lucide-react";
import { useId, useState } from "react";
import { isKrea2ModelFamily } from "./domain/model-family.js";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import { useDistributedCard } from "./useDistributedCard";
import { DISTRIBUTED_MODES, checkBadge, diagnosisSummary } from "./distributedDiagnosis";
import "./TrainingDistributedCard.css";

type Props = { state: ReturnType<typeof useTrainingWorkspace> };

export function TrainingDistributedCard({ state }: Props) {
  const [manualOpen, setManualOpen] = useState<boolean | null>(null);
  const defaultExpanded = Boolean(
    state.deviceState.issue ||
    state.draft.pipeline_parallel === true ||
    state.draft.pipeline_parallel === "true",
  );
  const expanded = manualOpen ?? defaultExpanded;
  const card = useDistributedCard({ state, open: expanded });
  const panelId = useId();
  const isKrea2 = isKrea2ModelFamily(card.family);
  const summaryTone = card.issue
    ? "danger"
    : card.diagnosis
      ? diagnosisSummary(card.diagnosis.checks).errors
        ? "danger"
        : diagnosisSummary(card.diagnosis.checks).warnings
          ? "warning"
          : "success"
      : "neutral";
  const checks = card.diagnosis?.checks || [];
  const risks = card.diagnosis?.hardware.risks || [];
  const ToggleIcon = expanded ? ChevronDown : ChevronRight;

  return (
    <section
      className="training-distributed-card"
      aria-label="设备与并行设置"
      aria-busy={card.diagnosisPending}
      data-expanded={expanded || undefined}
    >
      <div className="training-distributed-row">
        <strong className="training-distributed-title">
          <Cpu size={15} aria-hidden="true" />
          <span>训练设备与并行</span>
        </strong>
        <div className="training-distributed-modes" role="group" aria-label="分布式训练模式">
          {DISTRIBUTED_MODES.map((item) => {
            const disabled =
              card.busy ||
              (item.value === "ddp" && card.devices.length < 2) ||
              (item.value === "pp" && !card.ppAvailable);
            return (
              <button
                key={item.value}
                type="button"
                disabled={disabled}
                aria-pressed={card.mode === item.value}
                title={item.value === "pp" && card.ppBlockedReason ? card.ppBlockedReason : item.hint}
                onClick={() => {
                  card.changeMode(item.value);
                  if (item.value !== "single") setManualOpen(true);
                }}
              >
                {item.label}
              </button>
            );
          })}
        </div>
        <span className="training-distributed-summary" data-tone={summaryTone}>
          {card.summary}
        </span>
        <button
          type="button"
          className="training-distributed-toggle"
          aria-expanded={expanded}
          aria-controls={panelId}
          aria-label={expanded ? "收起设备与分布式详情" : "展开设备与分布式详情"}
          onClick={() => setManualOpen(!expanded)}
        >
          <ToggleIcon size={14} aria-hidden="true" />
          <span>{expanded ? "收起" : "选卡与详情"}</span>
        </button>
      </div>

      <div id={panelId} className="training-distributed-panel" hidden={!expanded}>
        {card.issue && (
          <p className="form-error" role="alert">
            {card.issue}
            {card.issue.startsWith("已选设备不可用") && (
              <button type="button" disabled={card.busy} onClick={state.deviceState.resetSelection}>
                清除失效选择
              </button>
            )}
          </p>
        )}

        <div
          className="training-distributed-gpus"
          role={card.mode === "single" ? "radiogroup" : "group"}
          aria-label="选择训练 GPU"
        >
          {card.devices.map((gpu) => {
            const snapshot = card.diagnosis?.hardware.gpus.find((item) => String(item.index) === gpu.id);
            return (
              <label key={gpu.id}>
                <input
                  type={card.mode === "single" ? "radio" : "checkbox"}
                  name="distributed-gpu"
                  aria-label={`GPU ${gpu.id} · ${gpu.name}`}
                  checked={card.selection.devices.some((item) => item.id === gpu.id && item.name === gpu.name)}
                  disabled={card.busy}
                  onChange={() => card.selectDevice(gpu)}
                />
                <span>GPU {gpu.id} · {gpu.name}</span>
                {gpu.memoryGb && <small>总显存 {gpu.memoryGb} GB</small>}
                {snapshot?.capability && (
                  <small className="training-distributed-attr">
                    sm_{snapshot.capability[0]}{snapshot.capability[1]}
                  </small>
                )}
              </label>
            );
          })}
          <button
            type="button"
            className="icon-button"
            aria-label="刷新 GPU 列表"
            title="刷新 GPU 列表"
            disabled={card.busy || state.deviceState.query.isFetching}
            onClick={() => void card.refreshDevices()}
          >
            <RefreshCw size={14} />
          </button>
        </div>

        {card.mode === "pp" && (
          <div className="training-distributed-pp">
            <label>
              <span>流水线阶段</span>
              <input
                type="number"
                min={2}
                value={card.settings?.pipeline_parallel_stages ?? 2}
                disabled={card.busy || card.settingsSaving}
                onChange={(event) => card.updatePpNumber("pipeline_parallel_stages", event.target.value)}
              />
            </label>
            <label>
              <span>微批数量</span>
              <input
                type="number"
                min={2}
                value={card.settings?.pipeline_parallel_microbatches ?? 4}
                disabled={card.busy || card.settingsSaving}
                onChange={(event) => card.updatePpNumber("pipeline_parallel_microbatches", event.target.value)}
              />
            </label>
            <label title={!isKrea2 ? "PP 块交换仅适用于 Krea-2 模型族" : undefined}>
              <span>PP 块交换</span>
              <input
                type="number"
                min={0}
                value={card.settings?.pipeline_parallel_blocks_to_swap ?? 0}
                disabled={card.busy || card.settingsSaving || !isKrea2}
                onChange={(event) => card.updatePpNumber("pipeline_parallel_blocks_to_swap", event.target.value)}
              />
            </label>
            <small>
              启用 PP 时普通 Block Swap 须保持 0；PP 块交换仅对 Krea-2 生效（16GB 双卡建议 4），修改会同步写入当前草稿与本机缺省值。
            </small>
          </div>
        )}

        {(card.diagnosisPending || card.diagnosisError || risks.length > 0 || checks.length > 0) && (
          <div className="training-distributed-checks" aria-live="polite">
            {card.diagnosisPending && <p role="status">正在诊断设备拓扑…</p>}
            {card.diagnosisError && <p className="form-error" role="alert">{card.diagnosisError}</p>}
            {[...risks, ...checks].map((check, index) => {
              const badge = checkBadge(check);
              return (
                <div key={`${check.key}-${index}`} data-tone={badge.tone}>
                  <p><span aria-hidden="true">{badge.icon}</span> {check.message}</p>
                  {check.detail && (
                    <details className="training-distributed-detail">
                      <summary>原始诊断详情</summary>
                      <pre>{check.detail}</pre>
                    </details>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {(card.wiredFamilies?.length || card.settingsError) ? (
          <footer className="training-distributed-footer">
            {card.wiredFamilies?.length ? (
              <small>
                PP 已支持模型族：{card.wiredFamilies.filter((item) => item.wired).map((item) => item.display_name).join("、") || "无"}
              </small>
            ) : null}
            {card.settingsError && <small role="alert">分布式设置保存失败：{card.settingsError}</small>}
          </footer>
        ) : null}
      </div>
    </section>
  );
}
