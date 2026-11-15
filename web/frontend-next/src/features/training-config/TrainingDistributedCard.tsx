import { ChevronDown, RefreshCw } from "lucide-react";
import { useId, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import { useDistributedCard } from "./useDistributedCard";
import { DISTRIBUTED_MODES, checkBadge, diagnosisSummary } from "./distributedDiagnosis";
import "./TrainingDistributedCard.css";

type Props = { state: ReturnType<typeof useTrainingWorkspace> };

export function TrainingDistributedCard({ state }: Props) {
  const [open, setOpen] = useState(false);
  const card = useDistributedCard({ state, open });
  const popoverId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const summaryTone = card.diagnosis
    ? diagnosisSummary(card.diagnosis.checks).errors
      ? "danger"
      : diagnosisSummary(card.diagnosis.checks).warnings
        ? "warning"
        : "success"
    : "neutral";
  return (
    <section className="training-distributed-card" aria-label="分布式训练" aria-busy={card.diagnosisPending}>
      <div className="training-distributed-row">
        <strong>分布式训练</strong>
        <div className="training-distributed-modes" role="group" aria-label="分布式训练模式">
          {DISTRIBUTED_MODES.map((item) => {
            const disabled = card.busy
              || (item.value === "ddp" && card.devices.length < 2)
              || (item.value === "pp" && !card.ppAvailable);
            return (
              <button key={item.value} type="button" disabled={disabled}
                aria-pressed={card.mode === item.value}
                title={item.value === "pp" && card.ppBlockedReason ? card.ppBlockedReason : item.hint}
                onClick={() => card.changeMode(item.value)}>{item.label}</button>
            );
          })}
        </div>
        <span className="training-distributed-state" data-tone={summaryTone}>
          {card.mode === "pp" ? "流水线并行" : card.mode === "ddp" ? "数据并行" : "单卡"}
        </span>
        <button ref={triggerRef} type="button" className="icon-button" aria-expanded={open}
          aria-controls={open ? popoverId : undefined} aria-label="展开分布式训练详情"
          onClick={() => setOpen((current) => !current)}>
          <ChevronDown size={14} aria-hidden="true" />
        </button>
      </div>
      {open && createPortal(
        <TrainingDistributedPopover card={card} popoverId={popoverId}
          anchorRef={triggerRef} onClose={() => setOpen(false)} />,
        document.body,
      )}
    </section>
  );
}

type PopoverProps = {
  card: ReturnType<typeof useDistributedCard>;
  popoverId: string;
  anchorRef: RefObject<HTMLButtonElement | null>;
  onClose: () => void;
};

function TrainingDistributedPopover({ card, popoverId, anchorRef, onClose }: PopoverProps) {
  const ref = useRef<HTMLElement>(null);
  const anchor = anchorRef.current;
  const rect = anchor?.getBoundingClientRect();
  const style = rect
    ? { position: "fixed" as const, top: rect.bottom + 6, right: Math.max(8, window.innerWidth - rect.right), zIndex: 60 }
    : undefined;
  const checks = card.diagnosis?.checks || [];
  const risks = card.diagnosis?.hardware.risks || [];
  return (
    <section ref={ref} id={popoverId} className="training-distributed-popover" role="dialog"
      aria-label="分布式训练详情" style={style}>
      <header>
        <strong>分布式训练详情</strong>
        <button type="button" className="icon-button" aria-label="收起分布式训练详情" onClick={onClose}>×</button>
      </header>
      {card.issue && <p className="form-error" role="alert">{card.issue}</p>}
      <div className="training-distributed-gpus" role="group" aria-label="选择训练 GPU">
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
              {snapshot && <small className="training-distributed-attr">
                sm_{snapshot.capability ? `${snapshot.capability[0]}${snapshot.capability[1]}` : "?"}
              </small>}
            </label>
          );
        })}
        <button type="button" className="icon-button" aria-label="刷新 GPU 列表" disabled={card.busy}
          onClick={() => void card.refreshDevices()}><RefreshCw size={14} /></button>
      </div>
      {card.mode !== "single" && (
        <div className="training-distributed-pp">
          <label>流水线阶段
            <input type="number" min={1} value={card.settings?.pipeline_parallel_stages ?? 2}
              disabled={card.busy || card.settingsSaving}
              onChange={(event) => card.updatePpNumber("pipeline_parallel_stages", event.target.value)} />
          </label>
          <label>微批数量
            <input type="number" min={1} value={card.settings?.pipeline_parallel_microbatches ?? 4}
              disabled={card.busy || card.settingsSaving}
              onChange={(event) => card.updatePpNumber("pipeline_parallel_microbatches", event.target.value)} />
          </label>
          <small>PP 参数保存在本机设置 [distributed] 节，仅作为卡片缺省值；真正拦截由训练入口兜底。</small>
        </div>
      )}
      <div className="training-distributed-checks" aria-live="polite">
        {card.diagnosisPending && <p role="status">诊断中…</p>}
        {card.diagnosisError && <p className="form-error" role="alert">{card.diagnosisError}</p>}
        {[...risks, ...checks].map((check, index) => {
          const badge = checkBadge(check);
          return <div key={`${check.key}-${index}`} data-tone={badge.tone}>
            <p><span aria-hidden="true">{badge.icon}</span> {check.message}</p>
            {check.detail && (
              <details className="training-distributed-detail">
                <summary>原始信息</summary>
                <pre>{check.detail}</pre>
              </details>
            )}
          </div>;
        })}
        {!card.diagnosisPending && !card.diagnosisError && !checks.length && card.mode !== "pp"
          && <p data-tone="info"><span aria-hidden="true">🔵</span> 单卡 / 数据并行模式无额外分布式检查。</p>}
      </div>
      <footer>
        {card.wiredFamilies?.length ? (
          <small>PP 已接线：{card.wiredFamilies.filter((item) => item.wired).map((item) => item.display_name).join("、") || "无"}</small>
        ) : null}
        {card.settingsError && <small role="alert">分布式设置保存失败：{card.settingsError}</small>}
      </footer>
    </section>
  );
}
