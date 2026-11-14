import { CircleCheck, CircleHelp, OctagonX, TriangleAlert } from "lucide-react";
import { Link } from "react-router-dom";
import type { TaskResultRetention } from "./monitorState";
import { outcomeLabel, outcomeTone } from "./monitorState";
import "./MonitorResultPanel.css";

const OUTCOME_ICONS = {
  completed: CircleCheck,
  stopped: CircleHelp,
  failed: OctagonX,
  interrupted: TriangleAlert,
  unconfirmed: CircleHelp,
} as const;

function formatFinishedTime(retention: TaskResultRetention): string | undefined {
  if (retention.finishedAtText?.trim()) return retention.finishedAtText;
  if (typeof retention.finishedAt === "number" && Number.isFinite(retention.finishedAt)
    && Math.abs(retention.finishedAt) < 8.64e12) {
    return new Date(retention.finishedAt * 1000).toLocaleString();
  }
  return undefined;
}

/**
 * Read-only retention banner for a finished task. The outcome label comes from
 * the final broadcast message (or a persisted interrupted meta) — never from
 * `status === "idle"`, which alone does not prove success.
 */
export function MonitorResultPanel({ retention, finalSyncError, keepingLive }: {
  retention: TaskResultRetention;
  finalSyncError?: string;
  keepingLive?: boolean;
}) {
  const Icon = OUTCOME_ICONS[retention.outcome];
  const tone = outcomeTone(retention.outcome);
  const finishedAt = formatFinishedTime(retention);
  return <section className="monitor-result-panel" role="status" data-tone={tone}
    aria-label="任务结果保留">
    <div className="monitor-result-head">
      <Icon size={20} aria-hidden="true" />
      <div className="monitor-result-text">
        <h2>本次结果已保留 · {outcomeLabel(retention.outcome)}</h2>
        <p>
          {retention.message || "任务已结束，本次曲线、样张与日志保留在下方只读展示。"}
          {finishedAt ? <span className="monitor-result-time"> · 结束时间：{finishedAt}</span> : null}
        </p>
        {finalSyncError ? <p className="monitor-result-sync-warning">
          最终同步未完成（{finalSyncError}），以下为最后一次成功读取的数据，不代表任务结果已确认。
        </p> : null}
      </div>
    </div>
    <div className="monitor-result-actions">
      <Link className="monitor-result-link" to={`/history/${encodeURIComponent(retention.taskId)}`}>打开历史详情</Link>
      <Link className="monitor-result-link" to="/history">历史任务列表</Link>
      <Link className="monitor-result-link" to="/training">训练配置</Link>
    </div>
    {keepingLive ? <p className="monitor-result-live-note">检测到新的活动任务后，将自动切换监控目标。</p> : null}
  </section>;
}
