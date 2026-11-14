import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { Activity, ArrowRight, Cpu, Radio, Square, Terminal } from "lucide-react";
import {
  fetchGpus, fetchHistoryResultMeta, fetchTrainingLogs, fetchTrainingMetrics, fetchTrainingStatus,
  liveMonitorKeys, monitorResultKeys, stopTraining,
  type GpuInventory, type TrainingStatus,
} from "./api";
import { TrainingMetricsCharts } from "../../components/TrainingMetricsCharts";
import { LogViewer } from "../../components/LogViewer";
import { QueryFeedback } from "../../components/QueryFeedback";
import { InlineConfirmDialog } from "../../components/InlineConfirmDialog";
import { MonitorSummary, GpuDetails } from "./MonitorSummary";
import { useMonitorRefresh } from "./useMonitorRefresh";
import { MonitorHistoryWorkspace } from "./MonitorHistoryWorkspace";
import { MonitorSamplePreview } from "./MonitorSamplePreview";
import { MonitorResultPanel } from "./MonitorResultPanel";
import { monitorPhase, monitorStateText, retainedResult, outcomeFromHistoryMeta } from "./monitorState";
import "./LiveMonitorPage.css";

const RUNNING_STATES = new Set(["running", "training", "compiling", "caching", "saving"]);

export function LiveMonitorPage() {
  const [params] = useSearchParams();
  const [stopTarget, setStopTarget] = useState<string | null>(null);
  const [visible, setVisible] = useState(() => document.visibilityState === "visible");
  const sourceTask = params.get("from_task");
  const client = useQueryClient();
  useEffect(() => {
    const onVisibilityChange = () => setVisible(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, []);
  const statusQuery = useQuery({
    queryKey: liveMonitorKeys.status,
    queryFn: ({ signal }) => fetchTrainingStatus(signal),
    refetchInterval: (query) => !visible || query.state.error ? false : 2000,
    retry: false,
  });
  const status = statusQuery.data;
  const { terminalEvent, ...connection } = useMonitorRefresh();
  const disconnected = Boolean(statusQuery.error);
  // A confirmed task identity must come from a successful status read.
  const confirmedTaskId = !statusQuery.error && status?.task_id ? status.task_id : undefined;
  const running = !disconnected && RUNNING_STATES.has(status?.status || "");
  // Finished snapshots keep the last task's identity, so the workspace stays
  // mounted read-only instead of collapsing to the empty state.
  const workspaceTaskId = confirmedTaskId;
  const metricsQuery = useQuery({
    queryKey: [...liveMonitorKeys.metrics, workspaceTaskId || "none"],
    queryFn: ({ signal }) => fetchTrainingMetrics(workspaceTaskId, signal),
    enabled: Boolean(workspaceTaskId) && !disconnected, refetchInterval: visible && running ? 5000 : false, retry: false,
  });
  const logsQuery = useQuery({
    queryKey: [...liveMonitorKeys.logs, workspaceTaskId || "none"],
    queryFn: ({ signal }) => fetchTrainingLogs(300, workspaceTaskId, signal),
    enabled: Boolean(workspaceTaskId) && !disconnected, refetchInterval: visible && running ? 2000 : false, retry: false,
  });
  const gpusQuery = useQuery({
    queryKey: liveMonitorKeys.gpus,
    queryFn: ({ signal }) => fetchGpus(signal),
    refetchInterval: visible ? running ? 2000 : 10000 : false, retry: false,
  });
  const stop = useMutation({
    mutationFn: (clickedTaskId: string) => stopTraining(clickedTaskId),
    onSettled: () => client.invalidateQueries({ queryKey: liveMonitorKeys.status }),
    retry: false,
  });
  const phase = monitorPhase({
    statusQueryPending: statusQuery.isPending, statusQueryError: disconnected,
    hasStatus: Boolean(status), status: status?.status, taskId: confirmedTaskId,
  });
  const state = monitorStateText(phase);
  useEffect(() => {
    if (stopTarget && (stopTarget !== confirmedTaskId || !running || disconnected)) setStopTarget(null);
  }, [stopTarget, confirmedTaskId, running, disconnected]);

  return <MonitorHistoryWorkspace>{(recentToggle) => <main className="monitor-page">
    <header className="monitor-header">
      <div className="monitor-header-main">
        <div className="monitor-title-row">
          <h1>当前监控</h1>
          <span className="monitor-telemetry-tag">Telemetry</span>
        </div>
        {workspaceTaskId ? <p className="monitor-identity"><Link to={`/history/${encodeURIComponent(workspaceTaskId)}`}>{workspaceTaskId}</Link>
          {status?.variant ? ` · ${status.variant}` : ""}{status?.preset ? ` · ${status.preset}` : ""}
          {status?.latest_progress?.label ? ` · ${String(status.latest_progress.label)}` : ""}
        </p> : null}
      </div>
      <div className="monitor-header-actions">
        <span className="monitor-state" role="status" aria-label="当前任务状态" data-state={state.dataState}>{state.label}</span>
        {workspaceTaskId && running ? <button type="button" className="monitor-stop" disabled={disconnected || stop.isPending}
          onClick={() => setStopTarget(workspaceTaskId)}><Square size={14} />{stop.isPending ? "正在停止" : "停止训练"}</button> : null}
        {recentToggle}
      </div>
    </header>
    <div className="monitor-connection-row">
      <p role="status" aria-label="实时连接状态" data-connection={connection.status}>{connection.status === "open" ? "实时连接已建立" : connection.status === "connecting"
        ? "正在连接；通过快照刷新" : "实时连接已断开；通过快照刷新"}</p>
      {statusQuery.dataUpdatedAt > 0 ? <span>任务状态最近成功读取 <time dateTime={new Date(statusQuery.dataUpdatedAt).toISOString()}>{new Date(statusQuery.dataUpdatedAt).toLocaleTimeString()}</time></span> : null}
    </div>
    {(statusQuery.error || statusQuery.isPending) ? <QueryFeedback query={statusQuery} label="任务状态" hasData={Boolean(status)} /> : null}
    {disconnected && status ? <p className="monitor-stale" role="status">
      任务状态未确认。以下为上次成功读取的任务快照，停止操作暂不可用。
    </p> : null}
    {sourceTask && status && !disconnected && sourceTask !== workspaceTaskId && <p className="data-scope" role="status">
      来源任务 {sourceTask} 已不是当前监控对象。{workspaceTaskId ? `当前显示 ${workspaceTaskId}。` : "当前没有已确认的任务。"}
      <Link to={`/history/${encodeURIComponent(sourceTask)}`}>返回来源任务</Link>
    </p>}
    {stop.error ? <section className="monitor-error" role="alert"><h2>停止训练失败</h2>
      <p>目标任务：{stop.variables}</p><p>{stop.error.message}</p>
    </section> : null}
    {workspaceTaskId
      ? <MonitorWorkspace taskId={workspaceTaskId} status={status} statusConfirmed={!disconnected} running={running}
        phase={phase} terminalEvent={terminalEvent} finalSyncError={finalSyncErrorOf(metricsQuery, logsQuery, running)}
        metricsQuery={metricsQuery} logsQuery={logsQuery} gpusQuery={gpusQuery} />
      : <IdlePanel phase={phase} hasStatus={Boolean(status)} status={status} gpusQuery={gpusQuery}
        confirmed={!disconnected && Boolean(status)} />}
    {stopTarget && stopTarget === confirmedTaskId && running && !disconnected ? <InlineConfirmDialog
      title="停止训练"
      message={`确定停止任务 ${stopTarget} 吗？`}
      confirmLabel="停止训练"
      danger
      onCancel={() => setStopTarget(null)}
      onConfirm={() => {
        if (confirmedTaskId === stopTarget && running && !stop.isPending) stop.mutate(stopTarget);
        setStopTarget(null);
      }}
    /> : null}
  </main>}</MonitorHistoryWorkspace>;
}

function finalSyncErrorOf(metricsQuery: UseQueryResult<Record<string, unknown>[], Error>, logsQuery: UseQueryResult<{ records: unknown[] }, Error>, running: boolean) {
  if (running) return undefined;
  if (metricsQuery.error && !metricsQuery.data) return metricsQuery.error.message;
  if (logsQuery.error && !logsQuery.data) return logsQuery.error.message;
  return undefined;
}

function MonitorWorkspace({ taskId, status, statusConfirmed, running, phase, terminalEvent, finalSyncError, metricsQuery, logsQuery, gpusQuery }: {
  taskId: string;
  status?: TrainingStatus;
  statusConfirmed: boolean;
  running: boolean;
  phase: string;
  terminalEvent: { taskId: string; state?: string; message?: string } | null;
  finalSyncError?: string;
  metricsQuery: UseQueryResult<Record<string, unknown>[], Error>;
  logsQuery: UseQueryResult<{ records: { id?: number; line?: string; type?: string; level?: string }[] }, Error>;
  gpusQuery: UseQueryResult<GpuInventory, Error>;
}) {
  const finished = phase === "finished";
  // A retained result must show a confirmed identity and a real end message.
  const broadcastRetention = useMemo(() => {
    if (!finished) return undefined;
    if (terminalEvent && terminalEvent.taskId === taskId && terminalEvent.message) {
      return retainedResult({
        taskId,
        message: terminalEvent.message,
        finishedAt: status?.last_output_at ? Number(status.last_output_at) : undefined,
      });
    }
    return undefined;
  }, [finished, terminalEvent, taskId, status]);
  // Hooks must run unconditionally; the history fallback decides internally
  // whether it is active (only when no broadcast retention exists).
  const historyRetention = useRetainedResultFromHistory(finished ? taskId : undefined, broadcastRetention);
  const retentionShown = broadcastRetention ?? historyRetention;
  const isTrainingJob = status?.job === "training";
  return <>
    {retentionShown ? <MonitorResultPanel retention={retentionShown} finalSyncError={finalSyncError} /> : null}
    <div className="monitor-workspace" data-has-charts={Boolean(isTrainingJob)}>
      <div className="monitor-primary-column">
        {status ? <MonitorSummary status={status} /> : null}
        {isTrainingJob ? <section className="monitor-chart-panel" aria-label="训练趋势">
          <header className="monitor-panel-header">
            <div className="monitor-panel-title">
              <Activity size={16} className="monitor-panel-icon" />
              <h2>训练趋势</h2>
            </div>
          </header>
          <QueryFeedback query={metricsQuery} label="指标" hasData={metricsQuery.data !== undefined} />
          {metricsQuery.data ? <TrainingMetricsCharts key={taskId} points={metricsQuery.data} total={status?.metric_count} /> : null}
        </section> : null}
        <MonitorSamplePreview taskId={taskId} active={running && statusConfirmed} />
      </div>
      <GpuPanel query={gpusQuery} status={status} confirmed={statusConfirmed} />
      <section className="monitor-log-panel" aria-label="实时日志">
        <header className="monitor-panel-header">
          <div className="monitor-panel-title">
            <Terminal size={16} className="monitor-panel-icon" />
            <h2>{finished ? "本次日志" : "实时日志"}</h2>
          </div>
          <Link className="monitor-log-history-link" to={`/history/${encodeURIComponent(taskId)}?view=logs`}>完整历史日志</Link>
        </header>
        <QueryFeedback query={logsQuery} label="日志" hasData={Boolean(logsQuery.data)} />
        {logsQuery.data ? <LogViewer key={taskId} lines={logsQuery.data.records} total={status?.log_count} /> : null}
      </section>
    </div>
  </>;
}

/**
 * When the terminal broadcast was missed (page reload after the process
 * exited, disconnect recovery or a refresh), read the persisted history meta
 * once and rebuild the real end outcome from its state + message. An "idle"
 * state still needs a parseable message; unknown states stay unconfirmed.
 */
function useRetainedResultFromHistory(taskId: string | undefined, existing: unknown) {
  const shouldQuery = Boolean(taskId) && !existing;
  const metaQuery = useQuery({
    queryKey: monitorResultKeys.meta(taskId || "none"),
    queryFn: ({ signal }) => fetchHistoryResultMeta(taskId!, signal),
    enabled: shouldQuery,
    staleTime: 60_000,
    retry: false,
  });
  const meta = metaQuery.data?.task;
  return useMemo(() => {
    if (!taskId || existing || !meta) return undefined;
    const outcome = outcomeFromHistoryMeta(meta.state, meta.message);
    if (outcome === "unconfirmed") return undefined;
    return retainedResult({
      taskId,
      message: meta.message,
      finishedAt: meta.finished_at,
      finishedAtText: meta.finished_at_text,
      historyOutcome: outcome,
    });
  }, [taskId, existing, meta]);
}

function IdlePanel({ phase, hasStatus, status, gpusQuery, confirmed }: {
  phase: string;
  hasStatus: boolean;
  status?: TrainingStatus;
  gpusQuery: UseQueryResult<GpuInventory, Error>;
  confirmed: boolean;
}) {
  const headings: Record<string, string> = {
    disconnected: "任务状态待确认。",
    "identity-unknown": hasStatus ? "任务身份未确认，控制暂不可用。" : "正在读取任务状态。",
    "no-task": "暂无当前任务。",
  };
  const subtitles: Record<string, string> = {
    disconnected: "无法连接到训练服务，请检查网络或服务状态；不会把未知状态当作任务结果。",
    "identity-unknown": "正在同步任务元数据，确认身份前不会读取任务数据。",
    "no-task": "训练系统处于空闲就绪状态，硬件与工作流就绪。您可直接配置新作业或在队列中查看任务。",
  };
  const key = headings[phase] ? phase : "no-task";
  return <div className="monitor-idle-device">
    <section className="monitor-idle" aria-label="空闲状态">
      <div className="monitor-idle-card">
        <div className="monitor-idle-hero">
          <div className="monitor-idle-icon" aria-hidden="true">
            <Radio size={24} />
          </div>
          <div className="monitor-idle-content">
            <h2>{headings[key]}</h2>
            <p className="monitor-idle-subtitle">{subtitles[key]}</p>
          </div>
        </div>
        {phase === "no-task" ? <nav aria-label="空闲时操作" className="monitor-idle-nav">
          <Link to="/training" className="monitor-idle-action-card">
            <div className="monitor-idle-action-text">
              <strong>训练配置</strong>
              <span>设置模型架构、数据集与训练参数</span>
            </div>
            <ArrowRight size={16} />
          </Link>
          <Link to="/queue" className="monitor-idle-action-card">
            <div className="monitor-idle-action-text">
              <strong>训练队列</strong>
              <span>管理已排队作业与调度策略</span>
            </div>
            <ArrowRight size={16} />
          </Link>
        </nav> : null}
      </div>
    </section>
    <GpuPanel query={gpusQuery} status={status} confirmed={confirmed} />
  </div>;
}

function GpuPanel({ query, status, confirmed }: {
  query: UseQueryResult<GpuInventory, Error>; status?: TrainingStatus; confirmed: boolean;
}) {
  const running = confirmed && RUNNING_STATES.has(status?.status || "");
  const selected = running && Array.isArray(status?.gpu_whitelist) ? status.gpu_whitelist : [];
  const sampledAt = query.data?.sampled_at;
  const unavailable = Boolean(query.error || query.data?.stale);
  return <section className="monitor-gpu-panel" aria-label="设备信息">
    <header className="monitor-panel-header">
      <div className="monitor-panel-title">
        <Cpu size={16} className="monitor-panel-icon" />
        <h2>设备信息</h2>
      </div>
      {typeof sampledAt === "number" && Number.isFinite(sampledAt) && !unavailable ? <time dateTime={new Date(sampledAt * 1000).toISOString()}>
        采样于 {new Date(sampledAt * 1000).toLocaleTimeString()}
      </time> : null}
    </header>
    <QueryFeedback query={query} label="GPU" hasData={Boolean(query.data)} />
    {query.data?.stale ? <p className="monitor-gpu-warning" role="status">GPU 采样暂不可用</p> : null}
    {running && selected.some((index) => !query.data?.gpus?.some((gpu) => gpu.index === index)) && !unavailable ?
      <p className="monitor-gpu-warning" role="status">当前任务所选 GPU 未全部出现在设备列表中</p> : null}
    <div className="monitor-gpu-list">
      {!unavailable && query.data?.gpus?.map((gpu, index) => <GpuDetails key={String(gpu.uuid ?? gpu.index ?? index)} gpu={gpu}
        participation={!confirmed || running && !selected.length ? "参与状态未确认" : !running ? "当前无任务" : selected.includes(Number(gpu.index)) ? "当前任务已选" : "未选用"} />)}
      {query.data && !unavailable && !query.data.gpus?.length ? <p className="monitor-empty">未检测到 GPU 信息</p> : null}
    </div>
  </section>;
}
