import { historyStateLabel } from "../training-history/historySummary";

export const RUNNING_STATES = new Set(["running", "training", "compiling", "caching", "saving"]);

export type MonitorPhase =
  | "no-task"
  | "identity-unknown"
  | "disconnected"
  | "running"
  | "finalizing"
  | "finished";

export type MonitorOutcome = "completed" | "stopped" | "failed" | "interrupted" | "unconfirmed";

export type TaskResultRetention = {
  taskId: string;
  outcome: MonitorOutcome;
  message: string;
  finishedAt?: number;
  finishedAtText?: string;
  finalized: boolean;
};

const finalizeHints: Array<{ hint: MonitorOutcome; pattern: RegExp }> = [
  { hint: "failed", pattern: /异常退出|失败|error/i },
  { hint: "stopped", pattern: /已停止/ },
  { hint: "completed", pattern: /完成/ },
];

// Broadcast terminal states are only "idle"/"error"; the real outcome lives in the message.
// Never infer "completed" from "idle" alone — an unparseable terminal message stays unconfirmed.
export function outcomeFromMessage(message?: string): MonitorOutcome {
  for (const { hint, pattern } of finalizeHints) {
    if (pattern.test(String(message || ""))) return hint;
  }
  return "unconfirmed";
}

export function isTerminalOutcome(outcome: MonitorOutcome): boolean {
  return outcome !== "unconfirmed";
}

export function outcomeLabel(outcome: MonitorOutcome): string {
  return {
    completed: "已完成",
    stopped: "已停止",
    failed: "失败",
    interrupted: "已中断",
    unconfirmed: "结果待确认",
  }[outcome];
}

/**
 * Derive the retained result of a finished task. A terminal outcome from the
 * live broadcast message or a persisted history meta may be provided; without
 * one the outcome stays unconfirmed (raw message shown verbatim).
 */
export function retainedResult(input: {
  taskId: string;
  state?: string;
  message?: string;
  finishedAt?: number;
  finishedAtText?: string;
  historyOutcome?: MonitorOutcome;
}): TaskResultRetention {
  if (input.historyOutcome && isTerminalOutcome(input.historyOutcome)) {
    return {
      taskId: input.taskId,
      outcome: input.historyOutcome,
      message: input.message
        || (input.historyOutcome === "interrupted"
          ? "服务重启时任务仍在运行，已标记为中断；本次结果以历史记录为准。"
          : "任务已结束，本次结果以历史记录为准。"),
      finishedAt: input.finishedAt,
      finishedAtText: input.finishedAtText,
      finalized: true,
    };
  }
  const outcome = outcomeFromMessage(input.message);
  return {
    taskId: input.taskId,
    outcome,
    // Keep the raw terminal message; an unparseable one is shown as-is so the
    // user can judge, and the panel must not claim a confirmed outcome.
    message: input.message || "任务已结束，但未能确认结束原因。",
    finishedAt: input.finishedAt,
    finishedAtText: input.finishedAtText,
    finalized: isTerminalOutcome(outcome) || Boolean(input.finishedAt || input.finishedAtText),
  };
}

/**
 * Pure state machine over confirmed status snapshots.
 * - "running" requires a confirmed task identity.
 * - "disconnected" marks an unknown status read; stale display data must not
 *   be presented as live or successful.
 * - "finished" covers idle snapshots that still carry the last task identity.
 */
export function monitorPhase(input: {
  statusQueryPending: boolean;
  statusQueryError: boolean;
  hasStatus: boolean;
  status?: string;
  taskId?: string;
}): MonitorPhase {
  if (input.statusQueryPending && !input.hasStatus) return "identity-unknown";
  if (input.statusQueryError) return "disconnected";
  if (!input.hasStatus) return "identity-unknown";
  if (RUNNING_STATES.has(input.status || "")) return input.taskId ? "running" : "identity-unknown";
  return input.taskId ? "finished" : "no-task";
}

export function monitorStateText(phase: MonitorPhase): { label: string; dataState: string } {
  switch (phase) {
    case "running": return { label: "运行中", dataState: "running" };
    case "disconnected": return { label: "状态待确认", dataState: "unknown" };
    case "identity-unknown": return { label: "身份未确认", dataState: "unknown" };
    case "finalizing": return { label: "最终同步中", dataState: "finalizing" };
    case "finished": return { label: "已结束", dataState: "finished" };
    default: return { label: "空闲", dataState: "idle" };
  }
}

export function outcomeTone(outcome: MonitorOutcome): "success" | "warning" | "danger" | "muted" {
  switch (outcome) {
    case "completed": return "success";
    case "stopped": return "warning";
    case "failed": return "danger";
    default: return "muted";
  }
}

/**
 * Map a persisted history meta state to a real outcome. The backend only
 * persists "idle"/"error"/"interrupted"; "idle" alone is not proof of success,
 * so the caller must combine it with the meta message (the retention builder
 * prefers an explicit message-derived outcome and stays unconfirmed otherwise).
 */
export function retentionOutcomeFromHistoryState(state?: string): MonitorOutcome | undefined {
  if (!state || state === "running") return undefined;
  if (state === "interrupted") return "interrupted";
  if (state === "stopped") return "stopped";
  if (state === "error" || state === "failed") return "failed";
  if (state === "idle" || state === "done" || state === "completed") return "completed";
  return undefined;
}

/**
 * Outcome for a persisted history meta that combines state and message.
 * - "interrupted"/"error" states are trusted directly (the backend writes them
 *   only for real interruption/failure).
 * - An "idle" state needs a parseable message: "训练完成" → completed,
 *   "训练已停止" → stopped, anything else stays unconfirmed (idle ≠ success).
 */
export function outcomeFromHistoryMeta(state?: string, message?: string): MonitorOutcome {
  if (state === "interrupted") return "interrupted";
  if (state === "error" || state === "failed") return "failed";
  if (state === "stopped") return "stopped";
  if (state === "idle" || state === "done" || state === "completed") {
    // idle alone is not proof of success; the message decides, else unconfirmed.
    return outcomeFromMessage(message);
  }
  return "unconfirmed";
}

export { historyStateLabel };
