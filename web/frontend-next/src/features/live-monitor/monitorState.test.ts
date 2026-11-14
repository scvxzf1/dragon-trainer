import { afterEach, describe, expect, it } from "vitest";

import {
  monitorPhase, monitorStateText, outcomeFromHistoryMeta, outcomeFromMessage, outcomeLabel,
  retainedResult, retentionOutcomeFromHistoryState,
} from "./monitorState";

describe("outcomeFromMessage", () => {
  it("classifies final broadcast messages without trusting bare idle", () => {
    expect(outcomeFromMessage("训练完成")).toBe("completed");
    expect(outcomeFromMessage("训练已停止")).toBe("stopped");
    expect(outcomeFromMessage("预处理完成")).toBe("completed");
    expect(outcomeFromMessage("训练异常退出 (code=1)")).toBe("failed");
    expect(outcomeFromMessage("训练异常退出 (code=137) [OOM]")).toBe("failed");
  });

  it("keeps an unparseable terminal message unconfirmed instead of guessing success", () => {
    expect(outcomeFromMessage("进程已退出")).toBe("unconfirmed");
    expect(outcomeFromMessage("")).toBe("unconfirmed");
    expect(outcomeFromMessage(undefined)).toBe("unconfirmed");
    expect(outcomeLabel("unconfirmed")).toBe("结果待确认");
  });
});

describe("retainedResult", () => {
  it("trusts a persisted history outcome with its real end time", () => {
    const result = retainedResult({
      taskId: "t1", message: "训练完成",
      finishedAt: 1760000000, finishedAtText: "2026-10-09 14:13:20", historyOutcome: "completed",
    });
    expect(result.outcome).toBe("completed");
    expect(result.finishedAtText).toBe("2026-10-09 14:13:20");
    expect(result.finalized).toBe(true);
  });

  it("keeps a persisted failed/stopped/interrupted meta outcome distinct", () => {
    expect(retainedResult({ taskId: "t", message: "训练异常退出 (code=1)", historyOutcome: "failed" }).outcome).toBe("failed");
    expect(retainedResult({ taskId: "t", message: "训练已停止", historyOutcome: "stopped" }).outcome).toBe("stopped");
    expect(retainedResult({ taskId: "t", historyOutcome: "interrupted" }).outcome).toBe("interrupted");
    expect(retainedResult({ taskId: "t", historyOutcome: "interrupted" }).message).toContain("中断");
  });

  it("does not finalize an unknown terminal outcome without a known end time", () => {
    const result = retainedResult({ taskId: "t2", message: "进程已退出" });
    expect(result.outcome).toBe("unconfirmed");
    expect(result.finalized).toBe(false);
    // The raw message is surfaced verbatim; the panel never invents an outcome.
    expect(result.message).toBe("进程已退出");
  });

  it("marks a failed message as finalized even without an end time", () => {
    const result = retainedResult({ taskId: "t3", message: "训练异常退出 (code=1)" });
    expect(result.outcome).toBe("failed");
    expect(result.finalized).toBe(true);
  });
});

describe("monitorPhase", () => {
  const base = { statusQueryPending: false, statusQueryError: false, hasStatus: true };
  it("distinguishes no task, unknown identity and disconnected states", () => {
    expect(monitorPhase({ ...base, status: "idle" })).toBe("no-task");
    expect(monitorPhase({ ...base, status: "idle", taskId: "t" })).toBe("finished");
    expect(monitorPhase({ ...base, status: "running", taskId: "t" })).toBe("running");
    expect(monitorPhase({ ...base, status: "running" })).toBe("identity-unknown");
    expect(monitorPhase({ ...base, statusQueryPending: true, hasStatus: false })).toBe("identity-unknown");
    expect(monitorPhase({ ...base, statusQueryError: true })).toBe("disconnected");
  });

  it("maps phases to honest state labels", () => {
    expect(monitorStateText("disconnected").dataState).toBe("unknown");
    expect(monitorStateText("identity-unknown").label).toBe("身份未确认");
    expect(monitorStateText("running").label).toBe("运行中");
    expect(monitorStateText("finished").label).toBe("已结束");
    expect(monitorStateText("no-task").label).toBe("空闲");
  });
});

describe("retentionOutcomeFromHistoryState", () => {
  it("maps persisted terminal states without faking success", () => {
    expect(retentionOutcomeFromHistoryState("running")).toBeUndefined();
    expect(retentionOutcomeFromHistoryState("interrupted")).toBe("interrupted");
    expect(retentionOutcomeFromHistoryState("stopped")).toBe("stopped");
    expect(retentionOutcomeFromHistoryState("error")).toBe("failed");
    expect(retentionOutcomeFromHistoryState("mystery")).toBeUndefined();
  });
});

describe("outcomeFromHistoryMeta", () => {
  it("trusts interrupted/error/stopped meta states directly", () => {
    expect(outcomeFromHistoryMeta("interrupted")).toBe("interrupted");
    expect(outcomeFromHistoryMeta("error")).toBe("failed");
    expect(outcomeFromHistoryMeta("stopped")).toBe("stopped");
  });

  it("does not treat a bare idle meta as success — the message decides", () => {
    expect(outcomeFromHistoryMeta("idle", "训练完成")).toBe("completed");
    expect(outcomeFromHistoryMeta("idle", "训练已停止")).toBe("stopped");
    expect(outcomeFromHistoryMeta("idle")).toBe("unconfirmed");
    expect(outcomeFromHistoryMeta("idle", "进程已退出")).toBe("unconfirmed");
    expect(outcomeFromHistoryMeta("idle", "训练异常退出 (code=1)")).toBe("failed");
  });

  it("keeps unknown meta states unconfirmed", () => {
    expect(outcomeFromHistoryMeta("mystery")).toBe("unconfirmed");
    expect(outcomeFromHistoryMeta(undefined)).toBe("unconfirmed");
  });
});
