import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { jsonResponse } from "../../test/renderInApp";
import { LiveMonitorPage } from "./LiveMonitorPage";

type FakeSocket = { onmessage: ((event: { data: string }) => void) | null; close: () => void };
let lastSocket: FakeSocket | null = null;

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
});

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
  lastSocket = null;
});

function bootServer() {
  const calls: string[] = [];
  // After the terminal broadcast the next HTTP snapshot reports the finished task.
  let statusPayload: Record<string, unknown> = { status: "running", job: "training", task_id: "task-A", variant: "v" };
  const setStatus = (payload: Record<string, unknown>) => { statusPayload = payload; };
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    calls.push(path);
    if (path === "/api/training/status") return jsonResponse(statusPayload);
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
    if (path.startsWith("/api/training/metrics")) return jsonResponse([{ step: 1, loss: 0.5, lr: 1e-4, kind: "train" }]);
    if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [{ id: 1, line: "step 1" }] });
    if (path.startsWith("/api/preview/images?")) return jsonResponse({ images: [{ file: "s1.png", name: "s1.png", sample: { step: 100 } }], total: 1 });
    if (path.includes("/history/task-A/logs")) return jsonResponse({ ok: true, total: 1, records: [{ id: 1, line: "step 1" }] });
    if (path.includes("/history/task-A")) return jsonResponse({
      task: { id: "task-A", job: "training", state: "idle", message: "训练完成", finished_at: 1760000000, finished_at_text: "2026-10-09 14:13:20" },
      metrics: [],
    });
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock, setStatus };
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const router = createMemoryRouter([{ path: "/monitor", element: <LiveMonitorPage /> }], { initialEntries: ["/monitor"] });
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  return client;
}

function openSocket() {
  vi.stubGlobal("WebSocket", class {
    onmessage: ((event: { data: string }) => void) | null = null;
    constructor() { lastSocket = this; }
    close() {}
  });
}

describe("task lifecycle retention", () => {
  it("keeps the finished workspace read-only with a real end message, samples, and no stop control", async () => {
    const { setStatus } = bootServer();
    openSocket();
    renderPage();
    await screen.findByRole("status", { name: "当前任务状态" });
    await waitFor(() => expect(screen.getByRole("button", { name: "停止训练" })).toBeEnabled());

    // Final broadcast for the running task, then the HTTP snapshot confirms idle.
    lastSocket!.onmessage?.({ data: JSON.stringify({ type: "status", state: "idle", task_id: "task-A", message: "训练完成" }) });
    setStatus({ status: "idle", job: "", task_id: "task-A", last_output_at: 1760000000 });
    await waitFor(() => expect(screen.getByRole("status", { name: "任务结果保留" })).toBeInTheDocument(), { timeout: 4000 });
    expect(screen.getByRole("status", { name: "任务结果保留" })).toHaveTextContent("本次结果已保留 · 已完成");
    expect(screen.getByRole("status", { name: "任务结果保留" })).toHaveTextContent("结束时间：2025/10/9 16:53:20");
    // Read-only: no stop control, samples and logs retained, full history entry exists.
    expect(screen.queryByRole("button", { name: "停止训练" })).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "最新样张" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "实时日志" })).toHaveTextContent("完整历史日志");
    expect(screen.getByRole("link", { name: "打开历史详情" })).toHaveAttribute("href", "/history/task-A");
    expect(screen.getByRole("status", { name: "当前任务状态" })).toHaveTextContent("已结束");
  });

  it("does not fabricate a completed result when the terminal message is unparseable", async () => {
    const { fetchMock, setStatus } = bootServer();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") return jsonResponse(currentStatus.status === "idle"
        ? { status: "idle", job: "", task_id: "task-A" }
        : { status: "running", job: "training", task_id: "task-A" });
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
      if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
      if (path.startsWith("/api/preview/images?")) return jsonResponse({ images: [], total: 0 });
      if (path.includes("/history/task-A/logs")) return jsonResponse({ ok: true, total: 0, records: [] });
      if (path.includes("/history/task-A")) return jsonResponse({ task: { id: "task-A", job: "training", state: "idle" }, metrics: [] });
      throw new Error(`Unexpected ${path}`);
    });
    const currentStatus = { status: "running" as string };
    openSocket();
    renderPage();
    await waitFor(() => expect(screen.getByRole("button", { name: "停止训练" })).toBeEnabled());
    lastSocket!.onmessage?.({ data: JSON.stringify({ type: "status", state: "idle", task_id: "task-A", message: "进程已退出" }) });
    currentStatus.status = "idle";
    const panel = await screen.findByRole("status", { name: "任务结果保留" }, { timeout: 4000 });
    expect(panel).toHaveTextContent("结果待确认");
    expect(panel).toHaveTextContent("进程已退出");
    expect(panel).not.toHaveTextContent("已完成");
  });

  it("surfaces a persisted interrupted meta when the terminal broadcast was missed", async () => {
    const { fetchMock } = bootServer();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") return jsonResponse({ status: "idle", job: "", task_id: "task-A" });
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
      if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
      if (path.startsWith("/api/preview/images?")) return jsonResponse({ images: [], total: 0 });
      if (path.includes("/history/task-A/logs")) return jsonResponse({ ok: true, total: 0, records: [] });
      if (path.includes("/history/task-A")) return jsonResponse({
        task: { id: "task-A", job: "training", state: "interrupted", message: "WebUI 上次退出时任务仍标记为运行中，已自动标记为中断。", finished_at_text: "2026-10-09 15:00:00" },
        metrics: [],
      });
      throw new Error(`Unexpected ${path}`);
    });
    openSocket();
    renderPage();
    const panel = await screen.findByRole("status", { name: "任务结果保留" }, { timeout: 4000 });
    expect(panel).toHaveTextContent("已中断");
    expect(panel).toHaveTextContent("结束时间：2026-10-09 15:00:00");
  });
});

describe("F1/F2 repair regressions", () => {
  it("F1: renders the task progress and live metric summary on the monitor workspace", async () => {
    const { setStatus } = bootServer();
    openSocket();
    renderPage();
    // Progress card and live metric fields exist on the running workspace.
    expect(await screen.findByRole("region", { name: "任务进度" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "实时指标" })).toBeInTheDocument();
    // Both remain present on the finished retained view (read-only summary).
    lastSocket!.onmessage?.({ data: JSON.stringify({ type: "status", state: "idle", task_id: "task-A", message: "训练完成" }) });
    setStatus({ status: "idle", job: "training", task_id: "task-A", latest_progress: { current: 10, total: 10 } });
    await waitFor(() => expect(screen.getByRole("status", { name: "任务结果保留" })).toBeInTheDocument(), { timeout: 4000 });
    expect(screen.getByRole("region", { name: "任务进度" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "实时指标" })).toBeInTheDocument();
  }, 10_000);

  it.each([
    ["completed", "训练完成", "本次结果已保留 · 已完成"],
    ["stopped", "训练已停止", "本次结果已保留 · 已停止"],
  ])("F2: rebuilds %s from an idle history meta with its known end time", async (_name, message, expected) => {
    const { fetchMock } = bootServer();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") return jsonResponse({ status: "idle", job: "", task_id: "task-A" });
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
      if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
      if (path.startsWith("/api/preview/images?")) return jsonResponse({ images: [], total: 0 });
      if (path.includes("/history/task-A/logs")) return jsonResponse({ ok: true, total: 0, records: [] });
      if (path.includes("/history/task-A")) return jsonResponse({
        task: { id: "task-A", job: "training", state: "idle", message, finished_at: 1760000000, finished_at_text: "2026-10-09 14:13:20" },
        metrics: [],
      });
      throw new Error(`Unexpected ${path}`);
    });
    openSocket();
    renderPage();
    const panel = await screen.findByRole("status", { name: "任务结果保留" }, { timeout: 4000 });
    expect(panel).toHaveTextContent(expected);
    expect(panel).toHaveTextContent("结束时间：2026-10-09 14:13:20");
  });

  it("F2: an idle history meta without a parseable message stays 结果待确认, never completed", async () => {
    const { fetchMock } = bootServer();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") return jsonResponse({ status: "idle", job: "", task_id: "task-A" });
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
      if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
      if (path.startsWith("/api/preview/images?")) return jsonResponse({ images: [], total: 0 });
      if (path.includes("/history/task-A/logs")) return jsonResponse({ ok: true, total: 0, records: [] });
      if (path.includes("/history/task-A")) return jsonResponse({
        task: { id: "task-A", job: "training", state: "idle", message: "", finished_at: 1760000000 },
        metrics: [],
      });
      throw new Error(`Unexpected ${path}`);
    });
    openSocket();
    renderPage();
    // No retention banner: idle without a message is not success and not retained as one.
    await waitFor(() => expect(screen.getByRole("status", { name: "当前任务状态" })).toHaveTextContent("已结束"), { timeout: 4000 });
    expect(screen.queryByRole("status", { name: "任务结果保留" })).not.toBeInTheDocument();
    expect(screen.queryByText("本次结果已保留 · 已完成")).not.toBeInTheDocument();
  });

  it("F2: rebuilds a failed result from an error history meta", async () => {
    const { fetchMock } = bootServer();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") return jsonResponse({ status: "idle", job: "", task_id: "task-A" });
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
      if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
      if (path.startsWith("/api/preview/images?")) return jsonResponse({ images: [], total: 0 });
      if (path.includes("/history/task-A/logs")) return jsonResponse({ ok: true, total: 0, records: [] });
      if (path.includes("/history/task-A")) return jsonResponse({
        task: { id: "task-A", job: "training", state: "error", message: "训练异常退出 (code=1)", finished_at_text: "2026-10-09 15:30:00" },
        metrics: [],
      });
      throw new Error(`Unexpected ${path}`);
    });
    openSocket();
    renderPage();
    const panel = await screen.findByRole("status", { name: "任务结果保留" }, { timeout: 4000 });
    expect(panel).toHaveTextContent("本次结果已保留 · 失败");
    expect(panel).toHaveTextContent("结束时间：2026-10-09 15:30:00");
  });

  it("F2: keeps an explicit stopped meta as 已停止, not 已中断", async () => {
    const { fetchMock } = bootServer();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") return jsonResponse({ status: "idle", job: "", task_id: "task-A" });
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
      if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
      if (path.startsWith("/api/preview/images?")) return jsonResponse({ images: [], total: 0 });
      if (path.includes("/history/task-A/logs")) return jsonResponse({ ok: true, total: 0, records: [] });
      if (path.includes("/history/task-A")) return jsonResponse({
        task: { id: "task-A", job: "training", state: "stopped", message: "训练已停止", finished_at_text: "2026-10-09 16:00:00" },
        metrics: [],
      });
      throw new Error(`Unexpected ${path}`);
    });
    openSocket();
    renderPage();
    const panel = await screen.findByRole("status", { name: "任务结果保留" }, { timeout: 4000 });
    expect(panel).toHaveTextContent("本次结果已保留 · 已停止");
    expect(panel).not.toHaveTextContent("已中断");
  });

  it("stays honest while disconnected and keeps the stop control unavailable", async () => {
    const { fetchMock } = bootServer();
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") return jsonResponse({ error: "down" }, 503);
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      throw new Error(`Unexpected ${path}`);
    });
    openSocket();
    renderPage();
    // While the first status read is pending the identity is honestly unknown;
    // after the read fails the state must say so and keep controls disabled.
    expect(await screen.findByRole("status", { name: "当前任务状态" })).toHaveTextContent("身份未确认");
    await waitFor(() => expect(screen.getByRole("status", { name: "当前任务状态" })).toHaveTextContent("状态待确认"));
    expect(screen.queryByRole("button", { name: "停止训练" })).not.toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "任务结果保留" })).not.toBeInTheDocument();
  });

  it("switches to a new task identity and keeps late responses of the old task from winning", async () => {
    const { fetchMock } = bootServer();
    let resolveOldMetrics: ((response: Response) => void) | undefined;
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/training/status") {
        // First read: old task. After the terminal event, a new task is running.
        return jsonResponse(statusPayload.status === "running"
          ? { status: "running", job: "training", task_id: "task-A" }
          : { status: "running", job: "training", task_id: "task-B" });
      }
      if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
      if (path.startsWith("/api/training/metrics")) {
        if (path.includes("task_id=task-A")) {
          return new Promise<Response>((resolve) => { resolveOldMetrics = resolve; });
        }
        return jsonResponse([{ step: 9, loss: 0.1, kind: "train" }]);
      }
      if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [{ id: 1, line: "new task log" }] });
      if (path.startsWith("/api/preview/images?")) {
        const taskId = String(new URLSearchParams(path.split("?")[1]).get("task_id"));
        return jsonResponse({ images: [{ file: `${taskId}.png`, name: `${taskId}.png` }], total: 1 });
      }
      if (path.includes("/history/")) return jsonResponse({ task: { id: "task-B", job: "training", state: "running" }, metrics: [] });
      throw new Error(`Unexpected ${path}`);
    });
    const statusPayload = { status: "running" as string };
    openSocket();
    renderPage();
    await waitFor(() => expect(screen.getByRole("button", { name: "停止训练" })).toBeEnabled());
    expect(screen.getByText("task-A")).toBeInTheDocument();

    // Old task finishes; a new task then starts (next status read returns task-B).
    lastSocket!.onmessage?.({ data: JSON.stringify({ type: "status", state: "idle", task_id: "task-A", message: "训练完成" }) });
    statusPayload.status = "started";
    await waitFor(() => expect(screen.getByText("task-B")).toBeInTheDocument());
    // The retention banner must be gone: the finished task no longer owns the monitor.
    expect(screen.queryByRole("status", { name: "任务结果保留" })).not.toBeInTheDocument();
    // The stale metrics request for task-A is resolved now — the page must show
    // only task-B data, never the late old-task payload.
    await actuallyResolve(resolveOldMetrics);
    expect(screen.queryByText("task-A")).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "最新样张" })).toHaveTextContent("task-B.png");
    // Even after the stale response settles, its task-A payload must not appear.
    await waitFor(() => expect(screen.queryByText("task-A.png")).not.toBeInTheDocument());
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByText("task-A.png")).not.toBeInTheDocument();
  }, 15_000);
});

async function actuallyResolve(resolve?: (response: Response) => void) {
  // Give the pending promise a settle tick; its value must be ignored by the page.
  resolve?.(jsonResponse([{ step: 0, loss: 99, kind: "train" }]));
  await new Promise((resolveTick) => setTimeout(resolveTick, 0));
}
