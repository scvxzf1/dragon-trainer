import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { jsonResponse, renderInApp } from "../../test/renderInApp";
import { LiveMonitorPage } from "./LiveMonitorPage";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
});

beforeEach(() => vi.stubGlobal("ResizeObserver", class {
  observe() {}
  unobserve() {}
  disconnect() {}
}));

it.each([
  ["interrupted", "已结束", "finished", false],
  ["failed", "已结束", "finished", false],
  ["error", "已结束", "finished", false],
  ["unavailable", "已结束", "finished", false],
  ["running", "运行中", "running", true],
])("renders %s as the correct non-running/running state", async (status, label, dataState, canStop) => {
  vi.stubGlobal("WebSocket", class {
    close() {}
  });
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path === "/api/training/status") return jsonResponse({ status, task_id: "task-1" });
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
    if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
    if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
    throw new Error(`Unexpected ${path}`);
  }));

  renderInApp(<LiveMonitorPage />);
  const state = await screen.findByRole("status", { name: "当前任务状态" });
  await waitFor(() => expect(state).toHaveTextContent(label));
  expect(state).toHaveAttribute("data-state", dataState);
  if (canStop) expect(screen.getByRole("button", { name: "停止训练" })).toHaveProperty("disabled", false);
  else expect(screen.queryByRole("button", { name: "停止训练" })).not.toBeInTheDocument();
});

it("pauses hidden polling and refreshes a stale snapshot on resume without leaking an in-flight refresh", async () => {
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
  let releaseStatus: ((response: Response) => void) | undefined;
  let statusCalls = 0;
  let metricsCalls = 0;
  let logsCalls = 0;
  let gpuCalls = 0;
  vi.stubGlobal("WebSocket", class { close() {} });
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path === "/api/training/status") {
      statusCalls += 1;
      if (statusCalls === 2) return new Promise<Response>((resolve) => { releaseStatus = resolve; });
      return Promise.resolve(jsonResponse({ status: "running", job: "training", task_id: "task-1" }));
    }
    if (path === "/api/training/gpus") {
      gpuCalls += 1;
      return Promise.resolve(jsonResponse({ gpus: [] }));
    }
    if (path.startsWith("/api/training/metrics")) {
      metricsCalls += 1;
      return Promise.resolve(jsonResponse([]));
    }
    if (path.startsWith("/api/training/logs")) {
      logsCalls += 1;
      return Promise.resolve(jsonResponse({ records: [] }));
    }
    throw new Error(`Unexpected ${path}`);
  }));

  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 15_000 }, mutations: { retry: false } },
  });
  const router = createMemoryRouter([{ path: "/", element: <LiveMonitorPage /> }]);
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  await screen.findByRole("status", { name: "当前任务状态" });
  await waitFor(() => expect([metricsCalls, logsCalls, gpuCalls]).toEqual([1, 1, 1]));

  Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  await new Promise((resolve) => setTimeout(resolve, 5200));
  expect([statusCalls, metricsCalls, logsCalls, gpuCalls]).toEqual([1, 1, 1, 1]);
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  await waitFor(() => expect(statusCalls).toBe(2));
  await waitFor(() => expect(releaseStatus).toBeTypeOf("function"));

  Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  await act(async () => releaseStatus?.(jsonResponse({ status: "running", job: "training", task_id: "task-1" })));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect([metricsCalls, logsCalls, gpuCalls]).toEqual([1, 1, 1]);

  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  await waitFor(() => expect(statusCalls).toBe(3));
  await waitFor(() => expect(metricsCalls).toBe(2));
  await waitFor(() => expect(logsCalls).toBe(2));
  await waitFor(() => expect(gpuCalls).toBe(2));
  client.clear();
}, 10_000);

it("refreshes GPU inventory after an idle monitor becomes visible", async () => {
  let statusCalls = 0;
  let gpuCalls = 0;
  vi.stubGlobal("WebSocket", class { close() {} });
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path === "/api/training/status") {
      statusCalls += 1;
      return Promise.resolve(jsonResponse({ status: "idle" }));
    }
    if (path === "/api/training/gpus") {
      gpuCalls += 1;
      return Promise.resolve(jsonResponse({ gpus: [] }));
    }
    throw new Error(`Unexpected ${path}`);
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 15_000 } } });
  const router = createMemoryRouter([{ path: "/", element: <LiveMonitorPage /> }]);
  render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  await waitFor(() => expect([statusCalls, gpuCalls]).toEqual([1, 1]));

  Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
  act(() => document.dispatchEvent(new Event("visibilitychange")));
  await waitFor(() => expect([statusCalls, gpuCalls]).toEqual([2, 2]));
  client.clear();
});
