import { useState } from "react";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { jsonResponse } from "../../test/renderInApp";
import { LiveMonitorPage } from "./LiveMonitorPage";
import { liveMonitorKeys } from "./api";

vi.mock("../../components/TrainingMetricsCharts", () => ({
  TrainingMetricsCharts: function ChartControls() {
    const [range, setRange] = useState("all");
    return <label>图表范围<select value={range} onChange={(event) => setRange(event.target.value)}>
      <option value="all">全部</option><option value="recent">最近</option>
    </select></label>;
  },
}));
vi.mock("../training-history/HistoryOverview", () => ({ HistoryOverview: () => <p>历史概览内容</p> }));
vi.mock("../training-history/HistorySnapshot", () => ({ HistorySnapshot: () => <p>历史配置内容</p> }));
vi.mock("../training-history/HistoryAssets", () => ({ HistoryAssets: () => <p>历史产物内容</p> }));
vi.mock("../training-history/HistoryLogs", () => ({ HistoryLogs: () => <p>历史日志内容</p> }));
vi.mock("../training-history/HistoryGpuMetrics", () => ({ HistoryGpuMetrics: () => <p>历史设备指标</p> }));

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

beforeEach(() => vi.stubGlobal("ResizeObserver", class {
  observe() {}
  unobserve() {}
  disconnect() {}
}));

function setup(historyResponse: () => Response) {
  vi.stubGlobal("WebSocket", class { close() {} });
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input);
    if (path.startsWith("/api/training/history?")) return historyResponse();
    if (path.startsWith("/api/training/history/")) return jsonResponse({
      task: { id: path.split("/").at(-1), name: "已选历史任务", job: "training", state: "idle" }, metrics: [],
    });
    if (path === "/api/training/status") return jsonResponse({ status: "running", task_id: "current", job: "training" });
    if (path === "/api/training/gpus") return jsonResponse({ gpus: [] });
    if (path.startsWith("/api/training/metrics")) return jsonResponse([]);
    if (path.startsWith("/api/training/logs")) return jsonResponse({ records: [] });
    throw new Error(`Unexpected ${path}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const router = createMemoryRouter([{ path: "/monitor", element: <LiveMonitorPage /> }], {
    initialEntries: ["/monitor?from_task=current"],
  });
  const view = render(<QueryClientProvider client={client}><RouterProvider router={router} /></QueryClientProvider>);
  return { ...view, client, router, fetchMock, user: userEvent.setup() };
}

it("requests the recent 20 tasks and presents them newest first with task status", async () => {
  const tasks = Array.from({ length: 25 }, (_, index) => ({ id: `task-${index}`, name: `任务 ${index}`, started_at: index + 1, state: "idle" }));
  const { user, fetchMock } = setup(() => jsonResponse({ tasks }));
  expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/history?"))).toBe(false);
  await user.click(screen.getByRole("button", { name: "最近训练" }));
  const sidebar = screen.getByRole("complementary", { name: "最近训练" });
  await waitFor(() => expect(within(sidebar).getAllByRole("listitem")).toHaveLength(20));
  const items = within(sidebar).getAllByRole("listitem");
  expect(items[0]).toHaveTextContent("任务 24");
  expect(items[19]).toHaveTextContent("任务 5");
  expect(items[0]).toHaveTextContent("完成");
  expect(fetchMock).toHaveBeenCalledWith("/api/training/history?limit=20&include_archived=1", expect.anything());
});

it("views every history tab without changing the route and restores the mounted monitor scroll and chart controls", async () => {
  const { user, router, container, client } = setup(() => jsonResponse({ tasks: [{ id: "past #1", name: "过去任务", started_at: 1 }] }));
  const range = await screen.findByLabelText("图表范围");
  await user.selectOptions(range, "recent");
  const monitor = container.querySelector<HTMLDivElement>(".monitor-current-view")!;
  await user.click(screen.getByRole("button", { name: "最近训练" }));
  monitor.scrollTop = 460;
  await user.click(await screen.findByRole("button", { name: /过去任务/ }));
  await screen.findByRole("heading", { name: "已选历史任务" });
  expect(monitor).toHaveAttribute("hidden");
  expect(screen.getByRole("button", { name: /过去任务/ })).toHaveAttribute("aria-pressed", "true");
  expect(range).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "检查点续训" })).not.toBeInTheDocument();
  expect(client.getQueryCache().find({ queryKey: liveMonitorKeys.status })?.getObserversCount()).toBe(1);
  for (const [label, content] of [["配置", "历史配置内容"], ["产物", "历史产物内容"], ["日志", "历史日志内容"], ["指标", "历史设备指标"], ["概览", "历史概览内容"]]) {
    await user.click(screen.getByRole("button", { name: label }));
    expect(await screen.findByText(content)).toBeInTheDocument();
    expect(router.state.location.pathname + router.state.location.search).toBe("/monitor?from_task=current");
  }
  monitor.scrollTop = 0;
  await user.click(screen.getByRole("button", { name: "返回当前监控" }));
  expect(monitor).not.toHaveAttribute("hidden");
  expect(monitor.scrollTop).toBe(460);
  expect(range).toHaveValue("recent");
  expect(screen.getByRole("complementary", { name: "最近训练" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /过去任务/ }));
  await screen.findByRole("button", { name: "返回当前监控" });
  await user.click(screen.getByRole("button", { name: "关闭最近训练" }));
  expect(monitor).not.toHaveAttribute("hidden");
  expect(monitor.scrollTop).toBe(460);
  expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "最近训练" })).toHaveAttribute("aria-expanded", "false");
});

it("offers a retry after a history read fails and then displays an empty list", async () => {
  let reads = 0;
  const { user } = setup(() => ++reads === 1 ? jsonResponse({ error: "读取失败" }, 500) : jsonResponse({ tasks: [] }));
  await user.click(screen.getByRole("button", { name: "最近训练" }));
  const sidebar = await screen.findByRole("complementary", { name: "最近训练" });
  await waitFor(() => expect(within(sidebar).getByRole("alert")).toHaveTextContent("无法读取最近训练"));
  await user.click(within(sidebar).getByRole("button", { name: "重新读取" }));
  expect(await within(sidebar).findByText("暂无历史任务")).toBeInTheDocument();
});
