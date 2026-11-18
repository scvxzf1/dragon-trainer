import { expect, test, type Page } from "@playwright/test";
import { mockWorkspace } from "./fixtures";

async function deviceFixture(page: Page) {
  const mocks = await mockWorkspace(page);
  let gpus = [{ index: 0, name: "GPU Alpha", memory_total_gb: 24 }, { index: 1, name: "GPU Beta", memory_total_gb: 32 }];
  const commands: { path: string; ids: string[]; body: Record<string, unknown> }[] = [];
  await page.route((url) => url.pathname === "/api/training/gpus", (route) => route.fulfill({ json: { gpus } }));
  await page.route((url) => ["/api/training/preflight", "/api/training/start", "/api/training/queue"].includes(url.pathname), (route) => {
    if (route.request().method() === "GET") return route.fallback();
    const path = new URL(route.request().url()).pathname;
    const body = route.request().postDataJSON() as Record<string, unknown>;
    commands.push({ path, ids: body.gpu_whitelist as string[], body });
    return route.fulfill({ json: path.endsWith("preflight")
      ? { ok: true, summary: { errors: 0, warnings: 0, checks: 1 }, checks: [{ level: "ok", key: "gpu", message: "设备检查通过" }] }
      : { ok: true, message: "模拟请求已完成" } });
  });
  return { ...mocks, commands, removeSecond: () => { gpus = gpus.slice(0, 1); } };
}

for (const width of [1440, 390]) {
  test(`shared GPU selection and launch contracts ${width}`, async ({ page }, info) => {
    const mocks = await deviceFixture(page);
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/next/training");
    const quick = page.getByRole("region", { name: "训练设备快捷选择" });
    await expect(quick.getByRole("radio", { name: /GPU 0/ })).toBeChecked();
    await quick.getByRole("radio", { name: /GPU 1/ }).check();
    await page.reload();
    await expect(quick.getByRole("radio", { name: /GPU 1/ })).toBeChecked();
    await page.getByRole("button", { name: "运行预检测", exact: true }).click();
    await expect(page.getByRole("dialog")).toContainText("设备检查通过");
    expect(mocks.commands.at(-1)?.ids).toEqual(["1"]);
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "立即启动", exact: true }).click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toContainText("单卡 · GPU 1 · GPU Beta");
    expect(await dialog.getByRole("radio").count()).toBe(0);
    await dialog.getByRole("checkbox").check();
    await dialog.getByRole("button", { name: "确认启动", exact: true }).click();
    await expect(dialog).toContainText("模拟请求已完成");
    const startCommand = mocks.commands.at(-1);
    expect(startCommand).toMatchObject({ path: "/api/training/start", ids: ["1"] });
    expect(startCommand?.body).toEqual({
      variant: "lora",
      preset: "default",
      methods_subdir: "imported",
      config_file: "configs/imported/studio-portrait.toml",
      confirmed: true,
      confirm_preprocess: true,
      gpu_whitelist: ["1"],
    });
    await page.keyboard.press("Escape");
    await quick.getByRole("button", { name: "多卡 · 数据并行" }).click();
    await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
    await quick.getByRole("checkbox", { name: /GPU 0/ }).check();
    await page.getByRole("tab", { name: "设备与性能", exact: true }).click();
    const details = page.getByRole("region", { name: "设备与并行设置" });
    await expect(details).toContainText("数据并行 · GPU 1 · GPU Beta / GPU 0 · GPU Alpha");
    await page.screenshot({ path: info.outputPath("devices-performance.png") });
    if (width < 800) {
      await page.getByRole("button", { name: /精度与计算后端/ }).scrollIntoViewIfNeeded();
      await page.screenshot({ path: info.outputPath("devices-performance-scrolled.png") });
    }
    await page.getByRole("button", { name: "加入队列", exact: true }).click();
    await expect(dialog).toContainText("数据并行");
    await dialog.getByRole("checkbox").check();
    await dialog.getByRole("button", { name: "确认入队", exact: true }).click();
    await expect(dialog).toContainText("模拟请求已完成");
    expect(mocks.commands.slice(-2).map(({ path, ids }) => ({ path, ids }))).toEqual([
      { path: "/api/training/preflight", ids: ["1", "0"] },
      { path: "/api/training/queue", ids: ["1", "0"] },
    ]);
    await page.keyboard.press("Escape");
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    expect(mocks.writes).toEqual([]);
    expect(mocks.unhandled).toEqual([]);
  });
}

test("missing saved GPU blocks launch instead of falling back", async ({ page }) => {
  const mocks = await deviceFixture(page);
  await page.goto("/next/training");
  const quick = page.getByRole("region", { name: "训练设备快捷选择" });
  await quick.getByRole("radio", { name: /GPU 1/ }).check();
  mocks.removeSecond();
  const refreshed = page.waitForRequest((request) => request.url().endsWith("/api/training/gpus?refresh=1"));
  await quick.getByRole("button", { name: "刷新 GPU 列表" }).click();
  await refreshed;
  await expect(quick).toContainText("已选设备不可用");
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await quick.getByRole("radio", { name: /GPU 0/ }).check();
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeEnabled();
  expect(mocks.commands).toEqual([]);
});

test("GPU inventory errors and empty inventory block execution but allow editing", async ({ page }) => {
  const mocks = await deviceFixture(page);
  await page.route((url) => url.pathname === "/api/training/gpus", (route) => route.fulfill({ status: 503, json: { error: "GPU inventory offline" } }));
  await page.goto("/next/training");
  const quick = page.getByRole("region", { name: "训练设备快捷选择" });
  await expect(quick).toContainText("无法读取 GPU");
  await expect(page.getByRole("button", { name: "运行预检测", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await page.getByRole("tab", { name: "训练计划", exact: true }).click();
  await expect(page.getByLabel("输出名称", { exact: true })).toBeEnabled();
  await page.route((url) => url.pathname === "/api/training/gpus", (route) => route.fulfill({ json: { gpus: [] } }));
  await quick.getByRole("button", { name: "刷新 GPU 列表" }).click();
  await expect(quick).toContainText("未检测到可用 GPU");
  expect(mocks.commands).toEqual([]);
});

test("GPU inventory loading exposes a busy device region until the read settles", async ({ page }) => {
  const mocks = await mockWorkspace(page);
  let release = () => {};
  const gate = new Promise<void>((resolve) => { release = resolve; });
  await page.route((url) => url.pathname === "/api/training/gpus", async (route) => {
    await gate;
    await route.fulfill({ json: { gpus: [{ index: 0, name: "GPU Alpha", memory_total_gb: 24 }] } });
  });

  await page.goto("/next/training");
  const quick = page.getByRole("region", { name: "训练设备快捷选择" });
  await expect(quick).toHaveAttribute("aria-busy", "true");
  await expect(quick).toContainText("正在读取 GPU");
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();

  release();
  await expect(quick).toHaveAttribute("aria-busy", "false");
  await expect(quick.getByRole("radio", { name: /GPU 0/ })).toBeVisible();
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training start failures do not auto-retry and require fresh confirmation", async ({ page }) => {
  const mocks = await deviceFixture(page);
  const attempts: { method: string; path: string; body: Record<string, unknown> }[] = [];
  let serverAccepted = false;
  let announceStartPending = () => {};
  let releaseStartResponse = () => {};
  const startPending = new Promise<void>((resolve) => { announceStartPending = resolve; });
  const startResponseGate = new Promise<void>((resolve) => { releaseStartResponse = resolve; });
  await page.route((url) => url.pathname === "/api/training/start", async (route) => {
    attempts.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON() as Record<string, unknown>,
    });
    if (attempts.length === 1) {
      announceStartPending();
      await startResponseGate;
      serverAccepted = true;
      return route.fulfill({ status: 500, json: { error: "injected post-launch failure" } });
    }
    return route.fulfill({ status: 409, json: { error: "已有任务在运行中" } });
  });
  await page.route((url) => url.pathname === "/api/training/status", (route) =>
    route.fulfill({ json: serverAccepted
      ? { task_id: "post-launch-task", status: "running", job: "training" }
      : { status: "idle" } }),
  );

  await page.goto("/next/training");
  await page.getByRole("button", { name: "立即启动", exact: true }).click();
  let dialog = page.getByRole("dialog", { name: "确认启动训练" });
  await expect(dialog).toContainText("设备检查通过");
  await dialog.getByRole("checkbox").check();
  const confirm = dialog.getByRole("button", { name: "确认启动", exact: true });
  await confirm.click();
  await startPending;
  await expect(dialog.getByRole("button", { name: "正在提交", exact: true })).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "取消", exact: true })).toBeDisabled();
  expect(attempts).toHaveLength(1);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible();
  expect(attempts).toHaveLength(1);

  releaseStartResponse();
  await expect(dialog.getByRole("alert")).toContainText("injected post-launch failure");
  await expect(dialog.getByRole("alert")).toContainText("启动结果可能未知");
  await expect(confirm).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "取消", exact: true })).toBeEnabled();
  expect(attempts).toHaveLength(1);
  await dialog.getByRole("button", { name: "取消", exact: true }).click();
  await expect(dialog).not.toBeVisible();

  await page.getByRole("button", { name: "立即启动", exact: true }).click();
  dialog = page.getByRole("dialog", { name: "确认启动训练" });
  await expect(dialog).toContainText("设备检查通过");
  await expect.poll(() => mocks.commands.filter((command) => command.path === "/api/training/preflight").length).toBe(2);
  await expect(dialog.getByRole("checkbox")).not.toBeChecked();
  await dialog.getByRole("checkbox").check();
  await dialog.getByRole("button", { name: "确认启动", exact: true }).click();
  await expect(dialog.getByRole("alert")).toContainText("已有任务在运行中");
  await expect(dialog.getByRole("button", { name: "确认启动", exact: true })).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "取消", exact: true })).toBeEnabled();
  const monitorLink = dialog.getByRole("link", { name: "查看当前监控，核对启动结果", exact: true });
  await expect(monitorLink).toBeVisible();
  expect(attempts).toHaveLength(2);
  expect(attempts.every((attempt) => attempt.method === "POST" && attempt.path === "/api/training/start")).toBe(true);
  expect(attempts.every((attempt) => attempt.body.confirmed && attempt.body.confirm_preprocess)).toBe(true);
  expect(mocks.commands.filter((command) => command.path === "/api/training/preflight")).toHaveLength(2);
  await monitorLink.click();
  await expect(page).toHaveURL(/\/next\/monitor$/);
  await expect(page.getByRole("heading", { name: "当前监控", exact: true })).toBeVisible();
  await expect(page.locator(".monitor-state")).toHaveText("运行中");
  await expect(page.getByRole("link", { name: "post-launch-task", exact: true })).toBeVisible();
  expect(attempts).toHaveLength(2);
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("training start result is reconciled in the monitor after acceptance and response loss", async ({ page }) => {
  const mocks = await deviceFixture(page);
  let serverAccepted = false;
  const attempts: { method: string; path: string; body: Record<string, unknown> }[] = [];
  await page.route((url) => url.pathname === "/api/training/start", (route) => {
    attempts.push({
      method: route.request().method(),
      path: new URL(route.request().url()).pathname,
      body: route.request().postDataJSON() as Record<string, unknown>,
    });
    serverAccepted = true;
    return route.abort();
  });
  await page.route((url) => url.pathname === "/api/training/status", (route) =>
    route.fulfill({ json: serverAccepted
      ? { task_id: "accepted-task", status: "running", job: "training" }
      : { status: "idle" } }),
  );

  await page.goto("/next/training");
  await page.getByRole("button", { name: "立即启动", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "确认启动训练" });
  await expect(dialog).toContainText("设备检查通过");
  await dialog.getByRole("checkbox").check();
  await dialog.getByRole("button", { name: "确认启动", exact: true }).click();

  await expect(dialog.getByRole("alert")).toContainText("操作结果尚未确认");
  await expect(dialog.getByRole("alert")).toContainText("启动结果可能未知");
  await expect(dialog.getByRole("button", { name: "确认启动", exact: true })).toBeDisabled();
  const monitorLink = dialog.getByRole("link", { name: "查看当前监控，核对启动结果", exact: true });
  await expect(monitorLink).toBeVisible();
  expect(attempts).toHaveLength(1);

  await monitorLink.click();
  await expect(page).toHaveURL(/\/next\/monitor$/);
  await expect(page.locator(".monitor-state")).toHaveText("运行中");
  await expect(page.getByRole("link", { name: "accepted-task", exact: true })).toBeVisible();
  expect(attempts).toHaveLength(1);
  expect(attempts[0].method).toBe("POST");
  expect(attempts[0].path).toBe("/api/training/start");
  expect(mocks.writes).toEqual([]);
  expect(mocks.unhandled).toEqual([]);
});

test("imported unavailable pipeline cannot launch and can explicitly return to single GPU", async ({ page }) => {
  const mocks = await deviceFixture(page);
  await page.route("**/api/config/merged?**", (route) => route.fulfill({ json: { model_family: "krea2_raw", pipeline_parallel: true } }));
  await page.goto("/next/training");
  const quick = page.getByRole("region", { name: "训练设备快捷选择" });
  await expect(quick).toContainText("当前配置启用了");
  await expect(page.getByRole("button", { name: "立即启动", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "加入队列", exact: true })).toBeDisabled();
  await quick.getByRole("button", { name: "单卡", exact: true }).click();
  await expect(quick).not.toContainText("当前配置启用了");
  await expect(page.getByRole("button", { name: "保存并启动", exact: true })).toBeEnabled();
  expect(mocks.commands).toEqual([]);
});
