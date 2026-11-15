import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { trainingContextKeys } from "../../api/trainingContext";
import { trainingConfigKeys } from "./api";
import { useTrainingContextStore } from "../../app/trainingContextStore";
import { TrainingWorkspace } from "./TrainingWorkspace";
import { useHotstartIntent } from "./hotstartIntent";
import { TRAINING_DETAILED_MANAGEMENT_KEY } from "./TrainingConfigLibrary";

function renderWorkspace() {
  const router = createMemoryRouter(
    [
      { path: "/training", element: <TrainingWorkspace /> },
      { path: "/other", element: <main>其他页面</main> },
    ],
    { initialEntries: ["/training"] },
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return {
    router,
    client,
    ...render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
  };
}

function createFetchMock(includeSecondConfig = false) {
  let content = 'output_name = "dragon-run"\nmax_train_steps = 1600\n';
  let merged = {
    output_name: "dragon-run",
    model_family: "krea2_raw",
    pretrained_model_name_or_path: "models/old-dit",
    qwen3: "models/old-qwen",
    vae: "models/old-vae",
    max_train_steps: 1600,
    train_batch_size: 1,
    dataset_config: "configs/datasets/alpha.toml",
    gradient_checkpointing: true,
  } as Record<string, unknown>;
  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || "GET";
      if (url === "/api/config/dataset-presets") return jsonResponse({ presets: [], groups: [] });
      if (url === "/api/training/gpus") return jsonResponse({ gpus: [{ index: 0, name: "Test GPU", memory_total_gb: 24 }] });
      if (url === "/api/config/model-families")
        return jsonResponse({
          items: [
            {
              name: "krea2_raw",
              supported_attention_modes: ["flash", "torch", "sdpa"],
            },
            { name: "anima" },
            { name: "z_image" },
          ],
        });
      if (url === "/api/settings/model-configs")
        return jsonResponse({
          items: [
            {
              id: "anima-main",
              name: "Anima 主模型",
              model_family: "anima",
              pretrained_model_name_or_path: "models/anima/dit",
              qwen3: "models/anima/qwen3",
              vae: "models/anima/vae",
            },
          ],
          groups: [],
          revision: "test",
          default_id: "",
        });
      if (url.startsWith("/api/config/steps?"))
        return jsonResponse({
          total_steps: 1600,
          train_image_count: 100,
          effective_batch_size: 1,
          steps_per_epoch: 100,
        });
      if (url === "/api/config/file-groups?kind=training")
        return jsonResponse([
          {
            id: "imported",
            label: "导入配置",
            methods_subdir: "imported",
            files: [
              {
                path: "configs/imported/train.toml",
                label: "train.toml",
                filename: "train.toml",
                method: "lora",
                methods_subdir: "imported",
                trainable: true,
                locked: false,
              },
              ...(includeSecondConfig
                ? [
                    {
                      path: "configs/imported/other.toml",
                      label: "other.toml",
                      filename: "other.toml",
                      method: "lora",
                      methods_subdir: "imported",
                      trainable: true,
                      locked: false,
                    },
                  ]
                : []),
            ],
          },
        ]);
      if (url === "/api/presets") return jsonResponse(["default", "low_vram"]);
      if (url.startsWith("/api/config/merged?")) {
        const preset = new URL(url, "http://localhost").searchParams.get(
          "preset",
        );
        return jsonResponse({
          ...merged,
          max_train_steps: preset === "low_vram" ? 800 : merged.max_train_steps,
        });
      }
      if (url.startsWith("/api/config/raw?") && method === "GET")
        return jsonResponse({
          file: "configs/imported/train.toml",
          content,
          meta: {
            path: "configs/imported/train.toml",
            label: "train.toml",
            method: "lora",
            methods_subdir: "imported",
            locked: false,
          },
        });
      if (url === "/api/config/raw/patch-preview" && method === "POST") {
        const body = requestJson(init);
        return jsonResponse(patchResult(body.file, body.values));
      }
      if (url === "/api/config/raw" && method === "PATCH") {
        const body = requestJson(init);
        const result = patchResult(body.file, body.values);
        content = result.content;
        merged = { ...merged, ...body.values };
        return jsonResponse(result);
      }
      if (url === "/api/config/raw/save-as" && method === "POST") {
        const body = requestJson(init);
        return jsonResponse({
          ok: true,
          file: body.file,
          message: "保存成功",
          warnings: [],
        });
      }
      if (url === "/api/training/preflight" && method === "POST")
        return jsonResponse({
          ok: false,
          variant: "lora",
          preset: "default",
          methods_subdir: "imported",
          summary: { errors: 1, warnings: 0, checks: 2 },
          checks: [
            {
              level: "error",
              key: "qwen3",
              message: "Qwen3 文本编码器 不存在",
              path: "models/qwen.safetensors",
            },
            {
              level: "ok",
              key: "source_image_dir",
              message: "源图像目录 存在",
              path: "image_dataset",
            },
          ],
          errors: [
            {
              level: "error",
              key: "qwen3",
              message: "Qwen3 文本编码器 不存在",
            },
          ],
          warnings: [],
        });
      return new Response(
        JSON.stringify({ ok: false, error: `unhandled ${method} ${url}` }),
        { status: 500 },
      );
    },
  );
  return fetchMock;
}

function patchResult(file: string, values: Record<string, unknown>) {
  const changed = Object.keys(values);
  const lines = Object.entries(values).map(([key, value]) =>
    typeof value === "string"
      ? `${key} = ${JSON.stringify(value)}`
      : `${key} = ${String(value)}`,
  );
  return {
    ok: true,
    file,
    message: "保存成功",
    content: `${lines.join("\n")}\n`,
    changed,
    warnings: [],
  };
}

function requestJson(init?: RequestInit) {
  return JSON.parse(String(init?.body || "{}"));
}

function jsonResponse(payload: unknown, status = 200) {
  return new Response(JSON.stringify(payload), { status });
}

describe("TrainingWorkspace", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    window.localStorage.clear();
    useHotstartIntent.setState({ intent: null });
  });

  it("clears saved resume even when the hotstart weight already matches, without auto-saving", async () => {
    const baseFetch = createFetchMock();
    const weight = "/output/weight.safetensors";
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith("/api/config/merged?")) {
        const response = await baseFetch(input, init);
        return jsonResponse({ ...await response.json(), network_weights: weight, resume: "/output/old-state" });
      }
      if (url === "/api/training/continue-lora/inspect")
        return jsonResponse({ ok: true, compatible: true, abs_path: weight, kind: "LoRA" });
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({ configFile: "configs/imported/train.toml", preset: "default" });
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const output = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(output).toBeEnabled());
    await user.clear(output);
    await user.type(output, "other-draft-change");
    const offer = () => useHotstartIntent.getState().offer({ configFile: "configs/imported/train.toml", preset: "default", variant: "lora", subdir: "imported", path: weight });
    act(offer);
    await user.click(await screen.findByRole("button", { name: "取消" }));
    expect(output).toHaveValue("other-draft-change");
    act(offer);
    await user.click(await screen.findByRole("button", { name: "应用到草稿" }));
    expect(fetchMock.mock.calls.some(([url, init]) => String(url) === "/api/config/raw" && init?.method === "PATCH")).toBe(false);
    expect(screen.getByText("有未保存修改")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "保存配置" }));
    await waitFor(() => expect(requestBody(fetchMock, "/api/config/raw", "PATCH").values).toEqual({ output_name: "other-draft-change", resume: "" }));
  }, 15_000);

  it("shows source-aware merged values without exposing a start action", async () => {
    vi.stubGlobal("fetch", createFetchMock());
    const { client } = renderWorkspace();

    expect(
      screen.getByRole("heading", { name: "训练配置" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByLabelText("数据集配置"),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "训练计划" }));
    expect(await screen.findByDisplayValue("dragon-run")).toBeInTheDocument();
    expect(screen.getByText("2 当前文件")).toBeInTheDocument();
    // Per-field "继承/预设" badges were replaced by the field-tag title; the
    // merged source is asserted in TrainingBlockSwapControl.test.tsx.
    expect(
      screen.getByRole("button", { name: "立即启动" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "加入队列" }),
    ).toBeInTheDocument();
  }, 15_000);

  it("previews and saves only changed fields, then runs structured preflight", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());

    await user.clear(outputName);
    await user.type(outputName, "dragon-edited");
    expect(screen.getByText("有未保存修改")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存并预检" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "预览变更" }));
    expect(await within(screen.getByRole("dialog")).findByText("1 项")).toBeInTheDocument();
    expect(
      screen.getByText("output_name", {
        selector: ".training-patch-preview code",
      }),
    ).toBeInTheDocument();
    expect(
      requestBody(fetchMock, "/api/config/raw/patch-preview", "POST").values,
    ).toEqual({ output_name: "dragon-edited" });

    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "关闭" }),
    );

    await user.click(screen.getByRole("button", { name: "保存配置" }));
    expect(await screen.findByText("保存成功")).toBeInTheDocument();
    expect(requestBody(fetchMock, "/api/config/raw", "PATCH").values).toEqual({
      output_name: "dragon-edited",
    });
    // After a successful save the toolbar label returns from "保存并预检" to "预检".
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "预检" })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: "预检" }));
    expect(await screen.findByText("需要处理")).toBeInTheDocument();
    expect(screen.getByText("Qwen3 文本编码器 不存在")).toBeInTheDocument();
    expect(screen.getByText("源图像目录 存在")).toBeInTheDocument();
  }, 15_000);

  it("saves boolean field edits as JSON booleans through the workspace handler", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();

    const vaeCache = await screen.findByLabelText("使用 VAE 缓存");
    await waitFor(() => expect(vaeCache).toBeEnabled());
    expect(vaeCache).not.toBeChecked();
    await user.click(vaeCache);
    expect(vaeCache).toBeChecked();

    await user.click(screen.getByRole("button", { name: "保存配置" }));
    await waitFor(() => expect(screen.getByText("保存成功")).toBeInTheDocument());

    const values = requestBody(fetchMock, "/api/config/raw", "PATCH").values;
    expect(values).toHaveProperty("use_vae_cache", true);
    expect(typeof values.use_vae_cache).toBe("boolean");
  });

  it("applies a model combination to the draft and saves all four model fields", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();

    const picker = await screen.findByLabelText("快速选择模型组合");
    await waitFor(() => expect(picker).toBeEnabled());
    await waitFor(() =>
      expect(
        picker,
      ).toContainElement(screen.getByRole("option", { name: /Anima 主模型/ })),
    );
    await user.selectOptions(picker, "anima-main");
    expect(await screen.findByText("有未保存修改")).toBeInTheDocument();
    expect(
      screen.getByText("4", { selector: ".training-config-source strong" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "保存配置" }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url) === "/api/config/raw" && init?.method === "PATCH",
        ),
      ).toBe(true),
    );
    expect(requestBody(fetchMock, "/api/config/raw", "PATCH").values).toEqual({
      model_family: "anima",
      pretrained_model_name_or_path: "models/anima/dit",
      qwen3: "models/anima/qwen3",
      vae: "models/anima/vae",
    });
  });

  it("keeps a stale training draft and blocks repeat saves after 409", async () => {
    const baseFetch = createFetchMock();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith("/api/config/raw?") && !init?.method) {
        const response = await baseFetch(input, init);
        return jsonResponse({ ...(await response.json()), revision: "read-revision" });
      }
      if (url === "/api/config/raw" && init?.method === "PATCH")
        return new Response(JSON.stringify({ ok: false, error: "文件已在其他位置修改" }), { status: 409 });
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.clear(outputName);
    await user.type(outputName, "unsaved-draft");
    await user.click(screen.getByRole("button", { name: "保存配置" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("文件已在其他位置修改"));
    expect(outputName).toHaveValue("unsaved-draft");
    expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "重新加载配置" })).toBeInTheDocument();
    expect(requestBody(fetchMock, "/api/config/raw", "PATCH").revision).toBe("read-revision");
    expect(fetchMock.mock.calls.filter(([url, init]) => String(url) === "/api/config/raw" && init?.method === "PATCH")).toHaveLength(1);
  });

  it("opens auxiliary dialogs and persists the library toggle", async () => {
    vi.stubGlobal("fetch", createFetchMock());
    const user = userEvent.setup();
    const view = renderWorkspace();
    await waitFor(() => expect(screen.getByLabelText("数据集配置")).toHaveValue("configs/datasets/alpha.toml"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "训练量估算" }));
    const dialog = screen.getByRole("dialog", { name: "训练量估算" });
    expect(await within(dialog).findByText("训练图片")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "预览与预检" }));
    expect(
      screen.getByRole("dialog", { name: "预览与预检" }),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    const toggle = screen.getByRole("button", { name: "收起配置库" });
    expect(toggle.closest(".training-command-bar")).not.toBeNull();
    expect(within(screen.getByRole("complementary", { name: "训练配置库" }))
      .queryByRole("button", { name: "收起配置库" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "收起配置库" }));
    expect(document.getElementById("training-config-library")).not.toBeVisible();
    expect(localStorage.getItem("dragon-next.training-library-expanded")).toBe(
      "false",
    );
    view.unmount();
    renderWorkspace();
    expect(screen.getByRole("button", { name: "展开配置库" })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    await user.click(screen.getByRole("button", { name: "展开配置库" }));
    expect(screen.getByRole("complementary", { name: "训练配置库" })).toBeVisible();
    expect(localStorage.getItem("dragon-next.training-library-expanded")).toBe(
      "true",
    );
  });

  it("hides training library management by default and persists its switch", async () => {
    vi.stubGlobal("fetch", createFetchMock());
    const user = userEvent.setup();
    const view = renderWorkspace();
    const library = await screen.findByRole("complementary", { name: "训练配置库" });
    await within(library).findByText("train.toml");
    expect(within(library).getByRole("switch", { name: "详细管理" })).not.toBeChecked();
    expect(within(library).getByRole("button", { name: "新建分组" })).toBeInTheDocument();
    expect(within(library).queryByRole("button", { name: "重命名当前分组" })).not.toBeInTheDocument();
    expect(within(library).queryByRole("button", { name: "拖动排序 train.toml" })).not.toBeInTheDocument();
    expect(within(library).queryByRole("combobox", { name: "移动 train.toml 到分组" })).not.toBeInTheDocument();
    expect(library.querySelector(".training-library-item")).toBeInTheDocument();

    await user.click(within(library).getByRole("switch", { name: "详细管理" }));
    expect(localStorage.getItem(TRAINING_DETAILED_MANAGEMENT_KEY)).toBe("true");
    expect(within(library).getByRole("button", { name: "重命名当前分组" })).toBeInTheDocument();
    expect(within(library).getByRole("button", { name: "拖动排序 train.toml" })).toBeInTheDocument();

    view.unmount();
    renderWorkspace();
    const restored = await screen.findByRole("complementary", { name: "训练配置库" });
    await within(restored).findByText("train.toml");
    expect(within(restored).getByRole("switch", { name: "详细管理" })).toBeChecked();
    await user.click(within(restored).getByRole("switch", { name: "详细管理" }));
    expect(localStorage.getItem(TRAINING_DETAILED_MANAGEMENT_KEY)).toBe("false");
    expect(within(restored).queryByRole("button", { name: "拖动排序 train.toml" })).not.toBeInTheDocument();
  });

  it("confirms dirty context switching asynchronously and keeps unload and SPA guards", async () => {
    vi.stubGlobal("fetch", createFetchMock(true));
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    const { router } = renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, "-dirty");

    const unload = new Event("beforeunload", { cancelable: true });
    expect(window.dispatchEvent(unload)).toBe(false);
    const config = screen.getByLabelText("当前训练配置");
    await user.selectOptions(config, "configs/imported/other.toml");
    let dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    expect(config).toHaveValue("configs/imported/train.toml");
    expect(within(dialog).getByText(/切换配置会丢失这些修改/)).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "取消" }));
    expect(config).toHaveValue("configs/imported/train.toml");
    expect(outputName).toHaveValue("dragon-run-dirty");

    const preset = screen.getByLabelText("当前硬件预设");
    await user.selectOptions(preset, "low_vram");
    dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    expect(preset).toHaveValue("default");
    expect(preset).toBeDisabled();
    expect(screen.getByLabelText("当前训练配置")).toBeDisabled();
    expect(confirm).not.toHaveBeenCalled();
    expect(within(dialog).getByRole("button", { name: "取消" })).toHaveFocus();

    await user.click(within(dialog).getByRole("button", { name: "取消" }));
    expect(screen.queryByRole("dialog", { name: "放弃未保存修改？" })).not.toBeInTheDocument();
    expect(preset).toHaveValue("default");
    expect(outputName).toHaveValue("dragon-run-dirty");

    await user.selectOptions(preset, "low_vram");
    dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "放弃未保存修改？" })).not.toBeInTheDocument();
    expect(preset).toHaveValue("default");
    // The trigger is disabled while the switch is pending, so focus cannot always
    // land back on it; the contract that matters here is "cancelled, nothing changed".
    expect(document.body).toContainElement(preset);

    await user.selectOptions(preset, "low_vram");
    dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    await user.click(within(dialog).getByRole("button", { name: "切换并放弃修改" }));
    await waitFor(() => expect(preset).toHaveValue("low_vram"));
    await waitFor(() => expect(outputName).toHaveValue("dragon-run"));
    expect(confirm).not.toHaveBeenCalled();

    await user.type(outputName, "-again");
    await user.selectOptions(config, "configs/imported/other.toml");
    dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    await user.click(within(dialog).getByRole("button", { name: "切换并放弃修改" }));
    await waitFor(() => expect(config).toHaveValue("configs/imported/other.toml"));
    await waitFor(() => expect(outputName).toHaveValue("dragon-run"));
    expect(confirm).not.toHaveBeenCalled();

    await user.type(outputName, "-again");
    await act(async () => {
      await router.navigate("/other");
    });
    expect(confirm).toHaveBeenCalledWith(
      "当前训练配置有未保存修改，离开会丢失这些修改。是否继续？",
    );
    expect(router.state.location.pathname).toBe("/training");
  });

  it("preserves a dirty draft when a pending config target disappears", async () => {
    const baseFetch = createFetchMock(true);
    let includeTarget = true;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (
        String(input) === "/api/config/file-groups?kind=training" &&
        !includeTarget
      ) {
        return Promise.resolve(
          jsonResponse([
            {
              id: "imported",
              label: "导入配置",
              methods_subdir: "imported",
              files: [
                {
                  path: "configs/imported/train.toml",
                  label: "train.toml",
                  filename: "train.toml",
                  method: "lora",
                  methods_subdir: "imported",
                  trainable: true,
                  locked: false,
                },
              ],
            },
          ]),
        );
      }
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const user = userEvent.setup();
    const { client } = renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, "-dirty");
    const config = screen.getByLabelText("当前训练配置");
    await user.selectOptions(config, "configs/imported/other.toml");
    const dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });

    includeTarget = false;
    await act(async () => {
      await client.invalidateQueries({ queryKey: trainingContextKeys.files() });
    });
    await waitFor(() =>
      expect(
        screen.queryByRole("option", { name: "other.toml" }),
      ).not.toBeInTheDocument(),
    );
    expect(dialog).toBeInTheDocument();

    await user.click(
      within(dialog).getByRole("button", { name: "切换并放弃修改" }),
    );
    expect(screen.queryByRole("dialog", { name: "放弃未保存修改？" })).not.toBeInTheDocument();
    expect(config).toHaveValue("configs/imported/train.toml");
    expect(outputName).toHaveValue("dragon-run-dirty");
    expect(screen.getByRole("button", { name: "保存配置" })).toBeEnabled();
  });

  it("keeps a removed dirty source selected, blocks writes, and allows switching to a listed target", async () => {
    const baseFetch = createFetchMock(true);
    let includeSource = true;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (
        String(input) === "/api/config/file-groups?kind=training" &&
        !includeSource
      ) {
        return Promise.resolve(jsonResponse([
          {
            id: "imported",
            label: "导入配置",
            methods_subdir: "imported",
            files: [
              {
                path: "configs/imported/other.toml",
                label: "other.toml",
                filename: "other.toml",
                method: "lora",
                methods_subdir: "imported",
                trainable: true,
                locked: false,
              },
            ],
          },
        ]));
      }
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const user = userEvent.setup();
    const { client } = renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, "-dirty");
    includeSource = false;
    await act(async () => {
      await client.invalidateQueries({ queryKey: trainingContextKeys.files() });
    });

    await waitFor(() =>
      expect(screen.getByLabelText("当前训练配置")).toHaveValue("configs/imported/train.toml"),
    );
    expect(screen.getByRole("option", { name: /train\.toml（已不可用）/ })).toBeInTheDocument();
    expect(outputName).toHaveValue("dragon-run-dirty");
    expect(outputName).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存并启动" })).toBeDisabled();

    const config = screen.getByLabelText("当前训练配置");
    await user.selectOptions(config, "configs/imported/other.toml");
    const dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    expect(config).toHaveValue("configs/imported/train.toml");
    expect(outputName).toHaveValue("dragon-run-dirty");
    await user.click(within(dialog).getByRole("button", { name: "切换并放弃修改" }));
    await waitFor(() => expect(config).toHaveValue("configs/imported/other.toml"));
    await waitFor(() => expect(outputName).toHaveValue("dragon-run"));
  });

  it("disables confirmation during an in-flight list refetch but keeps cancellation available", async () => {
    const baseFetch = createFetchMock(true);
    let releaseList!: (response: Response) => void;
    let holdList = false;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === "/api/config/file-groups?kind=training" && holdList) {
        holdList = false;
        return new Promise<Response>((resolve) => { releaseList = resolve; });
      }
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const user = userEvent.setup();
    const { client } = renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, "-dirty");
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/other.toml");
    const dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    holdList = true;
    let refetch!: Promise<void>;
    act(() => {
      refetch = client.invalidateQueries({ queryKey: trainingContextKeys.files() });
    });

    const confirm = within(dialog).getByRole("button", { name: "切换并放弃修改" });
    await waitFor(() => expect(confirm).toBeDisabled());
    expect(within(dialog).getByRole("button", { name: "取消" })).toBeEnabled();
    await user.click(within(dialog).getByRole("button", { name: "取消" }));
    expect(screen.queryByRole("dialog", { name: "放弃未保存修改？" })).not.toBeInTheDocument();
    expect(outputName).toHaveValue("dragon-run-dirty");
    releaseList(jsonResponse([
      {
        id: "imported",
        label: "导入配置",
        methods_subdir: "imported",
        files: [
          {
            path: "configs/imported/train.toml",
            label: "train.toml",
            filename: "train.toml",
            method: "lora",
            methods_subdir: "imported",
            trainable: true,
            locked: false,
          },
        ],
      },
    ]));
    await act(async () => { await refetch; });
    expect(screen.getByLabelText("当前训练配置")).toHaveValue("configs/imported/train.toml");
    expect(outputName).toHaveValue("dragon-run-dirty");
  });

  it("does not hydrate a clean draft from a target that disappears while its config is loading", async () => {
    const baseFetch = createFetchMock(true);
    let includeTarget = true;
    let releaseMerged!: (response: Response) => void;
    let holdTargetConfig = false;
    const targetMerged = new Promise<Response>((resolve) => { releaseMerged = resolve; });
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/config/file-groups?kind=training" && !includeTarget) {
        return Promise.resolve(jsonResponse([
          {
            id: "imported",
            label: "导入配置",
            methods_subdir: "imported",
            files: [
              {
                path: "configs/imported/train.toml",
                label: "train.toml",
                filename: "train.toml",
                method: "lora",
                methods_subdir: "imported",
                trainable: true,
                locked: false,
              },
            ],
          },
        ]));
      }
      if (url.startsWith("/api/config/merged?") && url.includes("other.toml") && holdTargetConfig) {
        holdTargetConfig = false;
        return targetMerged;
      }
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const user = userEvent.setup();
    const { client } = renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    expect(outputName).toHaveValue("dragon-run");
    holdTargetConfig = true;
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/other.toml");
    await waitFor(() => expect(fetchMock.mock.calls.some(([input]) =>
      String(input).startsWith("/api/config/merged?") && String(input).includes("other.toml"),
    )).toBe(true));

    includeTarget = false;
    await act(async () => {
      await client.invalidateQueries({ queryKey: trainingContextKeys.files() });
    });
    expect(screen.getByLabelText("当前训练配置")).toHaveValue("configs/imported/other.toml");
    expect(outputName).toHaveValue("dragon-run");
    expect(outputName).toBeDisabled();
    releaseMerged(jsonResponse({
      output_name: "other-run",
      model_family: "krea2_raw",
      max_train_steps: 900,
    }));
    await waitFor(() => expect(screen.getByLabelText("当前训练配置")).toHaveValue("configs/imported/other.toml"));
    expect(outputName).toHaveValue("dragon-run");
  });

  it("waits for warm target raw and merged data to refresh before hydrating or saving", async () => {
    const baseFetch = createFetchMock(true);
    let targetVersion = 1;
    let holdTargetRefresh = false;
    let releaseTargetRaw!: (response: Response) => void;
    let releaseTargetMerged!: (response: Response) => void;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method || "GET";
      if (url.startsWith("/api/config/raw?") && new URL(url, "http://localhost").searchParams.get("file") === "configs/imported/other.toml") {
        if (holdTargetRefresh && targetVersion === 2) {
          holdTargetRefresh = false;
          return new Promise<Response>((resolve) => { releaseTargetRaw = resolve; });
        }
        return Promise.resolve(jsonResponse({
          file: "configs/imported/other.toml",
          content: `output_name = "other-v${targetVersion}"\n`,
          revision: `other-rev-v${targetVersion}`,
          meta: {
            path: "configs/imported/other.toml",
            label: "other.toml",
            method: "lora",
            methods_subdir: "imported",
            locked: false,
          },
        }));
      }
      if (url.startsWith("/api/config/merged?") && new URL(url, "http://localhost").searchParams.get("config_file") === "configs/imported/other.toml") {
        if (targetVersion === 2 && !releaseTargetMerged) {
          return new Promise<Response>((resolve) => { releaseTargetMerged = resolve; });
        }
        return Promise.resolve(jsonResponse({
          output_name: `other-v${targetVersion}`,
          model_family: "krea2_raw",
          max_train_steps: targetVersion === 1 ? 800 : 900,
        }));
      }
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const user = userEvent.setup();
    const { client } = renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/other.toml");
    await waitFor(() => expect(outputName).toHaveValue("other-v1"));
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/train.toml");
    await waitFor(() => expect(outputName).toHaveValue("dragon-run"));
    await user.type(outputName, "-dirty");

    targetVersion = 2;
    holdTargetRefresh = true;
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/other.toml");
    const dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    await user.click(within(dialog).getByRole("button", { name: "切换并放弃修改" }));

    await waitFor(() => expect(screen.getByLabelText("当前训练配置")).toHaveValue("configs/imported/other.toml"));
    await waitFor(() => expect(releaseTargetRaw).toBeTypeOf("function"));
    await waitFor(() => expect(releaseTargetMerged).toBeTypeOf("function"));
    expect(outputName).toHaveValue("dragon-run-dirty");
    expect(outputName).toBeDisabled();
    expect(screen.getByLabelText("当前训练配置")).toBeDisabled();
    expect(screen.getByLabelText("当前硬件预设")).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存并启动" })).toBeDisabled();
    releaseTargetRaw(jsonResponse({
      file: "configs/imported/other.toml",
      content: 'output_name = "other-v2"\n',
      revision: "other-rev-v2",
      meta: { path: "configs/imported/other.toml", label: "other.toml", method: "lora", methods_subdir: "imported", locked: false },
    }));
    await waitFor(() => expect(outputName).toBeDisabled());
    releaseTargetMerged(jsonResponse({ output_name: "other-v2", model_family: "krea2_raw", max_train_steps: 900 }));
    await waitFor(() => expect(outputName).toHaveValue("other-v2"));
    await waitFor(() => expect(outputName).toBeEnabled());
    expect(screen.getByLabelText("当前训练配置")).toBeEnabled();
    expect(client.getQueryData(trainingConfigKeys.raw("configs/imported/other.toml"))).toEqual(
      expect.objectContaining({ revision: "other-rev-v2" }),
    );
    expect(client.getQueryData(trainingContextKeys.merged("configs/imported/other.toml", "default"))).toEqual(
      expect.objectContaining({ output_name: "other-v2", max_train_steps: 900 }),
    );

    await user.type(outputName, "-edited");
    await user.click(screen.getByRole("button", { name: "保存配置" }));
    await waitFor(() => expect(requestBody(fetchMock, "/api/config/raw", "PATCH").revision).toBe("other-rev-v2"));
    expect(requestBody(fetchMock, "/api/config/raw", "PATCH").values).toEqual({ output_name: "other-v2-edited" });
  });

  it("blocks writes when the preset catalog becomes empty without trapping config selection", async () => {
    const baseFetch = createFetchMock(true);
    let presetsEmpty = false;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      String(input) === "/api/presets" && presetsEmpty
        ? Promise.resolve(jsonResponse([]))
        : baseFetch(input, init),
    ));
    useTrainingContextStore.setState({ configFile: "configs/imported/train.toml", preset: "default" });
    const { client } = renderWorkspace();
    await screen.findByLabelText("当前训练配置");
    await userEvent.setup().click(screen.getByRole("tab", { name: "训练计划" }));
    await screen.findByRole("button", { name: "立即启动" });
    await waitFor(() => expect(screen.getByLabelText("当前训练配置")).toBeEnabled());

    presetsEmpty = true;
    await act(async () => {
      await client.invalidateQueries({ queryKey: trainingContextKeys.presets() });
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "立即启动" })).toBeDisabled());
    expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled();
    expect(screen.getByLabelText("当前训练配置")).toBeEnabled();
  });

  it("keeps a target uneditable and blocks writes when a warm target refresh fails", async () => {
    const baseFetch = createFetchMock(true);
    let failTargetRefresh = false;
    let sourceVersion = 1;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith("/api/config/raw?") && new URL(url, "http://localhost").searchParams.get("file") === "configs/imported/other.toml" && failTargetRefresh) {
        return Promise.resolve(jsonResponse({ ok: false, error: "目标配置读取失败" }, 500));
      }
      if (url.startsWith("/api/config/raw?") && new URL(url, "http://localhost").searchParams.get("file") === "configs/imported/train.toml") {
        return Promise.resolve(jsonResponse({
          file: "configs/imported/train.toml",
          content: `output_name = "${sourceVersion === 1 ? "dragon-run" : "dragon-run-v2"}"\nmax_train_steps = 1600\n`,
          revision: `train-rev-v${sourceVersion}`,
          meta: { path: "configs/imported/train.toml", label: "train.toml", method: "lora", methods_subdir: "imported", locked: false },
        }));
      }
      if (url.startsWith("/api/config/merged?") && new URL(url, "http://localhost").searchParams.get("config_file") === "configs/imported/train.toml") {
        return Promise.resolve(jsonResponse({
          output_name: sourceVersion === 1 ? "dragon-run" : "dragon-run-v2",
          model_family: "krea2_raw",
          max_train_steps: 1600,
        }));
      }
      if (url.startsWith("/api/config/raw?") && new URL(url, "http://localhost").searchParams.get("file") === "configs/imported/other.toml") {
        return Promise.resolve(jsonResponse({
          file: "configs/imported/other.toml",
          content: 'output_name = "other-v1"\n',
          revision: "other-rev-v1",
          meta: { path: "configs/imported/other.toml", label: "other.toml", method: "lora", methods_subdir: "imported", locked: false },
        }));
      }
      if (url.startsWith("/api/config/merged?") && new URL(url, "http://localhost").searchParams.get("config_file") === "configs/imported/other.toml") {
        return Promise.resolve(jsonResponse({ output_name: "other-v1", model_family: "krea2_raw", max_train_steps: 800 }));
      }
      return baseFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({ configFile: "configs/imported/train.toml", preset: "default" });
    const user = userEvent.setup();
    renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/other.toml");
    await waitFor(() => expect(outputName).toHaveValue("other-v1"));
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/train.toml");
    await waitFor(() => expect(outputName).toHaveValue("dragon-run"));
    await user.type(outputName, "-dirty");

    failTargetRefresh = true;
    sourceVersion = 2;
    await user.selectOptions(screen.getByLabelText("当前训练配置"), "configs/imported/other.toml");
    const dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    await user.click(within(dialog).getByRole("button", { name: "切换并放弃修改" }));

    expect(await screen.findByText("目标配置读取失败")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重试读取" })).toBeEnabled();
    const config = screen.getByLabelText("当前训练配置");
    expect(config).toBeEnabled();
    await user.selectOptions(config, "configs/imported/train.toml");
    const recoveryDialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    await user.click(within(recoveryDialog).getByRole("button", { name: "切换并放弃修改" }));
    await waitFor(() => expect(config).toHaveValue("configs/imported/train.toml"));
    await waitFor(() => expect(screen.getByLabelText("输出名称")).toHaveValue("dragon-run-v2"));
    expect(screen.getByLabelText("输出名称")).toBeEnabled();
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "PATCH")).toBe(false);
    expect(fetchMock.mock.calls.some(([input, init]) => String(input).startsWith("/api/training/") && init?.method === "POST")).toBe(false);
  });

  it("preserves a dirty draft when a pending preset target disappears", async () => {
    const baseFetch = createFetchMock();
    let includeTarget = true;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      String(input) === "/api/presets" && !includeTarget
        ? Promise.resolve(jsonResponse(["default"]))
        : baseFetch(input, init),
    );
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const user = userEvent.setup();
    const { client } = renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, "-dirty");
    const preset = screen.getByLabelText("当前硬件预设");
    await user.selectOptions(preset, "low_vram");
    const dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });

    includeTarget = false;
    await act(async () => {
      await client.invalidateQueries({ queryKey: trainingContextKeys.presets() });
    });
    await waitFor(() =>
      expect(
        screen.queryByRole("option", { name: "low_vram" }),
      ).not.toBeInTheDocument(),
    );
    expect(dialog).toBeInTheDocument();

    await user.click(
      within(dialog).getByRole("button", { name: "切换并放弃修改" }),
    );
    expect(screen.queryByRole("dialog", { name: "放弃未保存修改？" })).not.toBeInTheDocument();
    expect(preset).toHaveValue("default");
    expect(outputName).toHaveValue("dragon-run-dirty");
    expect(screen.getByRole("button", { name: "保存配置" })).toBeEnabled();
  });

  it("keeps a removed dirty source preset and blocks writes until switching to a listed preset", async () => {
    const baseFetch = createFetchMock();
    let includeSourcePreset = true;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      String(input) === "/api/presets" && !includeSourcePreset
        ? Promise.resolve(jsonResponse(["low_vram"]))
        : baseFetch(input, init),
    );
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const user = userEvent.setup();
    const { client } = renderWorkspace();

    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.type(outputName, "-dirty");
    await waitFor(() => expect(client.isFetching({ queryKey: trainingContextKeys.presets() })).toBe(0));
    includeSourcePreset = false;
    await act(async () => {
      client.setQueryData(trainingContextKeys.presets(), ["low_vram"]);
    });
    expect(client.getQueryData(trainingContextKeys.presets())).toEqual(["low_vram"]);

    const preset = screen.getByLabelText("当前硬件预设");
    await waitFor(() => expect(screen.getByRole("option", { name: /default.*已不可用/ })).toBeInTheDocument());
    expect(preset).toHaveValue("default");
    expect(outputName).toHaveValue("dragon-run-dirty");
    expect(outputName).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled();

    await user.selectOptions(preset, "low_vram");
    const dialog = screen.getByRole("dialog", { name: "放弃未保存修改？" });
    await user.click(within(dialog).getByRole("button", { name: "切换并放弃修改" }));
    await waitFor(() => expect(preset).toHaveValue("low_vram"));
    await waitFor(() => expect(outputName).toBeEnabled());
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
  });

  it("restores known page defaults to the draft only after confirmation, then saves explicitly", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    useTrainingContextStore.setState({
      configFile: "configs/imported/train.toml",
      preset: "default",
    });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const steps = await screen.findByLabelText("最大训练步数");
    await waitFor(() => expect(steps).toBeEnabled());

    // "恢复默认值" now lives inside the 保存菜单 popover, not on the toolbar.
    await user.click(screen.getByRole("button", { name: "保存菜单" }));
    const restore = await screen.findByRole("menuitem", { name: /恢复默认值/ });
    await user.click(restore);
    expect(confirm).toHaveBeenCalledWith(
      "恢复当前可编辑字段的页面默认值？这只会修改未保存草稿，恢复后仍需保存才生效。",
    );
    // Cancelled confirm leaves the loaded value (1600) untouched; only the
    // accepted call below applies the page default (0 = derive from epochs).
    expect(steps).toHaveValue(1600);
    expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled();

    confirm.mockReturnValue(true);
    // The menu closes after the first click, so reopen it before reusing the item.
    await user.click(screen.getByRole("button", { name: "保存菜单" }));
    await user.click(await screen.findByRole("menuitem", { name: /恢复默认值/ }));
    await waitFor(() => expect(steps).toHaveValue(0));
    expect(screen.getByRole("button", { name: "保存配置" })).toBeEnabled();
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "PATCH")).toBe(false);

    await user.click(screen.getByRole("button", { name: "保存配置" }));
    await waitFor(() =>
      expect(requestBody(fetchMock, "/api/config/raw", "PATCH").values).toEqual(
        expect.objectContaining({ max_train_steps: 0 }),
      ),
    );
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "保存配置" })).toBeDisabled(),
    );
  });

  it("keeps restore unavailable until the selected config is hydrated", async () => {
    // Pin the context explicitly: this case depends on a selected config file and
    // otherwise inherits whatever the previous test left in the module store.
    useTrainingContextStore.setState({ configFile: "configs/imported/train.toml", preset: "default" });
    const base = createFetchMock();
    let releaseMerged!: () => void;
    const gate = new Promise<void>((resolve) => { releaseMerged = resolve; });
    // A Response body can only be consumed once, and the merged query may be
    // re-requested (remount/retry), so hand out a fresh Response per call.
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).startsWith("/api/config/merged?")) {
        await gate;
        return jsonResponse({ model_family: "krea2_raw", max_train_steps: 1600 });
      }
      return base(input, init);
    }));
    const user = userEvent.setup();
    renderWorkspace();

    await user.click(screen.getByRole("button", { name: "保存菜单" }));
    const restore = await screen.findByRole("menuitem", { name: /恢复默认值/ });
    expect(restore).toBeDisabled();
    await act(async () => {
      releaseMerged();
      await gate;
    });
    // Restore stays disabled while raw/merged queries are pending and unlocks
    // once both settle; re-query the item because it is re-created per render.
    // Restore unlocks once raw + merged settle. Assert on the DOM attribute
    // because the item is re-created per render and toBeDisabled also accounts
    // for ancestor fieldset/aria state this popover never sets.
    // Both toolbars can expose menuitems, so scope the query to the save menu group.
    // The menu panel is portalled to <body>, so it is no longer a child of the
    // command group; scope by the menu's own accessible name instead.
    await waitFor(() => {
      const saveMenu = screen.getByRole("menu", { name: "保存菜单" });
      expect(
        within(saveMenu).getByRole("menuitem", { name: /恢复默认值/ }).hasAttribute("disabled"),
      ).toBe(false);
    });
  }, 20_000);

  it("blocks execution after a failed save and retains the draft", async () => {
    const base = createFetchMock();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      String(input) === "/api/config/raw" && init?.method === "PATCH"
        ? Promise.resolve(jsonResponse({ ok: false, error: "磁盘只读" }, 409))
        : base(input, init),
    );
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const output = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(output).toBeEnabled());
    await user.type(output, "-dirty");
    await user.click(screen.getByRole("button", { name: "保存并启动" }));
    expect(await screen.findByText("磁盘只读")).toBeInTheDocument();
    expect(output).toHaveValue("dragon-run-dirty");
    expect(
      fetchMock.mock.calls.some(([input, init]) =>
        String(input).startsWith("/api/training/") && init?.method === "POST",
      ),
    ).toBe(false);
  });

  it("renders one category and preserves drafts through keyboard and filtered switches", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await waitFor(() => expect(screen.getByLabelText("数据集配置")).toHaveValue("configs/datasets/alpha.toml"));
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    expect(screen.queryByLabelText("输出名称")).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    await user.type(screen.getByLabelText("输出名称"), "-draft");
    await user.click(screen.getByRole("tab", { name: "输入准备" }));
    expect(screen.queryByLabelText("输出名称")).not.toBeInTheDocument();
    // Arrow keys are handled on the tab buttons; focus the active tab first so the
    // keydown reaches the tablist instead of the clicked panel.
    screen.getByRole("tab", { name: "输入准备" }).focus();
    await user.keyboard("{ArrowRight}");
    await waitFor(() => expect(screen.getByRole("tab", { name: "方法配置" })).toHaveFocus());
    expect(screen.getByLabelText("LoRA rank")).toBeInTheDocument();
    await user.keyboard("{End}");
    expect(screen.getByRole("tabpanel", { name: "设备与性能" })).toBeInTheDocument();
    await user.type(screen.getByLabelText("搜索参数"), "output_name");
    expect(within(screen.getByRole("tabpanel")).getByRole("status")).toHaveTextContent("当前分类没有符合筛选条件");
    expect(screen.getByRole("tab", { name: /训练计划/ })).toHaveTextContent("(1)");
    await user.click(screen.getByRole("tab", { name: /训练计划/ }));
    expect(screen.getByLabelText("输出名称")).toHaveValue("dragon-run-draft");
    expect(screen.getAllByRole("tabpanel")).toHaveLength(1);
    await user.clear(screen.getByLabelText("搜索参数"));
    await user.selectOptions(screen.getByLabelText("参数视图"), "changed");
    expect(screen.getByLabelText("输出名称")).toHaveValue("dragon-run-draft");
    expect(screen.queryByLabelText("最大训练步数")).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([, init]) => init?.method && init.method !== "GET")).toBe(false);
  });

  it("saves a patched copy into imported configs without overwriting", async () => {
    const fetchMock = createFetchMock();
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    renderWorkspace();
    await user.click(screen.getByRole("tab", { name: "训练计划" }));
    const outputName = await screen.findByLabelText("输出名称");
    await waitFor(() => expect(outputName).toBeEnabled());
    await user.clear(outputName);
    await user.type(outputName, "copy-value");
    await user.click(screen.getByRole("button", { name: "保存菜单" }));
    await user.click(await screen.findByRole("menuitem", { name: /另存配置/ }));
    const dialog = screen.getByRole("dialog", { name: "另存训练配置" });
    await user.clear(within(dialog).getByRole("textbox", { name: "配置名称" }));
    await user.type(
      within(dialog).getByRole("textbox", { name: "配置名称" }),
      "dragon copy",
    );
    await user.click(within(dialog).getByRole("button", { name: "确认另存" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/config/raw/save-as",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    const body = requestBody(fetchMock, "/api/config/raw/save-as", "POST");
    expect(body.file).toBe("configs/imported/dragon_copy.toml");
    expect(body.content).toContain('output_name = "copy-value"');
  });
});

function requestBody(
  fetchMock: ReturnType<typeof vi.fn>,
  url: string,
  method: string,
) {
  const call = fetchMock.mock.calls.find(
    ([input, init]) =>
      String(input) === url && (init?.method || "GET") === method,
  );
  if (!call) throw new Error(`missing ${method} ${url}`);
  return requestJson(call[1]);
}
