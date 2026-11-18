import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { fetchGlobalSettings, settingsKeys } from "../features/settings/api";

const routeScales: Record<string, string> = {
  training: "config",
  datasets: "datasets",
  monitor: "training",
  models: "model_config",
  settings: "settings",
  history: "history_overview",
  "weight-analysis": "weight_analysis",
  "image-test": "image_test",
  environment: "environment",
};

export function validScale(value: unknown, fallback = 100): number {
  if (value === "" || value == null) return fallback;
  const number = Number(value);
  return Number.isFinite(number)
    ? Math.min(400, Math.max(25, number))
    : fallback;
}

export function pageScale(settings: Record<string, unknown>, path: string) {
  const global = validScale(settings.ui_scale);
  const url = new URL(path, "http://localhost");
  const page = url.pathname.split("/").filter(Boolean)[0];
  const historyTabs: Record<string, string> = {
    metrics: "analysis",
    artifacts: "preview",
    logs: "logs",
    config: "config_files",
  };
  const key =
    page === "history"
      ? `history_${historyTabs[url.searchParams.get("view") || ""] || "overview"}`
      : routeScales[page];
  return validScale(settings[`ui_scale_${key || ""}`], global);
}

export function useUIPreferences(path: string) {
  const { data = {} } = useQuery({
    queryKey: settingsKeys.global,
    queryFn: ({ signal }) => fetchGlobalSettings(signal),
    staleTime: 60_000,
  });
  useEffect(() => {
    const root = document.documentElement;
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => {
      root.dataset.nextMotion =
        data.dragon_motion_enabled !== false && !media.matches ? "on" : "off";
    };
    update();
    root.dataset.nextHelp = data.dragon_config_help_always_visible
      ? "always"
      : "contextual";
    root.dataset.nextTags = data.dragon_config_tags_always_visible
      ? "always"
      : "contextual";
    media.addEventListener("change", update);
    return () => {
      media.removeEventListener("change", update);
      delete root.dataset.nextMotion;
      delete root.dataset.nextHelp;
      delete root.dataset.nextTags;
    };
  }, [data]);
  const global = validScale(data.ui_scale);
  return { global: global / 100, content: pageScale(data, path) / global };
}
