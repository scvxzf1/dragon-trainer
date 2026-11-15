import { Cpu, RefreshCw } from "lucide-react";
import type { useTrainingWorkspace } from "./useTrainingWorkspace";
import "./TrainingDevices.css";

type Props = { state: ReturnType<typeof useTrainingWorkspace>; details?: boolean };

export function TrainingDevices({ state, details = false }: Props) {
  const { deviceState: device, busy, draft } = state;
  const pipeline = draft.pipeline_parallel === true || draft.pipeline_parallel === "true";
  const changeMode = (mode: "single" | "ddp") => {
    device.setMode(mode);
    if (pipeline) state.setDraft((current) => ({ ...current, pipeline_parallel: false }));
    state.preflight.reset();
  };
  if (details) return (
    <section className="training-devices training-devices-details" aria-label="设备与并行设置">
      <details>
        <summary>多卡与分布式训练</summary>
        <div className="training-device-detail"><strong>当前设备</strong><span>{pipeline ? "流水线并行（不可用）" : device.summary}</span></div>
        <div className="training-device-heading">
          <strong>并行模式</strong>
          <div className="training-device-modes" role="group" aria-label="训练并行模式">
            <button type="button" aria-pressed={!pipeline && device.selection.mode === "single"} disabled={busy}
              onClick={() => changeMode("single")}>单卡</button>
            <button type="button" aria-pressed={!pipeline && device.selection.mode === "ddp"}
              disabled={busy || device.devices.length < 2} title={device.devices.length < 2 ? "至少需要两张 GPU" : "数据并行"}
              onClick={() => changeMode("ddp")}>多卡 · 数据并行</button>
          </div>
          <button type="button" className="icon-button" aria-label="刷新 GPU 列表" title="刷新 GPU 列表"
            disabled={busy || device.query.isFetching} onClick={() => void device.refreshDevices()}><RefreshCw size={14} /></button>
        </div>
        <div className="training-device-options" role={device.selection.mode === "single" ? "radiogroup" : "group"} aria-label="选择训练 GPU">
          {device.devices.map((gpu) => (
            <label key={gpu.id}>
              <input type={device.selection.mode === "single" ? "radio" : "checkbox"}
                name="gpu-details" aria-label={`GPU ${gpu.id} · ${gpu.name}`}
                checked={device.selection.devices.some((item) => item.id === gpu.id && item.name === gpu.name)} disabled={busy}
                onChange={() => { device.selectDevice(gpu); state.preflight.reset(); }} />
              <span>GPU {gpu.id} · {gpu.name}</span>
              {gpu.memoryGb && <small>总显存 {gpu.memoryGb} GB</small>}
            </label>
          ))}
        </div>
        <div className="training-device-summary" aria-live="polite"><span>当前选择</span><strong>{device.summary}</strong></div>
        {device.issue && <p className="form-error" role={device.query.isPending ? "status" : "alert"}>
          {device.issue}
          {device.issue.startsWith("已选设备不可用") && <button type="button" disabled={busy} onClick={device.resetSelection}>清除失效选择</button>}
        </p>}
        <p className="training-device-detail training-device-unavailable"><span>流水线并行（实验）</span><small>请在下方「分布式训练」卡片选择流水线并行；未接线模型族会被拦截，admission 在训练入口兜底。</small></p>
      </details>
    </section>
  );
  return (
    <section className="training-devices" aria-label="训练设备快捷选择" aria-busy={device.query.isPending || device.query.isFetching}>
      <div className="training-device-heading">
        <strong><Cpu size={16} />训练设备</strong>
        <div className="training-device-modes" role="group" aria-label="训练并行模式">
          <button type="button" aria-pressed={!pipeline && device.selection.mode === "single"} disabled={busy}
            onClick={() => changeMode("single")}>单卡</button>
          <button type="button" aria-pressed={!pipeline && device.selection.mode === "ddp"}
            disabled={busy || device.devices.length < 2} title={device.devices.length < 2 ? "至少需要两张 GPU" : "数据并行"}
            onClick={() => changeMode("ddp")}>多卡 · 数据并行</button>
        </div>
        <button type="button" className="icon-button" aria-label="刷新 GPU 列表" title="刷新 GPU 列表"
          disabled={busy || device.query.isFetching} onClick={() => void device.refreshDevices()}><RefreshCw size={14} /></button>
      </div>
      <div className="training-device-options" role={device.selection.mode === "single" ? "radiogroup" : "group"} aria-label="选择训练 GPU">
        {device.devices.map((gpu) => (
          <label key={gpu.id}>
            <input type={device.selection.mode === "single" ? "radio" : "checkbox"}
              name="gpu-quick" aria-label={`GPU ${gpu.id} · ${gpu.name}`}
              checked={device.selection.devices.some((item) => item.id === gpu.id && item.name === gpu.name)} disabled={busy}
              onChange={() => { device.selectDevice(gpu); state.preflight.reset(); }} />
            <span>GPU {gpu.id} · {gpu.name}</span>
            {gpu.memoryGb && <small>总显存 {gpu.memoryGb} GB</small>}
          </label>
        ))}
      </div>
      {device.issue && <p className="form-error" role={device.query.isPending ? "status" : "alert"}>
        {device.issue}
        {device.issue.startsWith("已选设备不可用") && <button type="button" disabled={busy} onClick={device.resetSelection}>清除失效选择</button>}
      </p>}
      {pipeline && <p className="form-error" role="alert">当前配置启用了尚未接入主训练入口的流水线并行。请选择单卡或数据并行后保存。</p>}
    </section>
  );
}
