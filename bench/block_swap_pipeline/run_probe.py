"""Block-swap macro-pipeline probe.

Runs a real Anima training loop with the offloader profiler enabled and turns the
block_swap_profile.jsonl stream into per-step pipeline metrics:

  * how many retire+restore jobs run per step and per phase,
  * how much DMA time they occupy (h2d + d2h),
  * how much of that is *exposed* as host/GPU wait (uncovered),
  * how long the copy worker queues each job (submit_to_enqueue_ms), which is the
    direct evidence for the single-worker serialisation in Offloader,
  * the two U-turn windows: from the last forward job to the first backward job
    and from the last backward job to the first forward job of the next step.

Nothing here changes training behaviour; the probe only measures.

Usage:
    .venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 \
        --blocks-to-swap 12 --label swap12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from bench.gpu_telemetry import GpuTelemetry
from bench.plain_lora_speed.run_matrix import _summarize_progress

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = str(REPO_ROOT / ".venv" / "bin" / "python")
IMAGE_SIZE = (512, 512)
DEFAULT_ROOT = "output/bench/block_swap_pipeline"

# Per-family sources plus the two cache conventions that differ per family:
# the sidecar suffixes (see library/models/family_registry.py) and the text-cache
# entry point. Only Anima lives in scripts/preprocess/cache_text_embeddings.py;
# the other families each own a script, and they disagree on the encoder flag
# (--qwen3 for anima/krea2_raw, --qwen for z_image/qwen_image_2_1).
FAMILY_PRESETS = {
    "anima": {
        "dit": "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-base-v1.0-sha.safetensors",
        "te": "/home/scv/nvme0n1p1/ComfyUI/models/text_encoders/anima/anima_qwen_3_06b_base.safetensors",
        "vae": "models/vae/qwen_image_vae.safetensors",
        "latent_suffix": "_anima.npz",
        "te_suffix": "_anima_te.safetensors",
        "cached_llm_adapter": True,
        "te_module": "scripts.preprocess.cache_text_embeddings",
        "te_encoder_flag": "--qwen3",
        "runtime_extra": "",
    },
    "krea2_raw": {
        "dit": "models/diffusion_models/krea2_raw_bf16.safetensors",
        "te": "models/text_encoders/qwen3vl_4b_bf16.safetensors",
        "vae": "models/vae/qwen_image_vae.safetensors",
        "latent_suffix": "_anima.npz",
        "te_suffix": "_krea2_te.safetensors",
        "cached_llm_adapter": False,
        "te_module": "scripts.krea2.preprocess_te_cache",
        "te_encoder_flag": "--qwen3",
        "runtime_extra": "",
    },
    "z_image": {
        # Official Diffusers root: transformer/ + text_encoder/ + vae/ + tokenizer/.
        # The single-file ComfyUI components cannot serve the TE pass, because the
        # tokenizer is loaded from this same path (see _te_command).
        "dit": "/home/scv/nvme0n1p1/Z-Image",
        "te": "/home/scv/nvme0n1p1/Z-Image",
        "vae": "/home/scv/nvme0n1p1/Z-Image",
        "latent_suffix": "_z_image.npz",
        "te_suffix": "_z_image_te.safetensors",
        "cached_llm_adapter": False,
        "te_module": "scripts.z_image.preprocess_te_cache",
        "te_encoder_flag": "--qwen3",
        "runtime_extra": 'timestep_sampling = "uniform"\ndiscrete_flow_shift = 6.0',
    },
    "qwen_image_2_1": {
        # Verified against the loader contract in library/models/qwen_image_2_1/
        # weights.py: exactly 32 transformer blocks with packed img_mlp.gate_up
        # tensors. The 60-block ComfyUI 2511/Edit checkpoints on this box are a
        # different architecture and _validate_header rejects them.
        "dit": "/home/scv/nvme0n1p1/ComfyUI/local_models/diffusion_models/Qwen-Image-2.1/qwen_image_2.1_bf16.safetensors",
        "te": "/home/scv/nvme0n1p1/ComfyUI/local_models/text_encoders/Qwen-Image-2.1/qwen3vl_8b_bf16.safetensors",
        "vae": "/home/scv/nvme0n1p1/ComfyUI/local_models/vae/Qwen-Image-2.1/qwen_image_2.1_vae_bf16.safetensors",
        "latent_suffix": "_qwen_image_2_1.npz",
        "te_suffix": "_qwen_image_2_1_te.safetensors",
        "cached_llm_adapter": False,
        "te_module": "scripts.qwen_image_2_1.preprocess_te_cache",
        "te_encoder_flag": "--qwen3",
        # library/models/qwen_image_2_1/strategy.py::resolve_tokenizer_path walks up
        # from the TE file looking for a sibling tokenizer/ dir holding
        # tokenizer_config.json; the single-file ComfyUI TE has none, so the probe
        # stages a tokenizer next to the workspace copy of it.
        "tokenizer": "/home/scv/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
        # configs/base.toml ships timestep_sampling="sigmoid"; the family contract
        # (library/training/compat_matrix.py:223) admits uniform only, so pin it here.
        "runtime_extra": 'timestep_sampling = "uniform"',
    },
}

RUNTIME_TOML = """# Generated by bench/block_swap_pipeline/run_probe.py
model_family = "{model_family}"
pretrained_model_name_or_path = "{dit}"
qwen3 = "{te}"
vae = "{vae}"
no_half_vae = true
vae_disable_cache = true
output_dir = "{output_dir}"
output_name = "{output_name}"
logging_dir = "{logging_dir}"

network_module = "networks.lora_anima"
network_dim = 8
network_alpha = 4
network_train_unet_only = true
learning_rate = 1e-4
optimizer_type = "AdamW"
lr_scheduler = "constant"
max_train_steps = {steps}

mixed_precision = "bf16"
base_compute = "bf16"
save_precision = "bf16"
save_model_as = "safetensors"
gradient_checkpointing = true
blocks_to_swap = {blocks_to_swap}
torch_compile = false
attn_mode = "torch"

use_text_cache = true
use_vae_cache = true
cache_llm_adapter_outputs = {cache_llm_adapter}
use_shuffled_caption_variants = false
caption_dropout_rate = 0.0
masked_loss = false
weighting_scheme = "none"
{runtime_extra}

# --- profiler: one JSONL event per completed retire+restore job ---
block_swap_profile_jsonl = "auto"
"""

DATASET_TOML = """# Generated by bench/block_swap_pipeline/run_probe.py
[general]
caption_extension = ".txt"
keep_tokens = 0

[[datasets]]
resolution = {side}
batch_size = 1
enable_bucket = true
min_bucket_reso = 256
max_bucket_reso = 1024
bucket_reso_steps = 64
bucket_no_upscale = true
caption_extension = ".txt"
keep_tokens = 0

[[datasets.subsets]]
image_dir = "{image_dir}"
cache_dir = "{cache_dir}"
num_repeats = 1
"""


def _run(cmd: list[str], *, gpu: int, log: Path) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode


def _write_images(image_dir: Path, side: int = IMAGE_SIZE[0]) -> None:
    from PIL import Image

    size = (side, side)
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        pixels = [
            ((x * 7 + idx * 31) % 256, (y * 5 + idx * 17) % 256, ((x + y) * 3) % 256)
            for y in range(size[1])
            for x in range(size[0])
        ]
        img = Image.new("RGB", size)
        img.putdata(pixels)
        img.save(image_dir / f"img{idx}.png")
        (image_dir / f"img{idx}.txt").write_text(
            f"block swap pipeline probe image {idx}", encoding="utf-8"
        )


def _te_command(
    family: str,
    preset: dict,
    *,
    image_dir: Path,
    cache_dir: Path,
    te: Path,
    dit: str,
    dtype: str,
) -> list[str]:
    """Family-appropriate text-encoder cache command.

    Cache suffixes differ per family, and so do the entry points: only Anima
    caches through scripts/preprocess/cache_text_embeddings.py, and the encoder
    flag is --qwen3 for anima/krea2_raw but --qwen for z_image/qwen_image_2_1.
    """
    cmd = [
        PY, "-m", preset["te_module"],
        "--dir", str(image_dir), "--cache_dir", str(cache_dir),
        "--batch_size", "1", "--dtype", dtype,
        "--caption_shuffle_variants", "0", "--min_pixels", "0",
        preset["te_encoder_flag"], str(te),
    ]
    if family == "anima":
        cmd += ["--dit", str(dit), "--require_cache_identity"]
    return cmd


def _ensure_workspace(
    workspace: Path,
    sources: dict,
    *,
    gpu: int,
    dtype: str,
    side: int = IMAGE_SIZE[0],
    family: str = "anima",
) -> dict:
    """Create (once) the dataset + a reusable cache workspace."""
    image_dir = workspace / "images"
    # library/datasets/dreambooth.py:198 reads each image size from latent cache
    # files found in the *image* directory; with a separate cache_dir those sizes
    # stay unresolved and the trainer aborts with "Latent cache is incomplete".
    # Anima tolerates the split layout (historically used), the other families do
    # not (measured on krea2_raw and z_image), so they share one directory.
    cache_dir = workspace / "cache" if family == "anima" else image_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not (image_dir / "img0.png").exists():
        _write_images(image_dir, side)

    resolved: dict[str, Path] = {}
    for key in ("te", "vae"):
        src = Path(sources[key])
        dst = workspace / f"{key}.safetensors"
        if src.is_dir():
            # Diffusers component root (Z-Image/Qwen style): reference it in place;
            # copying multi-GB directories into the workspace buys nothing. A file
            # copy left behind by an older preset would also shadow the new source
            # in the cache identity, so drop it (this cost one stale-cache cycle).
            resolved[key] = src
            stale = workspace / f"{key}.safetensors"
            if stale.exists():
                stale.unlink()
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
        resolved[key] = dst
    te_copy = resolved["te"]
    vae_copy = resolved["vae"]

    preset = FAMILY_PRESETS[family]
    tokenizer_src = preset.get("tokenizer")
    if tokenizer_src:
        # resolve_tokenizer_path() looks for <te-parent>/tokenizer; stage it beside
        # the workspace TE copy so the family keeps its own tokenizer contract.
        staged = Path(te_copy).parent / "tokenizer"
        if not (staged / "tokenizer_config.json").is_file():
            staged.mkdir(parents=True, exist_ok=True)
            for item in sorted(Path(tokenizer_src).iterdir()):
                if item.is_file():
                    shutil.copy2(item, staged / item.name)
        resolved["tokenizer"] = staged

    dataset_toml = workspace / "dataset.toml"
    dataset_toml.write_text(
        DATASET_TOML.format(image_dir=image_dir, cache_dir=cache_dir, side=side),
        encoding="utf-8",
    )

    have_latents = any(cache_dir.glob("*" + preset["latent_suffix"]))
    have_te = any(cache_dir.glob("*" + preset["te_suffix"]))
    if not have_latents:
        rc = _run(
            [PY, "-m", "scripts.preprocess.cache_latents",
             "--dir", str(image_dir), "--cache_dir", str(cache_dir),
             "--batch_size", "1", "--dtype", dtype, "--require_cache_identity",
             "--vae", str(vae_copy), "--model_family", family],
            gpu=gpu, log=workspace / "pre-latents.log",
        )
        print(json.dumps({"stage": "preprocess_latents", "returncode": rc}), flush=True)
        if rc != 0:
            raise SystemExit("latent preprocessing failed; see " + str(workspace / "pre-latents.log"))
    if not have_te:
        rc = _run(
            _te_command(family, preset, image_dir=image_dir, cache_dir=cache_dir,
                        te=te_copy, dit=sources["dit"], dtype=dtype),
            gpu=gpu, log=workspace / "pre-te.log",
        )
        print(json.dumps({"stage": "preprocess_te", "returncode": rc}), flush=True)
        if rc != 0:
            raise SystemExit("TE preprocessing failed; see " + str(workspace / "pre-te.log"))
    return {
        "dataset_toml": dataset_toml,
        "cache": cache_dir,
        "image_dir": image_dir,
        "te": te_copy,
        "vae": vae_copy,
    }


STEP_TIME_RE = re.compile(r"([0-9.]+)(s/it|it/s)")


def parse_step_time(train_log: Path, *, tail: int = 8) -> dict:
    """Mean step time from the training progress lines (last N samples)."""
    if not train_log.exists():
        return {}
    per_it, per_s = [], []
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        # Only the training progress line ('steps:  10%|..| 1/10 [..,  1.60s/it]')
        # carries the real step time; LoRA-construction lines also print 'it/s'.
        if "steps:" not in line:
            continue
        for value, unit in STEP_TIME_RE.findall(line):
            try:
                number = float(value)
            except ValueError:
                continue
            if unit == "s/it":
                per_it.append(number)
            elif number > 0:
                per_s.append(1.0 / number)
    values = (per_it + per_s)[-tail:]
    if not values:
        return {}
    return {
        "train_step_ms_mean": round(statistics.fmean(values) * 1000.0, 1),
        "train_step_ms_samples": [round(v * 1000.0, 1) for v in values],
    }


def _segment_steps(events: list[dict]) -> list[list[dict]]:
    """Split the event stream into training steps.

    The offloader step field is not usable for grouping (it stays at its first
    value in practice), so segment on the phase cycle instead: a step starts at a
    forward job that follows a backward job.
    """
    steps: list[list[dict]] = []
    current: list[dict] = []
    seen_backward = False
    for ev in events:
        phase = str(ev.get("submit_phase") or "")
        if phase.startswith("forward") and seen_backward and current:
            steps.append(current)
            current, seen_backward = [], False
        if phase.startswith("backward"):
            seen_backward = True
        current.append(ev)
    if current:
        steps.append(current)
    return steps


def analyze_profile(path: Path, *, train_log: Path | None = None) -> dict:
    """Turn the JSONL event stream into pipeline metrics."""
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    events = [e for e in events if e.get("ev") == "block_swap"]
    if not events:
        return {"error": "no block_swap events", "path": str(path)}

    steps = []
    for step_index, evs in enumerate(_segment_steps(events), start=1):
        evs = sorted(evs, key=lambda e: e.get("queued_at") or 0.0)
        fwd = [e for e in evs if str(e.get("submit_phase", "")).startswith("forward")]
        bwd = [e for e in evs if str(e.get("submit_phase", "")).startswith("backward")]
        def total(key, items):
            return sum(float(e.get(key) or 0.0) for e in items)
        started = min((e.get("queued_at") or 0.0) for e in evs)
        finished = max((e.get("wait_returned_at") or 0.0) for e in evs)
        # duplicate restore = same target block restored twice in one step
        targets = [int(e.get("block_idx")) for e in evs]
        dupes = sorted({t for t in targets if targets.count(t) > 1})
        fwd_ready = [float(e.get("ready_at") or 0.0) for e in fwd]
        bwd_queued = [float(e.get("queued_at") or 0.0) for e in bwd]
        bwd_ready = [float(e.get("ready_at") or 0.0) for e in bwd]
        entry = {
            "step": step_index,
            "jobs": len(evs),
            "jobs_forward": len(fwd),
            "jobs_backward": len(bwd),
            "dma_busy_ms": total("d2h_ms", evs) + total("h2d_ms", evs),
            "d2h_ms": total("d2h_ms", evs),
            "h2d_ms": total("h2d_ms", evs),
            "h2d_bytes": int(sum(float(e.get("h2d_bytes") or 0.0) for e in evs)),
            "gb_per_s": round(
                (sum(float(e.get("h2d_bytes") or 0.0) for e in evs) / 1e9)
                / max(1e-9, (total("d2h_ms", evs) + total("h2d_ms", evs)) / 1000.0),
                2,
            ),
            "exposed_wait_ms": total("wait_ms", evs),
            "worker_queue_ms": total("submit_to_enqueue_ms", evs),
            "step_span_ms": (finished - started) * 1000.0,
            "forward_tail_idle_ms": (
                (min(bwd_queued) - max(fwd_ready)) * 1000.0
                if fwd and bwd and min(bwd_queued) > max(fwd_ready)
                else 0.0
            ),
            "duplicate_restores": dupes,
        }
        steps.append(entry)

    def agg(key):
        vals = [s[key] for s in steps]
        return round(statistics.fmean(vals), 3) if vals else 0.0

    result = {
        "path": str(path),
        "events": len(events),
        "steps": steps,
        "mean": {
            "jobs": agg("jobs"),
            "dma_busy_ms": agg("dma_busy_ms"),
            "h2d_bytes": agg("h2d_bytes"),
            "gb_per_s": agg("gb_per_s"),
            "d2h_ms": agg("d2h_ms"),
            "h2d_ms": agg("h2d_ms"),
            "exposed_wait_ms": agg("exposed_wait_ms"),
            "worker_queue_ms": agg("worker_queue_ms"),
            "step_span_ms": agg("step_span_ms"),
            "forward_tail_idle_ms": agg("forward_tail_idle_ms"),
        },
        "phases": sorted({str(e.get("submit_phase")) for e in events}),
        "prefetch_lead_values": sorted({int(e.get("prefetch_lead_blocks") or 0) for e in events}),
        "slot_count": sorted({int(e.get("slot_count") or 0) for e in events}),
    }
    if train_log is not None:
        result["train"] = parse_step_time(train_log)
    return result


def _runtime_toml(
    args, preset: dict, *, dit: str, te: Path, vae: Path, run_dir: Path
) -> str:
    """Render the runtime config for one probe run (family-aware keys included)."""
    text = RUNTIME_TOML.format(
        model_family=args.family,
        cache_llm_adapter=str(bool(preset["cached_llm_adapter"])).lower(),
        runtime_extra=preset["runtime_extra"],
        dit=dit,
        te=te,
        vae=vae,
        output_dir=run_dir / "train_out",
        output_name="block_swap_pipeline",
        logging_dir=run_dir / "logs",
        steps=args.steps,
        blocks_to_swap=args.blocks_to_swap,
    )
    if args.restore_mode:
        text += f'block_swap_restore_mode = "{args.restore_mode}"\n'
    if args.block_swap_auto:
        text += (
            "block_swap_auto = true\n"
            f"block_swap_vram_margin_mib = {args.margin_mib}\n"
            f"block_swap_auto_stop_ratio = {args.stop_ratio}\n"
        )
    if args.memory_probe:
        text += (
            'memory_probe_jsonl = "auto"\n'
            f"memory_probe_max_steps = {args.memory_probe_max_steps}\n"
        )
    return text


def _parse_window(text: str | None) -> tuple[int, int] | None:
    """Parse a ``lo-hi`` step window (e.g. ``6-50``) for steady-state metrics."""
    if not text:
        return None
    raw = text.strip()
    if "-" in raw:
        lo, hi = raw.split("-", 1)
    else:
        lo = hi = raw
    try:
        start, end = int(lo), int(hi)
    except ValueError as exc:
        raise SystemExit(f"bad --metric-window {text!r}; expected lo-hi, e.g. 6-50") from exc
    if start <= 0 or end < start:
        raise SystemExit(f"bad --metric-window {text!r}; expected positive and ordered")
    return start, end


def _first_existing(*paths: Path) -> Path:
    """Return the first path that exists (last one wins as the fallback)."""
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


def _summarize_memory_probe(path: Path) -> dict:
    """Compact VRAM/component view of the trainer's memory probe JSONL."""
    if not path.exists():
        return {"path": str(path), "missing": True}
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def peak(key: str) -> float | None:
        values = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
        return round(max(values), 4) if values else None

    by_label = [
        {
            "label": r.get("label"),
            "phase": r.get("phase"),
            "allocated_gb": r.get("cuda_allocated_gb"),
            "reserved_gb": r.get("cuda_reserved_gb"),
            "max_reserved_gb": r.get("cuda_max_reserved_gb"),
        }
        for r in records
        if r.get("ev") == "memory" and r.get("label")
    ]
    return {
        "path": str(path),
        "records": len(records),
        "peak_max_allocated_gb": peak("cuda_max_allocated_gb"),
        "peak_max_reserved_gb": peak("cuda_max_reserved_gb"),
        "min_free_gb": min(
            (float(r["cuda_free_gb"]) for r in records if isinstance(r.get("cuda_free_gb"), (int, float))),
            default=None,
        ),
        "by_label": by_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--blocks-to-swap", type=int, default=12)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--family", default="anima", choices=sorted(FAMILY_PRESETS),
                        help="model family preset: sources, cache suffixes, runtime keys")
    parser.add_argument("--dit", default=None, help="override the preset DiT path")
    parser.add_argument("--te", default=None, help="override the preset TE path")
    parser.add_argument("--vae", default=None, help="override the preset VAE path")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--side", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--block-swap-auto", action="store_true")
    parser.add_argument("--margin-mib", type=int, default=400)
    parser.add_argument("--stop-ratio", type=float, default=0.0)
    parser.add_argument(
        "--restore-mode",
        default=None,
        help="block_swap_restore_mode (foreach|slab); None keeps the project default",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="sample nvidia-smi (SM/mem clock, VRAM, util, power, PCIe link) for the train stage",
    )
    parser.add_argument(
        "--telemetry-gpu-index",
        type=int,
        default=None,
        help="nvidia-smi index to sample; defaults to --gpu. Only equal to --gpu when "
        "CUDA_DEVICE_ORDER=PCI_BUS_ID (otherwise CUDA and nvidia-smi index orders differ).",
    )
    parser.add_argument("--telemetry-interval-ms", type=int, default=500)
    parser.add_argument(
        "--memory-probe",
        action="store_true",
        help="enable the trainer memory probe JSONL (VRAM breakdown per load/train stage)",
    )
    parser.add_argument("--memory-probe-max-steps", type=int, default=5)
    parser.add_argument(
        "--metric-window",
        default=None,
        help="steady-state step window for the step-time summary, e.g. 6-50 (default: all steps)",
    )
    args = parser.parse_args()

    preset = FAMILY_PRESETS[args.family]
    sources = {
        "dit": args.dit or preset["dit"],
        "te": args.te or preset["te"],
        "vae": args.vae or preset["vae"],
    }
    # Anima keeps the historical workspace name so existing caches stay reusable;
    # every other family gets its own directory (caches must not be shared).
    base_workspace = "workspace" if args.family == "anima" else f"workspace-{args.family}"
    if args.side != IMAGE_SIZE[0]:
        base_workspace = f"{base_workspace}-{args.side}"
    workspace = Path(args.workspace or (Path(args.root) / base_workspace))
    workspace.mkdir(parents=True, exist_ok=True)
    label = args.label or f"swap{args.blocks_to_swap}"
    run_dir = Path(args.root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    (run_dir / "train_out").mkdir(parents=True, exist_ok=True)

    print(f"[probe] workspace={workspace}", flush=True)
    components = _ensure_workspace(
        workspace, sources, gpu=args.gpu, dtype=args.dtype,
        side=args.side, family=args.family,
    )

    runtime = run_dir / "config.runtime.toml"
    dataset_toml = workspace / "dataset.toml"
    runtime.write_text(
        _runtime_toml(
            args, preset, dit=sources["dit"], te=components["te"],
            vae=components["vae"], run_dir=run_dir,
        ),
        encoding="utf-8",
    )
    cmd = [
        PY, "train.py",
        "--config_file", str(runtime),
        "--dataset_config", str(dataset_toml),
        "--require_cache_identity",
    ]
    telemetry = (
        GpuTelemetry(
            args.telemetry_gpu_index if args.telemetry_gpu_index is not None else args.gpu,
            interval_ms=args.telemetry_interval_ms,
            csv_path=run_dir / "gpu_telemetry.csv",
        ).start()
        if args.telemetry
        else None
    )
    rc = _run(cmd, gpu=args.gpu, log=run_dir / "train.log")
    if telemetry is not None:
        telemetry.stop()
    print(json.dumps({"stage": "train", "returncode": rc}), flush=True)

    profile = run_dir / "logs" / "block_swap_pipeline.block_swap_profile.jsonl"
    metrics = (
        analyze_profile(profile, train_log=run_dir / "train.log")
        if profile.exists()
        else {"error": "profile missing", "train": parse_step_time(run_dir / "train.log")}
    )
    progress = _summarize_progress(
        _first_existing(
            run_dir / "logs" / "block_swap_pipeline.progress.jsonl",
            run_dir / "train_out" / "block_swap_pipeline.progress.jsonl",
        ),
        metric_window=_parse_window(args.metric_window),
        images_per_step=1.0,
    )
    memory = _summarize_memory_probe(
        run_dir / "logs" / "block_swap_pipeline.memory_probe.jsonl"
    )
    auto_lines = [
        line for line in (run_dir / "train.log").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if "block_swap_auto" in line
    ]
    result = {
        "label": label,
        "args": vars(args),
        "train_returncode": rc,
        "metrics": metrics,
        "speed": {
            "progress": progress,
            "telemetry": telemetry.summary() if telemetry is not None else None,
            "vram": memory,
        },
        "block_swap_auto": auto_lines[-3:],
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics.get("mean", metrics), indent=2, ensure_ascii=False), flush=True)
    speed_line = {
        "avg_step_sec": progress.get("avg_step_sec"),
        "median_step_sec": progress.get("median_step_sec"),
        "p90_step_sec": progress.get("p90_step_sec"),
        "images_per_hour": progress.get("images_per_hour"),
        "peak_reserved_gb": progress.get("peak_reserved_gb"),
        "vram_peak_max_reserved_gb": memory.get("peak_max_reserved_gb"),
        "h2d_gb_per_s": metrics.get("mean", {}).get("gb_per_s") if isinstance(metrics.get("mean"), dict) else None,
    }
    if telemetry is not None:
        tel = telemetry.summary()
        active = tel.get("active") or {}
        speed_line["sm_clock_mean_mhz"] = (tel.get("clocks.sm") or {}).get("mean")
        speed_line["mem_clock_mean_mhz"] = (tel.get("clocks.mem") or {}).get("mean")
        speed_line["util_gpu_mean_pct"] = tel.get("utilization_gpu_mean_pct")
        speed_line["sm_clock_active_mean_mhz"] = (active.get("clocks.sm") or {}).get("mean")
        speed_line["mem_clock_active_mean_mhz"] = (active.get("clocks.mem") or {}).get("mean")
        speed_line["util_gpu_active_mean_pct"] = (active.get("utilization.gpu") or {}).get("mean")
        speed_line["vram_used_active_peak_mib"] = (active.get("memory.used") or {}).get("max")
        speed_line["power_active_mean_w"] = (active.get("power.draw") or {}).get("mean")
        speed_line["pcie_width_current"] = tel.get("pcie.link.width.current")
        speed_line["pcie_gen_current"] = tel.get("pcie.link.gen.current")
    print(json.dumps(speed_line, ensure_ascii=False), flush=True)
    print(f"[probe] {'PASS' if rc == 0 else 'FAIL'} -> {run_dir}", flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
