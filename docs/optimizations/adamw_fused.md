# AdamW8bit → fused AdamW

状态：稳定

This doc records why `optimizer_type = "AdamW8bit"` was replaced with `"AdamW"` (running with `fused=True` on CUDA) as the project default, and when it still makes sense to choose the bitsandbytes-backed 8-bit optimizer.

## What was there

`configs/base.toml` defaulted to `optimizer_type = "AdamW8bit"`, which routes through `library/training/optimizers.py` to `bitsandbytes.optim.AdamW8bit`. State (m, v) was stored in 8-bit blockwise quantized form; on every step bnb dequantizes a block, runs the AdamW math, and re-quantizes. State VRAM cost ≈ 2 bytes/param.

## Why it was removed

A `--profile_steps` chrome trace under plain LoRA training showed a clean 11–12 percentage-point GPU-utilization drop coinciding exactly with the optimizer phase of every step. The one-off `scripts/analyze_profile_gaps.py` analyzer used for that historical trace was retired in commit `1eba1d04`; the measurements needed to justify the optimizer choice are preserved below.

| Phase | Duration | GPU util |
|---|---|---|
| forward + backward + everything pre-optimizer | ~860 ms | **99.0%** |
| `Optimizer.step#AdamW8bit.step` | ~308 ms | **88.0%** |
| **Total step** | **~1166 ms** | — |

Inside the 308 ms optimizer phase, the trace shows ~50 tiny `kOptimizerStatic8bit2StateBlockwise` kernels back-to-back, each ~8–30 µs, with kernel-to-kernel gaps of ~6 ms. bnb dispatches one launch per parameter block. The CPU can't refill the launch queue fast enough between micro-kernels, so the GPU sits at ~88 % across the entire optimizer phase. This was the source of the "GPU goes 100 % → 80 % every ~8 steps" symptom — actually every step, but step time is ~1.17 s and aliases against `nvidia-smi`'s 1 Hz refresh.

The 308 ms optimizer plateau was the dominant idle source — `loss.item()` syncs and `accelerator.unwrap_model` overhead at the step boundary added another ~1–3 ms per step but were rounding error compared to the optimizer phase.

## What we switched to

```toml
# configs/base.toml
optimizer_type = "AdamW"
```

In `library/training/optimizers.py`, the `"AdamW"` branch automatically injects `fused=True` whenever `torch.cuda.is_available()` is true and `"fused"` was not explicitly set in `optimizer_args`. `torch.optim.AdamW(..., fused=True)` runs the entire optimizer step as a single fused CUDA kernel — no per-block dispatch, no quantize/dequantize round-trip. The 308 ms plateau collapses to a few milliseconds.

If you ever want to override `fused` via `optimizer_args`, note that `get_optimizer_kwargs` parses values with `ast.literal_eval`, which only accepts Python literals — booleans must be capitalized (`["fused=False"]`, not `["fused=false"]`).

## Trade-off

- **VRAM**: state grows from ~2 bytes/param (bnb 8-bit) to 8 bytes/param (fp32 m + v). For LoRA-only training with `network_train_unet_only=true` and dim=32 (~30 M trainable params), that's about **+180 MB** — fits comfortably on a 16 GB card. For a full-DiT train it would be a few GB extra and may not fit. None of the shipped configs hit that case.
- **Speed**: step time drops from ~1.17 s to ~0.86 s on the same RTX 5060 Ti / FA2 / static-token-pad setup that produced the numbers. GPU utilization sits flat at ~99 %.
- **Numerics**: full-precision optimizer state. No quantization noise on the second moment, so anything sensitive to that (very small LR, long training runs accumulating error) will be slightly more stable. Not a measurable quality difference at our typical scales.

## bitsandbytes is optional, not the default

`bitsandbytes` is kept in `[project.dependencies]` for Linux/Windows so the WebUI's `AdamW8bit` option works after a normal `uv sync`. It is not used unless you explicitly select a bnb-backed optimizer. The bnb optimizer branches in `library/training/optimizers.py` (`AdamW8bit`, `Lion8bit`, `SGDNesterov8bit`, `PagedAdamW`, `PagedAdamW8bit`, `PagedAdamW32bit`, `PagedLion8bit`) still lazy-import `bitsandbytes` inside the selected branch, so normal AdamW/CAME/Lion runs do not pay the import cost.

Important dependency constraint: keep the PyTorch CUDA track on CUDA 13.0 (`cu130` / `cuda-toolkit 13.0.2`). bitsandbytes ships CUDA 13.0 binaries, but not CUDA 13.2 binaries, so resolving PyTorch/FA2 against `cu132` makes `AdamW8bit` unusable.

## If you want to use AdamW8bit

1. In your method / override TOML or the WebUI config, set:
   ```toml
   optimizer_type = "AdamW8bit"
   ```
   (`configs/base.toml` does not set `optimizer_args = ["fused=True"]`, so no manual `optimizer_args` cleanup is needed when switching to `AdamW8bit`.)
2. Run `uv sync` if you are on an older environment that was created before bitsandbytes was added back.

Expect the GPU-utilization dip to come back. If 8-bit state is genuinely needed for VRAM (full-DiT train on 16 GB), the dip is worth the trade. For LoRA, fused AdamW is strictly better.
