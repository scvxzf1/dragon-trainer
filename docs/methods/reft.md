# ReFT: Representation Fine-Tuning

状态：稳定
适用版本：当前 main

Block-level residual-stream intervention on the DiT, following LoReFT (Wu et al., NeurIPS 2024). A small learned low-rank edit is added to the output hidden state of selected DiT blocks, instead of modifying any weights inside them. Composes additively with LoRA/T-LoRA/HydraLoRA in the same training run and the same `.safetensors` file.

> **For the structural walkthrough** (math, forward pass, paper-form derivation, why activation-space, composition with the LoRA family, timestep masking on the ReFT bottleneck, why ReFT can't be merged into DiT weights), see **`docs/structure/reft.md`**. This doc is the usage / ops / tuning reference.

## Block selection: `reft_layers`

Not every block needs an edit. `reft_layers` in the method config picks which blocks get wrapped:

| Spec | Meaning |
|------|---------|
| `"all"` *(default)* | every block |
| `"first_N"` | first N blocks |
| `"last_N"` | last N blocks (task-signal heavy) |
| `"stride_K"` | every Kᵗʰ block starting from 0 |
| `"3,7,11,15"` | explicit list of block indices |

For the DiT used here (28 blocks), `"last_8"` is the recommended setting in `configs/gui-methods/reft.toml` and `configs/gui-methods/tlora_ortho_reft.toml` (and in the commented `# reft` toggle block of `configs/methods/lora.toml`). Late blocks are where the strongest task-specific routing already happens in the base model, so small edits there tend to move the final image more than edits in the first few (high-noise, coarse-composition) blocks.

## Tuning guidance

- **Style transferring but anatomy still drifting** → try `reft_layers = "all"` and a larger `reft_dim`.
- **Training already strong, just want a light anatomy rescue** → `"last_4"` or `"last_8"` with `reft_dim = 16–32` is cheap.
- **Per-block parameter cost** is `2 · reft_dim · embed_dim` plus a bias — small compared to a full-block LoRA bank.
- **`reft_dim` is fully decoupled** from `network_dim`; no reason they have to match. Start in the 32–64 range.

## File format

ReFT weights live alongside LoRA weights in the same output `.safetensors`. One sub-state-dict per selected block:

```
reft_unet_blocks_<idx>.rotate_layer.weight      # (reft_dim, embed_dim)
reft_unet_blocks_<idx>.learned_source.weight    # (reft_dim, embed_dim)
reft_unet_blocks_<idx>.learned_source.bias      # (reft_dim,)
reft_unet_blocks_<idx>.alpha                    # scalar
```

Loader behavior on these keys is strict: any `reft_*` key must match `reft_unet_blocks_<idx>.*`. Older per-Linear ReFT wiring is refused at load time with an explicit error pointing at retraining (`networks/lora_anima/`).

Dim inference on load: `reft_dim` is read from `rotate_layer.weight.shape[0]`, block indices from the key prefix, and `reft_layers` is rebuilt from the set of present indices. Nothing else is needed from the original training config.

Each block restores its own `alpha` and the derived `alpha / reft_dim` scale during state loading. The alpha buffer is floating-point even when configured with an integer, so fractional checkpoint values are not truncated. This applies to both adapter-file loading and native training-state resume.

## NF4, checkpointing and compile

ReFT edits activations; it never reads or overwrites packed NF4 base weights. Rank must satisfy `1 <= reft_dim <= embed_dim`. The embedding width comes from the block's `x_dim` or `features`, with `unet.config.features` as a fallback.

Krea-2 blocks opt into the inner `_forward` target so that the ReFT intervention is included in checkpoint recomputation and resident block compilation. The public block object and block-swap hooks remain intact. Other blocks retain their existing public `forward` target. Install and load adapters before compiling; a late Krea ReFT installation is explicitly rejected. Repeated application of the same module is idempotent, while installing a second ReFT on the same target is rejected.

Krea-2 remains **experimental** for this variant: its family registry permits network variant selection, but selection is not a validation guarantee. The bounded NF4/Flash/checkpoint/swap/compile checks and their limits are recorded in [the repair report](../findings/krea2_nf4_adapter_repair_20260905.md); they do not establish 300-step quality or general Krea inference support.

## Inference

### CLI (`inference.py`, `python tasks.py test`)

For supported model families, no special flag is needed (`python tasks.py test`, compatibility alias `make test`). `reft_*` keys are detected in the adapter file and the matching `ReFTModule`s are constructed, installed onto the DiT blocks, and their trained weights loaded. Use `network.set_multiplier(value)` and `network.set_enabled(bool)` to control both LoRA and ReFT branches together. Zero multiplier or a disabled module returns the unedited block output exactly. ReFT cannot be merged into base Linear weights.

### ComfyUI — requires the custom node

**Vanilla ComfyUI cannot load ReFT.** The built-in LoRA patcher rewrites `Linear.weight` in place, which is fundamentally the wrong operation here — ReFT is an *activation-space* intervention, not a weight patch. The standard loader silently ignores `reft_*` keys.

Use the **Anima Adapter Loader** node (`custom_nodes/comfyui-hydralora/`), which installs per-block forward hooks for ReFT alongside LoRA / HydraLoRA / postfix. Separate `strength_lora` and `strength_reft` sliders allow independent ablation. See `custom_nodes/comfyui-hydralora/README.md` for the node's installation paths and changelog.

## Configuration

In `configs/methods/lora.toml`, `add_reft` is **commented out (off) by default** (the active default stack there is OrthoHydra + T-LoRA with `network_dim = 32`, `min_rank = 16`). To train with ReFT enabled out-of-the-box, use the self-contained GUI variants:

```bash
python tasks.py lora-gui reft               # configs/gui-methods/reft.toml (reft_dim=64, last_8)
python tasks.py lora-gui tlora_ortho_reft   # configs/gui-methods/tlora_ortho_reft.toml (reft_dim=32, last_8)
# Compatibility aliases: make lora-gui GUI_PRESETS=reft / make lora-gui GUI_PRESETS=tlora_ortho_reft
```

Or uncomment the `# reft` block in `configs/methods/lora.toml` (`python tasks.py lora`):

```toml
add_reft      = true
reft_dim      = 64
reft_alpha    = 64
reft_layers   = "last_8"
```

`reft_alpha` defaults to `network_alpha` if omitted. The scale is `reft_alpha / reft_dim` = 1 at the default.

## Implementation

| File | Role |
|------|------|
| `networks/lora_modules/reft.py` | `ReFTModule` — intervention forward, orthogonality reg |
| `networks/lora_anima/` | Block selection (`_parse_reft_layers`), wrapping, loader |
| `networks/lora_anima/network.py` | `set_reft_timestep_mask()` — timestep masking on ReFT bottleneck |
| `library/anima/training.py` | CLI arg plumbing (`add_reft`, `reft_dim`, `reft_alpha`, `reft_layers`) |
| `library/training/forward/router_conditioning.py` | Calls `set_reft_timestep_mask()` / `clear_timestep_mask()` each step |
| `custom_nodes/comfyui-hydralora/adapter.py` | ComfyUI loader with per-block forward-hook install |

## Reference

Wu et al., *ReFT: Representation Finetuning for Language Models*, NeurIPS 2024.
