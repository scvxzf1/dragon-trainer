# Anima Tagger (ComfyUI)

Multi-label image tagger trained on the Anima caption distribution. Drop in an image, get back a comma-separated tag string in exactly Anima's training-time T5 format — `rating, count, characters, copyrights, @artists, generals`, underscores replaced by spaces.

Two nodes in the `anima` category:

| Node | Inputs | Outputs | Use |
|------|--------|---------|-----|
| **Anima Tagger Loader** | `tagger_dir` (STRING), `version` (`v1`/`v2`), `pe_ckpt` (STRING), `pe_aux_ckpt` (STRING), `use_pe_lora` (BOOLEAN), `pe_lora_path` | `tagger` (ANIMA_TAGGER) | Load the checkpoint once; ComfyUI memoizes the output so the tagger persists across graph runs. |
| **Anima Tagger Caption** | `tagger` (ANIMA_TAGGER), `image` (IMAGE) | `caption` (STRING) | Tag an image. Drop the STRING into any text input (`CLIPTextEncode`, `Anima DirectEdit.source_tag`, etc.). |

## What it's for

- **DirectEdit ψ_src (`source_tag`).** Wire `Anima Tagger Caption`'s `caption` (`STRING`) output into `source_tag` on [comfyui-anima-directedit](../comfyui-anima-directedit/README.md). DirectEdit's edit leverage collapses when ψ_src is structurally far from Anima's training-time embedding manifold — Anima Tagger produces in-distribution Anima captions vs. a generic WD-tagger.
- **Caption pre-fill for LoRA training.** Tag your dataset, paste into `.txt` sidecars.
- **Prompt scaffolding.** Wire the caption STRING into `CLIPTextEncode` to seed a generation from an existing image's tag set.

## Install

Drop `custom_nodes/comfyui-anima-tagger/` into your ComfyUI `custom_nodes/`, restart ComfyUI. The nodes appear under the `anima` category.

The node works in two install shapes:

1. **Inside the `dragon-trainer` repo** (dev / monorepo). It imports the live `library.captioning.anima_tagger`, so edits in the parent repo are picked up immediately.
2. **Standalone** (just this directory dropped into a vanilla ComfyUI `custom_nodes/`). It falls back to a bundled inference subset under `_vendor/` — no need to clone the parent repo or run `uv sync`. Pip deps are listed in `pyproject.toml` (ComfyUI ships everything except possibly `einops` / `timm` / `pyyaml`, all small).

Both checkpoints auto-download on first use:

- **Tagger checkpoint** (~26 MB) is fetched from [`sorryhyun/anima-tagger`](https://huggingface.co/sorryhyun/anima-tagger) into `tagger_dir` (default `models/captioners/anima-tagger-v2` or `-v1`, aligned with `version`) when any required file is missing.
- **PE-Core-L14-336** vision encoder (~1 GB) is fetched from `facebook/PE-Core-L14-336` into `pe_ckpt` (default `models/pe/PE-Core-L14-336.pt`), and **PE-Spatial-B16-512** into `pe_aux_ckpt` (`models/pe/PE-Spatial-B16-512.pt`) when using dual-encoder `v2`.

### For maintainers — keeping the vendor copy fresh

The `_vendor/` tree is generated from the live `dragon-trainer` source. Regenerate it before bumping the node version:

```bash
python tasks.py vendor-sync       # from the dragon-trainer repo root (compatibility alias: python scripts/sync_vendor.py)
```

## Checkpoint layout

`tagger_dir` should contain (the published `sorryhyun/anima-tagger` checkpoint already does — auto-downloaded if missing):

```
<tagger_dir>/
  config.json              # model config + training metadata           (required)
  model.safetensors        # AnimaTaggerHead state dict                 (required)
  vocab.json               # tag list with category + median_pos info   (required)
  rules.yaml               # caption-normalization rules snapshot       (required)
  thresholds.safetensors   # per-tag F1-optimal thresholds              (optional, falls back to 0.5)
  groups.yaml              # tag-group taxonomy → softmax argmax mode   (optional)
  pe_lora.safetensors      # PE-LoRA delta on PE-Core trailing blocks   (optional, gated on config.pe_lora=true)
```

Default `tagger_dir` resolves relative to the `dragon-trainer` repo root in dev install, or to ComfyUI root in standalone install (absolute paths used as-is). Re-train via `python tasks.py preprocess-tagger` + `python tasks.py tagger` (or `python -m scripts.anima_tagger.cli`) in the parent repo to produce a custom checkpoint; single-image CLI smoke test via `python tasks.py test-tagger --image <path>`.

## Usage

### Caption an image

```
[Load Image] ──┐
               ├─► [Anima Tagger Caption] ──► [Save Text File]
[Anima Tagger Loader] ──┘
       tagger_dir: models/captioners/anima-tagger-v1
```

### Drive a normal text-to-image generation from an existing image's tags

```
[Load Image] ──┐
               ├─► [Anima Tagger Caption] ──► caption ──► [CLIPTextEncode] ──► [KSampler] ──► …
[Anima Tagger Loader] ──┘
```

### Plug into DirectEdit (cross-package, 0.2.0+ `STRING` wiring)

```
[Anima Tagger Loader] ──► tagger ──┐
                                   ├─► [Anima Tagger Caption] ──► caption (STRING)
[Load Image] ────────────┬─────────┘                                  │
                         │                                            ▼
                         └────────► image ──► [Anima DirectEdit] ◄── source_tag
                                                    ▲
                                        target_tag: "<edited caption>"
```

Since `comfyui-anima-directedit` 0.2.0 removed the embedded `ANIMA_TAGGER` socket and `edit_text` input, wire `Anima Tagger Caption`'s `caption` (`STRING`) output into `Anima DirectEdit`'s `source_tag`, and supply the edited target caption string to `target_tag` (via string concat/replace or manual edit) — see [comfyui-anima-directedit](../comfyui-anima-directedit/README.md).

## Files

| File | Role |
|------|------|
| `nodes.py` | `AnimaTaggerLoader` + `AnimaTaggerCaption`. |
| `__init__.py` | Re-exports `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. |
| `pyproject.toml` | ComfyUI Registry metadata. |

## References

- **AnimaTagger architecture.** `docs/experimental/anima_tagger.md` in the parent repo.
- **DirectEdit integration.** `docs/experimental/directedit_editing_v3.md` (why ψ_src manifold-fit matters).
- **Trainer.** `python tasks.py tagger` / `python -m scripts.anima_tagger.cli` in the parent repo (`scripts/anima_tagger/cli.py`).
