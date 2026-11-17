# Colorization (EasyControl)

状态：实验
适用版本：当前 main；可运行边界以 `tasks.py --help` 和实时源码为准

Manga / illustration **colorization** as an EasyControl control task: feed a
black-and-white screentoned page (line art + halftone tone), get a colorized
image. This is the first project under `easycontrol_adapters/` — a workspace for
per-task EasyControl control adapters that share the shipped EasyControl network
(`networks/methods/easycontrol.py`) but differ in how the *condition* is built
(for cross-image subject/edit pairing utilities, see `easycontrol_adapters/tools/{blueprint,subject_pairs,subject_edit_pairs}.py`).

## The idea

Real B&W manga has **no color ground truth**, so we can't make training pairs by
desaturating. We invert the direction:

- **target** = the color illustrations we already have (`post_image_dataset/lora/`
  latents + captions — reused as-is, nothing re-cached).
- **condition** = a synthetic *mangafied* version of the same image (XDoG lineart
  + algorithmic screentone — the toned value range is split into a few luminance
  bands and each gets its own *pattern*: clustered dots, parallel-line/hatch tone,
  or cross-hatch, so darks/mids/lights carry distinct tones the way a real page picks
  tone by value), cached to `cond_cache_dir`.

Synthesizing a manga-like condition matches the inference distribution (real
screentoned pages) far better than naive grayscale. EasyControl's extended
self-attention cond stream does the conditioning; only the cond source changes
(cond ≠ target), wired via the new `cond_cache_dir` subset knob.

## Caption policy — color-only captions

The condition (mangafied lineart + screentone) already encodes *everything
spatial*: composition, poses, objects, layout. The one variable B&W manga can't
carry is **hue/chroma**. So the text channel for colorization is reduced to
**color tags only** — hair/eye/skin color, `<color> <garment>`, background color
(`color_caption.filter_to_colors`). Every surviving token is then a color fact
the model can't get from structure, which gives a strong **prompt→color** binding
instead of the weak, arbitrary steering you get when color tags are buried in a
full caption.

The color-only TE cache lives in its own `text_cache_dir`
(`post_image_dataset/colorize_text`) — a TE-only redirect, so the **latents still
come from the shared `lora/` cache** (nothing re-encoded there). Built by
`prep.py`'s text stage as a multi-variant cache (shipped defaults:
`--text_shuffle_variants 2`, `--text_tag_dropout_rate 0.5`): v0 = the full color
set, v1 = smart-shuffled with each color tag independently dropped at p≈0.5, so
the model learns to colorize from a *partial* color spec, not just a complete
palette.

Two independent knobs shape the per-step caption, and it's worth not conflating
them:

- **`caption_dropout_rate = 0.05`** — the *auto-color floor*. ~5% of steps drop
  the caption entirely (→ uncond `T5("")`), training the empty-prompt default.
  Keep it low: this knob does **not** balance full-vs-partial color, and a high
  rate (the old `0.9`) over-trains the unconditional path into weak, arbitrary
  steering.
- **`use_shuffled_caption_variants = true`** — the *full-vs-partial balance*. On
  the captioned steps, the loader draws **20% v0 (full colors) / 80% v1+
  (partial)** (`strategy.py`), so partial prompts ("pink hair" alone) work.

Composed, the per-step regime is **~5% empty / ~19% full / ~76% partial**.

At inference:

- **empty prompt** → auto-colorization (guesses *modal* colors).
- **color prompt** (`pink hair, blue eyes, white dress`) → reliably steers, and
  partial specs work because of the tag dropout above.

Note the information-theoretic floor: B&W manga doesn't contain hair/eye/costume
color, so the empty-prompt default guesses — correct it with a color prompt when
it matters. This is expected, not a bug.

## Screening engines & scope

Three interchangeable screening engines share the `(img_rgb, seed) -> cond_rgb` contract:

- **CPU `cv2`/`numpy` (`mangafy.py`)** — zero extra model downloads; XDoG lineart + per-band algorithmic screentone.
- **CUDA/PyTorch (`mangafy_gpu.py`)** — GPU port reusing `mangafy.resolve_params` / `_band_plan` verbatim for fast batch preprocessing without extra weights.
- **Learned SD + ControlNet (`screentone_sd.py`)** — on-manifold `Sketch2Manga` (`mangatone` SD1.5 + ControlNet lineart + selective screentone) engine; fetch extra weights via `python tasks.py exp-easycontrol-download EASYADAPTER=colorize`.
- Training targets (illustrations) have **no speech bubbles**; inference manga pages do. Current builds colorize the art and you composite the original B&W text back over the result at inference (speech-bubble pass-through remains deferred).

## Files

| File | Role |
|------|------|
| `mangafy.py` | CPU color RGB → B&W manga (XDoG lineart + dot/line/cross screentone), per-stem jitter |
| `mangafy_gpu.py` | CUDA/PyTorch port of `mangafy.py` with identical band/pattern structure |
| `screentone_sd.py` | Learned `Sketch2Manga` + ControlNet lineart + selective screentone engine (`exp-easycontrol-download EASYADAPTER=colorize`) |
| `color_caption.py` | reduce a full Anima caption to color tags only (`filter_to_colors`) |
| `prep.py` | mangafy `resized/` → VAE-encode into `colorize_cond/`; re-encode color-only captions into `colorize_text/` |

Configs: `configs/datasets/colorize.toml`, `configs/methods/colorize.toml` (base GUI EasyControl variant lives at `configs/gui-methods/easycontrol.toml`; `colorize` does not ship a separate GUI-method variant).

## Run

The colorization project rides the existing `exp-easycontrol*` commands via the
`EASYADAPTER=colorize` selector:

```bash
# 0. (Optional) Download Sketch2Manga weights if using the `sd` screentone engine:
python tasks.py exp-easycontrol-download EASYADAPTER=colorize

# 1. Build the condition latents (mangafy + VAE-encode + color-only TE cache). Idempotent.
python tasks.py exp-easycontrol-preprocess EASYADAPTER=colorize
#    or directly:  python easycontrol_adapters/colorization/prep.py
#    QA a few first:  python easycontrol_adapters/colorization/prep.py --limit 8
#    Inspect staged manga PNGs under post_image_dataset/colorize_staging/

# 2. Train (frozen DiT, adapter-only, color-only caption cache + 5% empty-prompt dropout).
python tasks.py exp-easycontrol EASYADAPTER=colorize

# 3. Inference — feed a real B&W manga page as the control image (empty prompt by default).
python tasks.py exp-test-easycontrol EASYADAPTER=colorize REF_IMAGE=path/to/manga_page.png
#    Color steer:  python tasks.py exp-test-easycontrol EASYADAPTER=colorize REF_IMAGE=path/to/manga_page.png --prompt "pink hair, blue eyes, white dress"
# (Compatibility aliases: make exp-easycontrol-preprocess / exp-easycontrol / exp-test-easycontrol EASYADAPTER=colorize)
```

### Inference settings

- `--easycontrol_image_match_size` — always on (the task sets it); picks the
  token bucket matching the page aspect ratio so tall manga pages don't squash.
- `--easycontrol_scale` (cond structure adherence) — **1.0** (trained default);
  1.1–1.2 if color bleeds past lines, 0.7–0.8 for looser/creative coloring.
- `--guidance_scale` (cfg) — **empty prompt → 1.0–1.5** (nothing for high cfg to
  push toward → oversaturates); **color prompt → 3.0–4.5** (this is what makes a
  color prompt bite).
- `--infer_steps` 20–28, `--sampler euler` — colorization is "easy" (structure
  given); more steps buy little.
- Feed a **real screentoned B&W page** as the cond (it's VAE-encoded as-is, no
  XDoG at inference). A flat grayscale photo / clean lineart is out-of-distribution.

`EASYADAPTER=colorize` ⇒ `configs/methods/colorize.toml` (train),
`easycontrol_adapters/colorization/prep.py` (preprocess), and the `anima_colorize`
checkpoint + `output/tests/colorize/` (inference). Unset ⇒ default ref==target
EasyControl.

## Deferred roadmap

- **Speech-bubble pass-through:** paste synthetic bubbles onto the manga cond and
  mask those regions from the loss, so the model leaves text untouched.
- Gradient/special screentones; difficulty curriculum (single clean figure →
  multi-figure dense pages).
