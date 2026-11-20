# Inference Reference

状态：稳定

## Basic usage

```bash
python inference.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --lora_weight output/ckpt/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 28 \
    --flow_shift 1.0 \
    --guidance_scale 4.0 \
    --sampler er_sde \
    --attn_mode flash \
    --save_path output/tests
```

The canned `python tasks.py test` target picks the latest bakeable LoRA in `output/ckpt/`
and runs against the values above (`INFERENCE_BASE` in
`scripts/tasks/_common.py`). The `inference.py` argparse defaults differ
(`--infer_steps 50`, `--flow_shift 3.0`, `--guidance_scale 3.5`,
`--attn_mode torch`, `--sampler euler`) — `python tasks.py test` is the more
representative starting point. `make` remains available as a compatibility
forwarding wrapper. These `test` / `test-*` commands run inference and generate
images; they are not pytest suites or development acceptance checks. Their
default model/weight assumptions do not imply support for every model family;
use a family-specific command only where that family's documented capabilities
allow it. The example above is for Anima.

## Multi-model standalone inference scope

Standalone CLI inference (`inference.py`) is registered per model family in
`library/models/family_registry.py`:

- **`anima`**: `single`, `batch` (`--from_file`), and `interactive` modes; `euler`, `er_sde`, and `lcm` samplers.
- **`krea2_raw`**: `single` mode + `euler` sampler (`library/models/krea2_raw/inference_runner.py`).
- **`qwen_image_2_1`**: `single` mode + `euler` sampler for T2I, Edit (1–4 references), and saved-latent decoding.
- **`z_image`**: `single` mode + `euler` sampler through `library/models/z_image/inference_runner.py`; `batch`/`interactive` and LoRA are not implemented.
- **`moss_image`**: `single` mode + `euler` sampler through `library/models/moss_image/inference_runner.py`; `batch`/`interactive` and LoRA are not implemented. Text-encoder quantization is selected with `--moss_te_quantization {nf4,bf16}` (default `nf4`). NF4 requires the TE stage on CUDA; CPU inference (`--device cpu` or `--text_encoder_cpu`) must explicitly use `--moss_te_quantization bf16` and BF16 TE compute. No implicit precision fallback is applied.

All five standalone runners are constructed through their family plugin. The
generic entrypoint owns mode dispatch and calls the family runner's `validate`
and `prepare` hooks. The runner-owned generation lifecycle currently covers
`single` only: `generate_latent` and `save_output` are called in that branch.
Anima `batch`/`interactive` remain legacy `process_batch_prompts` /
`process_interactive` paths and do not use the runner generation methods; do not
describe other families or modes as runner-backed until a dedicated batch /
interactive runner contract exists. The entrypoint does not contain a
five-family handler table. WebUI `image-test` remains a separate integration
surface and is not implied by CLI support.

### Krea-2-Raw standalone inference

Use `--model_family krea2_raw --sampler euler` with Krea-2 DiT, Qwen3-VL text
encoder, and the shared Qwen-Image VAE:

```bash
python inference.py --model_family krea2_raw \
    --dit /path/to/krea2-dit.safetensors \
    --text_encoder /path/to/qwen3-vl \
    --vae models/vae/qwen_image_vae.safetensors \
    --sampler euler --attn_mode flash \
    --prompt "your prompt" --image_size 1024 1024 \
    --save_path output/tests
```

Krea-2 uses its official automatic mu-shift (leave `--flow_shift` at its CLI
default `3.0`) and supports `flash` or `torch` (`sdpa` alias) attention. Optional
`--lora_weight` checkpoints must carry `ss_model_family=krea2_raw`. Batch /
interactive modes and Anima-only extras (`--spectrum`, `--spd`, `--dcw`,
`--smc_cfg`, `--cns`, `--pooled_text_proj`, `--pgraft`, `--tiled_diffusion`,
`--ip_adapter_weight`, `--easycontrol_weight`, `--soft_tokens_weight`) are
rejected explicitly.

### Qwen Image 2.1 standalone inference

Use `--model_family qwen_image_2_1 --runtime_dtype bf16 --sampler euler` with
Qwen Image 2.1 DiT, Qwen3-VL text encoder and its dedicated VAE. The default
task is `t2i`; `--qwen_image_2_1_task edit` requires either
`--reference_image PATH` or `--reference_images PATH [PATH...]` (1–4 images,
in the supplied order). These two reference flags are mutually exclusive;
T2I rejects references and other model families reject these Qwen flags.

Both T2I and Edit first load the text encoder strictly on CPU. With a CUDA
runtime device, the shared low-memory runtime automatically chooses core GPU
placement or Accelerate CPU offload using available memory, leaving the unused
`lm_head` on CPU. `--text_encoder_cpu` explicitly keeps encoding on CPU and
disables offload; `--device cpu` also uses CPU encoding. Positive and negative
prompts are encoded separately. Encoder hooks and weights are released before
loading the reference VAE or DiT, including when prompt encoding fails.

```bash
python inference.py --model_family qwen_image_2_1 \
    --dit /path/to/qwen-image-2.1 --text_encoder /path/to/qwen3-vl \
    --vae /path/to/qwen-vae --runtime_dtype bf16 --sampler euler \
    --qwen_image_2_1_task edit --reference_images first.png second.png \
    --prompt "Apply the style of image 2 to image 1" --image_size 512 768 \
    --save_path output/tests
```

Output dimensions are H/W and must be positive multiples of 32. References
use bounded single-frame image IO with EXIF orientation, then one shared
aspect-preserving BICUBIC resize for both VLM and VAE conditioning. Inputs
are never overwritten. Loading stays sequential: text encoder, reference
VAE, DiT, then output VAE. At most one strictly stamped plain LoRA is accepted;
LoKr/LoHa, batch/interactive mode, generic image-test and compile remain
unsupported. Full-weight NF4/BF16 standalone Edit produced a verified 512×512
PNG with one reference, one Euler step and CFG=1 on CMP 90HX. After connecting
automatic encoder offload, four-reference, two-step CFG=2 Edit and saved-latent
decoding also passed; the decoded PNG pixels match the original exactly.
These results do not establish Edit quality or general memory requirements; see the
[standalone inference findings](../findings/qwen_standalone_inference_20261005.md).

For saved Qwen latents, use `--model_family qwen_image_2_1 --latent_path
first.safetensors second.safetensors --vae /path/to/qwen-vae` with BF16/Euler
defaults. This reads the supplied files in order and loads only the dedicated
Qwen VAE, restoring its 64-channel mean/std normalization and 16x pixel stride.
Safetensors must contain `latent`; torch tensor files use safe `weights_only`
loading. Each latent must be finite floating `[1,64,H,W]` with positive H/W;
any saved height/width must match the 16x stride. All files are checked before
VAE loading. Saved seed/prompt/negative prompt metadata applies per file, and
PNG names include the input filename stem. `--output_type` keeps the same
images/latent/latent_images behavior. Edit/reference flags, LoRA weights and
prompt-file/interactive generation are rejected in this decode mode. The CLI
still requires `--text_encoder` and `--save_path`; the encoder is not loaded.

## Test targets

| Target | What it runs |
|--------|--------------|
| `python tasks.py test` | Latest LoRA |
| `python tasks.py test SPECTRUM=1` | Latest LoRA + Spectrum acceleration (mutually exclusive with `SPD=1`) |
| `python tasks.py test SPD=1` | Latest LoRA + training-free SPD multi-resolution sampling |
| `python tasks.py test MOD=1` | Latest LoRA + distilled `pooled_text_proj` (modulation guidance). Composes with `SPECTRUM=1` or `SPD=1`. |
| `python tasks.py test-mod` | Compatibility alias for `python tasks.py test MOD=1` |
| `python tasks.py test NOLORA=1` | Bare DiT (skips `--lora_weight`). Compose with `MOD=1` for a mod-only sample. |
| `python tasks.py test-smc-cfg` | Latest LoRA + SMC-CFG; composes with `SPECTRUM=1`, `SPD=1`, `MOD=1`, and `NOLORA=1`. |
| `python tasks.py test-hydra` | Latest HydraLoRA / FeRA `*_moe.safetensors` (router-live) |
| `python tasks.py test-merge MODEL_DIR=...` | Inference against a baked DiT under `MODEL_DIR=` |
| `python tasks.py test-dcw` | Latest LoRA + DCW scalar bias correction (task target passes λ = 0.01) |
| `python tasks.py test-dcw-v4` | Bare DiT + DCW v4 learnable calibrator by default; set `NOLORA=0` for latest LoRA |
| `python tasks.py test-spectrum-dcw` | Spectrum + DCW scalar |
| `python tasks.py test-dcw-v4-spectrum` | Spectrum + DCW v4 |
| `python tasks.py exp-test-spd` | Latest SPD trajectory LoRA on the SPD sampler at its trained schedule |
| `python tasks.py exp-test-soft` | Latest `soft_tokens` checkpoint (`--soft_tokens_weight`) |
| `python tasks.py exp-invert-directedit` | DirectEdit postfix-tail inversion probe |
| `python tasks.py exp-test-ip <ref_image>` | IP-Adapter (image-conditioned) |
| `python tasks.py exp-test-easycontrol <ref_image>` | EasyControl |
| `python tasks.py exp-test-directedit` | DirectEdit on a random source image |
| `python tasks.py exp-test-directedit-dry` | DirectEdit reconstruction sanity check |

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_weight` | — | Adapter path(s); space-separated for stacking |
| `--lora_multiplier` | 1.0 | Scalar (one per weight or one for all) |
| `--anima_selective_lora` | off | Enable Anima selective LoRA loading by preset or per-block selection |
| `--soft_tokens_weight` | — | Soft Tokens `.safetensors` bank spliced into cross-attention |
| `--ip_adapter_weight` | — | IP-Adapter checkpoint |
| `--ip_image` | — | Reference image for IP-Adapter |
| `--ip_scale` | `None` | Override IP-Adapter scale (defaults to `ss_ip_scale` in checkpoint metadata) |
| `--ip_image_match_size` | off | Match target H/W to the ref image bucket |
| `--easycontrol_weight` | — | EasyControl checkpoint |
| `--easycontrol_image` | — | Reference image for EasyControl |
| `--easycontrol_scale` | `None` | Override EasyControl scale (defaults to `ss_cond_scale` in checkpoint metadata) |
| `--easycontrol_image_match_size` | off | Match target H/W to ref bucket |
| `--infer_steps` | 50 | Denoising steps (28 via `python tasks.py test`) |
| `--guidance_scale` | 3.5 | Text CFG scale (4.0 via `python tasks.py test`) |
| `--flow_shift` | 3.0 | Flow-matching schedule shift (1.0 via `python tasks.py test`) |
| `--sampler` | `euler` | `euler` (deterministic ODE), `er_sde` (stochastic), `lcm` (x0 re-noise — distilled few-step models) |
| `--spd` | off | Spectral Progressive Diffusion multi-resolution denoising (forces `euler`; mutually exclusive with `--spectrum`) |
| `--cns` | `None` | Colored Noise Sampling on `er_sde` (`auto` or `cns_gamma.npz` path; no-op on deterministic samplers) |
| `--attn_mode` | `torch` | `torch` / `flash` / `flex` / `sageattn` / `xformers` (`sdpa` alias) |
| `--from_file` | — | Batch prompts from a text file |
| `--interactive` | off | REPL prompt loop |
| `--compile` | off | Whole-model `torch.compile` speedup |
| `--compile_blocks` | off | Compile each DiT block's `_forward` individually |
| `--text_encoder_cpu` | off | Keep the text encoder on CPU (low-VRAM) |
| `--vae_chunk_size` | — | Spatial chunk size for VAE encode/decode (even integer) |
| `--vae_disable_cache` | off | Skip the internal VAE cache |
| `--no_metadata` | off | Don't embed training metadata in the PNG |
| `--save_path` | — | Output directory |

`--fp8` and `--prefix_weight` were removed. The legacy `--dcw_v4 <head>`
alias still parses but maps to `--dcw_calibrator`.

## Modulation guidance

Distilled `pooled_text_proj` MLP steers AdaLN coefficients at inference. See
[`../methods/mod-guidance.md`](../methods/mod-guidance.md).

| Flag | Description |
|------|-------------|
| `--pooled_text_proj` | Path to the distilled MLP |
| `--mod_w` | Guidance strength (positive boosts) |
| `--mod_pos_prompt` / `--mod_neg_prompt` | Text targets for the AdaLN delta |
| `--mod_start_layer` / `--mod_end_layer` | Layer band to apply the delta on |
| `--mod_taper` / `--mod_taper_scale` / `--mod_final_w` | Schedule shaping |

## Spectrum acceleration

Training-free; Chebyshev feature forecasting skips most blocks on cached
steps. See [`../methods/spectrum.md`](../methods/spectrum.md).

| Flag | Description |
|------|-------------|
| `--spectrum` | Enable |
| `--spectrum_warmup` | Steps before caching starts |
| `--spectrum_window_size` / `--spectrum_flex_window` | Adaptive window schedule |
| `--spectrum_w` / `--spectrum_m` / `--spectrum_lam` | Forecasting hyperparameters |
| `--spectrum_stop_caching_step` | Last cached step |
| `--spectrum_calibration` | Bias adjustment |

## DCW (sampler-level SNR-t bias correction)

Composes with every adapter and with Spectrum. Two modes — see
[`../methods/dcw.md`](../methods/dcw.md).

| Flag | Description |
|------|-------------|
| `--dcw` | Scalar mode (one global λ) |
| `--dcw_lambda` | λ value (`inference.py` default: -0.015; `python tasks.py test-dcw` passes 0.01) |
| `--dcw_band_mask` | Haar subband mask: `'LL'` (default), `'HH'`, `'LH+HL+HH'`, or `'all'` |
| `--dcw_schedule` | Per-step shaping: `one_minus_sigma` (default), `sigma_i`, `const`, or `none` |
| `--dcw_calibrator` | Path to a v4 fusion-head safetensors (replaces scalar) |
| `--dcw_calibrator_gain` | Multiplicative scale on the calibrator's α̂ |

The shipped scalar default (-0.015) is tuned for CFG=1; at production CFG=4
the bias direction is (CFG × aspect)-dependent — non-square aspects want
small **positive** λ. The Spectrum ComfyUI node ships +0.01 as its scalar
default. Prefer `--dcw_calibrator` for production runs.

## P-GRAFT

Dynamic-hook variant that lets you cut LoRA off mid-trajectory (originally
for prefix inversion; useful as a debug switch).

```bash
python inference.py ... \
    --pgraft \
    --lora_cutoff_step 37    # LoRA active for steps 0–36, disabled 37+
```

## Tiled denoising and VAE chunked decode

Denoising-stage tiling and VAE spatial chunking are independent switches with
different units:

- **Denoising tiled diffusion** (`--tiled_diffusion`): runs MultiDiffusion-style
  overlapping tiles in **latent space** (even integers; `--tile_size` defaults
  to `128` latent = 1024px, `--tile_overlap` defaults to `16` latent = 128px,
  with `tile_overlap < tile_size`):
  ```bash
  --tiled_diffusion --tile_size 128 --tile_overlap 16
  ```
- **VAE chunked decode** (`--vae_chunk_size`): spatial chunking inside the VAE
  when full-frame VAE decode exceeds VRAM (even integer, e.g.
  `--vae_chunk_size 64`, optionally paired with `--vae_disable_cache`).

## Prompt-file format

One prompt per line (`--from_file`); inline options parsed by
`inference.py::parse_prompt_line` are `--w` (width), `--h` (height), `--d`
(seed), `--s` (steps), `--g` / `--l` (guidance scale), `--fs` (flow shift),
and `--n` (negative prompt):

```
a girl standing in a field --w 1024 --h 1024 --s 50 --g 5.0
another prompt --d 42 --fs 4.0 --n blurry, low quality
```

## LoRA in ComfyUI

Plain Anima LoRA `.safetensors` use kohya-ss `lora_unet_` key naming and
load directly into ComfyUI's stock `LoraLoader` — no conversion. For
HydraLoRA / FeRA / ReFT / postfix / prefix checkpoints (extra `router.*`,
`reft_*`, stacked `lora_ups.N.*` keys that the stock loader drops), use the
**Anima Adapter Loader** in `custom_nodes/comfyui-hydralora/`.

Spectrum KSampler + mod-guidance + in-node DCW (scalar default +0.01, plus
an `auto` mode that runs the v4 fusion head) live in
[ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler).
