# Embedding Inversion (historical)

> Current status: the old `make invert`, `make invert-ref`, `exp-test-ref`,
> the former `archive/inversion/invert_embedding.py` prototype, and inference
> `--prefix_weight` paths are not active commands in this tree. The archive
> prototype was removed in commit `03791a0a`. The runnable inversion-related
> probe is now:
>
> ```bash
> python tasks.py exp-invert-directedit
> ```
>
> It runs `scripts/inversion/invert_postfix_tail.py`, then compares DirectEdit
> dry-mode output with and without the learned postfix tail.

Finds the optimal text embedding (`crossattn_emb`) for a target image by optimizing in the post-T5, pre-DiT embedding space. The frozen DiT acts as a fixed decoder — only the embedding is updated via gradient descent on the flow-matching loss.

This reveals "how the DiT interprets the image" in embedding space, producing a `.safetensors` file that can be used as a conditioning input for inference or as an analysis tool for understanding model behavior.

## Quick start

```bash
python tasks.py exp-invert-directedit
```

Optional environment overrides include `REF_IMAGE`, `N_IMAGES`, and `K`; see
`python tasks.py --help` for the live command summary.

## How it works

1. **Load target** — either encode a raw image via VAE, or load cached latents from `post_image_dataset/`.
2. **Initialize embedding** — from cached text encoder output, a text prompt, a saved embedding, or zeros.
3. **Optimize** — for each step, sample random noise levels (sigmas), run the frozen DiT forward, compute MSE between predicted and target noise, and backpropagate through the DiT to update only the embedding.
4. **Save** — the best embedding (lowest loss) is saved as a `.safetensors` file with metadata.

### Optimization details

Each step samples `timesteps_per_step × grad_accum` random timesteps. The embedding is maintained in float32 for optimizer precision and cast to bfloat16 for the DiT forward pass. Gradient norm is clipped to 1.0.

## Two input modes

### Single image (`--image`)

Encodes the image via VAE on the fly. Requires `--vae`. The VAE is loaded, used, and freed before the DiT loads.

Historical script path: `archive/inversion/invert_embedding.py --image ...`
is not a current task entry.

### Batch from preprocessed directory (`--image_dir`)

Uses cached latents (`.npz`) and optionally cached text encoder outputs (`_anima_te.safetensors`) from `make preprocess`. No VAE needed at runtime. Skips images that already have an output file.

Historical script path: `archive/inversion/invert_embedding.py --image_dir ...`
is not a current task entry.

## VRAM modes

| `--blocks_to_swap` | Behavior | VRAM |
|---|---|---|
| `0` (default) | Full model on GPU + `torch.compile` | Lowest (~12.5 GB) |
| `N > 0` | Swap N blocks to CPU, no compile | Higher (more swap = less VRAM but slower) |
| `-1` | Gradient checkpointing + `torch.compile` | Medium |

`torch.compile` provides significant memory savings through operator fusion, often outweighing block swap for small swap counts. Block swap is useful when the model doesn't fit on GPU at all.

Historical Make overrides such as `make invert INVERT_SWAP=12` are no longer
registered.

## Embedding initialization

Priority order (first match wins):

| Flag | Source |
|------|--------|
| `--init_embedding path.safetensors` | Load from a previously saved embedding |
| `--init_prompt "text"` + `--text_encoder` | Encode prompt through text encoder + LLM adapter |
| (default) `--init_from_cache` | Use cached `crossattn_emb` from `_anima_te.safetensors` (batch mode) |
| `--init_zeros` | Start from zero embedding |

## Verification

Add `--verify` to generate an image from the inverted embedding after optimization (requires `--vae`):

Historical script path: `archive/inversion/invert_embedding.py --verify ...`
is not a current task entry.

## Block gradient logging

`--log_block_grads` logs per-block gradient norms during optimization, saved as `_block_grads.json` alongside the loss CSV. This shows which DiT blocks are most sensitive to the embedding — useful for understanding which blocks drive cross-attention behavior.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dit` | (required) | DiT checkpoint path |
| `--vae` | — | VAE path (required for `--image` mode and `--verify`) |
| `--text_encoder` | — | Text encoder path (for `--init_prompt`) |
| `--attn_mode` | flash | Attention backend |
| `--image` | — | Single target image path |
| `--image_dir` | — | Preprocessed dataset directory |
| `--num_images` | all | Process N images from `--image_dir` |
| `--shuffle` | off | Shuffle image order |
| `--steps` | 100 | Optimization steps per image |
| `--lr` | 0.01 | Learning rate |
| `--lr_schedule` | cosine | `cosine` or `constant` |
| `--timesteps_per_step` | 1 | Timesteps sampled per step (batched) |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--sigma_sampling` | uniform | `uniform` or `sigmoid` |
| `--blocks_to_swap` | 0 | Blocks to swap to CPU (0 = compile, -1 = grad ckpt) |
| `--verify` | off | Generate verification image after optimization |
| `--log_block_grads` | off | Log per-block gradient norms |
| `--output_dir` | inversions | Output directory |

## Output structure

```
inversions/
├── results/
│   ├── image1_inverted.safetensors   # optimized embedding
│   ├── image1_verify.png             # verification image (if --verify)
│   └── ...
└── logs/
    ├── image1.csv                    # per-step loss, lr, grad_norm
    ├── image1_block_grads.json       # per-block gradient norms (if --log_block_grads)
    └── ...
```

---

# Reference Inversion (K-slot prefix)

A "referencer" variant of embedding inversion: instead of optimizing all 512 token positions of the crossattn embedding, freeze a user-supplied text template and optimize **only K consecutive token vectors** against a single reference image. The resulting K vectors capture the image's subject/style in T5-compatible space; at inference they're spliced into a fresh user prompt, letting the subject travel into new scenes.

This is the original Textual Inversion recipe (Gal et al. 2022) ported to Anima. This section is kept as historical design context only: the old inference-side splicing flag `--prefix_weight` was removed, and the old reference-inversion Make targets are not registered in `tasks.py`.

## Quick start

The old quick-start commands (`make invert-ref`, `make exp-test-ref`,
`python tasks.py invert-ref`, `python tasks.py exp-test-ref`) are no longer
registered. Use `python tasks.py exp-invert-directedit` for the current
postfix-tail inversion probe.

## How it differs from full inversion

| Aspect | `invert` | `invert-ref` |
|---|---|---|
| Variables optimized | All 512 crossattn positions | K consecutive positions (default K=8) |
| Output key | `crossattn_emb` (512×D) | `prefix_embeds` (K×D) |
| Conditioning at inference | Replaces full crossattn | Historically spliced into front of user's prompt via removed `--prefix_weight` |
| Composes with user prompt | No (full embed is subject-specific) | Yes (user controls 512-K positions) |
| Compose with other LoRAs | Not designed for it | Yes — doesn't touch DiT weights |
| Typical use | Analysis / reconstruction | Subject/style reference for new scenes |

## How it works

1. **VAE encode** the reference image to latents, free VAE.
2. **Encode the template** (e.g. `"a photo"`) through Qwen3 + LLM adapter to produce a fixed `template_emb` of shape `[1, 512, D]`.
3. **Create K trainable slot vectors** `slots: [K, D]` (init: small random, zeros, or first-K-of-template).
4. **Optimize**: at each step, assemble `emb = [slots ; template[:512-K]]` and run the flow-matching loss against the reference image. Only `slots` is trainable.
5. **Save** the K slots as `prefix_embeds` + metadata (template, placeholder offset, stats).

The assembly in step 4 byte-for-byte matches what `PostfixNetwork.prepend_prefix` does at inference — trains exactly what runtime splices.

## Historical commands

### `make invert-ref`

This target is no longer registered. The variables below are kept only to
explain the old design.

| Env var | Default | Description |
|---|---|---|
| `REF_IMAGE` | (random pick) | Explicit reference image path |
| `REF_IMAGE_DIR` | `post_image_dataset` | Directory to pick from when `REF_IMAGE` unset |
| `REF_TEMPLATE` | `"a photo"` | Frozen text around the learned K slots |
| `REF_K` | `8` | Number of trainable slot vectors |
| `REF_STEPS` | `100` | Optimization steps |
| `REF_LR` | `0.01` | Learning rate |
| `REF_NAME` | `latest` | Output saved as `output/anima_ref_$(REF_NAME).safetensors` |
| `REF_SAVE_PATH` | — | Overrides `REF_NAME` with an explicit path |
| `REF_SWAP` | `0` | `blocks_to_swap` (same semantics as `invert`) |

Historically, when `REF_IMAGE` was unset, a random image was picked from
`REF_IMAGE_DIR` and frozen for the whole target run.

### `make exp-test-ref`

This target is no longer registered. It used the removed `--prefix_weight`
runtime path and should not be used as current guidance.

## Direct script usage

```text
# Historical only; archive/inversion/invert_reference.py is not a current task entry.
python archive/inversion/invert_reference.py \
    --image path/to/ref.png \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --template "a photo of <REF> in a scene" \
    --num_tokens 8 \
    --steps 100 --lr 0.01 \
    --save_path output/anima_ref_cat.safetensors \
    --verify
```

The `<REF>` marker in `--template` is stripped before encoding (the K slots always front-prepend at runtime today) but the marker's character offset is recorded in safetensors metadata (`ss_placeholder_char_offset`), reserved for a future placement-aware inference loader.

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--image` | (required) | Reference image path |
| `--dit` / `--vae` / `--text_encoder` | (required) | Model paths |
| `--template` | `"a photo"` | Frozen text around the K slots |
| `--num_tokens` / `-K` | `8` | Number of slot vectors |
| `--init_std` | `0.02` | Gaussian std for slot init (0 = zero-init) |
| `--init_from_template` | off | Init slots from first K positions of encoded template |
| `--steps` | `100` | Optimization steps |
| `--lr` | `0.01` | Learning rate |
| `--save_path` | (required) | Output safetensors path |
| `--verify` | off | Render a sanity-check image using `--verify_prompt` (default = `--template`) |
| `--blocks_to_swap` | `0` | Same semantics as `invert` |

## Output file format

Single `.safetensors` holding one tensor:

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `prefix_embeds` | `[K, D]` | bf16 | Historical prefix-mode checkpoint schema; the former `networks/methods/postfix.py` implementation was removed in commit `3f1bc4a5` |

Metadata includes `ss_network_module`, `ss_mode=prefix`, `ss_num_postfix_tokens=K`, `ss_embed_dim=D`, plus inversion-specific fields (`ss_reference_image`, `ss_template`, `ss_placeholder_char_offset`, `ss_best_loss`, `ss_steps`, `ss_lr`, `ss_seed`).

Historically this made the file interchangeable with prefix-mode checkpoints;
the `inference.py --prefix_weight` loader no longer exists.

## Caveats and future work

- **Prefix applied to positive AND negative conditioning.** `library/inference/generation.py` calls `prepend_prefix` on both `embed` and `negative_embed`. For pure prefix tuning (quality prior) that's fine; for a *reference* you may want the subject only on the positive path. Splitting the two is a small inference-side patch, not done yet.
- **Placement mode is metadata-only.** A `<REF>` marker in `--template` today just gets stripped — the K slots always front-prepend. The character offset is recorded so a future loader can splice the K vectors into the middle of a user's prompt where they write `<REF>`, instead of at position 0.
- **Text-space ceiling.** Like all textual inversion, this can't encode detail that T5 space wasn't trained to represent (exact pose, micro-geometry, pixel-precise composition). For stronger fidelity, a KV-cache reference-attention approach (concat ref K/V into self-attn) is the next rung up — trades extra compute per step for direct access to the DiT's visual representation space.
