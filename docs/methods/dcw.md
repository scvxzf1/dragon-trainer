# DCW — Post-Step SNR-t Bias Correction

状态：稳定
适用版本：当前 main

Training-free, sampler-level correction that closes the SNR-t bias of flow-matching DiTs by mixing each Euler step's `prev_sample` toward (or away from) the model's `x0_pred`.

Paper: [Elucidating the SNR-t Bias of Diffusion Probabilistic Models](https://arxiv.org/abs/2604.16044) (Yu et al., CVPR 2026)

**Historical evidence:** [DCW bias findings, 2026-05](../../_archive/docs/findings/dcw-bias-findings-202605.md)
(CFG=1 / no-LoRA bench, integrated signed gap −406 — the source of the
historical scalar default `λ = −0.015`). At production CFG=4 the picture is
**(CFG × aspect)-dependent**: paper-direction on non-square aspects,
paper-opposite on 1024² and at CFG=1. v4 per-aspect bucket priors land at small
positive λ_scalar on every CFG=4 bucket; the negative scalar is a CFG=1 artifact.
See §"Bias direction by CFG × aspect" below.

## Two modes

| Mode | What `λ` is | When to use |
|---|---|---|
| **scalar** (v0/v1) — `--dcw` | a single global constant tuned offline (`inference.py` argparse default `−0.015`; canned `python tasks.py test-dcw` overrides to `0.01`) | minimal/safe default; one-line ablation; fallback when v4 isn't calibrated |
| **v4 learnable** — `--dcw_calibrator <artifact>` | a function of `(aspect, prompt, observed prefix gap)` produced at runtime by a small MLP | per-prompt amplitude + per-trajectory steering; trained per checkpoint via `python tasks.py dcw` |

The math at the apply site is identical:

```
denoised   = latents − σ_i · v                       # x0_pred (FLUX velocity convention)
prev       = Euler/ER-SDE step                        # prev_sample
diff       = prev − denoised
diff_LL    = haar_idwt(LL(diff), 0, 0, 0)             # LL-only band mask
prev      += λ_i · diff_LL                            # DCW correction
```

What differs is how `λ_i` is produced. In scalar mode, `λ_i = λ · sched(σ_i)`. In calibrator mode, `λ_i = baseline_lambda · (1 − σ_i) + α_eff · gain · (1 − σ_i)` (clamped to `±0.05`) with the controller observing the first `k_warmup` steps' LL gap (or FEI low-band energy) before firing.

This page contains the current calibrator derivation, gates, and fallback ladder.

## Quick start

```bash
python tasks.py test-dcw                 # latest LoRA + scalar λ=0.01 (task target override)
python tasks.py test-dcw-v4              # bare DiT + calibrator by default; set NOLORA=0 for latest LoRA
python tasks.py test-spectrum-dcw        # Spectrum + scalar DCW composed
python tasks.py test-dcw-v4-spectrum     # Spectrum + DCW calibrator composed
# Compatibility aliases: make test-dcw / make test-dcw-v4 / make test-spectrum-dcw
```

Scalar mode also works on any `inference.py` invocation:

```bash
python inference.py --dcw                                      \
    --dcw_lambda -0.015 --dcw_schedule one_minus_sigma         \
    --dcw_band_mask LL                                         \
    ...  # other inference args
```

Calibrator mode (`python tasks.py test-dcw-v4` auto-resolves the newest `fusion_head.safetensors` under `output/dcw/`; when calling `inference.py` directly, pass the artifact file or run directory path; `--dcw_v4` is still accepted as a legacy alias):

```bash
python inference.py --dcw_calibrator output/dcw/<run>/fusion_head.safetensors \
    --dcw_calibrator_gain 1.0 \
    ...
```

The tqdm progress bar shows per-step `λ` (and `α` once the head fires) so the controller's trajectory is interpretable in real time.

## v4 architecture in one paragraph

Three input channels feed one shared MLP. (1) **Aspect prior** — per-bucket profile indexed by `(H, W)`. (2) **Prompt embedding** — mean-pooled `crossattn_emb` (`c_pool`, 1024-dim) plus auxiliary scalars `[caption_length, cos(c_pool, μ_centroid), token_l2_std]`. (3) **Observed prefix** — LL-band Haar norms of `noise_pred` (or 2-band FEI low-band energy in v6 artifacts) over the first `k_warmup` denoising steps (default `k=7`). The MLP predicts `α̂` at step `k_warmup`, scaled by `--dcw_calibrator_gain` along the `(1 − σ_i)` envelope across `[target_start, target_end)`, with per-step `λ_i` clamped to `±0.05` as an overshoot guard.

## Calibration: `python tasks.py dcw`

```bash
python tasks.py dcw                              # default calibration: 8 prompts × 2 seeds × 5 aspect buckets
python tasks.py dcw --n_images 32 --n_seeds 2    # larger pool (extra args forwarded directly)
python tasks.py dcw-train                        # train-only on existing output/dcw/ pool (~30s)
# Compatibility aliases: make dcw / make dcw ARGS="--n_images 32 --n_seeds 2" / make dcw-train
```

`python tasks.py dcw` runs `scripts/dcw/measure_bias.py --dump_per_sample_gaps` against the five `DCW_ASPECT_BUCKETS` at the production env (CFG=4, mod_w=3.0), then chains `scripts/dcw/train_fusion_head.py` on the pooled output. Current defaults: 8 prompts × 2 seeds × 5 buckets, collected with `--baseline_lambda 0.0`. All new outputs land in `output/dcw/<timestamp>-<label>/`; `train_fusion_head.py` reads `output/dcw/` by default (legacy roots such as `post_image_dataset/dcw/` only participate when present on disk or passed explicitly via `--results_root`).

End artifact: `output/dcw/<run>/fusion_head.safetensors` — single file containing the `FusionHead` weights, standardization stats, and metadata. `python tasks.py test-dcw-v4` auto-resolves the newest artifact by mtime.

## CLI

| Flag | Mode | Default | Notes |
|------|------|---------|-------|
| `--dcw` | scalar | off | Enable post-step correction with a constant `λ`. |
| `--dcw_lambda` | scalar | `-0.015` in `inference.py`; `python tasks.py test-dcw` passes `0.01` | Historical negative scalar came from CFG=1/no-LoRA findings; the canned task target overrides it for current smoke runs. |
| `--dcw_schedule` | scalar | `one_minus_sigma` | One of `one_minus_sigma`, `sigma_i`, `const`, `none`. |
| `--dcw_band_mask` | scalar | `LL` | Haar subband mask: `LL`, `HH`, `LH+HL+HH`, `all`. LL-only is strictly better than `all` on Anima — see §LL-only correction. |
| `--dcw_calibrator` / legacy `--dcw_v4` | calibrator | unset | Path to `fusion_head.safetensors` (or a directory containing one). When set, overrides scalar `--dcw_lambda` with per-step controller output (LL-only). |
| `--dcw_calibrator_gain` / legacy `--dcw_v4_alpha_gain` | calibrator | `1.0` | Multiplier on top of the head's predicted `α̂` (which is in `λ`-units, median `|α̂| ≈ 0.015`). `1.0` is identity; `2.0` doubles magnitude; negative flips sign. Final per-step `λ_i` is clamped to `±0.05`. |

The final step (`σ_{i+1} == 0`) is always skipped in both modes — at that step `prev == x0_pred` exactly, so DCW would be a numerical no-op.

## When to use which

The **scalar default `λ = −0.015`** was tuned on the CFG=1 / no-LoRA bench. Production CFG=4 has a (CFG × aspect)-dependent bias direction (see §"Bias direction by CFG × aspect" above): on CFG=4 non-square aspects the optimal λ is small and *positive*, so a uniformly-negative scalar pushes the wrong way. This is what causes scalar DCW to over-sharpen intentionally flat styles (channel/caststation-class artists) — not a "calibrator can't see style" failure (cross-attention embeddings clearly do represent artist style; otherwise generation couldn't render it), but uniform-λ-across-prompts mismatching the per-cell optimum.

Practical guidance:
- **Scalar mode**: helps when the target is detail-dense (busy compositions, intricate textures); leave off for intentionally simple/flat styles. The recommendation gates on prompt intent. Detail-dense + non-square at CFG=4 is the cell where the scalar's negative λ is most clearly wrong-direction; the perceptual win there comes from the LL-band restriction + `(1−σ)` schedule shape, not from the sign of λ.
- **v4 mode**: addresses the uniform-λ problem by predicting per-prompt α̂. Cross-attention features feed the head directly via `c_pool`, so style-intent is in scope; `g_obs` adds per-trajectory steering. The expected behaviour on flat-style prompts is α̂ near zero or sign-corrected. Until perceptual side-by-side validation passes across diverse styles, scalar remains the safer default for production runs of unknown style.

## Composition

DCW lives at the sampler boundary, not inside any module — composes with everything below.

| Composes with | How |
|---|---|
| `--sampler er_sde` | Applied post-`er_sde.step`. |
| `--tiled_diffusion` | Applied to post-merge latents, not per-tile. **v4 controller currently no-ops in tiled mode** (single-tile assumption in `c_pool`/`g_obs`); scalar still works. |
| `--spectrum` | Applied at the same sampler-step site on both actual-forward and cached-step branches. On cached steps, `x0_pred = latents − σ_i · noise_pred` carries Spectrum's prediction error; correction is bias-agnostic so this is fine. v4 hasn't been ablated on cached steps. |
| `--lora_weight` / Hydra / OrthoLoRA / T-LoRA / ReFT / postfix | Orthogonal — no module patching, no extra weights. v4 calibrates against the base DiT by default; per-LoRA calibration is `python tasks.py dcw --lora_weight <path>` (writes a separate artifact). |

Untested at v0:
- CFG ≠ 4 with v4 (the prototype was calibrated at CFG=4; falls back to bucket prior or scalar via the proposal's fallback ladder).
- Stacked LoRA / OrthoLoRA / T-LoRA / ReFT (one row per family).

## v4 status (prototype)

Trained on the historical prototype calibration pool — 176 rows, 40 unique stems, 8-fold prompt-stratified CV:

| Metric | Threshold | Prototype |
|---|---|---|
| r(α̂_p, mean_s r) per-prompt | ≥ 0.6 | **+0.89** ✓ |
| r(α̂_p,s, r_p,s) seed-conditional | ≥ 0.7 | **+0.88** ✓ |
| r(σ̂_p, std_s r) | ≥ 0.4 | −0.01 ✗ |
| NLL improvement vs N(0, σ²_pop) | ≥ 15% | +5.7% ✗ |

α̂ channels pass strongly, whereas the prototype σ̂² channel was under-supervised at single-seed-majority data. Consequently, `OnlineDCWCalibrator` (`dcw_v5_lambda_scalar` / `dcw_v6_fei_*`) predicts direct `λ`-unit `α̂` without runtime σ̂² shrinkage, relying on `--dcw_calibrator_gain` (default `1.0`) and the hard `±0.05` per-step clamp guard.

**Output clamp.** Per-step `λ_i` is bounded at `±0.05` (`_LAMBDA_CLAMP` in `library/inference/corrections/dcw_calibrator.py`) as a safety guard against tail overshoot.

## Anima form details (kept for reference)

### Bias direction by CFG × aspect

The bias direction is a **(CFG × aspect)** interaction, not a fixed property of Anima. Integrated signed LL gap on 28-step baselines:

| Setting | ∫ gap_LL | Direction | λ_scalar (LSQ) |
|---|---:|---|---:|
| CFG=1, no LoRA, no mod-guidance (historical run `20260503-1720`) | −406 | paper-opposite | −0.015 (shipped scalar) |
| CFG=4, 1024² (historical run `20260504-1648`) | −188 | paper-opposite | +0.0046 |
| CFG=4, 832×1248 HD portrait (`output/dcw/20260505-0130`) | +89 | paper-direction | +0.0059 |
| CFG=4, 1248×832 inv-HD landscape (`output/dcw/20260505-0612`) | +205 | paper-direction | +0.0127 |

Paper-direction means Yu et al.'s Key Finding 2 (`||v_θ(x̂_t)|| > ||v_θ(x_t_fwd)||`) holds — gap is positive late, closed by **positive** λ. Paper-opposite is the inverse, closed by negative λ. The `λ_scalar` column is the LSQ-optimal `(1−σ)`-weighted constant per cell; the v4 controller distributes a per-trajectory `α̂` on top.

The scalar default `λ = −0.015` was tuned against the CFG=1 bench before A2 measured the per-aspect CFG=4 baseline and is correct only in that regime; on CFG=4 non-square it pushes the gap further from zero. The v4 fallback ladder uses the per-aspect bucket prior at CFG=4 instead of the scalar. The [archived findings](../../_archive/docs/findings/dcw-bias-findings-202605.md) discuss several speculative mechanisms (manifold mismatch, max-padded cross-attention sinks and mod-guidance interaction), but none tightly explain why the CFG × aspect interaction inverts the sign.

### Why `(1 − σ)` schedule (scalar mode)

The bias is concentrated at low σ on Anima — `gap` is small around σ=0.5 and grows to ≈−64 by σ=0.04. The paper's `σ_i` decay would put correction in the wrong place; `const` overcorrects mid-trajectory and sign-flips the gap by step 15 (visible as over-smoothing). `(1 − σ)` weights late steps heaviest, matches the bias envelope, and dominated the 8-prompt visual panel. v4 inherits the `(1−σ)·λ_scalar` envelope as `base_lambda` and adds bucket + head corrections on top.

### Scalar λ calibration (when to retune)

The default `λ=-0.015` was derived from two independent estimates that agreed (perceptual winner of a wide sweep + closed-form fit on a narrow sweep). To re-tune for a different checkpoint / CFG-on / on a LoRA stack:

1. `python scripts/dcw/measure_bias.py --dcw_sweep --dcw_scalers 0 -0.010 -0.020` (or any 3+ anchors).
2. Read `λ*` from the printed line — `(1−σ)`-weighted least-squares optimum:
   ```
   s_i  = ∂gap/∂λ                            (finite-diff from any 2 anchors)
   w_i  = (1 − σ_i)
   λ*   = − Σ w_i · g_i · s_i  /  Σ w_i · s_i²       (over i ≥ N/2)
   ```
3. Confirm with a tighter sweep `{λ*−ε, λ*, λ*+ε}`.

For calibrator re-training, use `python tasks.py dcw` instead — it produces the per-aspect pool + fusion head in one go.

## LL-only correction (2026-05-03 finding)

A per-Haar-subband sweep (historical run `20260503-2102-band-mask-eyeball`) on a 4-image / 2-seed bench produced:

| Config | late-half integrated \|gap\| | Δ vs baseline | per-band signed gap (LL / LH / HL / HH) |
|---|---|---|---|
| baseline | 330.1 | — | −317 / −165 / −165 / −127 |
| **`λ=-0.01_one_minus_sigma_LL`** | **235.7** | **−28.6%** | **−225 / −120 / −122 / −92** *(all bands improved)* |
| `λ=-0.01_one_minus_sigma_all` | 340.6 | **+3.2%** | −180 / −240 / −242 / −222 *(LL improved, detail bands worsened)* |
| `λ=-0.01_one_minus_sigma_HH` | 363.6 | +10.2% | −300 / −146 / −147 / **−287** *(HH sign-flipped)* |

Restricting the correction to LL is **strictly better** by every metric we checked: lower late-half |gap|, no sign flips, all four per-band gaps improved vs baseline, and visually equivalent or slightly better on the 4-image panel. The mechanism: LL is an upstream causal lever — applying LL-only correction at step `i` propagates through the DiT's nonlinear forward and tightens all four band gaps at step `i+1` and after. Detail bands are downstream symptoms, not independent failures.

**Both modes ship LL-only.** Scalar default `--dcw_band_mask LL`; v4 controller hardcodes `LL` (the broadband ablation hasn't been re-run on v4 and isn't a near-term priority).

### LL-only λ magnitude (scalar)

| λ | late-half \|gap\| | Δ vs baseline | max \|gap\| |
|---|---|---|---|
| baseline | 330.1 | — | 64.0 |
| −0.005 | 281.5 | −14.7% | 53.0 |
| −0.010 | 235.7 | −28.6% | 42.1 |
| **−0.015** | **192.6** | **−41.7%** | **31.8** |

Closes 83% of the LL gap at the worst step (σ=0.04) and leaves headroom for per-LoRA calibration to push either direction. The closed-form solver predicts λ* ≈ −0.033 but that extrapolation crosses the nonlinear regime where `LL_const`-style overshoot kicks in (|λ · w(σ)| > ~0.01 late-step).

## Limitations / open questions

- **Multi-seed uncertainty supervision** — the prototype σ̂² channel failed Gate B on single-seed-majority data, so the shipping calibrator relies on direct `α̂` prediction + `±0.05` clamp rather than runtime variance shrinkage.
- **Tiled inference** — v4 controller no-ops; scalar still works. The tile-merge boundary makes single-tile `c_pool` / `g_obs` ill-defined.
- **CFG drift** — v4 calibrated at CFG=4 only. Other CFGs fall back to scalar (proposal §"Risks" #7).
- **Cached-Spectrum `x0_pred`** is biased by Chebyshev forecaster error. Empirically should still help (correction is bias-agnostic) but worth one explicit ablation row.
- **Sign-flip vs the paper** unresolved — the speculative mechanisms are preserved in the [archived findings](../../_archive/docs/findings/dcw-bias-findings-202605.md); the cleanest test (smaller / pixel-space DiT) is out of scope.

## Related code

| File | Role |
|---|---|
| `networks/dcw.py` | `apply_dcw` (the apply site, shared by both modes) + `FusionHead` (shared by trainer + inference) + `haar_LL_norm` |
| `library/inference/corrections/dcw_calibrator.py` | `OnlineDCWCalibrator` — loads artifact, observes warmup, fires head at step `k_warmup`, emits per-step `λ_i` |
| `library/inference/generation.py` | controller setup pre-loop + per-step apply at the DCW call site (non-tiled path) |
| `scripts/dcw/measure_bias.py` | offline trajectory dump + S_pop sweep — produces `gaps_per_sample.npz` consumed by the trainer |
| `scripts/dcw/train_fusion_head.py` | offline head training — produces `fusion_head.safetensors` |
| `scripts/tasks/dcw.py` | `python tasks.py dcw` / `python tasks.py dcw-train` task wrappers |
| `docs/methods/dcw.md` | Current calibrator derivation, gates, fallback ladder, evidence appendix |
