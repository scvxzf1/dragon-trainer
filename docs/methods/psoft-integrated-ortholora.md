# OrthoLoRA: Cayley Parameterization + SVD-Informed Init

状态：稳定
适用版本：当前 main

OrthoLoRA parameterization that replaces soft orthogonality regularization with a hard Cayley constraint and initializes bases from the pretrained weight's SVD. Gated behind `use_ortho = true`.

Inspired by PSOFT (Wu et al., "Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation", ICLR 2026).

> **For the structural walkthrough** (frozen SVD bases, Cayley transform derivation, forward pass, why exact inverse beats Neumann, expressiveness trade-off), see **`docs/structure/ortholora.md`**. This doc is the usage / ops / benchmarking reference.

## Usage

In any method config (e.g. `configs/methods/lora.toml` or `configs/gui-methods/ortholora.toml`):

```toml
use_ortho = true
network_alpha = 64
network_dim = 64
```

Keep `alpha = dim` for neutral scaling.

For self-contained OrthoLoRA or timestep-masked OrthoLoRA variants, run:

```bash
python tasks.py lora-gui ortholora          # OrthoLoRA only
python tasks.py lora-gui tlora              # OrthoLoRA + T-LoRA
# Compatibility alias: make lora-gui GUI_PRESETS=tlora
```

## Trainable parameter count

Per module at rank r, dim d:

| Component | Unconstrained Basis OrthoLoRA | Cayley OrthoLoRA (`use_ortho = true`) |
|-----------|-------------------------------|---------------------------------------|
| Trainable | r(d_in + d_out) + r | 2r² + r |
| Frozen buffers | r(d_in + d_out) + r | r(d_in + d_out) |
| Optimizer states | ~ trainable | ~ trainable |

At r=64, d=3072: trainable params drop from ~394K to ~8.3K per module. Frozen buffer size is similar. Optimizer memory is significantly reduced.

## VRAM overhead

Compared to standard LoRA at the same rank, Cayley OrthoLoRA uses slightly more activation memory (~150–200 MB at r=64 across 196 DiT modules) because `P_eff` and `Q_eff` are computed tensors saved in the autograd graph, not leaf parameters. Optimizer memory is lower (fewer trainable params). Net effect is modest.

## Save format

The native network state contains `S_p`, `S_q`, `lambda_layer`, `P_basis` and `Q_basis`. Enable `save_state = true` and resume from the saved state directory to restore this parameterization together with optimizer, scheduler and RNG state.

The exported adapter `.safetensors` instead converts those keys to standard LoRA (`lora_up.weight`, `lora_down.weight`) for compatible inference loaders. It preserves the rank-r update algebraically, without another SVD, but BF16 factorization and rounding are not bitwise identical to the native forward. This distilled file does **not** preserve the original Cayley parameters or optimizer state and must not be presented as an equivalent OrthoLoRA training resume.

## NF4 and runtime boundaries

- SVD initialization uses the logical dequantized weight, never the packed `Params4bit.data` shape. The temporary dense matrix is released after each module; the NF4 base remains unchanged.
- Rank must satisfy `1 <= network_dim <= min(in_features, out_features)` and is checked before initialization. SVD runs on the base weight's device, without silently selecting the default CUDA device. CPU-loaded models therefore perform CPU SVD, which can make initialization noticeably slower.
- Cayley solves run in FP32 even after adapter BF16 conversion. Bases and the activation path retain their configured compute dtype.
- `network.set_enabled(False)` or a zero multiplier disables the adapter branch without changing base weights.
- Krea-2's family registry permits network variant selection, including OrthoLoRA, though selection does not guarantee full validation across all combinations: OrthoLoRA on Krea-2 NF4 has bounded experimental verification rather than general production or quality certification. See [the repair report](../findings/krea2_nf4_adapter_repair_20260905.md) for numerical, export/reload and bounded hardware evidence.

## What to compare

When benchmarking against plain LoRA or soft-regularized OrthoLoRA:

1. **Training loss curves** — does Cayley plateau higher? (subspace restriction)
2. **Generated image quality** — the real test for creative concept learning
3. **Orthogonality** — verify `P_eff^T @ P_eff ~ I` (should be exact, by construction)

## Differences from the PSOFT paper

| Aspect | PSOFT | Our implementation |
|--------|-------|-------------------|
| Bases | Frozen A', B' from symmetric SVD decomposition | Frozen P_basis, Q_basis from standard SVD |
| Rotation | Single r × r Cayley matrix R | Two r × r Cayley matrices (S_p, S_q) |
| Relaxation | Tunable alpha, beta vectors (2r params) | Tunable lambda vector (r params) |
| Inverse | Neumann series (K=5) | Exact solve (r × r is cheap) |
| Residual | Frozen W_res preserved | No residual — zero-init lambda instead |
| Target | NLP/CV classification | Creative visual concept learning (LoRA) |

## Files

| File | What changed |
|------|-------------|
| `networks/lora_modules/ortho.py` | `OrthoLoRAModule` class |
| `networks/lora_anima/` | Config parsing, module selection, save conversion |
| `configs/methods/lora.toml` | `use_ortho` toggle |
