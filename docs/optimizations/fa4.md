# Flash Attention 4 — removed

状态：历史 / 已移除能力说明

This doc records why `attn_mode = "flash4"` and `trim_crossattn_kv` were removed from the training pipeline. The short version: **FA4 ran slower than FA2 on our targets**, and the cross-attention KV-trim trick that justified FA4's ergonomic complexity only worked under FA4, so it went with it.

## What was there

Two bundled features that shared a single dependency on FA4's `return_lse=True` path:

1. **FA4 forward kernel.** `flash_attn.cute.flash_attn_func` from a patched SM120 fork (`sorryhyun/flash-attention-sm120-fix`, based on `sisgrad`'s `dz/sm120_tma_optimized` branch). The stock `flash-attn` v4 does not ship SM120 (consumer Blackwell) kernels, so we pulled in a fork to get FA4 running on RTX 50-series cards at all.
2. **Cross-attention KV trim (`trim_crossattn_kv`).** Qwen3 text encoder outputs are zero-padded to 512 tokens, but real captions only fill 30–80. The trim path sliced the padded KV down to a bucketed length (`[64, 128, 256, 512]`), ran attention on the trimmed tensor, and then used FA4's returned log-sum-exp to apply a closed-form correction that rebuilt the contribution of the removed zero keys (they act as attention sinks in the pretrained softmax — dropping them without correction produces black images):

   ```
   out_corrected = out_trimmed * sigmoid(lse - log(N_pad))
   ```

   The correction is exact, not an approximation. It relied on FA4's LSE return, which FA2's Python interface doesn't expose, so the trim path was gated on `attn_mode == "flash4"`.

## Why it was removed

### FA4 was slower than FA2 in practice

On a single RTX 5060 Ti 16 GB with LoRA rank 32, batch 2, 182 steps, FA4 was effectively a wash with FA2 — and often a regression once compile/static-token effects were accounted for. Representative numbers from the old benchmark table (before removal):

| Configuration | 2nd Epoch |
|---|---|
| FA2 + compile (static tokens) | 5:01 |
| FA4 + compile (static tokens) | 5:15 |

The expected advantage from FA4 (the fused CUTLASS/TVM kernel) did not materialize on consumer Blackwell. A few things contributed:

- **SM120 kernel maturity.** The SM120 port is a community fork, not a tuned upstream kernel. The TMA-optimized path worked, but it wasn't meaningfully ahead of FA2 in forward throughput at our shapes (batch 2, 4096 image tokens, 64–512 text tokens).
- **Compile interaction.** FA4's CUTLASS kernel accesses raw data pointers via DLPack, which fake-tensor tracing cannot see through. We had to wrap the kernel in `@torch.compiler.disable` and take graph breaks around every attention call. The surrounding fused region still got compiled, but the graph breaks added up across 32 blocks × 2 attention calls per block × every step.
- **Shape-dependent branching.** The KV trim path had a separate code branch for the LSE correction, which caused torch.compile recompilations at bucketed KV shapes (64/128/256/512) until we flattened the branch. Even after the fix, the compile graph was more fragile than the FA2 path.

The KV trim itself *did* reduce FLOPs — roughly 4× less cross-attention compute on short captions. But cross-attention is a small fraction of total compute (the 4096-token self-attention dominates), so the end-to-end win was small, and what little win there was got eaten by the slower FA4 forward and the compile friction.

### The trim was coupled to FA4 and couldn't be salvaged

We could in principle port the LSE-sigmoid correction to FA2 by monkey-patching `_flash_attn_forward` to return LSE. We chose not to:

- FA2's Python interface doesn't return LSE, and adding it means poking at the C++ bindings.
- Without FA4, the remaining speedup is tiny (cross-attention isn't the bottleneck).
- The trim path complicates every caller (`train.py` has to compute `max_crossattn_seqlen`, pass `crossattn_seqlens`, and handle postfix/prefix offset accounting) for a win we can no longer measure.

Keeping it as a flag that everyone sets to `true` in configs, with an opaque code path that depends on a dead backend, wasn't worth the maintenance cost.

## What's left

- **Default attention on Anima / Krea-2 / Z-Image is FA2** (`attn_mode = "flash"` in `configs/base.toml`), via the upstream `flash-attn` 2.x wheel. Note that family method configs override this where needed: Qwen Image 2.1 defaults to `attn_mode = "torch"` (`flash` / `sdpa` also supported under BF16), and MOSS-Image supports only `torch` / `sdpa` (no `flash`).
- **Other Anima backends still work:** `torch` (SDPA), `flex` (PyTorch flex attention), `sageattn`, `xformers`. None of them use trimming; Anima runs the full 512-length cross-attention KV with attention-sink padding intact.
- **`trim_crossattn_kv` is gone.** The config flag, CLI arg, the caller plumbing (`max_crossattn_seqlen` / `crossattn_full_len` / `_KV_BUCKETS`), and the commented-out trim block in `library/anima/models.py` were all deleted (2026-05-20). `crossattn_seqlens` survives only as the per-sample text length consumed by the `attn_mode="flex"` BlockMask path — note that the flex mask *masks out* padding instead of treating it as a sink, which regresses quality, so it is not a substitute for the removed trim.

## If you want to bring FA4 back

The Anima dispatch stubs remain in tree, so re-enabling for local experiments follows these locations:

1. **Dependency.** Add an appropriate `flash-attn` v4 / SM120 wheel or fork dependency in `pyproject.toml`.
2. **Attention dispatch.** Restore the FA4 import block and the `flash4` branch in `networks/attention_dispatch.py` (currently guarded by a disabled stub).
3. **Train path.** In `library/training/model_loading.py`, the `attn_mode == "flash4"` guard currently raises `NotImplementedError`; replace that with a check against `_flash_attn_4_func_raw is not None` (and update `library/training/compat_matrix.py` if testing under non-BF16 modes).
4. **KV trim (optional).** The trim plumbing was deleted in 2026-05, so it is no longer a matter of uncommenting a block — you'd reintroduce it from scratch: a `trim_crossattn_kv` flag + CLI arg, the per-sample `crossattn_seqlens` / `max_crossattn_seqlen` pass-through in `library/training/forward/{text_conds,forward_kwargs}.py`, a `crossattn_full_len` field on `AttentionParams`, and the slice + LSE-sigmoid correction gated on `attn_mode == "flash4"` in `library/anima/models.py`. Given the measured win was tiny, reviving FA4 without the trim (full 512-length KV) is the saner default. See the pre-2026-05 git history for the original trim implementation.
5. **WebUI / configs.** Add `"flash4"` to the WebUI attention-mode catalog and the `--attn_mode` choices in `inference.py` and the training CLI surface. Any FA4 VRAM presets should live in the normal preset/config trees rather than a frontend-only registry.

Before doing any of that, benchmark FA4 on your own hardware. If the kernel has matured upstream — or if SM120 TMA has landed in stock `flash-attn` — it may be worth reviving. On our targets in 2026-04 it wasn't.
