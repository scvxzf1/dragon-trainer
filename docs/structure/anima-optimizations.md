# Anima performance & compile optimizations

状态：稳定
适用版本：当前 main

A tour of the non-obvious decisions the codebase makes to run fast on consumer GPUs. Four themes, each ending with the *why* — most of these look strange until you see what breaks when they're absent.

1. **QKV fusion** — fewer, wider GEMMs.
2. **FP32 accumulation in the right places** — bf16 for storage, fp32 for the reductions that bf16 would wreck.
3. **Constant-token bucketing (4032 / 4200 families)** — a tiny fixed set of token counts with no padding to leak into attention, paired with union dynamic-seq compilation.
4. **Compile-friendly code polish** — the dozen little rules that keep dynamo's guard cache from evicting.

![Anima performance & compile optimizations](../structure_images/optimization.png)

---

## 1. QKV fusion

### Self-attention: one fused GEMM

`Attention.__init__` (`library/anima/models.py`) — for a self-attention module with $d_\text{in} = d_\text{out} = 2048$, three `Linear(2048 → 2048)` projections would issue three separate GEMMs:

$$
Q = W_Q x,\quad K = W_K x,\quad V = W_V x
$$

Anima instead stacks the three projections into one weight $W_{QKV} \in \mathbb{R}^{6144 \times 2048}$ and fires a single matmul:

$$
\begin{bmatrix} Q \\ K \\ V \end{bmatrix} \;=\; W_{QKV}\,x, \qquad
W_{QKV} = \begin{bmatrix} W_Q \\ W_K \\ W_V \end{bmatrix}
$$

Split happens post-matmul on the feature axis (`Attention.compute_qkv` in `library/anima/models.py`):

```python
qkv = self.qkv_proj(x)                                                  # (..., 6144)
q, k, v = qkv.unflatten(-1, (3, self.n_heads, self.head_dim)).unbind(-3)  # three (..., 16, 128)
```

Why this is a win:

- **Arithmetic intensity.** One `[6144 × 2048]` GEMM has roughly the same FLOPs as three `[2048 × 2048]` GEMMs but fetches the input `x` from HBM only once instead of three times. On bf16 with large batch-seq, those reads dominate.
- **Kernel launch overhead.** Three GEMM launches vs. one — matters at short sequences and during compile tracing (fewer nodes in the graph).
- **Fused bias / norm friendliness.** `unflatten + unbind` is a pure view, so the subsequent `q_norm / k_norm / RoPE` operate on views of the same contiguous buffer.

### Cross-attention: KV fused, Q separate

Cross-attention reads $x \in \mathbb{R}^{2048}$ for Q and a *different* context $c \in \mathbb{R}^{1024}$ for K, V. You can't fuse Q with KV — different input dims, different matmuls. Anima fuses only what's fusable (`Attention` in `library/anima/models.py`):

$$
Q = W_Q\,x \in \mathbb{R}^{2048}, \qquad
\begin{bmatrix} K \\ V \end{bmatrix} = W_{KV}\,c \in \mathbb{R}^{4096}
$$

Split is symmetric (`Attention.compute_qkv`):

```python
q  = self.q_proj(x).unflatten(-1, (n_heads, head_dim))
kv = self.kv_proj(context)
k, v = kv.unflatten(-1, (2, n_heads, head_dim)).unbind(-3)
```

### AdaLN heads: one Linear → three modulations

The same trick on the modulation side. Each sub-layer needs `(shift, scale, gate)`, a triple of `D`-vectors. Instead of three separate `Linear`s per sub-layer, each `adaln_up_*` produces `3D` at once and splits via `.chunk(3, dim=-1)` (`Block._forward` in `library/anima/models.py`):

$$
[b_\star,\,s_\star,\,g_\star]\ =\ \text{adaln\_up}_\star(d_\star) + \text{adaln\_lora}_\star
\quad\in\mathbb{R}^{3D}
$$

With `adaLN-LoRA` (`use_adaln_lora = True`, `adaln_lora_dim = 256` across all shipped Anima weights), one fused down-proj `adaln_fused_down` (`SiLU → Linear(D → 3·256)`) produces the 256-dim bottleneck for all three sub-layers at once, and only the three `256 → 6144` up-projs remain per-sub-layer.

---

## 2. FP32 accumulation

Bf16 has 8 mantissa bits. That's fine for *storing* weights and activations, but it's catastrophic for long reductions — summing thousands of bf16 products accumulates rounding error proportional to $\sqrt{N} \cdot 2^{-8}$. Anima promotes to fp32 in the exact three places where this bites.

### 2.1 LoRA bottleneck matmuls

`networks/lora_modules/custom_autograd.py` (`lora_down_project`) and `networks/lora_modules/base.py`. Modules store weights in bf16 (or fp32 master when `lora_fp32_compute` is active) and run critical bottleneck reductions in fp32 while saving compact bf16 inputs on the autograd tape:

```python
lx = lora_down_project(x, self.lora_down.weight, inv_scale)    # fp32 matmul, saves bf16 x
...
return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)
```

The down-proj sums across up to $d_\text{in} = 8192$ (e.g. `mlp.layer2`). That's ~13 mantissa bits of noise floor — the LoRA delta is small by construction, so noise of that magnitude would swamp the signal. Fp32 accumulation adds an ignorable amount of compute (LoRA is ~0.1% of params) and rescues the gradient signal.

### 2.2 RMSNorm

`RMSNorm` (`library/anima/models.py`) — every norm upcasts before computing:

```python
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    output = self._norm(x.float())              # ← fp32 variance
    return (output * self.weight).to(x.dtype)   # ← cast back
```

$\text{mean}(x^2)$ at `D = 2048` is another long reduction — bf16 can over/underflow the squared intermediate when any channel is large. The cast back happens after `rsqrt`, so the rest of the block sees bf16.

### 2.3 Loss & sigma weighting

In `library/training/` and `library/anima/training.py`, $\sigma$ weighting is computed in fp32 (`weighting = (sigmas**-2.0).float()` in `library/runtime/noise.py`) and guidance deltas for CFG are upcast before subtraction. Both are low-volume pointwise ops where fp32 is free.

### The rule

Upcast to fp32 **exactly at reductions** — the dot products inside LoRA, the sum-of-squares inside RMSNorm. Leave bf16 everywhere else so HBM bandwidth stays halved.

---

## 3. Constant-token bucketing (4032 / 4200 families)

### The problem

Bucketed training allows images of different aspect ratios: `512×768`, `768×512`, `640×640`, etc. After `PatchEmbed` with patch size 16, each bucket produces a different sequence length:

$$
L_\text{bucket} = \frac{H}{16}\cdot\frac{W}{16}
$$

A naive implementation lets this shape propagate through the DiT. Every distinct sequence length then triggers `torch.compile` to retrace and recompile — and with 28 blocks × many buckets × 2 `requires_grad` states, you blow past dynamo's recompile limit and fall back to eager. Losing the compiled path is a ~2× regression.

### The fix: collapse to a few exact token counts, run them natively

`CONSTANT_TOKEN_BUCKETS` (`library/datasets/buckets.py`) is **two token-count families — 4032 (= 63·64) and 4200 (= 60·70)**. Every bucket resolution *exactly* fills its family's count, so there is **zero intra-bucket padding by construction**. Native shapes are the only mode: every forward runs at its real token count, so dynamo guards only on the token count — and the whole table collapses to **two** distinct counts → two compiled block graphs.

When `compile_blocks` is active, `library/anima/models.py`'s forward flattens `(B, T, H, W, D)` into a *fake-5D* `(B, 1, seq_len, 1, D)` tensor the block code already knows how to consume:

```python
B, T, H, W, D = x.shape
seq_len = T * H * W

x = x.flatten(1, 3)              # (B, seq_len, D)
x = x.unsqueeze(1).unsqueeze(3)  # (B, 1, seq_len, 1, D)
```

The fake-5D reshape is what makes the block stack key on token count alone rather than guarding `H` and `W` separately (which would recompile *per resolution*, 24 graphs instead of 2):

> The fake-5D shape `(B, 1, seq_len, 1, D)` is compatible with existing Block code because `rearrange("b t h w d -> b (t h w) d")` with `t=1, w=1` produces the same flat sequential order as the original.

Eager (uncompiled) forwards leave `_native_flatten = False` and skip the reshape entirely — bit-exact, slightly cheaper.

### Why not just pad everything to one shape?

The original design did exactly that — pad every bucket up to a single static target for *one* graph (`set_static_token_count(count, pad=True)` + `compile_core`). That path was **removed 2026-05-24**, for two reasons:

- **It can't run this table.** The 4200 family exceeds the legacy 4096 cap and would truncate.
- **The padding isn't free.** Under `attn_mode="flash"` (no padding mask) the zero-padded tokens are *not* harmless attention sinks — AdaLN shift + Q/K/V bias leak them into the real-token output (up to ~6.5% rel-L2 on the 4032-token buckets). Native shapes have no pad to leak and match the no-pad ground truth bit-for-bit.

### Cross-attention side: full-length KV

Cross-attention KV length (the text sequence) is fixed: the text encoder output is zero-padded to 512 tokens, so the cross-attn path is already shape-stable with no trimming needed. The padding tail acts as attention sinks, which the pretrained model expects.

There used to be a flash4-only **bucketed KV trim** here — sequences trimmed to one of `(128, 192, 256, 512)` (`_KV_BUCKETS`) with a sigmoid-based **LSE correction** in `networks/attention_dispatch.py` (via `crossattn_full_len`) re-adding the lost zero-key sinks. That path was bundled with FA4 and removed with it (2026-05-20) — `_KV_BUCKETS`, `crossattn_full_len`, the `trim_crossattn_kv` flag, and the trim block are all gone. Cross-attention now always runs full 512-length KV under FA2. See `docs/optimizations/fa4.md` for the postmortem.

---

## 4. Code polish for `torch.compile`

`configs/base.toml` sets both `torch_compile = true` and **`compile_dynamic_seq = true`** by default. `Anima.compile_blocks` (`library/anima/models.py`) compiles each resident block's `_forward` individually:

```python
def compile_blocks(self, backend="inductor", ..., dynamic_seq=False, ...):
    for block in target_blocks:
        compiled = torch.compile(block._forward, backend=backend, mode=mode, dynamic=False)
        if dynamic_seq:
            block._forward = _make_dynamic_seq_wrapper(compiled, seq_range=...)
        else:
            block._forward = compiled
```

Notice how **union dynamic-seq** (`compile_dynamic_seq = true`, Anima's production default) works: `torch.compile(..., dynamic=False)` is still passed on the inner `_forward` so non-sequence axes stay static, while an eager outer wrapper calls `torch._dynamo.mark_dynamic` on **only** the native-flattened sequence axis (`dim=2` of `(B, 1, seq_len, 1, D)`) over the union `[min_seq, max_seq]` range. Because that wrapper sits *inside* the checkpointing boundary, the dynamic-sequence marks are re-applied during backward recompute as well as forward, letting one compiled block graph serve the entire bucket union instead of recompiling per token-count family. (Setting `compile_dynamic_seq = false` falls back to static per-token-family graphs; when `blocks_to_swap > 0`, `compile_block_scope = "resident"` compiles only the GPU-resident head blocks by default so swapped tail blocks do not trigger device-guard recompiles.)

Why `_forward` and not `forward`:

> Compiles `_forward` (the actual attention/MLP computation) rather than `forward` (the checkpointing wrapper). This is critical because `unsloth_checkpoint` has `@torch._disable_dynamo`, which causes an immediate graph break if `forward` itself is compiled.

That's a load-bearing two-line change. If `forward` is the compile target, dynamo hits the disable decorator, emits a graph break, and compiles essentially nothing while still paying the full guard-check cost per step.

Five more rules the code follows:

### 4.1 Don't pre-compile flex_attention

`networks/attention_dispatch.py`:

```python
# Do NOT pre-compile flex_attention here. When blocks are individually
# compiled (compile_blocks / native-flatten mode), the outer torch.compile
# already traces into _flex_attention and fuses it.
# Pre-compiling causes nested compilation which exhausts dynamo's
# recompile limit (grad_mode guard × mask variants) and falls back to
# the slow unfused path.
compiled_flex_attention = _flex_attention
```

Nested compilation is a pit trap — dynamo tries to compile from the outside and hits an already-compiled callable inside, guards disagree, it gives up.

### 4.2 Kill Python dict caches inside compiled code

RoPE computation in `library/anima/models.py` — the Python dict cache is *skipped* when tracing:

```python
_compiling = torch.compiler.is_compiling()
if not _compiling:
    cached = self._cos_sin_cache.get(key)
    if cached is not None:
        return cached
```

Why: dict mutations are dynamo guard failures. Skipping the Python dict cache under trace avoids mid-training guard invalidations.

### 4.3 Normalize `requires_grad` once per step

`Anima.forward` (`library/anima/models.py`):

```python
# All blocks share the same compiled _forward, so a mismatch triggers
# dynamo recompilation.
if self.training:
    x_B_T_H_W_D = x_B_T_H_W_D.requires_grad_()
```

Block 0 sees a frozen patch-embed output (`requires_grad=False`); blocks 1+ see a LoRA-activated tensor. If the compiled `_forward` is reused across blocks, the grad-mode guard differs on block 0 and triggers a second compile. A single `requires_grad_()` up front unifies the guard.

### 4.4 Vectorize lookups — no `.item()` host syncs

`library/runtime/noise.py` shows the pattern:

```python
# a single broadcast-equality + argmax finds the right index per batch
# element without per-element .item() host syncs.
eq = schedule_timesteps.unsqueeze(0) == timesteps.unsqueeze(1)   # [B, N]
step_indices = eq.to(torch.int8).argmax(dim=-1)                  # [B]
sigma = sigmas[step_indices].flatten()
```

A `.item()` call forces a GPU→CPU sync and stalls the pipeline. Vectorizing into broadcast ops keeps everything on-device and lets dynamo trace a single graph.

### 4.5 Keep in-place / control-flow simple

Reading `Block._forward` and `Anima.forward_mini_train_dit` in `library/anima/models.py`, the patterns are:

- no `if x.shape[0] > 1: …` data-dependent branches inside the hot path;
- no Python-side scalar extraction (`.item()`, `.tolist()`);
- optional features (ReFT, mod-guidance, soft tokens) are gated by `is not None` checks at the top of the function, where dynamo can specialize the trace once based on the module's attribute state.

---

## Putting it together

| Optimization              | What it saves                              | Without it                                    |
| ------------------------- | ------------------------------------------ | --------------------------------------------- |
| QKV + KV fusion           | 2× fewer GEMMs, 3× fewer HBM reads on x    | Three small kernels per attention sub-layer   |
| Fp32 at reductions        | Gradient precision for LoRA / norm stats   | LoRA stalls, norms drift at D=2048            |
| Native 4032/4200 buckets + union `compile_dynamic_seq` | Single union dynamic-seq block graph (or 2 static graphs), no pad leak | Recompile storm, fallback to eager |
| `_forward` compile target | Real fusion past `unsloth_checkpoint`      | Graph break, guards still checked every step  |
| Tensor (not int) seq-len  | Symbolic tracking across buckets           | Per-bucket recompile                          |
| No dict cache under trace | Stable guards                              | Cache-miss guard invalidation mid-training    |
| Unified `requires_grad`   | One compile for resident blocks            | Block 0 vs. 1+ split cache                    |
| Vectorized sigma lookup   | No host sync                               | CPU ↔ GPU pipeline stall per step             |

The theme: **keep non-sequence shapes static, mark only the flattened sequence axis dynamic when needed, and give dynamo one dtype, one code path, and one grad mode.** Everything above is a different angle on that rule.
