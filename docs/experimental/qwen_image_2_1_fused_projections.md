# Qwen Image 2.1 frozen projection packing (experimental)

状态：实验（默认关闭，显式 opt-in）

The training-only `qwen_fused_projections` option defaults to `off`. It accepts
`mlp`, `qkv`, or `all` in a training TOML or as `--qwen_fused_projections all`
on the CLI. It is not a default recommendation and has no WebUI control.

This experiment requires Qwen Image 2.1, BF16 base compute and swap transfer,
full gradient checkpointing, and plain attention-only LoRA. It rejects low-bit
transfer, checkpoint offload, selective checkpointing, and unsupported adapter
targets before converting the model. Both conversions run after LoRA apply/load
and before block-swap CPU-master capture or block compilation. CPU-loaded base
weights may still have `requires_grad=True` at that point, so the installer
explicitly freezes converted weights.

With block swap, the transformer initially loads on CPU; without it, the
transformer loads on the accelerator. Packing temporarily materializes new
weights alongside the originals and can raise peak host memory or GPU memory.
Independent CPU numerical/regression checks and a small real 90HX swap/Inductor
probe pass. Full-size BF16 + swap24 + compile training with `all`, one retained
block and a 512 MiB payload budget completed four steps, with 5.66 GiB peak
allocated and 7.80 GiB peak reserved. State loading and execution also pass.
Default computation showed a resumed-gradient discrepancy; deterministic
diagnostics eliminated it. A subsequent complete six-step uninterrupted versus
step-four-resumed comparison matched final model/optimizer/scheduler state and
the last two losses exactly, with independent verification. This does not
promise bitwise default execution. Three alternating rounds (12 runs, 400 total steps, including four 60-step
runs) completed on the 90HX with BF16/full checkpoint/swap24/compile. Retaining
four blocks after packing reduced mean step time by 2.22–2.24% with torch and
0.56–1.22% with Flash, while adding about 1.76–1.82 GiB peak allocated versus
the unfused baseline. This falls below the approximately 5% adoption target.
Production adapter writer/reader round-trips into fresh split and packed models
also passed independent CPU output/gradient checks. Keep the option off for
ordinary runs; see the
[development acceptance record](../findings/qwen_image_2_1_stage3_acceptance_20260928.md).

## Saved projection experiment (stage 3C)

`qwen_saved_projection_blocks=N` selects the final N logical transformer
blocks. It requires `qwen_fused_projections="all"` and a positive
`qwen_projection_budget_mib`; both new options default to zero. The selected
blocks retain packed QKV (including the three existing LoRA branches) and
packed gate/up results outside checkpoint. Their attention and MLP tails each
use a non-reentrant checkpoint; unselected blocks keep full checkpointing.
No-grad inference, including KV-cache and segment paths, uses the original
fused block forward. Block identities and swap backward hooks remain intact.
For no-swap training, install the selected-block wrapper after the final
`enable_gradient_checkpointing()` call. Calling that method again later may
replace the wrapper on `_gradient_checkpointing_func`, silently restoring full
checkpointing for selected blocks. Reinstalling or restarting checkpointing
after projection setup is not supported by this experiment.

The budget checks only raw QKV plus gate/up outputs, using the actual batch,
sequence, projection widths and dtype before those large projections run. At
B=1, L=4107, H=4096, MLP ratio 3 and BF16, that payload is 288.78 MiB per
selected block. Additional inputs, LoRA activations, autograd metadata,
attention workspace and allocator fragmentation are not covered, so this is
not an OOM guarantee. Start with one block and measure peak memory. With
`torch_compile`, selected blocks compile the two projection and two tail
segments, not an enclosing whole-block graph; swap and budget dispatch stay
outside compiled regions. Full-size compiled traces confirm QKV and gate/up forward recomputation each
falls from 32 to 28 dispatches with four retained blocks; their dX counts stay
at 32. The measured end-to-end benefit remains small, so this is an explicitly
opt-in experiment rather than a default recommendation.
