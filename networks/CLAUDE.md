# `networks/` development rules

This file adds network-specific constraints to the repository-wide [`AGENTS.md`](../AGENTS.md). Read both before changing adapters, routing, attention dispatch, persistence, or merge behavior. Live code and tests override historical design notes.

## Scope and entry points

| Area | Primary entry |
| --- | --- |
| Registry and method resolution | [`__init__.py`](__init__.py), [`registry.py`](registry.py), [`registry_api.py`](registry_api.py), [`core_specs.py`](core_specs.py), plugin registration |
| LoRA-family config and lifecycle | [`lora_anima/config.py`](lora_anima/config.py), [`lora_anima/factory.py`](lora_anima/factory.py), [`lora_anima/network.py`](lora_anima/network.py) |
| Module construction | [`lora_anima/builders.py`](lora_anima/builders.py), [`lora_anima/module_builders.py`](lora_anima/module_builders.py) |
| LoRA variants | [`lora_modules/`](lora_modules/) |
| Non-LoRA methods | [`methods/`](methods/), [`plugins/`](plugins/) |
| Save/load and fused projections | [`lora_save.py`](lora_save.py), [`lora_anima/loading.py`](lora_anima/loading.py), [`attn_fuse.py`](attn_fuse.py) |
| Attention backends | [`attention_dispatch.py`](attention_dispatch.py) |

Do not maintain a complete variant inventory here. Use the [stable method index](../docs/methods/README.md), [experimental index](../docs/experimental/README.md), live registry, and `configs/` for current availability.

## LoRA-family routing contract

Routing is expressed only through three config axes:

| Key | Supported values |
| --- | --- |
| `use_moe_style` | `False`, `"shared_A"`, `"independent_A"` |
| `route_per_layer` | `True`, `False` |
| `router_source` | `"none"`, `"input"`, `"sigma"`, `"fei"`, `"crossattn_emb"` |

- Parse and validate the axes in `LoRANetworkCfg.from_kwargs`; resolve implementations through the network registry. Do not infer a new method from checkpoint metadata.
- `router_source="input"` requires layer-local routing. Network-level FEI, sigma, or pooled-text routing shares one router result across routing-aware modules.
- `router_source="crossattn_emb"` uses pooled post-adapter text features and must preserve cond/uncond branch handling in both training and inference.
- Retired stamps such as `ss_use_hydra` and `ss_use_fei_router` are not compatibility fallbacks. Current checkpoints stamp `ss_use_moe_style`, `ss_route_per_layer`, and `ss_router_source`.
- `use_ortho` remains an orthogonal module capability, not a fourth routing axis.

## Routing lifecycle

- For network-level FEI routing, compute FEI/router output once per denoising step with `network.set_fei(z_t)` before the adapted forward, then share the resulting weights by reference.
- Training and inference must update the same conditioning source at equivalent step boundaries. Check `library/training/router_conditioning.py`, `library/runtime/fei.py`, and `library/inference/generation.py` when changing this lifecycle.
- A zero-initialized global-router output is intentional. Do not change initialization, warmup, temperature, or normalization without an invariant test and routing-stat evidence.
- Never thread timestep or routing tensors through every Linear call when an existing shared-buffer/reference mechanism applies.

## Apply, compile, save, and load

- Apply the adapter and load adapter weights before `torch.compile`; otherwise the compiled graph can bypass monkey-patched forwards. Reuse the runtime harness instead of open-coding the order.
- [`attn_fuse.py`](attn_fuse.py) is the single source of truth for runtime fused `qkv_proj` / `kv_proj` and on-disk split `q/k/v_proj`. Save and load must use the same specs.
- Variant-specific transformation belongs beside the variant that owns the math. Keep `lora_save.py` as an orchestrator rather than adding another variant implementation there.
- Adding a new checkpoint layout requires save/load round-trip coverage, metadata coverage, and an explicit legacy-compatibility or refusal decision.
- Only adapters reducible to DiT Linear deltas may use the merge path. Non-foldable methods must fail clearly rather than silently dropping state.

## Attention and timestep invariants

- All attention call sites go through `dispatch_attention()` unless a documented method-specific reason requires otherwise.
- Layout is backend-dependent: SDPA/sageattn commonly use BHLD; xformers/flash-attn commonly use BLHD. State the incoming layout and test any new transpose path.
- Do not revive removed FA4/KV-trim behavior from historical notes. Current status and re-enable requirements live in [`docs/optimizations/fa4.md`](../docs/optimizations/fa4.md).
- T-LoRA uses one shared `_timestep_mask`, updated once per denoising step. New timestep-aware variants must reuse the buffer pattern in `lora_modules/base.py` and the setters in `lora_anima/factory.py` / `network.py`.
- Inference and merged weights run full rank unless a method explicitly implements and documents runtime masking.

## Change checklist

- Update registry/config/schema and reject unsupported combinations early.
- Check training construction, inference loading, save/load metadata, merge behavior, and optimizer parameter groups.
- Add focused tests under [`../tests/`](../tests/) for construction, numerics, lifecycle, metadata, and round trips as applicable.
- Update the stable or experimental method index instead of expanding this file with method history.
- If live `library/` or `networks/` changes affect ComfyUI nodes, run or report the need for `python tasks.py vendor-sync`; never patch `_vendor/` as an independent source.
