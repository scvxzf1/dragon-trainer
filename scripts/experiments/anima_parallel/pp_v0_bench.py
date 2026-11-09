"""Bounded real-GPU measurement for Anima PP v0.

Two modes, both driving the ``library.training.pp`` runtime:

* ``pp``     -- 2 ranks, Schedule1F1B, stage-local plain-MLP LoRA, fp16 +
                GradScaler, ``microbatch_size=1`` (so the global batch equals
                ``--microbatches``), single 4200-token bucket.
* ``single`` -- 1 rank, whole-model reference at the same global batch, used for
                the throughput ratio and the fp16 numerics reference.

This is a measurement harness, not a production entry point.  It writes one JSON
per rank plus (optionally) the step-0 adapter gradients, so the fp16 cosine /
relative-norm comparison can be computed offline.

Numbers produced with this script are recorded in
``docs/experimental/anima_pipeline_parallel_v0.md`` (v0 acceptance) and
``docs/findings/anima_pp_microbatch_scan_20261109.md`` (M=4/8/16/32 sweep).

Run (one process per rank; PP v0 needs exactly two ranks):

    for R in 0 1; do
      CUDA_VISIBLE_DEVICES=0,1 RANK=$R LOCAL_RANK=$R WORLD_SIZE=2 \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=29561 \
      .venv/bin/python -m scripts.experiments.anima_parallel.pp_v0_bench \
        --mode pp --output-dir <dir> --microbatches 16 --steps 2 --warmup-steps 1 \
        --dit-path <model.safetensors> --latent-path <latent.npz> --text-path <te.safetensors> &
    done
    wait

Note ``CUDA_VISIBLE_DEVICES=0,1`` -- on this host CUDA ordinals and nvidia-smi
indices differ (nvidia-smi 1,2 are the two V100s; CUDA 0,1 are).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.experiments.anima_parallel.common import (
    create_mlp_lora,
    load_cached_batches,
    load_model,
)

# Single 4200-token bucket: 150x112 latent with patch 2x2 -> 4200 tokens.
# Only the batch dimension is variable; microbatch_size is always 1 in PP v0, so
# the global batch equals --microbatches.
LATENT_CHANNELS = 16
LATENT_T_HW = (1, 150, 112)
CONTEXT_LEN = (512, 1024)


def batch_shapes(batch: int):
    """Wire-buffer shapes for a global batch of ``batch`` samples."""

    return (
        (batch, LATENT_CHANNELS, *LATENT_T_HW),
        (batch,),
        (batch, *CONTEXT_LEN),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pp", "single"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dit-path", type=Path, required=True)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--rank-dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--attn-mode", default="torch")
    parser.add_argument("--checkpoint-policy", default="full")
    parser.add_argument("--init-scale", type=float, default=65536.0)
    parser.add_argument("--capture-grads", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--microbatch-accumulate",
        action="store_true",
        help="single mode: same 4x(bs1) split + accumulation order as the PP run "
        "instead of one bs4 forward (the matched reference for track 2)",
    )
    parser.add_argument(
        "--autocast-backward",
        action="store_true",
        help="single mode: keep the backward inside the autocast region, matching "
        "what a PP schedule can do (forward and backward are interleaved in one "
        "schedule.step call)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_batches(args, count):
    return load_cached_batches(
        args.latent_path,
        args.text_path,
        count=count,
        seed=args.seed,
        dtype=torch.float16,
    )


def pack(batches, device):
    """Concatenate per-sample batches into one global-batch step input."""

    inputs = tuple(
        torch.cat([batch[0][i].to(device) for batch in batches]) for i in range(4)
    )
    target = torch.cat([batch[1].to(device) for batch in batches])
    return inputs, target


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def configure_checkpoint(model, policy):
    for block in model.blocks:
        block.gradient_checkpointing = policy == "full"


def build_base(args):
    model = load_model(args.dit_path, torch.device("cpu"), args.attn_mode)
    # load_model loads bf16; PP v0 is fp16 + GradScaler on this host (SM70 has no
    # native bf16: 8.4 TFLOP/s vs 91.2 for fp16, see
    # docs/findings/multigpu_v100_nvlink_baseline_20261107.md).
    model = model.to(dtype=torch.float16)
    network = create_mlp_lora(
        model, seed=args.seed + 101, rank_dim=args.rank_dim, alpha=args.alpha
    )
    return model, network


def adapter_grads(network):
    payload = {}
    for lora in network.unet_loras:
        for name, parameter in lora.named_parameters():
            if parameter.grad is not None:
                payload[f"{lora.lora_name}.{name}"] = (
                    parameter.grad.detach().float().cpu().clone()
                )
    return payload


# ---------------------------------------------------------------------------
# single-card reference
# ---------------------------------------------------------------------------


def run_single(args, device):
    model, network = build_base(args)
    configure_checkpoint(model, args.checkpoint_policy)
    model.to(device=device, dtype=torch.float16).train()
    network.to(device=device, dtype=torch.float32).train()
    model.pos_embedder._cos_sin_cache.clear()
    params = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=args.init_scale)

    batches = load_batches(args, (args.steps + args.warmup_steps) * args.microbatches)
    records = []
    grads = None
    for step in range(args.steps + args.warmup_steps):
        step_batches = batches[
            step * args.microbatches : (step + 1) * args.microbatches
        ]
        inputs, target = pack(step_batches, device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if args.microbatch_accumulate:
            # Same split/accumulation order as the 1F1B run: 4 x bs1, each loss
            # divided by 4, gradients summed in the master weights.
            with torch.autocast("cuda", dtype=torch.float16):
                total = None
                for index, batch in enumerate(step_batches):
                    one_inputs = tuple(value.to(device) for value in batch[0])
                    one_target = batch[1].to(device)
                    one_pred = model.forward_mini_train_dit(
                        one_inputs[0], one_inputs[1], one_inputs[2],
                        padding_mask=one_inputs[3],
                    )
                    one_loss = torch.nn.functional.mse_loss(
                        one_pred.float(), one_target.float()
                    ) / len(step_batches)
                    total = one_loss if total is None else total + one_loss
            scaler.scale(total).backward()
            loss = total * len(step_batches)
            del inputs, target
        elif args.autocast_backward:
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model.forward_mini_train_dit(
                    inputs[0], inputs[1], inputs[2], padding_mask=inputs[3]
                )
                loss = torch.nn.functional.mse_loss(
                    prediction.float(), target.float()
                )
                scaler.scale(loss).backward()
        else:
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model.forward_mini_train_dit(
                    inputs[0], inputs[1], inputs[2], padding_mask=inputs[3]
                )
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
            scaler.scale(loss).backward()
        if step == 0 and args.capture_grads:
            grads = adapter_grads(network)
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if step >= args.warmup_steps:
            records.append({"seconds": elapsed, "loss": float(loss.detach())})
        if step == 0:
            first_loss = float(loss.detach())
            torch.cuda.reset_peak_memory_stats(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    payload = {
        "mode": "single",
        "steps": records,
        "first_step_loss": first_loss,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "scale": float(scaler.get_scale()),
    }
    if grads is not None:
        torch.save(grads, args.output_dir / "grads-single.pt")
    return payload


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def log(rank, message):
    print(f"[r{rank}] {message}", flush=True)


def patch_prediction_meta_dtype():
    """Correct the last stage's prediction buffer dtype for fp16 mixed precision.

    ``boundary_metas`` takes one ``dtype`` for *both* the boundary activation and
    the last stage's prediction.  Under fp16 mixed precision those two are not the
    same tensor type: the boundary is the block-stack output, which the model
    upcasts to fp32 (``models.py:1519``), while the prediction comes out of
    ``final_layer`` under ``autocast(fp16)`` and is fp16.  A single ``dtype``
    therefore cannot describe both, so the harness pins the prediction meta to the
    dtype the stage actually produces.  Transport-buffer bookkeeping only -- no
    numerics are touched.
    """

    from library.training.pp import runtime as runtime_module
    from library.training.pp import schedule as schedule_module

    real = runtime_module.boundary_metas

    def boundary_metas(model, layout, **kwargs):
        inputs, outputs = real(model, layout, **kwargs)
        if layout.is_last:
            outputs = tuple(tensor.to(torch.float16) for tensor in outputs)
        return inputs, outputs

    runtime_module.boundary_metas = boundary_metas
    schedule_module.boundary_metas = boundary_metas


def build_pp(args, rank, device, group):
    from library.training.pp import (
        PipelineRuntimeConfig,
        build_anima_pipeline_runtime,
        release_foreign_blocks,
    )

    log(rank, "loading model")
    model, network = build_base(args)
    log(rank, "model+lora ready")

    config = PipelineRuntimeConfig(
        stages=2,
        microbatches=args.microbatches,
        schedule="1f1b",
        conditioning="local",
        mixed_precision="fp16",
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=args.init_scale) if rank == 1 else None

    def factory(stage_module):
        params = list(stage_module.trainable_parameters())
        if not params:
            raise RuntimeError(f"stage {rank} owns no trainable adapter parameters")
        return torch.optim.AdamW(params, lr=args.lr)

    latent_shape, timesteps_shape, context_shape = batch_shapes(args.microbatches)
    log(rank, "building runtime")
    runtime = build_anima_pipeline_runtime(
        model,
        config=config,
        world_size=2,
        rank=rank,
        device=device,
        latent_shape=latent_shape,
        # Post-D1/D2 the runtime takes the *pre-block compute* dtype and resolves
        # PpComputePolicy from it once (use_fp32=True for fp16, boundary fp32,
        # prediction fp16).  Passing the boundary dtype here would resolve
        # use_fp32=False and reintroduce D1.
        dtype=torch.float16,
        timesteps_shape=timesteps_shape,
        context_shape=context_shape,
        network=network,
        scaler=scaler,
        optimizer_factory=factory,
        group=group,
    )
    # The stage module must be built over the FULL 40-block model, otherwise the
    # placement plan is computed over an already-sliced block list and the split
    # silently becomes "10 of 20 blocks per stage" (the first smoke run did
    # exactly that).  Release + move only after the runtime owns its slice.
    log(rank, f"runtime ready, layout {runtime.layout.block_range}")
    release_foreign_blocks(model, runtime.layout)
    configure_checkpoint(model, args.checkpoint_policy)
    model.to(device=device, dtype=torch.float16).train()
    # Adapters keep fp32 master weights: torch's GradScaler refuses to unscale
    # fp16 gradients ("Attempting to unscale FP16 gradients"), so fp16 + scaler
    # means fp16 autocast compute with fp32 parameters, exactly like accelerate's
    # mixed_precision=fp16.
    network.to(device=device, dtype=torch.float32).train()
    model.pos_embedder._cos_sin_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    log(rank, f"stage moved to {device}")
    return runtime, network


def _stats(tag, value):
    tensors = value if isinstance(value, (tuple, list)) else (value,)
    for index, tensor in enumerate(tensors):
        if torch.is_tensor(tensor):
            finite = bool(torch.isfinite(tensor).all().item())
            print(
                f"[dbg] {tag}[{index}] dtype={tensor.dtype} shape={tuple(tensor.shape)} "
                f"finite={finite} min={float(tensor.min()):.4g} max={float(tensor.max()):.4g}",
                flush=True,
            )


def run_pp(args, rank, device, group):
    runtime, network = build_pp(args, rank, device, group)
    log(rank, f"compute_policy {runtime.compute_policy.as_dict()}")
    if args.debug:
        runtime.stage_module.register_forward_pre_hook(
            lambda module, inputs: _stats(f"r{rank} stage-in", inputs)
        )
        runtime.stage_module.register_forward_hook(
            lambda module, inputs, output: _stats(f"r{rank} stage-out", output)
        )
    bad = {"index": None, "in_max": None}

    def block_hook(module, inputs, output):
        if bad["index"] is not None:
            return
        if not bool(torch.isfinite(output).all()):
            bad["index"] = int(module._block_idx)
            bad["in_max"] = float(inputs[0].max())

    for block in runtime.stage_module.block_stage.blocks:
        block.register_forward_hook(block_hook)
    batches = load_batches(args, (args.steps + args.warmup_steps) * args.microbatches)
    records = []
    grads = None
    first_loss = None
    for step in range(args.steps + args.warmup_steps):
        inputs, target = pack(
            batches[step * args.microbatches : (step + 1) * args.microbatches], device
        )
        log(rank, f"step {step} start")
        runtime.zero_grad()
        if step == 0:
            torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses = []
        with torch.autocast("cuda", dtype=torch.float16):
            result = runtime.step(
                inputs[0],
                timesteps=inputs[1],
                crossattn_emb=inputs[2],
                # The last stage's prediction is model_pred_to_loss_4d, so the
                # target has to be the 4D latent, not the 5D one.
                target=target.squeeze(2),
                padding_mask=inputs[3],
                losses=losses,
            )
        if step == 0 and args.capture_grads:
            grads = adapter_grads(network)
        outcome = runtime.optimizer_step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        scale = float(runtime.scale)
        if step == 0:
            first_loss = (
                float(sum(losses)) / scale if losses else None
            )
        if step >= args.warmup_steps:
            records.append(
                {
                    "seconds": elapsed,
                    "skipped": bool(outcome.skipped),
                    "scale": scale,
                    "loss": float(sum(losses)) / scale if losses else None,
                }
            )
    times = torch.tensor([max(r["seconds"] for r in records)], dtype=torch.float64, device=device)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    payload = {
        "mode": "pp",
        "rank": rank,
        "steps": records,
        "wall_seconds_per_step_max_across_ranks": float(times.item()),
        "first_step_loss": first_loss,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "scale": float(runtime.scale),
        "growth_interval": runtime.transaction.growth_interval,
        "compute_policy": runtime.compute_policy.as_dict(),
        "first_non_finite_block": bad["index"],
        "first_non_finite_block_input_max": bad["in_max"],
        "topology": runtime.topology(),
        "skipped_any": any(r["skipped"] for r in records),
    }
    if grads is not None:
        torch.save(grads, args.output_dir / f"grads-rank{rank}.pt")
    return payload


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    if os.environ.get("PPV0_FAULT"):
        import faulthandler

        faulthandler.dump_traceback_later(float(os.environ["PPV0_FAULT"]), exit=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    group = None
    if args.mode == "pp":
        # PP v0 needs *two* backends from one group: torch.distributed.pipelining
        # moves CUDA activations with P2P (NCCL only) while pp/transactions.py
        # agrees the fp16 scale with CPU tensors (int32 all_reduce / float64
        # broadcast, Gloo only).  A single-backend group cannot serve both --
        # Gloo P2P on CUDA tensors aborts with "gloo::IoException ... Bad address",
        # and NCCL rejects CPU tensors.  torch >= 2.6 multi-backend groups
        # dispatch per tensor device, so one group covers both.
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            timeout=timedelta(seconds=180),
            device_id=device,
        )
        if dist.get_world_size() != 2:
            raise ValueError("pp mode requires exactly two ranks")
        group = dist.group.WORLD
    try:
        if args.mode == "pp":
            payload = run_pp(args, rank, device, group)
            (args.output_dir / f"result-rank{rank}.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            if rank == 0:
                print(json.dumps({k: v for k, v in payload.items() if k != "topology"}), flush=True)
        else:
            payload = run_single(args, device)
            (args.output_dir / "result-single.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(payload), flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # destroy_process_group() blocks forever when the peer is still inside a
        # collective, which hides the real traceback; die loudly instead.
        os._exit(1)
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
