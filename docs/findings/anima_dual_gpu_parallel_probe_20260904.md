# Anima dual-GPU PP2 / TP2 probe

Status: Experimental; historical bounded probe, not a production capability claim.

Date: 2026-09-04

> **DEPRECATED (PP 数字)**：本页 PP2 的 **1.806 s/step 已作废，不得作为任何基线引用**。
> 它来自 `scripts/experiments/anima_parallel_probe.py:69`（`_run_pp`）的**手写阻塞
> `dist.send` / `dist.recv`（`:96` / `:98` / `:102` / `:112`）+ `cuda.synchronize`
> 循环**，**完全没有使用
> `torch.distributed.pipelining`**，也不是 `Schedule1F1B`——它是单 microbatch 的
> fill-drain 往返。PP v0 的性能结论必须来自 `library/training/pp` 运行时本身的测量，
> 不能用本页数字外推。TP2 / TP2 INT8 的数字不受此标记影响（它们不声称 PP）。
> 取代说明见 [`docs/experimental/anima_pipeline_parallel_v0.md`](../experimental/anima_pipeline_parallel_v0.md)。

## Scope

This is a standalone experimental probe, not the production trainer runtime.
It used Anima-2.9B-preview-v1 (40 blocks), BF16 frozen base weights, plain MLP
LoRA rank 16, batch size 1, full gradient checkpointing, one warmup step, and
eight optimizer steps over the same cached sample and initialization.

The actual machine was heterogeneous:

- rank 0: NVIDIA CMP 170HX 64GB
- rank 1: NVIDIA GeForce RTX 3080 10GB
- topology: PHB, CUDA peer access unavailable

These measurements must not be relabeled as a homogeneous pair or as PCIe
3.0 x8 results.

## Results

| Mode | Schedule / transport | s/step | it/min | Peak allocated GiB rank0/rank1 | Peak reserved GiB rank0/rank1 | Wire GiB/rank, warmup included | Calls/rank | Rank0 blocking communication/wait share | Loss first -> last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PP2 ~~1.806 s/step~~ **DEPRECATED** — see the banner above; hand-written *blocking* `dist.send`/`dist.recv`, not `torch.distributed.pipelining` | 20/20 blocks, one-microbatch fill-drain | ~~1.806~~ | ~~33.22~~ | 3.90 / 3.91 | 3.98 / 4.00 | 0.29 | 18 | 72.2% | 0.123285 -> 0.084241 |
| TP2 | BF16 activation and exact LoRA-gradient collectives | 3.057 | 19.63 | 4.19 / 4.19 | 4.32 / 4.32 | 51.95 | 3960 | 81.4% | 0.123263 -> 0.084309 |
| TP2 INT8 | group-128 INT8 activation transport; exact BF16 LoRA gradients | 3.050 | 19.67 | 4.27 / 4.27 | 4.41 / 4.41 | 26.40 | 3960 | 81.1% | 0.123543 -> 0.084282 |

PP2 is a batch-size-one fill-drain measurement. It is deliberately not called
1F1B because there is only one microbatch. TP2 shards attention heads and MLP
features while keeping AdaLN, patch embed, and final projection replicated.

The group-128 INT8 path cut measured wire bytes by 49.2%, but improved total
step time by only about 0.23%. Quantize/dequantize work consumed the bandwidth
gain on this PHB/no-peer-access topology. The reported communication share is
the time spent inside blocking collectives or P2P calls, including fast-rank
wait time; it is not pure PCIe transfer time or link utilization. This is
transport quantization, not QAT.

## Numerical and image comparison

| Initial output comparison | rel-L2 | cosine |
|---|---:|---:|
| TP2 vs PP2 | 0.02057 | 0.999775 |
| TP2 INT8 vs PP2 | 0.04034 | 0.999172 |
| TP2 INT8 vs TP2 | 0.03967 | 0.999199 |

Each consolidated step-8 LoRA checkpoint was loaded by the normal independent
inference path. All three used seed 114, 1024x1024, 28 ER-SDE steps, CFG 4.0,
flow shift 1.0, FlashAttention, and the same prompt.

| Image comparison | MAE | RMSE | PSNR dB | cosine |
|---|---:|---:|---:|---:|
| TP2 vs PP2 | 0.005360 | 0.024682 | 32.15 | 0.996035 |
| TP2 INT8 vs PP2 | 0.003837 | 0.017553 | 35.11 | 0.997961 |
| TP2 INT8 vs TP2 | 0.005511 | 0.025399 | 31.90 | 0.995813 |

The composition, pose, lighting, and background remained stable in this one
seed. Local flower and fabric details changed. This one-seed, eight-step smoke
test is not a long-training quality evaluation.

## Decision

On this machine PP2 was clearly preferable. TP2 was dominated by frequent
fine-grained synchronization. INT8 transport did not produce a meaningful
end-to-end speedup and added about 4% initial-output rel-L2 versus PP2, so it is
rejected as a production default.

Reproducible outputs are under
`output/bench/anima_dual_gpu_compare/20260904-anima29-pp-tp/`, including the
three checkpoints, per-mode JSON, `samples.csv`, `metrics.json`,
`contact_sheet.png`, and `report.md`.
