# DoRA

状态：稳定
适用版本：当前 main

DoRA adds a learned per-output magnitude to a low-rank weight-direction update.
The implementation is `networks/lora_modules/dora.py`.

## Configuration

For a model family that supports DoRA:

```toml
dora_wd = true
network_dim = 16
network_alpha = 8
```

The row norm is intentionally detached during training. This is the existing
DoRA gradient convention, not a missing gradient through the NF4 implementation.

## Strength And Disable

Runtime strength scales the complete DoRA edit, including magnitude:

```text
output(strength) = base_output + strength * (full_DoRA_output - base_output)
```

Strength 0 or `network.set_enabled(False)` returns the exact base output.
Strength 1 retains full DoRA training behavior. Intermediate and negative
strengths now interpolate/extrapolate the complete edit; this changes their
behavior relative to older versions that scaled only the direction update.
Forward, dense merge and dense fuse use the same convention.

## NF4

The packed base is read only. Initialization dequantizes one logical weight at
a time, then retains only the magnitude and FP32 row-norm cache. The cache keeps
its FP32 values across device/dtype moves and is reconstructed from the base on
load rather than written to the adapter checkpoint.

During forward, the NF4 norm uses a bias-free rank-sized quantized matmul and
FP32 low-rank norm algebra. NF4 base computation stays BF16. Zero-initialized
adapters preserve the base output exactly, including BF16 bias rounding.

NF4 `merge_to` and `fuse_weight` are explicitly unsupported: these operations
write dense weights and cannot be applied to packed storage. Keep the adapter
live on NF4. This restriction is unrelated to fused AdamW or FlashAttention.

Krea-2's registry permits network variant selection, including DoRA. DoRA on Krea-2 NF4
has bounded experimental verification, not general production or quality
certification. See [the repair report](../findings/krea2_nf4_adapter_repair_20260905.md).
