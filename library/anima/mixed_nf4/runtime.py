"""Restore packed MLP modules without a second quantization pass."""
from __future__ import annotations
import torch
from safetensors.torch import load_file
from library.quantization.constants import STATE_KEYS
from library.quantization.format import inspect_mixed_nf4 as inspect_canonical_mixed_nf4
from .format import inspect_mixed_nf4 as inspect_legacy_mixed_nf4, quant_state


def inspect_mixed_nf4(path):
    """Read canonical output first, then accept the legacy Anima format."""
    from safetensors import safe_open

    with safe_open(str(path), framework='pt', device='cpu') as f:
        metadata = f.metadata() or {}
    if 'diffusion.mixed_nf4.manifest' in metadata:
        return inspect_canonical_mixed_nf4(path, family='anima')
    return inspect_legacy_mixed_nf4(path)


def restore_linear(data, qs, *, in_features, out_features, bias=None):
    from library.quantization.nf4 import restore_nf4_linear

    return restore_nf4_linear(
        data,
        qs,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device="cpu",
    )


def load_mixed_state(model, path, *, dtype, lora_weights_list, rename_hook, concat_hook, device):
    if dtype != torch.bfloat16:
        raise ValueError('Anima mixed NF4 requires BF16 compute')
    if lora_weights_list:
        raise ValueError('Load-time LoRA merge is unsupported for Anima mixed NF4')
    manifest = inspect_mixed_nf4(path)
    if manifest is None:
        raise ValueError('Not an Anima mixed NF4 checkpoint')
    canonical = 'source_tensors' in manifest
    state = load_file(str(path), device='cpu')
    quantized = {}
    for p in manifest['modules']:
        prefix = manifest.get('prefix', '')
        w = prefix + p + '.weight'
        shape = manifest['source_tensors'][w]['shape'] if canonical else manifest['tensor_shapes'][w]
        qs, _ = quant_state(state, w, shape)
        target = model.get_submodule(p)
        if (target.in_features, target.out_features) != (shape[1], shape[0]):
            raise ValueError(f'Mixed NF4 model shape mismatch: {p}')
        bias_key = prefix + p + '.bias'
        bias = state.get(bias_key)
        if (target.bias is None) != (bias is None):
            raise ValueError(f'Mixed NF4 bias mismatch: {p}')
        quantized[p] = restore_linear(state.pop(w), qs, in_features=target.in_features,
                                     out_features=target.out_features, bias=bias)
        for k in STATE_KEYS:
            del state[w + '.' + k]
        state.pop(bias_key, None)
    transformed, groups = {}, {}
    for k, v in state.items():
        fused, _ = concat_hook(k, None)
        if fused is None:
            transformed[rename_hook(k)] = v.to(device)
        else:
            groups.setdefault(fused, {})[k] = v
    for group in groups.values():
        k, v = concat_hook(next(iter(group)), group)
        transformed[k] = v.to(device)
    # assign=True preserves Params4bit only when it is not in the state_dict.
    for p, new in quantized.items():
        parent, attr = p.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, new.to(device))
    missing, unexpected = model.load_state_dict(transformed, strict=False, assign=True)
    expected_missing = {p + '.weight' for p in quantized}
    expected_missing.update(p + '.bias' for p, new in quantized.items() if new.bias is not None)
    return [k for k in missing if k not in expected_missing], unexpected
