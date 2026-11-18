"""CPU conversion and direct restoration of canonical mixed NF4 models."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .format import MANIFEST_KEY, inspect_mixed_nf4, validate_mixed_nf4_payload
from .nf4 import STATE_KEYS, quant_state, restore_nf4_linear
from .policy import ANIMA_SELECTIVE_POLICY, POLICY, model_paths


def _model_inventory(model, paths, *, conversion, prefix=''):
    params = dict(model.named_parameters())
    state = model.state_dict()
    if any('quant_state' in key or 'lora_' in key for key in state):
        raise ValueError('Expected ordinary model without packed weights or adapters')
    inventory = {}
    # Keep the manifest inventory canonical.  Stable key order makes the
    # metadata reproducible regardless of module registration order.
    for key in sorted(state):
        value = state[key]
        if conversion:
            if value.device.type != 'cpu':
                raise ValueError('Mixed NF4 conversion requires a CPU model')
            if value.is_floating_point() and (
                    (key in params and value.dtype != torch.bfloat16)
                    or (key not in params and value.dtype not in (torch.bfloat16, torch.float32))):
                raise ValueError(f'Invalid source floating tensor dtype: {key}')
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite source tensor: {key}')
        inventory[prefix + key] = {'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    return inventory


def _validate_model(model, manifest, paths):
    prefix = manifest.get('prefix', '')
    expected = _model_inventory(model, paths, conversion=False, prefix=prefix)
    actual = manifest['source_tensors']
    if set(expected) != set(actual) or any(expected[k]['shape'] != actual[k]['shape'] for k in expected):
        raise ValueError('Mixed NF4 source inventory differs from model')
    if paths != manifest['modules']:
        raise ValueError('Mixed NF4 policy differs from model')
    params = dict(model.named_parameters())
    for key, entry in actual.items():
        model_key = key.removeprefix(prefix)
        if model_key in params and params[model_key].is_floating_point() and entry['dtype'] != 'bfloat16':
            raise ValueError(f'Mixed NF4 floating parameters must be BF16: {key}')
        if model_key not in params and expected[key]['dtype'] != entry['dtype']:
            raise ValueError(f'Mixed NF4 buffer dtype differs from model: {key}')


def load_mixed_nf4_into(model, path, *, family, dtype, device):
    """Load into a plain meta model after exact header and qstate validation."""
    if dtype != torch.bfloat16:
        raise ValueError('Mixed NF4 requires BF16 compute')
    path = Path(path).expanduser()
    manifest = inspect_mixed_nf4(path, family=family)
    if manifest is None:
        raise ValueError('Not a canonical mixed NF4 checkpoint')
    paths = model_paths(model, family, manifest['policy'])
    if any(v.device.type != 'meta' for v in model.state_dict().values()):
        raise ValueError('Mixed NF4 loading requires a meta model')
    _validate_model(model, manifest, paths)
    restored, retained = {}, {}
    prefix = manifest.get('prefix', '')
    targets = {prefix + p + '.weight' for p in paths}
    biases = {prefix + p + '.bias' for p in paths if model.get_submodule(p).bias is not None}
    model_targets = {p + '.weight' for p in paths}
    model_biases = {p + '.bias' for p in paths if model.get_submodule(p).bias is not None}
    with safe_open(str(path), framework='pt', device='cpu') as f:
        # Validate every small qstate before installing modules or reading retained weights.
        states = {}
        for p in paths:
            key = prefix + p + '.weight'
            qs = {key + '.' + s: f.get_tensor(key + '.' + s) for s in STATE_KEYS}
            states[p], _ = quant_state(qs, key, manifest['source_tensors'][key]['shape'])
        for p in paths:
            target = model.get_submodule(p)
            restored[p] = restore_nf4_linear(f.get_tensor(prefix + p + '.weight'), states[p],
                in_features=target.in_features, out_features=target.out_features,
                bias=f.get_tensor(prefix + p + '.bias') if target.bias is not None else None, device=device)
        for key in manifest['source_tensors']:
            if key not in targets | biases:
                retained[key.removeprefix(prefix)] = f.get_tensor(key).to(device)
    for p, module in restored.items():
        if '.' in p:
            parent, attr = p.rsplit('.', 1)
            target_parent = model.get_submodule(parent)
        else:
            attr, target_parent = p, model
        setattr(target_parent, attr, module)
    missing, unexpected = model.load_state_dict(retained, strict=False, assign=True)
    if set(missing) != model_targets | model_biases or unexpected:
        raise ValueError(f'Mixed NF4 state mismatch: missing={missing}, unexpected={unexpected}')
    model._mixed_nf4_family = family
    model._mixed_nf4_policy = manifest['policy']
    return model


def _write_mixed_nf4_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    path,
    *,
    family,
    source_sha256,
    policy,
    modules,
    prefix='',
    metadata=None,
    quantize_device='cpu',
):
    """Write a canonical artifact from an already validated CPU state map."""
    import bitsandbytes
    from bitsandbytes.functional import quantize_4bit

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError('Mixed NF4 source state must be a non-empty mapping')
    if any('quant_state' in key or 'lora_' in key for key in state_dict):
        raise ValueError('Expected ordinary model without packed weights or adapters')
    output = Path(path)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if not isinstance(source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('source_sha256 must be 64 lowercase hexadecimal characters')
    if not isinstance(modules, list) or not modules or len(set(modules)) != len(modules):
        raise ValueError('Mixed NF4 policy modules must be a non-empty unique list')
    inventory = {}
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
            raise ValueError('Mixed NF4 conversion requires CPU tensor state')
        if value.is_floating_point() and value.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f'Invalid source floating tensor dtype: {key}')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite source tensor: {key}')
        inventory[key] = {
            'shape': list(value.shape),
            'dtype': str(value.dtype).removeprefix('torch.'),
        }
    expected_keys = set(inventory)
    if any(key not in expected_keys for key in (prefix + p + '.weight' for p in modules)):
        raise ValueError('Mixed NF4 policy module is missing from source state')
    manifest = {'version': 1, 'family': family, 'policy': policy, 'compute_dtype': 'bfloat16',
        'blocksize': 64, 'double_quant': True, 'modules': modules, 'source_tensors': inventory,
        'source_sha256': source_sha256, 'bitsandbytes_version': bitsandbytes.__version__,
        **({'prefix': prefix} if prefix else {})}
    payload = {key: value for key, value in state_dict.items()}
    for p in modules:
        key = prefix + p + '.weight'
        if inventory[key]['dtype'] != 'bfloat16' or len(inventory[key]['shape']) != 2:
            raise ValueError(f'NF4 policy target must be a BF16 matrix: {p}')
        if any(d % 64 for d in inventory[key]['shape']):
            raise ValueError(f'NF4 dimensions must be divisible by 64: {p}')
        packed, qs = quantize_4bit(payload[key].to(quantize_device), blocksize=64, compress_statistics=True,
                                    quant_type='nf4', quant_storage=torch.uint8)
        payload[key] = packed.cpu().contiguous()
        payload.update({key + '.' + k: v.cpu().contiguous() for k, v in qs.as_dict(packed=True).items()})
        quant_state(payload, key, inventory[key]['shape'])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + output.name + '.', dir=output.parent)
    os.close(fd)
    try:
        output_metadata = dict(metadata or {})
        if MANIFEST_KEY in output_metadata:
            raise ValueError(f'Existing metadata already contains {MANIFEST_KEY}')
        output_metadata[MANIFEST_KEY] = json.dumps(manifest, sort_keys=True)
        save_file(payload, temporary, metadata=output_metadata)
        validated = inspect_mixed_nf4(temporary, family=family)
        if validated is None:
            raise ValueError('Canonical mixed NF4 manifest was not published')
        validate_mixed_nf4_payload(
            temporary,
            family=family,
            manifest=validated,
        )
        os.link(temporary, output)  # Atomic publication with no clobber, including races.
        return {'quantized_weights': len(modules), 'bytes': output.stat().st_size,
                'output': str(output), 'source_sha256': source_sha256}
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_mixed_nf4_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    path,
    *,
    family,
    source_sha256,
    modules,
    policy=POLICY,
    prefix='',
    metadata=None,
    quantize_device='cpu',
):
    """Publish canonical mixed NF4 from a raw, CPU safetensors state map."""
    if Path(path).expanduser().exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    if family == 'anima' and policy == POLICY:
        policy = ANIMA_SELECTIVE_POLICY
    result = _write_mixed_nf4_state_dict(
        state_dict,
        path,
        family=family,
        source_sha256=source_sha256,
        policy=policy,
        modules=list(modules),
        prefix=prefix,
        metadata=metadata,
        quantize_device=quantize_device,
    )
    return result


def save_mixed_nf4(model, path, *, family, source_sha256, policy=None, prefix=''):
    """Quantize CPU BF16 MLP weights without changing the source model."""
    if Path(path).expanduser().exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    policy = POLICY if policy is None else policy
    if family == 'anima' and policy == POLICY:
        policy = ANIMA_SELECTIVE_POLICY
    paths = model_paths(model, family, policy)
    serialized_state = {
        prefix + key: value
        for key, value in model.state_dict().items()
    }
    result = save_mixed_nf4_state_dict(
        serialized_state,
        path,
        family=family,
        source_sha256=source_sha256,
        modules=paths,
        policy=policy,
        prefix=prefix,
    )
    return result
