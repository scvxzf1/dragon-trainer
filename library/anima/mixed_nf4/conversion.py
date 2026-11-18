"""One-matrix-at-a-time CPU conversion and atomic, non-overwriting publication."""
from __future__ import annotations
from pathlib import Path
import torch
from safetensors import safe_open
from library.quantization.constants import STATE_KEYS
from library.quantization.format import MANIFEST_KEY as CANONICAL_MANIFEST_KEY
from library.quantization.format import inspect_mixed_nf4 as inspect_canonical_mixed_nf4
from library.quantization.format import validate_mixed_nf4_payload
from library.quantization.policy import ANIMA_SELECTIVE_POLICY
from .format import inspect_mixed_nf4 as inspect_legacy_mixed_nf4
from .format import quantized_paths, quant_state, sha256


def validate_source(f) -> tuple[str, dict]:
    keys = set(f.keys())
    prefix = next((p for p in ('net.', 'model.diffusion_model.', '') if p + 'blocks.0.mlp.layer1.weight' in keys), None)
    if prefix is None or len(keys) != 685 or any('.weight.quant_state.' in k for k in keys):
        raise ValueError('Expected standard 685-tensor Anima BF16 checkpoint')
    shapes = {}
    for key in sorted(keys):
        sl = f.get_slice(key)
        if sl.get_dtype() != 'BF16':
            raise ValueError(f'Source must be entirely BF16: {key}')
        shapes[key] = sl.get_shape()
    for i in range(28):
        for layer, shape in ((1, [8192, 2048]), (2, [2048, 8192])):
            if shapes.get(f'{prefix}blocks.{i}.mlp.layer{layer}.weight') != shape:
                raise ValueError('Source requires 28 blocks and hidden width 2048')
    blocks = {k[len(prefix):].split('.')[1] for k in keys if k.startswith(prefix + 'blocks.')}
    if blocks != {str(i) for i in range(28)}:
        raise ValueError('Source block set mismatch')
    return prefix, shapes


def verify_checkpoint(path, source=None) -> dict:
    with safe_open(str(path), framework='pt', device='cpu') as f:
        metadata = f.metadata() or {}
    canonical = CANONICAL_MANIFEST_KEY in metadata
    manifest = (
        inspect_canonical_mixed_nf4(path, family='anima')
        if canonical else inspect_legacy_mixed_nf4(path)
    )
    if manifest is None:
        raise ValueError('Not an Anima mixed NF4 checkpoint')
    if canonical:
        validate_mixed_nf4_payload(path, family='anima', manifest=manifest)
        if source is not None:
            source_hash = sha256(source)
            if source_hash != manifest['source_sha256']:
                raise ValueError('Source SHA256 differs from manifest')
            with safe_open(str(source), framework='pt', device='cpu') as base:
                prefix, shapes = validate_source(base)
                if prefix != manifest.get('prefix', '') or {
                    key: {'shape': list(shape), 'dtype': 'bfloat16'}
                    for key, shape in shapes.items()
                } != manifest['source_tensors']:
                    raise ValueError('Source inventory differs from manifest')
        return {
            'quantized_weights': len(manifest['modules']),
            'retained_layer1_hotspots': 7,
            'bytes': Path(path).stat().st_size,
        }
    with safe_open(str(path), framework='pt', device='cpu') as f:
        for p in manifest['modules']:
            w = manifest['prefix'] + p + '.weight'
            state = {w + '.' + k: f.get_tensor(w + '.' + k) for k in STATE_KEYS}
            quant_state(state, w, manifest['tensor_shapes'][w])
        if source is not None:
            if sha256(source) != manifest['source_sha256']:
                raise ValueError('Source SHA256 differs from manifest')
            with safe_open(str(source), framework='pt', device='cpu') as base:
                prefix, shapes = validate_source(base)
                if shapes != manifest['tensor_shapes'] or prefix != manifest['prefix']:
                    raise ValueError('Source inventory differs from manifest')
                targets = {prefix + p + '.weight' for p in manifest['modules']}
                for k in base.keys():
                    if k not in targets and not torch.equal(base.get_tensor(k).view(torch.uint8), f.get_tensor(k).view(torch.uint8)):
                        raise ValueError(f'Retained source tensor changed: {k}')
    return {'quantized_weights': 49, 'retained_layer1_hotspots': 7, 'bytes': Path(path).stat().st_size}


def convert_checkpoint(source, output, *, device='cpu', threads=None, progress=None) -> dict:
    from library.anima.checkpoint import inspect_anima_checkpoint
    if threads is not None:
        if type(threads) is not int or threads <= 0:
            raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    from library.quantization.mixed_nf4 import _write_mixed_nf4_state_dict
    layout = inspect_anima_checkpoint(source)
    if layout.num_blocks != 28 or layout.model_channels != 2048:
        raise ValueError('Only standard 28-block/2048 Anima is supported')
    source_hash = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(str(source), framework='pt', device='cpu') as f:
        prefix, shapes = validate_source(f)
        validate_model_shapes(shapes)
        metadata = dict(f.metadata() or {})
        if any(k.startswith('anima.mixed_nf4') for k in metadata):
            raise ValueError('Source already contains mixed NF4 metadata')
        targets = {prefix + p + '.weight' for p in quantized_paths()}
        state = {}
        target_index = 0
        for key in f.keys():
            tensor = f.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite source tensor: {key}')
            state[key] = tensor
            if key in targets and progress:
                target_index += 1
                progress(target_index, len(targets), key)
    if sha256(source) != source_hash:
        raise ValueError('Source changed during conversion')
    result = _write_mixed_nf4_state_dict(
        state,
        output,
        family='anima',
        source_sha256=source_hash,
        policy=ANIMA_SELECTIVE_POLICY,
        modules=quantized_paths(),
        prefix=prefix,
        metadata=metadata,
        quantize_device=device,
    )
    result = verify_checkpoint(output, source)
    if sha256(source) != source_hash:
        raise ValueError('Source changed before publication')
    result.update(source_sha256=source_hash, output_sha256=sha256(output),
                  source_bytes=source.stat().st_size, output=str(output))
    result['saving_fraction'] = 1 - result['bytes'] / result['source_bytes']
    return result


def validate_model_shapes(shapes):
    """Check every source shape against the current meta-only standard model."""
    from accelerate import init_empty_weights
    from library.anima.models import Anima
    from library.anima.weights import _dit_concat_hook, _dit_rename_hook
    with init_empty_weights():
        reference = Anima(max_img_h=512, max_img_w=512, max_frames=128,
                          in_channels=16, out_channels=16, patch_spatial=2,
                          patch_temporal=1, num_blocks=28, attn_mode='torch')
    expected = {k: list(v.shape) for k, v in reference.state_dict().items()}
    actual, groups = {}, {}
    for k, shape in shapes.items():
        fused, _ = _dit_concat_hook(k, None)
        if fused is None:
            actual[_dit_rename_hook(k)] = shape
        else:
            groups.setdefault(fused, []).append(shape)
    for k, rows in groups.items():
        if any(row[1:] != rows[0][1:] for row in rows):
            raise ValueError(f'Source fused projection shapes disagree: {k}')
        actual[k] = [sum(row[0] for row in rows), *rows[0][1:]]
    optional = ('seq', 'dim_spatial_range', 'dim_temporal_range', 'inv_freq', 'pooled_text_proj')
    required = {k for k in expected if not any(token in k for token in optional)}
    if set(actual) != required or any(actual[k] != expected[k] for k in actual):
        raise ValueError('Source tensor structure differs from standard Anima model')
