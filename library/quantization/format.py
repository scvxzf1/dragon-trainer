"""Canonical, self-contained mixed NF4 safetensors manifest validation."""
from __future__ import annotations
import json
from pathlib import Path
import re
from safetensors import safe_open
from .constants import STATE_KEYS
from .policy import ANIMA_SELECTIVE_POLICY, ALL_LINEAR_POLICY, FAMILIES, POLICY, policy_paths

MANIFEST_KEY = 'diffusion.mixed_nf4.manifest'
DTYPES = {'bfloat16': 'BF16', 'float32': 'F32', 'int64': 'I64', 'int32': 'I32',
          'int16': 'I16', 'int8': 'I8', 'uint8': 'U8', 'bool': 'BOOL', 'float64': 'F64'}


def _validate_manifest(m, family):
    fields = {'version', 'family', 'policy', 'compute_dtype', 'blocksize', 'double_quant',
              'modules', 'source_tensors', 'source_sha256', 'bitsandbytes_version'}
    if not isinstance(m, dict) or not set(m).issubset(fields | {'prefix'}) or not fields.issubset(m):
        raise ValueError('Invalid mixed NF4 manifest fields')
    prefix = m.get('prefix', '')
    if not isinstance(prefix, str) or (prefix and not prefix.endswith('.')):
        raise ValueError('Invalid mixed NF4 tensor prefix')
    if (type(m['version']) is not int or m['version'] != 1 or m['family'] not in FAMILIES
            or (family is not None and m['family'] != family)
            or m['policy'] not in (POLICY, ANIMA_SELECTIVE_POLICY, ALL_LINEAR_POLICY)
            or (m['policy'] == ALL_LINEAR_POLICY and m['family'] != 'qwen_image_2_1')
            or (m['policy'] == ANIMA_SELECTIVE_POLICY and m['family'] != 'anima')
            or m['compute_dtype'] != 'bfloat16' or type(m['blocksize']) is not int
            or m['blocksize'] != 64 or m['double_quant'] is not True
            or not isinstance(m['source_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', m['source_sha256'])
            or not isinstance(m['bitsandbytes_version'], str) or not m['bitsandbytes_version']):
        raise ValueError('Unsupported mixed NF4 family, version or policy')
    inventory = m['source_tensors']
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('Invalid mixed NF4 source inventory')
    # ``save_file`` writes metadata byte-for-byte.  Require canonical ordering
    # so two conversions of the same model produce the same manifest and so a
    # hand-edited inventory cannot hide duplicate/reordered entries.
    if list(inventory) != sorted(inventory):
        raise ValueError('Mixed NF4 source inventory must be sorted')
    for key, entry in inventory.items():
        if (not isinstance(key, str) or not key or not isinstance(entry, dict)
                or set(entry) != {'shape', 'dtype'} or entry['dtype'] not in DTYPES
                or not isinstance(entry['shape'], list)
                or any(type(d) is not int or d <= 0 for d in entry['shape'])):
            raise ValueError('Invalid mixed NF4 source tensor shape or dtype')
        if entry['dtype'].startswith('float') and entry['dtype'] != 'float32':
            raise ValueError('Retained floating tensors must be BF16 or FP32 buffers')
    if not isinstance(m['modules'], list) or any(not isinstance(p, str) for p in m['modules']):
        raise ValueError('Invalid mixed NF4 policy modules')
    if len(set(m['modules'])) != len(m['modules']):
        raise ValueError('Mixed NF4 policy modules must be unique')
    root = FAMILIES[m['family']][0]
    normalized_keys = {
        key.removeprefix(prefix) if prefix else key for key in inventory
    }
    ids = {
        key.split('.')[1] for key in normalized_keys if key.startswith(root + '.')
    }
    if m['policy'] in (POLICY, ANIMA_SELECTIVE_POLICY):
        if not ids or ids != {str(i) for i in range(len(ids))}:
            raise ValueError('Incomplete mixed NF4 main block inventory')
        policy_candidates = policy_paths(m['family'], len(ids), m['policy'])
        available_targets = [
            path for path in policy_candidates
            if prefix + path + '.weight' in inventory
        ]
        if m['modules'] != available_targets:
            raise ValueError('Incomplete mixed NF4 policy modules')
    else:
        candidates = [
            key.removeprefix(prefix).removesuffix('.weight') for key, entry in inventory.items()
            if key.startswith(prefix) and key.endswith('.weight') and len(entry['shape']) == 2
            and entry['dtype'] == 'bfloat16'
        ]
        if m['modules'] != sorted(candidates) or len(candidates) != len(set(candidates)):
            raise ValueError('Incomplete all-linear NF4 policy modules')
    for path in m['modules']:
        entry = inventory.get(prefix + path + '.weight', {})
        shape = entry.get('shape', [])
        if (len(shape) != 2 or any(d % 64 for d in shape) or entry.get('dtype') != 'bfloat16'):
            raise ValueError(f'Invalid mixed NF4 policy shape: {path}')
    return m


def inspect_mixed_nf4(path, *, family=None):
    """Inspect headers before materializing any weights; reject unmarked packed data."""
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    # Bound the header before safetensors maps the file; never read tensor payloads.
    with path.open('rb') as stream:
        prefix = stream.read(8)
    if len(prefix) != 8 or not 0 < int.from_bytes(prefix, 'little') <= 64 * 1024 * 1024:
        raise ValueError('Invalid or oversized safetensors header')
    with safe_open(str(path), framework='np') as f:
        metadata, keys = f.metadata() or {}, set(f.keys())
        raw = metadata.get(MANIFEST_KEY)
        if raw is None:
            # The established Krea format has its own strict loader and namespaced state.
            if metadata.get('krea2_nf4_format') == 'krea2_raw_nf4':
                return None
            # Legacy Anima artifacts use ``anima.mixed_nf4.manifest`` and are
            # validated by the Anima-specific loader.  They must not enter the
            # canonical four-family preflight contract.
            if any(str(key).startswith('anima.mixed_nf4') for key in metadata):
                return None
            if (any(k.startswith('diffusion.mixed_nf4') for k in metadata)
                    or any('.weight.' + suffix in k for k in keys for suffix in STATE_KEYS)):
                raise ValueError('Packed tensors require a mixed NF4 manifest')
            return None
        if not isinstance(raw, str) or len(raw) > 8_000_000:
            raise ValueError('Invalid or oversized mixed NF4 manifest')
        try:
            m = _validate_manifest(json.loads(raw), family)
        except (TypeError, KeyError, IndexError, ValueError) as exc:
            raise ValueError(f'Invalid mixed NF4 manifest: {exc}') from exc
        prefix = m.get('prefix', '')
        expected = set(m['source_tensors'])
        targets = {prefix + p + '.weight' for p in m['modules']}
        for key in targets:
            expected.update(key + '.' + s for s in STATE_KEYS)
        if keys != expected:
            raise ValueError('Mixed NF4 tensor/qstate inventory mismatch')
        for key, entry in m['source_tensors'].items():
            sl = f.get_slice(key)
            shape, dtype = entry['shape'], DTYPES[entry['dtype']]
            if key in targets:
                shape, dtype = [shape[0] * shape[1] // 2, 1], 'U8'
            if sl.get_shape() != shape or sl.get_dtype() != dtype:
                raise ValueError(f'Mixed NF4 tensor shape/dtype mismatch: {key}')
        for key in targets:
            rows, columns = m['source_tensors'][key]['shape']
            blocks = rows * columns // 64
            specs = {'absmax': ([blocks], 'U8'), 'quant_map': ([16], 'F32'),
                     'nested_absmax': ([(blocks + 255) // 256], 'F32'),
                     'nested_quant_map': ([256], 'F32')}
            for suffix, (shape, dtype) in specs.items():
                sl = f.get_slice(key + '.' + suffix)
                if sl.get_shape() != shape or sl.get_dtype() != dtype:
                    raise ValueError(f'Invalid NF4 qstate header: {key}.{suffix}')
            meta = f.get_slice(key + '.quant_state.bitsandbytes__nf4')
            shape = meta.get_shape()
            if meta.get_dtype() != 'U8' or len(shape) != 1 or not 0 < shape[0] <= 65536:
                raise ValueError(f'Invalid NF4 packed metadata header: {key}')
        return m


def validate_mixed_nf4_payload(path, *, family=None, manifest=None):
    """Validate serialized NF4 qstate contents after header inspection.

    Header inspection deliberately avoids importing bitsandbytes.  Conversion
    and verification tools that already opt into the optional runtime can use
    this second step to validate the packed quantization metadata itself.  The
    function is shared by every family, including Anima's legacy-aware loader.
    """
    if manifest is None:
        manifest = inspect_mixed_nf4(path, family=family)
    if manifest is None:
        return None
    from .nf4 import quant_state

    prefix = manifest.get('prefix', '')
    with safe_open(str(path), framework='pt', device='cpu') as f:
        for module_path in manifest['modules']:
            key = prefix + module_path + '.weight'
            shape = manifest['source_tensors'][key]['shape']
            state = {
                key + '.' + suffix: f.get_tensor(key + '.' + suffix)
                for suffix in STATE_KEYS
            }
            quant_state(state, key, shape)
    return manifest
