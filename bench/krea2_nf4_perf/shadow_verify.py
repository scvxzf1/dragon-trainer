"""Shadow-verify scaled_cast bit-equality on real production inputs during training.

Wraps the shipped candidate implementation: every call also computes the torch
reference on the same live inputs and compares bits. This is a diagnostic run —
step timing is meaningless here; pass/fail is the bit comparison count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--max-mismatches', type=int, default=5)
    opts, train_args = parser.parse_known_args()
    if opts.result.exists():
        parser.error('evidence output exists; use a new path')

    import library.quantization._scaled_cast as scaled_cast_module

    candidate_impl = scaled_cast_module.scaled_cast
    stats = {'invocations': 0, 'bit_equal': 0, 'nan_class_equal': 0,
             'fallback_fast_path': 0, 'mismatches': [], 'shifts_seen': set()}

    def shadow(value, shift, dtype):
        candidate = candidate_impl(value, shift, dtype)
        reference = torch.ldexp(value.float(), shift).to(dtype)
        stats['invocations'] += 1
        same_bits = bool((candidate.view(torch.uint16) == reference.view(torch.uint16)).all())
        both_nan = bool(torch.isnan(candidate).logical_and(torch.isnan(reference)).all()) \
            if candidate.numel() else False
        if same_bits:
            stats['bit_equal'] += 1
        elif both_nan:
            stats['nan_class_equal'] += 1
        elif len(stats['mismatches']) < opts.max_mismatches:
            stats['mismatches'].append({
                'dtype': str(dtype), 'source_dtype': str(value.dtype),
                'shape': list(value.shape), 'contiguous': value.is_contiguous(),
                'shift': float(shift.float().item()) if shift.numel() == 1 else None,
                'mismatched_elements': int((candidate.view(torch.uint16) != reference.view(torch.uint16)).sum()),
            })
        try:
            stats['shifts_seen'].add(int(shift.item()))
        except Exception:
            pass
        if not value.is_contiguous() or value.is_neg() or value.numel() == 0:
            stats['fallback_fast_path'] += 1
        return candidate

    scaled_cast_module.scaled_cast = shadow
    sys.argv = [str(ROOT / 'train.py'), *train_args]
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    stats['shifts_seen'] = sorted(stats['shifts_seen'])
    stats['verdict'] = 'PASS' if not stats['mismatches'] else 'FAIL'
    with opts.result.open('x') as f:
        json.dump(stats, f, indent=2)
        f.write('\n')
    print(json.dumps(stats, indent=2))
    raise SystemExit(0 if not stats['mismatches'] else 1)


if __name__ == '__main__':
    main()
