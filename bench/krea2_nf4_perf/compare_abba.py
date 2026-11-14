"""Attribute divergence: A/A, B/B and A/B comparisons of captured final states."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from safetensors import safe_open


def compare(reference: Path, candidate: Path):
    groups = {k: dict(tensors=0, elements=0, unequal=0, max_abs=0.0,
                      diff_sq=0.0, ref_sq=0.0, cand_sq=0.0, dot=0.0, finite=True)
              for k in ('parameter', 'gradient')}
    with safe_open(str(reference), framework='numpy') as a, \
         safe_open(str(candidate), framework='numpy') as b:
        keys = list(a.keys())
        assert set(keys) == set(b.keys()), 'tensor key mismatch'
        for key in keys:
            group = groups[key.split('.', 1)[0]]
            x, y = a.get_tensor(key), b.get_tensor(key)
            assert x.shape == y.shape and x.dtype == y.dtype, key
            group['tensors'] += 1
            group['elements'] += x.size
            group['unequal'] += int(np.count_nonzero(x != y))
            group['finite'] &= bool(np.isfinite(x).all() and np.isfinite(y).all())
            xx, yy = x.astype(np.float64).reshape(-1), y.astype(np.float64).reshape(-1)
            delta = xx - yy
            group['max_abs'] = max(group['max_abs'], float(np.max(np.abs(delta))) if delta.size else 0.0)
            group['diff_sq'] += float(np.dot(delta, delta))
            group['ref_sq'] += float(np.dot(xx, xx))
            group['cand_sq'] += float(np.dot(yy, yy))
            group['dot'] += float(np.dot(xx, yy))
    for group in groups.values():
        assert group['tensors'] > 0
        group['rel_l2'] = math.sqrt(group['diff_sq'] / max(group['ref_sq'], 1e-300))
        denom = math.sqrt(group['ref_sq'] * group['cand_sq'])
        group['cosine'] = group['dot'] / denom if denom else None
    return groups


def timings(root: Path, name: str):
    rows = json.loads((root / name / 'timings.json').read_text())['rows']
    assert [r['step'] for r in rows] == list(range(6)), name
    measured = [r['seconds'] for r in rows[1:]]
    return dict(median_s=statistics.median(measured), mean_s=statistics.mean(measured),
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows),
                losses=[r['loss'] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    names = ('reference-0', 'candidate-0', 'reference-1', 'candidate-1-retry')
    for name in names:
        assert (args.root / name / 'final.safetensors').is_file(), name
    result = {'runs': {name: timings(args.root, name) for name in names}, 'comparisons': {}}
    pairs = {
        'AA_reference': ('reference-0', 'reference-1'),
        'BB_candidate': ('candidate-0', 'candidate-1-retry'),
        'AB_pair0': ('reference-0', 'candidate-0'),
        'AB_pair1': ('reference-1', 'candidate-1-retry'),
    }
    for label, (ref, cand) in pairs.items():
        result['comparisons'][label] = compare(
            args.root / ref / 'final.safetensors', args.root / cand / 'final.safetensors')
    ref_med = [result['runs'][n]['median_s'] for n in ('reference-0', 'reference-1')]
    cand_med = [result['runs'][n]['median_s'] for n in ('candidate-0', 'candidate-1-retry')]
    result['pooled_median_s'] = dict(reference=statistics.median(ref_med),
                                     candidate=statistics.median(cand_med))
    r, c = result['pooled_median_s'].values()
    result['latency_reduction_pct'] = (1 - c / r) * 100
    result['throughput_increase_pct'] = (r / c - 1) * 100
    result['scope'] = ('four six-step real runs, same cached image and seed; '
                       'final-state comparisons only, no long-run claim')
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
