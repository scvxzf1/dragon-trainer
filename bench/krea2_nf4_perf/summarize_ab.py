"""Summarize completed real-training ABBA timings and CPU tensor comparisons."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
from safetensors import safe_open


def compare_tensors(reference, candidate):
    groups = {k: dict(tensors=0, elements=0, unequal=0, max_abs=0.,
                      diff_sq=0., ref_sq=0., candidate_sq=0., dot=0., finite=True)
              for k in ('parameter', 'gradient')}
    with safe_open(reference, framework='numpy') as a, safe_open(candidate, framework='numpy') as b:
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
            group['max_abs'] = max(group['max_abs'], float(np.max(np.abs(delta))))
            group['diff_sq'] += float(np.dot(delta, delta))
            group['ref_sq'] += float(np.dot(xx, xx))
            group['candidate_sq'] += float(np.dot(yy, yy))
            group['dot'] += float(np.dot(xx, yy))
    for group in groups.values():
        assert group['tensors'] > 0, 'missing parameter/gradient evidence'
        group['rel_l2'] = math.sqrt(group['diff_sq'] / max(group['ref_sq'], 1e-300))
        denom = math.sqrt(group['ref_sq'] * group['candidate_sq'])
        group['cosine'] = group['dot'] / denom if denom else None
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    names = ('reference-0','candidate-0','candidate-1','reference-1')
    result = {'order': list(names), 'warmup_steps': 1, 'runs': {}}
    pooled = {'reference': [], 'candidate': []}
    losses = {}
    for name in names:
        data = json.loads((args.root/name/'timings.json').read_text())
        rows = data['rows']
        assert [r['step'] for r in rows] == list(range(6)), (name, rows)
        assert all(math.isfinite(r['loss']) and r['seconds'] > 0 for r in rows), name
        values = [r['seconds'] for r in rows[1:]]
        pooled[name.split('-')[0]].extend(values)
        losses[name] = [r['loss'] for r in rows]
        result['runs'][name] = dict(seconds=values, median_s=statistics.median(values),
                                   mean_s=statistics.mean(values), losses=losses[name],
                                   peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows))
    result['pooled_median_s'] = {k: statistics.median(v) for k,v in pooled.items()}
    result['pooled_mean_s'] = {k: statistics.mean(v) for k,v in pooled.items()}
    ref, cand = (result['pooled_median_s'][k] for k in ('reference','candidate'))
    result['latency_reduction_pct'] = (1-cand/ref)*100
    result['throughput_increase_pct'] = (ref/cand-1)*100
    result['loss_max_relative_difference'] = max(abs(x-y)/max(abs(x),1e-30)
        for x,y in zip(losses['reference-0'],losses['candidate-0']))
    result['numerical'] = compare_tensors(str(args.root/'reference-0/final.safetensors'),
                                         str(args.root/'candidate-0/final.safetensors'))
    result['scope'] = 'one real cached image, six steps, same GPU ABBA; no long-run quality or thermal claim'
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
