"""Read an nsys SQLite export without executing training or changing the DB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def analyze(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise ValueError("No CUPTI kernel table: this report is not GPU evidence")
        kernels = db.execute(
            "SELECT k.start,k.end,k.correlationId,s.value AS name FROM "
            "CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
            "ORDER BY k.start"
        ).fetchall()
        if not kernels:
            raise ValueError("Empty CUPTI kernel table")
        nvtx = []
        if "NVTX_EVENTS" in tables:
            nvtx = db.execute(
                "SELECT n.start,n.end,COALESCE(n.text,s.value) AS name FROM "
                "NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId "
                "WHERE n.end IS NOT NULL ORDER BY n.start"
            ).fetchall()
        step = [r for r in nvtx if str(r['name']).startswith('step=')]
        window = (step[0]['start'], step[0]['end']) if len(step) == 1 else None
        totals = {}
        for r in kernels:
            entry = totals.setdefault(r['name'], {'count': 0, 'total_ms': 0.0})
            entry['count'] += 1
            entry['total_ms'] += (r['end'] - r['start']) / 1e6
        intervals = [(r['start'], r['end']) for r in kernels]
        busy = union_ns(intervals)
        copies = []
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            copies = [(r[0], r[1]) for r in db.execute(
                'SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start')]
        gaps = []
        end = kernels[0]['end']
        for r in kernels[1:]:
            if r['start'] > end:
                gaps.append({'start_ns': end, 'end_ns': r['start'],
                             'ms': (r['start'] - end) / 1e6, 'next_kernel': r['name']})
            end = max(end, r['end'])
        result = {
            'source_sqlite': str(path), 'kernel_count': len(kernels),
            'kernel_sum_ms': sum((b-a)/1e6 for a,b in intervals),
            'kernel_union_ms': busy/1e6,
            'kernel_span_ms': (max(b for _,b in intervals)-min(a for a,_ in intervals))/1e6,
            'memcpy_union_ms': union_ns(copies)/1e6,
            'kernel_or_memcpy_union_ms': union_ns(intervals+copies)/1e6,
            'nvtx_step_count': len(step),
            'top_kernels': [dict(name=k, **v) for k,v in
                            sorted(totals.items(), key=lambda kv: -kv[1]['total_ms'])[:20]],
            'largest_kernel_gaps': sorted(gaps, key=lambda g: -g['ms'])[:12],
            'host_nvtx_ranges': [dict(name=r['name'], ms=(r['end']-r['start'])/1e6)
                                 for r in nvtx if r['name'] in {'forward','backward','optimizer'}
                                 or str(r['name']).startswith('step=')],
            'caveat': 'Host NVTX spans are not per-stage GPU time; kernel busy is not SM occupancy.',
        }
        if window:
            lo, hi = window
            clipped = [(max(a,lo),min(b,hi)) for a,b in intervals if a < hi and b > lo]
            result.update(nvtx_step_ms=(hi-lo)/1e6,
                          kernel_busy_fraction_in_step=union_ns(clipped)/(hi-lo))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new path')
    result = analyze(args.database)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
