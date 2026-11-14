"""GPU exhaustive scaled-cast equivalence and bounded operator A/B (no model)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from library.quantization._scaled_cast import scaled_cast
except ImportError:
    scaled_cast = None


@torch.no_grad()
def check_values():
    raw = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.uint16)
    cases = []
    for source, target in ((torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)):
        value = raw.view(source)
        for exponent in (-126, -125, -64, -26, -16, -1, 0, 1, 16, 26, 64, 125, 126):
            shift = torch.tensor(exponent, device='cuda', dtype=torch.int32)
            expected = torch.ldexp(value.float(), shift).to(target)
            actual = scaled_cast(value, shift, target)
            same_nan = torch.isnan(expected) & torch.isnan(actual)
            exact = expected.view(torch.uint16) == actual.view(torch.uint16)
            failures = int((~(same_nan | exact)).sum().item())
            assert failures == 0, (str(source), exponent, failures)
            cases.append(dict(source=str(source), shift=exponent, mismatches=failures))
    # Noncontiguous input must keep the torch fallback.
    value = torch.randn(64, 32, device='cuda', dtype=torch.bfloat16).t()
    shift = torch.tensor(20, device='cuda', dtype=torch.int32)
    assert torch.equal(scaled_cast(value, shift, torch.float16),
                       torch.ldexp(value.float(), shift).to(torch.float16))
    return cases


@torch.no_grad()
def time_ops():
    results = []
    for source, target, exp, amplitude in (
        (torch.bfloat16, torch.float16, 26, 1e-8),
        (torch.float16, torch.bfloat16, -26, 1.0),
    ):
        value = torch.randn(4608, 6144, device='cuda', dtype=source) * amplitude
        shift = torch.tensor(exp, device='cuda', dtype=torch.int32)
        reference = lambda: torch.ldexp(value.float(), shift).to(target)
        candidate = lambda: scaled_cast(value, shift, target)
        for _ in range(2):
            reference(); candidate()
        times = {'reference': [], 'candidate': []}
        for iteration in range(9):
            arms = [('reference', reference), ('candidate', candidate)]
            if iteration % 2:
                arms.reverse()
            for name, fn in arms:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); result = fn(); end.record(); end.synchronize()
                times[name].append(start.elapsed_time(end))
                del result
        results.append(dict(source=str(source), shape=list(value.shape), milliseconds=times,
                            medians_ms={k: statistics.median(v) for k,v in times.items()}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if scaled_cast is None:
        parser.error('Candidate was withdrawn; apply scaled_cast_candidate.patch in an isolated tree first')
    assert not args.output.exists(), args.output
    torch.cuda.reset_peak_memory_stats()
    cases = check_values()
    timings = time_ops()
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  bit_patterns_per_case=65536, cases=cases, timings=timings,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  note='Exact finite/Inf/signed-zero bits; NaNs compare as NaNs. Operator timings are NOT full-step gains.')
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
