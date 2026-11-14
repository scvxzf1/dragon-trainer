"""Time real train.py microbatches and optionally capture final adapter/gradients.

Only instrumentation and a process-local reference arm: the production trainer,
optimizer, cache loader and forward/backward paths are reused without a new loop.
All tensor copies for numerical comparison occur AFTER the timed final step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('reference', 'candidate'), required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--capture-final', type=Path)
    opts, train_args = parser.parse_known_args()
    if opts.result.exists() or (opts.capture_final and opts.capture_final.exists()):
        parser.error('evidence output exists; use new paths')
    try:
        from library.quantization import _scaled_cast
    except ImportError:
        _scaled_cast = None
    from library.training import loop

    if opts.arm == 'candidate':
        import inspect
        from library.quantization.nf4 import _Dir1MatMul4Bit
        if _scaled_cast is None or 'scaled_cast(' not in inspect.getsource(_Dir1MatMul4Bit.backward):
            parser.error('Candidate was withdrawn; apply scaled_cast_candidate.patch in an isolated tree first')
    elif _scaled_cast is not None:
        _scaled_cast.scaled_cast = lambda x, shift, dtype: torch.ldexp(x.float(), shift).to(dtype)
    original = loop._run_microbatch
    rows = []

    def measured(trainer, state, batch, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original(trainer, state, batch, **kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        rows.append(dict(step=int(state.global_step), seconds=seconds,
                         loss=float(result),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        payload = dict(arm=opts.arm, gpu=torch.cuda.get_device_name(),
                       torch=torch.__version__, rows=rows,
                       scope='production _run_microbatch, synchronized; excludes data loading and step bookkeeping')
        # This is a new run-owned result, updated between steps, never an old asset.
        opts.result.write_text(json.dumps(payload, indent=2) + '\n')
        if state.global_step + 1 == state.args.max_train_steps and opts.capture_final:
            from safetensors.torch import save_file
            net = state.accelerator.unwrap_model(state.network)
            tensors = {}
            for name, parameter in net.named_parameters():
                if parameter.requires_grad:
                    tensors['parameter.'+name] = parameter.detach().float().cpu().contiguous()
                    if parameter.grad is not None:
                        tensors['gradient.'+name] = parameter.grad.detach().float().cpu().contiguous()
            save_file(tensors, str(opts.capture_final))
        return result

    loop._run_microbatch = measured
    sys.argv = [str(ROOT/'train.py'), *train_args]
    runpy.run_path(str(ROOT/'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
