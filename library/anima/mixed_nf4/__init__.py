"""Experimental, fixed-policy Anima BF16/NF4 checkpoint support."""
from safetensors import safe_open

from library.quantization.format import inspect_mixed_nf4 as inspect_canonical_mixed_nf4
from .format import inspect_mixed_nf4 as inspect_legacy_mixed_nf4, quantized_paths
from .conversion import convert_checkpoint, verify_checkpoint
from .runtime import load_mixed_state


def inspect_mixed_nf4(path):
    with safe_open(str(path), framework='pt', device='cpu') as f:
        metadata = f.metadata() or {}
    if 'diffusion.mixed_nf4.manifest' in metadata:
        return inspect_canonical_mixed_nf4(path, family='anima')
    return inspect_legacy_mixed_nf4(path)

__all__ = ['inspect_mixed_nf4', 'quantized_paths', 'convert_checkpoint', 'verify_checkpoint', 'load_mixed_state']
