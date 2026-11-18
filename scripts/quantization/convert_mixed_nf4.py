"""Convert a local, ordinary BF16 transformer file to canonical mixed NF4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SUPPORTED_FAMILIES = ("anima", "krea2_raw", "z_image", "qwen_image_2_1")


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(*, family, source, output, threads=None, policy="main-mlp"):
    import torch
    from safetensors import safe_open
    from library.quantization.format import inspect_mixed_nf4
    from library.quantization.mixed_nf4 import save_mixed_nf4
    source, output = Path(source).expanduser(), Path(output).expanduser()
    if policy not in ("main-mlp", "all-linear"):
        raise ValueError("policy must be main-mlp or all-linear")
    if family == "anima" and policy != "main-mlp":
        raise ValueError("Anima supports only its fixed selective MLP policy")
    if policy == "all-linear" and family != "qwen_image_2_1":
        raise ValueError("all-linear policy is supported only for qwen_image_2_1")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not source.is_file() or source.suffix != ".safetensors":
        raise ValueError("Source must be a local single BF16 .safetensors file; directories are unsupported")
    if threads is not None:
        if threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
    if inspect_mixed_nf4(source) is not None:
        raise ValueError("Source is already mixed NF4")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        if (any("nf4" in key.lower() or "quant" in key.lower() for key in (handle.metadata() or {}))
                or any(handle.get_slice(key).get_dtype() != "BF16" for key in handle.keys())):
            raise ValueError("Source must contain ordinary BF16 tensors, without quantization")
    source_digest = _source_sha256(source)
    if family == "anima":
        from library.anima.mixed_nf4 import convert_checkpoint

        return convert_checkpoint(source, output, device="cpu", threads=threads)
    if family == "krea2_raw":
        from library.models.krea2_raw.weights import load_krea2_dit
        model = load_krea2_dit(source, dtype=torch.bfloat16, device="cpu")
    elif family == "z_image":
        from library.models.z_image.weights import load_z_image_transformer
        model = load_z_image_transformer(str(source), dtype=torch.bfloat16, device="cpu")
    elif family == "qwen_image_2_1":
        from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer
        model = load_qwen_image_2_1_transformer(str(source), dtype=torch.bfloat16, device="cpu")
    else:
        raise ValueError(f"Unsupported family: {family}")
    kwargs = {"family": family, "source_sha256": source_digest}
    if policy == "all-linear":
        kwargs["policy"] = "all-linear-nf4-v1"
    return save_mixed_nf4(model, output, **kwargs)


def verify(*, family, output, source=None):
    """Verify a canonical artifact without loading a model or rewriting it."""
    from library.quantization.format import (
        inspect_mixed_nf4,
        validate_mixed_nf4_payload,
    )

    output = Path(output).expanduser()
    if not output.is_file():
        raise FileNotFoundError(f"Mixed NF4 artifact does not exist: {output}")
    if family == "anima":
        from library.anima.mixed_nf4 import verify_checkpoint

        return verify_checkpoint(output, source)
    manifest = inspect_mixed_nf4(output, family=family)
    if manifest is None:
        raise ValueError(f"Not a canonical mixed NF4 artifact for {family}: {output}")
    validate_mixed_nf4_payload(output, family=family, manifest=manifest)
    return {
        "output": str(output),
        "family": family,
        "policy": manifest["policy"],
        "quantized_weights": len(manifest["modules"]),
        "source_sha256": manifest["source_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=SUPPORTED_FAMILIES)
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--policy", choices=("main-mlp", "all-linear"), default="main-mlp")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    if args.family not in SUPPORTED_FAMILIES:
        parser.error(f"unsupported mixed NF4 family: {args.family}")
    if args.verify_only:
        result = verify(family=args.family, output=args.output, source=args.source)
    else:
        if not args.source:
            parser.error("--source is required unless --verify-only is used")
        result = convert(
            family=args.family,
            source=args.source,
            output=args.output,
            threads=args.threads,
            policy=args.policy,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
