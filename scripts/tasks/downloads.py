"""Model download entry-points (Anima base, SAM3, MIT, PE-Core, Tagger vocab).

All targets shell out to ``hf download`` (rather than the SDK) so the user's
``hf auth login`` cache is honored.

Idempotency contract (see GH #21): every target skips when its final
destination files already exist, so a re-run *verifies* rather than re-fetching
gigabytes. This matters because several targets ``shutil.move`` files out of
``hf``'s ``--local-dir`` layout after download — once moved, ``hf download``
no longer sees them at the path it checks and would otherwise re-pull the whole
repo. Pass ``--force`` (e.g. ``make download-anima ARGS=--force``) to re-fetch
regardless. ``download-models`` continues past a failed component (a gated SAM3
without granted access shouldn't abort the Anima download) and reports the
failures at the end.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ._common import ROOT, run


_TAGGER_DOWNLOAD_TIMEOUT_SECONDS = 120
_TAGGER_REQUEST_TIMEOUT_SECONDS = 30


def _present(paths: list[Path]) -> bool:
    """True when every expected destination path already exists."""
    return all(p.exists() for p in paths)


def _skip(name: str, paths: list[Path], extra) -> bool:
    """Return True (caller should skip) when files exist and ``--force`` absent."""
    if "--force" in (extra or []):
        return False
    if _present(paths):
        print(f"  ✓ {name} already present (pass --force to re-download)")
        return True
    return False


def cmd_download_sam3(_extra):
    dst = ROOT / "models" / "sam3"
    # SAM3 is a gated repo; the full snapshot lands a config.json + weights.
    if _skip("SAM3", [dst / "config.json"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(["hf", "download", "facebook/sam3", "--local-dir", "models/sam3"])


def cmd_download_pe(_extra):
    # PE-Core-L14-336 — only the .pt checkpoint is needed; vision tower is
    # vendored at library/models/pe.py (no perception_models clone required).
    dst = ROOT / "models" / "pe"
    if _skip("PE-Core", [dst / "PE-Core-L14-336.pt"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "facebook/PE-Core-L14-336",
            "PE-Core-L14-336.pt",
            "--local-dir",
            "models/pe",
        ]
    )


def cmd_download_pe_spatial(_extra):
    # PE-Spatial-B16-512 — auxiliary encoder for the Anima Tagger's
    # dual-encoder configuration. Same vendored vision tower (different
    # config entry); only the .pt is fetched here.
    dst = ROOT / "models" / "pe"
    if _skip("PE-Spatial", [dst / "PE-Spatial-B16-512.pt"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "facebook/PE-Spatial-B16-512",
            "PE-Spatial-B16-512.pt",
            "--local-dir",
            "models/pe",
        ]
    )


def cmd_download_tagger(_extra):
    # Just the Anima Tagger v2 ``vocab.json`` (~0.7 MB) — the only piece
    # ``caption-index`` needs to classify tags. The full tagger model is not
    # fetched here (train it locally or pull it separately); this deliberately
    # won't clobber a local ``model.safetensors``.
    dst = ROOT / "models" / "captioners" / "anima-tagger-v2"
    if _skip("Anima Tagger vocab", [dst / "vocab.json"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    request_timeout = str(_TAGGER_REQUEST_TIMEOUT_SECONDS)
    env.setdefault("HF_HUB_ETAG_TIMEOUT", request_timeout)
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", request_timeout)
    try:
        run(
            [
                "hf",
                "download",
                "sorryhyun/anima-tagger",
                "v2/vocab.json",
                "--local-dir",
                "models/captioners/anima-tagger-v2",
            ],
            env=env,
            timeout=_TAGGER_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        print(
            "  ✗ Anima Tagger vocab download timed out after "
            f"{_TAGGER_DOWNLOAD_TIMEOUT_SECONDS}s"
        )
        raise SystemExit(124) from e
    # The file lands under the repo's ``v2/`` prefix; flatten it up one level.
    sub = dst / "v2"
    if sub.exists():
        for f in sub.iterdir():
            target = dst / f.name
            if target.exists():
                target.unlink()
            shutil.move(str(f), str(target))
        sub.rmdir()


def cmd_download_mit(_extra):
    dst = ROOT / "models" / "mit"
    if _skip("MIT", [dst / "model.pth"], _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "a-b-c-x-y-z/Manga-Text-Segmentation-2025",
            "model.pth",
            "--local-dir",
            "models/mit",
        ]
    )


def cmd_download_anima(_extra):
    models = ROOT / "models"
    # Final (post-move) destinations — this is what we verify against, NOT the
    # transient split_files/ layout hf downloads into (see module docstring).
    finals = [
        models / "diffusion_models" / "anima-base-v1.0.safetensors",
        models / "text_encoders" / "qwen_3_06b_base.safetensors",
        models / "vae" / "qwen_image_vae.safetensors",
    ]
    if _skip("Anima base (DiT + TE + VAE, ~5GB)", finals, _extra):
        return
    for d in ["diffusion_models", "text_encoders", "vae"]:
        (models / d).mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "circlestone-labs/Anima",
            "split_files/diffusion_models/anima-base-v1.0.safetensors",
            "split_files/text_encoders/qwen_3_06b_base.safetensors",
            "split_files/vae/qwen_image_vae.safetensors",
            "--local-dir",
            "models",
            "--include",
            "split_files/*",
        ]
    )
    split = models / "split_files"
    for subdir in ["diffusion_models", "text_encoders", "vae"]:
        src = split / subdir
        dst = models / subdir
        if src.exists():
            for f in src.iterdir():
                shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)


# Official base-model variants beyond anima-base-v1.0. Same 685-key DiT
# architecture — they differ only in weights and in the state-dict prefix
# ("model.diffusion_model." vs base's "net."), which the loader strips either
# way (library/anima/weights.py::_DIT_PREFIXES). So any of these is a valid
# LoRA training base.
ANIMA_VARIANTS = (
    "anima-aesthetic-v1.0",
    "anima-aesthetic-v1.0b",
    "anima-aesthetic-v1.1",
    "anima-turbo-v1.0",
    "anima-preview3-base",
)


def cmd_download_anima_variant(_extra):
    """Download an alternate Anima base DiT (aesthetic / turbo / preview)."""
    names = [a for a in (_extra or []) if not a.startswith("-")]
    if not names:
        print("Usage: make download-anima-variant ARGS=<name> [<name>...]")
        print("Available: " + ", ".join(ANIMA_VARIANTS))
        return
    unknown = [n for n in names if n not in ANIMA_VARIANTS]
    if unknown:
        raise SystemExit(
            f"Unknown Anima variant(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(ANIMA_VARIANTS)}"
        )
    models = ROOT / "models"
    dst = models / "diffusion_models"
    finals = [dst / f"{n}.safetensors" for n in names]
    if _skip(f"Anima variant(s) {', '.join(names)}", finals, _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "circlestone-labs/Anima",
            *[f"split_files/diffusion_models/{n}.safetensors" for n in names],
            "--local-dir",
            "models",
        ]
    )
    split = models / "split_files"
    src = split / "diffusion_models"
    if src.exists():
        for f in src.iterdir():
            shutil.move(str(f), str(dst / f.name))
    if split.exists():
        shutil.rmtree(split)
    print(
        "\nTrain against one with:\n"
        f"  make lora ARGS='--pretrained_model_name_or_path "
        f"models/diffusion_models/{names[0]}.safetensors'"
    )


def cmd_download_krea2(_extra):
    # Krea-2-Raw base checkpoint for the krea2_raw model family
    # (_archive/docs/proposal/krea2_raw_migration.md). Source is the Comfy-Org
    # repackaged single-file layout — NOT the gated `krea/Krea-2-Raw`
    # diffusers three-shard form, so no license acceptance is needed and the
    # loader is single-file key-mapping (not index.json shard expansion).
    # VAE filename `qwen_image_vae.safetensors` matches anima's exactly — same
    # AutoencoderKLQwenImage (R2 encode/decode interop still needs verifying,
    # but the file is byte-identical to anima's VAE download).
    models = ROOT / "models"
    finals = [
        models / "diffusion_models" / "krea2_raw_bf16.safetensors",
        models / "text_encoders" / "qwen3vl_4b_bf16.safetensors",
        models / "vae" / "qwen_image_vae.safetensors",
    ]
    if _skip("Krea-2-Raw (DiT bf16 + Qwen3-VL + VAE, ~35GB)", finals, _extra):
        return
    for d in ["diffusion_models", "text_encoders", "vae"]:
        (models / d).mkdir(parents=True, exist_ok=True)
    # Comfy-Org lays files directly under diffusion_models/ / text_encoders/ /
    # vae/, so `--local-dir models` lands them in the right place — no
    # split_files/ move step needed (unlike cmd_download_anima).
    run(
        [
            "hf",
            "download",
            "Comfy-Org/Krea-2",
            "diffusion_models/krea2_raw_bf16.safetensors",
            "text_encoders/qwen3vl_4b_bf16.safetensors",
            "vae/qwen_image_vae.safetensors",
            "--local-dir",
            "models",
        ]
    )


def cmd_download_sketch2manga(_extra):
    """Sketch2Manga screening weights for colorize EasyControl."""
    dst = ROOT / "models" / "sketch2manga"
    finals = [
        dst / "mangatone.ckpt",
        dst / "vae" / "mangatone_default.ckpt",
        dst / "control_v11p_sd15_lineart.pth",
    ]
    if _skip("Sketch2Manga screening weights (~5.7GB)", finals, _extra):
        return
    dst.mkdir(parents=True, exist_ok=True)
    run(
        [
            "hf",
            "download",
            "dreMaz/sketch2manga",
            "mangatone.ckpt",
            "vae/mangatone_default.ckpt",
            "--local-dir",
            "models/sketch2manga",
        ]
    )
    run(
        [
            "hf",
            "download",
            "lllyasviel/ControlNet-v1-1",
            "control_v11p_sd15_lineart.pth",
            "--local-dir",
            "models/sketch2manga",
        ]
    )


def cmd_download_models(_extra):
    # Continue-on-failure: a gated component the user hasn't been granted
    # (SAM3) or hasn't authed for must not abort the rest, and — crucially —
    # must not force a re-download of the components that DID succeed on the
    # retry (each is skip-if-present now). ``run`` calls ``sys.exit`` on a
    # non-zero subprocess, so we catch ``SystemExit`` per component.
    components = [
        ("Anima base", cmd_download_anima),
        ("Krea-2-Raw", cmd_download_krea2),
        ("SAM3 (gated)", cmd_download_sam3),
        ("MIT", cmd_download_mit),
        ("PE-Core", cmd_download_pe),
        ("PE-Spatial", cmd_download_pe_spatial),
        ("Anima Tagger vocab", cmd_download_tagger),
    ]
    failed: list[str] = []
    for name, fn in components:
        try:
            fn(_extra)
        except SystemExit as e:
            if e.code:
                failed.append(name)
                print(f"  ✗ {name} failed (exit {e.code}); continuing")
    if failed:
        print()
        print("The following downloads did not complete:")
        for name in failed:
            print(f"  - {name}")
        print()
        print("Common causes:")
        print("  - not authenticated: run `hf auth login` and re-run")
        print(
            "  - SAM3 is gated: request access at https://huggingface.co/facebook/sam3"
        )
        print("Successful components are cached; re-running only retries the failures.")
        raise SystemExit(1)
