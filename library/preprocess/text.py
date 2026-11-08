"""Cache text-encoder (Qwen3) outputs and the pooled-text sidecar.

Orchestration extracted from ``preprocess/cache_text_embeddings.py`` and
``preprocess/cache_pooled_text.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The scripts keep argparse + model
load + uncond staging; the caption-variant generation, the batched
tokenize→encode→(LLM-adapter)→save loop, and the pooled reduction live here.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch
from PIL import Image

from library import cache_identity as _identity
from library.io.cache import POOLED_CACHE_SUFFIX, TE_CACHE_SUFFIX, resolve_cache_path
from library.models.family_registry import get_model_family_spec
from library.preprocess.captions import CaptionSource, read_caption_source
from library.preprocess._dataset import PreprocessStats, walk_images
from library.preprocess._progress import ProgressFn
from library.preprocess.adaptive_batch import AutoBatcher, batch_size_arg, iter_batches
from library.training import numerical_observation
from library.training.prior_preservation import build_diff_output_prior_caption

logger = logging.getLogger(__name__)


def generate_caption_variants(
    caption: str | CaptionSource, num_variants: int, tag_dropout_rate: float
) -> list[str]:
    """Generate ``num_variants`` caption variants for stochastic train-time sampling.

    v0 = pristine original caption. v1..v{N-1} are smart-shuffled (preserving
    the @artist prefix and section anchors), then every tag *after* the prefix
    is independently dropped with probability ``tag_dropout_rate``. The
    ``@no-artist`` sentinel participates in the boundary but is stripped from
    every variant (including v0) before it is written.
    """
    if isinstance(caption, CaptionSource):
        if caption.captions is not None:
            return caption.caption_texts() or [""]
        if caption.structured is not None:
            return caption.structured.generate_variants(num_variants, tag_dropout_rate)
        caption = caption.text

    from library.anima import training as anima_train_utils

    sentinel = anima_train_utils.NO_ARTIST_SENTINEL

    tags = [t.strip() for t in caption.split(",")]
    split_idx = anima_train_utils.find_anima_prefix_end(tags)

    # v0 stays byte-identical to the source caption unless the sentinel is
    # actually present — re-joining would normalize whitespace around commas
    # for every existing dataset otherwise.
    if sentinel in tags:
        variants = [", ".join(anima_train_utils.strip_no_artist_sentinel(tags))]
    else:
        variants = [caption]

    for _ in range(max(0, num_variants - 1)):
        shuffled = anima_train_utils.anima_smart_shuffle_caption(tags.copy())
        if tag_dropout_rate > 0.0 and len(shuffled) > split_idx:
            kept = list(shuffled[:split_idx])
            for tag in shuffled[split_idx:]:
                if random.random() >= tag_dropout_rate:
                    kept.append(tag)
            if not kept:
                kept = shuffled[:1]
            shuffled = kept
        shuffled = anima_train_utils.strip_no_artist_sentinel(shuffled)
        variants.append(", ".join(shuffled))
    return variants


def _caption_variants_for_cache(
    caption: CaptionSource,
    num_variants: int,
    tag_dropout_rate: float,
) -> list[str]:
    if caption.captions is not None:
        return caption.caption_texts() or [""]
    if num_variants <= 0:
        return [caption.render()]
    return generate_caption_variants(caption, num_variants, tag_dropout_rate)


def _nonfinite_summary(tensor: torch.Tensor) -> dict[str, int]:
    """``{nan, inf, nonfinite, elements}`` of ``tensor`` in one host sync."""
    nan, inf = torch.stack(
        [torch.count_nonzero(torch.isnan(tensor)), torch.count_nonzero(torch.isinf(tensor))]
    ).tolist()
    return {
        "nan": int(nan),
        "inf": int(inf),
        "nonfinite": int(nan) + int(inf),
        "elements": int(tensor.numel()),
    }


def _report_cast_numeric(name: str, source: torch.Tensor, stored: torch.Tensor, dtype) -> None:
    """Count a text-cache tensor's non-finite values, split by cause.

    Two layers that used to be indistinguishable from the outside: values that were
    already non-finite *before* the storage cast (upstream encoder result: NaN or
    ±Inf) and values the cast itself turned non-finite (narrowing overflow — a finite
    source that does not fit the storage dtype). Observation only: the sidecar is
    still written exactly as before (a rejection policy is a separate user decision,
    not implemented here).
    """
    stored_summary = _nonfinite_summary(stored)
    if not stored_summary["nonfinite"]:
        return
    source_summary = _nonfinite_summary(source)
    site = f"preprocess.text.{name}"
    common = {
        "elements": stored_summary["elements"],
        "before_dtype": source.dtype,
        "after_dtype": dtype,
        "shape": tuple(stored.shape),
        "identity": name,
    }
    if source_summary["nan"]:
        numerical_observation.record(
            site,
            numerical_observation.KIND_SOURCE_NAN,
            nonfinite=source_summary["nan"],
            **common,
        )
    if source_summary["inf"]:
        numerical_observation.record(
            site,
            numerical_observation.KIND_SOURCE_INF,
            nonfinite=source_summary["inf"],
            **common,
        )
    introduced = stored_summary["nonfinite"] - source_summary["nonfinite"]
    if introduced > 0:
        numerical_observation.record(
            site,
            numerical_observation.KIND_CAST_OVERFLOW,
            nonfinite=introduced,
            detail="finite before the storage cast",
            **common,
        )


def _note_sidecar_numeric(cache_path: Path, payload: dict) -> None:
    """Count non-finite float tensors in one cache payload, with sidecar identity.

    Identity (the sidecar path) is what the earlier "no message at all" gap left out:
    without it a corrupt sidecar cannot be traced back to an image. Nothing is
    rejected or rewritten here — the established write policy is unchanged.
    """
    problems: dict[str, str] = {}
    for name, value in payload.items():
        if not (isinstance(value, torch.Tensor) and value.is_floating_point()):
            continue
        summary = _nonfinite_summary(value)
        if not summary["nonfinite"]:
            continue
        problems[name] = f"{summary['nonfinite']}/{summary['elements']}"
        site = f"preprocess.text.sidecar.{name}"
        common = {
            "elements": summary["elements"],
            "before_dtype": value.dtype,
            "after_dtype": value.dtype,
            "shape": tuple(value.shape),
            "identity": str(cache_path),
        }
        if summary["nan"]:
            numerical_observation.record(
                site, numerical_observation.KIND_SOURCE_NAN, nonfinite=summary["nan"], **common
            )
        if summary["inf"]:
            numerical_observation.record(
                site, numerical_observation.KIND_SOURCE_INF, nonfinite=summary["inf"], **common
            )
    if problems:
        logger.debug("text cache sidecar %s carries non-finite tensors %s", cache_path, problems)


def _encode_batch(
    captions: list[str],
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    llm_adapter,
    device: torch.device,
    cache_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Tokenize, encode through Qwen3, optionally run the LLM adapter. CPU tensors out."""
    tokens_and_masks = tokenize_strategy.tokenize(captions)
    with torch.no_grad():
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
            encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens_and_masks
            )
        )

        crossattn_emb = None
        if llm_adapter is not None:
            crossattn_emb = llm_adapter(
                source_hidden_states=prompt_embeds,
                target_input_ids=t5_input_ids.to(device, dtype=torch.long),
                target_attention_mask=t5_attn_mask.to(device),
                source_attention_mask=attn_mask,
            )
            crossattn_emb[~t5_attn_mask.to(device).bool()] = 0
            crossattn_stored = crossattn_emb.to(dtype=cache_dtype)
            _report_cast_numeric(
                "crossattn_emb", crossattn_emb, crossattn_stored, cache_dtype
            )
            crossattn_emb = crossattn_stored.cpu()

    prompt_stored = prompt_embeds.to(dtype=cache_dtype)
    _report_cast_numeric("prompt_embeds", prompt_embeds, prompt_stored, cache_dtype)
    return (
        prompt_stored.cpu(),
        attn_mask.to(dtype=torch.int32).cpu(),
        t5_input_ids.to(dtype=torch.long).cpu(),
        t5_attn_mask.to(dtype=torch.int32).cpu(),
        crossattn_emb,
    )


def _te_cache_path(image_path: Path, cache_dir: Path | None, image_dir: Path) -> Path:
    if cache_dir is None:
        return image_path.with_name(image_path.stem + TE_CACHE_SUFFIX)
    return Path(
        resolve_cache_path(
            str(image_path),
            TE_CACHE_SUFFIX,
            cache_dir=str(cache_dir),
            image_dir=str(image_dir),
        )
    )


def _te_cache_reusable(
    cache_path: Path, identity: dict | None, require_identity: bool
) -> bool:
    """True when an existing TE sidecar was produced by the current encoder.

    Reads only the safetensors header, so this is cheap next to encoding. An
    unstamped (pre-identity) sidecar is reused with a one-off warning unless
    require_identity asks for a full rebuild; an unreadable sidecar is always
    re-encoded.
    """
    if identity is None:
        return True
    try:
        from safetensors import safe_open

        with safe_open(str(cache_path), framework="pt") as handle:
            metadata = handle.metadata()
    except Exception as e:  # corrupt/truncated sidecar -> re-encode
        logger.warning("re-encoding unreadable text cache %s: %s", cache_path, e)
        return False

    recorded = _identity.read_stamp(metadata)
    if _identity.evaluate(
        recorded, identity, require_identity=require_identity
    ) == "stale":
        return False
    if _identity.compare(recorded, identity) == "absent":
        _identity.warn_legacy("text encoder", cache_path)
    return True


def cache_text_embeddings(
    data_dir: Path,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    *,
    llm_adapter=None,
    device: torch.device,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    batch_size: int | str = "auto",
    caption_shuffle_variants: int = 0,
    caption_tag_dropout_rate: float = 0.0,
    prefer_json_caption: bool = False,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    diff_output_preservation_trigger: str | None = None,
    diff_output_preservation_class: str | None = None,
    min_pixels: int = 500_000,
    verbose: bool = True,
    progress: ProgressFn | None = None,
    overwrite: bool = False,
    identity_inputs: dict | None = None,
    require_cache_identity: bool = False,
) -> PreprocessStats:
    """Encode captions for every image under ``data_dir``.

    Strategies + encoder + (optional) ``llm_adapter`` are supplied loaded + on
    ``device``. Images below ``min_pixels`` are skipped (mirrors the resize
    filter). Missing or empty caption sidecars are encoded as empty captions so
    the cached set stays aligned with the image dataset. With
    ``caption_shuffle_variants > 0`` each cache holds N variants
    (v0 pristine, v1..v{N-1} shuffled + optionally tag-dropped). Returns counts;
    pass ``progress`` for a per-image bar.
    """
    batch_size = batch_size_arg(batch_size)
    tuner = AutoBatcher(device, label="anima/text") if batch_size == "auto" else None
    candidates = walk_images(data_dir, recursive=recursive, pattern=path_pattern)

    entries: list[tuple[Path, CaptionSource]] = []
    skipped_small = 0
    for p in candidates:
        if min_pixels > 0:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:
                logger.warning("could not read %s: %s", p.name, e)
                continue
            if w * h < min_pixels:
                skipped_small += 1
                continue

        # A missing or empty caption file is a valid explicit empty caption
        # (unconditional / style-LoRA training) — encode "" rather than
        # dropping the image, so the cached set matches the training dataset.
        caption = read_caption_source(
            p,
            prefer_json_caption=prefer_json_caption,
            caption_source_mode=caption_source_mode,
            caption_extension=caption_extension,
            captions_root=data_dir,
            warn=logger.warning,
        )
        entries.append((p, caption))

    if skipped_small and verbose:
        print(
            f"Skipping {skipped_small} images below {min_pixels:,} pixels "
            f"({min_pixels / 1e6:.2f}MP) -- same filter as resize_images.py."
        )

    stats = PreprocessStats(seen=len(entries))
    caption_dropout_rate = torch.tensor(0.0, dtype=torch.float32)
    n_variants = caption_shuffle_variants
    tag_dropout_rate = float(caption_tag_dropout_rate)
    cache_prior_crossattn = bool(
        llm_adapter is not None and str(diff_output_preservation_class or "").strip()
    )
    cache_dtype = getattr(text_encoder, "dtype", None) or torch.bfloat16

    from safetensors.torch import save_file

    anima_cache_spec = get_model_family_spec("anima")
    # Model identity: which text encoder / tokenizer / DiT-backing-LLM-adapter
    # produced these outputs. Swapping any of them must invalidate the sidecar.
    identity = None
    if identity_inputs:
        identity = _identity.text_cache_identity(
            family=anima_cache_spec.name,
            text_encoder=identity_inputs.get("text_encoder"),
            tokenizer=identity_inputs.get("tokenizer"),
            t5_tokenizer=identity_inputs.get("t5_tokenizer"),
            vocab_pack=identity_inputs.get("vocab_pack"),
            dit=identity_inputs.get("dit"),
            llm_adapter_outputs=llm_adapter is not None,
            diff_output_preservation_trigger=diff_output_preservation_trigger,
            diff_output_preservation_class=diff_output_preservation_class,
            compute_dtype=cache_dtype,
            caption_shuffle_variants=n_variants,
            tag_dropout_rate=tag_dropout_rate,
        )
    cache_metadata = _identity.stamp(
        anima_cache_spec.text_cache.metadata(anima_cache_spec.name), identity
    )

    if progress is not None:
        progress(0, total=len(entries))

    def encode(captions):
        def forward(group):
            return _encode_batch(group, tokenize_strategy, encoding_strategy,
                                 text_encoder, llm_adapter, device, cache_dtype)

        return tuner.encode_all(captions, forward) if tuner is not None else forward(captions)

    for batch in iter_batches(entries, lambda: tuner.batch_size if tuner else batch_size):

        # Skip already-cached entries.
        to_encode: list[tuple[Path, CaptionSource, Path]] = []
        for img_path, caption in batch:
            cache_path = _te_cache_path(img_path, cache_dir, data_dir)
            if (
                cache_path.exists()
                and not overwrite
                and _te_cache_reusable(cache_path, identity, require_cache_identity)
            ):
                stats.skipped += 1
                if progress is not None:
                    progress(1, detail=f"skip {img_path.name}")
            else:
                to_encode.append((img_path, caption, cache_path))

        if not to_encode:
            continue

        variants_by_entry = [
            _caption_variants_for_cache(caption, n_variants, tag_dropout_rate)
            for _, caption, _ in to_encode
        ]
        use_variant_cache = n_variants > 0 or any(len(variants) > 1 for variants in variants_by_entry)

        if not use_variant_cache:
            captions = [variants[0] for variants in variants_by_entry]
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                encode(captions)
            )
            prior_crossattn_emb = None
            if cache_prior_crossattn:
                prior_captions = [
                    build_diff_output_prior_caption(
                        caption,
                        trigger=diff_output_preservation_trigger,
                        class_prompt=diff_output_preservation_class,
                    )
                    for caption in captions
                ]
                *_, prior_crossattn_emb = encode(prior_captions)

            for i, (img_path, _, cache_path) in enumerate(to_encode):
                save_dict = {
                    "prompt_embeds": prompt_embeds[i],
                    "attn_mask": attn_mask[i],
                    "t5_input_ids": t5_input_ids[i],
                    "t5_attn_mask": t5_attn_mask[i],
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if crossattn_emb is not None:
                    save_dict["crossattn_emb"] = crossattn_emb[i]
                if prior_crossattn_emb is not None:
                    save_dict["prior_crossattn_emb"] = prior_crossattn_emb[i]
                _note_sidecar_numeric(cache_path, save_dict)
                save_file(save_dict, str(cache_path), metadata=cache_metadata)
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{img_path.name}")
        else:
            all_captions: list[str] = []
            for variants in variants_by_entry:
                all_captions.extend(variants)

            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                encode(all_captions)
            )
            prior_crossattn_emb = None
            if cache_prior_crossattn:
                prior_captions = [
                    build_diff_output_prior_caption(
                        caption,
                        trigger=diff_output_preservation_trigger,
                        class_prompt=diff_output_preservation_class,
                    )
                    for caption in all_captions
                ]
                *_, prior_crossattn_emb = encode(prior_captions)

            offset = 0
            for img_path, caption, cache_path in to_encode:
                variants = variants_by_entry.pop(0)
                variant_count = len(variants)
                save_dict = {
                    "num_variants": torch.tensor(variant_count, dtype=torch.int64),
                    # Marker: v0 is the pristine original caption (no shuffle,
                    # no tag dropout). Loaders use this to switch on weighted
                    # 20%/80% sampling between v0 and v1..v{N-1}.
                    "v0_intact": torch.tensor(1, dtype=torch.int8),
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if caption.captions is not None:
                    save_dict["caption_multi_source"] = torch.tensor(1, dtype=torch.int8)
                for vi in range(variant_count):
                    flat_idx = offset + vi
                    save_dict[f"prompt_embeds_v{vi}"] = prompt_embeds[flat_idx]
                    save_dict[f"attn_mask_v{vi}"] = attn_mask[flat_idx]
                    save_dict[f"t5_input_ids_v{vi}"] = t5_input_ids[flat_idx]
                    save_dict[f"t5_attn_mask_v{vi}"] = t5_attn_mask[flat_idx]
                    if crossattn_emb is not None:
                        save_dict[f"crossattn_emb_v{vi}"] = crossattn_emb[flat_idx]
                    if prior_crossattn_emb is not None:
                        save_dict[f"prior_crossattn_emb_v{vi}"] = prior_crossattn_emb[flat_idx]
                _note_sidecar_numeric(cache_path, save_dict)
                save_file(save_dict, str(cache_path), metadata=cache_metadata)
                offset += variant_count
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{img_path.name} ({variant_count}v)")

    return stats


def _emit_pooled(te_path: Path, pooled_path: Path) -> bool:
    from safetensors.torch import load_file, save_file

    sd = load_file(str(te_path))
    out: dict = {}

    if "num_variants" in sd:
        n = int(sd["num_variants"])
        out["num_variants"] = sd["num_variants"]
        for vi in range(n):
            key = f"crossattn_emb_v{vi}"
            if key in sd:
                out[f"pooled_v{vi}"] = sd[key].amax(dim=0).contiguous()
        if not any(k.startswith("pooled_v") for k in out):
            return False
    elif "crossattn_emb_v0" in sd:
        out["pooled_v0"] = sd["crossattn_emb_v0"].amax(dim=0).contiguous()
    elif "crossattn_emb" in sd:
        out["pooled"] = sd["crossattn_emb"].amax(dim=0).contiguous()
    else:
        return False

    save_file(out, str(pooled_path))
    return True


def cache_pooled_text(
    cache_dir: Path,
    *,
    overwrite: bool = False,
    progress: ProgressFn | None = None,
) -> PreprocessStats:
    """Write ``{stem}_anima_pooled.safetensors`` next to each TE cache.

    ``pooled_v{i} = crossattn_emb_v{i}.amax(dim=0)`` for every variant present.
    Walks ``cache_dir`` recursively (nested caches mirror the source tree). Pure
    tensor reduction — no GPU / text encoder. ``failed`` counts TE files that
    carry no ``crossattn`` key. Returns counts; pass ``progress`` for a bar.
    """
    te_files = sorted(cache_dir.rglob(f"*{TE_CACHE_SUFFIX}"))
    stats = PreprocessStats(seen=len(te_files))

    if progress is not None:
        progress(0, total=len(te_files))

    for te_path in te_files:
        stem = te_path.name.removesuffix(TE_CACHE_SUFFIX)
        pooled_path = te_path.parent / (stem + POOLED_CACHE_SUFFIX)
        if pooled_path.exists() and not overwrite:
            stats.skipped += 1
        elif _emit_pooled(te_path, pooled_path):
            stats.written += 1
        else:
            stats.failed += 1
        if progress is not None:
            progress(1, detail=stem)

    return stats
