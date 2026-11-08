import glob
import json
import logging
import os
import random
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

from library.anima.text_strategies import LatentsCachingStrategy
from library.datasets import base as _base
from library.datasets.base import BaseDataset
from library.datasets.image_utils import (
    _assert_unique_stems,
    glob_images,
)
from library.datasets.qwen_image_edit import inspect_edit_pairs, require_complete_edit_pairs
from library.datasets.subsets import (
    DreamBoothSubset,
    ImageInfo,
    filter_paths_by_glob,
    split_train_val,
)

logger = logging.getLogger(__name__)


def _register_reg_infos(
    entries: Sequence[Tuple["DreamBoothDataset", ImageInfo, DreamBoothSubset]],
    target_repeats: int,
) -> int:
    """Register regularization samples round-robin up to the repeat target."""
    if target_repeats <= 0 or not entries:
        return 0

    registered = 0
    first_loop = True
    while registered < target_repeats:
        for dataset, info, subset in entries:
            if first_loop:
                dataset.register_image(info, subset)
                registered += info.num_repeats
            else:
                info.num_repeats += 1
                registered += 1
            if registered >= target_repeats:
                break
        first_loop = False
    return registered


def _latent_cache_paths(
    subset, cache_suffix: str, recursive: bool
) -> Tuple[List[str], List[str], Optional[str]]:
    """Latent sidecar paths for one subset, split by location.

    Returns ``(next_to_image, under_cache_dir, cache_dir)``.

    The subset may redirect its caches into ``cache_dir`` (the documented layout:
    ``cache_dir`` mirrors the source tree, ``charA/img1.png`` →
    ``cache_dir/charA/img1_1024x1024_anima.npz``). Scanning only ``image_dir``
    therefore found 0/N sidecars for such a dataset and every size had to be read
    by opening the image again (t13 F-T13-4). The two locations are returned
    separately because they are matched differently: next-to-image sidecars share
    a path prefix with their image, cache_dir sidecars share a *stem*.
    """
    image_dir = subset.image_dir
    if recursive:
        beside = glob.glob(
            os.path.join(image_dir, "**", "*" + cache_suffix), recursive=True
        )
    else:
        beside = glob.glob(os.path.join(image_dir, "*" + cache_suffix))

    cache_dir = getattr(subset, "cache_dir", None)
    if not cache_dir:
        return beside, [], None
    cache_dir = os.fspath(cache_dir)
    if not os.path.isdir(cache_dir):
        logger.warning(
            "subset cache_dir=%r does not exist; only %d cache file(s) next to the "
            "images can be used for image sizes",
            cache_dir,
            len(beside),
        )
        return beside, [], cache_dir
    # Always recursive: a cache_dir mirrors the source tree, so the sidecars of a
    # flat image_dir can still sit in subdirectories.
    pooled = glob.glob(
        os.path.join(cache_dir, "**", "*" + cache_suffix), recursive=True
    )
    return beside, pooled, cache_dir


def _cache_stem(npz_path: str, cache_suffix: str) -> str:
    """Image stem of a latent sidecar (``img1_1024x1024_anima.npz`` → ``img1``)."""
    base = os.path.basename(npz_path)
    if base.endswith(cache_suffix):
        base = base[: -len(cache_suffix)]
    # ``_1024x1024`` / ``_768x1024`` resolution infix.
    return base.rsplit("_", 1)[0]


def _index_cache_dir_sidecars(
    pooled: List[str], cache_dir: str, cache_suffix: str
) -> Tuple[dict, dict]:
    """``{(mirrored subdir, stem): path}`` and ``{stem: [paths]}`` for cache_dir hits."""
    by_dir_and_stem: dict = {}
    by_stem: dict = {}
    for npz_path in pooled:
        stem = _cache_stem(npz_path, cache_suffix)
        relative = os.path.relpath(os.path.dirname(npz_path), cache_dir)
        if relative == os.curdir:
            relative = ""
        by_dir_and_stem.setdefault((relative, stem), npz_path)
        by_stem.setdefault(stem, []).append(npz_path)
    return by_dir_and_stem, by_stem


def _match_cache_dir_sidecar(
    img_path: str,
    image_dir: str,
    recursive: bool,
    by_dir_and_stem: dict,
    by_stem: dict,
) -> Optional[str]:
    """The cache_dir sidecar for one image, preferring the mirrored subdirectory."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    relative = ""
    if recursive:
        relative = os.path.relpath(os.path.dirname(img_path), image_dir)
        if relative == os.curdir:
            relative = ""
    exact = by_dir_and_stem.get((relative, stem))
    if exact is not None:
        return exact
    options = by_stem.get(stem) or []
    if len(options) == 1:
        # A flat cache_dir (written without the source root) is still valid.
        return options[0]
    if len(options) > 1:
        logger.warning(
            "ambiguous latent cache for %r under cache_dir (%d candidates: %s); "
            "this image's size will be read from the image",
            img_path,
            len(options),
            ", ".join(os.path.basename(option) for option in options[:3]),
        )
    return None


def set_image_sizes_from_cache(
    subset,
    img_paths: Sequence[str],
    *,
    recursive: bool,
    strategy,
) -> List[Optional[Tuple[int, int]]]:
    """Image sizes read from the latent cache file names, without opening images.

    Two locations are consulted: sidecars next to the images (matched by path
    prefix, the long-standing fast path) and sidecars under ``subset.cache_dir``
    (matched by stem, because that layout mirrors the source tree instead of
    sharing a path prefix). Sizes that neither location knows are left ``None``
    and are later read by opening the image in ``make_buckets``.
    """
    logger.info("get image size from name of cache files")
    sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

    npz_paths, pooled_paths, cache_dir = _latent_cache_paths(
        subset, strategy.cache_suffix, recursive
    )
    npz_paths.sort(key=lambda item: item.rsplit("_", maxsplit=2)[0])
    npz_path_index = 0

    beside_count = 0
    for i, img_path in enumerate(tqdm(img_paths)):
        stem_len = len(os.path.splitext(img_path)[0])
        found = False
        while npz_path_index < len(npz_paths):
            if npz_paths[npz_path_index][:stem_len] > img_path[:stem_len]:
                break
            if npz_paths[npz_path_index][:stem_len] == img_path[:stem_len]:
                found = True
                break
            npz_path_index += 1

        if found:
            w, h = strategy.get_image_size_from_disk_cache_path(
                img_path, npz_paths[npz_path_index]
            )
        else:
            w, h = None, None

        if w is not None and h is not None:
            sizes[i] = (w, h)
            beside_count += 1
    logger.info(f"set image size from cache files: {beside_count}/{len(img_paths)}")

    # A subset that redirects its caches into ``cache_dir`` has no sidecar next to
    # the images at all, so the pass above cannot match anything (t13 F-T13-4).
    if pooled_paths:
        by_dir_and_stem, by_stem = _index_cache_dir_sidecars(
            pooled_paths, cache_dir, strategy.cache_suffix
        )
        cache_dir_count = 0
        for i, img_path in enumerate(img_paths):
            if sizes[i] is not None:
                continue
            npz_path = _match_cache_dir_sidecar(
                img_path, subset.image_dir, recursive, by_dir_and_stem, by_stem
            )
            if npz_path is None:
                continue
            w, h = strategy.get_image_size_from_disk_cache_path(img_path, npz_path)
            if w is not None and h is not None:
                sizes[i] = (w, h)
                cache_dir_count += 1
        logger.info(
            f"set image size from cache_dir={cache_dir}: "
            f"{cache_dir_count}/{len(img_paths)}"
        )

    unknown = sum(1 for size in sizes if size is None)
    if unknown:
        # Slow-but-correct path: ``make_buckets`` opens the image for a size that is
        # still unknown. Logged separately from the cache hits above so a
        # mislocated cache directory is visible instead of just slow.
        logger.info(
            f"image size unknown for {unknown}/{len(img_paths)} image(s); "
            "those are read by opening the image"
        )
    return sizes


def read_caption(img_path, caption_extension, enable_wildcard):
    """Read the caption sidecar for ``img_path``.

    Returns ``None`` when no sidecar file exists. An empty (or
    whitespace-only) caption file is a valid *explicit empty caption*
    (unconditional / style-LoRA training) and resolves to ``""`` rather than
    raising — callers treat ``""`` as a real caption, not a missing one.
    """
    base_name = os.path.splitext(img_path)[0]
    base_name_face_det = base_name
    tokens = base_name.split("_")
    if len(tokens) >= 5:
        base_name_face_det = "_".join(tokens[:-4])
    cap_paths = [
        base_name + caption_extension,
        base_name_face_det + caption_extension,
    ]

    caption = None
    for cap_path in cap_paths:
        if os.path.isfile(cap_path):
            with open(cap_path, "rt", encoding="utf-8") as f:
                try:
                    lines = f.readlines()
                except UnicodeDecodeError as e:
                    logger.error("illegal char in file (not UTF-8)")
                    raise e
                if enable_wildcard:
                    caption = "\n".join(
                        [line.strip() for line in lines if line.strip() != ""]
                    )
                else:
                    caption = lines[0].strip() if lines else ""
            break
    return caption


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        network_multiplier: float,
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
        validation_split_num: int = 0,
        resolution: int = 1024,
        enable_bucket: bool = True,
        min_bucket_reso: int = 256,
        max_bucket_reso: int = 2048,
        bucket_reso_steps: int = 64,
        bucket_no_upscale: bool = False,
    ) -> None:
        super().__init__(
            network_multiplier,
            debug_dataset,
            resize_interpolation,
            resolution=resolution,
            enable_bucket=enable_bucket,
            min_bucket_reso=min_bucket_reso,
            max_bucket_reso=max_bucket_reso,
            bucket_reso_steps=bucket_reso_steps,
            bucket_no_upscale=bucket_no_upscale,
        )

        self.batch_size = batch_size
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split
        self.validation_split_num = int(validation_split_num or 0)

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info("using cached image info for this subset")
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        "image info file not found. You can ignore this warning if this is the first time to use this subset"
                        + ""
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                pattern = getattr(subset, "path_pattern", "*") or "*"
                if pattern != "*":
                    meta_paths = list(metas.keys())
                    keep = filter_paths_by_glob(
                        meta_paths, subset.image_dir, pattern
                    )
                    metas = {
                        p: metas[p] for p, k in zip(meta_paths, keep) if k
                    }
                    logger.info(
                        f"path_pattern={pattern!r} kept {len(metas)}/"
                        f"{len(meta_paths)} cached entries from "
                        f"{subset.image_dir}"
                    )
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [
                    meta["resolution"] for meta in metas.values()
                ]
            else:
                recursive = getattr(subset, "recursive", False)
                img_paths = glob_images(subset.image_dir, "*", recursive=recursive)
                if recursive:
                    _assert_unique_stems(img_paths, source_label=subset.image_dir)
                pattern = getattr(subset, "path_pattern", "*") or "*"
                if pattern != "*":
                    keep = filter_paths_by_glob(
                        img_paths, subset.image_dir, pattern
                    )
                    pre_n = len(img_paths)
                    img_paths = [p for p, k in zip(img_paths, keep) if k]
                    logger.info(
                        f"path_pattern={pattern!r} kept {len(img_paths)}/"
                        f"{pre_n} images from {subset.image_dir}"
                    )
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    sizes = set_image_sizes_from_cache(
                        subset, img_paths, recursive=recursive, strategy=strategy
                    )

            if self.validation_split > 0.0 or self.validation_split_num > 0:
                if subset.is_reg is True:
                    if self.is_training_dataset is False:
                        img_paths = []
                        sizes = []
                else:
                    img_paths, sizes = split_train_val(
                        img_paths,
                        sizes,
                        self.is_training_dataset,
                        self.validation_split,
                        self.validation_seed,
                        validation_split_num=self.validation_split_num,
                    )

            # sample_ratio shrinks only the training pool. The validation pool
            # is pinned by validation_split_num / validation_split — applying
            # sample_ratio there would silently reduce the user-requested val
            # count (e.g. PRESET=half + validation_split_num=16 → 8 items),
            # which is surprising for CMMD where val size controls estimator
            # variance.
            if (
                subset.sample_ratio < 1.0
                and len(img_paths) > 0
                and self.is_training_dataset
            ):
                sample_count = max(1, int(len(img_paths) * subset.sample_ratio))
                dataset = list(zip(img_paths, sizes))
                prevstate = random.getstate()
                random.seed(self.validation_seed)
                random.shuffle(dataset)
                random.setstate(prevstate)
                img_paths, sizes = zip(*dataset[:sample_count])
                img_paths = list(img_paths)
                sizes = list(sizes)
                logger.info(
                    f"sampled {sample_count} images (sample_ratio={subset.sample_ratio}) from {subset.image_dir}"
                )

            logger.info(
                f"found directory {subset.image_dir} contains {len(img_paths)} image files"
            )

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [
                    img_path
                    for img_path, caption in zip(img_paths, captions)
                    if caption is None or caption == ""
                ]
            else:
                # Subset may redirect TE caches to a separate `cache_dir` (e.g.
                # captions live in image_dataset/ but resized images live in
                # post_image_dataset/resized/ with caches in
                # post_image_dataset/lora/). When a TE cache exists for the
                # image's relative path, missing .txt sidecars are expected,
                # not an error — training reads the cached prompt embeddings.
                # Key by (rel_subdir, stem) to support nested cache layouts
                # where two images can share a stem in different subdirs.
                cache_dir = getattr(subset, "text_cache_dir", None) or getattr(
                    subset, "cache_dir", None
                )
                te_suffix = subset.text_cache_suffix
                if subset.model_family == "moss_image":
                    from library.anima.text_strategies import TextEncoderOutputsCachingStrategy
                    from library.io.cache_names import text_cache_suffix

                    te_suffix = text_cache_suffix(
                        subset.model_family, TextEncoderOutputsCachingStrategy.get_strategy(),
                        moss_te_quantization=getattr(subset, "moss_te_quantization", "nf4"),
                    )
                te_cached_keys: set[tuple[str, str]] = set()
                if cache_dir and os.path.isdir(cache_dir):
                    cache_root = os.fspath(cache_dir)
                    for dirpath, _dirnames, filenames in os.walk(cache_root):
                        try:
                            rel_dir = os.path.relpath(dirpath, cache_root)
                        except ValueError:
                            rel_dir = ""
                        if rel_dir == ".":
                            rel_dir = ""
                        rel_dir = rel_dir.replace(os.sep, "/")
                        for name in filenames:
                            if name.endswith(te_suffix):
                                te_cached_keys.add(
                                    (rel_dir, name.removesuffix(te_suffix))
                                )

                image_root = getattr(subset, "image_dir", None)
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(
                        img_path, subset.caption_extension, subset.enable_wildcard
                    )
                    if image_root:
                        try:
                            img_rel_dir = os.path.relpath(
                                os.path.dirname(img_path), image_root
                            )
                        except ValueError:
                            img_rel_dir = ""
                        if img_rel_dir == ".":
                            img_rel_dir = ""
                        img_rel_dir = img_rel_dir.replace(os.sep, "/")
                    else:
                        img_rel_dir = ""
                    img_stem = os.path.splitext(os.path.basename(img_path))[0]
                    has_te_cache = (img_rel_dir, img_stem) in te_cached_keys
                    if cap_for_img is None and subset.class_tokens is None:
                        if not has_te_cache:
                            logger.warning(
                                f"neither caption file nor class tokens are found. use empty caption for {img_path}"
                            )
                            missing_captions.append(img_path)
                        captions.append("")
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            if not has_te_cache:
                                missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = (
                    number_of_missing_captions - number_of_missing_captions_to_show
                )

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used."
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(
                            missing_caption
                            + f"... and {remaining_missing_captions} more"
                        )
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info("cache image info for")
                sizes = [
                    self.get_image_size(img_path)
                    for img_path in tqdm(img_paths, desc="get image size")
                ]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info("cache image info done for")

            if _base._ARTIST_FILTER is not None:
                pre = len(img_paths)
                kept = [
                    (p, c, s)
                    for p, c, s in zip(img_paths, captions, sizes)
                    if _base._caption_has_artist(c, _base._ARTIST_FILTER)
                ]
                if kept:
                    img_paths, captions, sizes = (list(t) for t in zip(*kept))
                else:
                    img_paths, captions, sizes = [], [], []
                logger.info(
                    f"artist_filter='{_base._ARTIST_FILTER}' → kept {len(img_paths)}/{pre} "
                    f"images from {subset.image_dir}"
                )

            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            num_repeats = subset.num_repeats if self.is_training_dataset else 1
            if num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found"
                )
                continue

            reference_paths = {}
            if subset.reference_image_dir:
                if subset.is_reg:
                    raise ValueError(
                        "Qwen Image 2.1 Edit does not support regularization subsets; "
                        f"remove reference_image_dir from {subset.image_dir}"
                    )
                if not os.path.isdir(subset.reference_image_dir):
                    raise ValueError(
                        f"Qwen edit reference image directory does not exist: "
                        f"{subset.reference_image_dir}"
                    )
                pair_report = inspect_edit_pairs(
                    img_paths,
                    target_dir=subset.image_dir,
                    reference_dir=subset.reference_image_dir,
                    recursive=subset.recursive,
                )
                require_complete_edit_pairs(pair_report)
                reference_paths = pair_report.pairs
                if pair_report.unused_references:
                    logger.warning(
                        "Qwen edit reference directory has %d unused image(s): %s",
                        len(pair_report.unused_references),
                        ", ".join(pair_report.unused_references[:5]),
                    )

            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(
                    img_path,
                    num_repeats,
                    caption,
                    subset.is_reg,
                    img_path,
                    subset.caption_dropout_rate,
                )
                info.reference_image_path = reference_paths.get(img_path)
                if info.reference_image_path and not str(caption or "").strip():
                    raise ValueError(
                        "Qwen Image 2.1 Edit requires a non-empty target caption/instruction: "
                        f"{img_path}"
                    )
                info.resize_interpolation = (
                    subset.resize_interpolation
                    if subset.resize_interpolation is not None
                    else self.resize_interpolation
                )
                if getattr(subset, "mask_dir", None):
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    # Prefer the nested path that mirrors subset.image_dir →
                    # mask_dir; fall back to the flat layout so legacy
                    # masks/merged/ etc. caches keep working.
                    candidates: list[str] = []
                    image_dir = getattr(subset, "image_dir", None)
                    if image_dir:
                        try:
                            rel = os.path.relpath(os.path.dirname(img_path), image_dir)
                        except ValueError:
                            rel = ""
                        if rel and rel != "." and not rel.startswith(".."):
                            candidates.append(
                                os.path.join(subset.mask_dir, rel, f"{stem}_mask.png")
                            )
                    candidates.append(os.path.join(subset.mask_dir, f"{stem}_mask.png"))
                    for mask_path in candidates:
                        if os.path.exists(mask_path):
                            info.mask_path = mask_path
                            break
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        images_split_name = "train" if self.is_training_dataset else "validation"
        logger.info(f"{num_train_images} {images_split_name} images with repeats.")

        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images with repeats.")
        if num_train_images > 0 and num_train_images < num_reg_images:
            logger.warning("some of reg images are not used")

        if num_reg_images == 0:
            logger.warning("no regularization images")
            self._pending_reg_infos = []
        elif num_train_images > 0:
            _register_reg_infos(
                [(self, info, subset) for info, subset in reg_infos],
                num_train_images,
            )
            self._pending_reg_infos = []
        else:
            # WebUI rows are separate dataset members. Defer reg-only members
            # until every member is known so they can balance the whole group.
            self._pending_reg_infos = reg_infos

        self.num_reg_images = num_reg_images


def balance_reg_datasets(datasets: Sequence[DreamBoothDataset]) -> int:
    """Balance deferred reg-only members against unpaired training repeats."""
    total_train_repeats = sum(dataset.num_train_images for dataset in datasets)
    registered_reg_repeats = sum(
        info.num_repeats
        for dataset in datasets
        for info in dataset.image_data.values()
        if info.is_reg
    )
    pending = [
        (dataset, info, subset)
        for dataset in datasets
        for info, subset in getattr(dataset, "_pending_reg_infos", [])
    ]
    pending_reg_repeats = sum(info.num_repeats for _, info, _ in pending)
    target_repeats = max(0, total_train_repeats - registered_reg_repeats)
    registered = _register_reg_infos(pending, target_repeats)

    for dataset in datasets:
        dataset._pending_reg_infos = []

    if pending and target_repeats < pending_reg_repeats:
        logger.warning("some cross-dataset regularization images are not used")
    elif registered:
        logger.info(
            "registered %s cross-dataset regularization image repeats",
            registered,
        )
    return registered
