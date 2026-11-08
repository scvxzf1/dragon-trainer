import math
import os
import random
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np

# Bucket resolutions as (W, H), grouped into two token-count families: 4032
# (= 63*64) and 4200 (= 60*70). Both are highly composite, so each factors into
# many near-square→elongated patch grids — and crucially every bucket *exactly*
# fills its token count, so there is zero intra-bucket padding by construction.
#
# This table is designed for native shapes (the only mode): it collapses to
# just TWO distinct token counts → two compiled block graphs (via
# compile_blocks' flatten), with no padding and therefore no flash pad leak.
# The rope per-axis cap is 256 patches (max_img/patch_spatial); the largest dim
# here is 2016px → 126.
#
# Two families instead of one because a single token count's divisors near √N
# are sparse (4032 alone jumps aspect 1.29→1.75); interleaving 4032 and 4200
# densely covers aspect space at the cost of one extra graph. Landscape mirrors
# (swap W, H) are included explicitly. Token count = (W//16)*(H//16).
#
# NOTE: DCW_ASPECT_BUCKETS below now draws its top-5 from this table (every
# entry is a real training bucket), so `make dcw` recalibration produces rows
# for every aspect_id. Do not reorder the DCW table (shipped fusion-head
# checkpoints key off it).
CONSTANT_TOKEN_BUCKETS = [
    # ---- 4032-token family (63*64) ----
    (1008, 1024),  # 63 x 64, ar 0.98 (nearest to square)
    (1024, 1008),  #          ar 1.02
    (896, 1152),  # 56 x 72, ar 0.78
    (1152, 896),  #          ar 1.29
    (768, 1344),  # 48 x 84, ar 0.57
    (1344, 768),  #          ar 1.75
    (672, 1536),  # 42 x 96, ar 0.44
    (1536, 672),  #          ar 2.29
    (576, 1792),  # 36 x 112, ar 0.32
    (1792, 576),  #           ar 3.11
    (512, 2016),  # 32 x 126, ar 0.25
    (2016, 512),  #           ar 3.94
    # ---- 4200-token family (60*70) ----
    (960, 1120),  # 60 x 70, ar 0.86
    (1120, 960),  #          ar 1.17
    (896, 1200),  # 56 x 75, ar 0.75
    (1200, 896),  #          ar 1.34
    (800, 1344),  # 50 x 84, ar 0.60
    (1344, 800),  #          ar 1.68
    (672, 1600),  # 42 x 100, ar 0.42
    (1600, 672),  #           ar 2.38
    (640, 1680),  # 40 x 105, ar 0.38
    (1680, 640),  #           ar 2.62
    (560, 1920),  # 35 x 120, ar 0.29
    (1920, 560),  #           ar 3.43
]

# DCW v4 calibration aspect-bucket set.
#
# Top 5 (H, W) resolutions by frequency in post_image_dataset/lora/ (recounted
# 2026-05-23; every entry is a CONSTANT_TOKEN_BUCKETS training bucket). List
# order *is* the canonical aspect_id index — DCW v4's per-aspect statistics
# (fusion_head.safetensors per-bucket μ_g, σ²_prior, λ_scalar) key off this
# order, so a reorder invalidates every shipped fusion-head checkpoint.
#
# Read by both the calibration data-gen path (scripts/tasks/dcw.py drives
# `make dcw` over these buckets) and the fusion-head trainer
# (scripts/dcw/fusion_data.py uses the dict for the (H, W) → aspect_id
# lookup that decides which run rows feed the trainer). Inference itself
# is bucket-agnostic post-cleanup — see project_dcw_bucket_prior_cosmetic.
DCW_ASPECT_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (1200, 896),  # 0 — 896x1200 portrait (most common, 4200-tok)
    (1344, 800),  # 1 — 800x1344 tall portrait (4200-tok)
    (896, 1200),  # 2 — 1200x896 landscape (4200-tok)
    (1344, 768),  # 3 — 768x1344 tall portrait (4032-tok)
    (1152, 896),  # 4 — 896x1152 portrait (4032-tok)
)
DCW_ASPECT_NAMES: Tuple[str, ...] = tuple(f"{h}x{w}" for h, w in DCW_ASPECT_BUCKETS)
DCW_ASPECT_TABLE: dict = {hw: i for i, hw in enumerate(DCW_ASPECT_BUCKETS)}
N_DCW_ASPECTS: int = len(DCW_ASPECT_BUCKETS)


def snap_sample_size(width: int, height: int) -> Tuple[int, int]:
    """Snap a requested sample (W, H) to the DiT's 16px pixel grid.

    The single definition of the snap ``_sample_image_inference`` applies before
    sampling — shared with the compile token budget so both sides agree on the
    seq len a sample prompt will actually run at.
    """
    return max(64, width - width % 16), max(64, height - height % 16)


def token_counts_for_sample_prompts(prompts) -> set:
    """Distinct DiT token counts the training sample prompts will request.

    ``prompts`` are ``train_util.load_prompts`` dicts; width/height default to
    512, matching ``_sample_image_inference``. Folded into the torch.compile
    token budget so a sample resolution outside the training buckets (e.g.
    ``--w 1024 --h 1536`` over 1024-tier data → 6144 tokens vs a (4032, 4200)
    range) widens the compiled range instead of crashing mid-training with a
    dynamic-seq ConstraintViolationError.
    """
    counts: set = set()
    for prompt_dict in prompts:
        if not isinstance(prompt_dict, dict):
            continue
        try:
            w, h = snap_sample_size(
                int(prompt_dict.get("width", 512) or 512),
                int(prompt_dict.get("height", 512) or 512),
            )
        except (TypeError, ValueError):
            continue
        counts.add((w // 16) * (h // 16))
    return counts


def cluster_token_bands(counts, rel_gap: float = 0.10) -> "list[tuple[int, int]]":
    """Cluster token counts into tight, data-driven dynamic-seq bands.

    Counts are sorted and split when the relative gap from the previous count
    exceeds ``rel_gap``.  This deliberately operates on the counts actually
    present in a run rather than on the canonical bucket table: sample prompts
    and auxiliary/demoted resolutions can therefore form their own singleton
    band without widening an unrelated tier across a dead zone.
    """
    ordered = sorted({int(count) for count in counts})
    if not ordered:
        return []

    bands: list[tuple[int, int]] = []
    lo = previous = ordered[0]
    for count in ordered[1:]:
        if (count - previous) > rel_gap * previous:
            bands.append((lo, previous))
            lo = count
        previous = count
    bands.append((lo, previous))
    return bands


def band_for_seq(bands, seq: int) -> "tuple[int, int] | None":
    """Return the sorted non-overlapping band containing ``seq``.

    ``None`` means that the sequence is in an inter-band gap or outside the
    compiled budget.  ``cluster_token_bands`` produces the required ordering.
    """
    import bisect

    if not bands:
        return None
    starts = [int(band[0]) for band in bands]
    index = bisect.bisect_right(starts, int(seq)) - 1
    if index < 0:
        return None
    band = bands[index]
    if int(seq) > int(band[1]):
        return None
    return (int(band[0]), int(band[1]))


def widen_bands(bands, extra: int) -> "list[tuple[int, int]]":
    """Increase each band's upper bound for constant register-token tails.

    Register tokens are inserted after the pre-insert blocks, so only the
    upper bound grows.  Refuse a widening that would touch or overlap the next
    band; silently merging bands would defeat the purpose of per-band guards.
    """
    normalized = [(int(lo), int(hi)) for lo, hi in bands]
    if extra <= 0:
        return normalized
    for (_, hi), (next_lo, _) in zip(normalized, normalized[1:]):
        if hi + int(extra) >= next_lo:
            raise ValueError(
                f"extra_seq_tokens={extra} >= inter-band gap "
                f"({next_lo - hi} between hi={hi} and next lo={next_lo}); "
                "bands would merge — widen the clustering gap or drop "
                "--compile_seq_bands for this run"
            )
    return [(lo, hi + int(extra)) for lo, hi in normalized]


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    """Generate bucket resolutions for multi-aspect-ratio training.
    Moved from model_util.py to avoid dependency."""
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


def make_constant_token_bucket_resolutions(
    max_reso,
    min_size=256,
    max_size=2048,
    divisible=16,
):
    if max_reso is None or max_reso == (1024, 1024):
        return list(CONSTANT_TOKEN_BUCKETS)

    max_width, max_height = max_reso
    scale = math.sqrt((max_width * max_height) / (1024 * 1024))
    resos = set()
    for width, height in CONSTANT_TOKEN_BUCKETS:
        scaled_width = max(divisible, round(width * scale / divisible) * divisible)
        scaled_height = max(divisible, round(height * scale / divisible) * divisible)
        if (
            min_size <= scaled_width <= max_size
            and min_size <= scaled_height <= max_size
        ):
            resos.add((scaled_width, scaled_height))

    if not resos:
        fallback = max(divisible, round(min(max_reso) / divisible) * divisible)
        resos.add((fallback, fallback))

    return sorted(resos)


class BucketManager:
    def __init__(
        self, max_reso=None, min_size=None, max_size=None, reso_steps=None
    ) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], (
                    "the max_size should be larger than the width of max_reso"
                )
                assert max_size >= max_reso[1], (
                    "the max_size should be larger than the height of max_reso"
                )
            if min_size is not None:
                assert max_size >= min_size, (
                    "the max_size should be larger than the min_size"
                )

        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        if pinned_order_enabled():
            # Measurement gate (contract §13 V-2.2 / F-T31-2): keep the bucket in
            # its insertion order so the sample at global position N is the same
            # image in a one-worker and a two-worker run.
            return
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self, constant_token_buckets: bool = False):
        if constant_token_buckets:
            resos = make_constant_token_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size
            )
        else:
            resos = make_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size, self.reso_steps
            )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        reso = (image_width, image_height)
        if reso in self.predefined_resos_set:
            pass
        else:
            ar_errors = self.predefined_aspect_ratios - aspect_ratio
            predefined_bucket_id = np.abs(ar_errors).argmin()
            reso = self.predefined_resos[predefined_bucket_id]

        ar_reso = reso[0] / reso[1]
        if aspect_ratio > ar_reso:
            scale = reso[1] / image_height
        else:
            scale = reso[0] / image_width

        resized_size = (
            int(image_width * scale + 0.5),
            int(image_height * scale + 0.5),
        )

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


#: Opt-in *measurement* gate: pin the data stream so a single-worker and a
#: multi-worker run consume the same image at the same global sample position.
#: Default off — with the variable unset every function below is a no-op and the
#: training path is byte-for-byte what it was.
PINNED_ORDER_ENV = "ANIMA_MULTIGPU_PINNED_ORDER"

_TRUE = {"1", "true", "yes", "on"}


def pinned_order_enabled(env: Optional[dict] = None) -> bool:
    """Whether the pinned-order measurement gate is open."""
    values = dict(os.environ if env is None else env)
    return str(values.get(PINNED_ORDER_ENV, "")).strip().lower() in _TRUE


def pinned_buckets_indices(dataset: Any) -> list:
    """The dataset's batch descriptors in natural ``(bucket, batch)`` order.

    ``DatasetBuckets.create_buckets`` shuffles ``buckets_indices`` with
    ``random.seed(seed + epoch)``, and the list length depends on the batch size
    (one descriptor per batch: 12 at batch 2, 24 at batch 1 for 24 images). Two
    permutations of different lengths cannot agree, so the descriptor order has to
    be pinned as well — otherwise "disable the DataLoader shuffle" still yields
    different images per slot (t31 F-T31-2 measured exactly that).
    """
    indices = list(getattr(dataset, "buckets_indices", []) or [])
    return sorted(indices, key=lambda item: (item.bucket_index, item.batch_index))


def apply_pinned_order(dataset_group: Any) -> int:
    """Sort every member's batch descriptors into natural order; 0 when the gate is off.

    Returns the number of descriptors re-ordered so a run can record whether the
    measurement gate actually took effect.
    """
    if not pinned_order_enabled():
        return 0
    changed = 0
    for dataset in getattr(dataset_group, "datasets", []) or []:
        indices = getattr(dataset, "buckets_indices", None)
        if not indices:
            continue
        ordered = pinned_buckets_indices(dataset)
        if ordered != list(indices):
            dataset.buckets_indices = ordered
            changed += len(ordered)
    return changed


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int
