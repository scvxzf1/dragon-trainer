import logging
from typing import Dict, List, Optional

import torch

from library.strategies import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TokenizeStrategy,
)
from library.datasets.buckets import BucketManager
from library.datasets import runtime_flags
from library.datasets.image_utils import (
    validate_interpolation_fn,
    IMAGE_TRANSFORMS,
)
from library.datasets.dataset_buckets import DatasetBucketsMixin
from library.datasets.dataset_caption import DatasetCaptionMixin
from library.datasets.dataset_getitem import DatasetGetItemMixin
from library.datasets.dataset_cache import DatasetCacheMixin
from library.datasets.dataset_image_io import DatasetImageIOMixin
from library.datasets.feature_sidecars import FeatureSidecarMixin
from library.datasets.subsets import (
    AugHelper,
    BaseSubset,
    DreamBoothSubset,
    ImageInfo,
)

logger = logging.getLogger(__name__)

HIGH_VRAM = runtime_flags.HIGH_VRAM

# Module-level artist filter — set from train.py (`--artist_filter`). When non-empty,
# `load_dreambooth_dir` keeps only images whose caption contains the `@<artist>` tag.
_ARTIST_FILTER: Optional[str] = None


def set_artist_filter(artist: Optional[str]) -> None:
    global _ARTIST_FILTER
    if artist is None or artist == "":
        _ARTIST_FILTER = None
        return
    _ARTIST_FILTER = artist if artist.startswith("@") else f"@{artist}"


def _caption_has_artist(caption: Optional[str], needle: str) -> bool:
    if not caption:
        return False
    needle_lc = needle.lower()
    for tag in caption.split(","):
        if tag.strip().lower() == needle_lc:
            return True
    return False


def enable_high_vram():
    global HIGH_VRAM
    runtime_flags.enable_high_vram()
    HIGH_VRAM = runtime_flags.HIGH_VRAM


class BaseDataset(FeatureSidecarMixin, DatasetCacheMixin, DatasetBucketsMixin, DatasetCaptionMixin, DatasetImageIOMixin, DatasetGetItemMixin, torch.utils.data.Dataset):
    def __init__(
        self,
        network_multiplier: float,
        debug_dataset: bool,
        resize_interpolation: Optional[str] = None,
        resolution: int = 1024,
        enable_bucket: bool = True,
        min_bucket_reso: int = 256,
        max_bucket_reso: int = 2048,
        bucket_reso_steps: int = 64,
        bucket_no_upscale: bool = False,
    ) -> None:
        super().__init__()

        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[DreamBoothSubset] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.bucket_manager: BucketManager = None  # not initialized
        self.bucket_info = None  # for metadata
        self.resolution = int(resolution or 1024)
        self.enable_bucket = bool(enable_bucket)
        self.min_bucket_reso = int(min_bucket_reso or 256)
        self.max_bucket_reso = int(max_bucket_reso or 2048)
        self.bucket_reso_steps = int(bucket_reso_steps or 64)
        self.bucket_no_upscale = bool(bucket_no_upscale)

        self.current_epoch: int = 0

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        if resize_interpolation is not None:
            assert validate_interpolation_fn(resize_interpolation), (
                f'Resize interpolation "{resize_interpolation}" is not a valid interpolation'
            )
        self.resize_interpolation = resize_interpolation

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, DreamBoothSubset] = {}

        self.replacements = {}

        # Functional-loss inversion supervision (postfix-func).
        # Set via `dataset.inversion_dir = ...` after construction; None disables.
        self.inversion_dir: Optional[str] = None
        self.inversion_num_runs: int = 3

        # BYG unpaired-editing per-image text conditionings. Set via
        # `dataset.byg_text_dir = ...` after construction; None disables.
        self.byg_text_dir: Optional[str] = None
        self._byg_roles = (
            "src_caption",
            "tgt_caption",
            "instruction",
            "reverse_instruction",
        )

        # IP-Adapter cached PE/vision features (sibling sidecars). Set via
        # `dataset.ip_features_cache_to_disk = True; dataset.ip_features_encoder = "pe"`
        # after construction. When enabled, __getitem__ loads
        # ``{stem}_anima_{encoder}.safetensors`` for every image and exposes
        # the stacked features as ``example["ip_features"]`` so train.py can
        # skip live PE encoding (and the dataset can keep cache_latents=true).
        self.ip_features_cache_to_disk: bool = False
        self.ip_features_encoder: str = "pe"
        # Force the cached-latent branches to ALSO load the source image into
        # ``example["images"]`` (in addition to ``example["latents"]``). Used
        # by IP-Adapter live PE encoding (PE-LoRA, or `cache_latents=true`
        # alongside non-cached PE features) so VAE latents stay cached while
        # the PE encoder gets a fresh image every step. Caller is responsible
        # for ensuring `subset.random_crop=False` so the live image matches
        # the deterministic crop baked into the cached latent.
        self.force_load_images_for_ip: bool = False

        # IP-Adapter distinct-pair (identity) training. When an
        # IdentityPairSampler is attached via ``setup_identity_pairs`` the
        # reference fed to the IP path (``example["ip_features"]``) is decoupled
        # from the VAE target: with probability ``ip_pair_prob`` a *different*
        # image of the target's identity supplies the PE features, removing the
        # self-pair copy shortcut. ``self`` (no sampler) = bit-identical legacy
        # behavior. See docs/proposal/ip-adapter-identity-pairs.md.
        self.identity_pair_sampler = None  # IdentityPairSampler | None
        self.ip_pair_prob: float = 0.8
        self.ip_pair_caption_strip_p: float = 0.0
        self.ip_pair_is_validation: bool = False
        self._ip_pair_strip_warned: bool = False

        # Soft-tokens contrastive negatives. When a sampler is attached via
        # ``setup_contrastive_negatives`` each example carries
        # ``neg_crossattn_emb`` of shape (B, k, S, D): k cached text embeddings
        # of *unrelated* images, used as InfoNCE negatives. Reuses the
        # IdentityPairSampler's ``shuffled`` policy (Phase 1). Decoupled from the
        # VAE target — same cached-feature-swap trick as IP-Adapter pairs, but
        # the swapped feature is the text embedding, not the PE feature. See
        # docs/proposal/soft_tokens_contrastive.md.
        self.contrastive_neg_sampler = None  # IdentityPairSampler | None
        self.contrastive_neg_k: int = 1
        self.contrastive_neg_mode: str = "shuffled"

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

        self.tokenize_strategy = None
        self.text_encoder_output_caching_strategy = None
        self.latents_caching_strategy = None

    def set_current_strategies(self):
        self.tokenize_strategy = TokenizeStrategy.get_strategy()
        self.text_encoder_output_caching_strategy = (
            TextEncoderOutputsCachingStrategy.get_strategy()
        )
        self.latents_caching_strategy = LatentsCachingStrategy.get_strategy()

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:
            if epoch > self.current_epoch:
                logger.info(
                    "epoch is incremented. current_epoch: {}, epoch: {}".format(
                        self.current_epoch, epoch
                    )
                )
                num_epochs = epoch - self.current_epoch
                for _ in range(num_epochs):
                    self.current_epoch += 1
                    self.shuffle_buckets()
            else:
                logger.warning(
                    "epoch is not incremented. current_epoch: {}, epoch: {}".format(
                        self.current_epoch, epoch
                    )
                )
                self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def __len__(self):
        return self._length
