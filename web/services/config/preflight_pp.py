"""Krea-2 PP batch geometry checks without model loading."""

from typing import Any

from library.training.pp.krea2_batch import batch_divisibility_message


def krea2_batch_checks(cfg: dict[str, Any]) -> list[dict[str, str]]:
    if str(cfg.get("model_family", "")).lower() not in {"krea2", "krea2_raw"}:
        return []
    # WebUI materializes its dataset with prefer_train_batch_size=True.
    # Check that batch, rather than the source blueprint's overridden value.
    batch = int(cfg.get("train_batch_size", 1) or 1)
    microbatches = int(cfg.get("pipeline_parallel_microbatches", 4))
    message = batch_divisibility_message(batch, microbatches)
    return [{
        "level": "error",
        "key": "pipeline_parallel_microbatches",
        "code": "pp_krea2_batch_not_divisible",
        "message": "Krea-2 微批数量与 batch 不匹配。"
        "请调整微批数量或训练 batch；" + message,
    }] if message else []


def cached_runtime_vae_is_unused(
    cfg: dict[str, Any], checks: list[dict[str, Any]], *,
    runtime_file: bool, dataset_count: int,
) -> bool:
    """Only a frozen runtime bypasses preprocessing and may skip VAE loading."""
    from types import SimpleNamespace

    from library.training.train_bootstrap import sample_preview_enabled
    from web.services.config.pp_cache_validation import krea2_runtime_latents_complete

    def enabled(value: Any) -> bool:
        return str(value or "").lower() not in {"", "0", "false", "none", "off", "no"}

    if (
        not runtime_file or dataset_count < 1
        or str(cfg.get("model_family", "")).lower() not in {"krea2", "krea2_raw"}
        or not enabled(cfg.get("pipeline_parallel"))
        or not enabled(cfg.get("cache_latents", cfg.get("use_vae_cache")))
        or not enabled(cfg.get("cache_text_encoder_outputs", cfg.get("use_text_cache")))
        or enabled(cfg.get("skip_cache_check"))
        or enabled(cfg.get("use_cmmd", True))
        or sample_preview_enabled(SimpleNamespace(**cfg))
    ):
        return False
    for index in range(1, dataset_count + 1):
        prefix = "" if index == 1 else f"dataset_{index}_"
        for kind in ("latent_cache", "text_cache"):
            matching = [check for check in checks if check["key"] == prefix + kind]
            if len(matching) != 1 or matching[0]["level"] != "ok":
                return False
    return krea2_runtime_latents_complete(cfg)
