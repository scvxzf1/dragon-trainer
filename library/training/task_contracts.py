"""Training-task capabilities and dataset contracts shared by CLI and WebUI."""

from collections.abc import Mapping

from library.config.normalize import parse_bool
from library.models.family_registry import (
    get_model_family_spec,
    normalize_registered_family,
)


def _get(config, key, default=None):
    return (
        config.get(key, default)
        if isinstance(config, Mapping)
        else getattr(config, key, default)
    )


def _enabled(value):
    return parse_bool(value, default=False)


def dataset_requires_edit(config) -> bool:
    datasets = _get(config, "datasets", [])
    return isinstance(datasets, (list, tuple)) and any(
        str(subset.get("reference_image_dir") or "").strip()
        for dataset in datasets
        if isinstance(dataset, Mapping)
        for subset in (dataset.get("subsets") or [])
        if isinstance(subset, Mapping)
    )


def configured_training_task(config, spec) -> str:
    # Preserve the legacy runtime default; data inspection never silently changes it.
    return (
        str(_get(config, spec.training_task_key, "t2i") or "t2i").strip().lower()
        if spec.training_task_key
        else "t2i"
    )


def task_config_values(family: str, task: str) -> dict[str, str]:
    spec = get_model_family_spec(
        normalize_registered_family(family, allow_aliases=True)
    )
    if task not in spec.supported_tasks:
        raise ValueError(
            f"当前模型 {spec.display_name} 不支持{'编辑数据集' if task == 'edit' else task}训练"
        )
    if task == "edit" and (
        spec.name not in EDIT_CONTRACTS or not spec.training_task_key
    ):
        raise ValueError(f"{spec.display_name} 编辑训练契约尚未接入")
    return {spec.training_task_key: task} if spec.training_task_key else {}


def check_training_task(out, config, spec) -> None:
    task = configured_training_task(config, spec)
    requires_edit = dataset_requires_edit(config)
    requested = "edit" if requires_edit else task
    if task not in {"t2i", "edit"}:
        out.error(
            "invalid_training_task",
            spec.training_task_key or "model_family",
            "训练任务必须为 t2i 或 edit",
        )
        return
    if requested not in spec.supported_tasks:
        out.error(
            "unsupported_training_task",
            "model_family",
            f"当前数据集需要编辑训练；当前模型 {spec.display_name} 不支持编辑数据集训练",
        )
        return
    if task == "edit":
        contract = EDIT_CONTRACTS.get(spec.name)
        if contract is None:
            out.error(
                "training_task_unavailable",
                "model_family",
                f"{spec.display_name} 编辑训练契约尚未接入",
            )
            return
        contract(out, config, spec)
    elif requires_edit:
        out.error(
            f"{spec.name}_edit_task_mismatch",
            spec.training_task_key or "dataset_config",
            "数据集包含编辑参考图，但训练任务为 t2i；请将训练任务改为 edit",
        )


def _check_qwen_paired_edit(out, config, spec) -> None:
    label = f"{spec.display_name} Edit"

    def forbid(key, active, message):
        if active:
            out.error(f"{spec.name}_{key}", key, message)

    forbid(
        "edit_text_cache",
        not _enabled(
            _get(config, "cache_text_encoder_outputs", _get(config, "use_text_cache"))
        ),
        f"{label} 需要启用 Qwen3-VL 条件缓存",
    )
    forbid(
        "edit_latent_cache",
        not _enabled(_get(config, "cache_latents", _get(config, "use_vae_cache"))),
        f"{label} 需要启用 VAE latent 缓存",
    )
    datasets = _get(config, "datasets")
    if not isinstance(datasets, (list, tuple)):
        forbid(
            "edit_subsets",
            not _get(config, "dataset_config"),
            f"{label} requires a dataset config",
        )
        return
    general = _get(config, "general", {})
    rows = [
        (dataset, subset)
        for dataset in datasets
        if isinstance(dataset, Mapping)
        for subset in (dataset.get("subsets") or [])
        if isinstance(subset, Mapping)
    ]
    forbid("edit_subsets", not rows, f"{label} requires at least one dataset subset")
    for index, (dataset, subset) in enumerate(rows):
        prefix = f"子集 {index + 1}"
        forbid(
            "edit_reference_dir",
            not str(subset.get("reference_image_dir") or "").strip(),
            f"{prefix} 缺少参考图目录（编辑前）",
        )
        forbid(
            "edit_regularization",
            _enabled(subset.get("is_reg")),
            f"{label} 不支持正则数据子集（{prefix}）",
        )
        batch = _get(config, "train_batch_size") or dataset.get(
            "batch_size", _get(general, "batch_size", 1)
        )
        try:
            valid_batch = int(batch) == 1
        except (TypeError, ValueError):
            valid_batch = False
        forbid(
            "edit_batch_size",
            not valid_batch,
            f"{label} requires batch_size=1（{prefix}）",
        )
        forbid(
            "edit_augmentation",
            any(
                _enabled(_get(source, key))
                for source in (config, general, dataset, subset)
                for key in ("flip_aug", "color_aug", "random_crop")
            ),
            f"{label} requires deterministic reference/target transforms（{prefix}）",
        )


# A capability declaration alone must never enable an unwired training task.
EDIT_CONTRACTS = {"qwen_image_2_1": _check_qwen_paired_edit}
