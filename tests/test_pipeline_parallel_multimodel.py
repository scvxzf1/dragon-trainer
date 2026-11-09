from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.anima.pipeline_parallel import AnimaBlockStage
from library.models.family_registry import (
    MODEL_FAMILY_REGISTRY,
    model_family_capability_catalog,
)
from library.models.krea2_raw.pipeline_parallel import Krea2BlockStage
from library.models.pipeline_parallel import (
    PipelineParallelConfig,
    build_pipeline_block_stage,
    make_pipeline_plan,
    validate_pipeline_parallel_config,
)
from library.models.z_image.pipeline_parallel import ZImageBlockStage
from library.training.compat_matrix import check_training_compat


def _pipeline_config(family: str, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_family": family,
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 4,
        "pipeline_parallel_schedule": "1f1b",
        "pipeline_parallel_split": "balanced",
        "blocks_to_swap": 0,
        "torch_compile": False,
        "selective_checkpoint": "off",
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
        "network_train_unet_only": True,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("family", ["anima", "krea2", "krea2_raw", "zimage", "z_image"])
def test_shared_pipeline_validator_accepts_registered_family_aliases(family: str) -> None:
    config = validate_pipeline_parallel_config(
        _pipeline_config(family),
        world_size=2,
    )

    assert config == PipelineParallelConfig(True, 2, 4, "1f1b", "balanced")


@pytest.mark.parametrize(
    ("family", "num_blocks", "container", "ranges"),
    [
        ("anima", 28, "blocks", ((0, 14), (14, 28))),
        ("anima", 40, "blocks", ((0, 20), (20, 40))),
        ("krea2", 28, "blocks", ((0, 13), (13, 28))),
        ("zimage", 30, "layers", ((0, 15), (15, 30))),
    ],
)
def test_shared_pipeline_plan_uses_family_topology(
    family: str,
    num_blocks: int,
    container: str,
    ranges: tuple[tuple[int, int], ...],
) -> None:
    plan = make_pipeline_plan(family=family, stages=2, num_blocks=num_blocks)

    assert plan.block_container == container
    assert plan.ranges == ranges
    assert plan.indices_for_stage(1) == tuple(range(*ranges[1]))


def test_capability_catalog_exposes_all_registered_pipeline_planners() -> None:
    catalog = {item["name"]: item for item in model_family_capability_catalog()}

    assert set(catalog) == set(MODEL_FAMILY_REGISTRY)
    assert catalog["anima"]["pipeline_parallel"]["known_num_blocks"] == [28, 40]
    assert catalog["krea2_raw"]["pipeline_parallel"]["runtime_available"] is False
    assert catalog["z_image"]["pipeline_parallel"]["block_container"] == "layers"


@pytest.mark.parametrize("family", ["krea2_raw", "z_image"])
def test_compat_matrix_keeps_every_family_runtime_fail_closed(family: str) -> None:
    result = check_training_compat(_pipeline_config(family), world_size=2)
    pipeline_codes = {
        item.code for item in result.errors if item.key == "pipeline_parallel"
    }

    assert pipeline_codes == {"pipeline_parallel_runtime_unavailable"}


def test_compat_matrix_releases_only_the_anima_runtime() -> None:
    """Anima publishes ``torch_pipelining_1f1b``; the compat gate follows it."""

    result = check_training_compat(_pipeline_config("anima"), world_size=2)
    pipeline_codes = {
        item.code for item in result.errors if item.key == "pipeline_parallel"
    }

    assert pipeline_codes == set()
    warnings = {
        item.code for item in result.warnings if item.key == "pipeline_parallel"
    }
    assert warnings == {"pipeline_parallel_experimental_runtime"}


@pytest.mark.parametrize("family", ["krea2", "zimage"])
def test_compat_matrix_accepts_declared_family_aliases(family: str) -> None:
    result = check_training_compat(_pipeline_config(family), world_size=2)
    pipeline_codes = {
        item.code for item in result.errors if item.key == "pipeline_parallel"
    }

    assert "invalid_model_family" not in {item.code for item in result.errors}
    assert pipeline_codes == {"pipeline_parallel_runtime_unavailable"}


def test_shared_validator_reports_one_cross_family_config_error_code() -> None:
    result = check_training_compat(
        _pipeline_config("z_image", blocks_to_swap=1),
        world_size=2,
    )

    assert {
        item.code for item in result.errors if item.key == "pipeline_parallel"
    } == {"pipeline_parallel_config"}


class _AnimaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[object, ...]] = []

    def forward(
        self,
        hidden,
        embedding,
        crossattn_emb,
        attn_params,
        **kwargs,
    ):
        self.calls.append((crossattn_emb, attn_params, kwargs))
        return hidden + embedding * self.weight


class _KreaBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden, tvec, freqs, mask):
        del freqs, mask
        return hidden + tvec * self.weight


class _ZImageBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        hidden,
        attn_mask,
        freqs_cis,
        adaln_input,
        noise_mask,
        adaln_noisy,
        adaln_clean,
    ):
        del attn_mask, freqs_cis, noise_mask, adaln_noisy, adaln_clean
        return hidden + adaln_input * self.weight


def test_anima_stage_preserves_family_call_contract_and_global_keys() -> None:
    blocks = nn.ModuleList([_AnimaBlock() for _ in range(4)])
    model = SimpleNamespace(blocks=blocks)
    plan, stage = build_pipeline_block_stage(
        model,
        family="anima",
        stage_index=1,
        stages=2,
    )
    crossattn = object()
    attn_params = object()
    output = stage(
        torch.tensor(1.0),
        torch.tensor(0.0),
        crossattn,
        attn_params,
        block_embeddings=[torch.tensor(2.0), torch.tensor(3.0)],
    )

    assert isinstance(stage, AnimaBlockStage)
    assert plan.ranges == ((0, 2), (2, 4))
    assert output.item() == pytest.approx(6.0)
    assert stage.blocks[0] is blocks[2]
    assert stage.state_dict_key_map()["blocks.0.weight"] == "blocks.2.weight"
    assert stage.blocks[0].calls[0][:2] == (crossattn, attn_params)


def test_krea_stage_uses_blocks_container_and_legacy_signature() -> None:
    blocks = nn.ModuleList([_KreaBlock() for _ in range(4)])
    plan, stage = build_pipeline_block_stage(
        SimpleNamespace(blocks=blocks),
        family="krea2",
        stage_index=0,
        stages=2,
    )
    output = stage(
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(0.0),
        None,
    )

    assert isinstance(stage, Krea2BlockStage)
    assert plan.ranges == ((0, 1), (1, 4))
    assert output.item() == pytest.approx(3.0)


def test_z_image_stage_uses_layers_container_and_global_keys() -> None:
    layers = nn.ModuleList([_ZImageBlock() for _ in range(4)])
    plan, stage = build_pipeline_block_stage(
        SimpleNamespace(layers=layers),
        family="zimage",
        stage_index=1,
        stages=2,
    )
    output = stage(
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(2.0),
    )

    assert isinstance(stage, ZImageBlockStage)
    assert plan.ranges == ((0, 2), (2, 4))
    assert output.item() == pytest.approx(5.0)
    assert stage.blocks[0] is layers[2]
    assert stage.state_dict_key_map()["blocks.0.weight"] == "layers.2.weight"


def test_stage_builder_fails_closed_for_unknown_family() -> None:
    with pytest.raises(ValueError, match="pipeline_parallel model_family"):
        build_pipeline_block_stage(
            SimpleNamespace(blocks=nn.ModuleList()),
            family="unknown",
            stage_index=0,
            stages=2,
        )
