from __future__ import annotations

from argparse import Namespace

import pytest
import torch
from torch import nn

from library.models.krea2_raw.pipeline_parallel import (
    Krea2PipelineParallelConfig,
    build_krea2_block_stage,
    make_krea2_pipeline_plan,
    validate_krea2_pipeline_config,
)
from library.training import cli_entry
from library.training.cli_args import setup_parser
from library.training.compat_matrix import check_training_compat


def _valid_config(**overrides):
    config = {
        "pipeline_parallel": True,
        "pipeline_parallel_stages": 2,
        "pipeline_parallel_microbatches": 4,
        "pipeline_parallel_schedule": "1f1b",
        "pipeline_parallel_split": "balanced",
        "model_family": "krea2_raw",
        "blocks_to_swap": 0,
        "torch_compile": False,
        "selective_checkpoint": "off",
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
        "network_train_unet_only": True,
    }
    config.update(overrides)
    return config


def _codes(items) -> set[str]:
    return {item.code for item in items}


def test_pipeline_config_defaults_match_krea_method_and_webui() -> None:
    config = Krea2PipelineParallelConfig.from_config({})

    assert config == Krea2PipelineParallelConfig(
        enabled=False,
        stages=2,
        microbatches=4,
        schedule="1f1b",
        split="balanced",
    )


def test_pipeline_config_normalizes_mapping_and_namespace_values() -> None:
    mapping = Krea2PipelineParallelConfig.from_config(
        {
            "pipeline_parallel": "yes",
            "pipeline_parallel_stages": "2",
            "pipeline_parallel_microbatches": "8",
            "pipeline_parallel_schedule": " 1F1B ",
            "pipeline_parallel_split": " BALANCED ",
        }
    )
    namespace = Krea2PipelineParallelConfig.from_config(
        Namespace(pipeline_parallel="false", pipeline_parallel_microbatches=2)
    )

    assert mapping == Krea2PipelineParallelConfig(True, 2, 8, "1f1b", "balanced")
    assert namespace.enabled is False
    assert namespace.microbatches == 2


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pipeline_parallel": "invalid"}, "pipeline_parallel must be a boolean"),
        (
            {"pipeline_parallel_stages": "invalid"},
            "pipeline_parallel_stages must be an integer",
        ),
        (
            {"pipeline_parallel_microbatches": ""},
            "pipeline_parallel_microbatches must be an integer",
        ),
    ],
)
def test_pipeline_config_rejects_malformed_mapping_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Krea2PipelineParallelConfig.from_config(_valid_config(**overrides))


@pytest.mark.parametrize("family", ["krea2", "krea2_raw"])
def test_pipeline_validator_accepts_krea_aliases_and_two_rank_topology(
    family: str,
) -> None:
    validated = validate_krea2_pipeline_config(
        _valid_config(model_family=family),
        world_size=2,
    )

    assert validated.enabled is True
    assert validated.microbatches == 4


@pytest.mark.parametrize(
    ("overrides", "world_size", "message"),
    [
        ({"model_family": "anima"}, 2, "supported only"),
        ({"pipeline_parallel_stages": 3}, 3, "must be 2"),
        ({"pipeline_parallel_microbatches": 0}, 2, "between 1 and 1024"),
        ({"pipeline_parallel_microbatches": 1025}, 2, "between 1 and 1024"),
        ({"pipeline_parallel_schedule": "gpipe"}, 2, "schedule must be one of"),
        ({"pipeline_parallel_schedule": ""}, 2, "schedule must be one of"),
        ({"pipeline_parallel_split": "manual"}, 2, "split must be one of"),
        ({}, 1, "must equal the distributed world size"),
    ],
)
def test_pipeline_validator_rejects_invalid_topology(
    overrides: dict[str, object], world_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_krea2_pipeline_config(
            _valid_config(**overrides),
            world_size=world_size,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"blocks_to_swap": 1},
        {"torch_compile": True},
        {"selective_checkpoint": "every_other"},
        {"cpu_offload_checkpointing": True},
        {"unsloth_offload_checkpointing": True},
        {"network_train_unet_only": False},
    ],
)
def test_pipeline_validator_rejects_unsupported_combinations(override) -> None:
    with pytest.raises(ValueError, match="currently cannot be combined"):
        validate_krea2_pipeline_config(_valid_config(**override), world_size=2)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"blocks_to_swap": "invalid"}, "blocks_to_swap must be an integer"),
        ({"torch_compile": "invalid"}, "torch_compile must be a boolean"),
    ],
)
def test_pipeline_validator_rejects_malformed_compatibility_values(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_krea2_pipeline_config(_valid_config(**override), world_size=2)


def test_pipeline_validator_checks_expected_block_count() -> None:
    with pytest.raises(ValueError, match="exceeds Krea-2 block count 1"):
        validate_krea2_pipeline_config(
            _valid_config(),
            world_size=2,
            num_blocks=1,
        )


def test_disabled_pipeline_skips_topology_constraints() -> None:
    validated = validate_krea2_pipeline_config(
        {
            "pipeline_parallel": False,
            "model_family": "anima",
            "pipeline_parallel_stages": 99,
        },
        world_size=1,
    )

    assert validated.enabled is False


def test_balanced_two_stage_plan_is_contiguous_13_15() -> None:
    plan = make_krea2_pipeline_plan(stages=2, num_blocks=28)

    assert plan.ranges == ((0, 13), (13, 28))
    assert plan.indices_for_stage(0) == tuple(range(13))
    assert plan.indices_for_stage(1) == tuple(range(13, 28))
    assert plan.ranges[0][1] == plan.ranges[1][0]
    assert plan.ranges[-1][1] == plan.num_blocks


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stages": 0, "num_blocks": 28}, "stages must be positive"),
        ({"stages": 3, "num_blocks": 2}, "must be >= stages"),
    ],
)
def test_pipeline_plan_rejects_invalid_sizes(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_krea2_pipeline_plan(**kwargs)


def test_pipeline_plan_rejects_invalid_stage_index() -> None:
    plan = make_krea2_pipeline_plan(stages=2, num_blocks=28)

    with pytest.raises(ValueError, match="stage_index"):
        plan.range_for_stage(2)


class _RecordingBlock(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(delta))
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def forward(self, combined, tvec, freqs, mask=None):
        self.calls.append((tvec, freqs, mask))
        return combined + self.delta


class _DummyKreaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            _RecordingBlock(float(index + 1)) for index in range(4)
        )


def test_block_stage_preserves_order_inputs_and_parameter_identity() -> None:
    model = _DummyKreaModel()
    plan, stage = build_krea2_block_stage(model, stage_index=1, stages=2)
    combined = torch.tensor([1.0])
    tvec = torch.tensor([2.0])
    freqs = torch.tensor([3.0])
    mask = torch.tensor([True])

    output = stage(combined, tvec, freqs, mask)

    assert plan.ranges == ((0, 1), (1, 4))
    assert output.item() == pytest.approx(10.0)
    assert stage.stage_index == 1
    assert stage.block_range == (1, 4)
    assert stage.global_block_indices == (1, 2, 3)
    assert stage.blocks[0] is model.blocks[1]
    assert stage.blocks[0].delta is model.blocks[1].delta
    assert list(stage.state_dict()) == [
        "blocks.0.delta",
        "blocks.1.delta",
        "blocks.2.delta",
    ]
    assert stage.state_dict_key_map() == {
        "blocks.0.delta": "blocks.1.delta",
        "blocks.1.delta": "blocks.2.delta",
        "blocks.2.delta": "blocks.3.delta",
    }
    assert all(block.calls == [(tvec, freqs, mask)] for block in model.blocks[1:])
    assert model.blocks[0].calls == []


def test_block_stage_requires_model_blocks() -> None:
    with pytest.raises(TypeError, match="must expose a blocks ModuleList"):
        build_krea2_block_stage(nn.Identity(), stage_index=0, stages=2)


def test_cli_defaults_and_choices_match_pipeline_contract() -> None:
    parser = setup_parser()
    args = parser.parse_args([])
    actions = {action.dest: action for action in parser._actions}

    assert args.pipeline_parallel is False
    assert args.pipeline_parallel_stages == 2
    assert args.pipeline_parallel_microbatches == 4
    assert actions["pipeline_parallel_stages"].choices == [2]
    assert actions["pipeline_parallel_schedule"].choices == ["1f1b"]
    assert actions["pipeline_parallel_split"].choices == ["balanced"]


def test_pipeline_fields_are_in_runtime_config_schema() -> None:
    from library.config import schema as config_schema

    parser = setup_parser()
    config_schema.populate_schema(parser)
    schema = config_schema.get_schema()

    assert {
        "pipeline_parallel",
        "pipeline_parallel_stages",
        "pipeline_parallel_microbatches",
        "pipeline_parallel_schedule",
        "pipeline_parallel_split",
    } <= schema.keys()
    assert schema["pipeline_parallel"].type == "bool"
    assert schema["pipeline_parallel_stages"].type == "int"
    assert schema["pipeline_parallel_stages"].choices == (2,)
    assert schema["pipeline_parallel_microbatches"].type == "int"
    assert schema["pipeline_parallel_schedule"].choices == ("1f1b",)
    assert schema["pipeline_parallel_split"].choices == ("balanced",)


def test_compat_matrix_keeps_valid_pipeline_fail_closed() -> None:
    result = check_training_compat(_valid_config(attn_mode="torch"))

    assert "pipeline_parallel_config" not in _codes(result.errors)
    assert "pipeline_parallel_runtime_unavailable" in _codes(result.errors)


def test_compat_matrix_reports_invalid_config_before_runtime_gate() -> None:
    result = check_training_compat(_valid_config(attn_mode="torch", torch_compile=True))

    assert "pipeline_parallel_config" in _codes(result.errors)
    assert "pipeline_parallel_runtime_unavailable" not in _codes(result.errors)


def test_compat_matrix_rejects_malformed_pipeline_toggle() -> None:
    result = check_training_compat(
        _valid_config(attn_mode="torch", pipeline_parallel="invalid")
    )

    assert "pipeline_parallel_config" in _codes(result.errors)
    assert "pipeline_parallel_runtime_unavailable" not in _codes(result.errors)


def test_compat_matrix_checks_pipeline_world_size_before_runtime_gate() -> None:
    result = check_training_compat(
        _valid_config(attn_mode="torch"),
        world_size=1,
    )

    assert "pipeline_parallel_config" in _codes(result.errors)
    assert "pipeline_parallel_runtime_unavailable" not in _codes(result.errors)


def test_compat_matrix_plans_anima_through_the_pp_v0_runtime() -> None:
    """Anima is the one family whose stage-local PP runtime exists.

    The compatibility layer therefore stops rejecting it -- but the trainer entry
    still refuses a PP run with a named code instead of falling back to DDP (see
    ``tests/test_anima_pp_wiring.py``).
    """

    result = check_training_compat(_valid_config(model_family="anima"))

    assert "pipeline_parallel_config" not in _codes(result.errors)
    assert "pipeline_parallel_runtime_unavailable" not in _codes(result.errors)


def test_cli_refuses_to_fall_back_to_data_parallel(monkeypatch) -> None:
    args = Namespace(**_valid_config(attn_mode="torch"))

    class _Parser:
        def parse_args(self, _argv):
            return args

    trainer_created = False

    def trainer_factory():
        nonlocal trainer_created
        trainer_created = True
        return object()

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        cli_entry._config_schema, "populate_schema", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        cli_entry, "verify_command_line_training_args", lambda _args: None
    )
    monkeypatch.setattr(
        cli_entry, "read_config_from_file", lambda parsed, _parser: parsed
    )

    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        cli_entry.run_training_cli(
            setup_parser=_Parser,
            trainer_factory=trainer_factory,
            install_stop_signal_handlers=lambda: None,
            install_crash_reporter=lambda _argv: None,
            argv=["train.py"],
        )

    assert trainer_created is False


def test_cli_normalizes_string_false_without_entering_pipeline_gate(monkeypatch) -> None:
    args = Namespace(
        **_valid_config(pipeline_parallel="false", attn_mode="torch")
    )

    class _Parser:
        def parse_args(self, _argv):
            return args

    trained = False

    class _Trainer:
        def train(self, trained_args):
            nonlocal trained
            trained = True
            assert trained_args is args

    monkeypatch.setattr(
        cli_entry._config_schema, "populate_schema", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        cli_entry, "verify_command_line_training_args", lambda _args: None
    )
    monkeypatch.setattr(
        cli_entry, "read_config_from_file", lambda parsed, _parser: parsed
    )

    cli_entry.run_training_cli(
        setup_parser=_Parser,
        trainer_factory=_Trainer,
        install_stop_signal_handlers=lambda: None,
        install_crash_reporter=lambda _argv: None,
        argv=["train.py"],
    )

    assert trained is True
