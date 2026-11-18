"""CPU contract checks; numerical acceptance is documented separately."""
import os
from types import SimpleNamespace

import pytest
import torch

from library.quantization.qwen_nf4_compile_policy import qwen_nf4_compile_experiment_errors
from library.quantization.training import validate_mixed_nf4_training
from library.models.qwen_image_2_1.nf4_compile_runtime import (
    prepare_qwen_nf4_compile_experiment, qwen_nf4_compile_session,
)


def args(**overrides):
    values = dict(model_family="qwen_image_2_1", qwen_nf4_compile_experiment=True,
                  torch_compile=True, base_compute="bf16", mixed_precision="bf16",
                  attn_mode="torch", gradient_checkpointing=True,
                  compile_dynamic_seq=False, compile_seq_bands=False,
                  compile_block_scope="all", dynamo_backend="inductor",
                  blocks_to_swap=24, block_swap_restore_mode="foreach",
                  network_dim=16, network_alpha=16, qwen_image_2_1_task="edit")
    values.update(overrides)
    return SimpleNamespace(**values)


def model_and_network():
    from networks.lora_modules.lora import LoRAModule
    model = torch.nn.Module()
    model._mixed_nf4_family = "qwen_image_2_1"
    model.gradient_checkpointing = True
    model.blocks_to_swap = 24
    model._qwen_image_2_1_block_swap_adapter = SimpleNamespace(
        offloader=SimpleNamespace(restore_mode="foreach", transfer_dtype="bf16"))
    model.transformer_blocks = torch.nn.ModuleList()
    adapters = []
    for index in range(32):
        block = torch.nn.Module()
        block.img_norm1 = torch.nn.LayerNorm(2, elementwise_affine=False)
        block.img_norm2 = torch.nn.LayerNorm(2, elementwise_affine=False)
        block.img_mlp = torch.nn.Module()
        block.img_mlp.activation_fn = torch.nn.SiLU()
        block.attn = torch.nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(block.attn, name, torch.nn.Linear(2, 2))
        block.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        model.transformer_blocks.append(block)
        for name, module in block.attn.named_modules():
            if isinstance(module, torch.nn.Linear):
                lora = LoRAModule(f"{index}_{name}", module, lora_dim=16, alpha=16)
                lora.apply_to()
                adapters.append(lora)
    return model, SimpleNamespace(unet_loras=adapters, text_encoder_loras=[])


@pytest.mark.parametrize("change", [dict(model_family="anima"), dict(mixed_precision="fp16"),
    dict(base_compute="nf4"), dict(compile_dynamic_seq=True), dict(compile_seq_bands=True),
    dict(compile_block_scope="resident"), dict(allow_unverified_nf4_compile=True),
    dict(network_args=["use_lokr=true"]), dict(auto_block_swap=True), dict(attn_mode="flash")])
def test_illegal_contract_is_not_bypassable(change):
    config = args(**change)
    assert qwen_nf4_compile_experiment_errors(config.model_family, config)


def test_default_gate_and_actual_weights():
    model = SimpleNamespace(_mixed_nf4_family="qwen_image_2_1")
    with pytest.raises(ValueError, match="torch_compile disabled"):
        validate_mixed_nf4_training(model, args(qwen_nf4_compile_experiment=False))
    validate_mixed_nf4_training(model, args())
    with pytest.raises(ValueError, match="actual mixed NF4"):
        validate_mixed_nf4_training(SimpleNamespace(), args())


@pytest.mark.parametrize("failure", [False, True])
def test_session_restores_and_nests(monkeypatch, failure):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        with qwen_nf4_compile_session(args()):
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            with qwen_nf4_compile_session(args()):
                pass
            assert torch.are_deterministic_algorithms_enabled()
            if failure:
                raise RuntimeError("intentional")
    except RuntimeError:
        assert failure
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert before == (torch.are_deterministic_algorithms_enabled(),
                      torch.is_deterministic_algorithms_warn_only_enabled())


@pytest.mark.parametrize("workspace,initialized", [(":16:8", False), (None, True)])
def test_late_or_conflicting_environment_refused(monkeypatch, workspace, initialized):
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    with pytest.raises(ValueError):
        with qwen_nf4_compile_session(args()):
            pytest.fail("invalid environment entered")


def test_patch_atomic_across_norm_and_activation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    model, network = model_and_network()
    monkeypatch.setattr("library.models.qwen_image_2_1.nf4_compile_runtime._verify_packed_model", lambda model: None)
    before = model.transformer_blocks[0].img_norm1.forward
    model.transformer_blocks[-1].img_mlp.activation_fn = torch.nn.SiLU(inplace=True)
    with qwen_nf4_compile_session(args()):
        with pytest.raises(ValueError, match="inplace=False"):
            prepare_qwen_nf4_compile_experiment(model, network, args())
    assert model.transformer_blocks[0].img_norm1.forward == before


@pytest.mark.parametrize("as_mapping", [False, True])
def test_compile_installs_joint_options_and_first_order(monkeypatch, as_mapping):
    from library.models.qwen_image_2_1.compile import compile_qwen_image_2_1_blocks
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(kw) or fn)
    model, network = model_and_network()
    monkeypatch.setattr("library.models.qwen_image_2_1.nf4_compile_runtime._verify_packed_model", lambda model: None)
    # The fixture has no swap wrapper; real runtime offloader remains checked.
    del model._qwen_image_2_1_block_swap_adapter
    model._qwen_image_2_1_block_swap_adapter = SimpleNamespace(
        offloader=SimpleNamespace(restore_mode="foreach", transfer_dtype="bf16"),
        inner_forwards=[block.forward for block in model.transformer_blocks])
    config = vars(args()) if as_mapping else args()
    with qwen_nf4_compile_session(config):
        assert compile_qwen_image_2_1_blocks(model, dynamic_seq=False,
            experiment_args=config, network=network) == 32
        x = torch.randn(2, 2, requires_grad=True)
        block = model.transformer_blocks[0]
        actual = block.img_mlp.activation_fn(block.img_norm1(x))
        expected = torch.nn.functional.silu(torch.nn.functional.layer_norm(x, (2,)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), x)[0],
            torch.autograd.grad(expected.sum(), x)[0], rtol=0, atol=0)
    assert len(calls) == 32
    assert all(call["options"] == {"emulate_precision_casts": True,
        "triton.persistent_reductions": False} for call in calls)


def test_web_legal_and_illegal_experiment(tmp_path):
    from tests.test_mixed_nf4_preflight_risks import header_checkpoint, collect
    from tests.nf4_preflight_test_support import write_ordinary_checkpoint
    path = header_checkpoint(tmp_path / "mixed.safetensors", "qwen_image_2_1")
    checks = collect(vars(args()), path)
    assert len(checks) == 1 and checks[0]["level"] == "warning"
    checks = collect(vars(args(compile_dynamic_seq=True)), path)
    assert any(check["level"] == "error" for check in checks)
    assert not any(check.get("overridable") for check in checks)
    write_ordinary_checkpoint(path)
    checks = collect(vars(args()), path)
    assert checks[0]["level"] == "error" and (
        "mixed-NF4" in checks[0]["message"] or "actual mixed NF4" in checks[0]["message"]
    )


def test_session_facade_early_and_failure_reporting(monkeypatch):
    from library.training import train_session
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    reported = []
    def reporting(trainer, config, callback):
        try:
            return callback(trainer, config)
        except RuntimeError:
            reported.append("failure")
            raise
    def fail(trainer, config):
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        raise RuntimeError("intentional")
    monkeypatch.setattr(train_session, "reported_session", reporting)
    monkeypatch.setattr(train_session, "_run_training_session", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        train_session.run_training_session(SimpleNamespace(), args())
    assert reported == ["failure"] and "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_cli_context_precedes_precision_and_factory(monkeypatch):
    import argparse
    from library.training import cli_entry
    from library.training.adaptive_runtime import precision
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    config = args(artist_filter=None)
    monkeypatch.setattr(cli_entry._config_schema, "populate_schema", lambda *a, **k: None)
    monkeypatch.setattr(cli_entry, "verify_command_line_training_args", lambda args: None)
    monkeypatch.setattr(cli_entry, "read_config_from_file", lambda *a: config)
    def check(config):
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setattr(precision, "resolve_adaptive_precision", check)
    def factory():
        check(config)
        return SimpleNamespace(train=check)
    cli_entry.run_training_cli(setup_parser=argparse.ArgumentParser, trainer_factory=factory,
        install_stop_signal_handlers=lambda: None, install_crash_reporter=lambda args: None,
        argv=["train"])
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_marker_cannot_disguise_ordinary_model():
    from library.models.qwen_image_2_1.nf4_compile_runtime import _verify_packed_model
    model, _network = model_and_network()
    with pytest.raises(ValueError, match="232 packed"):
        _verify_packed_model(model)


@pytest.mark.parametrize("mode", [None, "default"])
def test_nf4_default_mode_accepts_actual_torch_compile_options(mode):
    from library.models.qwen_image_2_1.compile import compile_qwen_image_2_1_blocks

    model = torch.nn.Module()
    model._mixed_nf4_family = "qwen_image_2_1"
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])
    # Construction uses the real torch.compile argument validation, without
    # executing a kernel or pretending this fixture is GPU numerical evidence.
    assert compile_qwen_image_2_1_blocks(model, mode=mode) == 1


def test_segmented_block_refused_before_any_native_patch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr("library.models.qwen_image_2_1.nf4_compile_runtime._verify_packed_model", lambda model: None)
    model, network = model_and_network()
    before = model.transformer_blocks[0].img_norm1.forward
    model.transformer_blocks[-1]._qwen_segmented_forward = object()
    with qwen_nf4_compile_session(args()):
        with pytest.raises(ValueError, match="segmented block"):
            prepare_qwen_nf4_compile_experiment(model, network, args())
    assert model.transformer_blocks[0].img_norm1.forward == before


@pytest.mark.parametrize("value", ["0", "0.0", "0e0", "-0.0"])
def test_serialized_zero_dropout_is_allowed(value):
    config = args(network_dropout=value, network_args=[
        f"dropout={value}", f"rank_dropout={value}", f"module_dropout={value}"])
    assert qwen_nf4_compile_experiment_errors(config.model_family, config) == []


@pytest.mark.parametrize("value", ["0.01", "-0.01", "nan", "inf", "invalid"])
def test_nonzero_or_invalid_dropout_stays_rejected(value):
    config = args(network_dropout=value, network_args=[
        f"dropout={value}", f"rank_dropout={value}", f"module_dropout={value}"])
    errors = qwen_nf4_compile_experiment_errors(config.model_family, config)
    assert all(f"{key} disabled" in errors for key in
               ("network_dropout", "dropout", "rank_dropout", "module_dropout"))


def test_malformed_network_args_produce_structured_compatibility_error():
    from library.training.compat_matrix import check_training_compat

    config = args(network_args=["dropout"])
    result = check_training_compat(config)
    assert not result.ok
    assert any(issue.code == "qwen_nf4_compile_experiment" for issue in result.errors)
