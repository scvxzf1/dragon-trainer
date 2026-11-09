from __future__ import annotations

from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
import traceback

import pytest
import torch
import torch.distributed as dist

from scripts.experiments.anima_parallel.checkpoint import (
    load_distributed_checkpoint,
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from scripts.experiments.anima_parallel.stages import (
    AnimaOwnedStage,
    build_schedule,
    stage_examples,
)


def _model(seed=123, checkpoint=True):
    from library.anima.models import Anima
    from scripts.experiments.anima_parallel.common import create_mlp_lora

    torch.manual_seed(seed)
    model = Anima(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=4,
        out_channels=4,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=24,
        num_blocks=4,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=24,
        pos_emb_learnable=True,
        use_adaln_lora=True,
        adaln_lora_dim=8,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    model.requires_grad_(False)
    network = create_mlp_lora(model, seed=seed + 1, rank_dim=4, alpha=4.0)
    with torch.no_grad():
        for lora in network.unet_loras:
            lora.lora_up.weight.normal_(std=0.02)
    if checkpoint:
        model.enable_gradient_checkpointing()
    return model.train(), network


def _batch(count):
    generator = torch.Generator().manual_seed(999)
    inputs = (
        torch.randn(count, 4, 1, 4, 4, generator=generator),
        torch.rand(count, generator=generator),
        torch.randn(count, 3, 24, generator=generator),
        torch.zeros(count, 1, 4, 4),
    )
    target = torch.randn(count, 4, 1, 4, 4, generator=generator)
    return inputs, target


def _baseline_step(model, optimizer, inputs, target):
    optimizer.zero_grad(set_to_none=True)
    outputs = []
    for index in range(len(target)):
        x, timestep, context, mask = (value[index : index + 1] for value in inputs)
        output = model(x, timestep, context, padding_mask=mask)
        (
            torch.nn.functional.mse_loss(output, target[index : index + 1])
            / len(target)
        ).backward()
        outputs.append(output.detach())
    return torch.cat(outputs)


def _pipeline_step(schedule, rank, optimizer, inputs, target):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    return (
        schedule.step(*inputs, losses=losses)
        if rank == 0
        else schedule.step(target=target, losses=losses)
    )


def _worker(rank, init_path, root, microbatches, checkpoint, split):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        baseline, baseline_network = _model(checkpoint=checkpoint)
        expected_optimizer = torch.optim.AdamW(baseline_network.parameters(), lr=1e-3)
        inputs, target = _batch(microbatches)
        model, network = _model(checkpoint=checkpoint)
        examples = tuple(value[:1] for value in inputs)
        boundary, output, rope = stage_examples(model, examples)
        module = AnimaOwnedStage(model, network, rank=rank, split=split, rope=rope)
        del network, model
        optimizer = torch.optim.AdamW(module.adapters.parameters(), lr=1e-3)
        schedule = build_schedule(
            module, examples, boundary, output, microbatches=microbatches
        )
        expected = _baseline_step(baseline, expected_optimizer, inputs, target)
        actual = _pipeline_step(schedule, rank, optimizer, inputs, target)
        if rank == 1:
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        expected_parameters = dict(baseline_network.named_parameters())
        for name, parameter in module.adapters.named_parameters():
            assert parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad, expected_parameters[name].grad, rtol=2e-4, atol=2e-7
            )
        optimizer.step()
        expected_optimizer.step()
        for name, parameter in module.adapters.named_parameters():
            torch.testing.assert_close(
                parameter, expected_parameters[name], rtol=2e-5, atol=2e-6
            )

        contract = {"microbatches": microbatches, "checkpoint": checkpoint}
        checkpoint_path = Path(root) / f"rank{rank}.pt"
        save_stage_checkpoint(
            checkpoint_path, module, optimizer, step=1, contract=contract
        )
        _pipeline_step(schedule, rank, optimizer, inputs, target)
        optimizer.step()
        uninterrupted = module.adapter_state()
        assert (
            load_stage_checkpoint(checkpoint_path, module, optimizer, contract=contract)
            == 1
        )
        _pipeline_step(schedule, rank, optimizer, inputs, target)
        optimizer.step()
        for name, value in module.adapter_state().items():
            torch.testing.assert_close(value, uninterrupted[name], rtol=0, atol=0)
        # Reconstruct ownership, optimizer and schedule as a restarted worker would.
        fresh_model, fresh_network = _model(checkpoint=checkpoint)
        fresh_boundary, fresh_output, fresh_rope = stage_examples(fresh_model, examples)
        fresh = AnimaOwnedStage(
            fresh_model, fresh_network, rank=rank, split=split, rope=fresh_rope
        )
        fresh_optimizer = torch.optim.AdamW(fresh.adapters.parameters(), lr=1e-3)
        load_stage_checkpoint(
            checkpoint_path, fresh, fresh_optimizer, contract=contract
        )
        fresh_schedule = build_schedule(
            fresh, examples, fresh_boundary, fresh_output, microbatches=microbatches
        )
        _pipeline_step(fresh_schedule, rank, fresh_optimizer, inputs, target)
        fresh_optimizer.step()
        for name, value in fresh.adapter_state().items():
            torch.testing.assert_close(value, uninterrupted[name], rtol=0, atol=0)

        save_stage_checkpoint(
            Path(root) / f"stage-rank{rank}.pt",
            module,
            optimizer,
            step=2,
            contract=contract,
        )
        if rank == 0:
            Path(root, "result.json").write_text("{}")
        dist.barrier()
        assert (
            load_distributed_checkpoint(
                Path(root), module, optimizer, contract=contract, group=dist.group.WORLD
            )
            == 2
        )
        with pytest.raises(ValueError, match="PP resume rejected"):
            load_distributed_checkpoint(
                Path(root),
                module,
                optimizer,
                contract=contract if rank == 0 else {"wrong": True},
                group=dist.group.WORLD,
            )
        try:
            load_stage_checkpoint(
                checkpoint_path, module, optimizer, contract={"wrong": True}
            )
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched checkpoint contract was accepted")
        Path(root, f"rank{rank}.ok").touch()
    except Exception:
        Path(root, f"rank{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.probe
@pytest.mark.parametrize(
    ("microbatches", "checkpoint", "split"),
    # torch 2.10 rejects ``n_microbatches < num_stages`` in
    # ``PipelineScheduleSingle.__init__``, which Schedule1F1B *and* ScheduleGPipe
    # both inherit, so a 2-stage pipeline cannot run microbatches=1 at all.
    # microbatches=2 is the smallest legal step and issues the GPipe action order;
    # 3 is the smallest count where 1F1B actually overlaps.
    [(2, True, 2), (3, True, 2), (4, False, 3)],
)
def test_pp_output_gradients_update_and_resume(
    tmp_path, microbatches, checkpoint, split
):
    ctx = get_context("spawn")
    processes = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                str(tmp_path / "init"),
                str(tmp_path),
                microbatches,
                checkpoint,
                split,
            ),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=55)
        errors = "\n".join(path.read_text() for path in tmp_path.glob("*.error"))
        assert all(process.exitcode == 0 for process in processes), errors
        assert len(list(tmp_path.glob("*.ok"))) == 2
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_owned_stage_has_only_local_parameters():
    model, network = _model()
    inputs, _ = _batch(1)
    _, _, rope = stage_examples(model, inputs)
    remote_ids = {
        id(parameter) for block in model.blocks[2:] for parameter in block.parameters()
    }
    remote_ids.update(
        id(parameter)
        for lora in network.unet_loras[4:]
        for parameter in lora.parameters()
    )
    stage = AnimaOwnedStage(model, network, rank=0, split=2, rope=rope)
    assert not remote_ids.intersection(
        id(parameter) for parameter in stage.parameters()
    )
    assert set(stage.model._modules) == {
        "blocks",
        "x_embedder",
        "pos_embedder",
        "t_embedder",
        "t_embedding_norm",
    }
    assert not hasattr(stage.model, "final_layer")


def test_merged_adapter_uses_formal_inference_loader(tmp_path):
    from types import SimpleNamespace
    from networks.lora_anima.factory import create_network_from_weights
    from scripts.experiments.anima_pipeline_bench import export_adapter

    full, network = _model(checkpoint=False)
    inputs, _ = _batch(1)
    with torch.no_grad():
        expected = full(*inputs[:3], padding_mask=inputs[3])
    merged = {}
    for rank in range(2):
        model, adapters = _model(checkpoint=False)
        _, _, rope = stage_examples(model, inputs)
        stage = AnimaOwnedStage(model, adapters, rank=rank, split=2, rope=rope)
        state = stage.adapter_state()
        assert not set(state).intersection(merged)
        merged.update(state)
    assert set(merged) == set(network.state_dict())
    path = tmp_path / "adapter.safetensors"
    export_adapter(path, merged, SimpleNamespace(rank_dim=4, alpha=4.0, mode="pp"), 1)
    # Remove the original monkey patch before exercising the production factory.
    for lora in network.unet_loras:
        lora.org_module_ref[0].forward = lora.org_forward
    loaded, state = create_network_from_weights(
        1.0, str(path), None, [], full, for_inference=True
    )
    loaded.apply_to([], full, apply_text_encoder=False, apply_unet=True)
    info = loaded.load_state_dict(state, strict=False)
    assert not info.missing_keys and not info.unexpected_keys
    with torch.no_grad():
        actual = full(*inputs[:3], padding_mask=inputs[3])
    # The public export intentionally stores BF16 adapter tensors.
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize(
    "failure", ["compiled", "attention", "base", "extra_adapter", "dropout"]
)
def test_stage_rejects_unsupported_contract(failure):
    model, network = _model()
    inputs, _ = _batch(1)
    _, _, rope = stage_examples(model, inputs)
    if failure == "compiled":
        model._native_flatten = True
    elif failure == "attention":
        model.attn_mode = "flex"
    elif failure == "base":
        next(model.parameters()).requires_grad_(True)
    elif failure == "extra_adapter":
        network.unet_loras = network.unet_loras[:-1]
    else:
        network.unet_loras[0].rank_dropout = 0.1
    with pytest.raises(ValueError, match="PP probe"):
        AnimaOwnedStage(model, network, rank=0, split=2, rope=rope)


def test_stage_rejects_equal_token_count_different_geometry():
    model, network = _model()
    inputs, _ = _batch(1)
    _, _, rope = stage_examples(model, inputs)
    stage = AnimaOwnedStage(model, network, rank=0, split=2, rope=rope)
    stage.input_shapes = tuple(tuple(value.shape) for value in inputs)
    changed = (
        inputs[0].reshape(1, 4, 1, 2, 8),
        *inputs[1:3],
        inputs[3].reshape(1, 1, 2, 8),
    )
    with pytest.raises(ValueError, match="fixed-shape"):
        stage(*changed)


def test_optional_hardware_snapshot_does_not_abort_results(monkeypatch):
    from scripts.experiments import anima_pipeline_bench

    def failed(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(anima_pipeline_bench.subprocess, "run", failed)
    assert "unavailable: OSError" in anima_pipeline_bench.hardware_snapshot()
