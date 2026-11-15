"""CPU/meta regression for master placement; this is not CUDA execution proof."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training.pp import PipelineRuntimeConfig
from library.training.pp.contract import validate_master_weight_dtype
from library.training.pp.errors import CODE_STAGE_CONTRACT, PipelineParallelError
from tests.moss_pp_test_support import (
    BATCH, HEIGHT, WIDTH, MAX_TEXT_TOKENS,
    init_gloo, run_pair_expect_failure, tiny_lora, tiny_model,
)


def test_wrong_master_device_is_refused_without_cuda():
    parameter = torch.nn.Parameter(torch.empty(2, device="meta"))
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_master_weight_dtype([parameter], mixed_precision="fp16", device="cpu")
    assert excinfo.value.code == CODE_STAGE_CONTRACT


def test_cpu_index_hint_and_legacy_dtype_only_call_remain_valid():
    parameter = torch.nn.Parameter(torch.ones(2))
    validate_master_weight_dtype([parameter], mixed_precision="fp16", device="cpu:0")
    validate_master_weight_dtype([parameter], mixed_precision="fp16")


def test_default_cuda_index_is_checked_as_metadata_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    parameter = SimpleNamespace(
        requires_grad=True, dtype=torch.float32, device=torch.device("cuda:1")
    )
    validate_master_weight_dtype([parameter], mixed_precision="fp16", device="cuda")
    parameter.device = torch.device("cuda:0")
    with pytest.raises(PipelineParallelError) as excinfo:
        validate_master_weight_dtype([parameter], mixed_precision="fp16", device="cuda")
    assert excinfo.value.code == CODE_STAGE_CONTRACT


@pytest.mark.parametrize("rank", [0, 1])
def test_cuda_fixture_places_network_before_builder(rank, monkeypatch):
    # Mock device movement and builder; do not allocate a CUDA tensor here.
    from tests import test_moss_image_pp_runtime as cuda_tests

    events = []
    device = torch.device(f"cuda:{rank}")

    class Network:
        def to(self, *, device):
            events.append(("move", device))
            return self

    network = Network()
    result = object()

    def build(model, **kwargs):
        assert events == [("move", device)]
        assert kwargs["network"] is network and kwargs["device"] == device
        assert callable(kwargs["optimizer_factory"])
        events.append(("build", device))
        return result

    monkeypatch.setattr(cuda_tests, "build_moss_pipeline_runtime", build)
    monkeypatch.setattr(
        torch.amp, "GradScaler",
        lambda _: SimpleNamespace(set_growth_interval=lambda interval: None),
    )
    assert cuda_tests._cuda_runtime(rank, device, object(), network, growth_interval=4) is result
    assert events == [("move", device), ("build", device)]


@pytest.mark.parametrize("losses, fails", [
    ([torch.tensor(1.0)], True),
    ([torch.tensor(1.0), torch.tensor(float("nan"))], True),
    ([torch.tensor(1.0), torch.tensor(2.0)], False),
])
def test_cuda_loss_record_guard_runs_before_optimizer(losses, fails, monkeypatch):
    from contextlib import nullcontext
    from tests import test_moss_image_pp_runtime as cuda_tests

    called = False

    def update():
        nonlocal called
        called = True
        return SimpleNamespace(skipped=False, scale=1.0)

    runtime = SimpleNamespace(
        is_last=True, zero_grad=lambda: None, optimizer_step=update,
        step=lambda *args, **kwargs: kwargs["losses"].extend(losses),
    )
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    if fails:
        with pytest.raises(AssertionError):
            cuda_tests._cuda_run_steps(runtime, None, None, None, None, None, 1)
    else:
        records = cuda_tests._cuda_run_steps(runtime, None, None, None, None, None, 1)
        assert records == [{"scale_used": 1.0, "losses": [1.0, 2.0]}]
    assert called is (not fails)


def test_device_only_move_keeps_real_base_half_and_lora_masters_fp32():
    model = tiny_model().to(dtype=torch.float16)
    network = tiny_lora(model)
    network.to(device="cpu")
    assert all(parameter.dtype == torch.float16 for parameter in model.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in network.parameters())
    validate_master_weight_dtype(network.parameters(), mixed_precision="fp16", device="cpu")


def _worker_misplaced_adapter(rank, init_path, output_dir):
    import torch.distributed as dist
    from library.training.pp import moss_runtime

    try:
        init_gloo(init_path, rank)
        model = tiny_model()
        network = tiny_lora(model)
        if rank == 1:
            next(
                module for module in network.unet_loras
                if module.lora_name.startswith("lora_unet_layers_21_")
            ).to(device="meta")
        factory_called = False

        def factory(stage):
            nonlocal factory_called
            factory_called = True
            return torch.optim.AdamW(list(stage.trainable_parameters()), lr=1e-3)

        def forbidden_policy(*args, **kwargs):
            raise AssertionError("must refuse before any policy collective")

        moss_runtime.assert_compute_policy_agreement = forbidden_policy
        try:
            moss_runtime.build_moss_pipeline_runtime(
                model,
                config=PipelineRuntimeConfig(
                    stages=2, microbatches=2, schedule="1f1b",
                    conditioning="local", mixed_precision="fp16",
                ),
                rank=rank, world_size=2, device=torch.device("cpu"),
                latent_shape=(BATCH, 128, HEIGHT, WIDTH), max_text_tokens=MAX_TEXT_TOKENS,
                dtype=torch.float32, network=network,
                scaler=torch.amp.GradScaler("cpu") if rank == 1 else None,
                optimizer_factory=factory,
            )
        except PipelineParallelError as exc:
            if rank == 1:
                assert factory_called is False
            Path(output_dir, f"rank{rank}.code").write_text(exc.code)
            raise
        raise AssertionError("misplaced adapter unexpectedly accepted")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_single_rank_misplaced_master_aborts_both_before_factory_or_policy(tmp_path):
    run_pair_expect_failure(tmp_path, _worker_misplaced_adapter)
    assert [Path(tmp_path, f"rank{rank}.code").read_text() for rank in range(2)] == [
        CODE_STAGE_CONTRACT, CODE_STAGE_CONTRACT,
    ]
