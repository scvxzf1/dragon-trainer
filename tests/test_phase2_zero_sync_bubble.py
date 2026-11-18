from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import warnings

import pytest
import torch
from torch import nn

from library.models.qwen_image_2_1 import attention_backend as qwen_attn
from library.models.z_image import family as z_family
from library.training import loop as training_loop
from library.training import scaler_ownership, window_accounting
from library.training.batch_step import _prepare_latents


@contextmanager
def _capture_cuda_sync_warnings():
    prev = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            yield caught
        finally:
            torch.cuda.set_sync_debug_mode(prev)


def _count_sync_warnings(caught) -> int:
    return sum(
        1
        for item in caught
        if "synchronizing CUDA operation" in str(item.message)
    )


def test_backward_is_enqueued_before_loss_item_extraction(monkeypatch) -> None:
    events: list[str] = []

    class _TrackedLoss:
        dtype = torch.float32
        device = torch.device("cpu")
        requires_grad = True

        def detach(self):
            return self

        def item(self):
            events.append("loss_item")
            return 0.25

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 2))
            self.w.grad = torch.ones_like(self.w)

        def get_trainable_params(self):
            return [self.w]

    net = _Net()

    class _Acc:
        sync_gradients = True
        mixed_precision = "bf16"
        scaler = None
        num_processes = 1
        device = torch.device("cpu")

        @staticmethod
        def accumulate(_m):
            return nullcontext()

        @staticmethod
        def unwrap_model(m):
            return m

        @staticmethod
        def backward(_loss):
            events.append("backward")

        @staticmethod
        def clip_grad_norm_(_params, _max_norm):
            events.append("clip")

    class _Opt:
        param_groups = [{"params": [net.w]}]

        @staticmethod
        def step():
            events.append("opt_step")

        @staticmethod
        def zero_grad(set_to_none=True):
            del set_to_none

    trainer = SimpleNamespace(
        memory_probe=None,
        peak_probe=None,
        _cudagraph_mark_step=False,
        _state=SimpleNamespace(personalization_observer={}),
        on_step_start=lambda *_a, **_k: None,
        process_batch=lambda *_a, **_k: (events.append("forward") or _TrackedLoss()),
        run_after_backward=lambda *_a, **_k: events.append("after_backward_hook"),
    )
    state = SimpleNamespace(
        args=SimpleNamespace(max_grad_norm=1.0, log_every_n_steps=1, max_train_steps=2),
        accelerator=_Acc(),
        network=net,
        training_model=net,
        optimizer=_Opt(),
        lr_scheduler=SimpleNamespace(step=lambda: None),
        text_encoder=None,
        unet=None,
        train_ctx=object(),
        on_step_start_for_network=lambda *_a: None,
        profile_started=False,
        global_step=0,
        is_tracking=False,
        nonfinite_grad_skips=0,
    )
    monkeypatch.setattr(training_loop, "debug_finite_enabled", lambda _a: False)
    monkeypatch.setattr(training_loop, "synchronize_optimizer_gradients", lambda *_a: None)

    training_loop._run_microbatch(trainer, state, batch=object())
    assert events.index("forward") < events.index("backward")
    assert events.index("backward") < events.index("after_backward_hook")
    assert events.index("after_backward_hook") < events.index("loss_item")


def test_precomputed_total_norm_matches_torch_clip_grad_norm(monkeypatch) -> None:
    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.tensor([3.0, 4.0]))
            self.b = nn.Parameter(torch.tensor([12.0]))
            self.a.grad = torch.tensor([3.0, 4.0])
            self.b.grad = torch.tensor([12.0])

    net = _Net()
    calls = {"foreach_norm": 0}
    orig_foreach_norm = torch._foreach_norm

    def _counting_foreach_norm(*args, **kwargs):
        calls["foreach_norm"] += 1
        return orig_foreach_norm(*args, **kwargs)

    monkeypatch.setattr(torch, "_foreach_norm", _counting_foreach_norm)
    acc = SimpleNamespace(scaler=None, num_processes=1)
    fin = scaler_ownership.inspect_grad_ownership(acc, net)
    assert fin.bad is False
    assert fin.precomputed_total_norm is not None
    torch.testing.assert_close(fin.precomputed_total_norm, torch.tensor(13.0))
    assert calls["foreach_norm"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for sync_debug_mode check")
def test_prepare_latents_and_z_image_conditions_zero_cuda_sync() -> None:
    device = torch.device("cuda")
    acc = SimpleNamespace(
        device=device,
        print=lambda *_a: None,
        autocast=nullcontext,
    )
    trainer = SimpleNamespace(shift_scale_latents=lambda _args, lat: lat)
    args = SimpleNamespace(vae_batch_size=None)
    batch = {"latents": torch.randn(2, 16, 8, 8, dtype=torch.float16).pin_memory()}

    # Warm up CUDA context before entering strict sync-warn scope.
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)

    with _capture_cuda_sync_warnings() as caught_latents:
        lat = _prepare_latents(args, trainer, None, batch, acc, torch.float16)
    assert lat.device.type == "cuda"
    assert _count_sync_warnings(caught_latents) == 0

    ctx = SimpleNamespace(
        args=SimpleNamespace(),
        accelerator=acc,
        weight_dtype=torch.bfloat16,
        text_encoders=[],
        tokenize_strategy=None,
        text_encoding_strategy=None,
    )
    cpu_hiddens = torch.randn(2, 12, 32, dtype=torch.bfloat16)
    cpu_mask = torch.tensor([[1] * 7 + [0] * 5, [1] * 9 + [0] * 3], dtype=torch.bool)
    with _capture_cuda_sync_warnings() as caught_z:
        prompts = z_family._prepare_text_conditions(
            trainer, ctx, {}, (cpu_hiddens, cpu_mask), is_train=True
        )
    assert [p.shape for p in prompts] == [(7, 32), (9, 32)]
    assert all(p.device.type == "cuda" for p in prompts)
    assert _count_sync_warnings(caught_z) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for sync_debug_mode check")
def test_mid_window_microbatch_and_log_step_zero_cuda_sync(monkeypatch) -> None:
    device = torch.device("cuda")
    w = nn.Parameter(torch.ones(16, 16, device=device))

    class _Net(nn.Module):
        def __init__(self, param: nn.Parameter) -> None:
            super().__init__()
            self.w = param

        def get_trainable_params(self):
            return [self.w]

    net = _Net(w)
    x = torch.randn(4, 16, device=device)

    # Warm up CUDA matmul/backward allocator & cuBLAS handle first.
    (x @ w).square().mean().backward()
    w.grad = None
    torch.cuda.synchronize(device)

    class _MidWindowAcc:
        sync_gradients = False
        mixed_precision = "bf16"
        scaler = None
        num_processes = 1

        def __init__(self) -> None:
            self.device = device

        @staticmethod
        def accumulate(_m):
            return nullcontext()

        @staticmethod
        def unwrap_model(m):
            return m

        @staticmethod
        def backward(loss):
            loss.backward()

    trainer = SimpleNamespace(
        memory_probe=None,
        peak_probe=None,
        _cudagraph_mark_step=False,
        _state=SimpleNamespace(personalization_observer={}),
        on_step_start=lambda *_a, **_k: None,
        process_batch=lambda *_a, **_k: (x @ net.w).square().mean(),
        run_after_backward=lambda *_a, **_k: None,
        progress_sink=None,
    )
    state = SimpleNamespace(
        args=SimpleNamespace(max_grad_norm=1.0, log_every_n_steps=1, max_train_steps=10),
        accelerator=_MidWindowAcc(),
        network=net,
        training_model=net,
        optimizer=SimpleNamespace(step=lambda: None, zero_grad=lambda **_k: None),
        lr_scheduler=SimpleNamespace(step=lambda: None),
        text_encoder=None,
        unet=None,
        train_ctx=object(),
        on_step_start_for_network=lambda *_a: None,
        profile_started=False,
        global_step=0,
        is_tracking=True,
        nonfinite_grad_skips=0,
        loss_recorder=SimpleNamespace(moving_average=0.5),
        progress_bar=SimpleNamespace(set_postfix=lambda **_k: None),
        recent_step_seconds=[],
        train_dataset_group=None,
        stage_index=-1,
    )
    monkeypatch.setattr(training_loop, "debug_finite_enabled", lambda _a: False)

    window = window_accounting.WindowBuffer()
    with _capture_cuda_sync_warnings() as caught:
        loss = training_loop._run_microbatch(
            trainer, state, batch=object(), window=window
        )
        window.add_deferred(object(), loss, epoch=0, step=0)
        training_loop._log_step(
            trainer,
            state,
            loss=loss,
            step=0,
            epoch=0,
            keys_scaled=0,
            mean_norm=None,
            maximum_norm=None,
            max_mean_logs={},
        )
    assert _count_sync_warnings(caught) == 0
    tail_val = window.materialize_losses()
    assert tail_val is None
    assert len(window.losses) == 1 and window.losses[0] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Qwen mask sync test")
def test_qwen_flash_varlen_caches_mask_metadata_across_60_layers(monkeypatch) -> None:
    from networks import attention_dispatch

    device = torch.device("cuda")
    monkeypatch.setattr(
        attention_dispatch, "flash_attn_available_for_tensor", lambda _t: True
    )
    monkeypatch.setattr(
        attention_dispatch,
        "flash_attn_varlen_func",
        lambda q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0: q.clone(),
    )

    q = torch.randn(2, 16, 4, 32, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 12, 4, 32, device=device, dtype=torch.bfloat16)
    v = torch.randn(2, 12, 4, 32, device=device, dtype=torch.bfloat16)
    mask = torch.ones(2, 12, device=device, dtype=torch.bool)
    mask[1, 8:] = False
    torch.cuda.synchronize(device)

    with _capture_cuda_sync_warnings() as caught:
        for _ in range(60):
            out = qwen_attn._flash_varlen_attention(q, k, v, mask=mask)
    assert out.shape == q.shape
    # Layer 0 resolves mask lengths + valid_indices once (2 syncs: min/max + nonzero);
    # the remaining 59 layers incur ZERO synchronizing CUDA operations at all.
    assert _count_sync_warnings(caught) <= 2

    with _capture_cuda_sync_warnings() as caught_cached:
        for _ in range(60):
            _ = qwen_attn._flash_varlen_attention(q, k, v, mask=mask)
    assert _count_sync_warnings(caught_cached) == 0
