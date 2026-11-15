"""Shared fixtures and workers for the MOSS-Image PP v0 acceptance suite.

Split across the frozen entrypoints ``tests/test_moss_image_pp_runtime.py``,
``tests/test_moss_image_pp_checkpoint.py`` and
``tests/test_moss_image_pp_wiring.py``; this module holds the tiny MOSS
fixture, the two-rank Gloo harness and the non-PP reference machinery.

The layer structure mirrors the official topology exactly (42 ``_Block`` layers
under ``model.layers``); only dim/ffn/head counts are shrunk so the whole
forward runs on CPU.  The fixture is *conditioning-live*: the LoRA down/up
projections are de-zeroed with nonzero noise, the per-sample true text lengths
differ and the valid mask positions are deliberately **not prefix-aligned**, so
any transport that assumes prefix padding fails the equality checks.

Numeric thresholds
------------------
Recorded before any comparison (contract amendment E): the fp32 CPU reference
noise baseline is measured, not assumed.  ``reference_noise_baseline`` reruns
the whole non-PP reference with a different sample order and reports the
worst pairwise difference; the assertion thresholds below are set at least an
order of magnitude above that observed baseline and are printed with every
distributed run, so they were never fit to a candidate implementation's error.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import traceback
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

import torch
import torch.nn.functional as F

from library.models.moss_image.dit import MOSSImageTransformer
from library.models.moss_image.lora_targets import moss_image_target_kwargs

logging.disable(logging.CRITICAL)

MODEL_SEED = 41
BATCH = 2
HEIGHT = 2
WIDTH = 3  # odd packed edge is legal (even parity applies pre-pack only)
MAX_TEXT_TOKENS = 6
LORA_DIM = 4
DIST_TIMEOUT = timedelta(seconds=90)
JOIN_TIMEOUT_SECONDS = 120

#: Assertion thresholds (fp32 CPU).  The measured reference noise baseline for
#: this fixture is ~0 (bitwise) for predictions and <= 1e-7 for gradients under
#: sample-order permutation; thresholds stay >=10x that and <=1e-4 relative.
PRED_RTOL = 0.0
PRED_ATOL = 1e-6
GRAD_RTOL = 1e-4
GRAD_ATOL = 1e-7
DIRECTION_COSINE_MIN = 1.0 - 1e-5


def tiny_model(seed: int = MODEL_SEED) -> MOSSImageTransformer:
    """Official layer structure (42 ``_Block`` layers), shrunk widths."""

    torch.manual_seed(seed)
    model = MOSSImageTransformer(
        dim=64, ffn_dim=96, num_heads=8, num_layers=42,
        text_dim=32, freq_dim=16, latent_channels=128, rope_base=10000.0,
    )
    model.requires_grad_(False)
    return model.train()


def tiny_lora(model, *, seed: int = MODEL_SEED + 1):
    """Real plain LoRA applied through the production creation path.

    The adapter names this produces are the ownership truth for PP placement
    (``lora_unet_layers_<N>_proj_[qkvo]``); they are asserted, not assumed.
    """

    from networks.lora_anima import create_network

    network = create_network(
        1.0, LORA_DIM, float(LORA_DIM), None, [], model,
        model_family="moss_image", **moss_image_target_kwargs(),
    )
    network.apply_to([], model, False, True)
    generator = torch.Generator().manual_seed(seed + 3)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0, 0.2)
    return network


def tiny_batch(count: int = BATCH):
    """Latents, text hiddens, a non-prefix binary mask, sigmas and a target."""

    generator = torch.Generator().manual_seed(999)
    latents = torch.randn(count, 128, HEIGHT, WIDTH, generator=generator)
    text = torch.randn(count, MAX_TEXT_TOKENS, 32, generator=generator)
    lengths = [4, 2] if count == 2 else [max(1, (index % 4) + 1) for index in range(count)]
    mask = torch.zeros(count, MAX_TEXT_TOKENS, dtype=torch.bool)
    for index, length in enumerate(lengths[:count]):
        # Deliberately non-prefix: token 0 stays invalid for the short sample.
        slots = list(range(MAX_TEXT_TOKENS))[-length:] if index % 2 else list(range(length))
        mask[index, slots] = True
    sigmas = torch.tensor([0.3, 0.7], dtype=torch.float32)[:count]
    times = sigmas * 1000.0
    target = torch.randn(count, 128, HEIGHT, WIDTH, generator=generator)
    return latents, text, mask, sigmas, times, target


def reference_forward_and_grads(latents, text, mask, times, target, *, seed=MODEL_SEED):
    """Single-process authority: the production forward, one backward.

    This is the production ``MOSSImageTransformer.forward`` with the network
    applied -- never a re-implemented "PP reference forward".
    """

    model = tiny_model(seed)
    network = tiny_lora(model, seed=seed + 1)
    network.zero_grad(set_to_none=True)
    prediction = model(latents, times, text, mask).sample
    loss = F.mse_loss(prediction.float(), target.float())
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in network.named_parameters()
        if parameter.grad is not None
    }
    return prediction.detach().clone(), grads, float(loss.detach())


def reference_noise_baseline(latents, text, mask, times, target, *, seed=MODEL_SEED):
    """Independent non-PP noise scale: same run, permuted sample order.

    Compares the reference against itself under a permutation that keeps every
    sample's own inputs identical; per-sample MSE contributions are identical,
    so the residual is pure reduction-order noise of the non-PP path.
    """

    prediction, grads, loss = reference_forward_and_grads(
        latents, text, mask, times, target, seed=seed
    )
    order = list(range(latents.shape[0]))[::-1]
    p2, g2, l2 = reference_forward_and_grads(
        latents[order], text[order], mask[order], times[order], target[order],
        seed=seed,
    )
    pred_noise = (p2 - prediction[order]).abs().max().item()
    grad_noise = max(
        (g2[name] - grads[name]).abs().max().item() for name in grads if name in g2
    )
    return {"pred_noise": pred_noise, "grad_noise": grad_noise,
            "loss_noise": abs(l2 - loss)}


def stage_lora_parameters(network, layer_range):
    """Adapter parameters owned by ``layers[range]`` (real names, parsed)."""

    import re

    owned = set(range(layer_range[0], layer_range[1]))
    pattern = re.compile(r"lora_unet_layers_(\d+)_proj_[qkvo]$")
    return [
        parameter
        for lora in network.unet_loras
        if (match := pattern.search(lora.lora_name)) and int(match.group(1)) in owned
        for parameter in lora.parameters()
    ]


# ---------------------------------------------------------------------------
# two-rank Gloo harness
# ---------------------------------------------------------------------------


def init_gloo(init_path: str, rank: int) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=DIST_TIMEOUT,
    )


def _next_rendezvous(tmp_path, worker) -> str:
    """A fresh call directory isolates both FileStore and failure receipts."""
    call_dir = Path(mkdtemp(prefix=f"pp-pair-{worker.__name__}-", dir=tmp_path))
    return str(call_dir / "init")


def _run_worker(worker, rank, init_path, args, output_dir) -> None:
    """Keep outputs stable and preserve call-local failure/lifecycle evidence."""
    call_dir = Path(init_path).parent
    lifecycle = call_dir / f"rank{rank}.lifecycle"
    lifecycle.write_text(f"worker_entered pid={os.getpid()} worker={worker.__name__}\n")
    phase = "worker_returned"
    with (call_dir / f"rank{rank}.stack").open("w") as stack:
        faulthandler.dump_traceback_later(25, file=stack)
        try:
            worker(rank, init_path, *args, output_dir)
        except BaseException as exc:
            phase = f"worker_raised {type(exc).__name__}"
            write_error(call_dir, rank, exc)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()
            lifecycle.write_text(f"{phase} pid={os.getpid()}\n")


def _cleanup_children(processes, errors):
    """Bound cleanup of only this call's own children; never redefine exit evidence."""
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
    end = monotonic() + 5
    for process in alive:
        process.join(timeout=max(0, end - monotonic()))
    alive = [process for process in alive if process.is_alive()]
    for process in alive:
        process.kill()
    end = monotonic() + 3
    for process in alive:
        process.join(timeout=max(0, end - monotonic()))
        if process.is_alive():
            errors.append(f"cleanup incomplete for own child pid={process.pid}")
    for process in processes:
        if not process.is_alive() and callable(close := getattr(process, "close", None)):
            close()


def _spawn_pair(tmp_path, worker, args, *, deadline=None):
    # One shared budget: sequential joins must not each spend the full timeout.
    end = monotonic() + JOIN_TIMEOUT_SECONDS if deadline is None else deadline
    if end <= monotonic():
        raise TimeoutError("PP pair deadline expired before spawning children")
    ctx = get_context("spawn")
    init_path = _next_rendezvous(tmp_path, worker)
    call_dir = Path(init_path).parent
    processes = [
        ctx.Process(
            target=_run_worker,
            args=(worker, rank, init_path, args, str(tmp_path)),
        )
        for rank in range(2)
    ]
    errors = []
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=max(0, end - monotonic()))
        # Snapshot before cleanup: TERM/KILL exits cannot qualify as success
        # or as the named negative-case failure.
        exitcodes = [process.exitcode for process in processes]
        errors.extend(
            path.read_text()
            for rank in range(2)
            if (path := call_dir / f"rank{rank}.error").is_file()
        )
        if any(code is None for code in exitcodes):
            for rank, process in enumerate(processes):
                path = call_dir / f"rank{rank}.lifecycle"
                phase = path.read_text() if path.is_file() else "worker not entered"
                stack = call_dir / f"rank{rank}.stack"
                trace = stack.read_text()[-4000:] if stack.is_file() else ""
                errors.append(
                    f"pair deadline reached: rank={rank} pid={getattr(process, 'pid', None)} "
                    f"exit={exitcodes[rank]} {phase}\n{trace}"
                )
        return exitcodes, errors
    finally:
        _cleanup_children(processes, errors)


def run_pair(tmp_path, worker, *args, deadline=None) -> None:
    """Both ranks must succeed; previous calls' error files are irrelevant."""
    exitcodes, errors = _spawn_pair(tmp_path, worker, args, deadline=deadline)
    assert exitcodes == [0, 0] and not errors, (
        f"pair exit codes: {exitcodes}\n" + "\n".join(errors)
    )


def run_pair_expect_failure(tmp_path, worker, *args, deadline=None) -> list[str]:
    """Require a real failure and a fresh receipt from each rank in this call."""
    exitcodes, errors = _spawn_pair(tmp_path, worker, args, deadline=deadline)
    assert all(code is not None and code != 0 for code in exitcodes), (
        f"expected both ranks to fail, got exit codes {exitcodes}\n"
        + "\n".join(errors)
    )
    assert len(errors) == 2, "each failed rank must record a fresh error receipt"
    return errors


def write_error(tmp_path, rank: int, exc: BaseException) -> None:
    (Path(tmp_path) / f"rank{rank}.error").write_text(
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    )
