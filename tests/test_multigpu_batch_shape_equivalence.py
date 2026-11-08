"""Does one batch-2 forward equal two batch-1 forwards? (t26, CPU only)

t25 narrowed the cross-world divergence to the *loss itself*: the single-worker
run reported ``0.19995`` where the two dual-worker ranks reported ``0.25614`` and
``0.26525`` (mean ``0.26070``), with the same step-0 adapter hash, the same slots
and the same ``sigma=0.5``.

These two tests answer the two mechanisms that can be settled without a GPU, with
the **real** code paths and no toy reimplementation:

1. :func:`...losses.LossComposer.compose` — is the composed scalar invariant to the
   batch shape? (a per-sample ``[B]`` vector, then ``.mean()``; the weighting is
   per sample, ``loss_weights`` is 1.0 per sample)
2. :func:`...runtime.noise.get_noisy_model_input_and_timesteps` — does the noisy
   input / timestep / sigma for a sample depend on the batch it travels in?

Both are batch-shape invariant, which is the finding: the loss arithmetic cannot
produce a 23% gap, so the gap has to sit in the per-sample *content* (noise,
latent, text) that the two runs fed to the same slot — which is why the probe now
records per-sample content fingerprints and the actually-injected noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.runtime.noise import get_noisy_model_input_and_timesteps
from library.training.losses import LossComposer, LossContext


class _Scheduler:
    config = SimpleNamespace(num_train_timesteps=1000)
    timesteps = torch.arange(999, -1, -1).float()


def _args():
    return SimpleNamespace(
        timestep_sampling="uniform",
        sigmoid_scale=1.0,
        sigmoid_bias=0.0,
        discrete_flow_shift=1.0,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        t_min=None,
        t_max=None,
        ip_noise_gamma=None,
        loss_type="l2",
        masked_loss=False,
        foreground_loss_weight=1.0,
        mask_loss_normalize="none",
    )


def _ctx(args, pred, target, weighting, loss_weights):
    return LossContext(
        args=args,
        batch={},
        model_pred=pred,
        target=target,
        timesteps=torch.full((pred.shape[0],), 0.5),
        weighting=weighting,
        huber_c=None,
        loss_weights=loss_weights,
        network=None,
    )


def test_the_real_loss_composer_is_invariant_to_the_batch_shape():
    args = _args()
    composer = LossComposer(["flow_match"])
    torch.manual_seed(0)
    batch, channels, height, width = 2, 4, 3, 3
    prediction = torch.randn(batch, channels, height, width)
    target = torch.randn(batch, channels, height, width)
    weighting = torch.tensor([0.6, 0.9]).view(batch, 1, 1, 1)
    loss_weights = torch.ones(batch)

    together = composer.compose(
        _ctx(args, prediction, target, weighting, loss_weights)
    ).item()
    halves = [
        composer.compose(
            _ctx(
                args,
                prediction[index : index + 1],
                target[index : index + 1],
                weighting[index : index + 1],
                loss_weights[index : index + 1],
            )
        ).item()
        for index in range(batch)
    ]

    # Per-sample [B] vector, then the mean over B: the two orders agree to fp32.
    assert together == pytest.approx(sum(halves) / batch, rel=1e-6)


@pytest.mark.parametrize("sigma", [0.5, 0.25, 0.9])
def test_the_real_noise_path_is_invariant_to_the_batch_shape(sigma):
    args = _args()
    scheduler = _Scheduler()
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 2, 2)
    noise = torch.randn(2, 4, 2, 2)
    sigmas = torch.tensor([sigma, sigma])

    together = get_noisy_model_input_and_timesteps(
        args, scheduler, latents, noise, torch.device("cpu"), torch.float32,
        sigma_override=sigmas,
    )
    for index in range(2):
        alone = get_noisy_model_input_and_timesteps(
            args, scheduler, latents[index : index + 1], noise[index : index + 1],
            torch.device("cpu"), torch.float32,
            sigma_override=sigmas[index : index + 1],
        )
        assert torch.equal(together[0][index], alone[0][0])
        assert torch.equal(together[1][index], alone[1][0])
        assert torch.equal(together[2][index].reshape(-1), alone[2].reshape(-1))


def test_a_per_sample_sigma_mismatch_is_what_the_comparison_must_catch():
    """Two samples, same sigma, different noise: the losses differ a lot.

    The per-sample MSE of the flow target is wildly different per sample, so a
    slot that is fed the wrong sample shows up as a large loss gap — which is why
    the probe records per-sample content, not just the slot id.
    """
    args = _args()
    scheduler = _Scheduler()
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn(2, 4, 4, 4)
    sigmas = torch.tensor([0.5, 0.5])
    noisy, timesteps, _ = get_noisy_model_input_and_timesteps(
        args, scheduler, latents, noise, torch.device("cpu"), torch.float32,
        sigma_override=sigmas,
    )
    targets = noise - latents
    per_sample = (noisy - targets).pow(2).mean(dim=[1, 2, 3])
    assert per_sample[0].item() != pytest.approx(per_sample[1].item(), rel=1e-3)
