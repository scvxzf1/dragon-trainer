from __future__ import annotations

import pytest
import torch

from library.training.flow_step import mix_latents, unit_weighting, velocity_target


def test_flow_step_shared_4d_math() -> None:
    latents = torch.ones(2, 4, 3, 5)
    noise = torch.full_like(latents, 3.0)
    sigmas = torch.tensor([0.25, 0.75])

    mixed = mix_latents(latents, noise, sigmas)
    assert torch.equal(mixed[0], torch.full_like(mixed[0], 1.5))
    assert torch.equal(mixed[1], torch.full_like(mixed[1], 2.5))
    assert torch.equal(velocity_target(noise, latents), torch.full_like(latents, 2.0))
    assert unit_weighting(sigmas, latents).shape == (2, 1, 1, 1)


@pytest.mark.parametrize(
    "latents, noise, sigmas",
    [
        (torch.ones(2, 4, 3), torch.ones(2, 4, 3), torch.ones(2)),
        (torch.ones(2, 4, 3, 5), torch.ones(1, 4, 3, 5), torch.ones(2)),
        (torch.ones(2, 4, 3, 5), torch.ones(2, 4, 3, 5), torch.ones(1)),
    ],
)
def test_flow_step_rejects_invalid_shapes(latents, noise, sigmas) -> None:
    with pytest.raises(ValueError, match="flow"):
        mix_latents(latents, noise, sigmas)
