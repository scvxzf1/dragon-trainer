"""Krea-2 PP probe cut: the two halves must reproduce the monolith exactly."""

from __future__ import annotations

import torch

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT
from library.models.krea2_raw.pp_stages import (
    krea2_forward_first,
    krea2_forward_last,
    krea2_token_cap,
)

LAYERS = 4
TXT_TOKENS = 5
IMG_TOKENS = 4  # 2x2 patch grid


def _tiny_model(seed: int = 7) -> SingleStreamDiT:
    torch.manual_seed(seed)
    config = SingleMMDiTConfig(
        features=128,
        tdim=32,
        txtdim=64,
        heads=4,
        multiplier=2,
        layers=LAYERS,
        patch=2,
        channels=16,
        kvheads=2,
        txtlayers=2,
        txtheads=4,
        txtkvheads=4,
    )
    model = SingleStreamDiT(config)
    model.eval()
    return model


def _inputs(model: SingleStreamDiT, batch: int = 2, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    total = TXT_TOKENS + IMG_TOKENS
    img = torch.randn(
        batch, IMG_TOKENS, model.PATCH_SIZE**2 * model.LATENT_CHANNELS, generator=generator
    )
    context = torch.randn(
        batch, TXT_TOKENS, model.config.txtlayers, model.config.txtdim, generator=generator
    )
    t = torch.rand(batch, generator=generator)
    pos = torch.randn(batch, total, 3, generator=generator)
    mask = torch.ones(batch, total, dtype=torch.bool)
    return img, context, t, pos, mask


def test_token_cap_matches_the_monolith_padding() -> None:
    assert krea2_token_cap(5, 4) == 256  # 9 -> next multiple of 256
    assert krea2_token_cap(128, 128) == 256
    assert krea2_token_cap(200, 100) == 512  # 300 -> 512


def test_split_reproduces_the_monolith_bit_exactly() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)

    with torch.no_grad():
        full = model(img, context, t, pos, mask)
        state = krea2_forward_first(
            model,
            model.blocks[:2],
            img,
            timesteps=t,
            context=context,
            pos=pos,
            mask=mask,
        )
        split = krea2_forward_last(model, model.blocks[2:], state)

    assert torch.equal(full, split), (full - split).abs().max().item()


def test_split_is_independent_of_the_cut_point() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    with torch.no_grad():
        reference = model(img, context, t, pos, mask)
        for cut in range(0, LAYERS + 1):
            state = krea2_forward_first(
                model,
                model.blocks[:cut],
                img,
                timesteps=t,
                context=context,
                pos=pos,
                mask=mask,
            )
            split = krea2_forward_last(model, model.blocks[cut:], state)
            assert torch.equal(reference, split), f"cut={cut}"


def test_state_shapes_are_fixed_for_one_geometry() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    with torch.no_grad():
        state = krea2_forward_first(
            model,
            model.blocks[:1],
            img,
            timesteps=t,
            context=context,
            pos=pos,
            mask=mask,
        )
    cap = krea2_token_cap(TXT_TOKENS, IMG_TOKENS)
    assert state.hidden.shape == (2, cap, model.config.features)
    # tproj emits the six-way modulation vector: 6 * features per sample.
    assert state.tvec.shape == (2, 1, 6 * model.config.features)
    assert state.mask.shape == (2, cap)
    assert state.tembed.shape == (2, 1, model.config.features)
    assert (state.txtlen, state.imglen) == (TXT_TOKENS, IMG_TOKENS)


def test_gradients_flow_across_the_cut() -> None:
    model = _tiny_model()
    img, context, t, pos, mask = _inputs(model)
    img.requires_grad_(True)
    state = krea2_forward_first(
        model, model.blocks[:2], img, timesteps=t, context=context, pos=pos, mask=mask
    )
    out = krea2_forward_last(model, model.blocks[2:], state)
    out.sum().backward()
    assert img.grad is not None and img.grad.abs().sum() > 0
