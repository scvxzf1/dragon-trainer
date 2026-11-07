"""Per-sample noise/timestep fixture used by the single-vs-dual comparison (t25).

The L3 numeric gate needs the *same* ``(noise, timestep)`` for a given sample in
every run. These tests pin the three properties that make that trustworthy:
the schema is enforced, the injection really changes the draw, and the fixture is
**off** unless explicitly enabled.
"""

from __future__ import annotations

import types

import pytest
import torch

from library.runtime import noise as noise_utils
from library.training.multigpu import noise_fixture as fx


def _write(tmp_path, rows):
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_fixture_is_disabled_unless_the_env_var_is_set():
    assert fx.active({}) is None
    assert fx.active({fx.ENV: "   "}) is None


def test_valid_fixture_loads_per_sample_records(tmp_path):
    path = _write(
        tmp_path,
        [
            '{"sample_id": "a", "noise": [0.1, 0.2], "timestep": 0.5}',
            '{"sample_id": "b", "noise": [0.3, 0.4], "timestep": 0.25}',
        ],
    )
    fixture = fx.active({fx.ENV: str(path)})
    assert fixture is not None
    assert sorted(fixture.records) == ["a", "b"]
    assert fixture.records["b"]["timestep"] == 0.25


@pytest.mark.parametrize(
    "rows, code",
    [
        (['{"sample_id": "a", "noise": [], "timestep": 0.5}'], fx.CODE_FIXTURE_INVALID),
        (['{"sample_id": "", "noise": [0.1], "timestep": 0.5}'], fx.CODE_FIXTURE_INVALID),
        (['{"sample_id": "a", "noise": [0.1]}'], fx.CODE_FIXTURE_INVALID),
        (
            [
                '{"sample_id": "a", "noise": [0.1], "timestep": 0.5}',
                '{"sample_id": "a", "noise": [0.2], "timestep": 0.6}',
            ],
            fx.CODE_FIXTURE_INVALID,
        ),
        (['{"sample_id": "a", "noise": [0.1], "timestep": 1.5}'], fx.CODE_FIXTURE_INVALID),
    ],
)
def test_bad_fixture_is_a_named_error(tmp_path, rows, code):
    path = _write(tmp_path, rows)
    with pytest.raises(fx.NoiseFixtureError) as excinfo:
        fx.load_fixture(str(path))
    assert excinfo.value.code == code


def test_apply_pins_noise_and_sigma_per_sample(tmp_path):
    path = _write(
        tmp_path,
        [
            '{"sample_id": "a", "noise": [1.0, 0.0, 0.0, 0.0], "timestep": 0.25}',
            '{"sample_id": "b", "noise": [0.0, 2.0, 0.0, 0.0], "timestep": 0.75}',
        ],
    )
    fixture = fx.active({fx.ENV: str(path)})
    latents = torch.zeros(2, 1, 2, 2)
    drawn = torch.randn(2, 1, 2, 2)

    noise, sigmas = fixture.apply(drawn, latents, {"sample_ids": ["a", "b"]})

    assert torch.equal(noise[0].flatten(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(noise[1].flatten(), torch.tensor([0.0, 2.0, 0.0, 0.0]))
    assert sigmas.tolist() == [0.25, 0.75]
    # The original draw is not mutated in place.
    assert not torch.equal(drawn[0].flatten(), noise[0].flatten())


def test_apply_refuses_a_batch_it_cannot_pin(tmp_path):
    path = _write(tmp_path, ['{"sample_id": "a", "noise": [1.0], "timestep": 0.5}'])
    fixture = fx.active({fx.ENV: str(path)})
    latents = torch.zeros(1, 1, 1, 1)

    with pytest.raises(fx.NoiseFixtureError) as excinfo:
        fixture.apply(torch.zeros(1, 1, 1, 1), latents, {"sample_ids": ["zzz"]})
    assert excinfo.value.code == fx.CODE_FIXTURE_SAMPLE_MISSING

    with pytest.raises(fx.NoiseFixtureError) as excinfo:
        fixture.apply(torch.zeros(1, 1, 1, 1), latents, {"latents": torch.zeros(1)})
    assert excinfo.value.code == fx.CODE_FIXTURE_BATCH_IDS_MISSING


def test_sigma_override_drives_the_noising_arithmetic():
    """A pinned sigma must reach the noisy input, not just the returned timestep."""
    args = types.SimpleNamespace(
        timestep_sampling="uniform",
        sigmoid_bias=0.0,
        sigmoid_scale=1.0,
        t_min=None,
        t_max=None,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
    )
    scheduler = types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=1000))
    latents = torch.ones(2, 1, 2, 2)
    noise = torch.zeros(2, 1, 2, 2)
    override = torch.tensor([0.25, 0.75])

    noisy, timesteps, sigmas = noise_utils.get_noisy_model_input_and_timesteps(
        args, scheduler, latents, noise, torch.device("cpu"), torch.float32,
        sigma_override=override,
    )

    assert timesteps.tolist() == [0.25, 0.75]
    # (1 - sigma) * latents + sigma * 0, per sample.
    assert torch.allclose(noisy[0], torch.full_like(latents[0], 0.75))
    assert torch.allclose(noisy[1], torch.full_like(latents[1], 0.25))
    assert sigmas.flatten().tolist() == [0.25, 0.75]


def test_sample_id_resolution_uses_the_declared_keys():
    assert fx.batch_sample_ids({"sample_ids": ["a", "b"]}) == ["a", "b"]
    assert fx.batch_sample_ids({"image_names": "a.png"}) == ["a.png"]
    with pytest.raises(fx.NoiseFixtureError):
        fx.batch_sample_ids({"latents": torch.zeros(2)})


# ---------------------------------------------------------------------------
# 生产形状 batch：id 必须来自 Anima dataloader 真实产出的键（t25 复核项）
# ---------------------------------------------------------------------------


def _production_batch(*, debug_dataset=True, batch_size=2, bucket=("img/a.png", "img/b.png")):
    """Build a batch through the **production** batch assembler, not by hand.

    ``BaseDataset.__getitem__`` returns ``assemble_training_example(...)``; building
    the dict here means the id key this fixture resolves is the key the trainer
    really emits, and a drift in that key breaks this test instead of silently
    disabling the L3 fixture.
    """
    import types as _types

    from library.datasets.dataset_getitem_stack import assemble_training_example

    dataset = _types.SimpleNamespace(
        debug_dataset=debug_dataset,
        network_multiplier=1.0,
        batch_size=batch_size,
        _byg_roles=(),
    )
    return assemble_training_example(
        dataset=dataset,
        custom_attributes=[{} for _ in range(batch_size)],
        loss_weights=[1.0] * batch_size,
        text_encoder_outputs_list=[None] * batch_size,
        input_ids_list=[None] * batch_size,
        alpha_mask_list=[None] * batch_size,
        images=[None] * batch_size,
        latents_list=[torch.zeros(1, 2, 2) for _ in range(batch_size)],
        cond_latents_list=[None] * batch_size,
        qwen_edit_reference_latents=[None] * batch_size,
        captions=["a cat"] * batch_size,
        original_sizes_hw=[(8, 8)] * batch_size,
        crop_top_lefts=[(0, 0)] * batch_size,
        target_sizes_hw=[(8, 8)] * batch_size,
        flippeds=[False] * batch_size,
        inversion_runs_list=[None] * batch_size,
        ip_features_list=[None] * batch_size,
        ip_features_shuffled_list=[None] * batch_size,
        neg_crossattn_list=[None] * batch_size,
        neg_jaccard_list=[None] * batch_size,
        byg_tuple_list=[None] * batch_size,
        bucket=list(bucket),
        image_index=0,
    )


def test_real_batch_ids_come_from_image_keys(tmp_path):
    batch = _production_batch()
    assert "image_keys" in batch  # the key the Anima dataloader really produces
    assert fx.batch_sample_ids(batch) == ["img/a.png", "img/b.png"]

    path = _write(
        tmp_path,
        [
            '{"sample_id": "img/a.png", "noise": [1.0, 0.0, 0.0, 0.0], "timestep": 0.5}',
            '{"sample_id": "img/b.png", "noise": [0.0, 2.0, 0.0, 0.0], "timestep": 0.25}',
        ],
    )
    fixture = fx.active({fx.ENV: str(path)})
    noise, sigmas = fixture.apply(
        torch.randn_like(batch["latents"]), batch["latents"], batch
    )
    assert torch.equal(noise[0].flatten(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(noise[1].flatten(), torch.tensor([0.0, 2.0, 0.0, 0.0]))
    assert sigmas.tolist() == [0.5, 0.25]


def test_real_batch_missing_a_record_is_a_named_error(tmp_path):
    batch = _production_batch()
    path = _write(tmp_path, ['{"sample_id": "img/a.png", "noise": [1.0, 0.0, 0.0, 0.0], "timestep": 0.5}'])
    fixture = fx.active({fx.ENV: str(path)})
    with pytest.raises(fx.NoiseFixtureError) as excinfo:
        fixture.apply(torch.randn_like(batch["latents"]), batch["latents"], batch)
    assert excinfo.value.code == fx.CODE_FIXTURE_SAMPLE_MISSING
    assert "img/b.png" in str(excinfo.value)


def test_batch_without_ids_still_fails_closed(tmp_path):
    """Honest gate: ``image_keys`` only exists in the do-not-train debug mode.

    A real training batch carries no per-sample id, so the fixture must refuse
    instead of pinning nothing and reporting a green number.
    """
    batch = _production_batch(debug_dataset=False)
    assert "image_keys" not in batch
    path = _write(tmp_path, ['{"sample_id": "img/a.png", "noise": [1.0, 0.0, 0.0, 0.0], "timestep": 0.5}'])
    fixture = fx.active({fx.ENV: str(path)})
    with pytest.raises(fx.NoiseFixtureError) as excinfo:
        fixture.apply(torch.randn_like(batch["latents"]), batch["latents"], batch)
    assert excinfo.value.code == fx.CODE_FIXTURE_BATCH_IDS_MISSING


# ---------------------------------------------------------------------------
# 光标模式：真实训练 batch 没有 id 时，用已提交的 sampler 位置定标
# ---------------------------------------------------------------------------


def test_default_id_mode_is_the_batch_and_is_env_selectable():
    assert fx.ids_mode({}) == fx.IDS_BATCH
    assert fx.ids_mode({fx.ENV_IDS: "cursor"}) == fx.IDS_CURSOR
    with pytest.raises(fx.NoiseFixtureError) as excinfo:
        fx.ids_mode({fx.ENV_IDS: "guess"})
    assert excinfo.value.code == fx.CODE_FIXTURE_IDS_MODE


def test_cursor_ids_identify_the_same_global_slot_in_both_world_sizes():
    # single worker: local batch 3 == global batch 3
    single = fx.sample_ids_from_position(local_batch_index=3, rank=0, world_size=1, batch_size=2)
    # two workers: rank 1's local batch 1 is global batch 3
    dual = fx.sample_ids_from_position(local_batch_index=1, rank=1, world_size=2, batch_size=2)
    assert single == dual == ["s00000006", "s00000007"]


def test_cursor_mode_pins_the_draw_without_any_batch_id(tmp_path):
    batch = _production_batch(debug_dataset=False)  # a real training batch
    path = _write(
        tmp_path,
        [
            '{"sample_id": "s00000006", "noise": [1.0, 0.0, 0.0, 0.0], "timestep": 0.5}',
            '{"sample_id": "s00000007", "noise": [0.0, 2.0, 0.0, 0.0], "timestep": 0.25}',
        ],
    )
    fixture = fx.active({fx.ENV: str(path)})
    noise, sigmas = fixture.apply(
        torch.randn_like(batch["latents"]),
        batch["latents"],
        batch,
        mode=fx.IDS_CURSOR,
        local_batch_index=1,
        rank=1,
        world_size=2,
    )
    assert torch.equal(noise[0].flatten(), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(noise[1].flatten(), torch.tensor([0.0, 2.0, 0.0, 0.0]))
    assert sigmas.tolist() == [0.5, 0.25]


def test_cursor_mode_without_a_position_is_a_named_error(tmp_path):
    batch = _production_batch(debug_dataset=False)
    path = _write(tmp_path, ['{"sample_id": "s00000000", "noise": [1.0, 0.0, 0.0, 0.0], "timestep": 0.5}'])
    fixture = fx.active({fx.ENV: str(path)})
    with pytest.raises(fx.NoiseFixtureError) as excinfo:
        fixture.apply(
            torch.randn_like(batch["latents"]), batch["latents"], batch, mode=fx.IDS_CURSOR
        )
    assert excinfo.value.code == fx.CODE_FIXTURE_BATCH_IDS_MISSING


def test_fixture_stays_off_when_the_env_var_is_unset(monkeypatch):
    """Zero intervention: no env → no fixture object → the draw is untouched."""
    monkeypatch.delenv(fx.ENV, raising=False)
    monkeypatch.delenv(fx.ENV_IDS, raising=False)
    assert fx.active() is None
