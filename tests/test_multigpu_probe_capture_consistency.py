"""Capture-point consistency: 1 worker and 2 workers must see identical bytes (t28).

t27 reported ``input_divergence`` at ``noise_per_sample``: the same slot carried a
different noise, latent and text on the dual-worker side, and those tensors matched
nothing on disk. Before that can be called a product defect, the *capture point* has
to be proven identical — otherwise the probe is measuring its own bookkeeping.

This test settles it on CPU with the real code paths, no GPU and no model:

* the real :meth:`noise_fixture.NoiseFixture.apply` injection point,
* the real :func:`library.runtime.noise.get_noisy_model_input_and_timesteps`,
* the probe's own hashing/raw-summary helpers,
* under a real two-process Gloo group with ``accelerate``-style rank semantics.

For every slot it asserts that the injected noise, the latent and the text have
byte-identical ``head8``/``sum``/hash whether one worker or two workers ran, and
that nothing in the chain mutates the latent in place.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bench.multigpu_reference import determinism_probe as probe
from library.runtime.noise import get_noisy_model_input_and_timesteps
from library.training.multigpu import noise_fixture

LATENT_SHAPE = (16, 4, 3)  # (C, H, W); 4*3 = 12 elements per channel
CHANNELS, HEIGHT, WIDTH = LATENT_SHAPE
NUMEL = CHANNELS * HEIGHT * WIDTH


class _Scheduler:
    config = SimpleNamespace(num_train_timesteps=1000)
    timesteps = torch.arange(999, -1, -1).float()


def _args():
    return SimpleNamespace(
        timestep_sampling="uniform", sigmoid_scale=1.0, sigmoid_bias=0.0,
        discrete_flow_shift=1.0, weighting_scheme="none", logit_mean=0.0,
        logit_std=1.0, mode_scale=1.29, t_min=None, t_max=None, ip_noise_gamma=None,
    )


def _write_cache(root: Path, count: int) -> dict[str, Path]:
    """A tiny stand-in latent cache with one npz per sample."""
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for index in range(count):
        generator = torch.Generator().manual_seed(1000 + index)
        latents = torch.randn((CHANNELS, HEIGHT, WIDTH), generator=generator)
        stem = f"sample{index:02d}_0896x1200"
        path = root / f"{stem}_anima.npz"
        np.savez(path, latents=latents.numpy(), original_size=np.array([1200, 896]),
                 crop_ltrb=np.array([0, 0, 896, 1200]))
        (root / f"{stem}.txt").write_text(f"caption {index}\n", encoding="utf-8")
        paths[stem] = path
    return paths


def _write_fixture(path: Path, count: int) -> None:
    rows = []
    for index in range(count):
        rows.append(
            {
                "sample_id": f"s{index:08d}",
                "timestep": 0.5,
                # A constant per-slot value, exactly like the real t13 fixture.
                "noise": [0.001 * (index + 1)] * NUMEL,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _capture(cache_dir: str, fixture_path: str, *, rank: int, world: int, consumed: int):
    """One micro-batch through the real chain; returns what the probe records."""
    os.environ[noise_fixture.ENV] = fixture_path
    os.environ[noise_fixture.ENV_IDS] = noise_fixture.IDS_CURSOR
    noise_fixture.reset_cache()

    fixture = noise_fixture.active()
    ids = noise_fixture.sample_ids_from_position(
        local_batch_index=consumed, rank=rank, world_size=world, batch_size=1
    )
    index = probe.LatentIndex(cache_dir).build()
    slot = ids[0]
    row = int(slot[1:])
    stems = sorted(path.name[: -len("_anima.npz")] for path in Path(cache_dir).glob("*_anima.npz"))
    latents = torch.from_numpy(np.load(Path(cache_dir) / f"{stems[row]}_anima.npz")["latents"])
    text = torch.full((1, 3), float(row + 1))

    batch = {"latents": latents.unsqueeze(0), "te": text}
    before = probe._raw_summary(batch["latents"][0])

    noise = torch.randn_like(batch["latents"])
    pinned, sigmas = fixture.apply(
        noise,
        batch["latents"],
        batch,
        mode=noise_fixture.ID_CURSOR if hasattr(noise_fixture, "ID_CURSOR") else "cursor",
        local_batch_index=consumed,
        rank=rank,
        world_size=world,
    )
    noisy, timesteps, sampler_sigmas = get_noisy_model_input_and_timesteps(
        _args(), _Scheduler(), batch["latents"], pinned, torch.device("cpu"),
        torch.float32, sigma_override=sigmas,
    )
    after = probe._raw_summary(batch["latents"][0])

    inputs = probe._inputs_fingerprint(
        batch, consumed, rank, world,
        noise_fixture,
        injected={
            "slots": ids,
            "noise": [probe.tensor_fingerprint(item) for item in pinned],
            "noise_raw": [probe._raw_summary(item) for item in pinned],
            "sigmas": [round(float(value), 9) for value in sigmas.reshape(-1)],
        },
        cache_index=index,
    )
    return {
        "slot": slot,
        "noise_raw": inputs["noise_raw"],
        "latent_raw": inputs["latent_raw"],
        "text_raw": inputs["text_raw"],
        "noise_hash": inputs["noise_per_sample_hash"],
        "latent_hash": inputs["latent_per_sample"],
        "text_hash": inputs["text_per_sample"],
        "timestep": inputs["timestep_injected"],
        "provenance": inputs["latent_provenance"],
        "latent_before": before,
        "latent_after": after,
        "noisy_sum": round(float(noisy.float().sum()), 6),
        "injected_sigmas": [round(float(value), 9) for value in sampler_sigmas.reshape(-1)],
    }


def _gloo_worker(rank, world_size, init_file, cache_dir, fixture_path, out_dir):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        captured = _capture(cache_dir, fixture_path, rank=rank, world=world_size, consumed=0)
    finally:
        dist.destroy_process_group()
    # Each rank writes its own file: passing a shared path would make the last
    # rank overwrite the first, which is exactly the sort of capture-side mistake
    # this test exists to rule out.
    Path(out_dir, f"out-{world_size}-{rank}.json").write_text(
        json.dumps(captured), encoding="utf-8"
    )


def _run_world(world_size: int, tmp: Path, cache_dir: Path, fixture: Path) -> list[dict]:
    import torch.multiprocessing as mp

    init_file = tmp / f"init{world_size}"
    init_file.write_text("", encoding="utf-8")
    if world_size > 1:
        mp.spawn(
            _gloo_worker,
            args=(world_size, str(init_file), str(cache_dir), str(fixture), str(tmp)),
            nprocs=world_size,
            join=True,
        )
    else:
        _gloo_worker(0, 1, str(init_file), str(cache_dir), str(fixture), str(tmp))
    paths = [tmp / f"out-{world_size}-{rank}.json" for rank in range(world_size)]
    return [json.loads(path.read_text()) for path in paths if path.exists()]


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    monkeypatch.delenv(noise_fixture.ENV, raising=False)
    monkeypatch.delenv(probe.ENV_LATENT_CACHE, raising=False)
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, count=4)
    fixture = tmp_path / "fixture.jsonl"
    _write_fixture(fixture, count=4)
    yield tmp_path, cache_dir, fixture
    noise_fixture.reset_cache()
    os.environ.pop(noise_fixture.ENV, None)
    os.environ.pop(noise_fixture.ENV_IDS, None)


def test_one_and_two_workers_capture_byte_identical_values(harness):
    tmp, cache_dir, fixture = harness
    single = _run_world(1, tmp, cache_dir, fixture)
    dual = _run_world(2, tmp, cache_dir, fixture)

    assert len(single) == 1 and len(dual) == 2
    by_slot = {entry["slot"]: entry for entry in single + dual}
    assert set(by_slot) == {"s00000000", "s00000001"}

    # The two workers really did take different samples...
    assert dual[0]["slot"] != dual[1]["slot"]
    # ...while each slot's content is byte-identical to the single-worker run.
    for world in (1, 2):
        for entry in single + dual:
            other = by_slot[entry["slot"]]
            assert entry["noise_raw"] == other["noise_raw"]
            assert entry["latent_raw"] == other["latent_raw"]
            assert entry["text_raw"] == other["text_raw"]
            assert entry["noise_hash"] == other["noise_hash"]
            assert entry["latent_hash"] == other["latent_hash"]
            assert entry["timestep"] == other["timestep"] == [0.5]


def test_the_capture_point_sees_the_pinned_noise_and_a_clean_latent(harness):
    tmp, cache_dir, fixture = harness
    single = _run_world(1, tmp, cache_dir, fixture)
    for entry in single:
        # 1. The noise captured at the injection point is the fixture value, not a
        #    derived tensor: a constant 0.001*(slot+1) fills every element.
        row = int(entry["slot"][1:])
        head, total = entry["noise_raw"][0]["head8"], entry["noise_raw"][0]["sum"]
        assert head == pytest.approx([0.001 * (row + 1)] * 8)
        assert total == pytest.approx(0.001 * (row + 1) * NUMEL, rel=1e-6)
        assert entry["noise_raw"][0]["shape"] == list(LATENT_SHAPE)
        # 2. The latent is read unmodified: before == after, and the provenance
        #    index resolves it to the very npz that was loaded.
        assert entry["latent_before"] == entry["latent_after"]
        provenance = entry["provenance"][0]
        assert provenance is not None, "the latent must resolve to a cache file"
        assert provenance["stem"] == f"sample{row:02d}_0896x1200"
        assert provenance["npz_path"].endswith(f"sample{row:02d}_0896x1200_anima.npz")
        assert provenance["caption_path"].endswith(f"sample{row:02d}_0896x1200.txt")


def test_the_index_reports_a_foreign_latent_as_unmatched(harness, tmp_path):
    """The t27 symptom: a latent that no cache file explains."""
    tmp, cache_dir, fixture = harness
    index = probe.LatentIndex(cache_dir).build()
    foreign = probe.tensor_fingerprint(torch.full(LATENT_SHAPE, 0.123))
    assert index.resolve(foreign) is None

    records = [
        {
            "schema": probe.SCHEMA, "global_step": 0, "rank": 0,
            "slots": ["s00000000"], "latent_per_sample": [foreign],
        }
    ]
    report = probe.provenance_report(records, index)
    assert report["summary"] == {"matched": 0, "unmatched": 1}
    assert report["steps"]["0"][0]["rows"][0]["verdict"] == "not_from_this_cache"


def test_the_record_states_its_capture_points_and_fixture_digest(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.jsonl"
    _write_fixture(fixture, count=2)
    monkeypatch.setenv(noise_fixture.ENV, str(fixture))
    provenance = probe._fixture_provenance()
    import hashlib

    assert provenance["sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert provenance["ids_mode"] == "batch"
    assert "NoiseFixture.apply" in probe.CAPTURE_POINTS["noise"]
    assert "none:" in probe.CAPTURE_POINTS["in_place_noise_injection"]


def test_a_latent_is_never_mistaken_for_text():
    """``"latents"`` contains ``"te"``; a substring match made the two collide.

    In the t27 record ``text_per_sample`` therefore carried the latent's hash, so
    "the text differs too" was the latent difference counted twice.
    """
    assert probe.is_text_key("te")
    assert probe.is_text_key("text_embeddings")
    assert probe.is_text_key("crossattn")
    assert probe.is_text_key("qwen3_text")
    for key in ("latents", "latent", "noise", "alpha_masks", "cond_latents", "ip_features"):
        assert not probe.is_text_key(key), key
