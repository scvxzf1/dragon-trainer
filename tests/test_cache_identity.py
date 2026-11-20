"""CPU-only contracts for model-identity cache invalidation.

Covers the shared primitives in library.cache_identity plus the two sidecar
paths that consume them: VAE latent .npz caches and text-encoder .safetensors
caches. The Anima vocab-pack producer/consumer round trip lives in
tests/test_anima_vocab_pack_cache_identity.py.
"""

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library import cache_identity as identity
from library.anima.strategy import (
    AnimaLatentsCachingStrategy,
    AnimaTextEncoderOutputsCachingStrategy,
)
from library.preprocess.latents import _latent_cached, _save_batch


# --- primitives -----------------------------------------------------------


def test_path_identity_tracks_file_and_directory_changes(tmp_path):
    weight = tmp_path / "te.safetensors"
    weight.write_bytes(b"a" * 16)
    before = identity.path_identity(weight)

    weight.write_bytes(b"b" * 32)
    after = identity.path_identity(weight)

    assert before["exists"] and after["exists"]
    assert (before["size"], before["mtime_ns"]) != (after["size"], after["mtime_ns"])
    assert before != after


def test_path_identity_inventories_diffusers_directory(tmp_path):
    root = tmp_path / "vae"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"w" * 8)
    before = identity.path_identity(root)

    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"w" * 16)
    after = identity.path_identity(root)

    assert before["files"] != after["files"]


def test_missing_path_is_recorded_not_raised(tmp_path):
    record = identity.path_identity(tmp_path / "nope")
    assert record == {"path": str(tmp_path / "nope"), "exists": False}


def test_compare_is_a_subset_check():
    recorded = {"a": 1, "b": 2, "provenance": "x"}
    assert identity.compare(recorded, {"a": 1}) == "match"
    assert identity.compare(recorded, {"a": 1, "b": 2}) == "match"
    assert identity.compare(recorded, {"a": 9}) == "mismatch"
    assert identity.compare(recorded, {"missing": 1}) == "mismatch"
    assert identity.compare(None, {"a": 1}) == "absent"
    assert identity.compare({}, {"a": 1}) == "absent"


def test_evaluate_legacy_policy():
    assert identity.evaluate(None, {"a": 1}) == "reuse"
    assert identity.evaluate(None, {"a": 1}, require_identity=True) == "stale"
    assert identity.evaluate({"a": 1}, {"a": 1}) == "reuse"
    assert identity.evaluate({"a": 2}, {"a": 1}) == "stale"


def test_stamp_round_trip_is_json_and_schema_versioned():
    metadata = identity.stamp({"model_family": "anima"}, {"stage": "text-encoder"})
    assert metadata["model_family"] == "anima"
    assert metadata["cache_identity_schema"] == identity.SCHEMA_VERSION
    assert identity.read_stamp(metadata) == {"stage": "text-encoder"}
    assert identity.read_stamp({"cache_identity": "{not json"}) is None
    assert identity.read_stamp({}) is None


def test_dtype_normalization_matches_torch_and_string_forms():
    assert identity.normalize_dtype(torch.bfloat16) == identity.normalize_dtype("BF16")
    assert identity.normalize_dtype("torch.float32") == "float32"
    assert identity.normalize_dtype(None) == ""


# --- text-encoder safetensors caches --------------------------------------


def _te_tensors():
    return {
        "prompt_embeds": torch.zeros(2, 3, dtype=torch.bfloat16),
        "attn_mask": torch.ones(2, dtype=torch.int32),
        "t5_input_ids": torch.zeros(2, dtype=torch.int64),
        "t5_attn_mask": torch.ones(2, dtype=torch.int32),
        "caption_dropout_rate": torch.tensor(0.0, dtype=torch.float32),
    }


def test_te_cache_rejects_changed_text_encoder(tmp_path):
    encoder = tmp_path / "qwen3.safetensors"
    encoder.write_bytes(b"e" * 32)
    cache = tmp_path / "a_anima_te.safetensors"

    writer = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, False, identity_inputs={"text_encoder": str(encoder)}
    )
    save_file(
        _te_tensors(),
        cache,
        metadata=writer.stamp_metadata({"model_family": "anima", "cache_schema": "anima_te_v1"}),
    )
    assert writer.is_disk_cached_outputs_expected(str(cache))

    reader = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, False, identity_inputs={"text_encoder": str(encoder)}
    )
    assert reader.is_disk_cached_outputs_expected(str(cache))

    encoder.write_bytes(b"f" * 64)
    assert not reader.is_disk_cached_outputs_expected(str(cache))


def test_te_cache_legacy_unstamped_is_reused_unless_required(tmp_path):
    cache = tmp_path / "a_anima_te.safetensors"
    save_file(
        _te_tensors(),
        cache,
        metadata={"model_family": "anima", "cache_schema": "anima_te_v1"},
    )

    lenient = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, False, identity_inputs={"text_encoder": "models/te.safetensors"}
    )
    strict = AnimaTextEncoderOutputsCachingStrategy(
        True,
        1,
        False,
        identity_inputs={"text_encoder": "models/te.safetensors"},
        require_cache_identity=True,
    )
    assert lenient.is_disk_cached_outputs_expected(str(cache))
    assert not strict.is_disk_cached_outputs_expected(str(cache))


def test_te_cache_identity_survives_skip_cache_check(tmp_path):
    encoder = tmp_path / "qwen3.safetensors"
    encoder.write_bytes(b"e" * 32)
    cache = tmp_path / "a_anima_te.safetensors"
    writer = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, False, identity_inputs={"text_encoder": str(encoder)}
    )
    save_file(
        _te_tensors(),
        cache,
        metadata=writer.stamp_metadata({"model_family": "anima", "cache_schema": "anima_te_v1"}),
    )

    # An explicit identity contract wins over skip_disk_cache_validity_check.
    skipping = AnimaTextEncoderOutputsCachingStrategy(
        True,
        1,
        True,
        identity_inputs={"text_encoder": str(encoder)},
        require_cache_identity=True,
    )
    assert skipping.is_disk_cached_outputs_expected(str(cache))

    encoder.write_bytes(b"f" * 64)
    assert not skipping.is_disk_cached_outputs_expected(str(cache))


def test_skip_cache_check_stays_the_escape_hatch_without_require_identity(tmp_path):
    encoder = tmp_path / "qwen3.safetensors"
    encoder.write_bytes(b"e" * 32)
    cache = tmp_path / "a_anima_te.safetensors"
    writer = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, False, identity_inputs={"text_encoder": str(encoder)}
    )
    save_file(
        _te_tensors(),
        cache,
        metadata=writer.stamp_metadata({"model_family": "anima", "cache_schema": "anima_te_v1"}),
    )
    encoder.write_bytes(b"f" * 64)
    lenient = AnimaTextEncoderOutputsCachingStrategy(
        True, 1, True, identity_inputs={"text_encoder": str(encoder)}
    )
    assert lenient.is_disk_cached_outputs_expected(str(cache))


def test_skip_cache_check_keeps_no_read_fast_path_without_identity(tmp_path):
    cache = tmp_path / "garbage_anima_te.safetensors"
    cache.write_bytes(b"not a safetensors file")
    strategy = AnimaTextEncoderOutputsCachingStrategy(True, 1, True)
    # No identity to assert -> the pre-existing no-IO path is preserved.
    assert strategy.is_disk_cached_outputs_expected(str(cache))


def test_te_cache_without_identity_inputs_stays_schema_only(tmp_path):
    cache = tmp_path / "a_anima_te.safetensors"
    save_file(
        _te_tensors(),
        cache,
        metadata={"model_family": "anima", "cache_schema": "anima_te_v1"},
    )
    strategy = AnimaTextEncoderOutputsCachingStrategy(True, 1, False)
    assert strategy.expected_cache_identity(set()) is None
    assert strategy.is_disk_cached_outputs_expected(str(cache))


# --- VAE latent .npz caches -----------------------------------------------


def _write_latent_npz(path, latent_dims, cache_identity=None):
    """Write an NPZ with one latents_<H>x<W> key per (latent H, latent W) pair."""
    payload = {}
    for h, w in latent_dims:
        payload[f"latents_{h}x{w}"] = np.zeros((16, 1, h, w), dtype=np.float32)
        payload[f"original_size_{h}x{w}"] = np.array([w * 8, h * 8])
        payload[f"crop_ltrb_{h}x{w}"] = np.zeros(4, dtype=np.int64)
    if cache_identity is not None:
        payload[identity.NPZ_KEY] = identity.encode_npz(cache_identity)
    np.savez(path, **payload)


def _vae_dir(tmp_path, name, payload):
    root = tmp_path / name
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(payload)
    return root


def test_latent_cache_rejected_when_vae_identity_changes(tmp_path):
    vae_a = _vae_dir(tmp_path, "vae_a", b"a" * 16)
    vae_b = _vae_dir(tmp_path, "vae_b", b"b" * 16)
    npz = tmp_path / "img_0064x0064_anima.npz"
    _write_latent_npz(
        npz, [(8, 8)], identity.latent_cache_identity(family="anima", vae=str(vae_a))
    )

    same = AnimaLatentsCachingStrategy(
        True,
        1,
        False,
        cache_identity=identity.latent_cache_identity(family="anima", vae=str(vae_a)),
    )
    other = AnimaLatentsCachingStrategy(
        True,
        1,
        False,
        cache_identity=identity.latent_cache_identity(family="anima", vae=str(vae_b)),
    )
    assert same.is_disk_cached_latents_expected((64, 64), str(npz), False, False)
    assert not other.is_disk_cached_latents_expected((64, 64), str(npz), False, False)


def test_latent_cache_legacy_npz_policy(tmp_path):
    npz = tmp_path / "img_0064x0064_anima.npz"
    _write_latent_npz(npz, [(8, 8)])
    stamp = identity.latent_cache_identity(family="anima", vae="models/vae.safetensors")

    lenient = AnimaLatentsCachingStrategy(True, 1, False, cache_identity=stamp)
    strict = AnimaLatentsCachingStrategy(
        True, 1, False, cache_identity=stamp, require_cache_identity=True
    )
    assert lenient.is_disk_cached_latents_expected((64, 64), str(npz), False, False)
    assert not strict.is_disk_cached_latents_expected((64, 64), str(npz), False, False)


def test_latent_save_drops_other_resolutions_when_vae_changed(tmp_path):
    npz = tmp_path / "img_anima.npz"
    vae_a = {"path": "/models/vae_a", "exists": True}
    vae_b = {"path": "/models/vae_b", "exists": True}
    _write_latent_npz(
        npz,
        [(8, 8), (8, 16)],
        {"schema": "1", "stage": "vae-latents", "family": "anima", "vae": vae_a},
    )

    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    strategy.save_latents_to_disk(
        str(npz),
        torch.zeros(1, 16, 8, 8),
        [64, 64],
        [0, 0, 64, 64],
        key_reso_suffix="_8x8",
        cache_identity={"schema": "1", "stage": "vae-latents", "family": "anima", "vae": vae_b},
    )

    with np.load(npz) as loaded:
        keys = set(loaded.files)
        assert "latents_8x8" in keys
        assert "latents_8x16" not in keys  # stale other-resolution key dropped
        assert identity.read_npz_stamp(loaded)["vae"] == vae_b


def test_latent_save_keeps_other_resolutions_when_vae_matches(tmp_path):
    npz = tmp_path / "img_anima.npz"
    vae = {"path": "/models/vae", "exists": True}
    stamp = {"schema": "1", "stage": "vae-latents", "family": "anima", "vae": vae}
    _write_latent_npz(npz, [(8, 8), (8, 16)], stamp)

    strategy = AnimaLatentsCachingStrategy(True, 1, False)
    strategy.save_latents_to_disk(
        str(npz),
        torch.zeros(1, 16, 8, 8),
        [64, 64],
        [0, 0, 64, 64],
        key_reso_suffix="_8x8",
        cache_identity=stamp,
    )

    with np.load(npz) as loaded:
        assert "latents_8x16" in set(loaded.files)


def test_preprocess_latent_probe_and_save_honour_vae_identity(tmp_path):
    vae_a = {"path": "/models/vae_a", "exists": True}
    vae_b = {"path": "/models/vae_b", "exists": True}
    stamp_a = {"schema": "1", "stage": "vae-latents", "family": "anima", "vae": vae_a}
    stamp_b = {"schema": "1", "stage": "vae-latents", "family": "anima", "vae": vae_b}
    npz = tmp_path / "img_0064x0064_anima.npz"
    _write_latent_npz(npz, [(8, 8), (8, 16)], stamp_a)

    assert _latent_cached(npz, 64, 64, cache_identity=stamp_a)
    assert not _latent_cached(npz, 64, 64, cache_identity=stamp_b)

    _save_batch(
        [(npz, np.zeros((1, 16, 8, 8), dtype=np.float32), (64, 64))],
        cache_identity=stamp_b,
    )
    with np.load(npz) as loaded:
        keys = set(loaded.files)
        assert "latents_8x16" not in keys
        assert identity.read_npz_stamp(loaded)["vae"] == vae_b


def test_preprocess_latent_probe_without_identity_keeps_legacy_behaviour(tmp_path):
    npz = tmp_path / "img_0064x0064_anima.npz"
    _write_latent_npz(npz, [(8, 8)])
    assert _latent_cached(npz, 64, 64)
    assert not _latent_cached(npz, 128, 128)


def test_decode_npz_rejects_non_identity_payloads():
    assert identity.decode_npz("") is None
    assert identity.decode_npz(np.array("not json")) is None
    assert identity.decode_npz(np.zeros((2, 2))) is None
    assert identity.decode_npz('[1, 2]') is None
    assert identity.decode_npz('{"a": 1}') == {"a": 1}
