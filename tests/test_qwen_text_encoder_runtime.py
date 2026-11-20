"""CPU contracts for the low-memory Qwen3-VL cache encoder."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from library.models.qwen_image_2_1 import strategy, text_encoder_runtime as runtime


@pytest.fixture(scope="module")
def tiny_encoder():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 64, "max_position_embeddings": 128,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [1, 1, 2], "rope_type": "default"},
        },
        vision_config={
            "hidden_size": 32, "intermediate_size": 64, "depth": 2,
            "num_heads": 4, "patch_size": 2, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "out_hidden_size": 32,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=63, vision_start_token_id=62, vision_end_token_id=61,
        tie_word_embeddings=False,
    )
    torch.manual_seed(45)
    return Qwen3VLForConditionalGeneration(config).eval()


def _old_pre_norm_output(model, kwargs):
    norm = model.model.language_model.norm
    handle = norm.register_forward_hook(lambda _module, args, _output: args[0])
    try:
        with torch.no_grad():
            return model(**kwargs, output_hidden_states=True, use_cache=False).hidden_states[-1]
    finally:
        handle.remove()


@pytest.mark.parametrize("multimodal", [False, True])
def test_core_pre_norm_matches_full_encoder_without_lm_head_or_hidden_collection(
    tiny_encoder, monkeypatch, multimodal
):
    model = tiny_encoder
    kwargs = {
        "input_ids": torch.tensor([[0, 0, 5, 62, 63, 61, 7]]) if multimodal
        else torch.tensor([[0, 0, 5, 6, 7]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1]]) if multimodal
        else torch.tensor([[0, 0, 1, 1, 1]]),
    }
    if multimodal:
        kwargs.update(
            pixel_values=torch.randn(4, 24),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.tensor([[0, 0, 0, 0, 1, 0, 0]]),
        )
    expected = _old_pre_norm_output(model, kwargs)

    def forbidden_head(*_args, **_kwargs):
        raise AssertionError("cache encoding must not invoke lm_head")

    monkeypatch.setattr(model.lm_head, "forward", forbidden_head)
    norm = model.model.language_model.norm
    handle = norm.register_forward_hook(lambda _module, args, _output: args[0])
    try:
        with torch.no_grad():
            outputs = strategy._qwen_encoder_forward(model, {
                **kwargs, "output_hidden_states": False, "use_cache": False,
            })
    finally:
        handle.remove()
    assert outputs.hidden_states is None
    assert torch.equal(strategy._qwen_last_hidden_state(outputs), expected)


def test_offloaded_execution_device_does_not_read_meta_parameters(monkeypatch):
    class MetaModel:
        _qwen_execution_device = torch.device("cpu")

        def parameters(self):
            raise AssertionError("offloaded parameter devices must not be inspected")

    assert strategy._qwen_execution_device(MetaModel()) == torch.device("cpu")


def test_runtime_cpu_context_cleans_metadata_on_error(monkeypatch, tiny_encoder):
    from library.models.qwen_image_2_1 import weights

    monkeypatch.setattr(weights, "load_qwen_image_2_1_text_encoder", lambda *_a, **_k: tiny_encoder)
    with pytest.raises(ZeroDivisionError):
        with runtime.text_encoder_for_cache("unused", dtype=torch.float32, device="cpu", offload="off") as model:
            assert model._qwen_execution_device == torch.device("cpu")
            assert model._qwen_cache_offloaded is False
            1 / 0
    assert not hasattr(tiny_encoder, "_qwen_execution_device")
    assert not hasattr(tiny_encoder, "_qwen_cache_offloaded")


def test_cpu_offload_rejects_non_cuda_execution_device(tiny_encoder):
    with pytest.raises(ValueError, match="requires a CUDA execution device"):
        runtime._should_offload(tiny_encoder.model, torch.device("cpu"), "on")


def test_runtime_removes_offload_hooks_after_error(monkeypatch, tiny_encoder):
    import accelerate
    from accelerate import hooks
    from library.models.qwen_image_2_1 import weights

    events = []
    monkeypatch.setattr(weights, "load_qwen_image_2_1_text_encoder", lambda *_a, **_k: tiny_encoder)
    monkeypatch.setattr(runtime, "_should_offload", lambda *_a: True)
    monkeypatch.setattr(accelerate, "cpu_offload", lambda core, **_k: events.append(("install", core)))
    monkeypatch.setattr(hooks, "remove_hook_from_module", lambda core, **_k: events.append(("remove", core)))
    with pytest.raises(RuntimeError, match="cache failed"):
        with runtime.text_encoder_for_cache("unused", dtype=torch.float32, device="cpu") as model:
            assert model._qwen_cache_offloaded is True
            raise RuntimeError("cache failed")
    assert events == [("install", tiny_encoder.model), ("remove", tiny_encoder.model)]
    assert not hasattr(tiny_encoder, "_qwen_execution_device")
    assert not hasattr(tiny_encoder, "_qwen_cache_offloaded")


class _SmallEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        self.lm_head = torch.nn.Linear(2, 2)


def test_real_partial_offload_install_preserves_original_error_and_releases_encoder(monkeypatch):
    from accelerate import hooks

    model = _SmallEncoder()
    reference = weakref.ref(model)
    original_set = hooks.set_module_tensor_to_device
    failure = RuntimeError("injected offload init failure")
    meta_calls = 0

    def fail_partial(module, name, device, *args, _failure=failure, **kwargs):
        nonlocal meta_calls
        if str(device) == "meta":
            meta_calls += 1
            if meta_calls == 2:
                assert next(module.parameters()).is_meta
                assert not hasattr(module, "_hf_hook")
                raise _failure
        return original_set(module, name, device, *args, **kwargs)

    monkeypatch.setattr(runtime, "_should_offload", lambda *_a: True)
    with monkeypatch.context() as patch:
        patch.setattr(hooks, "set_module_tensor_to_device", fail_partial)
        with pytest.raises(RuntimeError, match="injected offload init failure") as raised:
            with runtime.loaded_text_encoder_for_cache(model, dtype=torch.float32, device="cpu"):
                pytest.fail("partially initialized encoder must never be yielded")
        assert raised.value is failure
        assert any("meta tensors" in note for note in failure.__notes__)
        assert not any(hasattr(module, "_hf_hook") or hasattr(module, "_old_forward") for module in model.modules())
        assert not hasattr(model, "_qwen_execution_device")
        assert not hasattr(model, "_qwen_cache_offloaded")
        assert model.lm_head.weight.device.type == "cpu"
    del raised, failure, fail_partial, model
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("encode_error", [False, True])
def test_real_offload_context_restores_weights_and_hooks(monkeypatch, encode_error):
    model = _SmallEncoder()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    failure = ValueError("encode failed")
    monkeypatch.setattr(runtime, "_should_offload", lambda *_a: True)

    def encode():
        with runtime.loaded_text_encoder_for_cache(model, dtype=torch.float32, device="cpu") as prepared:
            assert prepared.lm_head.weight.device.type == "cpu"
            assert next(prepared.model.parameters()).is_meta
            prepared.model(torch.ones(1, 2))
            if encode_error:
                raise failure

    if encode_error:
        with pytest.raises(ValueError, match="encode failed") as raised:
            encode()
        assert raised.value is failure
    else:
        encode()
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    assert not any(hasattr(module, "_hf_hook") or hasattr(module, "_old_forward") for module in model.modules())
    assert not hasattr(model, "_qwen_execution_device")
    assert not hasattr(model, "_qwen_cache_offloaded")


@pytest.mark.parametrize("encode_error", [False, True])
def test_cleanup_failure_is_exposed_without_masking_encode_error(monkeypatch, encode_error):
    model = _SmallEncoder()
    failure = ValueError("encode failed")
    cleanup_failure = RuntimeError("CPU cleanup failed")
    events = []

    def failed_move(*_args, **_kwargs):
        raise cleanup_failure

    monkeypatch.setattr(model, "to", failed_move)
    monkeypatch.setattr(runtime.gc, "collect", lambda: events.append("collect"))
    with pytest.raises((ValueError, RuntimeError)) as raised:
        with runtime.loaded_text_encoder_for_cache(model, dtype=torch.float32, device="cpu"):
            if encode_error:
                raise failure
    assert raised.value is (failure if encode_error else cleanup_failure)
    if encode_error:
        assert any("CPU cleanup failed" in note for note in failure.__notes__)
    assert events == ["collect"]
    assert not hasattr(model, "_qwen_execution_device")
    assert not hasattr(model, "_qwen_cache_offloaded")


@pytest.mark.parametrize("preview_error", [False, True])
def test_training_preview_uses_loaded_encoder_offload_without_whole_model_cuda(
    monkeypatch, preview_error
):
    import accelerate
    from accelerate import hooks
    from library.anima import text_strategies
    from library.models.qwen_image_2_1 import preview_conditions, weights
    from library.training import text_encoder_cache
    from library import env

    events = []

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

        def to(self, device, *args, **kwargs):
            assert torch.device(device).type != "cuda", "whole encoder moved to CUDA"
            events.append("encoder_cpu")
            return super().to(device, *args, **kwargs)

    model = Encoder()
    monkeypatch.setattr(weights, "load_qwen_image_2_1_text_encoder", lambda *_a, **_k: pytest.fail("duplicate load"))
    monkeypatch.setattr(runtime, "_resolve_device", lambda value: torch.device(value))
    monkeypatch.setattr(runtime, "_should_offload", lambda *_a: True)
    monkeypatch.setattr(accelerate, "cpu_offload", lambda core, **_k: events.append("install"))
    monkeypatch.setattr(hooks, "remove_hook_from_module", lambda core, **_k: events.append("remove"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(text_encoder_cache, "clean_memory_on_device", lambda _device: events.append("clean"))
    monkeypatch.setattr(text_encoder_cache, "sample_preview_enabled", lambda _args: True)
    monkeypatch.setattr(env, "resolve_model_family", lambda _args: "qwen_image_2_1")
    monkeypatch.setattr(text_strategies.TokenizeStrategy, "get_strategy", lambda: object())
    monkeypatch.setattr(text_strategies.TextEncodingStrategy, "get_strategy", lambda: object())

    def preview(_trainer, _args, encoder, _tokenizer, _strategy):
        assert encoder is model
        assert encoder._qwen_cache_offloaded is True
        assert encoder._qwen_execution_device == torch.device("cuda:1")
        events.append("preview")
        if preview_error:
            raise ValueError("preview failed")

    monkeypatch.setattr(preview_conditions, "cache_preview_text", preview)

    class Accelerator:
        device = torch.device("cuda:1")
        num_processes = 1

        def autocast(self):
            from contextlib import nullcontext
            return nullcontext()

        def wait_for_everyone(self):
            events.append("barrier")

    class Dataset:
        def new_cache_text_encoder_outputs(self, encoders, _accelerator):
            assert encoders == [None]

    trainer = SimpleNamespace(sample_prompts_snapshot=None, sample_prompts_te_outputs=None)
    args = SimpleNamespace(cache_text_encoder_outputs=True, sample_prompts="unused", qwen_text_encoder_cache_policy="cpu_offload")
    if preview_error:
        with pytest.raises(RuntimeError, match="Qwen.*preview text caching failed") as error:
            text_encoder_cache.cache_text_encoder_outputs_if_needed(trainer, args, Accelerator(), [model], Dataset())
        assert isinstance(error.value.__cause__, ValueError)
    else:
        text_encoder_cache.cache_text_encoder_outputs_if_needed(trainer, args, Accelerator(), [model], Dataset())

    assert events[:4] == ["install", "preview", "remove", "encoder_cpu"]
    assert events[4:6] == ["empty_cache", "clean"]
    assert ("barrier" in events) is not preview_error
    assert not hasattr(model, "_qwen_execution_device")
    assert not hasattr(model, "_qwen_cache_offloaded")


def test_single_process_cache_adapter_provides_barrier():
    from scripts.qwen_image_2_1.preprocess_edit_cache import _SingleProcess

    adapter = _SingleProcess(torch.device("cpu"))
    assert adapter.num_processes == 1
    assert adapter.process_index == 0
    assert adapter.wait_for_everyone() is None


def test_actual_cpu_offload_core_matches_eager_text_encoding(tiny_encoder):
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module

    model = tiny_encoder
    tokens = [torch.tensor([[0, 0, 5, 6, 7]]), torch.tensor([[0, 0, 1, 1, 1]])]
    tokenizer = SimpleNamespace(drop_idx=0)
    encoder = strategy.QwenImage21TextEncodingStrategy()
    eager = encoder.encode_tokens(tokenizer, [model], tokens)
    multimodal = {
        "input_ids": torch.tensor([[5, 62, 63, 61, 7]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "pixel_values": torch.randn(4, 24),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0]]),
        "output_hidden_states": False,
        "use_cache": False,
    }
    with torch.no_grad():
        eager_multimodal = strategy._qwen_encoder_forward(model, multimodal).last_hidden_state

    cpu_offload(model.model, execution_device=torch.device("cpu"), offload_buffers=False)
    model._qwen_execution_device = torch.device("cpu")
    try:
        assert next(model.model.parameters()).is_meta
        offloaded = encoder.encode_tokens(tokenizer, [model], tokens)
        assert all(torch.equal(actual, expected) for actual, expected in zip(offloaded, eager))
        with torch.no_grad():
            offloaded_multimodal = strategy._qwen_encoder_forward(model, multimodal).last_hidden_state
        assert torch.equal(offloaded_multimodal, eager_multimodal)
    finally:
        remove_hook_from_module(model.model, recurse=True)
        del model._qwen_execution_device
