import pytest
import torch

from library.inference.family_runner import FamilyInferenceRunner, is_cuda_oom


class _StageModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_family_runner_releases_stage_before_returning_result():
    cleanup = []
    runtime = FamilyInferenceRunner("test_family", torch.device("cpu"), cleanup=cleanup.append)
    stage = _StageModule()

    result = runtime.run_stage(
        "dit",
        lambda: stage,
        lambda loaded: (loaded.weight.detach().clone(), loaded.training),
    )

    assert result[0].item() == 1
    assert result[1] is True
    assert stage.weight.device == torch.device("cpu")
    assert cleanup == [torch.device("cpu")]


def test_family_runner_releases_stage_on_failure():
    cleanup = []
    runtime = FamilyInferenceRunner("test_family", torch.device("cpu"), cleanup=cleanup.append)
    stage = _StageModule()

    with pytest.raises(RuntimeError, match="stage failed"):
        runtime.run_stage(
            "dit",
            lambda: stage,
            lambda _loaded: (_ for _ in ()).throw(RuntimeError("stage failed")),
        )

    assert stage.weight.device == torch.device("cpu")
    assert cleanup == [torch.device("cpu")]


def test_family_runner_cleans_allocator_when_loader_fails():
    cleanup = []
    runtime = FamilyInferenceRunner(
        "test_family",
        torch.device("cpu"),
        cleanup=cleanup.append,
    )

    with pytest.raises(RuntimeError, match="load failed"):
        runtime.run_stage(
            "dit",
            lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
            lambda _loaded: pytest.fail("operation must not run"),
        )

    assert cleanup == [torch.device("cpu")]


def test_family_runner_retries_only_cuda_oom():
    cleanup = []
    runtime = FamilyInferenceRunner("test_family", torch.device("cpu"), cleanup=cleanup.append)
    calls = []

    def operation(attempt):
        calls.append(attempt)
        if attempt == 0:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return "ok"

    assert runtime.run_with_oom_retry(operation, max_retries=1) == "ok"
    assert calls == [0, 1]
    assert cleanup == [torch.device("cpu")]


def test_family_runner_stage_oom_retries_with_one_cleanup_per_attempt():
    cleanup = []
    runtime = FamilyInferenceRunner(
        "test_family",
        torch.device("cpu"),
        cleanup=cleanup.append,
    )
    attempts = []

    def loader():
        return _StageModule()

    def operation(_stage):
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return "ok"

    assert runtime.run_stage(
        "dit",
        loader,
        operation,
        max_oom_retries=1,
    ) == "ok"
    assert attempts == [0, 1]
    assert cleanup == [torch.device("cpu"), torch.device("cpu")]


def test_family_runner_text_failure_clears_allocator():
    cleanup = []
    runtime = FamilyInferenceRunner("test_family", torch.device("cpu"), cleanup=cleanup.append)

    with pytest.raises(RuntimeError, match="encoding failed"):
        runtime.run_text(lambda: (_ for _ in ()).throw(RuntimeError("encoding failed")))

    assert cleanup == [torch.device("cpu")]


def test_family_runner_does_not_retry_cpu_oom():
    cleanup = []
    runtime = FamilyInferenceRunner("test_family", torch.device("cpu"), cleanup=cleanup.append)
    calls = []

    def operation(attempt):
        calls.append(attempt)
        raise torch.OutOfMemoryError("CPU out of memory")

    with pytest.raises(torch.OutOfMemoryError, match="CPU out of memory"):
        runtime.run_with_oom_retry(operation, max_retries=2)

    assert calls == [0]
    assert cleanup == []


@pytest.mark.parametrize(
    "error,expected",
    [
        (torch.OutOfMemoryError("CUDA out of memory"), True),
        (RuntimeError("CUDA out of memory"), True),
        (RuntimeError("CPU out of memory"), False),
        (RuntimeError("CUDA kernel failed"), False),
    ],
)
def test_is_cuda_oom_is_fail_closed(error, expected):
    assert is_cuda_oom(error) is expected


def test_runner_keeps_family_identity():
    runtime = FamilyInferenceRunner("qwen_image_2_1", torch.device("cpu"))
    assert runtime.family == "qwen_image_2_1"
