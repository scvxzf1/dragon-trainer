"""Import-and-construct smoke contract for the training bootstrap (t34 / F-T33-2).

t32 edited ``library/training/bootstrap.py`` and silently dropped
``@dataclass(frozen=True)`` from ``DatasetBuildResult``. Nothing caught it: the
business suites never import the bootstrap module, and ``git diff --check`` only
looks at whitespace. The breakage was ``TypeError: DatasetBuildResult() takes no
arguments`` inside ``prepare_dataset``, i.e. every real run died before its first
forward.

These tests are the missing net. They are deliberately dumb — import the module,
walk its own source for ``@dataclass`` declarations, and construct the result
objects — because that is exactly the failure mode that a business-level suite
misses.

Rule from t34 on: any change to a training-load-path module
(``library/training/bootstrap.py``, ``library/datasets/buckets.py``, ...) must
carry an import + construct smoke case in its verify, not just business tests and
``git diff --check``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect

import pytest
import torch

BOOTSTRAP_MODULE = "library.training.bootstrap"

#: Every dataclass this module is expected to declare. Deleting a decorator makes
#: the class disappear from the AST *and* from ``dataclasses.fields`` — pinning the
#: names here is what turns "silently lost the decorator" into a test failure (the
#: t33 accident: ``DatasetBuildResult`` became a plain class, and the first real run
#: died with ``TypeError: DatasetBuildResult() takes no arguments``).
EXPECTED_DATACLASSES = {
    "AcceleratorPrepareResult",
    "DatasetBuildResult",
    "NetworkBuildResult",
    "OptimizerBuildResult",
}


def _bootstrap():
    return importlib.import_module(BOOTSTRAP_MODULE)


def test_the_bootstrap_module_imports():
    module = _bootstrap()
    assert hasattr(module, "TrainingBootstrap")
    assert hasattr(module.TrainingBootstrap, "prepare_dataset")
    # The pinned-order measurement sampler lives here; it must stay constructible.
    sampler = module._PinnedOrderSampler(4)
    assert list(sampler) == [0, 1, 2, 3]
    assert isinstance(sampler, torch.utils.data.Sampler)


def test_dataset_build_result_is_a_frozen_dataclass_with_seven_fields():
    module = _bootstrap()
    cls = module.DatasetBuildResult

    assert dataclasses.is_dataclass(cls), (
        "DatasetBuildResult lost its @dataclass decorator: prepare_dataset() "
        "constructs it positionally and would raise TypeError on every real run"
    )
    fields = [field.name for field in dataclasses.fields(cls)]
    assert fields == [
        "train_dataset_group",
        "val_dataset_group",
        "current_epoch",
        "current_step",
        "collator",
        "use_user_config",
        "use_dreambooth_method",
    ]

    result = cls(None, None, 0, 0, None, False, False)
    assert result.current_epoch == 0
    assert result.use_dreambooth_method is False

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.current_epoch = 5


def test_every_dataclass_declared_in_the_bootstrap_is_a_dataclass():
    """Catch the whole class of accident, not just ``DatasetBuildResult``."""
    module = _bootstrap()
    source = inspect.getsource(module)
    tree = ast.parse(source)

    declared = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(ast.unparse(decorator))
        if any("dataclass" in text for text in decorators):
            declared.append(node.name)

    assert declared, "the bootstrap should still declare dataclasses"
    missing = EXPECTED_DATACLASSES - set(declared)
    assert not missing, (
        f"these classes lost their @dataclass decorator: {sorted(missing)} — a "
        "decorator deletion turns them into plain classes, so every positional "
        "construction raises TypeError on a real run"
    )
    for name in declared:
        obj = getattr(module, name, None)
        assert obj is not None, f"{name} is declared but not importable"
        assert dataclasses.is_dataclass(obj), (
            f"{name} is decorated with @dataclass in source but is not a dataclass "
            "at runtime — the decorator or the class body was disturbed"
        )


def test_the_bucket_module_imports_and_the_pinned_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("ANIMA_MULTIGPU_PINNED_ORDER", raising=False)
    buckets = importlib.import_module("library.datasets.buckets")

    manager = buckets.BucketManager((1024, 1024), 256, 2048, 64)
    assert manager.buckets == []

    assert buckets.pinned_order_enabled() is False
    assert buckets.apply_pinned_order(object()) == 0
    assert buckets.PINNED_ORDER_ENV == "ANIMA_MULTIGPU_PINNED_ORDER"

    monkeypatch.setenv("ANIMA_MULTIGPU_PINNED_ORDER", "1")
    assert buckets.pinned_order_enabled() is True


def test_the_gate_branch_in_the_loader_construction_reads_the_buckets_gate():
    """The bootstrap must consult the same gate the bucket layer does."""
    source = inspect.getsource(_bootstrap())
    assert "pinned_order_enabled()" in source
    assert "apply_pinned_order(train_dataset_group)" in source
    assert "_PinnedOrderSampler(len(train_dataset_group))" in source
    # The production branch stays as it was.
    assert "train_dataset_group, shuffle=True, **dataloader_kwargs," in source
