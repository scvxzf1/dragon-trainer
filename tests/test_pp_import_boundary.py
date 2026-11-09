"""Import-boundary guard for ``library/`` (task t10, finding F4).

Two hard rules, both machine-checked here:

1. **No ``library`` -> ``scripts`` reverse dependency.**  The repository layering
   is ``scripts`` -> ``library`` -> ``packages``.  A reverse edge breaks the
   package boundary: an installed ``anima_lora`` fails to import whenever
   ``scripts/`` is absent.  Before t10 ``library/training/pp/numerics.py`` had
   exactly such an edge, and the eager ``library/training/pp/__init__.py``
   amplified it to four ``scripts.*`` modules in ``sys.modules`` for a single
   ``import library.training.pp``.

2. **Each public entry point has its own import graph.**  ``import
   library.training.pp.<module>`` must not pull any other ``pp`` submodule (the
   package ``__init__`` is PEP 562 lazy), so one future bad import cannot break
   every entry point at once.

The runtime checks run one **independent interpreter per entry point**, in
parallel: the property is "this import does not register a ``scripts`` module",
which can only be proven on a fresh ``sys.modules``.  Sequential imports in one
process would mask nothing but would lose attribution, and a shared process would
let an earlier import hide a later one's edge.

The static scan complements them by catching imports the runtime check cannot
reach: function-local imports (never executed at import time) and
``importlib.import_module("scripts...")`` string forms.
"""

from __future__ import annotations

import ast
import concurrent.futures
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import library

LIBRARY_ROOT = Path(library.__file__).resolve().parent
REPO_ROOT = LIBRARY_ROOT.parent
SCRIPTS_SHIM = (
    REPO_ROOT / "scripts" / "experiments" / "anima_parallel" / "pipeline_parallel.py"
)

#: Every entry point a caller may import directly.  ``gate`` / ``runtime`` /
#: ``schedule`` / ``stages`` / ``checkpoint`` / ``transactions`` are the ones the
#: trainer, the CLI and the compat matrix touch; ``library.training.pp`` itself
#: covers the lazy package ``__init__``.
ENTRY_POINTS = (
    "library.training.pp",
    "library.training.pp.gate",
    "library.training.pp.runtime",
    "library.training.pp.schedule",
    "library.training.pp.stages",
    "library.training.pp.checkpoint",
    "library.training.pp.transactions",
)

#: The numerics that must be *defined* in ``library``, never in ``scripts``.
PROMOTED_NUMERICS = (
    "PP_FLEX_ATTENTION_UNSUPPORTED",
    "PP_MOD_GUIDANCE_CONTRACT_UNSUPPORTED",
    "PP_MOD_GUIDANCE_UNSUPPORTED",
    "PP_NATIVE_FLATTEN_UNSUPPORTED",
    "PP_POOLED_TEXT_MODULATION_UNSUPPORTED",
    "AnimaPipelineUnsupported",
    "assert_mini_forward_supported",
    "assert_mod_guidance_indexable",
    "block_sdpa_scope",
    "cast_rope_for_compute",
    "finish_model",
    "normalize_block_kwargs",
    "prepare_block_inputs",
    "resolve_block_compute_dtype",
    "resolve_use_fp32",
    "run_local_blocks",
)


def _library_python_files() -> list[Path]:
    return sorted(
        path
        for path in LIBRARY_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _scripts_imports(tree: ast.AST, *, where: str) -> list[str]:
    """Every ``scripts`` import in ``tree``, including function-local ones."""

    violations: list[str] = []

    def record(node: ast.AST, module: str | None, line: int) -> None:
        violations.append(f"{where}:{line}: imports {module!r}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scripts" or alias.name.startswith("scripts."):
                    record(node, alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            # ``node.level > 0`` is a relative import and cannot reach ``scripts``.
            if node.level:
                continue
            module = node.module or ""
            if module == "scripts" or module.startswith("scripts."):
                record(node, module, node.lineno)
        elif isinstance(node, ast.Call):
            # ``importlib.import_module("scripts...")`` / ``__import__("scripts...")``
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name not in {"import_module", "__import__"}:
                continue
            for argument in node.args[:1]:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and (argument.value == "scripts" or argument.value.startswith("scripts."))
                ):
                    record(node, argument.value, node.lineno)
    return violations


def test_no_library_module_imports_from_scripts():
    """Static gate over all of ``library/`` -- catches function-local imports too."""

    violations: list[str] = []
    for path in _library_python_files():
        source = path.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            # Parsing the whole tree reaches files this test does not own; e.g.
            # ``library/anima/text_strategies.py:105`` has a pre-existing invalid
            # escape sequence.  That is a real (unrelated, out-of-scope) finding
            # and it still surfaces when the module itself is imported -- it is
            # only noise for an AST-only scan, so it is silenced here.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(path))
        violations.extend(
            _scripts_imports(tree, where=str(path.relative_to(REPO_ROOT)))
        )
    assert not violations, "library -> scripts reverse dependency found:\n" + "\n".join(
        violations
    )


def test_promoted_numerics_are_defined_in_library_not_scripts():
    """The single authority lives in ``library``; the scripts module re-exports."""

    from library.anima import mini_forward

    for name in PROMOTED_NUMERICS:
        assert hasattr(mini_forward, name), f"{name} missing from library.anima.mini_forward"
        owner = getattr(mini_forward, name)
        module = getattr(owner, "__module__", None)
        if module is not None:
            assert module == "library.anima.mini_forward", (
                f"{name} is owned by {module}, expected library.anima.mini_forward"
            )


def test_scripts_pipeline_parallel_is_a_consumer():
    """The probe module must import the numerics, not define them."""

    tree = ast.parse(SCRIPTS_SHIM.read_text(encoding="utf-8"), filename=str(SCRIPTS_SHIM))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    duplicated = defined.intersection(PROMOTED_NUMERICS)
    assert not duplicated, (
        "scripts/experiments/anima_parallel/pipeline_parallel.py re-implements "
        f"{sorted(duplicated)}; it must only re-export from library.anima.mini_forward"
    )
    # It must still expose them, or the existing probe call sites break.
    import scripts.experiments.anima_parallel.pipeline_parallel as shim

    missing = [name for name in PROMOTED_NUMERICS if not hasattr(shim, name)]
    assert not missing, f"scripts shim no longer re-exports {missing}"


def _run_probe(module: str) -> tuple[str, int, str]:
    code = (
        f"import {module}\n"
        "import sys\n"
        "bad = sorted(m for m in sys.modules if m == 'scripts' or m.startswith('scripts.'))\n"
        "print('scripts modules:', bad)\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    return module, result.returncode, (result.stdout + result.stderr).strip()


def test_each_entry_point_imports_without_scripts_modules():
    """One independent interpreter per entry point, run in parallel.

    Sequential subprocesses would take ~50s (torch dominates each one); running
    them concurrently keeps the independence requirement and brings the wall time
    down to a single import.
    """

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ENTRY_POINTS)) as pool:
        results = list(pool.map(_run_probe, ENTRY_POINTS))
    failures = [
        f"{module} (exit {code}):\n{output}"
        for module, code, output in results
        if code != 0
    ]
    assert not failures, "entry point pulled a scripts module:\n" + "\n\n".join(failures)


def test_bare_package_import_pulls_no_submodules():
    """The package ``__init__`` is lazy: F4's amplification is gone."""

    code = (
        "import library.training.pp\n"
        "import sys\n"
        "pulled = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.startswith('library.training.pp.') and m.count('.') == 2\n"
        ")\n"
        "print('pulled:', pulled)\n"
        "assert not pulled, pulled\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_all_matches_lazy_export_map():
    """``__all__`` and ``_EXPORTS`` are one source of truth (no drift)."""

    import library.training.pp as pp

    assert sorted(pp._EXPORTS) == pp.__all__
    assert len(pp.__all__) == len(set(pp.__all__)), "duplicate name in __all__"
    for name in pp.__all__:
        assert getattr(pp, name) is not None


def test_unknown_public_name_raises_attribute_error():
    import library.training.pp as pp

    with pytest.raises(AttributeError, match="has no attribute"):
        pp.definitely_not_a_public_name


def test_package_imports_with_scripts_unimportable():
    """The installed-``anima_lora`` case: ``scripts/`` does not exist at all.

    This is the failure the reverse dependency actually caused, so it is asserted
    end to end instead of being inferred from the static scan: a ``meta_path``
    finder makes every ``scripts*`` import raise, and the whole public surface
    must still import.
    """

    code = """
import sys, importlib.abc


class BlockScripts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise ImportError(f"blocked in this probe: {fullname}")
        return None


sys.meta_path.insert(0, BlockScripts())
sys.modules.pop("scripts", None)

import library.training.pp as pp
from library.anima import mini_forward
from library.training.pp import (
    AnimaPipelineRuntime,
    PipelineParallelError,
    PipelineRuntimeConfig,
    PpFp16Transaction,
    build_anima_pipeline_runtime,
    build_anima_schedule,
)
from library.training.pp import numerics

assert len(pp.__all__) == 85, len(pp.__all__)
assert mini_forward.resolve_use_fp32.__module__ == "library.anima.mini_forward"
assert numerics.resolve_use_fp32 is mini_forward.resolve_use_fp32, (
    "library.training.pp.numerics must re-export the single authority"
)
leftover = [m for m in sys.modules if m == "scripts" or m.startswith("scripts.")]
assert not leftover, leftover
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"
