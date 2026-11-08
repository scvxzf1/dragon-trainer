# Contributing

Thanks for contributing to Dragon Trainer (formerly Anima LoRA WebUI). This document defines the requirements for opening and reviewing a pull request. Project priorities live in [`docs/contribution-priorities.md`](docs/contribution-priorities.md); they are not part of the PR contract.

## Source of truth

Read [`AGENTS.md`](AGENTS.md) before changing code. It is the repository-wide source of truth for safety boundaries, architecture invariants, subsystem entry points, documentation lifecycle, and validation policy. A nearer `AGENTS.md` or `CLAUDE.md` adds directory-specific constraints.

When documentation and code disagree, verify the live implementation and tests, then update the stale documentation in the same PR when it is in scope.

## Before opening a PR

1. Open or comment on an [Issue](https://github.com/scvxzf1/dragon-trainer/issues) for work larger than Tier 1. Agree on the scope before implementing a new method or changing numerics.
2. Keep the change focused. Do not mix feature work with unrelated refactors, formatting, generated files, user data, model weights, caches, or training outputs.
3. Use the repository's Python 3.13 `uv` environment. For an unchanged lockfile, maintainers should prefer `uv sync --locked`; dependency changes must update both `pyproject.toml` and `uv.lock` through `uv add` or an explicit lock operation.
4. Do not introduce a generic `requirements.txt` or document ad-hoc `pip install` flows. The tracked `requirements-v100.txt` is a deliberate hardware-specific exception.
5. Inspect `git diff` before committing. Never include credentials, local absolute paths, `.env`, `.anima-webui-settings.toml`, model files, or runtime output.

## Validation rules

- Follow the risk-based [validation policy](AGENTS.md#验证策略). Low-risk changes use diff review and necessary syntax, type, or build checks; they do not require new tests or a test suite. When a specific contract needs testing, reuse the smallest relevant cases with the project interpreter:

  ```bash
  timeout 60 .venv/bin/python -m pytest tests/<relevant_test>.py -q
  ```

- Larger suites belong to focused acceptance or release validation when a concrete risk or an agreed release requirement calls for them. Crossing subsystem boundaries alone does not trigger broader tests. For example, select the core layer when its contracts need acceptance:

  ```bash
  timeout 300 .venv/bin/python tasks.py test-core
  ```

- Run checks only over the affected scope; do not reformat unrelated files:

  ```bash
  .venv/bin/python -m ruff check <changed_paths>
  .venv/bin/python -m ruff format --check <changed_paths>
  git diff --check -- <changed_paths>
  ```

- Documentation changes must keep the document reachable from [`docs/README.md`](docs/README.md) or a section index. Check changed paths with `git diff --check`; for moves or structural changes, check affected local links and navigation. The broader documentation-integrity checks are available for focused acceptance, not a routine step for every edit.
- GPU, model-download, and real-training validation belongs to an explicitly scoped acceptance stage when material to the claimed capability. Preserve the authorization and resource boundaries in `AGENTS.md`. State exactly what was and was not run; never report an unexecuted smoke test as passing.
- Acceptance can run asynchronously while implementation continues. Wait for evidence only when a dependent decision, a critical contract, or release requires it; otherwise distinguish implementation completion from pending acceptance. A passing focused check needs no repetition without new changes, failures, or unresolved concerns.
- If live `library/` or `networks/` changes affect a ComfyUI node, run or explicitly call out the need for `python tasks.py vendor-sync`.

## Tier 1 — focused fixes

Examples include bugs, typos, UI labels, error messages, small CLI changes, and documentation corrections.

Requirements:

- minimal scope with no drive-by refactor;
- checks scoped to the actual risk under `AGENTS.md`, with no blanket requirement to add tests for each new behavior;
- targeted browser verification when the changed interaction needs it;
- a short training smoke test only when needed for the scoped acceptance claim and a suitable environment is available; otherwise identify the pending acceptance boundary;
- user-visible behavior and documentation updated together.

## Tier 1.5 — efficiency or algorithm revision

This tier covers changes to an existing method's compute path, memory use, scheduling, or numerics without adding a new method.

Requirements:

1. **Reproducible bench:** add to an existing `bench/<topic>/` or create a focused bench. Report before/after values for the claimed metric; a percentage without raw, reproducible measurements is insufficient.
2. **Invariant test:** add or extend a test for numerical equivalence, schedule behavior, memory behavior, or another explicit regression boundary. State tolerances when exact equivalence is impossible.
3. **Correct harness:** DiT-loading benches use `bench/_anima.py` so adapter apply/load occurs before compile. Results use `bench/_common.py` where applicable.
4. **Documentation:** update the relevant method, optimization, or architecture document. Do not add a new top-level document unless the behavior needs a distinct user-facing entry.
5. **PR evidence:** include the exact commands, before/after output, environment, and result path.
6. **Compatibility statement:** state whether fixed-seed outputs, loss curves, checkpoints, config, or public APIs change.

A paper citation is welcome but not mandatory when the empirical evidence independently supports the revision.

## Tier 2 — new adapter or method

A new adapter/network implementation or a new method family is Tier 2. Discuss it in an Issue before implementation.

Requirements:

1. **Rationale and prior art:** cite the paper and upstream code when available. A method without prior art needs stronger empirical evidence and prior maintainer agreement.
2. **Dedicated bench:** add `bench/<method>/README.md`, runnable scripts, standard `result.json` output, interpretation, failure criteria, and a cited baseline result. DiT-loading scripts use `bench/_anima.py`.
3. **Method documentation:** add a stable document under `docs/methods/` or an explicitly bounded experimental document under `docs/experimental/`, and link it from the corresponding section index.
4. **Tests:** cover registry/config construction, at least one forward smoke path, metadata/save-load behavior, and important method invariants without requiring a real base-model download in unit tests.
5. **Task entry points:** expose supported train/test flows through `tasks.py`; `Makefile` aliases may remain as compatibility shims but must not be the only documented entry.
6. **Merge story:** prove merge equivalence for foldable adapters, or update refusal/compatibility logic and document why the method cannot be folded into DiT weights.
7. **Anima result:** provide a reproducible comparison against the appropriate baseline. “It trains without crashing” is not sufficient evidence.
8. **Integration surfaces:** check config schema, WebUI exposure, inference loading, checkpoint metadata, docs, tests, and ComfyUI vendor impact.

## Tier 3 — new model-family support

The [family registry](library/models/family_registry.py) defines the registered model families and capability boundaries. Adding a new model family is Tier 3 and requires maintainer agreement in an Issue before implementation. The existing Krea-2 path does not imply that every adapter or runtime feature is portable; see [`docs/multi_model_support.md`](docs/multi_model_support.md) for current support boundaries and [`_archive/docs/proposal/krea2_raw_migration.md`](_archive/docs/proposal/krea2_raw_migration.md) for migration background.

Fixes and documented capability extensions for an existing family use Tier 1, 1.5, or 2 according to their actual behavioral and numerical impact. A new family proposal must define its training and inference boundary, cache and checkpoint compatibility, adapter support matrix, configuration/WebUI surface, reproducible validation, and explicit refusals for unsupported features.

## PR description checklist

Copy the applicable items into the PR description:

- [ ] Tier identified and scope agreed when required.
- [ ] Summary explains the user-visible or architectural change.
- [ ] Relevant commands and exact results are listed.
- [ ] Validation limitations are stated; no unexecuted check is described as passing.
- [ ] Compatibility and migration impact are described.
- [ ] Documentation/config/WebUI/task surfaces are synchronized where applicable.
- [ ] No credentials, local paths, user data, model/cache/output files, debug prints, commented-out code, or unrelated formatting churn are included.
- [ ] Tier 1.5: reproducible before/after bench, invariant test, and numerical compatibility statement included.
- [ ] Tier 2: method rationale, bench README/result, method doc, tests, task entries, merge story, and Anima comparison included.
- [ ] Tier 3: maintainer agreement, family capability matrix, cache/checkpoint compatibility, and train/inference validation included.
- [ ] Custom-node impact checked and `vendor-sync` status reported.

## License

By contributing, you agree that your changes are licensed under the same terms as this repository; see [`LICENSE`](LICENSE).
