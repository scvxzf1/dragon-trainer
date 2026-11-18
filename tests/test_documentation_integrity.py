from __future__ import annotations

import re
import subprocess
import tomllib
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import tasks
from library.models.family_registry import MODEL_FAMILY_REGISTRY
from library.training.pp.gate import TRAINER_PP_WIRED_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
ARCHIVE_ROOT = ROOT / "_archive"
ARCHIVE_DOCS_ROOT = ARCHIVE_ROOT / "docs"
MAINTENANCE_DOCS = (
    ROOT / "AGENTS.md",
    ROOT / "CLAUDE.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "networks" / "CLAUDE.md",
    ROOT / "custom_nodes" / "comfyui-hydralora" / "CLAUDE.md",
)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REF_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)(?:\s+#+\s*)?$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
TEST_FILE_TOKEN_RE = re.compile(r"`(tests/test_[A-Za-z0-9_]+\.py)`")
TASKS_CMD_RE = re.compile(r"(?:python(?:\.exe)?\s+)?tasks\.py\s+([a-z0-9][a-z0-9_-]*)")

UNTRACKED_LINK_PREFIXES = (
    "output/",
    "post_image_dataset/",
    "image_dataset/",
    "logs/",
    "models/",
    ".venv/",
    ".venv-v100/",
    ".worktrees/",
)


def _tracked_files() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        (ROOT / raw.decode("utf-8")).resolve()
        for raw in result.stdout.split(b"\0")
        if raw
    }


def _markdown_files() -> list[Path]:
    tracked_md = sorted(
        path for path in _tracked_files() if path.suffix.lower() == ".md" and path.exists()
    )
    if tracked_md:
        return tracked_md
    return sorted(
        {
            ROOT / "README.md",
            *MAINTENANCE_DOCS,
            *DOCS_ROOT.rglob("*.md"),
            *ARCHIVE_DOCS_ROOT.rglob("*.md"),
        }
    )


def _content_lines(path: Path):
    fenced = False
    marker = ""
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        match = FENCE_RE.match(line)
        if match:
            current = match.group(1)
            if not fenced:
                fenced = True
                marker = current
            elif current == marker:
                fenced = False
                marker = ""
            continue
        if not fenced:
            yield line_number, line


def _iter_raw_links(path: Path):
    for line_number, line in _content_lines(path):
        stripped_code = re.sub(r"`[^`]*`", "", line)
        for match in LINK_RE.finditer(stripped_code):
            raw = match.group(1).strip()
            raw = raw.split(' "', 1)[0].split(" '", 1)[0]
            yield line_number, raw
        ref_match = REF_DEF_RE.match(stripped_code)
        if ref_match:
            raw = ref_match.group(1).strip().strip("<>")
            yield line_number, raw


def _local_link_targets(path: Path):
    for line_number, raw in _iter_raw_links(path):
        if not raw or raw.startswith(
            ("#", "http://", "https://", "mailto:", "data:", "javascript:")
        ):
            continue
        path_part, separator, fragment = raw.partition("#")
        target = (path.parent / unquote(path_part)).resolve()
        yield line_number, raw, target, unquote(fragment) if separator else ""


def _github_slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"<[^>]*>", "", text).strip().lower()
    output: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character in "_-":
            output.append(character)
        elif character.isspace():
            output.append("-")
        elif category[0] in "LN" or category in {"Mn", "Mc"}:
            output.append(character)
    return "".join(output)


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for _, line in _content_lines(path):
        match = HEADING_RE.match(line)
        if not match:
            continue
        base = _github_slug(match.group(1))
        duplicate_index = counts.get(base, 0)
        counts[base] = duplicate_index + 1
        anchors.add(base if duplicate_index == 0 else f"{base}-{duplicate_index}")
    return anchors


def test_markdown_fences_are_balanced():
    failures: list[str] = []
    for path in _markdown_files():
        marker = ""
        start_line = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            match = FENCE_RE.match(line)
            if not match:
                continue
            current = match.group(1)
            if not marker:
                marker = current
                start_line = line_number
            elif current == marker:
                marker = ""
                start_line = 0
        if marker:
            failures.append(f"{path.relative_to(ROOT)}:{start_line}")
    assert not failures, "unclosed Markdown fences:\n" + "\n".join(failures)


def test_local_markdown_links_and_anchors_exist():
    missing: list[str] = []
    out_of_tree_or_runtime: list[str] = []
    bad_anchors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    tracked = _tracked_files()

    for path in _markdown_files():
        rel_doc = path.relative_to(ROOT)
        for line_number, raw, target, fragment in _local_link_targets(path):
            try:
                rel_target = target.relative_to(ROOT).as_posix()
            except ValueError:
                out_of_tree_or_runtime.append(
                    f"{rel_doc}:{line_number}: external local path {raw}"
                )
                continue

            if any(
                rel_target == prefix.rstrip("/") or rel_target.startswith(prefix)
                for prefix in UNTRACKED_LINK_PREFIXES
            ):
                out_of_tree_or_runtime.append(
                    f"{rel_doc}:{line_number}: runtime/untracked path {raw}"
                )
                continue

            if not target.exists():
                missing.append(f"{rel_doc}:{line_number}: {raw}")
                continue

            if target.is_file() and target not in tracked:
                out_of_tree_or_runtime.append(
                    f"{rel_doc}:{line_number}: untracked file target {raw}"
                )
                continue

            if fragment and target.suffix.lower() == ".md":
                anchors = anchor_cache.setdefault(target, _anchors(target))
                fragment = fragment.split(" ", 1)[0]
                if fragment not in anchors:
                    bad_anchors.append(f"{rel_doc}:{line_number}: {raw}")

    errors: list[str] = []
    if missing:
        errors.append("missing local Markdown targets:\n  " + "\n  ".join(missing))
    if out_of_tree_or_runtime:
        errors.append(
            "links to out-of-repo or untracked runtime paths (use backticks instead):\n  "
            + "\n  ".join(out_of_tree_or_runtime)
        )
    if bad_anchors:
        errors.append("missing Markdown anchors:\n  " + "\n  ".join(bad_anchors))
    assert not errors, "\n\n".join(errors)


def test_every_document_is_reachable_from_docs_index():
    documents = {
        path.resolve()
        for root in (DOCS_ROOT, ARCHIVE_DOCS_ROOT)
        for path in root.rglob("*.md")
    }
    edges: dict[Path, set[Path]] = {path: set() for path in documents}
    for path in documents:
        for _, _, target, _ in _local_link_targets(path):
            if target in documents:
                edges[path].add(target)

    start = (DOCS_ROOT / "README.md").resolve()
    reached: set[Path] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(edges[current] - reached)

    unreachable = sorted(path.relative_to(ROOT) for path in documents - reached)
    assert not unreachable, "unreachable documentation:\n" + "\n".join(
        map(str, unreachable)
    )


def test_section_indexes_list_every_document():
    failures: list[str] = []
    for section in sorted(path for path in DOCS_ROOT.iterdir() if path.is_dir()):
        index = section / "README.md"
        if not index.exists():
            continue
        linked: set[Path] = set()
        for _, _, target, _ in _local_link_targets(index):
            try:
                target.relative_to(section.resolve())
            except ValueError:
                continue
            if target.suffix.lower() == ".md":
                linked.add(target)
        actual = {path.resolve() for path in section.rglob("*.md") if path != index}
        for path in sorted(actual - linked):
            failures.append(f"{index.relative_to(ROOT)} missing {path.relative_to(ROOT)}")
    assert not failures, "incomplete documentation indexes:\n" + "\n".join(
        failures
    )


def test_no_duplicate_filenames_between_experimental_and_proposal():
    exp_names = {
        path.name
        for path in (DOCS_ROOT / "experimental").glob("*.md")
        if path.name != "README.md"
    }
    prop_names = {
        path.name
        for path in (DOCS_ROOT / "proposal").glob("*.md")
        if path.name != "README.md"
    }
    collisions = sorted(exp_names & prop_names)
    assert not collisions, (
        "duplicate document names across docs/experimental and docs/proposal "
        f"(archive frozen specs once wired): {collisions}"
    )


def test_archive_has_no_orphan_markdown_outside_archive_docs():
    orphans = sorted(
        path.relative_to(ROOT)
        for path in ARCHIVE_ROOT.rglob("*.md")
        if not path.resolve().is_relative_to(ARCHIVE_DOCS_ROOT.resolve())
    )
    assert not orphans, "orphan Markdown outside _archive/docs/:\n" + "\n".join(
        map(str, orphans)
    )


def test_lifecycle_sensitive_sections_have_status_labels():
    failures: list[str] = []
    markers = ("状态：", "Status:", "Current status:", "> Current status:")
    for section_name in ("experimental", "findings", "proposal"):
        for path in (DOCS_ROOT / section_name).rglob("*.md"):
            header = "\n".join(
                path.read_text(encoding="utf-8", errors="replace").splitlines()[:25]
            )
            if not any(marker in header for marker in markers):
                failures.append(str(path.relative_to(ROOT)))
    assert not failures, "missing lifecycle status labels:\n" + "\n".join(failures)


def test_archived_documents_are_linked_in_archive_indexes() -> None:
    archive_index = (DOCS_ROOT / "archive-index.md").read_text(encoding="utf-8")
    failures: list[str] = []

    for section in sorted(path for path in ARCHIVE_DOCS_ROOT.iterdir() if path.is_dir()):
        readme = section / "README.md"
        if not readme.exists():
            failures.append(f"missing archive section index: {readme.relative_to(ROOT)}")
            continue
        linked = {
            target.resolve()
            for _, _, target, _ in _local_link_targets(readme)
            if target.suffix.lower() == ".md"
        }
        direct_docs = {
            path.resolve() for path in section.glob("*.md") if path.name != "README.md"
        }
        for missing in sorted(direct_docs - linked):
            failures.append(
                f"{readme.relative_to(ROOT)} does not link {missing.relative_to(ROOT)}"
            )
        if section.name not in archive_index:
            failures.append(
                f"docs/archive-index.md does not mention archive section {section.name!r}"
            )

    for path in sorted((ARCHIVE_DOCS_ROOT / "proposal").glob("*.md")):
        if path.name != "README.md" and path.name not in archive_index:
            failures.append(f"docs/archive-index.md missing archived proposal {path.name}")

    assert not failures, "archive index drift:\n" + "\n".join(failures)


def test_active_docs_reference_existing_pytest_files() -> None:
    active_roots = [
        ROOT / "README.md",
        *MAINTENANCE_DOCS,
        *sorted((ROOT / "web" / "frontend-next").rglob("*.md")),
        *(
            path
            for section in ("guidelines", "configuration", "methods", "experimental", "features", "optimizations")
            for path in sorted((DOCS_ROOT / section).rglob("*.md"))
        ),
    ]
    missing: list[str] = []
    for doc in active_roots:
        for line_number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in TEST_FILE_TOKEN_RE.finditer(line):
                rel = match.group(1)
                if not (ROOT / rel).exists():
                    missing.append(f"{doc.relative_to(ROOT)}:{line_number}: {rel}")
    assert not missing, "active docs reference deleted test files:\n" + "\n".join(missing)


def test_active_docs_tasks_commands_exist_in_registry() -> None:
    known_commands = set(tasks.COMMANDS) | {"--help"}
    checked_docs = [
        ROOT / "README.md",
        *MAINTENANCE_DOCS,
        *(
            path
            for section in ("guidelines", "configuration", "methods", "features", "optimizations")
            for path in sorted((DOCS_ROOT / section).rglob("*.md"))
        ),
        *sorted((ROOT / "custom_nodes").glob("*/README.md")),
        ROOT / "examples" / "README.md",
    ]
    unknown: list[str] = []
    for doc in checked_docs:
        if not doc.exists():
            continue
        for line_number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in TASKS_CMD_RE.finditer(line):
                cmd = match.group(1)
                if cmd not in known_commands:
                    unknown.append(f"{doc.relative_to(ROOT)}:{line_number}: tasks.py {cmd}")
    assert not unknown, "docs reference unknown tasks.py commands:\n" + "\n".join(unknown)


def test_model_family_and_pp_contracts_match_live_registry() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    multi = (DOCS_ROOT / "multi_model_support.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for family_name, spec in MODEL_FAMILY_REGISTRY.items():
        assert f"`{family_name}`" in agents, f"AGENTS.md missing model family {family_name}"
        assert f"`{family_name}`" in multi, f"multi_model_support.md missing {family_name}"
        assert (
            family_name in readme or spec.display_name in readme
        ), f"README.md missing model family {family_name} / {spec.display_name}"

    pp_available = {
        name
        for name, spec in MODEL_FAMILY_REGISTRY.items()
        if spec.pipeline_parallel is not None and spec.pipeline_parallel.runtime_available
    }
    assert pp_available == set(TRAINER_PP_WIRED_FAMILIES)
    assert "当前三族的 registry capability 都显式标记 `runtime_available=false`" not in multi


def test_current_optimization_document_matches_live_config_surfaces():
    document = (DOCS_ROOT / "optimization-configs-current.md").read_text(
        encoding="utf-8"
    )
    base = tomllib.loads((ROOT / "configs/base.toml").read_text(encoding="utf-8"))
    presets = tomllib.loads(
        (ROOT / "configs/presets.toml").read_text(encoding="utf-8")
    )

    custom_down = str(bool(base["use_custom_down_autograd"])).lower()
    custom_down_line = next(
        line for line in document.splitlines() if line.startswith("| `use_custom_down_autograd`")
    )
    assert f"base `{custom_down}`" in custom_down_line

    preset_line = next(
        line for line in document.splitlines() if line.startswith("| `preset` |")
    )
    assert not [name for name in presets if f"`{name}`" not in preset_line]

    method_line = next(
        line for line in document.splitlines() if line.startswith("| `method` |")
    )
    methods = {path.stem for path in (ROOT / "configs/methods").glob("*.toml")}
    assert not [name for name in sorted(methods) if f"`{name}`" not in method_line]

    variant_line = next(
        line for line in document.splitlines() if line.startswith("| 自包含变体文件 |")
    )
    variants = {path.stem for path in (ROOT / "configs/gui-methods").glob("*.toml")}
    assert not [name for name in sorted(variants) if f"`{name}`" not in variant_line]

    router_line = next(
        line for line in document.splitlines() if line.startswith("| `router_source` |")
    )
    config_source = (ROOT / "networks/lora_anima/config.py").read_text(
        encoding="utf-8"
    )
    literal = re.search(r"RouterSource\s*=\s*Literal\[([^\]]+)\]", config_source)
    assert literal is not None
    router_sources = set(re.findall(r'["\']([^"\']+)["\']', literal.group(1)))
    assert not [name for name in sorted(router_sources) if f"`{name}`" not in router_line]


def test_configuration_and_training_docs_match_current_defaults():
    external = (DOCS_ROOT / "configuration/external-configs.md").read_text(
        encoding="utf-8"
    )
    priority = [
        external.index(".anima-webui-settings.toml [paths].configs_root"),
        external.index("`ANIMA_CONFIGS_ROOT` 环境变量"),
        external.index("默认 `configs/` 目录"),
    ]
    assert priority == sorted(priority)

    base = tomllib.loads((ROOT / "configs/base.toml").read_text(encoding="utf-8"))
    training = (DOCS_ROOT / "guidelines/training.md").read_text(encoding="utf-8")
    dataset = base["datasets"][0]
    if not base["use_cmmd"] and dataset["validation_split_num"] == 0:
        assert "Validation is **disabled by default**" in training
    assert '"crossattn_emb"' in training
