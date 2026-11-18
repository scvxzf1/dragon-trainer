"""Q-R5: WebUI route registry completeness after setup_routes."""

from __future__ import annotations

from aiohttp import web

from web.routes import setup_routes


def _registered_routes(app: web.Application) -> set[tuple[str, str]]:
    """Return {(METHOD, canonical_path)} for all registered resources."""
    out: set[tuple[str, str]] = set()
    for route in app.router.routes():
        resource = getattr(route, "resource", None)
        if resource is None:
            continue
        method = str(getattr(route, "method", "") or "").upper()
        if not method or method == "*":
            # aiohttp may expose HEAD/OPTIONS helpers; keep only concrete methods.
            continue
        path = str(resource.canonical)
        out.add((method, path))
    return out


# Critical surface used by WebUI bootstrap / training / config / safety domains.
# Keep this list intentional: missing any of these is a release-blocking regression.
REQUIRED_ROUTES: set[tuple[str, str]] = {
    # catalog / config
    ("GET", "/api/methods"),
    ("GET", "/api/methods/{method}/variants"),
    ("GET", "/api/presets"),
    ("GET", "/api/config/model-families"),
    ("GET", "/api/config/merged"),
    ("GET", "/api/config/steps"),
    ("GET", "/api/config/raw"),
    ("PUT", "/api/config/raw"),
    ("PATCH", "/api/config/raw"),
    ("DELETE", "/api/config/raw"),
    ("POST", "/api/config/raw/save-as"),
    ("GET", "/api/config/file-groups"),
    ("GET", "/api/config/field-help"),
    ("GET", "/api/config/groups"),
    ("GET", "/api/config/dataset-presets"),
    ("GET", "/api/config/dataset-presets/images"),
    ("GET", "/api/config/dataset-presets/image"),
    ("GET", "/api/config/dataset-presets/thumbnail"),
    ("GET", "/api/config/dataset-masks"),
    ("GET", "/api/config/dataset-masks/image"),
    ("PUT", "/api/config/dataset-masks/image"),
    ("POST", "/api/config/dataset-masks/apply"),
    ("GET", "/api/config/model-blocks"),
    ("GET", "/api/config/sample-references/preview"),
    ("GET", "/api/config/output-runs"),
    # settings / preview / analysis / environment / image-test / data-management
    ("GET", "/api/settings/global"),
    ("PUT", "/api/settings/global"),
    ("GET", "/api/settings/distributed"),
    ("PUT", "/api/settings/distributed"),
    ("GET", "/api/settings/model-configs"),
    ("PUT", "/api/settings/model-configs"),
    ("GET", "/api/settings/model-files"),
    ("GET", "/api/settings/model-path-status"),
    ("GET", "/api/settings/directories"),
    ("GET", "/api/data-management/roots"),
    ("GET", "/api/data-management/entries"),
    ("POST", "/api/data-management/cleanup/preview"),
    ("POST", "/api/data-management/cleanup"),
    ("POST", "/api/data-management/retention/preview"),
    ("POST", "/api/data-management/retention/cleanup"),
    ("GET", "/api/preview/settings"),
    ("PUT", "/api/preview/settings"),
    ("GET", "/api/preview/images"),
    ("DELETE", "/api/preview/images"),
    ("GET", "/api/preview/image"),
    ("GET", "/api/preview/weights"),
    ("GET", "/api/analysis/weights"),
    ("POST", "/api/analysis/inspect"),
    ("GET", "/api/environment/check"),
    ("GET", "/api/image-test/status"),
    ("POST", "/api/image-test/start"),
    ("POST", "/api/image-test/stop"),
    # tagging workbench: captioning is canonical, tagging is a compatibility alias
    ("GET", "/api/captioning/settings"),
    ("POST", "/api/captioning/jobs"),
    ("GET", "/api/tagging/settings"),
    ("POST", "/api/tagging/jobs"),
    # training / queue / history / ws
    ("POST", "/api/training/preflight"),
    ("POST", "/api/training/distributed/diagnosis"),
    ("POST", "/api/training/start"),
    ("POST", "/api/training/stop"),
    ("GET", "/api/training/status"),
    ("GET", "/api/training/queue"),
    ("POST", "/api/training/queue/start"),
    ("POST", "/api/training/queue/batch/start"),
    ("POST", "/api/training/queue/batch-start"),  # alias kept for compatibility
    ("POST", "/api/training/queue/settings"),
    ("POST", "/api/training/queue/{item_id}/retry"),
    ("DELETE", "/api/training/queue/{item_id}"),
    ("GET", "/api/training/history"),
    ("GET", "/api/training/history/{task_id}"),
    ("GET", "/api/training/history/{task_id}/logs"),
    ("GET", "/api/training/history/{task_id}/logs/search"),
    ("DELETE", "/api/training/history/{task_id}"),
    ("GET", "/ws/training"),
}


def test_setup_routes_registers_required_api_surface():
    app = web.Application()
    setup_routes(app)
    registered = _registered_routes(app)
    missing = sorted(REQUIRED_ROUTES - registered)
    assert not missing, f"missing required routes: {missing}"


def test_setup_routes_registers_expected_domain_prefixes():
    app = web.Application()
    setup_routes(app)
    paths = {path for _method, path in _registered_routes(app)}
    for prefix in (
        "/api/methods",
        "/api/presets",
        "/api/config/",
        "/api/settings/",
        "/api/data-management/",
        "/api/preview/",
        "/api/analysis/",
        "/api/environment/",
        "/api/image-test/",
        "/api/captioning/",
        "/api/tagging/",
        "/api/training/",
        "/ws/training",
    ):
        assert any(path == prefix or path.startswith(prefix) for path in paths), (
            f"no routes registered for prefix {prefix!r}"
        )


def test_setup_routes_has_sane_route_count_floor():
    """Guard against accidental mass-unregister while allowing growth."""
    app = web.Application()
    setup_routes(app)
    registered = _registered_routes(app)
    # Current tree registers ~90 concrete method+path pairs; floor leaves room
    # if a few experimental endpoints move, but catches empty/partial setup.
    assert len(registered) >= 70, f"route count too low: {len(registered)}"


def test_queue_batch_start_alias_both_registered():
    app = web.Application()
    setup_routes(app)
    registered = _registered_routes(app)
    assert ("POST", "/api/training/queue/batch/start") in registered
    assert ("POST", "/api/training/queue/batch-start") in registered
