"""Global settings routes."""

from __future__ import annotations

import asyncio
import errno

from aiohttp import web

from web.services import config_service, settings_service, training_service
from web.services.model_config_service import (
    ModelConfigConflictError,
    ModelConfigFileError,
    get_model_configs,
    save_model_configs,
)
from web.services.settings_service import (
    DistributedSettingsError,
    get_distributed_settings,
    get_global_settings,
    save_distributed_settings,
    save_global_settings,
)
from web.services.model_file_service import list_model_files
from web.services import settings_paths


def setup_settings_routes(app: web.Application) -> None:
    app.router.add_get("/api/settings/global", handle_global_settings_get)
    app.router.add_put("/api/settings/global", handle_global_settings_put)
    app.router.add_get("/api/settings/model-configs", handle_model_configs_get)
    app.router.add_put("/api/settings/model-configs", handle_model_configs_put)
    app.router.add_get("/api/settings/model-files", handle_model_files_get)
    app.router.add_get("/api/settings/model-path-status", handle_model_path_status_get)
    app.router.add_get("/api/settings/directories", handle_settings_directories_get)
    app.router.add_get("/api/settings/distributed", handle_distributed_settings_get)
    app.router.add_put("/api/settings/distributed", handle_distributed_settings_put)


async def handle_distributed_settings_get(request: web.Request) -> web.Response:
    del request
    return web.json_response(get_distributed_settings())


async def handle_distributed_settings_put(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        return web.json_response(await asyncio.to_thread(save_distributed_settings, data))
    except DistributedSettingsError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except (ValueError, OSError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_model_files_get(request: web.Request) -> web.Response:
    try:
        offset = int(request.query.get("offset", "0"))
        limit = int(request.query.get("limit", "100"))
        payload = await asyncio.to_thread(
            list_model_files, request.query.get("directory"), offset=offset, limit=limit,
        )
        return web.json_response(payload)
    except ModelConfigFileError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except PermissionError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=403)
    except (FileNotFoundError, NotADirectoryError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except OSError as e:
        invalid_path = e.errno in {errno.ENAMETOOLONG, errno.EINVAL}
        message = "模型目录路径无效" if invalid_path else "暂时无法读取模型目录，请稍后重试"
        return web.json_response({"ok": False, "error": message}, status=400 if invalid_path else 503)
    except (ValueError, RuntimeError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_model_path_status_get(request: web.Request) -> web.Response:
    try:
        payload = await asyncio.to_thread(
            settings_paths.check_model_path, request.query.get("path")
        )
        return web.json_response(payload)
    except PermissionError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=403)
    except ModelConfigFileError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=503)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        invalid_path = e.errno in {errno.ENAMETOOLONG, errno.EINVAL}
        message = "路径无效" if invalid_path else "暂时无法读取路径，请稍后重试"
        return web.json_response({"ok": False, "error": message}, status=400 if invalid_path else 503)


async def handle_settings_directories_get(request: web.Request) -> web.Response:
    try:
        offset = int(request.query.get("offset", "0"))
        limit = int(request.query.get("limit", "100"))
        payload = await asyncio.to_thread(
            settings_paths.list_directories,
            request.query.get("directory"),
            offset=offset,
            limit=limit,
        )
        return web.json_response(payload)
    except PermissionError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=403)
    except (FileNotFoundError, NotADirectoryError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        invalid_path = e.errno in {errno.ENAMETOOLONG, errno.EINVAL}
        message = "目录路径无效" if invalid_path else "暂时无法读取目录，请稍后重试"
        return web.json_response({"ok": False, "error": message}, status=400 if invalid_path else 503)


async def handle_global_settings_get(request: web.Request) -> web.Response:
    try:
        return web.json_response(get_global_settings())
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


async def handle_global_settings_put(request: web.Request) -> web.Response:
    data = await request.json()
    try:
        payload = save_global_settings(data)
        tagging_service = request.app.get("tagging_service")
        set_job_retention = getattr(tagging_service, "set_job_retention", None)
        if callable(set_job_retention):
            set_job_retention(payload.get("tagging_max_retained_jobs"))
        if "configs_root" in data:
            config_service.set_configs_root(settings_service.SETTINGS_FILE.parent)
            training_service.reload_runtime_storage_state(request.app.get("training_service"))
        return web.json_response(payload)
    except settings_service.GlobalSettingsConflictError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_model_configs_get(request: web.Request) -> web.Response:
    try:
        return web.json_response(get_model_configs())
    except (ModelConfigFileError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


async def handle_model_configs_put(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        return web.json_response(save_model_configs(data))
    except (ModelConfigConflictError, ModelConfigFileError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
