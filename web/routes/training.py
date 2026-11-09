"""Training control and WebSocket routes."""

from __future__ import annotations

import asyncio

from urllib.parse import quote

from aiohttp import web
from web.routes.training_history_list import history_list_response
from web.routes.training_artifacts import handle_history_artifact_manifest
from web.routes.training_queue_commands import queue_command_response

from library.runtime.launch import resolve_training_world_size_for_gpu_selection
from web.services.config_service import is_web_runtime_config, preflight_training_config
from web.services.training.risk_acknowledgement import acknowledge_request_risks
from web.services.training.gpu import normalize_gpu_whitelist
from web.services.training_service import inspect_continue_lora_weight


def _preflight_world_size(gpu_whitelist) -> int:
    return resolve_training_world_size_for_gpu_selection(
        normalize_gpu_whitelist(gpu_whitelist)
    )


def setup_training_routes(app: web.Application) -> None:
    app.router.add_post("/api/training/preflight", handle_preflight)
    app.router.add_post("/api/training/continue-lora/inspect", handle_continue_lora_inspect)
    app.router.add_post("/api/training/start", handle_start)
    app.router.add_post("/api/training/resume", handle_resume)
    app.router.add_post("/api/training/preprocess", handle_preprocess)
    app.router.add_post("/api/training/stop", handle_stop)
    app.router.add_get("/api/training/status", handle_status)
    app.router.add_get("/api/training/metrics", handle_metrics)
    app.router.add_get("/api/training/logs", handle_logs)
    app.router.add_get("/api/training/gpus", handle_gpus)
    app.router.add_get("/api/training/queue", handle_queue_status)
    app.router.add_post("/api/training/queue", handle_queue_post)
    app.router.add_post("/api/training/queue/start", handle_queue_start)
    app.router.add_post("/api/training/queue/batch/start", handle_queue_batch_start)
    app.router.add_post("/api/training/queue/batch-start", handle_queue_batch_start)
    app.router.add_post("/api/training/queue/resume", handle_queue_resume)
    app.router.add_post("/api/training/queue/settings", handle_queue_settings)
    app.router.add_post("/api/training/queue/cancel-all", handle_queue_cancel_all)
    app.router.add_post("/api/training/queue/abort-after-current", handle_queue_abort_after_current)
    app.router.add_post("/api/training/queue/force-abort", handle_queue_force_abort)
    app.router.add_post("/api/training/queue/cancel-waiting", handle_queue_cancel_waiting)
    app.router.add_post("/api/training/queue/clear", handle_queue_clear)
    app.router.add_post("/api/training/queue/clear-completed", handle_queue_clear_completed)
    app.router.add_post("/api/training/queue/clear-canceled", handle_queue_clear_canceled)
    app.router.add_post("/api/training/queue/pause", handle_queue_pause)
    app.router.add_post("/api/training/queue/{item_id}/move", handle_queue_move)
    app.router.add_post("/api/training/queue/{item_id}/retry", handle_queue_retry)
    app.router.add_delete("/api/training/queue/{item_id}", handle_queue_cancel)
    app.router.add_get("/api/training/history", handle_history_list)
    app.router.add_post("/api/training/history/batch", handle_history_batch)
    app.router.add_get("/api/training/history/collections/settings", handle_history_collection_settings_get)
    app.router.add_put("/api/training/history/collections/settings", handle_history_collection_settings_put)
    app.router.add_get("/api/training/history/config-group/timeline", handle_config_group_timeline)
    app.router.add_get("/api/training/history/{task_id}/logs/search", handle_history_log_search)
    from web.routes.training_history_system import history_system_response
    app.router.add_get("/api/training/history/{task_id}/system", history_system_response)
    app.router.add_get("/api/training/history/{task_id}/logs", handle_history_logs)
    app.router.add_get("/api/training/history/{task_id}/logs/download", handle_history_log_download)
    app.router.add_get("/api/training/history/{task_id}/artifacts/{artifact_key}", handle_history_artifact)
    app.router.add_get("/api/training/history/{task_id}/artifacts", handle_history_artifact_manifest)
    app.router.add_get("/api/training/history/{task_id}", handle_history_detail)
    app.router.add_get("/api/training/history/{task_id}/resume-options", handle_history_resume_options)
    app.router.add_patch("/api/training/history/{task_id}", handle_history_update)
    app.router.add_delete("/api/training/history/{task_id}", handle_history_delete)
    app.router.add_get("/ws/training", handle_ws)


async def handle_preflight(request: web.Request) -> web.Response:
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    config_file = str(data.get("config_file") or "").strip() or None
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response(_cli_only_spd_payload(variant, preset, methods_subdir), status=400)
    try:
        result = preflight_training_config(
            variant,
            preset,
            methods_subdir,
            config_file=config_file,
            world_size=_preflight_world_size(data.get("gpu_whitelist")),
            gpu_whitelist=data.get("gpu_whitelist"),
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({
            "ok": False,
            "variant": variant,
            "preset": preset,
            "methods_subdir": methods_subdir,
            "summary": {"errors": 1, "warnings": 0, "checks": 1},
            "checks": [{
                "level": "error",
                "key": "preflight",
                "message": f"预检测失败: {e}",
            }],
            "errors": [{
                "level": "error",
                "key": "preflight",
                "message": f"预检测失败: {e}",
            }],
            "warnings": [],
        }, status=400)


async def handle_continue_lora_inspect(request: web.Request) -> web.Response:
    data = await request.json()
    path = str(data.get("path") or "").strip()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    config_file = str(data.get("config_file") or "").strip() or None
    try:
        return web.json_response(inspect_continue_lora_weight(
            path,
            variant=variant,
            preset=preset,
            methods_subdir=methods_subdir,
            config_file=config_file,
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_start(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    extra_args = data.get("extra_args", [])
    gpu_whitelist = data.get("gpu_whitelist")
    config_file = str(data.get("config_file") or "").strip() or None
    confirmed = bool(data.get("confirmed", False))
    confirm_preprocess = bool(data.get("confirm_preprocess", False))
    continue_info = _continue_lora_info_from_request(data)
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response({"ok": False, "error": _cli_only_spd_message()}, status=400)
    try:
        preflight = preflight_training_config(
            variant,
            preset,
            methods_subdir,
            config_file=config_file,
            world_size=_preflight_world_size(gpu_whitelist),
            gpu_whitelist=gpu_whitelist,
        )
        remaining, extra_args = acknowledge_request_risks(data, preflight, extra_args)
        if not remaining.get("ok", False):
            return web.json_response({
                "ok": False,
                "error": "预检测发现错误，已阻止训练启动",
                "preflight": preflight,
            }, status=400)
        needs_preprocess = not config_file or not is_web_runtime_config(config_file)
        if not confirmed or (needs_preprocess and not confirm_preprocess):
            return web.json_response({
                "ok": False,
                "error": "请先确认训练前预检测结果",
                "preflight": preflight,
                "requires_confirmation": True,
                "requires_preprocess_confirmation": needs_preprocess,
            }, status=409)
        if needs_preprocess:
            await svc.start_preprocess(
                variant,
                preset,
                methods_subdir,
                extra_args,
                True,
                gpu_whitelist=gpu_whitelist,
                config_file=config_file,
                continue_info=continue_info,
            )
            return web.json_response({
                "ok": True,
                "job": "preprocess",
                "train_after": True,
                "message": "预处理已启动，完成后会自动开始训练",
            })
        await svc.start(
            variant,
            preset,
            extra_args,
            methods_subdir,
            gpu_whitelist=gpu_whitelist,
            config_file=config_file,
            continue_info=continue_info,
            use_runtime_dir=False,
        )
        return web.json_response({"ok": True, "job": "training", "train_after": False, "message": "训练已启动"})
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_resume(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    task_id = str(data.get("task_id") or "").strip()
    checkpoint = str(data.get("checkpoint") or "").strip()
    duration_overrides = data.get("duration_overrides")
    gpu_whitelist = data.get("gpu_whitelist")
    if not task_id:
        return web.json_response({"ok": False, "error": "缺少 task_id"}, status=400)
    try:
        payload = await svc.resume_from_history_task(
            task_id,
            checkpoint or None,
            duration_overrides=duration_overrides,
            gpu_whitelist=gpu_whitelist,
        )
        return web.json_response(payload)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"删除运行缓存失败: {e}"}, status=500)
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


async def handle_preprocess(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    extra_args = data.get("extra_args", [])
    train_after = bool(data.get("train_after", False))
    confirmed = bool(data.get("confirmed", False))
    confirm_train_after = bool(
        data.get("confirm_train_after", False)
        or data.get("confirm_preprocess", False)
    )
    gpu_whitelist = data.get("gpu_whitelist")
    config_file = str(data.get("config_file") or "").strip() or None
    continue_info = _continue_lora_info_from_request(data)
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response({"ok": False, "error": _cli_only_spd_message()}, status=400)
    try:
        preflight = preflight_training_config(
            variant,
            preset,
            methods_subdir,
            config_file=config_file,
            world_size=_preflight_world_size(gpu_whitelist),
            gpu_whitelist=gpu_whitelist,
        )
    except Exception as e:
        return web.json_response({"ok": False, "error": f"预处理预检测失败: {e}"}, status=400)
    remaining, extra_args = acknowledge_request_risks(data, preflight, extra_args)
    if not _preflight_allows_preprocess(remaining):
        return web.json_response({
            "ok": False,
            "error": "当前配置还有预处理无法自动解决的问题，请先修正预检测错误",
            "preflight": preflight,
        }, status=400)
    if train_after and (not confirmed or not confirm_train_after):
        return web.json_response({
            "ok": False,
            "error": "请先确认预处理完成后自动开始训练",
            "preflight": preflight,
            "requires_confirmation": True,
            "requires_preprocess_confirmation": True,
            "requires_train_after_confirmation": True,
        }, status=409)
    try:
        await svc.start_preprocess(
            variant,
            preset,
            methods_subdir,
            extra_args,
            train_after,
            gpu_whitelist=gpu_whitelist,
            config_file=config_file,
            continue_info=continue_info,
        )
        message = "预处理已启动，完成后会自动开始训练" if train_after else "预处理已启动"
        return web.json_response({"ok": True, "message": message})
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


def _preflight_allows_preprocess(result: dict) -> bool:
    checks = result.get("checks") or []
    errors = result.get("errors") or []
    allowed_error_keys = {"training_images", "resized_image_dir"}
    if any(item.get("key") not in allowed_error_keys for item in errors):
        return False
    return any(
        item.get("key") == "source_image_dir" and item.get("level") == "ok"
        for item in checks
    )


def _continue_lora_info_from_request(data: dict) -> dict | None:
    path = str(data.get("continue_from_weight_abs_path") or "").strip()
    if not path:
        return None
    return {
        "continue_from_weight_abs_path": path,
        "continue_from_weight_name": str(data.get("continue_from_weight_name") or "").strip(),
        "continue_from_weight_kind": str(data.get("continue_from_weight_kind") or "").strip(),
    }


def _is_cli_only_spd(variant: str, methods_subdir: str) -> bool:
    return str(methods_subdir or "") == "methods" and str(variant or "") == "spd"


def _cli_only_spd_message() -> str:
    return "SPD 是 scripts/distill_spd.py 使用的 CLI 实验配置，当前 Web 训练入口不会用 train.py 启动它。请通过 tasks.py exp-spd 或对应 CLI 流程运行。"


def _cli_only_spd_payload(variant: str, preset: str, methods_subdir: str) -> dict:
    message = _cli_only_spd_message()
    item = {"level": "error", "key": "spd", "message": message}
    return {
        "ok": False,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "summary": {"errors": 1, "warnings": 0, "checks": 1},
        "checks": [item],
        "errors": [item],
        "warnings": [],
    }


def _preflight_requiring_preprocess(result: dict) -> dict:
    checks = list(result.get("checks") or [])
    item = {
        "level": "error",
        "key": "runtime_preprocess",
        "message": "Web 训练会写入独立运行目录；请先点击“开始预处理”，完成后会自动训练。",
    }
    checks.append(item)
    errors = list(result.get("errors") or [])
    errors.append(item)
    summary = dict(result.get("summary") or {})
    summary["errors"] = len(errors)
    summary["checks"] = len(checks)
    return {
        **result,
        "ok": False,
        "summary": summary,
        "checks": checks,
        "errors": errors,
    }


async def handle_stop(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    expected_task_id = request.query.get("task_id")
    if request.can_read_body:
        try:
            payload = await request.json()
        except (ValueError, TypeError) as exc:
            raise web.HTTPBadRequest(text="停止请求必须是包含 task_id 的 JSON 对象") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="停止请求必须是包含 task_id 的 JSON 对象")
        body_task_id = payload.get("task_id")
        if not isinstance(body_task_id, str) or not body_task_id.strip():
            raise web.HTTPBadRequest(text="task_id 必须是非空字符串")
        if expected_task_id is not None and expected_task_id != body_task_id:
            raise web.HTTPBadRequest(text="查询参数与请求正文中的 task_id 不一致")
        expected_task_id = body_task_id
    if expected_task_id is not None and not expected_task_id.strip():
        raise web.HTTPBadRequest(text="task_id 必须是非空字符串")
    try:
        if expected_task_id:
            await svc.stop(expected_task_id=expected_task_id)
        else:
            await svc.stop()
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc)) from exc
    return web.json_response({"ok": True, "message": "训练已停止", "task_id": expected_task_id})


async def handle_status(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(svc.get_status_snapshot())


def _check_task_query(request: web.Request, svc) -> str | None:
    expected = request.query.get("task_id")
    if not expected:
        return None
    current = (svc.get_status_snapshot() or {}).get("task_id")
    if current != expected:
        raise web.HTTPConflict(text="当前训练任务已发生变化，请刷新后重试")
    return expected


async def handle_metrics(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    _check_task_query(request, svc)
    if request.query.get("all") == "1":
        return web.json_response(svc.get_metrics_history(include_all=True))
    return web.json_response(svc.get_metrics_history())


async def handle_logs(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = _check_task_query(request, svc)
    try:
        after = int(request.query.get("after", "0") or 0)
        limit = int(request.query.get("limit", "1000") or 1000)
    except (TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text="after 和 limit 必须是整数") from exc
    return web.json_response({
        "records": svc.get_log_records(after=after, limit=limit),
        **({"task_id": task_id} if task_id else {}),
    })


async def handle_gpus(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response({"ok": True, **await svc.gpu_inventory(force=request.query.get("refresh") == "1")})


async def handle_queue_status(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(svc.get_queue_snapshot())


async def handle_queue_post(request: web.Request) -> web.Response:
    data = await request.json()
    if isinstance(data.get("items"), list):
        return await _handle_queue_batch_start_data(request, data)
    return await _handle_queue_start_data(request, data)


async def handle_queue_start(request: web.Request) -> web.Response:
    data = await request.json()
    return await _handle_queue_start_data(request, data)


async def _handle_queue_start_data(request: web.Request, data: dict) -> web.Response:
    svc = request.app["training_service"]
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    extra_args = data.get("extra_args", [])
    gpu_whitelist = data.get("gpu_whitelist")
    config_file = str(data.get("config_file") or "").strip() or None
    confirmed = bool(data.get("confirmed", False))
    confirm_preprocess = bool(
        data.get("confirm_preprocess", False)
        or data.get("confirm_train_after", False)
    )
    start_paused = bool(data.get("start_paused", False))
    continue_info = _continue_lora_info_from_request(data)
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response({"ok": False, "error": _cli_only_spd_message()}, status=400)
    try:
        preflight = preflight_training_config(
            variant,
            preset,
            methods_subdir,
            config_file=config_file,
            world_size=_preflight_world_size(gpu_whitelist),
            gpu_whitelist=gpu_whitelist,
        )
        remaining, extra_args = acknowledge_request_risks(data, preflight, extra_args)
        needs_preprocess = not config_file or not is_web_runtime_config(config_file)
        if not remaining.get("ok", False):
            if not (needs_preprocess and confirm_preprocess and _preflight_allows_preprocess(remaining)):
                return web.json_response({
                    "ok": False,
                    "error": "预检测发现错误，已阻止加入队列",
                    "preflight": preflight,
                }, status=400)
        if not confirmed or (needs_preprocess and not confirm_preprocess):
            return web.json_response({
                "ok": False,
                "error": "请先确认训练前预检测结果",
                "preflight": preflight,
                "requires_confirmation": True,
                "requires_preprocess_confirmation": needs_preprocess,
            }, status=409)
        item_auto_retry = data.get("auto_retry") if "auto_retry" in data else None
        item_max_attempts = data.get("max_attempts") if "max_attempts" in data else None
        item_retry_backoff = data.get("retry_backoff_sec") if "retry_backoff_sec" in data else None
        payload = await svc.enqueue_training(
            variant,
            preset,
            methods_subdir,
            extra_args=extra_args,
            config_file=config_file,
            gpu_whitelist=gpu_whitelist,
            continue_info=continue_info,
            requires_preprocess=needs_preprocess,
            start_paused=start_paused,
            auto_retry=item_auto_retry,
            max_attempts=int(item_max_attempts) if item_max_attempts is not None else None,
            retry_backoff_sec=float(item_retry_backoff) if item_retry_backoff is not None else None,
        )
        return web.json_response(payload)
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_batch_start(request: web.Request) -> web.Response:
    data = await request.json()
    return await _handle_queue_batch_start_data(request, data)


async def _handle_queue_batch_start_data(request: web.Request, data: dict) -> web.Response:
    svc = request.app["training_service"]
    items = data.get("items", [])
    if not isinstance(items, list) or not items:
        return web.json_response({"ok": False, "error": "items 必须是非空数组"}, status=400)
    default_preset = str(data.get("preset") or "default").strip() or "default"
    gpu_whitelist = data.get("gpu_whitelist")
    start_paused = bool(data.get("start_paused", True))
    prepared: list[dict] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            return web.json_response({
                "ok": False,
                "error": "队列项格式不合法",
                "failed_index": index,
            }, status=400)
        variant = str(raw.get("variant") or "").strip()
        preset = str(raw.get("preset") or default_preset).strip() or "default"
        methods_subdir = str(raw.get("methods_subdir") or "gui-methods").strip() or "gui-methods"
        config_file = str(raw.get("config_file") or "").strip() or None
        if not variant:
            return web.json_response({
                "ok": False,
                "error": "缺少 variant",
                "failed_index": index,
                "failed_item": raw,
            }, status=400)
        if _is_cli_only_spd(variant, methods_subdir):
            return web.json_response({
                "ok": False,
                "error": _cli_only_spd_message(),
                "failed_index": index,
                "failed_item": raw,
            }, status=400)
        try:
            preflight = preflight_training_config(
                variant,
                preset,
                methods_subdir,
                config_file=config_file,
                world_size=_preflight_world_size(gpu_whitelist),
                gpu_whitelist=gpu_whitelist,
            )
        except Exception as e:
            return web.json_response({
                "ok": False,
                "error": f"预检测失败: {e}",
                "failed_index": index,
                "failed_item": raw,
            }, status=400)
        remaining, extra_args = acknowledge_request_risks(raw, preflight, raw.get("extra_args") if isinstance(raw.get("extra_args"), list) else [])
        requires_preprocess = not config_file or not is_web_runtime_config(config_file)
        confirm_preprocess = bool(
            raw.get("confirm_preprocess", True)
            or raw.get("confirm_train_after", False)
        )
        if not remaining.get("ok", False):
            if not (requires_preprocess and confirm_preprocess and _preflight_allows_preprocess(remaining)):
                return web.json_response({
                    "ok": False,
                    "error": "预检测发现错误，已阻止批量加入队列",
                    "failed_index": index,
                    "failed_item": raw,
                    "preflight": preflight,
                }, status=400)
        prepared.append({
            "variant": variant,
            "preset": preset,
            "methods_subdir": methods_subdir,
            "config_file": config_file,
            "extra_args": extra_args,
            "requires_preprocess": requires_preprocess,
            "continue_info": _continue_lora_info_from_request(raw),
            "label": str(raw.get("label") or raw.get("filename") or "").strip(),
        })
    try:
        return web.json_response(await svc.enqueue_training_batch(
            prepared,
            default_preset=default_preset,
            gpu_whitelist=gpu_whitelist,
            start_paused=start_paused,
        ))
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_resume(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    task_id = str(data.get("task_id") or "").strip()
    checkpoint = str(data.get("checkpoint") or "").strip()
    duration_overrides = data.get("duration_overrides")
    gpu_whitelist = data.get("gpu_whitelist")
    if not task_id:
        return web.json_response({"ok": False, "error": "缺少 task_id"}, status=400)
    try:
        return web.json_response(await svc.enqueue_resume_from_history_task(
            task_id,
            checkpoint or None,
            duration_overrides=duration_overrides,
            gpu_whitelist=gpu_whitelist,
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_move(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    direction = str(data.get("direction") or "").strip()
    if direction not in {"up", "down", "top", "bottom"}:
        return web.json_response({"ok": False, "error": "direction 必须是 up、down、top 或 bottom"}, status=400)
    try:
        return web.json_response(await svc.move_queue_item(request.match_info["item_id"], direction))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_retry(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        return web.json_response(await svc.retry_queue_item(request.match_info["item_id"]))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_cancel_waiting(request: web.Request) -> web.Response:
    return await queue_command_response(request, "cancel_waiting_queue_items")


async def handle_queue_cancel_all(request: web.Request) -> web.Response:
    return await queue_command_response(request, "cancel_all_queue_items")


async def handle_queue_abort_after_current(request: web.Request) -> web.Response:
    return await queue_command_response(request, "abort_queue_after_current")


async def handle_queue_force_abort(request: web.Request) -> web.Response:
    return await queue_command_response(request, "force_abort_queue")


async def handle_queue_clear(request: web.Request) -> web.Response:
    return await queue_command_response(request, "clear_finished_queue_items")


async def handle_queue_clear_completed(request: web.Request) -> web.Response:
    return await queue_command_response(request, "clear_completed_queue_items")


async def handle_queue_clear_canceled(request: web.Request) -> web.Response:
    return await queue_command_response(request, "clear_canceled_queue_items")


async def handle_queue_settings(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    paused = data.get("paused") if "paused" in data else None
    failure_policy = data.get("failure_policy") if "failure_policy" in data else None
    auto_retry = data.get("auto_retry") if "auto_retry" in data else None
    max_attempts = data.get("max_attempts") if "max_attempts" in data else None
    retry_backoff_sec = data.get("retry_backoff_sec") if "retry_backoff_sec" in data else None
    try:
        return web.json_response(await svc.set_queue_settings(
            paused=bool(paused) if paused is not None else None,
            failure_policy=str(failure_policy) if failure_policy is not None else None,
            auto_retry=bool(auto_retry) if auto_retry is not None else None,
            max_attempts=int(max_attempts) if max_attempts is not None else None,
            retry_backoff_sec=float(retry_backoff_sec) if retry_backoff_sec is not None else None,
        ))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_cancel(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        data = await request.json()
    except Exception:
        data = {}
    try:
        return web.json_response(await svc.cancel_queue_item(
            request.match_info["item_id"],
            delete_runtime=bool(data.get("delete_runtime")),
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"删除运行缓存失败: {e}"}, status=500)


async def handle_queue_pause(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    paused = bool(data.get("paused", False))
    return web.json_response(await svc.set_queue_paused(paused))


async def handle_history_list(request: web.Request) -> web.Response:
    return await history_list_response(request)


async def handle_history_batch(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        data = await request.json()
        return web.json_response(svc.batch_update_history_tasks(data))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_collection_settings_get(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(svc.get_history_collection_settings())


async def handle_history_collection_settings_put(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        data = await request.json()
        return web.json_response(svc.save_history_collection_settings(data))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"保存集合设置失败: {e}"}, status=500)


def _positive_query_int(value) -> int | None:
    try:
        number = int(str(value or "").strip())
    except ValueError:
        return None
    return number if number > 0 else None


async def handle_history_detail(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        payload = svc.get_history_task(task_id)
        include_logs = str(request.query.get("include_logs") or "1").lower() not in {"0", "false", "no"}
        if not include_logs:
            payload["logs"] = []
            limits = payload.setdefault("limits", {})
            limits["logs_returned"] = 0
            limits["logs_truncated"] = bool(limits.get("logs_total"))
        return web.json_response(payload)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_logs(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    offset_value = request.query.get("offset")
    try:
        offset = int(offset_value) if offset_value is not None else None
        limit = _positive_query_int(request.query.get("limit"))
        if offset is not None and offset < 0:
            raise ValueError("日志偏移量不能小于 0")
        return web.json_response(await asyncio.to_thread(svc.get_history_log_page, task_id, offset=offset, limit=limit))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_log_search(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        cursor = int(request.query.get("cursor") or 0)
        return web.json_response(await asyncio.to_thread(svc.find_history_log_match,
            task_id,
            query=request.query.get("query") or "",
            cursor=cursor,
            direction=request.query.get("direction") or "forward",
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_log_download(request: web.Request) -> web.StreamResponse:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        path = svc.get_history_log_path(task_id)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(f'{task_id}.logs.jsonl')}",
    }
    return web.FileResponse(path, headers=headers)


async def handle_history_artifact(request: web.Request) -> web.StreamResponse:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    artifact_key = request.match_info["artifact_key"]
    try:
        path = svc.get_history_artifact_path(task_id, artifact_key)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    disposition = "attachment" if str(request.query.get("download") or "").lower() in {"1", "true", "yes"} else "inline"
    headers = {
        "Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(path.name)}",
        "Content-Type": "text/plain; charset=utf-8",
    }
    return web.FileResponse(path, headers=headers)


async def handle_config_group_timeline(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    methods_subdir = str(request.query.get("methods_subdir") or "").strip()
    variant = str(request.query.get("variant") or "").strip()
    preset = str(request.query.get("preset") or "default").strip() or "default"
    group_key = str(request.query.get("group_key") or "").strip()
    include_archived = str(request.query.get("include_archived") or "0").lower() in {"1", "true", "yes"}
    task_ids = _timeline_task_ids_from_query(request)
    if not task_ids and not group_key and (not methods_subdir or not variant):
        return web.json_response({"ok": False, "error": "缺少 group_key 或 methods_subdir/variant"}, status=400)
    try:
        return web.json_response(svc.get_config_group_timeline(
            methods_subdir,
            variant,
            preset,
            group_key=group_key,
            include_archived=include_archived,
            task_ids=task_ids,
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


def _timeline_task_ids_from_query(request: web.Request) -> list[str]:
    values: list[str] = []
    for key in ("task_id", "task_ids"):
        values.extend(request.query.getall(key, []))
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value or "").split(","):
            task_id = part.strip()
            if not task_id or task_id in seen:
                continue
            out.append(task_id)
            seen.add(task_id)
    return out


async def handle_history_resume_options(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        return web.json_response(svc.get_resume_options(task_id))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_update(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        data = await request.json()
        return web.json_response(svc.update_history_task(task_id, data))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_delete(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        return web.json_response(svc.delete_history_task(task_id))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    svc = request.app["training_service"]
    svc.subscribe(ws)

    try:
        async for msg in ws:
            pass
    finally:
        svc.unsubscribe(ws)

    return ws
