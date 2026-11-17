"""Merge Web launch overrides with the trainer's argparse rules, without IO."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any

# The Web request owns these selectors and operations. Allowing them in argv
# would make the worker load a different file or exit/write before training.
_WEB_OWNED = frozenset({
    "config_file", "method", "preset", "methods_subdir", "help",
    "print_config", "output_config", "config_snapshot", "config_strict",
})


def merge_training_extra_args(
    cfg: dict[str, Any], extra_args: Any,
) -> dict[str, Any]:
    """Apply explicit argv values, including false/zero and repeated options.

    Web argv requires full option names; the standalone CLI keeps its existing
    abbreviation policy. No config file is read or written by this function.
    """
    if extra_args is None:
        return dict(cfg)
    if not isinstance(extra_args, list) or any(
        not isinstance(arg, str) for arg in extra_args
    ):
        raise ValueError("extra_args 必须是字符串数组")
    if not extra_args:
        return dict(cfg)

    from library.training.cli_args import setup_parser

    parser = setup_parser()
    parser.allow_abbrev = False

    def fail(message=None, **_kwargs):
        raise ValueError(f"训练附加参数无效: {message or '不能退出参数检查'}")

    parser.error = fail
    parser.exit = lambda status=0, message=None: fail(message)
    provided: set[str] = set()
    for arg in extra_args:
        action = parser._option_string_actions.get(arg.split("=", 1)[0])
        if action is None:
            continue
        if action.dest in _WEB_OWNED:
            raise ValueError(
                f"extra_args 不能覆盖 {action.dest}；请使用 WebUI 的配置选择入口"
            )
        provided.add(action.dest)
    parsed = vars(parser.parse_args(extra_args, namespace=argparse.Namespace(**deepcopy(cfg))))
    # Keep sparse configs sparse; do not write every argparse default back into
    # runtime TOML. Explicitly supplied defaults must still override old values.
    return {key: parsed[key] for key in set(cfg) | provided}


def without_materialized_dataset_arg(extra_args: list[str] | None) -> list[str]:
    """A generated runtime dataset replaces the source CLI dataset path."""
    args = list(extra_args or [])
    result = []
    index = 0
    while index < len(args):
        if args[index] == "--dataset_config":
            index += 2
        elif args[index].startswith("--dataset_config="):
            index += 1
        else:
            result.append(args[index])
            index += 1
    return result
