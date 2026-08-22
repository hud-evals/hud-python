"""Shared task resolution for CLI commands."""

from __future__ import annotations

import ast
import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from hud.eval import Task, Taskset


class TaskResolutionError(ValueError):
    """The requested task or source cannot be resolved."""


def parse_task_args(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise TaskResolutionError(f"--args must be valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise TaskResolutionError("--args must be a JSON object")
    return parsed


def collect_taskset(source: str) -> Taskset:
    from hud.eval import Taskset

    try:
        return Taskset.from_file(source)
    except (FileNotFoundError, ValueError) as exc:
        raise TaskResolutionError(str(exc)) from None


def find_local_env_url(port: int = 8765) -> str | None:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return f"tcp://127.0.0.1:{port}"
    except OSError:
        return None


def _python_defines_environment(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "Environment")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "Environment")
        )
        for node in ast.walk(tree)
    )


def spawn_target(source: str | Path) -> Path:
    resolved = Path(source).resolve()
    if resolved.is_dir():
        return resolved
    if resolved.suffix != ".py":
        return resolved.parent
    if _python_defines_environment(resolved):
        return resolved
    env_py = resolved.parent / "env.py"
    return env_py if env_py.is_file() else resolved.parent


def select_local_task(
    task: str,
    source: str,
    args: dict[str, Any],
) -> Task:
    taskset = collect_taskset(source)
    if not taskset:
        raise TaskResolutionError(f"No tasks found in {source}")
    matches = [
        candidate
        for index, (slug, candidate) in enumerate(taskset.items())
        if task in (slug, candidate.id, str(index))
    ]
    if not matches:
        available = ", ".join(sorted({candidate.id for candidate in taskset}))
        raise TaskResolutionError(f"No task matching {task!r} (available: {available})")
    selected = matches[0]
    if args:
        selected = selected.model_copy(update={"args": args})
    return selected


def normalize_control_url(value: str) -> str:
    parts = urlsplit(value if "://" in value else f"tcp://{value}")
    if parts.scheme != "tcp":
        raise TaskResolutionError("--url must use the tcp:// control-channel scheme")
    if parts.hostname is None:
        raise TaskResolutionError("--url must include a host")
    try:
        port = parts.port or 8765
    except ValueError as exc:
        raise TaskResolutionError(f"--url has an invalid port: {exc}") from None
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"tcp://{host}:{port}"


__all__ = [
    "TaskResolutionError",
    "collect_taskset",
    "find_local_env_url",
    "normalize_control_url",
    "parse_task_args",
    "select_local_task",
    "spawn_target",
]
