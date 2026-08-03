"""Shared Task resolution and placement for lifecycle CLIs."""

from __future__ import annotations

import ast
import json
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class TaskResolutionError(ValueError):
    """The requested Task or local source cannot be resolved."""


def parse_task_args(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise TaskResolutionError(f"--args must be valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise TaskResolutionError("--args must be a JSON object")
    return parsed


def collect_taskset(source: str) -> Any:
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
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr
            if isinstance(callee, ast.Attribute)
            else None
        )
        if name == "Environment":
            return True
    return False


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
) -> tuple[Any, Path]:
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
    return selected, Path(source)


def attached_task(task: str, args: dict[str, Any]) -> Any:
    from hud.eval import Task

    env_name = task.split(":", 1)[0] if ":" in task else "attached"
    return Task(env=env_name, id=task, args=args)


def normalize_control_url(value: str) -> str:
    parts = urlsplit(value if "://" in value else f"tcp://{value}")
    if parts.scheme != "tcp":
        raise TaskResolutionError("--url must use the tcp:// control-channel scheme")
    if parts.hostname is None:
        raise TaskResolutionError("--url must include a host")
    return f"tcp://{parts.hostname}:{parts.port or 8765}"


__all__ = [
    "TaskResolutionError",
    "attached_task",
    "collect_taskset",
    "find_local_env_url",
    "normalize_control_url",
    "parse_task_args",
    "select_local_task",
    "spawn_target",
]
