"""Shared helpers for Harbor task integration."""

from __future__ import annotations

import hashlib
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """A valid env name (lowercase ``[a-z0-9-]``) from a dataset dir name."""
    normalized = re.sub(r"[^a-z0-9-]", "", name.strip().lower().replace(" ", "-").replace("_", "-"))
    return re.sub(r"-+", "-", normalized).strip("-") or "harbor"


def _is_harbor_task(path: Path) -> bool:
    return path.is_dir() and (path / "task.toml").exists() and (path / "instruction.md").exists()


def _task_dirs(path: str | Path) -> list[Path]:
    root = Path(path)
    if _is_harbor_task(root):
        return [root]
    if root.is_dir():
        return sorted(d for d in root.iterdir() if d.is_dir() and _is_harbor_task(d))
    return []


def _hash_directory(path: Path) -> str:
    """Content-hash a directory for grouping tasks by identical environments."""
    hasher = hashlib.sha256()
    if not path.exists():
        return "empty"
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            hasher.update(str(file_path.relative_to(path)).encode())
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class _HarborTask:
    """One parsed Harbor task dir."""

    task_id: str
    config: dict[str, Any]
    env_hash: str


def _parse_task(task_dir: Path) -> _HarborTask | None:
    if not (task_dir / "instruction.md").is_file():
        LOGGER.warning("failed to read instruction.md in %s", task_dir)
        return None
    try:
        config: dict[str, Any] = tomllib.loads((task_dir / "task.toml").read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        LOGGER.warning("failed to parse task.toml in %s", task_dir)
        config = {}
    env_dir = task_dir / "environment"
    return _HarborTask(
        task_id=task_dir.name,
        config=config,
        env_hash=_hash_directory(env_dir) if env_dir.exists() else "no-env",
    )
