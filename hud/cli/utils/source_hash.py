"""Utilities to compute a fast, deterministic source hash for environments.

This intentionally focuses on the typical HUD environment layout and aims to be fast:
- Always include: Dockerfile, pyproject.toml
- Include directories: controller/, environment/, src/
- Exclude common build/runtime caches and lock files

Note: This is not a full Docker build context hash and does not parse .dockerignore.
It is sufficient to detect meaningful changes for HUD environments quickly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "dist",
    "build",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

EXCLUDE_FILE_SUFFIXES = {
    ".pyc",
    ".log",
}

EXCLUDE_FILES = {
    "hud.lock.yaml",
}

INCLUDE_FILES = {"Dockerfile", "pyproject.toml"}
INCLUDE_DIRS = {"server", "mcp", "controller", "environment"}


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield files to include in the source hash.

    The order is not guaranteed; callers should sort for deterministic hashing.
    """
    # Always include top-level files if present
    for name in INCLUDE_FILES:
        p = root / name
        if p.is_file():
            yield p

    # Include known directories
    for d in INCLUDE_DIRS:
        dp = root / d
        if not dp.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(dp):
            # prune excluded dirs in-place
            dirnames[:] = [dn for dn in dirnames if dn not in EXCLUDE_DIRS]
            for fn in filenames:
                if fn in EXCLUDE_FILES:
                    continue
                if any(fn.endswith(suf) for suf in EXCLUDE_FILE_SUFFIXES):
                    continue
                yield Path(dirpath) / fn


def list_source_files(root: Path) -> list[Path]:
    """Return a sorted list of files used for the source hash.

    Sorting is by relative path to ensure deterministic ordering.
    """
    root = root.resolve()
    files = list(iter_source_files(root))
    files.sort(key=lambda p: str(p.resolve().relative_to(root)).replace("\\", "/"))
    return files


def compute_source_hash(directory: str | Path) -> str:
    """Compute a deterministic SHA-256 hash over relevant source files.

    Args:
        directory: Environment directory root.

    Returns:
        Hex digest string.
    """
    root = Path(directory).resolve()
    files = list_source_files(root)

    hasher = hashlib.sha256()
    for p in files:
        rel = str(p.resolve().relative_to(root)).replace("\\", "/")
        hasher.update(rel.encode("utf-8"))
        with open(p, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)

    return hasher.hexdigest()
