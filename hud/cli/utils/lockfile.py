"""Shared lock file helpers: loading, path resolution, image extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import yaml

LOCK_FILENAME = "hud.lock.yaml"


def load_lock(path: Path) -> dict[str, Any]:
    """Load and parse a hud.lock.yaml file. Raises on missing/invalid."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_lock(directory: Path) -> Path | None:
    """Find hud.lock.yaml in *directory* or its parent. Returns None if not found."""
    for candidate in [directory, directory.parent]:
        lock = candidate / LOCK_FILENAME
        if lock.exists():
            return lock
    return None


def get_local_image(lock_data: dict[str, Any]) -> str:
    """Extract the local image reference from lock data.

    Checks ``images.local`` (new format) then ``image`` (legacy).
    Returns empty string if neither exists.
    """
    return lock_data.get("images", {}).get("local") or lock_data.get("image", "")
