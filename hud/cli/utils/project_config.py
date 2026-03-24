"""Project-level ``.hud/config.json`` management.

Stores only IDs — names are resolved at command time, never persisted.

Schema::

    {"registryId": "abc123-...", "tasksetId": "def456-...", "syncEnv": true}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"
LEGACY_FILENAME = "deploy.json"
HUD_DIR = ".hud"


def _find_config_dir(directory: Path | None = None) -> Path:
    """Return the ``.hud/`` directory for the given project directory."""
    base = (directory or Path.cwd()).resolve()
    return base / HUD_DIR


def load_project_config(directory: Path | None = None) -> dict[str, Any]:
    """Load project config from ``.hud/config.json``.

    Falls back to ``.hud/deploy.json`` for migration. Returns empty dict
    if neither exists.
    """
    hud_dir = _find_config_dir(directory)
    config_path = hud_dir / CONFIG_FILENAME
    legacy_path = hud_dir / LEGACY_FILENAME

    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Failed to parse %s, returning empty config", config_path)
            return {}

    if legacy_path.exists():
        try:
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        _migrate_legacy(legacy_path, config_path, data)
        return data

    return {}


def save_project_config(
    data: dict[str, Any],
    directory: Path | None = None,
) -> Path | None:
    """Merge ``data`` into ``.hud/config.json`` and return the path.

    Only updates the keys present in ``data``; existing keys are preserved.
    Returns None if nothing changed (all values already match).
    """
    hud_dir = _find_config_dir(directory)
    config_path = hud_dir / CONFIG_FILENAME

    existing = load_project_config(directory)
    merged = {**existing, **data}

    if merged == existing and config_path.exists():
        return None

    hud_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(merged, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def get_registry_id(directory: Path | None = None) -> str | None:
    """Read the stored registry ID from project config."""
    return load_project_config(directory).get("registryId")


def get_taskset_id(directory: Path | None = None) -> str | None:
    """Read the stored taskset ID from project config."""
    return load_project_config(directory).get("tasksetId")


def _migrate_legacy(legacy_path: Path, config_path: Path, data: dict[str, Any]) -> None:
    """Migrate ``.hud/deploy.json`` to ``.hud/config.json``."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        legacy_path.unlink()
        LOGGER.info("Migrated .hud/deploy.json → .hud/config.json")
    except Exception as e:
        LOGGER.warning("Failed to migrate deploy.json → config.json: %s", e)
