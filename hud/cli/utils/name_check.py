"""Check environment/taskset name mismatches between local code and platform.

Used by ``hud deploy``, ``hud sync tasks``, and ``hud sync env`` to detect
when local ``Environment("old-name")`` references don't match the deployed
environment name.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path  # noqa: TC003 — runtime use

import httpx

LOGGER = logging.getLogger(__name__)

ENV_NAME_PATTERN = re.compile(r'Environment\(["\']([^"\']+)["\']\)')


def resolve_registry_name(
    registry_id: str,
    api_url: str,
    headers: dict[str, str],
) -> str | None:
    """Fetch the current name for a registry ID from the platform."""
    try:
        resp = httpx.get(
            f"{api_url}/registry/envs/{registry_id}",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("name_display") or data.get("name")
    except Exception:
        return None


def find_env_name_references(
    directory: Path,
) -> list[tuple[Path, int, str, str]]:
    """Scan Python files for Environment("name") references.

    Returns list of (file_path, line_number, full_line, matched_name).
    """
    results: list[tuple[Path, int, str, str]] = []
    py_files = list(directory.glob("*.py")) + list(directory.glob("*/*.py"))

    for py_file in py_files:
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: S112
            continue
        for i, line in enumerate(lines, 1):
            results.extend(
                (py_file, i, line.strip(), match.group(1))
                for match in ENV_NAME_PATTERN.finditer(line)
            )

    return results
