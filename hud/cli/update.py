"""Warn on stderr when a newer hud is on PyPI.

Skipped for help/version, in CI, and when ``HUD_SKIP_VERSION_CHECK`` is set.
A failed check never blocks a command.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
from packaging.version import parse as parse_version

from hud import __version__

_CACHE = Path(".hud") / ".cache" / "version_check.json"
_TTL_S = 6 * 60 * 60
_PYPI = "https://pypi.org/pypi/hud/json"


def notify_if_outdated(argv: list[str]) -> None:
    """Print an upgrade hint when the installed hud is behind PyPI."""
    if "CI" in os.environ or os.environ.get("HUD_SKIP_VERSION_CHECK"):
        return
    if not any(not arg.startswith("-") for arg in argv[1:]):
        return
    with contextlib.suppress(Exception):
        cache = Path.home() / _CACHE
        latest: str | None = None
        try:
            data = json.loads(cache.read_text())
            if time.time() - data["checked_at"] <= _TTL_S:
                latest = data["latest"]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        fetched = latest is None
        if latest is None:
            latest = httpx.get(_PYPI, timeout=httpx.Timeout(1.0, connect=0.5)).json()["info"][
                "version"
            ]
        if parse_version(latest) > parse_version(__version__):
            tool_install = "uv/tools/" in sys.prefix.replace("\\", "/")
            in_project_venv = not tool_install and (
                sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ
            )
            upgrade = "uv sync --upgrade-package hud" if in_project_venv else "uv tool upgrade hud"
            sys.stderr.write(
                f"A new version of hud is available: {latest} (current: {__version__})\n"
                f"Run: {upgrade}\n"
            )
        if fetched:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"latest": latest, "checked_at": time.time()}))
