"""Resolve ``source_framework`` from a local environment tree.

v1: ``Dockerfile.hud`` → ``"hud"``. Callers treat ``None`` as failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

SourceFrameworkName = Literal["hud"]

_HUD_MARKER = "Dockerfile.hud"


def resolve_source_framework(path: Path | str) -> SourceFrameworkName | None:
    """Return ``"hud"`` if ``Dockerfile.hud`` exists under *path*, else ``None``."""
    root = Path(path).expanduser().resolve()
    if (root / _HUD_MARKER).is_file():
        return "hud"
    return None


__all__ = [
    "SourceFrameworkName",
    "resolve_source_framework",
]
