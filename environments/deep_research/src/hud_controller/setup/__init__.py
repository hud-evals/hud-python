"""Setup layer for deep research environment."""

from __future__ import annotations

from hud.tools.base import BaseHub

setup = BaseHub("setup")

from . import navigate  # noqa: E402

__all__ = ["setup"]

