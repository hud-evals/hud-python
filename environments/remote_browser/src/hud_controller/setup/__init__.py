"""Setup layer for remote browser environment.

This module exposes:
- ``setup``, the BaseHub instance for setup operations
"""

from __future__ import annotations

from hud.tools.base import BaseHub

setup = BaseHub("setup")

# Import all setup functions to register them
from . import navigate, cookies, load_html, interact, sheets

__all__ = ["setup"]
