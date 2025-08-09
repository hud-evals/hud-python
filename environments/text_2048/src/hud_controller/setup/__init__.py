"""Setup layer for 2048 environment.

This module exposes:
- ``setup_hub`` – the BaseHub instance for setup operations
"""

from __future__ import annotations

from hud.tools.base import BaseHub

setup = BaseHub("setup")

# Import all setup functions to register them
from . import board

__all__ = ["setup"]
