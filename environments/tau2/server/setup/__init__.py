"""Setup layer"""

from __future__ import annotations
from hud.tools.base import BaseHub

setup = BaseHub("setup")

from . import load

__all__ = ["setup"]