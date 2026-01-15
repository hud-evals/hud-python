"""Setup layer for PDF computer use environment."""

from hud.tools.base import BaseHub

setup = BaseHub("setup")

from . import load_pdf

__all__ = ["setup"]
