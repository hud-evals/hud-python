"""Setup layer for PDF environment."""

from hud.tools.base import BaseHub

setup = BaseHub("setup")

# Import setup functions to register them
from . import load_pdf

__all__ = ["setup"]
