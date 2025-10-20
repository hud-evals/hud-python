"""Setup tools for the sheet environment."""

from __future__ import annotations
from hud.tools import BaseHub

setup = BaseHub("setup")

from .load_spreadsheet import load_spreadsheet

__all__ = ["setup"]
