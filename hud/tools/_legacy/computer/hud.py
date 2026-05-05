"""Legacy HUD computer import path."""

from __future__ import annotations

from hud.tools.computer import ComputerTool


class HudComputerTool(ComputerTool):
    """Compatibility shim for the old public HUD computer tool name."""


__all__ = ["HudComputerTool"]
