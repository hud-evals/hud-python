"""Compatibility shims for old coding tool names."""

from __future__ import annotations

from hud.tools._legacy.coding.apply_patch import ApplyPatchTool, DiffError
from hud.tools._legacy.coding.gemini import GeminiEditTool, GeminiShellTool, GeminiWriteTool
from hud.tools._legacy.coding.shell import ShellTool

__all__ = [
    "ApplyPatchTool",
    "DiffError",
    "GeminiEditTool",
    "GeminiShellTool",
    "GeminiWriteTool",
    "ShellTool",
]
