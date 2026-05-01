"""Compatibility shims for old memory tool names."""

from __future__ import annotations

from hud.tools.memory import MemoryCommand, MemoryTool

ClaudeMemoryCommand = MemoryCommand


class ClaudeMemoryTool(MemoryTool):
    """Compatibility shim for old Claude memory environment registrations."""


class GeminiMemoryTool(MemoryTool):
    """Compatibility shim for old Gemini memory environment registrations."""

    def __init__(
        self,
        memory_dir: str = ".",
        memory_filename: str = "GEMINI.md",
    ) -> None:
        del memory_filename
        super().__init__(memories_dir=memory_dir)


__all__ = ["ClaudeMemoryCommand", "ClaudeMemoryTool", "GeminiMemoryTool"]
