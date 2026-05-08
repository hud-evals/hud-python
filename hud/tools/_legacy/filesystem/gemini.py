"""Gemini filesystem compatibility shims."""

from __future__ import annotations

from hud.tools._legacy.filesystem.base import GlobTool, GrepTool, ListTool, ReadTool


class GeminiReadTool(ReadTool):
    """Compatibility shim for old Gemini read_file environment registrations."""


class GeminiReadManyTool(ReadTool):
    """Compatibility shim for old Gemini read_many_files environment registrations."""

    def __init__(
        self,
        base_path: str = ".",
        max_files: int = 100,
        max_total_lines: int = 10000,
    ) -> None:
        del max_files, max_total_lines
        super().__init__(base_path=base_path)


class GeminiSearchTool(GrepTool):
    """Compatibility shim for old Gemini grep_search environment registrations."""


class GeminiGlobTool(GlobTool):
    """Compatibility shim for old Gemini glob environment registrations."""


class GeminiListTool(ListTool):
    """Compatibility shim for old Gemini list_directory environment registrations."""


__all__ = [
    "GeminiGlobTool",
    "GeminiListTool",
    "GeminiReadManyTool",
    "GeminiReadTool",
    "GeminiSearchTool",
]
