"""Gemini-style write_file tool implementation.

Based on Gemini CLI's write_file tool:
https://github.com/google-gemini/gemini-cli
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from mcp.types import ContentBlock  # noqa: TC002 - used at runtime by FunctionTool

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType

from .utils import resolve_path_safely, write_file_sync


class GeminiWriteTool(BaseTool):
    """Gemini CLI-style file writing tool.

    Creates or overwrites a file with the provided content.
    Creates parent directories if they don't exist.

    Parameters (matching Gemini CLI):
        file_path: Path to the file to write (required)
        content: The content to write to the file (required)

    Native specs: Uses function calling (no native API), role="writer"
                  for mutual exclusion with other write tools.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="writer"),
    }

    _base_directory: str

    def __init__(self, base_directory: str = ".") -> None:
        super().__init__(
            env=None,
            name="write_file",
            title="WriteFile",
            description=(
                "Creates a new file or overwrites an existing file with the provided content. "
                "Creates parent directories if they don't exist. "
                "Use this for creating new files. "
                "For editing existing files, prefer the replace tool."
            ),
        )
        self._base_directory = str(Path(base_directory).resolve())

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base directory with containment check."""
        return resolve_path_safely(file_path, Path(self._base_directory))

    async def __call__(
        self,
        file_path: str,
        content: str,
    ) -> list[ContentBlock]:
        """Write content to a file.

        Args:
            file_path: Path to the file to write
            content: The content to write to the file

        Returns:
            List of ContentBlocks with result message
        """
        if not file_path or not file_path.strip():
            raise ToolError("The 'file_path' parameter must be non-empty.")

        path = self._resolve_path(file_path)

        if path.exists() and path.is_dir():
            raise ToolError(f"Path is a directory: {file_path}")

        is_new = not path.exists()
        write_file_sync(path, content)

        action = "Created" if is_new else "Overwrote"
        line_count = content.count("\n") + (1 if content else 0)
        result = f"{action} file: {file_path} ({line_count} lines)"

        return ContentResult(output=result).to_content_blocks()


__all__ = ["GeminiWriteTool"]
