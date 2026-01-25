"""Read tool for filesystem access.

Matches the `read` tool from OpenCode and similar coding agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from mcp.types import ContentBlock

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.native_types import NativeToolSpecs
from hud.tools.types import ContentResult, ToolError


class ReadTool(BaseTool):
    """Read file contents with optional line range support.

    This tool provides read-only access to files, matching the `read` tool
    from OpenCode and similar coding agents.

    Parameters:
        file_path: Path to the file to read (required)
        start_line: Optional starting line (1-indexed)
        end_line: Optional ending line (1-indexed, inclusive)

    Example:
        >>> tool = ReadTool(base_path="./workspace")
        >>> result = await tool(file_path="src/main.py")
        >>> result = await tool(file_path="src/main.py", start_line=10, end_line=20)
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    _base_path: Path
    _max_lines: int

    def __init__(
        self,
        base_path: str = ".",
        max_lines: int = 10000,
    ) -> None:
        """Initialize ReadTool.

        Args:
            base_path: Base directory for relative paths
            max_lines: Maximum lines to return (default: 10000)
        """
        super().__init__(
            env=None,
            name="read",
            title="Read File",
            description=(
                "Read file contents. Use start_line/end_line for large files. "
                "Returns file content with line numbers."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_lines = max_lines

    async def __call__(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> list[ContentBlock]:
        """Read file contents.

        Args:
            file_path: Path to the file to read
            start_line: Optional starting line (1-indexed)
            end_line: Optional ending line (1-indexed, inclusive)

        Returns:
            List of ContentBlocks with file contents
        """
        if not file_path:
            raise ToolError("file_path is required")

        path = resolve_path_safely(file_path, self._base_path)

        if not path.exists():
            raise ToolError(f"File not found: {file_path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"Cannot read binary file: {file_path}")
        except PermissionError:
            raise ToolError(f"Permission denied: {file_path}")

        lines = content.split("\n")
        total_lines = len(lines)

        # Apply line range if specified
        if start_line is not None or end_line is not None:
            start = max(0, (start_line or 1) - 1)  # Convert to 0-indexed
            end = min(total_lines, end_line or total_lines)
            lines = lines[start:end]
            line_offset = start + 1
        else:
            line_offset = 1

        # Truncate if too many lines
        if len(lines) > self._max_lines:
            lines = lines[: self._max_lines]
            truncated = True
        else:
            truncated = False

        # Format with line numbers
        numbered_lines = [
            f"{i + line_offset:4d} | {line}" for i, line in enumerate(lines)
        ]
        output = "\n".join(numbered_lines)

        # Add metadata
        header = f"File: {file_path} ({total_lines} lines total)"
        if start_line or end_line:
            header += f" [showing lines {line_offset}-{line_offset + len(lines) - 1}]"
        if truncated:
            header += f" [truncated to {self._max_lines} lines]"

        return ContentResult(output=f"{header}\n\n{output}").to_content_blocks()


__all__ = ["ReadTool"]
