"""Read tool for filesystem access (OpenCode-style).

Matches OpenCode's read tool specification:
https://github.com/anomalyco/opencode

Key features:
- Absolute path required for filePath
- 0-based offset, default 2000 line limit
- 5-digit zero-padded line numbers (00001|)
- Max 2000 char line length (truncated)
- Output wrapped in <file>...</file> tags
- Image support via base64
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mcp.types import ImageContent, TextContent  # noqa: TC002

from hud.tools.filesystem.base import BaseReadTool, ReadResult
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs


class ReadTool(BaseReadTool):
    """Read file contents matching OpenCode's read tool.

    Reads a file from the local filesystem with pagination support.
    Returns content with 5-digit zero-padded line numbers.

    Parameters:
        filePath: Absolute path to the file to read (required)
        offset: 0-based line number to start reading from (optional)
        limit: Number of lines to read, defaults to 2000 (optional)

    Example:
        >>> tool = ReadTool(base_path="./workspace")
        >>> result = await tool(filePath="/path/to/file.py")
        >>> result = await tool(filePath="/path/to/file.py", offset=100, limit=50)
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    def __init__(self, base_path: str = ".") -> None:
        """Initialize ReadTool.

        Args:
            base_path: Base directory for relative paths
        """
        super().__init__(
            base_path=base_path,
            name="read",
            title="Read",
            description=(
                "Reads a file from the local filesystem. The filePath parameter must be "
                "an absolute path. By default reads up to 2000 lines. Use offset and limit "
                "for pagination. Lines longer than 2000 chars are truncated."
            ),
        )

    def format_output(self, result: ReadResult, path: str) -> str:
        """Format output in OpenCode style with <file> tags and line numbers.

        Args:
            result: ReadResult from read_with_pagination
            path: Original path string for display

        Returns:
            Formatted output with line numbers and <file> tags
        """
        # Format with 5-digit zero-padded line numbers (OpenCode format: 00001|)
        numbered_lines = [
            f"{(i + result.start_offset + 1):05d}| {line}" for i, line in enumerate(result.lines)
        ]

        output = "<file>\n"
        output += "\n".join(numbered_lines)

        last_read_line = result.start_offset + len(result.lines)
        has_more_lines = result.total_lines > last_read_line

        if result.truncated_by_bytes:
            output += (
                f"\n\n(Output truncated at {self._max_bytes} bytes. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        elif has_more_lines or result.truncated:
            output += (
                f"\n\n(File has more lines. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        else:
            output += f"\n\n(End of file - total {result.total_lines} lines)"

        output += "\n</file>"
        return output

    async def __call__(
        self,
        filePath: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[TextContent | ImageContent]:
        """Read file contents.

        Args:
            filePath: Absolute path to the file to read
            offset: 0-based line number to start reading from
            limit: Number of lines to read (default: 2000)

        Returns:
            List of TextContent (or ImageContent for images) with file contents
        """
        if not filePath:
            raise ToolError("filePath is required")

        path = self.resolve_path(filePath)

        if not path.exists():
            raise ToolError(f"File not found: {filePath}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {filePath}")

        # Handle images
        if self.is_image(path):
            result = self.read_image(path)
            return result.to_content_blocks()  # type: ignore[return-value]

        # Read with pagination
        result = self.read_with_pagination(
            path,
            offset=offset or 0,
            limit=limit,
        )

        output = self.format_output(result, filePath)
        return list(ContentResult(output=output).to_text_blocks())


__all__ = ["ReadTool"]
