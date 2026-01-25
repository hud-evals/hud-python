"""Read tool for filesystem access.

Matches OpenCode's read tool specification exactly:
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

import base64
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs

DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 50 * 1024  # 50KB


class ReadTool(BaseTool):
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

    _base_path: Path

    def __init__(
        self,
        base_path: str = ".",
    ) -> None:
        """Initialize ReadTool.

        Args:
            base_path: Base directory for relative paths
        """
        super().__init__(
            env=None,
            name="read",
            title="Read",
            description=(
                "Reads a file from the local filesystem. The filePath parameter must be "
                "an absolute path. By default reads up to 2000 lines. Use offset and limit "
                "for pagination. Lines longer than 2000 chars are truncated."
            ),
        )
        self._base_path = Path(base_path).resolve()

    async def __call__(
        self,
        filePath: str,  # noqa: N803 - matches OpenCode param name
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ContentBlock]:
        """Read file contents.

        Args:
            filePath: Absolute path to the file to read
            offset: 0-based line number to start reading from
            limit: Number of lines to read (default: 2000)

        Returns:
            List of ContentBlocks with file contents
        """
        if not filePath:
            raise ToolError("filePath is required")

        path = resolve_path_safely(filePath, self._base_path)

        if not path.exists():
            raise ToolError(f"File not found: {filePath}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {filePath}")

        # Check for image files
        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"):
            try:
                image_data = path.read_bytes()
                b64_content = base64.b64encode(image_data).decode("utf-8")
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".bmp": "image/bmp",
                    ".svg": "image/svg+xml",
                }
                mime = mime_types.get(suffix, "application/octet-stream")
                return ContentResult(
                    output=f"Image read successfully: {filePath}",
                    system=f"data:{mime};base64,{b64_content[:100]}...",
                ).to_content_blocks()
            except Exception as e:
                raise ToolError(f"Failed to read image: {e}") from None

        # Check for binary files
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"Cannot read binary file: {filePath}") from None
        except PermissionError:
            raise ToolError(f"Permission denied: {filePath}") from None

        lines = content.split("\n")
        total_lines = len(lines)

        # Apply offset/limit (OpenCode uses 0-based offset)
        read_limit = limit if limit is not None else DEFAULT_READ_LIMIT
        start_offset = offset if offset is not None else 0

        # Collect lines with byte limit
        raw: list[str] = []
        total_bytes = 0
        truncated_by_bytes = False

        for i in range(start_offset, min(total_lines, start_offset + read_limit)):
            line = lines[i]
            # Truncate long lines (OpenCode max 2000 chars)
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "..."

            line_bytes = len(line.encode("utf-8")) + (1 if raw else 0)
            if total_bytes + line_bytes > MAX_BYTES:
                truncated_by_bytes = True
                break

            raw.append(line)
            total_bytes += line_bytes

        # Format with 5-digit zero-padded line numbers (OpenCode format: 00001|)
        numbered_lines = [
            f"{(i + start_offset + 1):05d}| {line}" for i, line in enumerate(raw)
        ]

        # Build output with <file> tags (OpenCode format)
        output = "<file>\n"
        output += "\n".join(numbered_lines)

        last_read_line = start_offset + len(raw)
        has_more_lines = total_lines > last_read_line

        if truncated_by_bytes:
            output += (
                f"\n\n(Output truncated at {MAX_BYTES} bytes. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        elif has_more_lines:
            output += (
                f"\n\n(File has more lines. "
                f"Use 'offset' parameter to read beyond line {last_read_line})"
            )
        else:
            output += f"\n\n(End of file - total {total_lines} lines)"

        output += "\n</file>"

        return ContentResult(output=output).to_content_blocks()


__all__ = ["ReadTool"]
