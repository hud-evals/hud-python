"""Grep tool for searching file contents (OpenCode-style).

Matches OpenCode's grep tool specification:
https://github.com/anomalyco/opencode

Key features:
- Fast content search using regex
- Results sorted by modification time (recent first)
- Grouped output by file with line numbers
- Max 100 results, max 2000 char line length
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mcp.types import TextContent  # noqa: TC002

from hud.tools.filesystem.base import BaseSearchTool, FileMatch
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs


class GrepTool(BaseSearchTool):
    """Search file contents matching OpenCode's grep tool.

    Fast content search tool that searches file contents using regex.
    Results are sorted by modification time (most recent first).

    Parameters:
        pattern: Regular expression pattern to search for (required)
        path: Directory to search in (optional, defaults to workspace)
        include: Glob pattern to filter files (e.g., "*.py", "*.{ts,tsx}")

    Example:
        >>> tool = GrepTool(base_path="./workspace")
        >>> result = await tool(pattern="def main", include="*.py")
        >>> result = await tool(pattern="TODO|FIXME", path="src/")
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = 100,
        max_files: int = 1000,
    ) -> None:
        """Initialize GrepTool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum matching lines to return
            max_files: Maximum files to search
        """
        super().__init__(
            base_path=base_path,
            max_results=max_results,
            max_files=max_files,
            name="grep",
            title="Grep",
            description=(
                "Fast content search tool. Searches file contents using regular expressions. "
                "Supports full regex syntax. Filter files by pattern with 'include' parameter. "
                "Returns file paths and line numbers sorted by modification time."
            ),
        )

    def format_output(self, matches: list[FileMatch], pattern: str) -> str:
        """Format output in OpenCode style (grouped by file, sorted by mtime).

        Args:
            matches: List of FileMatch objects
            pattern: Original search pattern

        Returns:
            Formatted output grouped by file
        """
        if not matches:
            return "No files found"

        # Sort by mtime (most recent first) - OpenCode behavior
        sorted_matches = sorted(matches, key=lambda x: x.mtime, reverse=True)
        truncated = len(matches) >= self._max_results

        lines = [f"Found {len(sorted_matches)} matches"]
        lines.append("")

        current_file = ""
        for match in sorted_matches:
            if current_file != match.path:
                if current_file:
                    lines.append("")
                current_file = match.path
                lines.append(f"{current_file}:")

            lines.append(f"  Line {match.line_num}: {match.line_text}")

        if truncated:
            lines.append("")
            lines.append("(Results are truncated. Consider using a more specific path or pattern.)")

        return "\n".join(lines)

    async def __call__(
        self,
        pattern: str,
        path: str | None = None,
        include: str | None = None,
    ) -> list[TextContent]:
        """Search file contents for a pattern.

        Args:
            pattern: Regular expression pattern to search for
            path: Directory to search in (defaults to base path)
            include: Glob pattern to filter files (e.g., "*.py")

        Returns:
            List of TextContent with matching lines grouped by file
        """
        regex = self.compile_pattern(pattern)
        search_path = self.resolve_path(path or ".")

        if not search_path.exists():
            raise ToolError(f"Path not found: {path or '.'}")

        matches = self.search_files(search_path, regex, include)
        output = self.format_output(matches, pattern)

        return ContentResult(output=output).to_text_blocks()


__all__ = ["GrepTool"]
