"""Grep tool for searching file contents.

Matches OpenCode's grep tool specification exactly:
https://github.com/anomalyco/opencode

Key features:
- Fast content search using regex
- Results sorted by modification time (recent first)
- Grouped output by file with line numbers
- Max 100 results, max 2000 char line length
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs

MAX_LINE_LENGTH = 2000
MAX_RESULTS = 100


class GrepTool(BaseTool):
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

    _base_path: Path
    _max_files: int

    def __init__(
        self,
        base_path: str = ".",
        max_files: int = 1000,
    ) -> None:
        """Initialize GrepTool.

        Args:
            base_path: Base directory for relative paths
            max_files: Maximum files to search (default: 1000)
        """
        super().__init__(
            env=None,
            name="grep",
            title="Grep",
            description=(
                "Fast content search tool. Searches file contents using regular expressions. "
                "Supports full regex syntax. Filter files by pattern with 'include' parameter. "
                "Returns file paths and line numbers sorted by modification time."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_files = max_files

    async def __call__(
        self,
        pattern: str,
        path: str | None = None,
        include: str | None = None,
    ) -> list[ContentBlock]:
        """Search file contents for a pattern.

        Args:
            pattern: Regular expression pattern to search for
            path: Directory to search in (defaults to base path)
            include: Glob pattern to filter files (e.g., "*.py")

        Returns:
            List of ContentBlocks with matching lines grouped by file
        """
        if not pattern:
            raise ToolError("pattern is required")

        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Invalid regex pattern: {e}") from None

        search_path = resolve_path_safely(path or ".", self._base_path)

        if not search_path.exists():
            raise ToolError(f"Path not found: {path or '.'}")

        # Collect files to search
        if search_path.is_file():
            files = [search_path]
        else:
            files = []
            for f in search_path.rglob("*"):
                if len(files) >= self._max_files:
                    break
                if not f.is_file():
                    continue
                # Skip hidden files and common non-text directories
                if any(part.startswith(".") for part in f.parts):
                    continue
                if any(
                    part in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                    for part in f.parts
                ):
                    continue
                if include and not fnmatch.fnmatch(f.name, include):
                    continue
                files.append(f)

        # Search files and collect matches with mtime
        matches: list[dict[str, str | int | float]] = []

        for file in files:
            try:
                content = file.read_text(encoding="utf-8")
                mtime = os.path.getmtime(file)
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            try:
                rel_path = str(file.relative_to(self._base_path))
            except ValueError:
                rel_path = str(file)

            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    # Truncate long lines (OpenCode max 2000 chars)
                    line_text = line.strip()
                    if len(line_text) > MAX_LINE_LENGTH:
                        line_text = line_text[:MAX_LINE_LENGTH] + "..."

                    matches.append({
                        "path": rel_path,
                        "mtime": mtime,
                        "line_num": i,
                        "line_text": line_text,
                    })

                    if len(matches) >= MAX_RESULTS:
                        break

            if len(matches) >= MAX_RESULTS:
                break

        # Sort by modification time (most recent first) - OpenCode behavior
        matches.sort(key=lambda x: float(x["mtime"]), reverse=True)

        truncated = len(matches) >= MAX_RESULTS
        final_matches = matches[:MAX_RESULTS]

        # Format output (grouped by file like OpenCode)
        if not final_matches:
            output = "No files found"
        else:
            lines = [f"Found {len(final_matches)} matches"]
            lines.append("")

            current_file = ""
            for match in final_matches:
                if current_file != match["path"]:
                    if current_file:
                        lines.append("")
                    current_file = str(match["path"])
                    lines.append(f"{current_file}:")

                lines.append(f"  Line {match['line_num']}: {match['line_text']}")

            if truncated:
                lines.append("")
                lines.append(
                    "(Results are truncated. Consider using a more specific path or pattern.)"
                )

            output = "\n".join(lines)

        return ContentResult(output=output).to_content_blocks()


__all__ = ["GrepTool"]
