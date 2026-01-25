"""Grep tool for searching file contents.

Matches the `grep` tool from OpenCode and similar coding agents.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs


class GrepTool(BaseTool):
    """Search file contents using regular expressions.

    This tool searches files for patterns, matching the `grep` tool
    from OpenCode and similar coding agents.

    Parameters:
        pattern: Regular expression pattern to search for (required)
        path: Directory or file to search in (default: ".")
        include: Optional glob pattern to filter files (e.g., "*.py")

    Example:
        >>> tool = GrepTool(base_path="./workspace")
        >>> result = await tool(pattern="def main", include="*.py")
        >>> result = await tool(pattern="TODO|FIXME", path="src/")
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    _base_path: Path
    _max_results: int
    _max_files: int

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = 100,
        max_files: int = 1000,
    ) -> None:
        """Initialize GrepTool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum matching lines to return (default: 100)
            max_files: Maximum files to search (default: 1000)
        """
        super().__init__(
            env=None,
            name="grep",
            title="Search Files",
            description=(
                "Search file contents using regex. Returns matching lines with "
                "file path and line number. Use 'include' to filter by file type."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_results = max_results
        self._max_files = max_files

    async def __call__(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
    ) -> list[ContentBlock]:
        """Search file contents for a pattern.

        Args:
            pattern: Regular expression pattern to search for
            path: Directory or file to search in
            include: Optional glob pattern to filter files (e.g., "*.py")

        Returns:
            List of ContentBlocks with matching lines
        """
        if not pattern:
            raise ToolError("pattern is required")

        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Invalid regex pattern: {e}") from None

        search_path = resolve_path_safely(path, self._base_path)

        if not search_path.exists():
            raise ToolError(f"Path not found: {path}")

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

        # Search files
        results: list[str] = []
        files_searched = 0
        files_with_matches = 0

        for file in files:
            try:
                content = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            files_searched += 1
            file_had_match = False

            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    if not file_had_match:
                        files_with_matches += 1
                        file_had_match = True

                    # Format: relative_path:line_num: content
                    try:
                        rel_path = file.relative_to(self._base_path)
                    except ValueError:
                        rel_path = file
                    results.append(f"{rel_path}:{i}: {line.strip()}")

                    if len(results) >= self._max_results:
                        break

            if len(results) >= self._max_results:
                break

        # Format output
        if not results:
            output = f"No matches found for pattern: {pattern}"
        else:
            header = (
                f"Found {len(results)} matches in {files_with_matches} files "
                f"(searched {files_searched} files)"
            )
            if len(results) >= self._max_results:
                header += f" [limited to {self._max_results} results]"
            output = f"{header}\n\n" + "\n".join(results)

        return ContentResult(output=output).to_content_blocks()


__all__ = ["GrepTool"]
