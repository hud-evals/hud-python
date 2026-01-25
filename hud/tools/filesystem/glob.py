"""Glob tool for finding files by pattern (OpenCode-style).

Matches OpenCode's glob tool specification:
https://github.com/anomalyco/opencode

Key features:
- Fast file pattern matching
- Results sorted by modification time (recent first)
- Supports glob patterns like "**/*.js"
- Max 100 results
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

from hud.tools.filesystem.base import BaseGlobTool
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from mcp.types import ContentBlock

    from hud.tools.native_types import NativeToolSpecs


class GlobTool(BaseGlobTool):
    """Find files matching OpenCode's glob tool.

    Fast file pattern matching tool that works with any codebase size.
    Returns matching file paths sorted by modification time (most recent first).

    Parameters:
        pattern: Glob pattern (e.g., "**/*.py", "src/*.ts") (required)
        path: Base directory to search from (optional, defaults to workspace)

    Example:
        >>> tool = GlobTool(base_path="./workspace")
        >>> result = await tool(pattern="**/*.py")
        >>> result = await tool(pattern="src/**/*.ts", path="frontend/")
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = 100,
    ) -> None:
        """Initialize GlobTool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum files to return
        """
        super().__init__(
            base_path=base_path,
            max_results=max_results,
            name="glob",
            title="Glob",
            description=(
                "Fast file pattern matching tool. Supports glob patterns like '**/*.js' "
                "or 'src/**/*.ts'. Returns matching file paths sorted by modification time."
            ),
        )

    def format_output(self, matches: list[tuple[Path, float]], pattern: str) -> str:
        """Format output in OpenCode style (relative paths, sorted by mtime).

        Args:
            matches: List of (path, mtime) tuples
            pattern: Original glob pattern

        Returns:
            Formatted output with relative paths
        """
        if not matches:
            return "No files found"

        # Sort by mtime (most recent first) - OpenCode behavior
        sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
        truncated = len(matches) >= self._max_results

        # Convert to relative paths
        rel_paths = []
        for path, _mtime in sorted_matches:
            try:
                rel_paths.append(str(path.relative_to(self._base_path)))
            except ValueError:
                rel_paths.append(str(path))

        output = "\n".join(rel_paths)

        if truncated:
            output += "\n\n(Results are truncated. Consider using a more specific path or pattern.)"

        return output

    async def __call__(
        self,
        pattern: str,
        path: str | None = None,
    ) -> list[ContentBlock]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/*.ts")
            path: Base directory to search from (defaults to workspace)

        Returns:
            List of ContentBlocks with matching file paths
        """
        base = self.resolve_path(path or ".")

        if not base.exists():
            raise ToolError(f"Directory not found: {path or '.'}")
        if not base.is_dir():
            raise ToolError(f"Not a directory: {path or '.'}")

        matches = self.find_files(base, pattern)
        output = self.format_output(matches, pattern)

        return ContentResult(output=output).to_content_blocks()


__all__ = ["GlobTool"]
