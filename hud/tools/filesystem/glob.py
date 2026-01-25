"""Glob tool for finding files by pattern.

Matches OpenCode's glob tool specification exactly:
https://github.com/anomalyco/opencode

Key features:
- Fast file pattern matching
- Results sorted by modification time (recent first)
- Supports glob patterns like "**/*.js"
- Max 100 results
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs

MAX_RESULTS = 100


class GlobTool(BaseTool):
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

    _base_path: Path

    def __init__(
        self,
        base_path: str = ".",
    ) -> None:
        """Initialize GlobTool.

        Args:
            base_path: Base directory for relative paths
        """
        super().__init__(
            env=None,
            name="glob",
            title="Glob",
            description=(
                "Fast file pattern matching tool. Supports glob patterns like '**/*.js' "
                "or 'src/**/*.ts'. Returns matching file paths sorted by modification time."
            ),
        )
        self._base_path = Path(base_path).resolve()

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
        if not pattern:
            raise ToolError("pattern is required")

        base = resolve_path_safely(path or ".", self._base_path)

        if not base.exists():
            raise ToolError(f"Directory not found: {path or '.'}")
        if not base.is_dir():
            raise ToolError(f"Not a directory: {path or '.'}")

        # Find matching files with mtime
        matches: list[tuple[Path, float]] = []
        try:
            for match in base.glob(pattern):
                # Skip hidden files/directories
                if any(part.startswith(".") for part in match.parts):
                    continue
                # Skip common non-source directories
                if any(
                    part in ("node_modules", "__pycache__", "venv", ".venv")
                    for part in match.parts
                ):
                    continue

                if not match.is_file():
                    continue

                try:
                    mtime = os.path.getmtime(match)
                except OSError:
                    mtime = 0

                matches.append((match, mtime))

                if len(matches) >= MAX_RESULTS:
                    break
        except Exception as e:
            raise ToolError(f"Invalid glob pattern: {e}") from None

        # Sort by modification time (most recent first) - OpenCode behavior
        matches.sort(key=lambda x: x[1], reverse=True)

        truncated = len(matches) >= MAX_RESULTS

        if not matches:
            output = "No files found"
        else:
            # Convert to relative paths
            rel_paths = []
            for m, _mtime in matches:
                try:
                    rel_paths.append(str(m.relative_to(self._base_path)))
                except ValueError:
                    rel_paths.append(str(m))

            output = "\n".join(rel_paths)

            if truncated:
                output += "\n\n(Results are truncated. Consider using a more specific path or pattern.)"

        return ContentResult(output=output).to_content_blocks()


__all__ = ["GlobTool"]
