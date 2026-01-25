"""Glob tool for finding files by pattern.

Matches the `glob` tool from OpenCode and similar coding agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs


class GlobTool(BaseTool):
    """Find files matching a glob pattern.

    This tool finds files by pattern, matching the `glob` tool
    from OpenCode and similar coding agents.

    Parameters:
        pattern: Glob pattern (e.g., "**/*.py", "src/*.ts") (required)
        path: Base directory to search from (default: ".")

    Example:
        >>> tool = GlobTool(base_path="./workspace")
        >>> result = await tool(pattern="**/*.py")
        >>> result = await tool(pattern="src/**/*.ts", path="frontend/")
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    _base_path: Path
    _max_results: int

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = 500,
    ) -> None:
        """Initialize GlobTool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum files to return (default: 500)
        """
        super().__init__(
            env=None,
            name="glob",
            title="Find Files",
            description=(
                "Find files matching a glob pattern. "
                "Use **/ for recursive search (e.g., '**/*.py')."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_results = max_results

    async def __call__(
        self,
        pattern: str,
        path: str = ".",
    ) -> list[ContentBlock]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/*.ts")
            path: Base directory to search from

        Returns:
            List of ContentBlocks with matching file paths
        """
        if not pattern:
            raise ToolError("pattern is required")

        base = resolve_path_safely(path, self._base_path)

        if not base.exists():
            raise ToolError(f"Directory not found: {path}")
        if not base.is_dir():
            raise ToolError(f"Not a directory: {path}")

        # Find matching files
        matches: list[Path] = []
        try:
            for match in base.glob(pattern):
                # Skip hidden files/directories
                if any(part.startswith(".") for part in match.parts):
                    continue
                # Skip common non-source directories
                if any(
                    part in ("node_modules", "__pycache__", "venv", ".venv") for part in match.parts
                ):
                    continue

                matches.append(match)
                if len(matches) >= self._max_results:
                    break
        except Exception as e:
            raise ToolError(f"Invalid glob pattern: {e}") from None

        # Sort and format output
        matches.sort()

        if not matches:
            output = f"No files found matching: {pattern}"
        else:
            # Convert to relative paths
            rel_paths = []
            for m in matches:
                try:
                    rel_paths.append(str(m.relative_to(self._base_path)))
                except ValueError:
                    rel_paths.append(str(m))

            header = f"Found {len(matches)} files matching: {pattern}"
            if len(matches) >= self._max_results:
                header += f" [limited to {self._max_results} results]"

            output = f"{header}\n\n" + "\n".join(rel_paths)

        return ContentResult(output=output).to_content_blocks()


__all__ = ["GlobTool"]
