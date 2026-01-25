"""List tool for directory contents (OpenCode-style).

Matches OpenCode's list tool specification:
https://github.com/anomalyco/opencode

Key features:
- Absolute path parameter (optional, defaults to workspace)
- Array of glob patterns to ignore
- Tree structure output with indentation
- Default ignore patterns for common directories
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

from hud.tools.filesystem.base import BaseListTool
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from mcp.types import ContentBlock

    from hud.tools.native_types import NativeToolSpecs

# OpenCode's default ignore patterns
OPENCODE_IGNORE_PATTERNS = [
    "node_modules/",
    "__pycache__/",
    ".git/",
    "dist/",
    "build/",
    "target/",
    "vendor/",
    "bin/",
    "obj/",
    ".idea/",
    ".vscode/",
    ".zig-cache/",
    "zig-out",
    ".coverage",
    "coverage/",
    "tmp/",
    "temp/",
    ".cache/",
    "cache/",
    "logs/",
    ".venv/",
    "venv/",
    "env/",
]


class ListTool(BaseListTool):
    """List directory contents matching OpenCode's list tool.

    Lists files and directories in a tree structure with indentation.
    Supports ignore patterns for filtering results.

    Parameters:
        path: Absolute path to directory (optional, defaults to workspace)
        ignore: Array of glob patterns to ignore (optional)

    Example:
        >>> tool = ListTool(base_path="./workspace")
        >>> result = await tool(path="/path/to/dir")
        >>> result = await tool(path="/path/to/dir", ignore=["*.log", "temp/"])
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    def __init__(
        self,
        base_path: str = ".",
        max_entries: int = 100,
    ) -> None:
        """Initialize ListTool.

        Args:
            base_path: Base directory for relative paths
            max_entries: Maximum entries to return
        """
        super().__init__(
            base_path=base_path,
            max_entries=max_entries,
            name="list",
            title="List",
            description=(
                "Lists files and directories in a given path. The path parameter must be "
                "absolute; omit it to use the current workspace directory. "
                "You can optionally provide an array of glob patterns to ignore."
            ),
        )

    def format_output(
        self,
        entries: list[tuple[str, bool]],
        directory: Path,
        path_str: str,
    ) -> str:
        """Format output in OpenCode style (tree with indentation).

        Args:
            entries: List of (relative_path, is_dir) tuples
            directory: Directory that was listed
            path_str: Original path string for display

        Returns:
            Formatted tree output
        """
        if not entries:
            return f"Empty directory: {path_str or '.'}"

        truncated = len(entries) >= self._max_entries

        # Build tree with indentation
        lines = [f"{directory}/"]

        for file_path, is_dir in entries:
            # Count depth by number of /
            parts = file_path.rstrip("/").split("/")
            depth = len(parts) - 1
            indent = "  " * (depth + 1)
            name = parts[-1]

            if is_dir:
                lines.append(f"{indent}{name}/")
            else:
                lines.append(f"{indent}{name}")

        output = "\n".join(lines)

        if truncated:
            output += f"\n\n(Limited to {self._max_entries} entries)"

        return output

    async def __call__(
        self,
        path: str | None = None,
        ignore: list[str] | None = None,
    ) -> list[ContentBlock]:
        """List directory contents.

        Args:
            path: Absolute path to directory (defaults to workspace)
            ignore: Array of glob patterns to ignore

        Returns:
            List of ContentBlocks with directory tree
        """
        search_path = self.resolve_path(path or ".")

        if not search_path.exists():
            raise ToolError(f"Directory not found: {path or '.'}")
        if not search_path.is_dir():
            raise ToolError(f"Not a directory: {path or '.'}")

        # Combine default and custom ignore patterns
        ignore_patterns = list(OPENCODE_IGNORE_PATTERNS) + (ignore or [])

        entries = self.list_directory(search_path, ignore=ignore_patterns)
        output = self.format_output(entries, search_path, path or ".")

        return ContentResult(output=output).to_content_blocks()


__all__ = ["ListTool"]
