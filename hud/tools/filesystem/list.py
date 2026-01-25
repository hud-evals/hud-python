"""List tool for directory contents.

Matches OpenCode's list tool specification exactly:
https://github.com/anomalyco/opencode

Key features:
- Absolute path parameter (optional, defaults to workspace)
- Array of glob patterns to ignore
- Tree structure output with indentation
- Default ignore patterns for common directories
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from hud.tools.native_types import NativeToolSpecs

# OpenCode's default ignore patterns
IGNORE_PATTERNS = [
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

MAX_ENTRIES = 100


class ListTool(BaseTool):
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

    _base_path: Path

    def __init__(
        self,
        base_path: str = ".",
    ) -> None:
        """Initialize ListTool.

        Args:
            base_path: Base directory for relative paths
        """
        super().__init__(
            env=None,
            name="list",
            title="List",
            description=(
                "Lists files and directories in a given path. The path parameter must be "
                "absolute; omit it to use the current workspace directory. "
                "You can optionally provide an array of glob patterns to ignore."
            ),
        )
        self._base_path = Path(base_path).resolve()

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
        search_path = resolve_path_safely(path or ".", self._base_path)

        if not search_path.exists():
            raise ToolError(f"Directory not found: {path or '.'}")
        if not search_path.is_dir():
            raise ToolError(f"Not a directory: {path or '.'}")

        # Combine default and custom ignore patterns
        ignore_patterns = list(IGNORE_PATTERNS) + (ignore or [])

        def should_ignore(name: str, is_dir: bool) -> bool:
            """Check if a file/directory should be ignored."""
            # Skip hidden files
            if name.startswith("."):
                return True

            for pattern in ignore_patterns:
                # Handle directory patterns ending with /
                if pattern.endswith("/"):
                    if is_dir and fnmatch.fnmatch(name, pattern.rstrip("/")):
                        return True
                else:
                    if fnmatch.fnmatch(name, pattern):
                        return True
            return False

        # Collect files using recursive walk (like OpenCode's ripgrep approach)
        files: list[str] = []

        def collect_files(dir_path: Path, prefix: str = "") -> None:
            """Recursively collect files."""
            if len(files) >= MAX_ENTRIES:
                return

            try:
                entries = list(dir_path.iterdir())
            except PermissionError:
                return

            # Sort: directories first, then files, alphabetically
            dirs = []
            regular_files = []
            for entry in entries:
                if should_ignore(entry.name, entry.is_dir()):
                    continue
                if entry.is_dir():
                    dirs.append(entry)
                else:
                    regular_files.append(entry)

            dirs.sort(key=lambda x: x.name.lower())
            regular_files.sort(key=lambda x: x.name.lower())

            # Process directories first
            for d in dirs:
                if len(files) >= MAX_ENTRIES:
                    break
                rel_path = prefix + d.name + "/"
                files.append(rel_path)
                collect_files(d, rel_path)

            # Then files
            for f in regular_files:
                if len(files) >= MAX_ENTRIES:
                    break
                files.append(prefix + f.name)

        collect_files(search_path)

        truncated = len(files) >= MAX_ENTRIES

        # Build tree structure output (OpenCode format)
        if not files:
            output = f"Empty directory: {path or '.'}"
        else:
            # Build tree with indentation
            lines = [f"{search_path}/"]

            for file_path in files:
                # Count depth by number of /
                parts = file_path.rstrip("/").split("/")
                depth = len(parts) - 1
                indent = "  " * (depth + 1)
                name = parts[-1]

                if file_path.endswith("/"):
                    lines.append(f"{indent}{name}/")
                else:
                    lines.append(f"{indent}{name}")

            output = "\n".join(lines)

            if truncated:
                output += f"\n\n(Limited to {MAX_ENTRIES} entries)"

        return ContentResult(output=output).to_content_blocks()


__all__ = ["ListTool"]
