"""List tool for directory contents.

Matches the `list` tool from OpenCode and similar coding agents.
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


class ListTool(BaseTool):
    """List directory contents.

    This tool lists files and directories, matching the `list` tool
    from OpenCode and similar coding agents.

    Parameters:
        path: Directory to list (default: ".")
        pattern: Optional glob pattern to filter results
        recursive: Whether to list recursively (default: False)

    Example:
        >>> tool = ListTool(base_path="./workspace")
        >>> result = await tool(path="src/")
        >>> result = await tool(path=".", pattern="*.py")
    """

    native_specs: ClassVar[NativeToolSpecs] = {}  # Function calling only

    _base_path: Path
    _max_entries: int

    def __init__(
        self,
        base_path: str = ".",
        max_entries: int = 500,
    ) -> None:
        """Initialize ListTool.

        Args:
            base_path: Base directory for relative paths
            max_entries: Maximum entries to return (default: 500)
        """
        super().__init__(
            env=None,
            name="list",
            title="List Directory",
            description=(
                "List directory contents. Shows files (f) and directories (d) "
                "with optional pattern filtering."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_entries = max_entries

    async def __call__(
        self,
        path: str = ".",
        pattern: str | None = None,
        recursive: bool = False,
    ) -> list[ContentBlock]:
        """List directory contents.

        Args:
            path: Directory to list
            pattern: Optional glob pattern to filter results
            recursive: Whether to list recursively

        Returns:
            List of ContentBlocks with directory listing
        """
        dir_path = resolve_path_safely(path, self._base_path)

        if not dir_path.exists():
            raise ToolError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise ToolError(f"Not a directory: {path}")

        # Collect entries
        entries: list[tuple[str, Path]] = []

        iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()

        for entry in iterator:
            # Skip hidden files
            if entry.name.startswith("."):
                continue
            # Skip common non-source directories
            if any(
                part in ("node_modules", "__pycache__", "venv", ".venv") for part in entry.parts
            ):
                continue
            # Apply pattern filter
            if pattern and not fnmatch.fnmatch(entry.name, pattern):
                continue

            entries.append((entry.name, entry))

            if len(entries) >= self._max_entries:
                break

        # Sort: directories first, then files, alphabetically
        entries.sort(key=lambda x: (not x[1].is_dir(), x[0].lower()))

        # Format output
        if not entries:
            output = f"Empty directory: {path}"
            if pattern:
                output = f"No entries matching '{pattern}' in: {path}"
        else:
            lines = []
            for name, entry in entries:
                if entry.is_dir():
                    prefix = "d"
                    suffix = "/"
                else:
                    prefix = "f"
                    suffix = ""
                    # Add file size for files
                    try:
                        size = entry.stat().st_size
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size // 1024}K"
                        else:
                            size_str = f"{size // (1024 * 1024)}M"
                        suffix = f"  ({size_str})"
                    except OSError:
                        suffix = ""

                if recursive:
                    try:
                        rel_path = entry.relative_to(dir_path)
                        lines.append(f"{prefix} {rel_path}{suffix}")
                    except ValueError:
                        lines.append(f"{prefix} {name}{suffix}")
                else:
                    lines.append(f"{prefix} {name}{suffix}")

            header = f"Directory: {path} ({len(entries)} entries)"
            if len(entries) >= self._max_entries:
                header += f" [limited to {self._max_entries}]"

            output = f"{header}\n\n" + "\n".join(lines)

        return ContentResult(output=output).to_content_blocks()


__all__ = ["ListTool"]
