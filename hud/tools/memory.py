"""Memory environment tools for persistent file-backed storage."""

from __future__ import annotations

import logging
import shutil
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, get_args

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding import EditTool, write_file_async
from hud.tools.types import ContentResult, ToolError

LOGGER = logging.getLogger(__name__)


class BaseMemoryTool(BaseTool):
    """Abstract base for all memory tools.

    Subclasses implement file-backed memory operations. Provider-native memory
    tools live on agent harnesses and call this environment primitive.
    """

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> list[ContentBlock]:
        """Execute a memory operation."""
        ...


class BaseFileMemoryTool(BaseMemoryTool):
    """Base class for file-based memory tools.

    Provides common functionality for tools that store memories as files:
    - Path resolution with security checks
    - Directory management
    - File reading/writing utilities
    """

    _base_path: Path
    _memory_section_header: str

    def __init__(
        self,
        base_path: str | Path = ".",
        memory_section_header: str = "## Memories",
        **kwargs: Any,
    ) -> None:
        """Initialize file-based memory tool.

        Args:
            base_path: Base directory for memory files
            memory_section_header: Markdown header for memory section
            **kwargs: Passed to parent classes (for cooperative inheritance)
        """
        # Pass kwargs to parent for cooperative multiple inheritance
        # This allows EditTool + BaseFileMemoryTool to work together
        super().__init__(
            env=kwargs.get("env"),
            name="memory",
            title="Memory",
            meta={"capability": "memory"},
        )
        self._base_path = Path(base_path).resolve()
        self._memory_section_header = memory_section_header

        # Ensure base directory exists
        self._base_path.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str) -> Path:
        """Resolve and validate a path within the memory directory.

        Prevents directory traversal attacks.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path escapes the base directory
        """
        relative = path.lstrip("/") if path.startswith("/") else path
        resolved = (self._base_path / relative).resolve()

        # Security check - prevent traversal
        try:
            resolved.relative_to(self._base_path)
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}") from None

        return resolved

    def read_memory_file(self, path: Path) -> str:
        """Read memory file contents.

        Args:
            path: Path to file

        Returns:
            File contents as string, or empty string if file doesn't exist
        """
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except Exception as e:
            LOGGER.warning("Failed to read memory file %s: %s", path, e)
            return ""

    def write_memory_file(self, path: Path, content: str) -> None:
        """Write content to memory file.

        Creates parent directories if needed.

        Args:
            path: Path to file
            content: Content to write
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


MemoryCommand = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "delete",
    "rename",
]


class MemoryTool(EditTool, BaseFileMemoryTool):
    """Environment tool for persistent memory files.

    Extends EditTool with memory-specific functionality:
    - All paths must be within /memories directory
    - Supports delete and rename commands (instead of undo_edit)
    - Custom directory listing with file sizes

    Commands:
        view: Show directory contents or file contents
        create: Create a new file
        str_replace: Replace text in a file
        insert: Insert text at a specific line
        delete: Delete a file or directory
        rename: Rename or move a file/directory
    """

    def __init__(
        self,
        memories_dir: str | Path = "/memories",
        file_history: dict[Path, list[str]] | None = None,
    ) -> None:
        """Initialize MemoryTool.

        Args:
            memories_dir: Base directory for memory files (default: /memories)
            file_history: Optional dictionary tracking edit history per file
        """
        _file_history = file_history or defaultdict(list)

        EditTool.__init__(self, file_history=_file_history)
        BaseFileMemoryTool.__init__(
            self,
            base_path=memories_dir,
            memory_section_header="## Memories",
        )

        self.env = _file_history
        self.name = "memory"
        self.title = "Memory"
        self.description = "Store and retrieve persistent information across conversations"

    def _resolve_memory_path(self, path: str) -> Path:
        """Validate and resolve a path within the memories directory."""
        if path.startswith("/memories"):
            relative_path = path[len("/memories") :].lstrip("/")
        else:
            relative_path = path.lstrip("/")

        return self.resolve_path(relative_path)

    def validate_path(self, command: str, path: Path) -> None:
        """Override parent validation; memory paths are resolved before operations."""
        return

    async def __call__(
        self,
        *,
        command: MemoryCommand,  # type: ignore[override]
        path: str | None = None,
        view_range: list[int] | None = None,
        file_text: str | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        insert_text: str | None = None,
        old_path: str | None = None,
        new_path: str | None = None,
    ) -> list[ContentBlock]:
        """Execute a memory command."""
        if command == "view":
            if path is None:
                path = "/memories"
            result = await self._memory_view(path, view_range)
            return result.to_content_blocks()

        if command == "create":
            if path is None:
                raise ToolError("path is required for command: create")
            if file_text is None:
                raise ToolError("file_text is required for command: create")
            resolved = self._resolve_memory_path(path)
            if resolved.exists():
                raise ToolError(f"Error: File {path} already exists")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await write_file_async(resolved, file_text)
            self.file_history[resolved].append(file_text)
            result = ContentResult(output=f"File created successfully at: {path}")
            return result.to_content_blocks()

        if command == "str_replace":
            if path is None:
                raise ToolError("path is required for command: str_replace")
            if old_str is None:
                raise ToolError("old_str is required for command: str_replace")
            resolved = self._resolve_memory_path(path)
            if not resolved.exists() or resolved.is_dir():
                raise ToolError(
                    f"Error: The path {path} does not exist. Please provide a valid path."
                )
            result = await self.replace(resolved, old_str, new_str)
            if result.output:
                result = ContentResult(output=result.output.replace("The file", "The memory file"))
            return result.to_content_blocks()

        if command == "insert":
            if path is None:
                raise ToolError("path is required for command: insert")
            if insert_line is None:
                raise ToolError("insert_line is required for command: insert")
            if insert_text is None:
                raise ToolError("insert_text is required for command: insert")
            resolved = self._resolve_memory_path(path)
            if not resolved.exists() or resolved.is_dir():
                raise ToolError(f"Error: The path {path} does not exist")
            result = await self.insert(resolved, insert_line, insert_text)
            return result.to_content_blocks()

        if command == "delete":
            if path is None:
                raise ToolError("path is required for command: delete")
            result = await self._memory_delete(path)
            return result.to_content_blocks()

        if command == "rename":
            if old_path is None:
                raise ToolError("old_path is required for command: rename")
            if new_path is None:
                raise ToolError("new_path is required for command: rename")
            result = await self._memory_rename(old_path, new_path)
            return result.to_content_blocks()

        allowed = ", ".join(get_args(MemoryCommand))
        raise ToolError(f"Unrecognized command {command}. Allowed commands: {allowed}")

    async def _memory_view(self, path: str, view_range: list[int] | None = None) -> ContentResult:
        """View directory contents or file contents with memory-specific formatting."""
        resolved = self._resolve_memory_path(path)

        if not resolved.exists():
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")

        if resolved.is_dir():
            if view_range:
                raise ToolError(
                    "The view_range parameter is not allowed when path points to a directory."
                )
            lines = []
            for item in sorted(resolved.rglob("*")):
                relative = item.relative_to(resolved)
                if len(relative.parts) > 2:
                    continue
                if any(part.startswith(".") for part in relative.parts):
                    continue

                try:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}K"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}M"
                except OSError:
                    size_str = "?"

                lines.append(f"{size_str}\t{path}/{relative}")

            header = (
                f"Here're the files and directories up to 2 levels deep in {path}, "
                "excluding hidden items and node_modules:\n"
            )
            return ContentResult(output=header + "\n".join(lines))

        return await self.view(resolved, view_range)

    async def _memory_delete(self, path: str) -> ContentResult:
        """Delete a file or directory."""
        resolved = self._resolve_memory_path(path)

        if not resolved.exists():
            raise ToolError(f"Error: The path {path} does not exist")

        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

        return ContentResult(output=f"Successfully deleted {path}")

    async def _memory_rename(self, old_path: str, new_path: str) -> ContentResult:
        """Rename or move a file/directory."""
        old_resolved = self._resolve_memory_path(old_path)
        new_resolved = self._resolve_memory_path(new_path)

        if not old_resolved.exists():
            raise ToolError(f"Error: The path {old_path} does not exist")
        if new_resolved.exists():
            raise ToolError(f"Error: The destination {new_path} already exists")

        new_resolved.parent.mkdir(parents=True, exist_ok=True)
        old_resolved.rename(new_resolved)

        return ContentResult(output=f"Successfully renamed {old_path} to {new_path}")


__all__ = [
    "BaseFileMemoryTool",
    "BaseMemoryTool",
    "MemoryCommand",
    "MemoryTool",
]
