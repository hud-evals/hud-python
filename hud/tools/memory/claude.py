"""Claude Memory tool for persistent storage across conversations.

This tool provides file-based memory storage with Claude's native memory API:
- Path validation to restrict access to /memories directory
- Commands: view, create, str_replace, insert, delete, rename
- Custom directory listing with file sizes

See: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/memory-tool
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar, Literal, get_args

if TYPE_CHECKING:
    from pathlib import Path

from hud.tools.coding.edit import EditTool
from hud.tools.coding.utils import write_file_async
from hud.tools.memory.base import BaseFileMemoryTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType

if TYPE_CHECKING:
    from mcp.types import ContentBlock

ClaudeMemoryCommand = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "delete",
    "rename",
]


class ClaudeMemoryTool(EditTool, BaseFileMemoryTool):
    """Persistent memory tool for Claude agents.

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

    Native specs: Claude (memory_20250818)
    Role: "memory" (unique role for memory operations)

    Requires beta header: context-management-2025-06-27
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.CLAUDE: NativeToolSpec(
            api_type="memory_20250818",
            api_name="memory",
            beta="context-management-2025-06-27",
            role="memory",
        ),
    }

    def __init__(
        self,
        memories_dir: str | Path = "/memories",
        file_history: dict[Path, list[str]] | None = None,
    ) -> None:
        """Initialize ClaudeMemoryTool.

        Args:
            memories_dir: Base directory for memory files (default: /memories)
            file_history: Optional dictionary tracking edit history per file
        """
        # Store file history before parent inits (BaseFileMemoryTool may reset self.env)
        _file_history = file_history or defaultdict(list)

        # Initialize EditTool with file history
        EditTool.__init__(self, file_history=_file_history)

        # Initialize BaseFileMemoryTool for path handling
        BaseFileMemoryTool.__init__(
            self,
            base_path=memories_dir,
            memory_section_header="## Memories",
        )

        # Restore file history (BaseFileMemoryTool may have reset self.env)
        self.env = _file_history

        # Override name/title/description for memory
        self.name = "memory"
        self.title = "Memory"
        self.description = "Store and retrieve persistent information across conversations"

    def _resolve_memory_path(self, path: str) -> Path:
        """Validate and resolve a path within the memories directory.

        For backwards compatibility - delegates to resolve_path().
        """
        # Handle /memories prefix
        if path.startswith("/memories"):
            relative_path = path[len("/memories") :].lstrip("/")
        else:
            relative_path = path.lstrip("/")

        return self.resolve_path(relative_path)

    def validate_path(self, command: str, path: Path) -> None:
        """Override parent validation - we use _resolve_memory_path instead."""
        return

    async def __call__(
        self,
        *,
        command: ClaudeMemoryCommand,  # type: ignore[override]
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
        """Execute a memory command.

        Args:
            command: The command to execute
            path: Path for view, create, str_replace, insert, delete
            view_range: Line range for view [start, end]
            file_text: Content for create
            old_str: Text to replace for str_replace
            new_str: Replacement text for str_replace
            insert_line: Line number for insert
            insert_text: Text to insert for insert command
            old_path: Source path for rename
            new_path: Destination path for rename

        Returns:
            List of MCP ContentBlocks with the result
        """
        if command == "view":
            if path is None:
                path = "/memories"
            result = await self._memory_view(path, view_range)
            return result.to_content_blocks()

        elif command == "create":
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

        elif command == "str_replace":
            if path is None:
                raise ToolError("path is required for command: str_replace")
            if old_str is None:
                raise ToolError("old_str is required for command: str_replace")
            resolved = self._resolve_memory_path(path)
            if not resolved.exists() or resolved.is_dir():
                raise ToolError(
                    f"Error: The path {path} does not exist. Please provide a valid path."
                )
            result = await self.str_replace(resolved, old_str, new_str)
            if result.output:
                result = ContentResult(output=result.output.replace("The file", "The memory file"))
            return result.to_content_blocks()

        elif command == "insert":
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

        elif command == "delete":
            if path is None:
                raise ToolError("path is required for command: delete")
            result = await self._memory_delete(path)
            return result.to_content_blocks()

        elif command == "rename":
            if old_path is None:
                raise ToolError("old_path is required for command: rename")
            if new_path is None:
                raise ToolError("new_path is required for command: rename")
            result = await self._memory_rename(old_path, new_path)
            return result.to_content_blocks()

        allowed = ", ".join(get_args(ClaudeMemoryCommand))
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
            # Custom directory listing with sizes
            lines = []
            for item in sorted(resolved.rglob("*")):
                # Limit to 2 levels deep
                relative = item.relative_to(resolved)
                if len(relative.parts) > 2:
                    continue
                # Skip hidden files
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

        # File content - reuse parent's view logic
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


__all__ = ["ClaudeMemoryCommand", "ClaudeMemoryTool"]
