"""Environment file-edit tool."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args

from mcp.types import ContentBlock  # noqa: TC002 - used at runtime by FunctionTool

from hud.agents.types import ContentResult, ToolError

from ..base import BaseTool

from .utils import SNIPPET_LINES, make_snippet, read_file_async, write_file_async

Command = Literal[
    "read",
    "view",
    "create",
    "write",
    "delete",
    "replace",
    "insert",
    "undo",
]


class EditTool(BaseTool):
    """Environment tool for viewing, creating, and editing files.

    Uses str_replace operations for precise text modifications.
    Maintains a history of file edits for undo functionality.
    """

    def __init__(
        self,
        file_history: dict[Path, list[str]] | None = None,
        base_path: str | Path | None = None,
        name: str = "edit",
        title: str = "File Editor",
        description: str = "View, create, and edit files with undo support",
    ) -> None:
        """Initialize EditTool with optional file history.

        Args:
            file_history: Optional dictionary tracking edit history per file.
                         If not provided, a new history will be created.
        """
        super().__init__(
            env=file_history or defaultdict(list),
            name=name,
            title=title,
            description=description,
            meta={"capability": "editor"},
        )
        self.base_path = Path(base_path).resolve() if base_path is not None else None

    @property
    def file_history(self) -> dict[Path, list[str]]:
        """Get the file edit history."""
        return self.env

    async def __call__(
        self,
        *,
        command: Command | None = None,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        insert_line: int | None = None,
        insert_text: str | None = None,
    ) -> list[ContentBlock]:
        if command is None:
            raise ToolError("Parameter `command` is required.")

        _path = self._resolve_path(Path(path))
        self.validate_path(command, _path)

        if command == "read":
            result = await self.read(_path)
            return result.to_content_blocks()
        elif command == "view":
            result = await self.view(_path, view_range)
            return result.to_content_blocks()
        elif command == "create":
            if file_text is None:
                raise ToolError("Parameter `file_text` is required for command: create")
            await write_file_async(_path, file_text)
            self.file_history[_path].append(file_text)
            return ContentResult(
                output=f"File created successfully at: {_path}"
            ).to_content_blocks()
        elif command == "write":
            if file_text is None:
                raise ToolError("Parameter `file_text` is required for command: write")
            old_text = await read_file_async(_path) if _path.exists() else ""
            _path.parent.mkdir(parents=True, exist_ok=True)
            _path.write_text(file_text)
            self.file_history[_path].append(old_text)
            result = ContentResult(output=f"File written successfully at: {_path}")
            return result.to_content_blocks()
        elif command == "delete":
            if _path.is_dir():
                raise ToolError(f"The path {_path} is a dir and cannot be deleted by edit.")
            old_text = await read_file_async(_path)
            _path.unlink()
            self.file_history[_path].append(old_text)
            result = ContentResult(output=f"File deleted successfully at: {_path}")
            return result.to_content_blocks()
        elif command == "replace":
            if old_text is None:
                raise ToolError("Parameter `old_text` is required for command: replace")
            result = await self.replace(_path, old_text, new_text)
            return result.to_content_blocks()
        elif command == "insert":
            if insert_line is None:
                raise ToolError("Parameter `insert_line` is required for command: insert")
            if insert_text is None:
                raise ToolError("Parameter `insert_text` is required for command: insert")
            result = await self.insert(_path, insert_line, insert_text)
            return result.to_content_blocks()
        elif command == "undo":
            result = await self.undo_edit(_path)
            return result.to_content_blocks()

        raise ToolError(
            f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: "
            f"{', '.join(get_args(Command))}"
        )

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute() or self.base_path is None:
            return path
        resolved = (self.base_path / path).resolve()
        if resolved != self.base_path and self.base_path not in resolved.parents:
            raise ToolError(f"Path traversal detected: {path}")
        return resolved

    def validate_path(self, command: str, path: Path) -> None:
        """Check that the path/command combination is valid."""
        if not path.is_absolute():
            if sys.platform == "win32":
                raise ToolError(
                    f"The path {path} is not an absolute path. "
                    f"On Windows, use a full path like C:\\Users\\...\\{path.name}"
                )
            suggested_path = Path("") / path
            raise ToolError(
                f"The path {path} is not an absolute path, it should start with `/`. "
                f"Maybe you meant {suggested_path}?"
            )
        if not path.exists() and command in {"read", "view", "delete", "replace", "insert"}:
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
        if path.exists() and command == "create":
            raise ToolError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        if path.is_dir() and command != "view":
            raise ToolError(
                f"The path {path} is a dir and only the `view` command can be used on dirs."
            )

    async def read(self, path: Path) -> ContentResult:
        """Read a file without snippet formatting."""
        return ContentResult(output=await read_file_async(path))

    async def view(self, path: Path, view_range: list[int] | None = None) -> ContentResult:
        """Implement the view command."""
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )
            import shlex

            from ..utils import run

            safe_path = shlex.quote(str(path))
            _, stdout, stderr = await run(rf"find {safe_path} -maxdepth 2 -not -path '*/\.*'")
            if not stderr:
                stdout = (
                    f"Here's the files and directories up to 2 levels deep in {path}, "
                    f"excluding hidden items:\n{stdout}\n"
                )
            return ContentResult(output=stdout, error=stderr)

        file_content = await read_file_async(path)
        init_line = 1

        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError("Invalid `view_range`. It should be a list of two integers.")
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range

            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its first element `{init_line}` "
                    f"should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` "
                    f"should be smaller than the number of lines in the file: `{n_lines_file}`"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` "
                    f"should be larger or equal than its first `{init_line}`"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return ContentResult(output=make_snippet(file_content, str(path), init_line))

    async def replace(self, path: Path, old_text: str, new_text: str | None) -> ContentResult:
        """Replace a unique text fragment in a file."""
        file_content = (await read_file_async(path)).expandtabs()
        old_text = old_text.expandtabs()
        new_text = new_text.expandtabs() if new_text is not None else ""

        occurrences = file_content.count(old_text)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_text `{old_text}` did not appear verbatim in "
                f"{path}."
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [idx + 1 for idx, line in enumerate(file_content_lines) if old_text in line]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_text `{old_text}` "
                f"in lines {lines}. Please ensure it is unique"
            )

        new_file_content = file_content.replace(old_text, new_text)
        await write_file_async(path, new_file_content)
        self.file_history[path].append(file_content)

        replacement_line = file_content.split(old_text)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_text.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        success_msg = f"The file {path} has been edited. "
        success_msg += make_snippet(snippet, f"a snippet of {path}", start_line + 1)
        success_msg += (
            "Review the changes and make sure they are as expected. "
            "Edit the file again if necessary."
        )

        return ContentResult(output=success_msg)

    async def insert(self, path: Path, insert_line: int, insert_text: str) -> ContentResult:
        """Implement the insert command."""
        file_text = (await read_file_async(path)).expandtabs()
        insert_text = insert_text.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range "
                f"of lines of the file: {[0, n_lines_file]}"
            )

        insert_text_lines = insert_text.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line] + insert_text_lines + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + insert_text_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        await write_file_async(path, new_file_text)
        self.file_history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += make_snippet(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += (
            "Review the changes and make sure they are as expected (correct indentation, "
            "no duplicate lines, etc). Edit the file again if necessary."
        )
        return ContentResult(output=success_msg)

    async def undo_edit(self, path: Path) -> ContentResult:
        """Implement the undo_edit command."""
        if not self.file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self.file_history[path].pop()
        await write_file_async(path, old_text)

        return ContentResult(
            output=f"Last edit to {path} undone successfully. {make_snippet(old_text, str(path))}"
        )


__all__ = ["SNIPPET_LINES", "Command", "EditTool"]
