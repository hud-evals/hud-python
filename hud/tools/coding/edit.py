"""Edit tool for Claude agents.

This tool conforms to Anthropic's text_editor tool specification and is used
when running with Claude models that support native str_replace editing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Literal, get_args

from mcp.types import ContentBlock  # noqa: TC002 - used at runtime by FunctionTool

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType

from .utils import SNIPPET_LINES, make_snippet, read_file_async, write_file_async

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]


class EditTool(BaseTool):
    """A filesystem editor tool for viewing, creating, and editing files.

    Uses str_replace operations for precise text modifications.
    Maintains a history of file edits for undo functionality.

    Native specs: Claude (text_editor_20250728)
    Role: "editor" (mutually exclusive with ApplyPatchTool)
    Supported models: Claude 3.5 Sonnet, 3.7 Sonnet, Sonnet 4, Opus 4
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.CLAUDE: NativeToolSpec(
            api_type="text_editor_20250728",
            api_name="str_replace_based_edit_tool",
            role="editor",
            supported_models=(
                "claude-3-5-sonnet-*",
                "claude-3-7-sonnet-*",
                "claude-sonnet-4-*",
                "claude-opus-4-*",
                "claude-4-5-sonnet-*",
                "claude-4-5-opus-*",
            ),
        ),
    }

    def __init__(self, file_history: dict[Path, list[str]] | None = None) -> None:
        """Initialize EditTool with optional file history.

        Args:
            file_history: Optional dictionary tracking edit history per file.
                         If not provided, a new history will be created.
        """
        super().__init__(
            env=file_history or defaultdict(list),
            name="edit",  # Generic name; Claude uses api_name override
            title="File Editor",
            description="View, create, and edit files with undo support",
        )

    @property
    def file_history(self) -> dict[Path, list[str]]:
        """Get the file edit history."""
        return self.env

    async def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
    ) -> list[ContentBlock]:
        _path = Path(path)
        self.validate_path(command, _path)

        if command == "view":
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
        elif command == "str_replace":
            if old_str is None:
                raise ToolError("Parameter `old_str` is required for command: str_replace")
            result = await self.str_replace(_path, old_str, new_str)
            return result.to_content_blocks()
        elif command == "insert":
            if insert_line is None:
                raise ToolError("Parameter `insert_line` is required for command: insert")
            if new_str is None:
                raise ToolError("Parameter `new_str` is required for command: insert")
            result = await self.insert(_path, insert_line, new_str)
            return result.to_content_blocks()
        elif command == "undo_edit":
            result = await self.undo_edit(_path)
            return result.to_content_blocks()

        raise ToolError(
            f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: "
            f"{', '.join(get_args(Command))}"
        )

    def validate_path(self, command: str, path: Path) -> None:
        """Check that the path/command combination is valid."""
        if not path.is_absolute():
            suggested_path = Path("") / path
            raise ToolError(
                f"The path {path} is not an absolute path, it should start with `/`. "
                f"Maybe you meant {suggested_path}?"
            )
        if not path.exists() and command != "create":
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
        if path.exists() and command == "create":
            raise ToolError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        if path.is_dir() and command != "view":
            raise ToolError(
                f"The path {path} is a dir and only the `view` command can be used on dirs."
            )

    async def view(self, path: Path, view_range: list[int] | None = None) -> ContentResult:
        """Implement the view command."""
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )
            import shlex

            from hud.tools.utils import run

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

    async def str_replace(self, path: Path, old_str: str, new_str: str | None) -> ContentResult:
        """Implement the str_replace command."""
        file_content = (await read_file_async(path)).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in "
                f"{path}."
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [idx + 1 for idx, line in enumerate(file_content_lines) if old_str in line]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in lines {lines}. Please ensure it is unique"
            )

        new_file_content = file_content.replace(old_str, new_str)
        await write_file_async(path, new_file_content)
        self.file_history[path].append(file_content)

        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        success_msg = f"The file {path} has been edited. "
        success_msg += make_snippet(snippet, f"a snippet of {path}", start_line + 1)
        success_msg += (
            "Review the changes and make sure they are as expected. "
            "Edit the file again if necessary."
        )

        return ContentResult(output=success_msg)

    async def insert(self, path: Path, insert_line: int, new_str: str) -> ContentResult:
        """Implement the insert command."""
        file_text = (await read_file_async(path)).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range "
                f"of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line] + new_str_lines + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
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
