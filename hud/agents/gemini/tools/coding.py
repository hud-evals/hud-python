"""Gemini coding tools — shell, edit, write — backed by SSHClient."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, ClassVar

from hud.agents.tools import SSHTool
from hud.agents.tools.base import result_text, tool_err

from .base import GeminiToolSpec, declaration, required_str

if TYPE_CHECKING:
    from google.genai import types as genai_types

    from hud.types import MCPToolResult

GEMINI_SHELL_SPEC = GeminiToolSpec(api_type="run_shell_command", api_name="run_shell_command")
GEMINI_EDIT_SPEC = GeminiToolSpec(api_type="replace", api_name="replace")
GEMINI_WRITE_SPEC = GeminiToolSpec(api_type="write_file", api_name="write_file")


class GeminiShellTool(SSHTool):
    name = "run_shell_command"
    description: ClassVar[str] = (
        "Execute a shell command. The command runs in the environment shell and may "
        "optionally be scoped to a directory."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "description": {"type": "string", "description": "Brief user-facing description."},
            "dir_path": {"type": "string", "description": "Directory to run the command in."},
        },
        "required": ["command"],
    }

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_SHELL_SPEC

    def to_params(self) -> genai_types.Tool:
        return declaration(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("command is required")
        dir_path = arguments.get("dir_path")
        if isinstance(dir_path, str) and dir_path:
            command = f"cd {shlex.quote(dir_path)} && {command}"
        return await self.bash(command)


class GeminiEditTool(SSHTool):
    name = "replace"
    description: ClassVar[str] = (
        "Replaces text within a file. Use old_string as exact literal context. "
        "Set old_string to an empty string to create a new file."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to modify."},
            "instruction": {"type": "string", "description": "Semantic description."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_EDIT_SPEC

    def to_params(self) -> genai_types.Tool:
        return declaration(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        file_path = required_str(arguments, "file_path")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        if old_string == "":
            return await self.file_write(file_path, str(new_string))
        return await self._str_replace_unique(file_path, str(old_string), str(new_string))

    async def _str_replace_unique(self, path: str, old: str, new: str) -> MCPToolResult:
        existing = await self.file_read(path)
        if existing.isError:
            return existing
        text = result_text(existing)
        count = text.count(old)
        if count == 0:
            return tool_err(f"old text not found in {path}")
        if count > 1:
            return tool_err(f"old text matches {count} times in {path}; must be unique")
        return await self.file_write(path, text.replace(old, new, 1))


class GeminiWriteTool(SSHTool):
    name = "write_file"
    description: ClassVar[str] = "Creates or overwrites a file with the provided content."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "File contents."},
        },
        "required": ["file_path", "content"],
    }

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_WRITE_SPEC

    def to_params(self) -> genai_types.Tool:
        return declaration(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        return await self.file_write(
            required_str(arguments, "file_path"),
            arguments.get("content") or "",
        )


__all__ = ["GeminiEditTool", "GeminiShellTool", "GeminiWriteTool"]
