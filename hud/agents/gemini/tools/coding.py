"""Gemini coding tools — shell, edit, write — backed by SSHClient."""

from __future__ import annotations

import shlex
from typing import Any, ClassVar

from google.genai import types as genai_types

from hud.agents.tools import SSHTool
from hud.agents.tools.base import result_text, tool_err
from hud.types import MCPToolResult

from .base import GeminiToolSpec

GEMINI_SHELL_SPEC = GeminiToolSpec(api_type="run_shell_command", api_name="run_shell_command")
GEMINI_EDIT_SPEC = GeminiToolSpec(api_type="replace", api_name="replace")
GEMINI_WRITE_SPEC = GeminiToolSpec(api_type="write_file", api_name="write_file")


def _decl(name: str, description: str, parameters: dict[str, Any]) -> genai_types.Tool:
    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name=name,
                description=description,
                parameters_json_schema=parameters,
            ),
        ],
    )


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
        return _decl(self.name, self.description, self.parameters)

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
        return _decl(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        file_path = _required_str(arguments, "file_path")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        if old_string == "":
            return await self.file_write(file_path, str(new_string))
        existing = await self.file_read(file_path)
        if existing.isError:
            return existing
        text = result_text(existing)
        if str(old_string) not in text:
            return tool_err(f"old_string not found in {file_path}")
        return await self.file_write(file_path, text.replace(str(old_string), str(new_string), 1))


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
        return _decl(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        return await self.file_write(
            _required_str(arguments, "file_path"),
            arguments.get("content") or "",
        )


def _required_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value


__all__ = ["GeminiEditTool", "GeminiShellTool", "GeminiWriteTool"]
