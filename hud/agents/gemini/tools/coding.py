"""Agent-side Gemini coding tools."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hud.agents.tools.base import CallTool
    from hud.types import MCPToolResult

from .base import GeminiTool, GeminiToolSpec

GEMINI_SHELL_SPEC = GeminiToolSpec(api_type="run_shell_command", api_name="run_shell_command")
GEMINI_EDIT_SPEC = GeminiToolSpec(api_type="replace", api_name="replace")
GEMINI_WRITE_SPEC = GeminiToolSpec(api_type="write_file", api_name="write_file")


class GeminiShellTool(GeminiTool):
    """Translate Gemini CLI shell calls into the generic bash env primitive."""

    name = "run_shell_command"
    capability = "shell"
    description = (
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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("command is required")
        dir_path = arguments.get("dir_path")
        if isinstance(dir_path, str) and dir_path:
            command = f"cd {shlex.quote(dir_path)} && {command}"
        return await super().execute(call_tool, {"command": command})


class GeminiEditTool(GeminiTool):
    """Translate Gemini CLI replace calls into the generic edit env primitive."""

    name = "replace"
    capability = "editor"
    description = (
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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        file_path = _required_str(arguments, "file_path")
        old_string = arguments.get("old_string")
        new_string = arguments.get("new_string")
        if old_string == "":
            return await super().execute(
                call_tool,
                {
                    "command": "create",
                    "path": file_path,
                    "file_text": new_string or "",
                },
            )
        return await super().execute(
            call_tool,
            {
                "command": "replace",
                "path": file_path,
                "old_text": old_string,
                "new_text": new_string,
            },
        )


class GeminiWriteTool(GeminiTool):
    """Translate Gemini CLI write_file calls into the generic edit env primitive."""

    name = "write_file"
    capability = "editor"
    description = "Creates or overwrites a file with the provided content."
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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        return await super().execute(
            call_tool,
            {
                "command": "write",
                "path": _required_str(arguments, "file_path"),
                "file_text": arguments.get("content") or "",
            },
        )


def _required_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value
