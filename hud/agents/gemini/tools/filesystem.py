"""Gemini filesystem tools — read, search, glob, list — backed by SSHClient."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, ClassVar

from hud.agents.tools.base import AgentToolSpec, result_text
from hud.agents.tools.ssh import SSHTool
from hud.types import TextContent, ToolResult

from .coding import required_str, tool_decl

if TYPE_CHECKING:
    from google.genai import types as genai_types

GEMINI_READ_SPEC = AgentToolSpec(api_type="read_file", api_name="read_file")
GEMINI_SEARCH_SPEC = AgentToolSpec(api_type="grep_search", api_name="grep_search")
GEMINI_GLOB_SPEC = AgentToolSpec(api_type="glob", api_name="glob")
GEMINI_LIST_SPEC = AgentToolSpec(api_type="list_directory", api_name="list_directory")


class GeminiReadTool(SSHTool):
    name = "read_file"
    description: ClassVar[str] = "Reads and returns the content of a specified file."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "start_line": {"type": "integer", "description": "1-based line to start at."},
            "end_line": {"type": "integer", "description": "1-based inclusive line to end at."},
        },
        "required": ["file_path"],
    }

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return GEMINI_READ_SPEC

    def to_params(self) -> genai_types.Tool:
        return tool_decl(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = required_str(arguments, "file_path")
        result = await self.file_read(path)
        if result.isError:
            return result
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        if isinstance(start, int) and start > 0:
            lines = result_text(result).splitlines(keepends=True)
            offset = start - 1
            limit = (end - start + 1) if isinstance(end, int) and end >= start else len(lines)
            sliced = lines[offset : offset + limit]
            return ToolResult(
                content=[TextContent(type="text", text="".join(sliced))],
            )
        return result


class GeminiSearchTool(SSHTool):
    name = "grep_search"
    description: ClassVar[str] = "Searches file contents using a regular expression pattern."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "dir_path": {"type": "string", "description": "Directory to search."},
            "include_pattern": {"type": "string", "description": "Glob filter."},
        },
        "required": ["pattern"],
    }

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return GEMINI_SEARCH_SPEC

    def to_params(self) -> genai_types.Tool:
        return tool_decl(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = required_str(arguments, "pattern")
        dir_path = arguments.get("dir_path") or "."
        include = arguments.get("include_pattern")
        return await self.file_grep(
            pattern, str(dir_path), include if isinstance(include, str) else None
        )


class GeminiGlobTool(SSHTool):
    name = "glob"
    description: ClassVar[str] = "Find files matching a glob pattern."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "dir_path": {"type": "string", "description": "Directory to search."},
        },
        "required": ["pattern"],
    }

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return GEMINI_GLOB_SPEC

    def to_params(self) -> genai_types.Tool:
        return tool_decl(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = required_str(arguments, "pattern")
        dir_path = arguments.get("dir_path") or "."
        cmd = f"find {shlex.quote(str(dir_path))} -name {shlex.quote(pattern)}"
        return await self.bash(cmd)


class GeminiListTool(SSHTool):
    name = "list_directory"
    description: ClassVar[str] = "Lists files and directories in a given path."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "dir_path": {"type": "string", "description": "Directory to list."},
        },
        "required": ["dir_path"],
    }

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return GEMINI_LIST_SPEC

    def to_params(self) -> genai_types.Tool:
        return tool_decl(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return await self.file_list(required_str(arguments, "dir_path"))
