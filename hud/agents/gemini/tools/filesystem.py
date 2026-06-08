"""Gemini filesystem tools — read, search, glob, list — backed by SSHClient."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from hud.agents.tools import SSHTool
from hud.agents.tools.base import result_text, tool_ok

from .base import GeminiToolSpec, declaration, required_str

if TYPE_CHECKING:
    from google.genai import types as genai_types

    from hud.types import MCPToolResult

GEMINI_READ_SPEC = GeminiToolSpec(api_type="read_file", api_name="read_file")
GEMINI_SEARCH_SPEC = GeminiToolSpec(api_type="grep_search", api_name="grep_search")
GEMINI_GLOB_SPEC = GeminiToolSpec(api_type="glob", api_name="glob")
GEMINI_LIST_SPEC = GeminiToolSpec(api_type="list_directory", api_name="list_directory")


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
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_READ_SPEC

    def to_params(self) -> genai_types.Tool:
        return declaration(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = required_str(arguments, "file_path")
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        if isinstance(start, int) and start > 0:
            limit = (end - start + 1) if isinstance(end, int) and end >= start else None
            return await self._read_slice(path, offset=start - 1, limit=limit)
        return await self._read_slice(path)

    async def _read_slice(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> MCPToolResult:
        result = await self.file_read(path)
        if result.isError or (offset <= 0 and limit is None):
            return result
        lines = result_text(result).splitlines(keepends=True)
        end = offset + limit if limit is not None and limit > 0 else len(lines)
        return tool_ok("".join(lines[offset:end]))


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
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_SEARCH_SPEC

    def to_params(self) -> genai_types.Tool:
        return declaration(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        pattern = required_str(arguments, "pattern")
        dir_path = arguments.get("dir_path") or "."
        include = arguments.get("include_pattern")
        cmd = f"grep -rn {_shell_quote(pattern)} {_shell_quote(str(dir_path))}"
        if isinstance(include, str) and include:
            cmd += f" --include={_shell_quote(include)}"
        return await self.bash(cmd)


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
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_GLOB_SPEC

    def to_params(self) -> genai_types.Tool:
        return declaration(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        pattern = required_str(arguments, "pattern")
        dir_path = arguments.get("dir_path") or "."
        cmd = f"find {_shell_quote(str(dir_path))} -name {_shell_quote(pattern)}"
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
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_LIST_SPEC

    def to_params(self) -> genai_types.Tool:
        return declaration(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        return await self.file_list(required_str(arguments, "dir_path"))


def _shell_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


__all__ = ["GeminiGlobTool", "GeminiListTool", "GeminiReadTool", "GeminiSearchTool"]
