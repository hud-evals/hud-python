"""OpenAI-compatible filesystem tools — backed by SSHClient."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Any, ClassVar

from hud.agents.tools import SSHTool
from hud.agents.tools.base import AgentToolSpec, result_text, tool_ok

if TYPE_CHECKING:
    from hud.types import MCPToolResult


class _FilesystemTool(SSHTool):
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return AgentToolSpec(api_type="function", api_name=cls.name)

    def to_params(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ReadTool(_FilesystemTool):
    name = "read"
    description = "Reads a file from the local filesystem. Use offset and limit for pagination."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "Absolute path to the file to read."},
            "offset": {
                "type": "integer",
                "description": "0-based line offset to start reading from.",
            },
            "limit": {"type": "integer", "description": "Maximum number of lines to read."},
        },
        "required": ["filePath"],
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = arguments.get("filePath")
        if not isinstance(path, str) or not path:
            raise ValueError("filePath is required")
        offset = arguments.get("offset")
        if isinstance(offset, int) and offset >= 0:
            limit = arguments.get("limit")
            return await self._read_slice(
                path,
                offset=offset,
                limit=limit if isinstance(limit, int) and limit > 0 else None,
            )
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


class GrepTool(_FilesystemTool):
    name = "grep"
    description = "Searches file contents using a regular expression and returns matching lines."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for.",
            },
            "path": {"type": "string", "description": "Directory to search in."},
            "include": {"type": "string", "description": "Glob pattern for files to include."},
        },
        "required": ["pattern"],
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str):
            raise ValueError("pattern is required")
        path = arguments.get("path") or "."
        cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(str(path))}"
        include = arguments.get("include")
        if isinstance(include, str) and include:
            cmd += f" --include={shlex.quote(include)}"
        return await self.bash(cmd)


class GlobTool(_FilesystemTool):
    name = "glob"
    description = "Finds files matching a glob pattern."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match."},
            "path": {"type": "string", "description": "Directory to search from."},
        },
        "required": ["pattern"],
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str):
            raise ValueError("pattern is required")
        path = arguments.get("path") or "."
        return await self.bash(f"find {shlex.quote(str(path))} -name {shlex.quote(pattern)}")


class ListTool(_FilesystemTool):
    name = "list"
    description = "Lists files and directories in a given path."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list."},
        },
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = arguments.get("path") or "."
        return await self.file_list(str(path))


__all__ = ["GlobTool", "GrepTool", "ListTool", "ReadTool"]
