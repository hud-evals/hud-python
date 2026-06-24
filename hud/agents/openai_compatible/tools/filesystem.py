"""OpenAI-compatible OpenCode-style workspace tools backed by SSHClient."""

from __future__ import annotations

import math
import posixpath
import shlex
from typing import Any, ClassVar

import mcp.types as mcp_types

from hud.agents.tools import SSHTool
from hud.agents.tools.base import AgentToolSpec, result_text, tool_err
from hud.types import MCPToolResult

DEFAULT_READ_LIMIT = 2000


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
    description = (
        "Reads a file or directory from the workspace. Use offset and limit for pagination."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "filePath": {
                "type": "string",
                "description": "The absolute path to the file or directory to read.",
            },
            "offset": {
                "type": "integer",
                "description": "The line number to start reading from (1-indexed).",
                "minimum": 0,
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of lines to read (defaults to 2000).",
                "minimum": 1,
            },
        },
        "required": ["filePath"],
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = arguments.get("filePath")
        if not isinstance(path, str) or not path:
            raise ValueError("filePath is required")
        offset = _read_offset(arguments.get("offset"))
        limit = _positive_int(arguments.get("limit"), default=DEFAULT_READ_LIMIT, name="limit")
        if not (await self.bash(f"test -d {shlex.quote(path)}")).isError:
            return await self._read_directory(path, offset=offset, limit=limit)
        result = await self.file_read(path)
        if result.isError:
            return result
        text = result_text(result)
        lines = text.splitlines()
        start = offset - 1
        if start > len(lines) and not (len(lines) == 0 and offset == 1):
            return tool_err(f"Offset {offset} is out of range for this file ({len(lines)} lines)")
        sliced = lines[start : start + limit]
        last = offset + len(sliced) - 1
        more = last < len(lines)
        body = [
            f"<path>{path}</path>",
            "<type>file</type>",
            "<content>",
            *[f"{i + offset}: {line}" for i, line in enumerate(sliced)],
        ]
        if more:
            body.append(
                f"\n(Showing lines {offset}-{last} of {len(lines)}. "
                f"Use offset={last + 1} to continue.)"
            )
        else:
            body.append(f"\n(End of file - total {len(lines)} lines)")
        body.append("</content>")
        return MCPToolResult(content=[mcp_types.TextContent(type="text", text="\n".join(body))])

    async def _read_directory(self, path: str, *, offset: int, limit: int) -> MCPToolResult:
        result = await self.file_list(path)
        if result.isError:
            return result
        entries = result_text(result).splitlines()
        if entries == ["(empty)"]:
            entries = []
        start = offset - 1
        sliced = entries[start : start + limit]
        truncated = start + len(sliced) < len(entries)
        body = [
            f"<path>{path}</path>",
            "<type>directory</type>",
            "<entries>",
            *sliced,
        ]
        if truncated:
            body.append(
                f"\n(Showing {len(sliced)} of {len(entries)} entries. "
                f"Use offset={offset + len(sliced)} to continue.)"
            )
        else:
            body.append(f"\n({len(entries)} entries)")
        body.append("</entries>")
        return MCPToolResult(content=[mcp_types.TextContent(type="text", text="\n".join(body))])


class BashTool(_FilesystemTool):
    name = "bash"
    description = (
        "Executes a shell command in the workspace. Prefer read, grep, glob, edit, "
        "and write for filesystem operations."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute."},
            "timeout": {
                "type": "integer",
                "description": "Optional timeout in milliseconds.",
                "minimum": 1,
            },
            "workdir": {
                "type": "string",
                "description": "The working directory to run the command in.",
            },
        },
        "required": ["command"],
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("command is required")
        timeout = arguments.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, int) or timeout < 1:
                raise ValueError("timeout must be a positive integer")
            seconds = max(1, math.ceil(timeout / 1000))
            command = f"timeout {seconds}s bash -lc {shlex.quote(command)}"
        workdir = arguments.get("workdir")
        if isinstance(workdir, str) and workdir:
            command = f"cd {shlex.quote(workdir)} && {command}"
        return await self.bash(command)


class EditTool(_FilesystemTool):
    name = "edit"
    description = (
        "Replaces text within a file. Use oldString as exact literal context. "
        "Set replaceAll to true to replace every occurrence."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "filePath": {
                "type": "string",
                "description": "The absolute path to the file to modify.",
            },
            "oldString": {"type": "string", "description": "The text to replace."},
            "newString": {
                "type": "string",
                "description": "The text to replace it with (must be different from oldString).",
            },
            "replaceAll": {
                "type": "boolean",
                "description": "Replace all occurrences of oldString (default false).",
            },
        },
        "required": ["filePath", "oldString", "newString"],
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = arguments.get("filePath")
        if not isinstance(path, str) or not path:
            raise ValueError("filePath is required")
        old = arguments.get("oldString")
        new = arguments.get("newString")
        if not isinstance(old, str):
            raise ValueError("oldString is required")
        if not isinstance(new, str):
            raise ValueError("newString is required")
        if old == new:
            return tool_err("No changes to apply: oldString and newString are identical.")
        if old == "":
            exists = not (await self.bash(f"test -e {shlex.quote(path)}")).isError
            if exists:
                return tool_err(
                    "oldString cannot be empty when editing an existing file. "
                    "Provide exact text to replace, or use write for full-file replacement."
                )
            mkdir = await self._ensure_parent(path)
            if mkdir.isError:
                return mkdir
            return await self.file_write(path, new)

        existing = await self.file_read(path)
        if existing.isError:
            return existing
        text = result_text(existing)
        count = text.count(old)
        if count == 0:
            return tool_err(f"oldString not found in {path}")
        replace_all = arguments.get("replaceAll") is True
        if count > 1 and not replace_all:
            return tool_err(f"oldString matches {count} times in {path}; set replaceAll to true")
        next_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        return await self.file_write(path, next_text)

    async def _ensure_parent(self, path: str) -> MCPToolResult:
        parent = posixpath.dirname(path)
        if not parent or parent in {".", "/"}:
            return MCPToolResult(content=[])
        return await self.bash(f"mkdir -p {shlex.quote(parent)}")


class WriteTool(_FilesystemTool):
    name = "write"
    description = "Creates or overwrites a file with the provided content."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The content to write to the file."},
            "filePath": {
                "type": "string",
                "description": "The absolute path to the file to write.",
            },
        },
        "required": ["content", "filePath"],
    }

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        path = arguments.get("filePath")
        if not isinstance(path, str) or not path:
            raise ValueError("filePath is required")
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("content is required")
        mkdir = await self._ensure_parent(path)
        if mkdir.isError:
            return mkdir
        return await self.file_write(path, content)

    async def _ensure_parent(self, path: str) -> MCPToolResult:
        parent = posixpath.dirname(path)
        if not parent or parent in {".", "/"}:
            return MCPToolResult(content=[])
        return await self.bash(f"mkdir -p {shlex.quote(parent)}")


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


def _positive_int(value: Any, *, default: int, name: str) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_offset(value: Any) -> int:
    if value is None or value == 0:
        return 1
    return _positive_int(value, default=1, name="offset")


__all__ = ["BashTool", "EditTool", "GlobTool", "GrepTool", "ReadTool", "WriteTool"]
