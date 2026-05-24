"""Agent-side Gemini filesystem tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hud.agents.tools.base import CallTool
    from hud.types import MCPToolResult

from hud.agents.tools import GroupedCapabilityMixin

from .base import GeminiTool, GeminiToolSpec

GEMINI_READ_SPEC = GeminiToolSpec(api_type="read_file", api_name="read_file")
GEMINI_SEARCH_SPEC = GeminiToolSpec(api_type="grep_search", api_name="grep_search")
GEMINI_GLOB_SPEC = GeminiToolSpec(api_type="glob", api_name="glob")
GEMINI_LIST_SPEC = GeminiToolSpec(api_type="list_directory", api_name="list_directory")


class GeminiFilesystemTool(GroupedCapabilityMixin, GeminiTool):
    """Gemini function tool backed by one filesystem environment primitive."""

    capability = "filesystem"
    env_tool_names: ClassVar[tuple[str, ...]]


class GeminiReadTool(GeminiFilesystemTool):
    """Translate Gemini read_file calls into the generic read env primitive."""

    name = "read_file"
    env_tool_names = ("read",)
    description = "Reads and returns the content of a specified file."
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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        offset = int(start) - 1 if isinstance(start, int) and start > 0 else None
        limit = None
        if offset is not None and isinstance(start, int) and isinstance(end, int) and end >= start:
            limit = end - start + 1
        return await super().execute(
            call_tool,
            {
                "filePath": _required_str(arguments, "file_path"),
                "offset": offset,
                "limit": limit,
            },
        )


class GeminiSearchTool(GeminiFilesystemTool):
    """Translate Gemini grep_search calls into the generic grep env primitive."""

    name = "grep_search"
    env_tool_names = ("grep",)
    description = "Searches file contents using a regular expression pattern."
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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        return await super().execute(
            call_tool,
            {
                "pattern": _required_str(arguments, "pattern"),
                "path": arguments.get("dir_path"),
                "include": arguments.get("include_pattern"),
            },
        )


class GeminiGlobTool(GeminiFilesystemTool):
    """Translate Gemini glob calls into the generic glob env primitive."""

    name = "glob"
    env_tool_names = ("glob",)
    description = "Find files matching a glob pattern."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "dir_path": {"type": "string", "description": "Directory to search."},
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether matching is case-sensitive.",
            },
        },
        "required": ["pattern"],
    }

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_GLOB_SPEC

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        return await super().execute(
            call_tool,
            {
                "pattern": _required_str(arguments, "pattern"),
                "path": arguments.get("dir_path"),
                "case_sensitive": arguments.get("case_sensitive", True),
            },
        )


class GeminiListTool(GeminiFilesystemTool):
    """Translate Gemini list_directory calls into the generic list env primitive."""

    name = "list_directory"
    env_tool_names = ("list",)
    description = "Lists files and directories in a given path."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "dir_path": {"type": "string", "description": "Directory to list."},
            "ignore": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns to ignore.",
            },
        },
        "required": ["dir_path"],
    }

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_LIST_SPEC

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        return await super().execute(
            call_tool,
            {
                "path": _required_str(arguments, "dir_path"),
                "ignore": arguments.get("ignore"),
            },
        )


def _required_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value
