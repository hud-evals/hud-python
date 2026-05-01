"""OpenAI-compatible coding tools inspired by OpenCode's filesystem tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hud.agents.tools import AgentTool, AgentToolSpec, GroupedCapabilityMixin

from .types import OpenAICompatibleToolParam

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam
    from openai.types.shared_params.function_parameters import FunctionParameters

READ_PARAMETERS: FunctionParameters = {
    "type": "object",
    "properties": {
        "filePath": {
            "type": "string",
            "description": "Absolute path to the file to read.",
        },
        "offset": {
            "type": "integer",
            "description": "0-based line offset to start reading from.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read.",
        },
    },
    "required": ["filePath"],
}

GREP_PARAMETERS: FunctionParameters = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regular expression pattern to search for.",
        },
        "path": {
            "type": "string",
            "description": "Directory to search in.",
        },
        "include": {
            "type": "string",
            "description": "Glob pattern for files to include.",
        },
    },
    "required": ["pattern"],
}

GLOB_PARAMETERS: FunctionParameters = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match.",
        },
        "path": {
            "type": "string",
            "description": "Directory to search from.",
        },
    },
    "required": ["pattern"],
}

LIST_PARAMETERS: FunctionParameters = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to list.",
        },
        "ignore": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Glob patterns to ignore.",
        },
    },
}


class FilesystemTool(GroupedCapabilityMixin, AgentTool[OpenAICompatibleToolParam]):
    """Function tool backed by a HUD filesystem environment tool."""

    description: ClassVar[str]
    parameters: ClassVar[FunctionParameters]
    env_tool_names: ClassVar[tuple[str, ...]]

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return AgentToolSpec(api_type="function", api_name=cls.name)

    def to_params(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ReadTool(FilesystemTool):
    """Expose a read function over the environment read tool."""

    name = "read"
    capability = "filesystem"
    env_tool_names = ("read",)
    description = (
        "Reads a file from the local filesystem. Use offset and limit for pagination."
    )
    parameters: ClassVar[FunctionParameters] = READ_PARAMETERS


class GrepTool(FilesystemTool):
    """Expose a grep function over the environment grep tool."""

    name = "grep"
    capability = "filesystem"
    env_tool_names = ("grep",)
    description = (
        "Searches file contents using a regular expression and returns matching lines."
    )
    parameters: ClassVar[FunctionParameters] = GREP_PARAMETERS


class GlobTool(FilesystemTool):
    """Expose a glob function over the environment glob tool."""

    name = "glob"
    capability = "filesystem"
    env_tool_names = ("glob",)
    description = "Finds files matching a glob pattern."
    parameters: ClassVar[FunctionParameters] = GLOB_PARAMETERS


class ListTool(FilesystemTool):
    """Expose a list function over the environment list tool."""

    name = "list"
    capability = "filesystem"
    env_tool_names = ("list",)
    description = "Lists files and directories in a given path."
    parameters: ClassVar[FunctionParameters] = LIST_PARAMETERS


__all__ = [
    "GLOB_PARAMETERS",
    "GREP_PARAMETERS",
    "LIST_PARAMETERS",
    "READ_PARAMETERS",
    "FilesystemTool",
    "GlobTool",
    "GrepTool",
    "ListTool",
    "ReadTool",
]
