"""OpenAI-compatible coding tools inspired by OpenCode's filesystem tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hud.agents.tools import AgentToolSpec

from .base import OpenAICompatibleTool

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam
    from openai.types.shared_params.function_parameters import FunctionParameters


class _FilesystemTool(OpenAICompatibleTool):
    """Function tool backed by a HUD filesystem environment tool."""

    description: ClassVar[str]
    parameters: ClassVar[FunctionParameters]

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


class ReadTool(_FilesystemTool):
    """Expose a read function over the environment read tool."""

    name = "read"
    capability = "filesystem.read"
    description = "Reads a file from the local filesystem. Use offset and limit for pagination."
    parameters: ClassVar[FunctionParameters] = {
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


class GrepTool(_FilesystemTool):
    """Expose a grep function over the environment grep tool."""

    name = "grep"
    capability = "filesystem.grep"
    description = "Searches file contents using a regular expression and returns matching lines."
    parameters: ClassVar[FunctionParameters] = {
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


class GlobTool(_FilesystemTool):
    """Expose a glob function over the environment glob tool."""

    name = "glob"
    capability = "filesystem.glob"
    description = "Finds files matching a glob pattern."
    parameters: ClassVar[FunctionParameters] = {
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


class ListTool(_FilesystemTool):
    """Expose a list function over the environment list tool."""

    name = "list"
    capability = "filesystem.list"
    description = "Lists files and directories in a given path."
    parameters: ClassVar[FunctionParameters] = {
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
