"""Agent-side Claude native coding tools backed by environment tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mcp.types import TextContent

from hud.types import MCPToolResult

from .base import CallTool, ClaudeTool, ClaudeToolSpec, call_tool

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaToolBash20250124Param,
        BetaToolTextEditor20250728Param,
    )


CLAUDE_BASH_SPEC = ClaudeToolSpec(
    api_type="bash_20250124",
    api_name="bash",
    supported_models=(
        "*claude-opus-4-7*",
        "*claude-opus-4-6*",
        "*claude-sonnet-4-5*",
        "*claude-sonnet-4-6*",
        "*claude-haiku-4-5*",
    ),
)

CLAUDE_TEXT_EDITOR_SPECS: tuple[ClaudeToolSpec, ...] = (
    ClaudeToolSpec(
        api_type="text_editor_20250728",
        api_name="str_replace_based_edit_tool",
        supported_models=(
            "*claude-opus-4-7*",
            "*claude-opus-4-6*",
            "*claude-sonnet-4-5*",
            "*claude-sonnet-4-6*",
            "*claude-haiku-4-5*",
        ),
    ),
)

CLAUDE_TEXT_EDITOR_SPEC = CLAUDE_TEXT_EDITOR_SPECS[0]

CLAUDE_TEXT_EDITOR_NAMES = {
    "text_editor_20250728": "str_replace_based_edit_tool",
}

CLAUDE_TEXT_EDITOR_COMMANDS = {
    "text_editor_20250728": frozenset({"view", "create", "str_replace", "insert"}),
}


class ClaudeBashTool(ClaudeTool):
    """Claude bash provider tool backed by an environment shell tool."""

    name = "bash"
    capability = "shell"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        if CLAUDE_BASH_SPEC.supports_model(model):
            return CLAUDE_BASH_SPEC
        return None

    def __init__(self, *, env_tool_name: str, spec: ClaudeToolSpec) -> None:
        del spec
        super().__init__(env_tool_name=env_tool_name, spec=CLAUDE_BASH_SPEC)

    def to_params(self) -> BetaToolBash20250124Param:
        return cast(
            "BetaToolBash20250124Param",
            {
                "type": "bash_20250124",
                "name": self.name,
            },
        )

    async def execute(
        self,
        caller: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        if not arguments.get("restart") and "command" not in arguments:
            return MCPToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="command is required unless restart is true",
                    )
                ],
                isError=True,
            )
        return await call_tool(caller, self.env_tool_name, arguments)


class ClaudeTextEditorTool(ClaudeTool):
    """Claude text editor provider tool backed by an environment editor tool."""

    name = "str_replace_based_edit_tool"
    capability = "editor"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        for spec in CLAUDE_TEXT_EDITOR_SPECS:
            if spec.supports_model(model):
                return spec
        return None

    def __init__(self, *, env_tool_name: str, spec: ClaudeToolSpec) -> None:
        super().__init__(env_tool_name=env_tool_name, spec=spec)

    @property
    def provider_name(self) -> str:
        return CLAUDE_TEXT_EDITOR_NAMES.get(self.spec.api_type, self.spec.api_name)

    def to_params(self) -> BetaToolTextEditor20250728Param:
        return cast(
            "BetaToolTextEditor20250728Param",
            {
                "type": self.spec.api_type,
                "name": self.provider_name,
            },
        )

    async def execute(
        self,
        caller: CallTool,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        command = arguments.get("command")
        allowed_commands = CLAUDE_TEXT_EDITOR_COMMANDS.get(self.spec.api_type)
        if allowed_commands is not None and command not in allowed_commands:
            return MCPToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"{self.spec.api_type} does not support command {command!r}. "
                            f"Supported commands: {', '.join(sorted(allowed_commands))}"
                        ),
                    )
                ],
                isError=True,
            )
        return await call_tool(caller, self.env_tool_name, _claude_editor_arguments(arguments))


def _claude_editor_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    command = arguments.get("command")
    match command:
        case "str_replace":
            translated = {
                "command": "replace",
                "path": arguments.get("path"),
                "old_text": arguments.get("old_str"),
            }
            if "new_str" in arguments:
                translated["new_text"] = arguments.get("new_str")
            return translated
        case "insert":
            return {
                "command": "insert",
                "path": arguments.get("path"),
                "insert_line": arguments.get("insert_line"),
                "insert_text": arguments.get("new_str"),
            }
        case _:
            return dict(arguments)


__all__ = [
    "CLAUDE_BASH_SPEC",
    "CLAUDE_TEXT_EDITOR_SPEC",
    "CLAUDE_TEXT_EDITOR_SPECS",
    "ClaudeBashTool",
    "ClaudeTextEditorTool",
]
