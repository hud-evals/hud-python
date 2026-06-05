"""Claude coding tools — bash + str_replace text editor — backed by ``SSHClient``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types

from hud.agents.tools import SSHTool
from hud.agents.tools.base import result_text, tool_err
from hud.types import MCPToolResult

from .base import ClaudeToolSpec

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaToolBash20250124Param,
        BetaToolTextEditor20250728Param,
    )


CLAUDE_BASH_SPEC = ClaudeToolSpec(
    api_type="bash_20250124",
    api_name="bash",
)

CLAUDE_TEXT_EDITOR_SPEC = ClaudeToolSpec(
    api_type="text_editor_20250728",
    api_name="str_replace_based_edit_tool",
)


class ClaudeBashTool(SSHTool):
    """Claude's native ``bash_20250124`` schema, executed over SSH."""

    name = "bash"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        del model
        return CLAUDE_BASH_SPEC

    def to_params(self) -> BetaToolBash20250124Param:
        return cast(
            "BetaToolBash20250124Param",
            {"type": self.spec.api_type, "name": self.name},
        )

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        if arguments.get("restart"):
            # SSH session lives across calls; "restart" is a no-op for us.
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text="(restart acknowledged)")],
            )
        command = arguments.get("command")
        if not command:
            return MCPToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text",
                        text="command is required unless restart is true",
                    ),
                ],
                isError=True,
            )
        return await self.bash(command)


class ClaudeTextEditorTool(SSHTool):
    """Claude's native ``text_editor_20250728`` schema, executed over SFTP."""

    name = "str_replace_based_edit_tool"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        del model
        return CLAUDE_TEXT_EDITOR_SPEC

    @property
    def provider_name(self) -> str:
        return self.spec.api_name

    def to_params(self) -> BetaToolTextEditor20250728Param:
        return cast(
            "BetaToolTextEditor20250728Param",
            {"type": self.spec.api_type, "name": self.provider_name},
        )

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        command = arguments.get("command")
        path = arguments.get("path")
        if not isinstance(path, str):
            return tool_err("`path` is required")

        match command:
            case "view":
                return await self.file_read(path)
            case "create":
                content = arguments.get("file_text", "")
                return await self.file_write(path, str(content))
            case "str_replace":
                return await self._str_replace(
                    path,
                    arguments.get("old_str", ""),
                    arguments.get("new_str", ""),
                )
            case "insert":
                line = arguments.get("insert_line")
                text = arguments.get("new_str", "")
                if not isinstance(line, int):
                    return tool_err("`insert_line` must be an integer")
                return await self._insert(path, line, str(text))
            case _:
                return tool_err(f"unknown editor command: {command!r}")

    async def _str_replace(self, path: str, old: str, new: str) -> MCPToolResult:
        existing = await self.file_read(path)
        if existing.isError:
            return existing
        text = result_text(existing)
        count = text.count(old)
        if count == 0:
            return tool_err(f"old_str not found in {path}")
        if count > 1:
            return tool_err(f"old_str matches {count} times in {path}; must be unique")
        return await self.file_write(path, text.replace(old, new, 1))

    async def _insert(self, path: str, line: int, text: str) -> MCPToolResult:
        existing = await self.file_read(path)
        if existing.isError:
            return existing
        lines = result_text(existing).splitlines(keepends=True)
        if line < 0 or line > len(lines):
            return tool_err(f"insert_line {line} out of range (file has {len(lines)} lines)")
        if text and not text.endswith("\n"):
            text += "\n"
        lines.insert(line, text)
        return await self.file_write(path, "".join(lines))


__all__ = ["CLAUDE_BASH_SPEC", "CLAUDE_TEXT_EDITOR_SPEC", "ClaudeBashTool", "ClaudeTextEditorTool"]
