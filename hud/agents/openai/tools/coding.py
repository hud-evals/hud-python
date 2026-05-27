"""OpenAI shell tool — backed by SSHClient."""

from __future__ import annotations

from typing import Any, cast

import mcp.types as mcp_types

from hud.agents.tools import SSHTool
from hud.agents.tools.ssh import result_text
from hud.types import MCPToolResult

from .base import OpenAIToolSpec

try:
    from openai.types.responses import FunctionShellToolParam, ToolParam
except Exception:
    ToolParam = Any  # type: ignore[assignment,misc]

OPENAI_SHELL_SPEC = OpenAIToolSpec(
    api_type="shell",
    api_name="shell",
    supported_models=("gpt-5.4", "gpt-5.4-*", "gpt-5.5", "gpt-5.5-*"),
)


class OpenAIShellTool(SSHTool):
    name = "shell"

    @classmethod
    def default_spec(cls, model: str) -> OpenAIToolSpec | None:
        return OPENAI_SHELL_SPEC if OPENAI_SHELL_SPEC.supports_model(model) else None

    def to_params(self) -> Any:
        return cast(
            "ToolParam",
            FunctionShellToolParam(type="shell", environment={"type": "local"}),
        )

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        def invalid_commands_result() -> MCPToolResult:
            text = "commands must be a list of strings"
            return _shell_result(
                text,
                is_error=True,
                structured={
                    "output": [_shell_output("", text, 1)],
                    "max_output_length": arguments.get("max_output_length"),
                },
            )

        commands = arguments.get("commands")
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list):
            return invalid_commands_result()
        raw_commands = cast("list[Any]", commands)
        if not all(isinstance(cmd, str) for cmd in raw_commands):
            return invalid_commands_result()
        command_list = cast("list[str]", raw_commands)

        outputs: list[dict[str, Any]] = []
        text_parts: list[str] = []
        is_error = False
        env_arguments: dict[str, Any] = {}
        timeout_ms = arguments.get("timeout_ms")
        if isinstance(timeout_ms, int):
            env_arguments["timeout_seconds"] = timeout_ms / 1000.0

        for command in command_list:
            if env_arguments.get("timeout_seconds"):
                full_cmd = f"timeout {int(env_arguments['timeout_seconds'])} {command}"
            else:
                full_cmd = command
            result = await self.bash(full_cmd)
            text = result_text(result)
            if result.isError:
                outputs.append(_shell_output("", text, 1))
                is_error = True
            else:
                outputs.append(_shell_output(text, "", 0))
            if text:
                text_parts.append(text)

        return _shell_result(
            "\n".join(text_parts),
            is_error=is_error,
            structured={
                "output": outputs,
                "max_output_length": arguments.get("max_output_length"),
            },
        )


def _shell_result(
    text: str,
    *,
    is_error: bool = False,
    structured: dict[str, Any] | None = None,
) -> MCPToolResult:
    payload = {"provider_tool": "shell", **(structured or {})}
    return MCPToolResult(
        content=[mcp_types.TextContent(type="text", text=text)] if text else [],
        isError=is_error,
        structuredContent=payload,
    )


def _shell_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "outcome": {"type": "exit", "exit_code": exit_code},
    }


__all__ = ["OPENAI_SHELL_SPEC", "OpenAIShellTool"]
