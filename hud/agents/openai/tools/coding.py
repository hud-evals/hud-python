"""Agent-owned OpenAI tools."""

from __future__ import annotations

from typing import Any, cast

from mcp.types import TextContent
from openai.types.responses import FunctionShellToolParam, ToolParam

from hud.types import MCPToolCall, MCPToolResult

from .base import CallTool, OpenAITool, OpenAIToolSpec, call_tool, result_text

OPENAI_SHELL_SPEC = OpenAIToolSpec(
    api_type="shell",
    api_name="shell",
    supported_models=(
        "gpt-5.4",
        "gpt-5.4-*",
        "gpt-5.5",
        "gpt-5.5-*",
    ),
)


class OpenAIShellTool(OpenAITool):
    """OpenAI shell provider tool backed by an environment bash tool."""

    name = "shell"
    capability = "shell"

    @classmethod
    def default_spec(cls, model: str) -> OpenAIToolSpec | None:
        if OPENAI_SHELL_SPEC.supports_model(model):
            return OPENAI_SHELL_SPEC
        return None

    def __init__(self, *, env_tool_name: str, spec: OpenAIToolSpec) -> None:
        del spec
        super().__init__(env_tool_name=env_tool_name, spec=OPENAI_SHELL_SPEC)

    def to_params(self) -> ToolParam:
        return cast(
            "ToolParam",
            FunctionShellToolParam(type="shell", environment={"type": "local"}),
        )

    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        commands = arguments.get("commands")
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list) or not all(isinstance(cmd, str) for cmd in commands):
            return _provider_result(
                "shell",
                "commands must be a list of strings",
                is_error=True,
                structured={
                    "output": [_shell_output("", "commands must be a list of strings", 1)],
                    "max_output_length": arguments.get("max_output_length"),
                },
            )

        outputs: list[dict[str, Any]] = []
        text_parts: list[str] = []
        is_error = False
        env_arguments: dict[str, Any] = {}
        timeout_ms = arguments.get("timeout_ms")
        if isinstance(timeout_ms, int):
            env_arguments["timeout_seconds"] = timeout_ms / 1000.0
        for command in commands:
            result = await call_tool(
                caller,
                self.env_tool_name,
                {"command": command, **env_arguments},
            )
            text = result_text(result)
            if result.isError:
                outputs.append(_shell_output("", text, 1))
                is_error = True
            else:
                outputs.append(_shell_output(text, "", 0))
            if text:
                text_parts.append(text)

        return _provider_result(
            "shell",
            "\n".join(text_parts),
            is_error=is_error,
            structured={
                "output": outputs,
                "max_output_length": arguments.get("max_output_length"),
            },
        )

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> dict[str, Any]:
        structured = result.structuredContent if isinstance(result.structuredContent, dict) else {}
        output = structured.get("output")
        if not isinstance(output, list):
            output = [_shell_output("", result_text(result), 1 if result.isError else 0)]

        response: dict[str, Any] = {
            "type": "shell_call_output",
            "call_id": call.id,
            "status": "completed",
            "output": output,
        }
        max_output_length = structured.get("max_output_length")
        if isinstance(max_output_length, int):
            response["max_output_length"] = max_output_length
        return response


def _provider_result(
    provider_tool: str,
    text: str,
    *,
    is_error: bool = False,
    structured: dict[str, Any] | None = None,
) -> MCPToolResult:
    payload = {"provider_tool": provider_tool, **(structured or {})}
    return MCPToolResult(
        content=[TextContent(type="text", text=text)] if text else [],
        isError=is_error,
        structuredContent=payload,
    )


def _shell_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "outcome": {"type": "exit", "exit_code": exit_code},
    }


__all__ = [
    "OPENAI_SHELL_SPEC",
    "OpenAIShellTool",
]
