"""Agent-owned OpenAI tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from mcp.types import TextContent
from openai.types.responses import FunctionShellToolParam, ResponseInputItemParam, ToolParam

from hud.types import MCPToolCall, MCPToolResult

from .base import OpenAITool, OpenAIToolSpec

if TYPE_CHECKING:
    from hud.agents.tools.base import CallTool

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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
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
            result = await super().execute(
                call_tool,
                {"command": command, **env_arguments},
            )
            text = _result_text(result)
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

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> ResponseInputItemParam:
        structured = result.structuredContent if isinstance(result.structuredContent, dict) else {}
        output = structured.get("output")
        if not isinstance(output, list):
            output = [_shell_output("", _result_text(result), 1 if result.isError else 0)]

        response: dict[str, Any] = {
            "type": "shell_call_output",
            "call_id": call.id,
            "status": "completed",
            "output": output,
        }
        max_output_length = structured.get("max_output_length")
        if isinstance(max_output_length, int):
            response["max_output_length"] = max_output_length
        return cast("ResponseInputItemParam", response)


def _shell_result(
    text: str,
    *,
    is_error: bool = False,
    structured: dict[str, Any] | None = None,
) -> MCPToolResult:
    payload = {"provider_tool": "shell", **(structured or {})}
    return MCPToolResult(
        content=[TextContent(type="text", text=text)] if text else [],
        isError=is_error,
        structuredContent=payload,
    )


def _result_text(result: MCPToolResult) -> str:
    parts = [block.text for block in result.content if isinstance(block, TextContent)]
    return "\n".join(part for part in parts if part)


def _shell_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "outcome": {"type": "exit", "exit_code": exit_code},
    }
