"""OpenAI shell tool — backed by SSHClient."""

from __future__ import annotations

import shlex
from typing import Any, cast

import mcp.types as mcp_types

from hud.agents.tools import SSHTool
from hud.agents.tools.ssh import MAX_SHELL_OUTPUT_LENGTH, bound_shell_output
from hud.types import MCPToolResult

from .base import OpenAIToolSpec

OPENAI_SHELL_SPEC = OpenAIToolSpec(
    api_type="shell",
    api_name="shell",
)


class OpenAIShellTool(SSHTool):
    name = "shell"

    @classmethod
    def default_spec(cls, model: str) -> OpenAIToolSpec:
        del model
        return OPENAI_SHELL_SPEC

    def to_params(self) -> Any:
        # openai.types.responses.FunctionShellToolParam, as a plain dict (TypedDicts
        # are dicts at runtime, and the param type isn't present in all SDK versions).
        return {"type": "shell", "environment": {"type": "local"}}

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        requested_limit = arguments.get("max_output_length")
        if requested_limit is None:
            max_output_length = MAX_SHELL_OUTPUT_LENGTH
        elif (
            isinstance(requested_limit, bool)
            or not isinstance(requested_limit, int)
            or requested_limit <= 0
        ):
            text = "max_output_length must be a positive integer"
            return _shell_result(
                [text],
                is_error=True,
                structured={"output": [shell_output("", text, 1)]},
            )
        else:
            max_output_length = min(requested_limit, MAX_SHELL_OUTPUT_LENGTH)

        def invalid_commands_result() -> MCPToolResult:
            text = "commands must be a list of strings"
            return _shell_result(
                [text],
                is_error=True,
                structured={
                    "output": [shell_output("", text, 1)],
                    "max_output_length": max_output_length,
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
        display_outputs: list[str] = []
        is_error = False
        timeout_seconds: float | None = None
        timeout_ms = arguments.get("timeout_ms")
        if isinstance(timeout_ms, int):
            timeout_seconds = timeout_ms / 1000.0

        for command in command_list:
            full_cmd = (
                f"timeout {timeout_seconds:g}s bash -lc {shlex.quote(command)}"
                if timeout_seconds
                else command
            )
            completed = await self.client.run(full_cmd, check=False)
            stdout = completed.stdout if isinstance(completed.stdout, str) else ""
            stderr = completed.stderr if isinstance(completed.stderr, str) else ""
            exit_code = completed.returncode
            assert exit_code is not None
            stdout, stderr = bound_shell_output(stdout, stderr, max_output_length)
            outputs.append(shell_output(stdout, stderr, exit_code))
            display_outputs.append(stdout + stderr)
            is_error = is_error or bool(exit_code)

        return _shell_result(
            display_outputs,
            is_error=is_error,
            structured={
                "output": outputs,
                "max_output_length": max_output_length,
            },
        )


def _shell_result(
    texts: list[str],
    *,
    is_error: bool = False,
    structured: dict[str, Any] | None = None,
) -> MCPToolResult:
    payload = {"provider_tool": "shell", **(structured or {})}
    return MCPToolResult(
        content=[mcp_types.TextContent(type="text", text=text) for text in texts if text],
        isError=is_error,
        structuredContent=payload,
    )


def shell_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "outcome": {"type": "exit", "exit_code": exit_code},
    }


__all__ = ["OPENAI_SHELL_SPEC", "OpenAIShellTool"]
