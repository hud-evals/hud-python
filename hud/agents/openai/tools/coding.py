"""Agent-owned OpenAI tools."""

from __future__ import annotations

from typing import Any, cast

from mcp.types import TextContent
from openai.types.responses import ApplyPatchToolParam, FunctionShellToolParam, ToolParam

from hud.types import MCPToolCall, MCPToolResult

from .apply_patch import _patch_to_commit, _text_to_patch
from .base import CallTool, OpenAITool, OpenAIToolSpec, call_tool, result_text

OPENAI_SHELL_SPEC = OpenAIToolSpec(
    api_type="shell",
    api_name="shell",
    supported_models=(
        "gpt-5.1",
        "gpt-5.1-*",
        "gpt-5.2",
        "gpt-5.2-*",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-*",
    ),
)

OPENAI_APPLY_PATCH_SPEC = OpenAIToolSpec(
    api_type="apply_patch",
    api_name="apply_patch",
    supported_models=(
        "gpt-5.1",
        "gpt-5.1-*",
        "gpt-5.2",
        "gpt-5.2-*",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-*",
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
        return cast("ToolParam", FunctionShellToolParam(type="shell"))

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


class OpenAIApplyPatchTool(OpenAITool):
    """OpenAI apply_patch provider tool backed by an environment editor tool."""

    name = "apply_patch"
    capability = "editor"

    @classmethod
    def default_spec(cls, model: str) -> OpenAIToolSpec | None:
        if OPENAI_APPLY_PATCH_SPEC.supports_model(model):
            return OPENAI_APPLY_PATCH_SPEC
        return None

    def __init__(self, *, env_tool_name: str, spec: OpenAIToolSpec) -> None:
        del spec
        super().__init__(env_tool_name=env_tool_name, spec=OPENAI_APPLY_PATCH_SPEC)

    def to_params(self) -> ToolParam:
        return cast("ToolParam", ApplyPatchToolParam(type="apply_patch"))

    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        operation = arguments.get("type")
        path = arguments.get("path")
        diff = arguments.get("diff")

        if not isinstance(operation, str):
            return _apply_patch_result("Missing operation type", status="failed")
        if not isinstance(path, str) or not path:
            return _apply_patch_result("Missing file path", status="failed")

        try:
            if operation == "delete_file":
                result = await call_tool(
                    caller,
                    self.env_tool_name,
                    {"command": "delete", "path": path},
                )
                return _apply_patch_result(result_text(result), result=result)

            if not isinstance(diff, str):
                return _apply_patch_result(
                    f"Missing diff for {operation} operation",
                    status="failed",
                )

            if operation == "create_file":
                content = _parse_create_diff(diff)
                result = await call_tool(
                    caller,
                    self.env_tool_name,
                    {"command": "create", "path": path, "file_text": content},
                )
                return _apply_patch_result(result_text(result), result=result)

            if operation == "update_file":
                read_result = await call_tool(
                    caller,
                    self.env_tool_name,
                    {"command": "read", "path": path},
                )
                if read_result.isError:
                    return _apply_patch_result(result_text(read_result), result=read_result)
                content = _apply_update_diff(path, result_text(read_result), diff)
                write_result = await call_tool(
                    caller,
                    self.env_tool_name,
                    {"command": "write", "path": path, "file_text": content},
                )
                return _apply_patch_result(result_text(write_result), result=write_result)

        except Exception as exc:
            return _apply_patch_result(str(exc), status="failed")

        return _apply_patch_result(f"Unknown operation type '{operation}'", status="failed")

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> dict[str, Any]:
        structured = result.structuredContent if isinstance(result.structuredContent, dict) else {}
        status = structured.get("status")
        if status not in {"completed", "failed"}:
            status = "failed" if result.isError else "completed"
        output = structured.get("output")
        if not isinstance(output, str):
            output = result_text(result)
        return {
            "type": "apply_patch_call_output",
            "call_id": call.id,
            "status": status,
            "output": output,
        }


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


def _apply_patch_result(
    output: str,
    *,
    status: str | None = None,
    result: MCPToolResult | None = None,
) -> MCPToolResult:
    if result is not None:
        status = "failed" if result.isError else "completed"
    status = status or "completed"
    return _provider_result(
        "apply_patch",
        output,
        is_error=status == "failed",
        structured={"status": status, "output": output},
    )


def _parse_create_diff(diff: str) -> str:
    lines = diff.strip().split("\n")
    content_lines: list[str] = []
    for line in lines:
        if not line and not content_lines:
            continue
        if line.startswith(("+", " ")):
            content_lines.append(line[1:])
        elif line == "":
            content_lines.append("")
    return "\n".join(content_lines)


def _apply_update_diff(path: str, current_content: str, diff: str) -> str:
    patch_text = f"*** Begin Patch\n*** Update File: {path}\n{diff}\n*** End Patch"
    patch, _ = _text_to_patch(patch_text, {path: current_content})
    commit = _patch_to_commit(patch, {path: current_content})
    change = commit.changes.get(path)
    if change is None:
        raise ValueError(f"Patch did not update {path}")
    return change.new_content or ""


__all__ = [
    "OPENAI_APPLY_PATCH_SPEC",
    "OPENAI_SHELL_SPEC",
    "OpenAIApplyPatchTool",
    "OpenAIShellTool",
]
