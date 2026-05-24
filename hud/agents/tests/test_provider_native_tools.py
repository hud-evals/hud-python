"""Native provider tool contracts for translation and model gating."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import pytest

from hud.agents.claude.tools.coding import ClaudeBashTool, ClaudeTextEditorTool
from hud.agents.gemini.tools.coding import GeminiShellTool
from hud.agents.gemini.tools.filesystem import GeminiReadTool
from hud.agents.gemini.tools.memory import GeminiMemoryTool
from hud.agents.openai.tools.coding import OpenAIShellTool
from hud.agents.tests.conftest import RecordingToolEnvironment, text_result
from hud.types import MCPToolCall


@pytest.mark.asyncio
async def test_openai_shell_translates_commands_timeout_and_structured_output() -> None:
    spec = OpenAIShellTool.default_spec("gpt-5.4")
    assert spec is not None
    tool = OpenAIShellTool(env_tool_name="bash", spec=spec)
    environment = RecordingToolEnvironment(
        results={
            "bash": text_result("pwd output"),
        },
    )

    result = await tool.execute(
        environment.call_tool,
        {"commands": ["pwd"], "timeout_ms": 2500, "max_output_length": 80},
    )
    formatted = tool.format_result(MCPToolCall(name="shell", id="call_1", arguments={}), result)

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("bash", {"command": "pwd", "timeout_seconds": 2.5})
    ]
    assert result.structuredContent == {
        "provider_tool": "shell",
        "output": [
            {"stdout": "pwd output", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
        ],
        "max_output_length": 80,
    }
    formatted_dict = cast("dict[str, Any]", formatted)
    assert formatted_dict["type"] == "shell_call_output"
    assert formatted_dict["call_id"] == "call_1"
    assert formatted_dict["max_output_length"] == 80


@pytest.mark.asyncio
async def test_openai_shell_rejects_invalid_commands_without_environment_call() -> None:
    spec = OpenAIShellTool.default_spec("gpt-5.4")
    assert spec is not None
    tool = OpenAIShellTool(env_tool_name="bash", spec=spec)
    environment = RecordingToolEnvironment()

    result = await tool.execute(environment.call_tool, {"commands": 123})

    assert result.isError is True
    assert environment.calls == []


@pytest.mark.asyncio
async def test_claude_text_editor_translates_str_replace_arguments() -> None:
    spec = ClaudeTextEditorTool.default_spec("claude-sonnet-4-6")
    assert spec is not None
    tool = ClaudeTextEditorTool(env_tool_name="edit", spec=spec)
    environment = RecordingToolEnvironment(results={"edit": text_result("edited")})

    result = await tool.execute(
        environment.call_tool,
        {
            "command": "str_replace",
            "path": "/tmp/file.txt",
            "old_str": "old",
            "new_str": "new",
        },
    )

    assert result.isError is False
    assert [(call.name, call.arguments) for call in environment.calls] == [
        (
            "edit",
            {
                "command": "replace",
                "path": "/tmp/file.txt",
                "old_text": "old",
                "new_text": "new",
            },
        )
    ]


@pytest.mark.asyncio
async def test_gemini_shell_scopes_command_to_directory() -> None:
    tool = GeminiShellTool(env_tool_name="bash", spec=GeminiShellTool.default_spec("gemini"))
    environment = RecordingToolEnvironment(results={"bash": text_result("ok")})

    await tool.execute(environment.call_tool, {"command": "ls -la", "dir_path": "/tmp/my dir"})

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("bash", {"command": "cd '/tmp/my dir' && ls -la"})
    ]


@pytest.mark.asyncio
async def test_gemini_read_translates_line_range_to_offset_and_limit() -> None:
    tool = GeminiReadTool(env_tool_name="read", spec=GeminiReadTool.default_spec("gemini"))
    environment = RecordingToolEnvironment(results={"read": text_result("lines")})

    await tool.execute(
        environment.call_tool,
        {"file_path": "/repo/file.py", "start_line": 3, "end_line": 7},
    )

    assert [(call.name, call.arguments) for call in environment.calls] == [
        ("read", {"filePath": "/repo/file.py", "offset": 2, "limit": 5})
    ]


@pytest.mark.asyncio
async def test_gemini_memory_persists_trimmed_fact_under_stable_path() -> None:
    tool = GeminiMemoryTool(env_tool_name="edit", spec=GeminiMemoryTool.default_spec("gemini"))
    environment = RecordingToolEnvironment(results={"edit": text_result("saved")})

    await tool.execute(environment.call_tool, {"fact": "  user likes concise tests  "})

    digest = hashlib.sha256(b"user likes concise tests").hexdigest()[:12]
    assert [(call.name, call.arguments) for call in environment.calls] == [
        (
            "edit",
            {
                "command": "create",
                "path": f"/memories/gemini-{digest}.md",
                "file_text": "user likes concise tests\n",
            },
        )
    ]


def test_native_tool_model_gating_uses_provider_supported_model_contracts() -> None:
    assert OpenAIShellTool.default_spec("gpt-5.4") is not None
    assert OpenAIShellTool.default_spec("gpt-4.1") is None
    assert ClaudeBashTool.default_spec("claude-sonnet-4-6") is not None
    assert ClaudeBashTool.default_spec("claude-3-5-sonnet") is None
