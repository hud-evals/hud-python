"""Each provider agent drives a real workspace through its own wire protocol.

A scripted provider behind the fake gateway answers the agent; the workspace is
served over real SSH. Rows assert the reward (the grader reads the file the
agent wrote), the step sequence the run exported, and the last request
body the provider received, which carries the whole conversation (or, for the
Responses API, the items new since the previous response).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP
from inline_snapshot import snapshot

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.claude import ClaudeToolSearchTool, ClaudeWebFetchTool, ClaudeWebSearchTool
from hud.agents.gemini.tools import (
    GeminiCodeExecutionTool,
    GeminiGoogleSearchTool,
    GeminiUrlContextTool,
)
from hud.agents.openai.tools import OpenAICodeInterpreterTool, OpenAIToolSearchTool
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from tests.agents.support import image, mcp_server, run_task, wire, workspace_env
from tests.harness import Turn, call, say, shell_call, steps

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.harness import HudEnv, ModelRequest, Models


@dataclass(frozen=True)
class Row:
    agent: Callable[[], Agent]
    script: Callable[[Path], list[Turn]]
    last_request: Any
    steps: Any


ROWS = {
    "claude": Row(
        agent=lambda: ClaudeAgent(
            ClaudeConfig(model="claude-sonnet-4-6", system_prompt="Be brief.", max_tokens=1024)
        ),
        script=lambda root: [
            replace(
                call(
                    "str_replace_based_edit_tool",
                    command="create",
                    path=f"{root}/a.txt",
                    file_text="PASS",
                ),
                reasoning="plan",
            ),
            call("str_replace_based_edit_tool", command="view", path=f"{root}/shot.png"),
            call("bash", command="cat a.txt"),
            say("done", reasoning="checked"),
        ],
        last_request=snapshot(
            {
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Do the task."}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"signature": "sig", "thinking": "plan", "type": "thinking"},
                            {
                                "id": "toolu_1",
                                "input": {
                                    "command": "create",
                                    "path": "<path0>/a.txt",
                                    "file_text": "PASS",
                                },
                                "name": "str_replace_based_edit_tool",
                                "type": "tool_use",
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {"type": "text", "text": "wrote 4 bytes to <path0>/a.txt"}
                                ],
                                "is_error": False,
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "id": "toolu_3",
                                "input": {"command": "view", "path": "<path0>/shot.png"},
                                "name": "str_replace_based_edit_tool",
                                "type": "tool_use",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_3",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "<image/png 40x30>",
                                        },
                                    }
                                ],
                                "is_error": False,
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "id": "toolu_5",
                                "input": {"command": "cat a.txt"},
                                "name": "bash",
                                "type": "tool_use",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_5",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": """\
$ cat a.txt
PASS
(exit 0)\
""",
                                    }
                                ],
                                "is_error": False,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                ],
                "model": "claude-sonnet-4-6",
                "system": "Be brief.",
                "tools": [
                    {"type": "bash_20250124", "name": "bash"},
                    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                ],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                "stream": True,
            }
        ),
        steps=snapshot(
            ["task", "user", "agent", "tool", "agent", "tool", "agent", "tool", "agent", "task"]
        ),
    ),
    "openai": Row(
        agent=lambda: OpenAIAgent(
            OpenAIConfig(model="gpt-5.6", system_prompt="Be brief.", prompt_cache_key="cache-1")
        ),
        script=lambda _root: [
            replace(shell_call("printf PASS > a.txt", "cat a.txt"), reasoning="plan"),
            say("done", reasoning="checked"),
        ],
        last_request=snapshot(
            {
                "input": [
                    {
                        "type": "shell_call_output",
                        "call_id": "call_shell",
                        "status": "completed",
                        "output": [
                            {
                                "stdout": "",
                                "stderr": "",
                                "outcome": {"type": "exit", "exit_code": 0},
                            },
                            {
                                "stdout": "PASS",
                                "stderr": "",
                                "outcome": {"type": "exit", "exit_code": 0},
                            },
                        ],
                        "max_output_length": 10485760,
                    }
                ],
                "instructions": "Be brief.",
                "max_output_tokens": None,
                "model": "gpt-5.6",
                "parallel_tool_calls": None,
                "previous_response_id": "resp_1",
                "prompt_cache_key": "cache-1",
                "temperature": None,
                "tools": [{"type": "shell", "environment": {"type": "local"}}],
            }
        ),
        steps=snapshot(["task", "user", "agent", "tool", "agent", "task"]),
    ),
    "gemini": Row(
        agent=lambda: GeminiAgent(
            GeminiConfig(model="gemini-3.1-pro-preview", system_prompt="Be brief.")
        ),
        script=lambda root: [
            replace(
                call("write_file", file_path=f"{root}/a.txt", content="PASS"), reasoning="plan"
            ),
            call("run_shell_command", command="cat a.txt", dir_path=str(root)),
            call("list_directory", dir_path=str(root)),
            say("done", reasoning="checked"),
        ],
        last_request=snapshot(
            {
                "contents": [
                    {"parts": [{"text": "Do the task."}], "role": "user"},
                    {
                        "parts": [
                            {"text": "plan", "thought": True},
                            {
                                "functionCall": {
                                    "args": {"file_path": "<path0>/a.txt", "content": "PASS"},
                                    "name": "write_file",
                                }
                            },
                        ],
                        "role": "model",
                    },
                    {
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "write_file",
                                    "response": {
                                        "success": True,
                                        "output": "wrote 4 bytes to <path0>/a.txt",
                                    },
                                }
                            }
                        ],
                        "role": "user",
                    },
                    {
                        "parts": [
                            {
                                "functionCall": {
                                    "args": {"command": "cat a.txt", "dir_path": "<path0>"},
                                    "name": "run_shell_command",
                                }
                            }
                        ],
                        "role": "model",
                    },
                    {
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "run_shell_command",
                                    "response": {
                                        "success": True,
                                        "output": """\
$ cd <path0> && cat a.txt
PASS
(exit 0)\
""",
                                    },
                                }
                            }
                        ],
                        "role": "user",
                    },
                    {
                        "parts": [
                            {
                                "functionCall": {
                                    "args": {"dir_path": "<path0>"},
                                    "name": "list_directory",
                                }
                            }
                        ],
                        "role": "model",
                    },
                    {
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "list_directory",
                                    "response": {
                                        "success": True,
                                        "output": """\
a.txt
shot.png\
""",
                                    },
                                }
                            }
                        ],
                        "role": "user",
                    },
                ],
                "systemInstruction": {"parts": [{"text": "Be brief."}], "role": "user"},
                "tools": [
                    {
                        "functionDeclarations": [
                            {
                                "name": "run_shell_command",
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string"},
                                        "description": {"type": "string"},
                                        "dir_path": {"type": "string"},
                                    },
                                    "required": ["command"],
                                },
                            },
                            {
                                "name": "replace",
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {"type": "string"},
                                        "instruction": {"type": "string"},
                                        "old_string": {"type": "string"},
                                        "new_string": {"type": "string"},
                                    },
                                    "required": ["file_path", "old_string", "new_string"],
                                },
                            },
                            {
                                "name": "write_file",
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {"type": "string"},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["file_path", "content"],
                                },
                            },
                            {
                                "name": "read_file",
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {"type": "string"},
                                        "start_line": {"type": "integer"},
                                        "end_line": {"type": "integer"},
                                    },
                                    "required": ["file_path"],
                                },
                            },
                            {
                                "name": "grep_search",
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "pattern": {"type": "string"},
                                        "dir_path": {"type": "string"},
                                        "include_pattern": {"type": "string"},
                                    },
                                    "required": ["pattern"],
                                },
                            },
                            {
                                "name": "glob",
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "pattern": {"type": "string"},
                                        "dir_path": {"type": "string"},
                                    },
                                    "required": ["pattern"],
                                },
                            },
                            {
                                "name": "list_directory",
                                "parameters_json_schema": {
                                    "type": "object",
                                    "properties": {"dir_path": {"type": "string"}},
                                    "required": ["dir_path"],
                                },
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 1.0,
                    "topP": 0.95,
                    "topK": 40.0,
                    "maxOutputTokens": 8192,
                    "thinkingConfig": {"include_thoughts": True},
                },
            }
        ),
        steps=snapshot(
            ["task", "user", "agent", "tool", "agent", "tool", "agent", "tool", "agent", "task"]
        ),
    ),
    "chat": Row(
        agent=lambda: OpenAIChatAgent(
            OpenAIChatConfig(model="qwen3.6-plus", system_prompt="Be brief.")
        ),
        script=lambda root: [
            replace(call("write", filePath=f"{root}/a.txt", content="PASS"), reasoning="plan"),
            call("read", filePath=f"{root}/shot.png"),
            call("bash", command="cat a.txt"),
            say("done", reasoning="checked"),
        ],
        last_request=snapshot(
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": [{"type": "text", "text": "Do the task."}]},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write",
                                    "arguments": '{"filePath": "<path0>/a.txt", "content": "PASS"}',
                                },
                            }
                        ],
                        "reasoning_content": "plan",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "wrote 4 bytes to <path0>/a.txt",
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_3",
                                "type": "function",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"filePath": "<path0>/shot.png"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_3",
                        "content": "Tool executed successfully",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Tool returned the following:"},
                            {"type": "image_url", "image_url": {"url": "<image/png 40x30>"}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_5",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command": "cat a.txt"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_5",
                        "content": """\
$ cat a.txt
PASS
(exit 0)\
""",
                    },
                ],
                "model": "qwen3.6-plus",
                "stream": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "timeout": {"type": "integer", "minimum": 1},
                                    "workdir": {"type": "string"},
                                },
                                "required": ["command"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "filePath": {"type": "string"},
                                    "offset": {"type": "integer", "minimum": 0},
                                    "limit": {"type": "integer", "minimum": 1},
                                },
                                "required": ["filePath"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "glob",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "pattern": {"type": "string"},
                                    "path": {"type": "string"},
                                },
                                "required": ["pattern"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "grep",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "pattern": {"type": "string"},
                                    "path": {"type": "string"},
                                    "include": {"type": "string"},
                                },
                                "required": ["pattern"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "edit",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "filePath": {"type": "string"},
                                    "oldString": {"type": "string"},
                                    "newString": {"type": "string"},
                                    "replaceAll": {"type": "boolean"},
                                },
                                "required": ["filePath", "oldString", "newString"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "write",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "filePath": {"type": "string"},
                                },
                                "required": ["content", "filePath"],
                            },
                        },
                    },
                ],
            }
        ),
        steps=snapshot(
            ["task", "user", "agent", "tool", "agent", "tool", "agent", "tool", "agent", "task"]
        ),
    ),
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test_a_provider_agent_works_the_workspace_over_its_wire_protocol(
    row: Row, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    env = workspace_env(
        root,
        files={"shot.png": image((40, 30))},
        passes=lambda root: (root / "a.txt").read_text() == "PASS",
    )
    models.script(row.script(root))

    run = await run_task(env, row.agent())

    assert run.reward == 1.0
    assert wire(models.requests()[-1].body, root) == row.last_request
    assert [step["source"] for step in steps(run.trace_id)] == row.steps


def lookup_tools() -> FastMCP:
    server = FastMCP("lookup")

    @server.tool()
    def lookup(key: str) -> str:
        """Look a key up."""
        return key

    @server.tool()
    def store(key: str, value: str) -> str:
        """Store a value."""
        return value

    return server


@dataclass(frozen=True)
class HostedRow:
    agent: Callable[[], Agent]
    tools: Any


HOSTED = {
    "claude": HostedRow(
        agent=lambda: ClaudeAgent(
            ClaudeConfig(
                model="claude-sonnet-4-6",
                hosted_tools=[
                    ClaudeWebSearchTool(max_uses=2, allowed_domains=["example.com"]),
                    ClaudeWebFetchTool(max_content_tokens=100, citations_enabled=True),
                    ClaudeWebSearchTool(supported_models=("claude-opus-*",)),
                ],
            )
        ),
        tools=snapshot(
            [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {
                    "name": "lookup",
                    "input_schema": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                    "eager_input_streaming": True,
                },
                {
                    "name": "store",
                    "input_schema": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["key", "value"],
                        "type": "object",
                    },
                    "eager_input_streaming": True,
                },
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 2,
                    "allowed_domains": ["example.com"],
                },
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_content_tokens": 100,
                    "citations": {"enabled": True},
                },
            ]
        ),
    ),
    "claude-tool-search-over-threshold": HostedRow(
        agent=lambda: ClaudeAgent(
            ClaudeConfig(
                model="claude-sonnet-4-6", hosted_tools=[ClaudeToolSearchTool(threshold=1)]
            )
        ),
        tools=snapshot(
            [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {
                    "name": "lookup",
                    "input_schema": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                    "eager_input_streaming": True,
                    "defer_loading": True,
                },
                {
                    "name": "store",
                    "input_schema": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["key", "value"],
                        "type": "object",
                    },
                    "eager_input_streaming": True,
                    "defer_loading": True,
                },
                {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
            ]
        ),
    ),
    "claude-tool-search-under-threshold": HostedRow(
        agent=lambda: ClaudeAgent(
            ClaudeConfig(
                model="claude-sonnet-4-6", hosted_tools=[ClaudeToolSearchTool(threshold=5)]
            )
        ),
        tools=snapshot(
            [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {
                    "name": "lookup",
                    "input_schema": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                    "eager_input_streaming": True,
                },
                {
                    "name": "store",
                    "input_schema": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["key", "value"],
                        "type": "object",
                    },
                    "eager_input_streaming": True,
                },
                {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
            ]
        ),
    ),
    "openai": HostedRow(
        agent=lambda: OpenAIAgent(
            OpenAIConfig(
                model="gpt-5.6",
                hosted_tools=[OpenAICodeInterpreterTool(container={"type": "auto"})],
            )
        ),
        tools=snapshot(
            [
                {"type": "shell", "environment": {"type": "local"}},
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                    "strict": True,
                },
                {
                    "type": "function",
                    "name": "store",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["key", "value"],
                        "type": "object",
                    },
                    "strict": True,
                },
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ]
        ),
    ),
    "openai-tool-search-over-threshold": HostedRow(
        agent=lambda: OpenAIAgent(
            OpenAIConfig(model="gpt-5.6", hosted_tools=[OpenAIToolSearchTool(threshold=1)])
        ),
        tools=snapshot(
            [
                {"type": "shell", "environment": {"type": "local"}},
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                    "strict": True,
                    "defer_loading": True,
                },
                {
                    "type": "function",
                    "name": "store",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["key", "value"],
                        "type": "object",
                    },
                    "strict": True,
                    "defer_loading": True,
                },
                {"type": "tool_search"},
            ]
        ),
    ),
    "openai-tool-search-under-threshold": HostedRow(
        agent=lambda: OpenAIAgent(
            OpenAIConfig(model="gpt-5.6", hosted_tools=[OpenAIToolSearchTool(threshold=5)])
        ),
        tools=snapshot(
            [
                {"type": "shell", "environment": {"type": "local"}},
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                    "strict": True,
                },
                {
                    "type": "function",
                    "name": "store",
                    "parameters": {
                        "additionalProperties": False,
                        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["key", "value"],
                        "type": "object",
                    },
                    "strict": True,
                },
                {"type": "tool_search"},
            ]
        ),
    ),
    "gemini": HostedRow(
        agent=lambda: GeminiAgent(
            GeminiConfig(
                model="gemini-3.1-pro-preview",
                hosted_tools=[
                    GeminiGoogleSearchTool(),
                    GeminiUrlContextTool(),
                    GeminiCodeExecutionTool(),
                ],
            )
        ),
        tools=snapshot(
            [
                {"googleSearch": {}},
                {"urlContext": {}},
                {"codeExecution": {}},
                {
                    "functionDeclarations": [
                        {
                            "name": "run_shell_command",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "description": {"type": "string"},
                                    "dir_path": {"type": "string"},
                                },
                                "required": ["command"],
                            },
                        },
                        {
                            "name": "replace",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "instruction": {"type": "string"},
                                    "old_string": {"type": "string"},
                                    "new_string": {"type": "string"},
                                },
                                "required": ["file_path", "old_string", "new_string"],
                            },
                        },
                        {
                            "name": "write_file",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["file_path", "content"],
                            },
                        },
                        {
                            "name": "read_file",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "start_line": {"type": "integer"},
                                    "end_line": {"type": "integer"},
                                },
                                "required": ["file_path"],
                            },
                        },
                        {
                            "name": "grep_search",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {
                                    "pattern": {"type": "string"},
                                    "dir_path": {"type": "string"},
                                    "include_pattern": {"type": "string"},
                                },
                                "required": ["pattern"],
                            },
                        },
                        {
                            "name": "glob",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {
                                    "pattern": {"type": "string"},
                                    "dir_path": {"type": "string"},
                                },
                                "required": ["pattern"],
                            },
                        },
                        {
                            "name": "list_directory",
                            "parameters_json_schema": {
                                "type": "object",
                                "properties": {"dir_path": {"type": "string"}},
                                "required": ["dir_path"],
                            },
                        },
                        {
                            "name": "lookup",
                            "parameters_json_schema": {
                                "additionalProperties": False,
                                "properties": {"key": {"type": "string"}},
                                "required": ["key"],
                                "type": "object",
                            },
                        },
                        {
                            "name": "store",
                            "parameters_json_schema": {
                                "additionalProperties": False,
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                                "required": ["key", "value"],
                                "type": "object",
                            },
                        },
                    ]
                },
            ]
        ),
    ),
}


@pytest.mark.parametrize("row", HOSTED.values(), ids=HOSTED.keys())
async def test_configured_hosted_tools_join_the_request(
    row: HostedRow, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    """A hosted tool the model does not support stays out; tool search defers MCP tools
    once there are more of them than its threshold."""
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])

    async with mcp_server(lookup_tools()) as tools:
        await run_task(workspace_env(tmp_path / "ws", capabilities=(tools,)), row.agent())

    (request,) = models.requests()
    assert wire(request.body["tools"]) == row.tools


INVALID_HOSTED = {
    "claude-both-domain-lists": (
        ClaudeAgent,
        ClaudeConfig,
        ClaudeWebSearchTool(allowed_domains=["a.com"], blocked_domains=["b.com"]),
        "[agent loop] ValueError: Use either allowed_domains or blocked_domains, not both.",
    ),
    "claude-fetch-both-domain-lists": (
        ClaudeAgent,
        ClaudeConfig,
        ClaudeWebFetchTool(allowed_domains=["a.com"], blocked_domains=["b.com"]),
        "[agent loop] ValueError: Use either allowed_domains or blocked_domains, not both.",
    ),
    "gemini-dynamic-threshold": (
        GeminiAgent,
        GeminiConfig,
        GeminiGoogleSearchTool(dynamic_threshold=0.5),
        "[agent loop] ValueError: dynamic_threshold is not supported by Gemini Google Search.",
    ),
}


@pytest.mark.parametrize(
    ("agent_cls", "config_cls", "hosted", "error"),
    INVALID_HOSTED.values(),
    ids=INVALID_HOSTED.keys(),
)
async def test_an_invalid_hosted_tool_fails_the_run_before_any_request(
    agent_cls: Any,
    config_cls: Any,
    hosted: Any,
    error: str,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")

    run = await run_task(
        workspace_env(tmp_path / "ws"), agent_cls(config_cls(hosted_tools=[hosted]))
    )

    assert (run.trace.status, run.trace.error) == ("error", error)
    assert models.requests() == []


def request_options(request: ModelRequest) -> dict[str, Any]:
    """A Chat Completions request without its conversation and tool list."""
    return {key: value for key, value in request.body.items() if key not in {"messages", "tools"}}


TRAINING = {
    "token-ids-continue-the-kv-cache": (
        OpenAIChatConfig(
            model="qwen3.6-plus",
            completion_kwargs={"extra_body": {"return_token_ids": True}, "temperature": 0.2},
        ),
        [
            Turn(
                tool_calls=call("bash", command="true").tool_calls,
                extra={"prompt_token_ids": [1, 2], "token_ids": [3]},
            ),
            Turn(text="done", extra={"prompt_token_ids": [1, 2, 3, 4], "token_ids": [5]}),
        ],
        snapshot(
            [
                {
                    "model": "qwen3.6-plus",
                    "logprobs": True,
                    "stream": False,
                    "temperature": 0.2,
                    "return_token_ids": True,
                },
                {
                    "model": "qwen3.6-plus",
                    "logprobs": True,
                    "stream": False,
                    "temperature": 0.2,
                    "return_token_ids": True,
                    "prompt_token_ids": [1, 2, 3],
                    "continuation_from": 2,
                },
            ]
        ),
    ),
    "multimodal-prompt-chunks-reset-the-continuation": (
        OpenAIChatConfig(
            model="qwen3.6-plus", completion_kwargs={"extra_body": {"return_token_ids": True}}
        ),
        [
            Turn(
                tool_calls=call("bash", command="true").tool_calls,
                extra={"prompt_chunks": [{"type": "text", "text": "hi"}], "token_ids": [3]},
            ),
            say("done"),
        ],
        snapshot(
            [
                {
                    "model": "qwen3.6-plus",
                    "logprobs": True,
                    "stream": False,
                    "return_token_ids": True,
                },
                {
                    "model": "qwen3.6-plus",
                    "logprobs": True,
                    "stream": False,
                    "return_token_ids": True,
                },
            ]
        ),
    ),
    "checkpoint-routes-in-the-body": (
        OpenAIChatConfig(model="qwen3.6-plus", checkpoint="ckpt-7"),
        [say("done")],
        snapshot([{"model": "qwen3.6-plus", "stream": False, "checkpoint": "ckpt-7"}]),
    ),
    "reserved-completion-kwargs-are-dropped": (
        OpenAIChatConfig(
            model="qwen3.6-plus",
            completion_kwargs={
                "model": "other",
                "stream": True,
                "tools": [],
                "messages": [],
                "max_tokens": 50,
            },
        ),
        [say("done")],
        snapshot([{"model": "qwen3.6-plus", "max_tokens": 50, "stream": False}]),
    ),
}


@pytest.mark.parametrize(("config", "turns", "expected"), TRAINING.values(), ids=TRAINING.keys())
async def test_chat_completion_options_reach_every_request(
    config: OpenAIChatConfig,
    turns: list[Turn],
    expected: Any,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(turns)

    await run_task(workspace_env(tmp_path / "ws"), OpenAIChatAgent(config))

    assert [request_options(request) for request in models.requests()] == expected


CITATIONS = {
    "openai": (
        lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6", citations_enabled=True)),
        lambda request: request.body.get("include"),
        snapshot(["web_search_call.action.sources"]),
    ),
    "gemini": (
        lambda: GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview", citations_enabled=True)),
        lambda request: [
            tool for tool in request.body["tools"] if "functionDeclarations" not in tool
        ],
        snapshot([{"googleSearch": {}}]),
    ),
}


@pytest.mark.parametrize(("agent", "part", "expected"), CITATIONS.values(), ids=CITATIONS.keys())
async def test_citations_ask_the_provider_for_sources(
    agent: Callable[[], Agent],
    part: Callable[[ModelRequest], Any],
    expected: Any,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])

    await run_task(workspace_env(tmp_path / "ws"), agent())

    (request,) = models.requests()
    assert part(request) == expected
