"""Tools from the environment's MCP servers, as each provider agent offers and calls them.

The environment serves MCP servers over HTTP. Rows pin the tool list each
provider received (names qualified by server once there are several, within
64 characters, OpenAI schemas in strict form), that every call reaches the
server that owns the tool with the run's trace id, and that a name clash
fails the run.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, Literal

import pytest
from dirty_equals import IsStr
from fastmcp import FastMCP
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from hud.agents import ClaudeAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.types import ClaudeConfig, OpenAIChatConfig, OpenAIConfig
from hud.capabilities.mcp import get_mcp_trace_id
from tests.agents.support import mcp_server, run_task, wire, workspace_env
from tests.harness import call, say

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from hud.capabilities import Capability
    from tests.harness import HudEnv, Models


class Address(BaseModel):
    street: str = Field(title="Street", min_length=1)


def database() -> FastMCP:
    server = FastMCP("database")

    @server.tool()
    def lookup(key: str) -> str:
        """Look a key up in the database."""
        return f"database:{key}"

    @server.tool()
    def trace() -> str:
        """Report the trace this call belongs to."""
        return str(get_mcp_trace_id())

    @server.tool()
    def register(
        name: str = Field(title="Name", min_length=2), address: Address | None = None
    ) -> str:
        """Register a person."""
        return f"registered {name}"

    return server


def search() -> FastMCP:
    server = FastMCP("search")

    @server.tool()
    def lookup(key: str) -> str:
        """Look a key up in the search index."""
        return f"search:{key}"

    return server


def named(*tools: str) -> Callable[[], FastMCP]:
    def build() -> FastMCP:
        server = FastMCP("named")
        for tool in tools:
            server.tool(name=tool, description=f"Run {tool}.")(lambda: tool)
        return server

    return build


async def serve(
    stack: AsyncExitStack,
    servers: dict[str, Callable[[], FastMCP]],
    transport: Literal["streamable-http", "sse"] = "streamable-http",
) -> tuple[Capability, ...]:
    return tuple(
        [
            await stack.enter_async_context(mcp_server(build(), name=name, transport=transport))
            for name, build in servers.items()
        ]
    )


def tool_name(tool: dict[str, Any]) -> str:
    return tool.get("name") or tool.get("function", {}).get("name", "")


AGENTS: dict[str, Callable[[], Agent]] = {
    "openai": lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6")),
    "claude": lambda: ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")),
    "chat": lambda: OpenAIChatAgent(OpenAIChatConfig(model="qwen3.6-plus")),
}

OFFERED = {
    "openai": snapshot(
        [
            {
                "type": "function",
                "name": "database__lookup",
                "description": "Look a key up in the database.",
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
                "name": "database__trace",
                "description": "Report the trace this call belongs to.",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {},
                    "type": "object",
                    "required": [],
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "database__register",
                "description": "Register a person.",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "anyOf": [
                                {
                                    "properties": {"street": {"type": "string"}},
                                    "required": ["street"],
                                    "type": "object",
                                    "additionalProperties": False,
                                },
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["name", "address"],
                    "type": "object",
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "search__lookup",
                "description": "Look a key up in the search index.",
                "parameters": {
                    "additionalProperties": False,
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "type": "object",
                },
                "strict": True,
            },
        ]
    ),
    "claude": snapshot(
        [
            {
                "name": "database__lookup",
                "description": "Look a key up in the database.",
                "input_schema": {
                    "additionalProperties": False,
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "type": "object",
                },
                "eager_input_streaming": True,
            },
            {
                "name": "database__trace",
                "description": "Report the trace this call belongs to.",
                "input_schema": {"additionalProperties": False, "properties": {}, "type": "object"},
                "eager_input_streaming": True,
            },
            {
                "name": "database__register",
                "description": "Register a person.",
                "input_schema": {
                    "additionalProperties": False,
                    "properties": {
                        "name": {"minLength": 2, "type": "string"},
                        "address": {
                            "anyOf": [
                                {
                                    "properties": {"street": {"minLength": 1, "type": "string"}},
                                    "required": ["street"],
                                    "type": "object",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                        },
                    },
                    "required": ["name"],
                    "type": "object",
                },
                "eager_input_streaming": True,
            },
            {
                "name": "search__lookup",
                "description": "Look a key up in the search index.",
                "input_schema": {
                    "additionalProperties": False,
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "type": "object",
                },
                "eager_input_streaming": True,
            },
        ]
    ),
    "chat": snapshot(
        [
            {
                "type": "function",
                "function": {
                    "name": "database__lookup",
                    "description": "Look a key up in the database.",
                    "parameters": {
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "database__trace",
                    "description": "Report the trace this call belongs to.",
                    "parameters": {"properties": {}, "type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "database__register",
                    "description": "Register a person.",
                    "parameters": {
                        "properties": {
                            "name": {"type": "string"},
                            "address": {
                                "properties": {"street": {"type": "string"}},
                                "required": ["street"],
                                "type": "object",
                                "default": None,
                            },
                        },
                        "required": ["name"],
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search__lookup",
                    "description": "Look a key up in the search index.",
                    "parameters": {
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "type": "object",
                    },
                },
            },
        ]
    ),
}


@pytest.mark.parametrize("provider", AGENTS.keys())
async def test_tools_from_two_servers_are_offered_qualified_and_reach_their_server(
    provider: str, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(
        [
            call("database__lookup", key="k1"),
            call("search__lookup", key="k2"),
            call("database__trace"),
            say("done"),
        ]
    )

    async with AsyncExitStack() as stack:
        tools = await serve(stack, {"database": database, "search": search})
        run = await run_task(workspace_env(tmp_path / "ws", capabilities=tools), AGENTS[provider]())

    requests = models.requests()
    first = requests[0]
    mcp_tools = [
        tool for tool in wire(first.body["tools"], descriptions=True) if "__" in tool_name(tool)
    ]
    assert mcp_tools == OFFERED[provider]
    answers = [text for request in requests for text in request.tool_results]
    if provider != "openai":
        answers = requests[-1].tool_results
    assert answers == [
        IsStr(regex=f".*{expected}") for expected in ("database:k1", "search:k2", run.trace_id)
    ]


async def test_a_single_server_offers_its_tools_unqualified(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("lookup", key="k1"), say("done")])

    async with AsyncExitStack() as stack:
        tools = await serve(stack, {"search": search}, transport="sse")
        await run_task(workspace_env(tmp_path / "ws", capabilities=tools), AGENTS["chat"]())

    first, last = models.requests()
    assert ("lookup" in first.tools, last.tool_results) == (True, ["search:k1"])


async def test_long_and_unusual_server_names_still_give_valid_tool_names(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])
    long_name = "lookup_" * 10

    async with AsyncExitStack() as stack:
        tools = await serve(
            stack,
            {
                "company.tools": named(long_name + "a"),
                "company-tools": named(long_name + "b"),
            },
        )
        await run_task(workspace_env(tmp_path / "ws", capabilities=tools), AGENTS["openai"]())

    (request,) = models.requests()
    names = [name for name in request.tools if name != "shell"]
    assert names == snapshot(
        [
            "company_tools__lookup_lookup_lookup_lookup_lookup_looku_525f784f",
            "company-tools__lookup_lookup_lookup_lookup_lookup_looku_442a1793",
        ]
    )
    assert all(len(name) <= 64 for name in names)


async def test_tool_names_that_clash_after_qualification_fail_the_run(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")

    async with AsyncExitStack() as stack:
        tools = await serve(stack, {"a": named("b__c"), "a__b": named("c")})
        run = await run_task(workspace_env(tmp_path / "ws", capabilities=tools), AGENTS["openai"]())

    assert (run.trace.status, run.trace.error, models.requests()) == (
        "error",
        "[agent loop] ValueError: MCP tool name collision after qualification: 'a__b__c'",
        [],
    )
