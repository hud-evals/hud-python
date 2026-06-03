"""The agent base contract: the ``Agent`` ABC, ``as_mcp_server``, gateway routing.

These cover the model-agnostic surface that doesn't need provider SDKs or network:
the stateless ``Agent`` contract, exposing native tools as an ``MCPServer``, and
``AgentType`` / ``create_agent`` resolution.
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp.types import TextContent

from hud.agents import OpenAIAgent, OpenAIChatAgent, create_agent
from hud.agents.base import Agent
from hud.native.tools.base import BaseTool
from hud.types import AgentType


class PingTool(BaseTool):
    async def __call__(self) -> list[TextContent]:  # name auto-derives to "ping"
        return [TextContent(type="text", text="pong")]


class _ServingAgent(Agent):
    native_tools = (PingTool,)

    async def __call__(self, run: Any) -> None:
        run.trace.content = "done"


# ─── the ABC contract ─────────────────────────────────────────────────


def test_agent_requires_call_implementation() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


async def test_agent_call_fills_trace() -> None:
    from types import SimpleNamespace

    run = SimpleNamespace(trace=SimpleNamespace(content=""))
    await _ServingAgent()(run)
    assert run.trace.content == "done"


# ─── as_mcp_server ────────────────────────────────────────────────────


async def test_as_mcp_server_exposes_native_tools() -> None:
    server = _ServingAgent().as_mcp_server()
    names = {tool.name for tool in await server.list_tools()}
    assert "ping" in names


async def test_as_mcp_server_accepts_tool_override_and_name() -> None:
    server = _ServingAgent().as_mcp_server(name="custom", tools=[PingTool()])
    assert server.name == "custom"
    assert {tool.name for tool in await server.list_tools()} == {"ping"}


def test_agent_without_native_tools_serves_empty() -> None:
    class _Bare(Agent):
        async def __call__(self, run: Any) -> None: ...

    server = _Bare().as_mcp_server()
    assert server is not None


# ─── AgentType resolution ─────────────────────────────────────────────


def test_agent_type_maps_value_to_class_and_provider() -> None:
    assert AgentType("openai").cls is OpenAIAgent
    assert AgentType("openai_compatible").cls is OpenAIChatAgent
    assert isinstance(AgentType("openai").gateway_provider, str)


# ─── create_agent routing ─────────────────────────────────────────────


def test_create_agent_unknown_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # No gateway models available -> a bare unknown model can't be resolved.
    monkeypatch.setattr("hud.agents.gateway._fetch_gateway_models", list)
    with pytest.raises(ValueError, match="not found"):
        create_agent("totally-unknown-model-xyz")


def test_create_agent_value_shortcut_builds_provider_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr("hud.agents.gateway.build_gateway_client", lambda _provider: sentinel)

    agent = create_agent("openai")  # AgentType.OPENAI shortcut

    assert isinstance(agent, OpenAIAgent)
    # The gateway client + validate flag are threaded into the agent's config.
    assert agent.config.model_client is sentinel
    assert agent.config.validate_api_key is False


def test_create_agent_resolves_gateway_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.agents.gateway import GatewayModelInfo, GatewayProviderInfo

    model = GatewayModelInfo(
        id="ft:custom-123",
        model_name="gpt-5.4",
        sdk_agent_type="openai_compatible",
        provider=GatewayProviderInfo(name="openai"),
    )
    monkeypatch.setattr("hud.agents.gateway._fetch_gateway_models", lambda: [model])
    monkeypatch.setattr("hud.agents.gateway.build_gateway_client", lambda _provider: object())

    agent = create_agent("ft:custom-123")

    assert isinstance(agent, OpenAIChatAgent)
    assert agent.config.model == "gpt-5.4"  # resolved to the model's real name
