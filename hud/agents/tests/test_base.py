"""The agent base contract: the ``Agent`` ABC and gateway routing.

These cover the model-agnostic surface that doesn't need provider SDKs or network:
the stateless ``Agent`` contract and ``AgentType`` / ``create_agent`` resolution.
"""

from __future__ import annotations

from typing import Any

import pytest

from hud.agents import OpenAIAgent, OpenAIChatAgent, create_agent
from hud.agents.base import Agent
from hud.types import AgentType


class _ServingAgent(Agent):
    async def __call__(self, run: Any, *, max_steps: int | None = None) -> None:
        del max_steps
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

    def build_client(_provider: str) -> object:
        return sentinel

    monkeypatch.setattr("hud.agents.gateway.build_gateway_client", build_client)

    agent = create_agent("openai")  # AgentType.OPENAI shortcut

    assert isinstance(agent, OpenAIAgent)
    # The gateway client is threaded into the agent's config.
    assert agent.config.model_client is sentinel


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

    def build_client(_provider: str) -> object:
        return object()

    monkeypatch.setattr("hud.agents.gateway.build_gateway_client", build_client)

    agent = create_agent("ft:custom-123")

    assert isinstance(agent, OpenAIChatAgent)
    assert agent.config.model == "gpt-5.4"  # resolved to the model's real name
