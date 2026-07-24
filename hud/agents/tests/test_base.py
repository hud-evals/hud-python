"""The agent base contract: the ``Agent`` ABC and gateway routing.

These cover the model-agnostic surface that doesn't need provider SDKs or
network: the stateless ``Agent`` contract and ``AgentType`` / ``create_agent``
resolution.
"""

from __future__ import annotations

from typing import Any

import pytest

from hud.agents import OpenAIAgent, OpenAIChatAgent, create_agent
from hud.agents.base import Agent
from hud.types import AgentType
from hud.utils.exceptions import HudAuthenticationError


class _FillingAgent(Agent):
    async def __call__(self, run: Any) -> None:
        run.trace.content = "done"


# ─── the ABC contract ─────────────────────────────────────────────────


def test_agent_requires_call_implementation() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


async def test_agent_call_fills_trace() -> None:
    from types import SimpleNamespace

    run = SimpleNamespace(trace=SimpleNamespace(content=""))
    await _FillingAgent()(run)
    assert run.trace.content == "done"


# ─── AgentType resolution ─────────────────────────────────────────────


def test_agent_type_maps_value_to_class_and_provider() -> None:
    assert AgentType("openai").cls is OpenAIAgent
    assert AgentType("openai_compatible").cls is OpenAIChatAgent
    assert isinstance(AgentType("openai").gateway_provider, str)


def test_missing_provider_dependency_points_at_agents_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base installs (no [agents] extra) get an actionable error, not a raw import failure."""
    import sys

    import hud.agents

    class _Blocker:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
            if fullname == "anthropic" or fullname.startswith("anthropic."):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    for module in list(sys.modules):
        if module == "anthropic" or module.startswith(("anthropic.", "hud.agents.claude")):
            monkeypatch.delitem(sys.modules, module)
    monkeypatch.setattr(sys, "meta_path", [_Blocker(), *sys.meta_path])
    if "ClaudeAgent" in vars(hud.agents):  # drop any cached lazy export
        monkeypatch.delitem(hud.agents.__dict__, "ClaudeAgent")

    with pytest.raises(ImportError, match=r"hud\[agents\]"):
        _ = hud.agents.ClaudeAgent

    with pytest.raises(ImportError, match=r"hud\[agents\]"):
        _ = AgentType.CLAUDE.cls


# ─── create_agent routing ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _gateway_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.agents.settings.api_key", "test-key")


def test_create_agent_unknown_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # No gateway models available -> a bare unknown model can't be resolved.
    monkeypatch.setattr("hud.agents.list_gateway_models", list)
    with pytest.raises(ValueError, match="not found"):
        create_agent("totally-unknown-model-xyz")


def test_create_agent_value_shortcut_leaves_client_out_of_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def _build_client(_provider: str) -> object:
        return sentinel

    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", _build_client)

    agent = create_agent("openai")  # AgentType.OPENAI shortcut

    assert isinstance(agent, OpenAIAgent)
    assert agent.config.model_client is None
    assert agent.openai_client is sentinel


def test_create_agent_resolves_gateway_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.utils.gateway import GatewayModelInfo, GatewayProviderInfo

    model = GatewayModelInfo(
        id="ft:custom-123",
        model_name="gpt-5.5",
        sdk_agent_type="openai_compatible",
        provider=GatewayProviderInfo(name="openai"),
    )
    monkeypatch.setattr("hud.agents.list_gateway_models", lambda: [model])

    def _build_client(_provider: str) -> object:
        return object()

    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", _build_client)

    agent = create_agent("ft:custom-123")

    assert isinstance(agent, OpenAIChatAgent)
    assert agent.config.model == "gpt-5.5"  # resolved to the model's real name
    assert agent.config.model_client is None


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("deepseek-v4", "deepseek/deepseek-v4-pro"),
        ("deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
        ("glm-5.2", "z-ai/glm-5.2"),
        ("kimi-k2.6", "moonshotai/kimi-k2.6"),
        ("minimax-m3", "MiniMax-M3"),
    ],
)
def test_create_agent_accepts_gateway_model_aliases(
    alias: str,
    canonical: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hud.utils.gateway import GatewayModelInfo, GatewayProviderInfo

    model = GatewayModelInfo(
        id=canonical,
        model_name=canonical,
        sdk_agent_type="openai_compatible",
        provider=GatewayProviderInfo(name="openai"),
    )
    monkeypatch.setattr("hud.agents.list_gateway_models", lambda: [model])

    def _build_client(_provider: str) -> object:
        return object()

    monkeypatch.setattr("hud.utils.gateway.build_gateway_client", _build_client)

    agent = create_agent(alias)

    assert isinstance(agent, OpenAIChatAgent)
    assert agent.config.model == canonical


def test_create_agent_requires_hud_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.agents.settings.api_key", None)

    with pytest.raises(HudAuthenticationError, match="HUD_API_KEY"):
        create_agent("openai")
