"""Which endpoint and credential an agent's model requests use.

Agents built with ``create_agent`` route through the HUD gateway; agents built
directly use a provider's own key when one is configured and the gateway
otherwise. Rows build an agent the way a user does, run one turn, and pin the
credential the provider received, whether the run's trace id rode along, and
the model it was asked for.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

from hud.agents import (
    ClaudeAgent,
    ClaudeCLIAgent,
    CodexCLIAgent,
    GeminiAgent,
    OpenAIAgent,
    OpenAIChatAgent,
    create_agent,
)
from hud.agents.base import Agent
from hud.agents.types import (
    AgentConfig,
    ClaudeCLIConfig,
    ClaudeConfig,
    CodexCLIConfig,
    OpenAIChatConfig,
    OpenAIConfig,
)
from hud.utils.exceptions import HudAuthenticationError
from tests.agents.support import run_task, workspace_env
from tests.harness import Reply, say

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.eval import Run
    from tests.harness import FakeServices, HudEnv, Models, Request

CATALOG = [
    {
        "id": "ft:custom-123",
        "model_name": "gpt-5.5",
        "sdk_agent_type": "openai_compatible",
        "provider": {"name": "openai"},
    },
    {
        "id": "deepseek/deepseek-v4-flash",
        "model_name": "deepseek/deepseek-v4-flash",
        "sdk_agent_type": "openai_compatible",
    },
    {"id": "MiniMax-M3", "model_name": "MiniMax-M3", "sdk_agent_type": "openai_compatible"},
    {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "sdk_agent_type": "claude"},
]


def serve_catalog(services: FakeServices) -> None:
    """The gateway model catalog, served two entries a page."""

    def page(request: Request) -> Reply:
        offset = int(request.query.get("offset", ["0"])[0])
        return Reply(json={"items": CATALOG[offset : offset + 2], "total": len(CATALOG)})

    services.route("api", "GET", "/v2/models", handler=page)


ROUTES = {
    "create-agent-shortcut": (
        lambda _url: create_agent("openai"),
        snapshot(
            {
                "protocol": "responses",
                "model": "gpt-5.6",
                "credential": "Bearer hud-key",
                "trace_id": True,
            }
        ),
    ),
    "create-agent-ignores-a-provider-key": (
        lambda _url: create_agent("openai_compatible"),
        snapshot(
            {
                "protocol": "chat",
                "model": "gpt-5.4-mini",
                "credential": "Bearer hud-key",
                "trace_id": True,
            }
        ),
    ),
    "create-agent-claude": (
        lambda _url: create_agent("claude"),
        snapshot(
            {
                "protocol": "anthropic",
                "model": "claude-sonnet-4-6",
                "credential": "hud-key",
                "trace_id": True,
            }
        ),
    ),
    "create-agent-gemini": (
        lambda _url: create_agent("gemini"),
        snapshot(
            {
                "protocol": "gemini",
                "model": "gemini-3.1-pro-preview",
                "credential": "hud-key",
                "trace_id": True,
            }
        ),
    ),
    "create-agent-catalog-id": (
        lambda _url: create_agent("ft:custom-123"),
        snapshot(
            {
                "protocol": "chat",
                "model": "gpt-5.5",
                "credential": "Bearer hud-key",
                "trace_id": True,
            }
        ),
    ),
    "create-agent-catalog-slug-without-provider": (
        lambda _url: create_agent("deepseek-v4-flash"),
        snapshot(
            {
                "protocol": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "credential": "Bearer hud-key",
                "trace_id": True,
            }
        ),
    ),
    "create-agent-catalog-name-any-case": (
        lambda _url: create_agent("minimax-m3"),
        snapshot(
            {
                "protocol": "chat",
                "model": "MiniMax-M3",
                "credential": "Bearer hud-key",
                "trace_id": True,
            }
        ),
    ),
    "reloaded-gateway-agent": (
        lambda _url: OpenAIAgent.load(create_agent("openai").dump()),
        snapshot(
            {
                "protocol": "responses",
                "model": "gpt-5.6",
                "credential": "Bearer hud-key",
                "trace_id": True,
            }
        ),
    ),
    "direct-agent-with-a-provider-key": (
        lambda _url: OpenAIAgent(OpenAIConfig()),
        snapshot(
            {
                "protocol": "responses",
                "model": "gpt-5.6",
                "credential": "Bearer openai-key",
                "trace_id": False,
            }
        ),
    ),
    "direct-claude-with-a-provider-key": (
        lambda _url: ClaudeAgent(ClaudeConfig()),
        snapshot(
            {
                "protocol": "anthropic",
                "model": "claude-sonnet-4-6",
                "credential": "anthropic-key",
                "trace_id": False,
            }
        ),
    ),
    "direct-gemini-without-a-provider-key": (
        lambda _url: GeminiAgent(),
        snapshot(
            {
                "protocol": "gemini",
                "model": "gemini-3.1-pro-preview",
                "credential": "hud-key",
                "trace_id": True,
            }
        ),
    ),
}


@pytest.mark.parametrize(("build", "expected"), ROUTES.values(), ids=ROUTES.keys())
async def test_an_agent_reaches_its_model_with_the_credential_its_routing_chose(
    build: Callable[[str], Agent],
    expected: Any,
    services: FakeServices,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(
        HUD_API_KEY="hud-key",
        OPENAI_API_KEY="openai-key",
        ANTHROPIC_API_KEY="anthropic-key",
        OPENAI_BASE_URL=models.url,
        ANTHROPIC_BASE_URL=models.url,
    )
    serve_catalog(services)
    models.script([say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), build(models.url))

    (request,) = models.requests()
    received = {
        "protocol": request.protocol,
        "model": request.model,
        "credential": request.headers.get("authorization")
        or request.headers.get("x-api-key")
        or request.headers.get("x-goog-api-key"),
        "trace_id": request.headers.get("trace-id") == run.trace_id,
    }
    assert received == expected


async def test_explicit_chat_credentials_reach_the_given_endpoint(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    models.script([say("done")])
    agent = OpenAIChatAgent(OpenAIChatConfig(api_key="explicit", base_url=models.url))

    await run_task(workspace_env(tmp_path / "ws"), agent)

    (request,) = models.requests()
    assert (request.headers["authorization"], "trace-id" in request.headers) == (
        "Bearer explicit",
        False,
    )


def test_create_agent_needs_a_hud_api_key() -> None:
    with pytest.raises(HudAuthenticationError, match="HUD_API_KEY is required"):
        create_agent("openai")


@pytest.mark.parametrize("hud_api_key", ["hud-key", None])
@pytest.mark.parametrize(
    "credentials",
    [{"api_key": "provider-key"}, {"base_url": "https://provider.example"}, {"api_key": None}],
)
def test_create_agent_refuses_provider_credentials(
    credentials: dict[str, Any], hud_api_key: str | None, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY=hud_api_key)

    with pytest.raises(ValueError, match="instantiate the provider agent directly"):
        create_agent("openai_compatible", **credentials)


def test_create_agent_names_close_matches_for_an_unknown_model(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="hud-key")
    serve_catalog(services)

    with pytest.raises(ValueError, match=r"'minimax-m4' not found .* Did you mean: MiniMax-M3"):
        create_agent("minimax-m4")


def test_a_chat_agent_refuses_a_foreign_key_for_the_hud_gateway(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="hud-key")

    with pytest.raises(ValueError, match="api_key is not allowed with HUD Gateway"):
        OpenAIChatAgent(OpenAIChatConfig(api_key="other", base_url=services.url("gateway")))


AGENTS_WITHOUT_CREDENTIALS = {
    "chat": (lambda: OpenAIChatAgent(), ValueError, "No API key found"),
    "claude": (lambda: ClaudeAgent(), HudAuthenticationError, "No API key for anthropic"),
    "openai": (lambda: OpenAIAgent(), HudAuthenticationError, "No API key for openai"),
    "gemini": (lambda: GeminiAgent(), HudAuthenticationError, "No API key for gemini"),
}


@pytest.mark.parametrize(
    ("build", "error", "message"),
    AGENTS_WITHOUT_CREDENTIALS.values(),
    ids=AGENTS_WITHOUT_CREDENTIALS.keys(),
)
def test_an_agent_without_any_credential_fails_at_construction(
    build: Callable[[], Agent], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        build()


def test_an_agent_with_a_custom_model_client_cannot_be_dumped() -> None:
    agent = OpenAIAgent(OpenAIConfig(model_client=object()))

    with pytest.raises(ValueError, match="a custom model_client cannot be serialized"):
        agent.dump()


class Answering(Agent):
    async def __call__(self, run: Run) -> None:
        run.trace.content = "done"


DUMPED = {
    "claude-cli": ClaudeCLIAgent(ClaudeCLIConfig(reasoning_effort="max", max_steps=3)),
    "codex-cli": CodexCLIAgent(CodexCLIConfig(sandbox="read-only")),
    "custom": Answering(AgentConfig(model="custom", timeout_seconds=30)),
}


@pytest.mark.parametrize("agent", DUMPED.values(), ids=DUMPED.keys())
def test_a_dumped_agent_loads_back_to_the_same_configuration(agent: Agent) -> None:
    loaded = type(agent).load(agent.dump())

    assert (type(loaded), loaded.dump()) == (type(agent), agent.dump())


def test_a_provider_agent_without_its_sdk_points_at_the_agents_extra(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "anthropic" or name.startswith("anthropic."):
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)

        sys.meta_path.insert(0, Blocker())
        import hud.agents

        try:
            hud.agents.ClaudeAgent
        except ImportError as exc:
            print(exc)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=tmp_path
    )

    assert result.stdout.strip() == (
        "ClaudeAgent requires the agents extra. Install with: pip install 'hud[agents]'"
    )
