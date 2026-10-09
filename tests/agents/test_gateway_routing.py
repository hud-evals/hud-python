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

from hud import Environment
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
    GeminiConfig,
    OpenAIChatConfig,
    OpenAIConfig,
)
from hud.eval import LocalRuntime, Task, Taskset, rollout
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
    "create-agent-catalog-display-name": (
        lambda _url: create_agent("Claude Sonnet 4.6"),
        snapshot(
            {
                "protocol": "anthropic",
                "model": "claude-sonnet-4-6",
                "credential": "hud-key",
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


BEDROCK_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"
AWS = {
    "AWS_ACCESS_KEY_ID": "AKIATEST",
    "AWS_SECRET_ACCESS_KEY": "secret",
    "AWS_REGION": "us-east-1",
}


def _env() -> Environment:
    env = Environment("routing")

    @env.template()
    async def answer():
        reply = yield "Reply with ok."
        yield 1.0 if reply == "ok" else 0.0

    return env


TASK = Task(env="routing", id="answer")

AGENTS: dict[str, Callable[[bool], Agent]] = {
    "openai": lambda gateway: OpenAIAgent(OpenAIConfig(model="m", max_steps=2, gateway=gateway)),
    "anthropic": lambda gateway: ClaudeAgent(
        ClaudeConfig(model="claude-m", max_steps=2, gateway=gateway)
    ),
    "gemini": lambda gateway: GeminiAgent(
        GeminiConfig(model="gemini-m", max_steps=2, gateway=gateway)
    ),
}
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
AUTH_HEADERS = {"openai": "authorization", "anthropic": "x-api-key", "gemini": "x-goog-api-key"}


def _point_provider_sdks_at(hud_env: HudEnv, provider: Models) -> None:
    hud_env.set(
        OPENAI_BASE_URL=provider.url,
        ANTHROPIC_BASE_URL=provider.url,
        GOOGLE_GEMINI_BASE_URL=provider.url,
    )


@pytest.mark.parametrize(
    ("family", "provider_key", "gateway", "reached"),
    [
        pytest.param(family, key, gateway, reached, id=f"{family}-{label}")
        for family in AGENTS
        for key, gateway, reached, label in [
            (True, False, "provider", "own-key"),
            (False, False, "gateway", "hud-key-only"),
            (True, True, "gateway", "forced-gateway"),
        ]
    ],
)
async def test_a_provider_key_routes_direct_unless_the_gateway_is_forced(
    models: Models,
    provider: Models,
    hud_env: HudEnv,
    family: str,
    provider_key: bool,
    gateway: bool,
    reached: str,
) -> None:
    _point_provider_sdks_at(hud_env, provider)
    hud_env.set(
        HUD_API_KEY="hud-key", **{PROVIDER_KEYS[family]: "own-key" if provider_key else None}
    )
    models.script([say("ok")])
    provider.script([say("ok")])

    run = await rollout(TASK, AGENTS[family](gateway), runtime=LocalRuntime(_env()))

    served = {"gateway": models.requests(), "provider": provider.requests()}
    assert run.reward == 1.0
    assert {name: len(requests) for name, requests in served.items()} == {
        name: int(name == reached) for name in served
    }
    (request,) = served[reached]
    credential = request.headers[AUTH_HEADERS[family]].removeprefix("Bearer ")
    assert credential == ("hud-key" if reached == "gateway" else "own-key")


@pytest.mark.parametrize(
    ("build", "settings", "error", "message"),
    [
        pytest.param(
            lambda: ClaudeAgent(ClaudeConfig(model=BEDROCK_ARN)),
            {"HUD_API_KEY": "hud-key"},
            HudAuthenticationError,
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required",
            id="bedrock-without-aws-credentials",
        ),
        pytest.param(
            lambda: ClaudeAgent(ClaudeConfig(model=BEDROCK_ARN, gateway=True)),
            {"HUD_API_KEY": "hud-key", **AWS},
            ValueError,
            "is a Bedrock inference profile; it cannot use the HUD gateway",
            id="bedrock-through-the-gateway",
        ),
    ],
)
def test_an_unroutable_model_is_refused_when_the_agent_is_built(
    hud_env: HudEnv,
    build: Callable[[], Agent],
    settings: dict[str, str],
    error: type[Exception],
    message: str,
) -> None:
    hud_env.set(**settings)

    with pytest.raises(error, match=message):
        build()


@pytest.mark.parametrize("family", list(AGENTS))
async def test_each_gateway_request_carries_its_own_runs_trace_id(
    models: Models, hud_env: HudEnv, family: str
) -> None:
    hud_env.set(HUD_API_KEY="hud-key")
    models.script([say("ok")])

    job = await Taskset("t", [TASK]).run(
        AGENTS[family](True), runtime=LocalRuntime(_env()), group=3
    )

    assert sorted(request.headers["trace-id"] for request in models.requests()) == sorted(
        str(run.trace_id) for run in job.runs
    )
    assert all("x-hud-parent-trace-id" not in r.headers for r in models.requests())


class Delegating(Agent):
    """Answers by running a nested rollout, as a subagent would."""

    def __init__(self) -> None:
        super().__init__()
        self.inner: list[Run] = []

    async def __call__(self, run: Run) -> None:
        agent = OpenAIChatAgent(OpenAIChatConfig(model="m", max_steps=2))
        inner = await rollout(TASK, agent, runtime=LocalRuntime(_env()))
        self.inner.append(inner)
        run.trace.content = inner.trace.content


async def test_a_nested_rollout_names_its_parent_trace(models: Models, hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")
    models.script([say("ok")])
    agent = Delegating()

    outer = await rollout(TASK, agent, runtime=LocalRuntime(_env()))

    (request,) = models.requests()
    (inner,) = agent.inner
    headers: dict[str, Any] = request.headers
    assert outer.reward == 1.0
    assert (headers["trace-id"], headers["x-hud-parent-trace-id"]) == (
        str(inner.trace_id),
        str(outer.trace_id),
    )
