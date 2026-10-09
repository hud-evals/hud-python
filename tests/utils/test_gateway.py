"""Which server a model request reaches, with which credentials and trace headers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.base import Agent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from hud.eval import LocalRuntime, Task, Taskset, rollout
from hud.utils.exceptions import HudAuthenticationError
from tests.harness import Reply, say

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.eval import Run
    from tests.harness import FakeServices, HudEnv, Models

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
            lambda: OpenAIAgent(OpenAIConfig(model="m")),
            {},
            HudAuthenticationError,
            "No API key for openai",
            id="no-key-at-all",
        ),
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


async def test_a_bedrock_profile_is_sent_to_bedrock_signed_with_aws_credentials(
    models: Models, provider_server: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="hud-key", ANTHROPIC_BEDROCK_BASE_URL=provider_server.url("api"), **AWS)
    provider_server.reset()
    provider_server.route("api", "POST", "/model/{rest:path}", Reply(status=400, json={}))

    run = await rollout(
        TASK,
        ClaudeAgent(ClaudeConfig(model=BEDROCK_ARN, max_steps=1)),
        runtime=LocalRuntime(_env()),
    )

    (request, *_) = provider_server.requests("api", "POST")
    assert run.reward == 0.0
    assert models.requests() == []
    assert request.params["rest"] == f"{BEDROCK_ARN}/invoke"
    assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIATEST/")


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
