"""Which endpoint and credential each agent family's model requests use, what it refuses
to run with, and the trace id every gateway request carries.

A family routes through the HUD gateway with ``HUD_API_KEY``, or reaches its
provider with the provider's own key, unless ``gateway=True`` forces the gateway.
Each row signs in the way a user does and reads, at the model's side, which
endpoint was reached and with which credential.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr

from hud.agents import OpenAIAgent, create_agent
from hud.eval import LocalRuntime, Task, Taskset
from hud.utils.exceptions import HudAuthenticationError
from tests.agents.families import (
    CLI_FAMILIES,
    TOOL_FAMILIES,
    CliDriver,
    NotApplicable,
    over_families,
    over_family_cases,
)
from tests.fixtures.environments import run_task, workspace_env
from tests.support import say
from tests.support.platform import API_KEY

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.agents.families import Driver
    from tests.support import HudEnv, Models, Platform


@dataclass(frozen=True)
class Signed:
    """How a user signs in (environment variables, ``None`` unsets one) and configures
    the agent, and the endpoint and credential its model is reached with."""

    variables: dict[str, str | None]
    route: str
    credential: str | None
    config: dict[str, Any] = field(default_factory=dict[str, Any])


#: A config value standing for the scripted provider's URL, or the fake gateway's.
PROVIDER, GATEWAY = "<provider>", "<gateway>"


def provider_agent(key: str) -> dict[str, Signed]:
    return {
        "own-key": Signed({"HUD_API_KEY": "hud-key", key: "own-key"}, "provider", "own-key"),
        "hud-key-only": Signed({"HUD_API_KEY": "hud-key", key: None}, "gateway", "hud-key"),
        "gateway-forced-over-own-key": Signed(
            {"HUD_API_KEY": "hud-key", key: "own-key"}, "gateway", "hud-key", {"gateway": True}
        ),
    }


ROUTES: dict[str, dict[str, Signed] | NotApplicable] = {
    "claude": provider_agent("ANTHROPIC_API_KEY"),
    "openai": provider_agent("OPENAI_API_KEY"),
    "gemini": provider_agent("GEMINI_API_KEY"),
    "openai_compatible": {
        "explicit-endpoint": Signed(
            {"HUD_API_KEY": "hud-key"},
            "provider",
            "explicit",
            {"api_key": "explicit", "base_url": PROVIDER},
        ),
        "hud-key-only": Signed({"HUD_API_KEY": "hud-key"}, "gateway", "hud-key"),
        "an-openai-key-is-not-its-own": Signed(
            {"HUD_API_KEY": "hud-key", "OPENAI_API_KEY": "openai-key"}, "gateway", "hud-key"
        ),
    },
    "claude_cli": provider_agent("ANTHROPIC_API_KEY"),
    "codex_cli": {
        **provider_agent("OPENAI_API_KEY"),
        "ambient-login": Signed({"HUD_API_KEY": None, "OPENAI_API_KEY": None}, "ambient", None),
    },
    "browser_use": {
        "own-key": Signed({"ANTHROPIC_API_KEY": "own-key"}, "provider", "own-key"),
        "explicit-key": Signed(
            {"ANTHROPIC_API_KEY": None}, "provider", "explicit", {"api_key": "explicit"}
        ),
    },
}


@pytest.mark.parametrize(("family", "case"), over_family_cases(ROUTES))
async def test_a_family_reaches_its_model_with_the_credential_its_routing_chose(
    family: str, case: Signed, driver: Driver, provider: Models, hud_env: HudEnv
) -> None:
    hud_env.set(
        OPENAI_BASE_URL=provider.url,
        ANTHROPIC_BASE_URL=provider.url,
        GOOGLE_GEMINI_BASE_URL=provider.url,
        **case.variables,
    )
    config = {
        key: provider.url if value == PROVIDER else value for key, value in case.config.items()
    }
    driver.answer("done")

    run = await driver.run(driver.agent(**config))

    (seen,) = driver.seen()
    assert run.trace.status == "completed"
    assert (seen.route, seen.credential) == (case.route, case.credential)
    if seen.route != "gateway":
        assert seen.trace_id is None


@pytest.mark.parametrize(
    ("family", "row"),
    over_families(
        {
            **dict.fromkeys(TOOL_FAMILIES),
            **dict.fromkeys(CLI_FAMILIES),
            "browser_use": NotApplicable("browser-use never routes through the HUD gateway"),
        }
    ),
)
async def test_each_gateway_request_carries_its_own_runs_trace_id(
    family: str, row: None, driver: Driver
) -> None:
    driver.answer("done")
    env = driver.environment()

    job = await Taskset("t", [Task(env=env.name, id="task")]).run(
        driver.agent(), runtime=LocalRuntime(env), group=3
    )

    sent = [seen.trace_id for seen in driver.seen()]
    assert all(seen.route == "gateway" for seen in driver.seen())
    assert sorted(uuid.UUID(str(trace)).hex for trace in sent) == sorted(
        uuid.UUID(str(run.trace_id)).hex for run in job.runs
    )


# ─── refusals ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Refused:
    """A configuration the agent refuses: the sign-in, the config, and the refusal.

    ``at="construction"`` raises ``error`` when the agent is built; ``at="run"``
    ends the run with ``message`` before the model is reached.
    """

    variables: dict[str, str | None]
    message: str
    config: dict[str, Any] = field(default_factory=dict[str, Any])
    at: str = "construction"
    error: type[Exception] = ValueError


BEDROCK_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude"
AWS = {
    "AWS_ACCESS_KEY_ID": "AKIATEST",
    "AWS_SECRET_ACCESS_KEY": "secret",
    "AWS_REGION": "us-east-1",
}
UNSIGNED = {"HUD_API_KEY": None}
GATEWAY_NEEDS_A_KEY = Refused(
    UNSIGNED, "HUD_API_KEY is required for HUD gateway routing", {"gateway": True}, "run"
)

REFUSED: dict[str, dict[str, Refused] | NotApplicable] = {
    "claude": {
        "no-credential": Refused(
            UNSIGNED, "No API key for anthropic", error=HudAuthenticationError
        ),
        "bedrock-without-aws-credentials": Refused(
            {"HUD_API_KEY": "hud-key"},
            "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION are required",
            {"model": BEDROCK_ARN},
            error=HudAuthenticationError,
        ),
        "bedrock-through-the-gateway": Refused(
            {"HUD_API_KEY": "hud-key", **AWS},
            "is a Bedrock inference profile; it cannot use the HUD gateway",
            {"model": BEDROCK_ARN, "gateway": True},
        ),
    },
    "openai": {
        "no-credential": Refused(UNSIGNED, "No API key for openai", error=HudAuthenticationError)
    },
    "gemini": {
        "no-credential": Refused(UNSIGNED, "No API key for gemini", error=HudAuthenticationError)
    },
    "openai_compatible": {
        "no-credential": Refused(UNSIGNED, "No API key found"),
        "a-foreign-key-for-the-hud-gateway": Refused(
            {"HUD_API_KEY": "hud-key"},
            "api_key is not allowed with HUD Gateway",
            {"api_key": "other", "base_url": GATEWAY},
        ),
    },
    "claude_cli": {"gateway-without-a-hud-key": GATEWAY_NEEDS_A_KEY},
    "codex_cli": {"gateway-without-a-hud-key": GATEWAY_NEEDS_A_KEY},
    "browser_use": {
        "no-anthropic-key": Refused(
            {"ANTHROPIC_API_KEY": None, "HUD_API_KEY": "hud-key"},
            "BrowserUseAgent needs an Anthropic API key",
            at="run",
        )
    },
}


@pytest.mark.parametrize(("family", "case"), over_family_cases(REFUSED))
async def test_an_unroutable_configuration_is_refused_before_the_model_is_reached(
    family: str, case: Refused, driver: Driver, models: Models, hud_env: HudEnv
) -> None:
    hud_env.set(**case.variables)
    config = {key: models.url if value == GATEWAY else value for key, value in case.config.items()}
    driver.answer("done")

    if case.at == "construction":
        with pytest.raises(case.error, match=case.message):
            driver.agent(**config)
    else:
        run = await driver.run(driver.agent(**config))
        assert case.message in (run.trace.error or "")
    assert driver.seen() == []


@pytest.mark.parametrize(
    ("family", "row"),
    over_families(
        {
            **dict.fromkeys(CLI_FAMILIES),
            **dict.fromkeys(TOOL_FAMILIES, NotApplicable("calls its provider's API directly")),
            "browser_use": NotApplicable("drives browser-use in this process"),
        }
    ),
)
async def test_a_missing_cli_names_the_runtime_platform(
    family: str, row: None, driver: Driver, tmp_path: Path
) -> None:
    assert isinstance(driver, CliDriver)
    driver.answer("done")
    env = workspace_env(tmp_path / "ws", env=driver.stub.shell_env(path="/usr/bin:/bin"))

    run = await run_task(env, driver.agent())

    assert run.trace.error == IsStr(
        regex=r".* is unavailable for runtime platform [a-z]+-\w+; install it in the "
        r"environment or provide a managed runtime bundle"
    )


# ─── create_agent ──────────────────────────────────────────────────────

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


def serve_catalog(platform: Platform) -> None:
    """The gateway model catalog, served two entries a page."""
    platform.page_size = 2
    for model in CATALOG:
        platform.add_model(**model)


BUILT = {
    "agent-type": (
        lambda: create_agent("openai"),
        ("responses", "gpt-5.6", f"Bearer {API_KEY}"),
    ),
    "agent-type-claude": (
        lambda: create_agent("claude"),
        ("anthropic", "claude-sonnet-4-6", API_KEY),
    ),
    "catalog-id": (
        lambda: create_agent("ft:custom-123"),
        ("chat", "gpt-5.5", f"Bearer {API_KEY}"),
    ),
    "catalog-name-any-case": (
        lambda: create_agent("minimax-m3"),
        ("chat", "MiniMax-M3", f"Bearer {API_KEY}"),
    ),
    "reloaded": (
        lambda: OpenAIAgent.load(create_agent("openai").dump()),
        ("responses", "gpt-5.6", f"Bearer {API_KEY}"),
    ),
}


@pytest.mark.parametrize(("build", "expected"), BUILT.values(), ids=BUILT.keys())
async def test_create_agent_resolves_a_model_and_routes_it_through_the_gateway(
    build: Callable[[], Agent],
    expected: tuple[str, str, str],
    platform: Platform,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(
        OPENAI_API_KEY="openai-key",
        ANTHROPIC_API_KEY="anthropic-key",
        OPENAI_BASE_URL=models.url,
        ANTHROPIC_BASE_URL=models.url,
    )
    serve_catalog(platform)
    models.script([say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), build())

    (request,) = models.requests()
    credential = request.headers.get("authorization") or request.headers.get("x-api-key")
    assert (request.protocol, request.model, credential) == expected
    assert request.headers["trace-id"] == run.trace_id
