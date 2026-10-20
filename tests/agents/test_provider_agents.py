"""What every provider agent does over its own API: how its tool calls reach the environment,
how its provider's failures end or retry a run, and where its requests go with which
credential."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import pytest

from hud.types import AgentType
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

#: (agent type, model, the variable holding the provider's own key)
PROVIDERS = [
    ("claude", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    ("openai", "gpt-5.6", "OPENAI_API_KEY"),
    ("gemini", "gemini-3.1-pro-preview", "GEMINI_API_KEY"),
    ("openai_compatible", "qwen3.6-plus", None),
]
EVERY_PROVIDER = pytest.mark.parametrize(
    ("agent_type", "model", "key"), PROVIDERS, ids=[row[0] for row in PROVIDERS]
)


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@EVERY_PROVIDER
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_conversation(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="done")])
    kind = AgentType(agent_type).cls
    task, agent = workspace_task(tmp_path / "ws"), kind(kind.config_cls(model=model))

    jobs = await asyncio.gather(task(prompt="first").run(agent), task(prompt="second").run(agent))

    assert [run.trace.status for job in jobs for run in job.runs] == ["completed", "completed"]
    assert sorted(request.prompt for request in models.requests()) == ["first", "second"]


#: (agent type, model) for the families whose tools drive a screen
COMPUTER_USERS = [
    ("claude", "claude-sonnet-4-6"),
    ("openai", "gpt-5.6"),
    ("gemini", "gemini-3.1-pro-preview"),
]


@EVERY_PROVIDER
async def test_a_reply_cut_at_the_token_cap_ends_the_run_as_truncated(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="part", truncated=True)])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(kind(kind.config_cls(model=model)))).runs

    assert (run.trace.status, run.trace.stop_reason, run.trace.content) == (
        "completed",
        "length",
        "part",
    )


@EVERY_PROVIDER
async def test_a_provider_fault_ends_the_run_with_its_error(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(fail=Reply(status=400, json={"error": {"message": "model unavailable"}}))])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(kind(kind.config_cls(model=model)))).runs

    assert run.trace.status == "error"
    assert "model unavailable" in (run.trace.error or "")


@EVERY_PROVIDER
async def test_an_overloaded_provider_is_asked_again(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    busy = Reply(status=503, headers={"retry-after-ms": "0"}, json={"error": {"message": "busy"}})
    models.queue([Turn(fail=busy), Turn(text="done")])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(kind(kind.config_cls(model=model)))).runs

    assert (run.trace.status, run.trace.content, len(models.requests())) == (
        "completed",
        "done",
        2,
    )


@pytest.mark.parametrize(
    ("agent_type", "model", "key"),
    [row for row in PROVIDERS if row[2] is not None],
    ids=[row[0] for row in PROVIDERS if row[2] is not None],
)
@pytest.mark.parametrize(
    ("own_key", "gateway", "route", "credential"),
    [
        ("own-key", False, "provider", "own-key"),
        (None, False, "gateway", "hud-key"),
        ("own-key", True, "gateway", "hud-key"),
    ],
    ids=["its-own-key", "only-a-hud-key", "the-gateway-forced-over-its-own-key"],
)
async def test_requests_go_to_the_provider_with_its_own_key_and_otherwise_to_the_gateway(
    agent_type: str,
    model: str,
    key: str,
    own_key: str | None,
    gateway: bool,
    route: str,
    credential: str,
    models: Models,
    provider: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(
        OPENAI_BASE_URL=provider.url,
        ANTHROPIC_BASE_URL=provider.url,
        GOOGLE_GEMINI_BASE_URL=provider.url,
        **{key: own_key},
    )
    models.script([Turn(text="done")])
    provider.script([Turn(text="done")])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(kind(kind.config_cls(model=model, gateway=gateway)))).runs

    (request,) = {"gateway": models, "provider": provider}[route].requests()
    assert (run.trace.status, request.credential) == ("completed", credential)
    assert ("trace-id" in request.headers) is (route == "gateway")


@EVERY_PROVIDER
async def test_each_gateway_request_carries_its_own_runs_trace_id(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="done")])
    task = workspace_task(tmp_path / "ws")
    kind = AgentType(agent_type).cls

    job = await task().run(kind(kind.config_cls(model=model)), group=3)

    sent = sorted(uuid.UUID(request.headers["trace-id"]).hex for request in models.requests())
    assert sent == sorted(uuid.UUID(str(run.trace_id)).hex for run in job.runs)
