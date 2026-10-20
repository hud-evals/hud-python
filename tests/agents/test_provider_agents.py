"""What every provider agent does over its own API: what its model is handed, how its tool
calls reach the environment, how its provider's failures end or retry a run, and where its
requests go with which credential. Each family in ``AgentType`` that calls a model API
itself is a row."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP

from hud.agents.types import ToolStep
from hud.capabilities import Capability, RFBClient
from hud.eval import LocalRuntime, Task, Taskset
from hud.types import AgentType
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Reply, Turn, click, fake_screen, mcp_server, say, shell

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, Models

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


def test_every_family_that_calls_a_model_api_is_a_row() -> None:
    assert {row[0] for row in PROVIDERS} == {kind.value for kind in AgentType if not kind.is_cli}


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@EVERY_PROVIDER
async def test_the_model_is_given_the_prompt_a_shell_and_the_environments_tools(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    lookups = FastMCP("tools")
    lookups.tool(name="lookup", description="Look a key up.")(lambda key: key)
    models.script([say("done")])
    kind = AgentType(agent_type).cls

    async with mcp_server(lookups) as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        await run_task(env, kind(kind.config_cls(model=model)), prompt="Fix the bug.")

    (request,) = models.requests()
    assert (request.prompt, "lookup" in request.tools) == ("Fix the bug.", True)
    assert request.shell_tool is not None


@EVERY_PROVIDER
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_conversation(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([say("done")])
    kind = AgentType(agent_type).cls
    env, agent = workspace_env(tmp_path / "ws"), kind(kind.config_cls(model=model))

    runs = await asyncio.gather(
        run_task(env, agent, prompt="first"), run_task(env, agent, prompt="second")
    )

    assert [run.trace.status for run in runs] == ["completed", "completed"]
    assert sorted(request.prompt for request in models.requests()) == ["first", "second"]


@EVERY_PROVIDER
async def test_a_shell_call_acts_on_the_workspace_and_its_result_reaches_the_model(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([shell("touch made && echo made-it"), say("done")])
    kind = AgentType(agent_type).cls

    run = await run_task(workspace_env(tmp_path / "ws"), kind(kind.config_cls(model=model)))

    assert (run.trace.status, run.trace.content) == ("completed", "done")
    assert (tmp_path / "ws" / "made").exists()
    assert sum(isinstance(step, ToolStep) for step in run.trace.steps) == 1
    assert "made-it" in json.dumps(models.requests()[-1].body)


#: (agent type, model) for the families whose tools drive a screen
COMPUTER_USERS = [
    ("claude", "claude-sonnet-4-6"),
    ("openai", "gpt-5.6"),
    ("gemini", "gemini-3.1-pro-preview"),
]


def test_every_family_that_drives_a_screen_is_a_computer_use_row() -> None:
    assert {row[0] for row in COMPUTER_USERS} == {
        kind.value for kind in AgentType if RFBClient in getattr(kind.cls, "clients", ())
    }


@pytest.mark.parametrize(
    ("agent_type", "model"), COMPUTER_USERS, ids=[r[0] for r in COMPUTER_USERS]
)
async def test_a_click_lands_on_the_screen_where_the_model_pointed(
    agent_type: str, model: str, models: Models, tmp_path: Path
) -> None:
    models.script([click(150, 25, screen=(200, 100)), say("done")])
    kind = AgentType(agent_type).cls

    async with fake_screen(width=200, height=100) as screen:
        env = workspace_env(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        run = await run_task(env, kind(kind.config_cls(model=model)))

    pressed = {(event.x, event.y) for event in screen.pointer_events() if event.buttons & 1}
    assert (run.trace.status, pressed) == ("completed", {(150, 25)})
    assert "image" in json.dumps(models.requests()[-1].body)


@EVERY_PROVIDER
async def test_a_reply_cut_at_the_token_cap_ends_the_run_as_truncated(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="part", truncated=True)])
    kind = AgentType(agent_type).cls

    run = await run_task(workspace_env(tmp_path / "ws"), kind(kind.config_cls(model=model)))

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

    run = await run_task(workspace_env(tmp_path / "ws"), kind(kind.config_cls(model=model)))

    assert run.trace.status == "error"
    assert "model unavailable" in (run.trace.error or "")


@EVERY_PROVIDER
async def test_an_overloaded_provider_is_asked_again(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    busy = Reply(status=503, headers={"retry-after-ms": "0"}, json={"error": {"message": "busy"}})
    models.queue([Turn(fail=busy), say("done")])
    kind = AgentType(agent_type).cls

    run = await run_task(workspace_env(tmp_path / "ws"), kind(kind.config_cls(model=model)))

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
    models.script([say("done")])
    provider.script([say("done")])
    kind = AgentType(agent_type).cls

    run = await run_task(
        workspace_env(tmp_path / "ws"), kind(kind.config_cls(model=model, gateway=gateway))
    )

    (request,) = {"gateway": models, "provider": provider}[route].requests()
    assert (run.trace.status, request.credential) == ("completed", credential)
    assert ("trace-id" in request.headers) is (route == "gateway")


@EVERY_PROVIDER
async def test_each_gateway_request_carries_its_own_runs_trace_id(
    agent_type: str, model: str, key: str | None, models: Models, tmp_path: Path
) -> None:
    models.script([say("done")])
    env = workspace_env(tmp_path / "ws")
    kind = AgentType(agent_type).cls

    job = await Taskset("t", [Task(env=env.name, id="task")]).run(
        kind(kind.config_cls(model=model)), runtime=LocalRuntime(env), group=3
    )

    sent = sorted(uuid.UUID(request.headers["trace-id"]).hex for request in models.requests())
    assert sent == sorted(uuid.UUID(str(run.trace_id)).hex for run in job.runs)
