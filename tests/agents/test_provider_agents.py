"""What every provider agent does when its provider misbehaves or it runs alongside
itself: a fault ends or retries the run, a capped reply ends it as truncated, and one
agent drives concurrent rollouts."""

from __future__ import annotations

import asyncio
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

PROVIDERS = [
    ("claude", "claude-sonnet-4-6"),
    ("openai", "gpt-5.6"),
    ("gemini", "gemini-3.1-pro-preview"),
    ("openai_compatible", "qwen3.6-plus"),
]
EVERY_PROVIDER = pytest.mark.parametrize(
    ("agent_type", "model"), PROVIDERS, ids=[row[0] for row in PROVIDERS]
)


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="hud-key")


@EVERY_PROVIDER
async def test_one_agent_drives_concurrent_rollouts_each_with_its_own_conversation(
    agent_type: str, model: str, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="done")])
    kind = AgentType(agent_type).cls
    task, agent = workspace_task(tmp_path / "ws"), kind(kind.config_cls(model=model))

    jobs = await asyncio.gather(task(prompt="first").run(agent), task(prompt="second").run(agent))

    assert [run.trace.status for job in jobs for run in job.runs] == ["completed", "completed"]
    assert sorted(request.prompt for request in models.requests()) == ["first", "second"]


@EVERY_PROVIDER
async def test_a_reply_cut_at_the_token_cap_ends_the_run_as_truncated(
    agent_type: str, model: str, models: Models, tmp_path: Path
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
    agent_type: str, model: str, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(fail=Reply(status=400, json={"error": {"message": "model unavailable"}}))])
    kind = AgentType(agent_type).cls

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(kind(kind.config_cls(model=model)))).runs

    assert run.trace.status == "error"
    assert "model unavailable" in (run.trace.error or "")


@EVERY_PROVIDER
async def test_an_overloaded_provider_is_asked_again(
    agent_type: str, model: str, models: Models, tmp_path: Path
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
