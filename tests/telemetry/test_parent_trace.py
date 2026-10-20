"""A rollout started inside another trace names that trace as its parent to everything it
reports to: the platform's trace record, the model gateway, and a hosted submission. Its
own trace is never its parent."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

from hud import Environment
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import HostedRuntime, LocalRuntime, Task, rollout
from hud.telemetry.context import set_trace_context
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.models import say

if TYPE_CHECKING:
    from tests.conftest import HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

TRACE_ID = "b" * 32
AMBIENTS = [("a" * 32, "a" * 32), (str(uuid.UUID(TRACE_ID)), None)]
AMBIENT_IDS = ["another-trace-is-the-parent", "its-own-trace-is-not-its-parent"]


def checked() -> Environment:
    env = Environment("checked")

    @env.template()
    async def check():
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    return env


@pytest.mark.parametrize(("ambient", "parent"), AMBIENTS, ids=AMBIENT_IDS)
async def test_the_trace_record_names_the_parent_trace(
    ambient: str, parent: str | None, reporting: FakeServices
) -> None:
    with set_trace_context(ambient):
        await rollout(
            Task(env="checked", id="check"),
            ScriptedAgent("yes"),
            runtime=LocalRuntime(checked()),
            trace_id=TRACE_ID,
        )

    (entered,) = reporting.bodies("api", "POST", "/v2/trace/{id}/enter")
    assert entered.get("parent_trace_id") == parent


@pytest.mark.parametrize(("ambient", "parent"), AMBIENTS, ids=AMBIENT_IDS)
async def test_the_model_gateway_is_told_the_parent_trace(
    ambient: str, parent: str | None, models: Models, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("yes")])

    with set_trace_context(ambient):
        await rollout(
            Task(env="checked", id="check"),
            OpenAIChatAgent(OpenAIChatConfig(model="m")),
            runtime=LocalRuntime(checked()),
            trace_id=TRACE_ID,
        )

    (request,) = models.requests()
    sent = request.headers.get("x-hud-parent-trace-id")
    assert (uuid.UUID(sent).hex if sent else None) == parent


@pytest.mark.parametrize(
    ("ambient", "parent"),
    [*AMBIENTS, ("external-run-id", None)],
    ids=[*AMBIENT_IDS, "an-id-that-is-not-a-trace-is-no-parent"],
)
async def test_a_hosted_submission_names_the_parent_trace(
    ambient: str, parent: str | None, services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    services.route("api", "POST", "/v2/rollouts/submit", json={})
    services.route("api", "GET", "/v2/trace/{id}", json={"status": "completed", "reward": 1.0})
    agent = OpenAIChatAgent(
        OpenAIChatConfig(model="m", api_key="secret", base_url="http://model.test")
    )

    with set_trace_context(ambient):
        await HostedRuntime(poll_interval=0).run(
            Task(env="checked", id="check"), agent, job_id="c" * 32, trace_id=TRACE_ID
        )

    (submitted,) = services.bodies("api", "POST", "/v2/rollouts/submit")
    sent = submitted.get("parent_trace_id")
    assert (uuid.UUID(sent).hex if sent else None) == parent
