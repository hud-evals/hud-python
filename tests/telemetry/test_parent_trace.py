"""A rollout started inside another trace names that trace as its parent to everything it
reports to: the platform's trace record, the model gateway, and the hosted submission.

Each row runs one rollout with trace id ``TRACE_ID`` under an ambient trace context
and reads the parent the consumer received.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import OpenAIChatConfig
from hud.eval import HostedRuntime, LocalRuntime, Task, rollout
from hud.telemetry.context import set_trace_context
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import lab, solve
from tests.support import say
from tests.support.platform import key

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.support import Models, Platform

ADD = Task(env="lab", id="add", args={"a": 2, "b": 3})
TRACE_ID = "b" * 32


async def trace_record(platform: Platform, models: Models) -> Any:
    await rollout(ADD, ScriptedAgent(solve), runtime=LocalRuntime(lab()), trace_id=TRACE_ID)
    return (platform.traces[key(TRACE_ID)].enter or {}).get("parent_trace_id")


async def gateway_request(platform: Platform, models: Models) -> Any:
    models.script([say("5")])
    agent = OpenAIChatAgent(OpenAIChatConfig(model="m", max_steps=2))
    await rollout(ADD, agent, runtime=LocalRuntime(lab()), trace_id=TRACE_ID)
    (request,) = models.requests()
    return request.headers.get("x-hud-parent-trace-id")


async def hosted_submission(platform: Platform, models: Models) -> Any:
    agent = OpenAIChatAgent(
        OpenAIChatConfig(model="m", api_key="secret", base_url="http://model.test")
    )
    await HostedRuntime(poll_interval=0).run(ADD, agent, job_id="c" * 32, trace_id=TRACE_ID)
    (trace,) = platform.traces.values()
    return (trace.submitted or {}).get("parent_trace_id")


CONSUMERS: dict[str, Callable[[Platform, Models], Awaitable[Any]]] = {
    "the trace record": trace_record,
    "the model gateway": gateway_request,
    "the hosted submission": hosted_submission,
}
AMBIENTS = {
    "another trace is the parent": ("a" * 32, "a" * 32, set(CONSUMERS)),
    "the rollout's own trace is not its parent": (
        str(uuid.UUID(TRACE_ID)),
        None,
        set(CONSUMERS),
    ),
    "an id that is not a trace is no parent": ("external-run-id", None, {"the hosted submission"}),
}


@pytest.mark.parametrize(
    ("consumer", "ambient", "parent"),
    [
        pytest.param(consumer, ambient, parent, id=f"{name}-{consumer}")
        for name, (ambient, parent, consumers) in AMBIENTS.items()
        for consumer in CONSUMERS
        if consumer in consumers
    ],
)
async def test_a_nested_rollout_names_its_parent_trace_to_each_consumer(
    consumer: str, ambient: str, parent: str | None, reporting: Platform, models: Models
) -> None:
    with set_trace_context(ambient):
        received = await CONSUMERS[consumer](reporting, models)

    assert (uuid.UUID(received).hex if received else None) == parent
