"""A rollout started inside another trace names that trace as its parent in the platform's
trace record. Its own trace is never its parent."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

from hud.eval import LocalRuntime, Task, rollout
from hud.telemetry.context import set_trace_context
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.environments import checked

if TYPE_CHECKING:
    from tests.fixtures.services import FakeServices

TRACE_ID = "b" * 32
AMBIENTS = [("a" * 32, "a" * 32), (str(uuid.UUID(TRACE_ID)), None)]
AMBIENT_IDS = ["another-trace-is-the-parent", "its-own-trace-is-not-its-parent"]


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
