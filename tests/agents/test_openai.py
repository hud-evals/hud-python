"""What only ``OpenAIAgent`` does over the Responses API: the tool search it offers, how it
chains responses and discards an unusable one, how a refused, capped or incomplete call
lands on the trace, and the citations it records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.agents import OpenAIAgent
from hud.agents.types import OpenAIConfig
from tests.agents.conftest import tool_results
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, shell_call

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "gpt-5.6"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


async def test_each_request_sends_only_what_is_new_since_the_response_it_continues(
    models: Models, tmp_path: Path
) -> None:
    models.script([shell_call("echo hi"), Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))

    first, second = (request.body for request in models.requests())
    assert (first.get("previous_response_id"), second["previous_response_id"]) == (None, "resp_1")
    assert [item["type"] for item in second["input"]] == ["shell_call_output"]


async def test_a_shell_calls_output_cap_keeps_the_head_and_tail_of_its_output(
    models: Models, tmp_path: Path
) -> None:
    models.script([shell_call("seq 1 1000", max_output_length=50), Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))).runs

    ((_, text),) = tool_results(run)
    assert len(text) <= 50
    assert text.startswith("1\n2\n") and text.endswith("999\n1000\n")
