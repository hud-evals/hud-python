"""``one_shot``: one model call, no tools, the reply is the answer.

Checked once on the chat completions agent, since the path is the shared loop's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.agents import OpenAIChatAgent
from hud.agents.types import AgentStep, OpenAIChatConfig
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


SCHEMA = {"type": "object", "properties": {"spans": {"type": "array"}}, "required": ["spans"]}


def one_shot(
    *, response_schema: dict[str, Any] | None = None, system_prompt: str | None = None
) -> OpenAIChatAgent:
    config = OpenAIChatConfig(
        model="qwen3.6-plus",
        one_shot=True,
        response_schema=response_schema,
        system_prompt=system_prompt,
    )
    return OpenAIChatAgent(config)


async def test_one_call_with_no_tools_and_the_reply_is_the_graded_answer(
    models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text='{"spans": []}'), Turn(text="never asked")])
    task = workspace_task(tmp_path / "ws")

    (run,) = (await task().run(one_shot(response_schema=SCHEMA))).runs

    (request,) = models.requests()
    assert request.tools == []
    system = request.body["messages"][0]
    assert system["role"] == "system" and '"required": ["spans"]' in system["content"]
    assert run.trace.content == '{"spans": []}'
    assert (run.trace.status, run.trace.stop_reason, run.reward) == ("completed", "done", 1.0)
    assert [type(step).__name__ for step in run.trace.steps] == [
        "Step",
        "Step",
        "AgentStep",
        "Step",
    ]  # task setup, prompt, the one model turn, task evaluate


async def test_the_system_prompt_survives_and_the_schema_is_optional(
    models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="yes")])
    task = workspace_task(tmp_path / "ws")

    await task().run(one_shot(system_prompt="Be terse."))

    (request,) = models.requests()
    assert request.body["messages"][0] == {"role": "system", "content": "Be terse."}


async def test_a_truncated_reply_is_still_the_answer_but_marked_length(
    models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text='{"spans": [', truncated=True)])
    task = workspace_task(tmp_path / "ws")

    (run,) = (await task().run(one_shot(response_schema=SCHEMA))).runs

    assert (run.trace.status, run.trace.stop_reason) == ("completed", "length")
    assert run.trace.content == '{"spans": ['


async def test_a_tool_call_in_a_one_shot_reply_is_not_executed(
    models: Models, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    models.script([call("bash", command="touch nope")])
    task = workspace_task(root)

    (run,) = (await task().run(one_shot())).runs

    assert len(models.requests()) == 1
    assert not (root / "nope").exists()
    assert isinstance(run.trace.steps[2], AgentStep)
    assert not run.trace.content


async def test_the_config_round_trips_for_hosted_execution() -> None:
    agent = one_shot(response_schema=SCHEMA)

    rebuilt = OpenAIChatAgent.load(agent.dump())

    assert rebuilt.config.one_shot is True
    assert rebuilt.config.response_schema == SCHEMA


def test_templates_expose_their_returns_schema_for_one_shot_prompts() -> None:
    from pydantic import BaseModel

    class Submission(BaseModel):
        spans: list[str]

    env = Environment("docs")

    @env.template(returns=Submission)
    async def scan(record_id: str):
        yield "go"
        yield 1.0

    assert scan.manifest_entry()["returns"]["required"] == ["spans"]
