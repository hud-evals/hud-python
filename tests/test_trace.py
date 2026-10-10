"""What a run's ``Trace`` answers about the steps it recorded."""

from __future__ import annotations

import pytest

from hud import Environment, Trace
from hud.eval import LocalRuntime, Task, rollout
from tests.harness import ScriptedAgent


def _env() -> Environment:
    env = Environment("trace")

    @env.template()
    async def answer(broken: bool = False):
        reply = yield "Reply with ok."
        if broken:
            raise RuntimeError("grader broke")
        yield 1.0 if reply == "ok" else 0.0

    return env


@pytest.mark.parametrize(
    ("broken", "sources", "status", "error"),
    [
        pytest.param(False, ["task", "user", "task"], "completed", None, id="graded"),
        pytest.param(
            True,
            ["task", "user", "system"],
            "error",
            "[grading] hud.clients.client.HudProtocolError: hud rpc error -32000: grader broke",
            id="broken",
        ),
    ],
)
async def test_a_runs_trace_answers_from_its_recorded_steps(
    broken: bool, sources: list[str], status: str, error: str | None
) -> None:
    task = Task(env="trace", id="answer", args={"broken": broken})

    run = await rollout(task, ScriptedAgent("ok"), runtime=LocalRuntime(_env()))

    trace = run.trace
    assert len(trace) == len(sources)
    assert trace.collect(lambda step: step.source) == sources
    assert [step.step_id for step in trace.steps] == [1, 2, 3]
    assert trace.final(lambda step: step.source) == sources[-1]
    assert trace.final(lambda step: step.extra.get("missing")) is None
    assert trace.status == status
    assert trace.is_error is (status == "error")
    assert trace.error == error
    assert all(step.ended_at is not None for step in trace.steps)


async def test_a_dumped_trace_reloads_with_its_steps_renumbered() -> None:
    run = await rollout(
        Task(env="trace", id="answer"), ScriptedAgent("ok"), runtime=LocalRuntime(_env())
    )
    dumped = run.trace.model_dump(mode="json")
    for step in dumped["steps"]:
        step.pop("step_id")

    reloaded = Trace.model_validate(dumped)

    assert [step.step_id for step in reloaded.steps] == [1, 2, 3]
    assert reloaded.content == "ok"
