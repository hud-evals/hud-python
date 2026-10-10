"""``RegradeAgent``: step filtering, copy isolation, and answer injection."""

from __future__ import annotations

from hud.agents.regrade import RegradeAgent
from hud.eval.run import Run
from hud.types import Step


def test_regrade_agent_filters_to_agent_tool_subagent_sources() -> None:
    steps = [
        Step(source="task"),
        Step(source="agent"),
        Step(source="tool"),
        Step(source="subagent"),
        Step(source="system"),
    ]
    agent = RegradeAgent(steps, answer="42")
    assert [s.source for s in agent._replay] == ["agent", "tool", "subagent"]


async def test_regrade_agent_copies_steps_so_originals_are_not_mutated() -> None:
    original = Step(source="agent")
    agent = RegradeAgent([original], answer="x")
    run = Run(None, "", {})

    await agent(run)

    # record() sets step_id on the replayed copy, not on the original
    assert original.step_id is None
    assert run.trace.steps[0].step_id == 1


async def test_regrade_agent_injects_answer_and_replays_steps_into_run() -> None:
    steps = [Step(source="agent"), Step(source="tool")]
    agent = RegradeAgent(steps, answer="the stored answer")
    run = Run(None, "", {})

    await agent(run)

    assert run.trace.content == "the stored answer"
    assert len(run.trace.steps) == 2
    assert run.trace.steps[0].source == "agent"
    assert run.trace.steps[1].source == "tool"
