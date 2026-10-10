"""``RegradeAgent``: replays a stored trajectory and injects a stored answer.

Used by the regrade engine to re-run the grader without re-running the LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.agents.base import Agent

if TYPE_CHECKING:
    from hud.eval.run import Run
    from hud.types import Step


class RegradeAgent(Agent):
    """Replays recorded steps and injects a stored answer, skipping the LLM entirely.

    Filters the supplied steps to agent, tool, and subagent sources; task setup
    and evaluate steps are excluded because ``Run.__aenter__`` / ``Run.__aexit__``
    re-record them naturally during the regrade rollout.
    """

    def __init__(self, steps: list[Step], answer: str | None) -> None:
        super().__init__()
        self._replay = [
            step.model_copy(update={"started_at": None, "ended_at": None})
            for step in steps
            if step.source in {"agent", "tool", "subagent"}
        ]
        self._answer = answer

    async def __call__(self, run: Run) -> None:
        for step in self._replay:
            run.record(step)
        run.trace.content = self._answer


__all__ = ["RegradeAgent"]
