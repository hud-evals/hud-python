"""Reference agents that call no model, for checking a task's grading.

``OracleAgent`` submits the task's reference solution, so a sound task grades it as a
pass; ``NopAgent`` does nothing, so a sound task grades it as a fail.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from hud.agents.base import Agent
from hud.agents.types import NopConfig, OracleConfig

if TYPE_CHECKING:
    from hud.eval.run import Run


class OracleAgent(Agent[OracleConfig]):
    """Has the environment run the task's reference solution and submits its answer.

    The task's template must register one with ``@template.solution``; a task without
    one fails the run with an error naming the task.
    """

    config_cls = OracleConfig

    async def __call__(self, run: Run) -> None:
        answer = await run.client.solve()
        run.trace.content = (
            answer if answer is None or isinstance(answer, str) else json.dumps(answer)
        )


class NopAgent(Agent[NopConfig]):
    """Submits no answer and leaves the environment untouched."""

    config_cls = NopConfig

    async def __call__(self, run: Run) -> None:
        return
