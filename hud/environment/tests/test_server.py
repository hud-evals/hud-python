"""The wire grade contract: ``tasks.grade`` frames carry a numeric ``score``.

The server normalizes every grade yield to the canonical frame and fails
loudly on authoring bugs (a grade that is neither a number, a ``.reward``
object, nor a ``{"score": ...}`` dict) instead of silently grading 0.0.
"""

from __future__ import annotations

import pytest

from hud.clients import HudProtocolError
from hud.environment import Environment
from hud.eval import Run

from .conftest import served


async def test_dict_grade_without_numeric_score_errors_loudly() -> None:
    env = Environment("badgrade")

    @env.task()
    async def reward_keyed():
        yield "go"
        yield {"reward": 1.0}  # wrong key: the wire grade frame is {"score": ...}

    async with served(env) as client:
        with pytest.raises(HudProtocolError, match="score"):
            async with Run(client, "reward_keyed", {}) as run:
                run.trace.content = "x"


async def test_non_numeric_grade_errors_loudly() -> None:
    env = Environment("badgrade")

    @env.task()
    async def stringy():
        yield "go"
        yield "great job"

    async with served(env) as client:
        with pytest.raises(HudProtocolError, match="yield a number"):
            async with Run(client, "stringy", {}) as run:
                run.trace.content = "x"


async def test_score_dict_passes_through_with_extra_keys() -> None:
    env = Environment("richgrade")

    @env.task()
    async def rich():
        yield "go"
        yield {"score": 0.5, "info": {"detail": "partial credit"}}

    async with served(env) as client:
        async with Run(client, "rich", {}) as run:
            run.trace.content = "x"
        assert run.reward == 0.5
        assert run.grade.info == {"detail": "partial credit"}
