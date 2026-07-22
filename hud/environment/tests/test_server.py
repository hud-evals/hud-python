"""The wire grade contract: ``tasks.grade`` frames carry a numeric ``score``.

The server normalizes every grade yield to the canonical frame and fails
loudly on authoring bugs (a grade that is neither a number, a ``.reward``
object, nor a ``{"score": ...}`` dict) instead of silently grading 0.0.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from hud.clients import HudProtocolError
from hud.environment import Answer, Environment
from hud.eval import Run
from hud.graders import EvaluationResult, SubScore

from .conftest import served


class _Payload(BaseModel):
    text: str


_Mode = Literal["upper", "lower"]


async def test_dict_grade_without_numeric_score_errors_loudly() -> None:
    env = Environment("badgrade")

    @env.template()
    async def reward_keyed():
        yield "go"
        yield {"reward": 1.0}  # wrong key: the wire grade frame is {"score": ...}

    async with served(env) as client:
        with pytest.raises(HudProtocolError, match="score"):
            async with Run(client, "reward_keyed", {}) as run:
                run.trace.content = "x"


async def test_non_numeric_grade_errors_loudly() -> None:
    env = Environment("badgrade")

    @env.template()
    async def stringy():
        yield "go"
        yield "great job"

    async with served(env) as client:
        with pytest.raises(HudProtocolError, match="yield a number"):
            async with Run(client, "stringy", {}) as run:
                run.trace.content = "x"


async def test_score_dict_passes_through_with_extra_keys() -> None:
    env = Environment("richgrade")

    @env.template()
    async def rich():
        yield "go"
        yield {"score": 0.5, "info": {"detail": "partial credit"}}

    async with served(env) as client:
        async with Run(client, "rich", {}) as run:
            run.trace.content = "x"
        assert run.reward == 0.5
        assert run.grade.info == {"detail": "partial credit"}


async def test_evaluation_result_info_reaches_evaluate_step() -> None:
    env = Environment("modelgrade")

    @env.template()
    async def graded():
        yield "go"
        yield EvaluationResult(
            reward=0.75,
            content="nice",
            info={"max_tile": 256},
            subscores=[
                SubScore(
                    name="judge",
                    value=0.75,
                    children=[
                        SubScore(
                            name="criterion",
                            value=1.0,
                            info={"reason": "because"},
                        )
                    ],
                    info={"model": "judge-model"},
                )
            ],
        )

    async with served(env) as client:
        async with Run(client, "graded", {}) as run:
            run.trace.content = "x"
        assert run.reward == 0.75
        assert run.grade.info == {"max_tile": 256}
        assert "info" not in run.evaluation["subscores"][0]
        assert "info" not in run.evaluation["subscores"][0]["children"][0]
        evaluate_step = run.trace.steps[-1]
        assert evaluate_step.task_call is not None
        assert evaluate_step.task_call.phase == "evaluate"
        assert evaluate_step.task_call.result == {
            "score": 0.75,
            "done": True,
            "content": "nice",
            "info": {"max_tile": 256},
            "isError": False,
            "subscores": [
                {
                    "name": "judge",
                    "weight": 1.0,
                    "value": 0.75,
                    "children": [
                        {
                            "name": "criterion",
                            "weight": 1.0,
                            "value": 1.0,
                            "children": None,
                            "info": {"reason": "because"},
                        }
                    ],
                    "info": {"model": "judge-model"},
                }
            ],
        }


def test_answer_holds_parsed_content_and_raw_string() -> None:
    answer = Answer(content={"final": "42"}, raw='{"final": "42"}')
    assert answer.content == {"final": "42"}
    assert answer.raw == '{"final": "42"}'


async def test_start_coerces_postponed_rich_annotations() -> None:
    env = Environment("coerce")

    @env.template()
    async def typed(mode: _Mode, payload: _Payload, retries: int | None = None):
        if mode == "upper":
            prompt = payload.text.upper()
        elif mode == "lower":
            prompt = payload.text.lower()
        else:
            raise ValueError(f"unexpected mode: {mode!r}")
        if retries is not None:
            prompt += "!" * retries
        yield prompt
        yield 1.0

    assert callable(typed)
    async with served(env) as client:
        async with Run(
            client,
            "typed",
            {
                "mode": '"upper"',
                "payload": '{"text":"hello"}',
                "retries": "3",
            },
        ) as run:
            run.trace.content = "x"
        assert run.prompt == "HELLO!!!"
