"""Graders as task authors use them: a template grades the agent's answer with ``combine``,
the bash grader, the LLM judge or a grader of its own, and the run's reward and
evaluation show the result."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.graders import (
    BashGrader,
    Grader,
    LLMJudgeGrader,
    SubScore,
    combine,
    combine_all,
    combine_any,
    exact_match,
)
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.models import Turn

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import ModelRequest, Models

ANSWER = "Paris is the capital."


def leaf(name: str, weight: float, value: float) -> dict[str, Any]:
    return {"name": name, "weight": weight, "value": value, "children": None}


async def test_repeated_graders_are_named_apart_and_weighted_into_the_reward() -> None:
    env = Environment("graders")

    @env.template()
    async def check():
        yield "go"
        yield await combine(
            BashGrader.grade(weight=0.3, command="true"),
            BashGrader.grade(weight=0.3, command="true"),
            SubScore(name="BashGrader-1", value=0.0, weight=0.4),
        )

    (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    assert run.reward == pytest.approx(0.6)
    assert run.evaluation["subscores"] == [
        leaf("BashGrader-2", 0.3, 1.0),
        leaf("BashGrader-3", 0.3, 1.0),
        leaf("BashGrader-1", 0.4, 0.0),
    ]


async def test_any_and_all_combine_their_children_into_one_weighted_subscore() -> None:
    env = Environment("graders")

    @env.template()
    async def check():
        answer = yield "Name the capital of France."
        either = combine_any(
            weight=0.5, subscores=[SubScore(name="a", value=0.0), SubScore(name="b", value=1.0)]
        )
        both = combine_all(
            weight=0.25,
            subscores=[SubScore(name="lint", value=1.0), SubScore(name="types", value=0.0)],
            name="checks",
        )
        exact = SubScore(name="format", value=exact_match(answer, ANSWER), weight=0.25)
        yield await combine(either, both, exact)

    (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    assert run.reward == pytest.approx(0.75)
    assert run.evaluation["subscores"] == [
        {
            "name": "any",
            "weight": 0.5,
            "value": 1.0,
            "children": [leaf("a", 1.0, 0.0), leaf("b", 1.0, 1.0)],
        },
        {
            "name": "checks",
            "weight": 0.25,
            "value": 0.0,
            "children": [leaf("lint", 1.0, 1.0), leaf("types", 1.0, 0.0)],
        },
        leaf("format", 0.25, 1.0),
    ]


@pytest.mark.parametrize(
    ("correct", "reward"), [(1.0, 0.8), (0.0, -0.2)], ids=["passing", "failing-goes-negative"]
)
async def test_a_negative_weight_is_a_penalty_taken_off_the_reward(
    correct: float, reward: float
) -> None:
    env = Environment("graders")

    @env.template()
    async def check():
        yield "go"
        yield await combine(
            SubScore(name="right", value=correct, weight=1.0),
            SubScore(name="penalty", value=1.0, weight=-0.2),
        )

    (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    assert run.reward == pytest.approx(reward)
    assert [s["weight"] for s in run.evaluation["subscores"]] == [1.0, -0.2]


async def test_weights_not_summing_to_one_are_normalized_with_a_warning_and_penalties_kept() -> (
    None
):
    env = Environment("graders")

    @env.template()
    async def check():
        yield "go"
        yield await combine(
            SubScore(name="a", value=1.0, weight=0.5),
            SubScore(name="b", value=1.0, weight=0.3),
            SubScore(name="c", value=0.0, weight=0.3),
            SubScore(name="penalty", value=1.0, weight=-0.1),
        )

    with pytest.warns(UserWarning, match="grader weights sum to 1.1000, not 1.0"):
        (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    assert run.reward == pytest.approx(0.8 / 1.1 - 0.1)
    assert [s["weight"] for s in run.evaluation["subscores"]] == pytest.approx(
        [0.5 / 1.1, 0.3 / 1.1, 0.3 / 1.1, -0.1]
    )


async def test_a_custom_grader_returns_a_number_a_pair_or_a_subscore() -> None:
    class Scaled(Grader):
        name = "scaled"

        @classmethod
        async def compute_score(cls, **kwargs: Any) -> Any:
            return 0.75

    class Explained(Grader):
        name = "explained"

        @classmethod
        async def compute_score(cls, **kwargs: Any) -> Any:
            return 0.5, {"source": "pair"}

    class Detailed(Grader):
        name = "detailed"

        @classmethod
        async def compute_score(cls, **kwargs: Any) -> Any:
            return SubScore(name="ignored", value=1.0, info={"reason": "did it"})

    env = Environment("graders")

    @env.template()
    async def check():
        yield "go"
        yield await combine(
            Scaled.grade(weight=0.25, payload=object()),
            Explained.grade(weight=0.25, level=2),
            Detailed.grade(weight=0.25, name="renamed"),
            SubScore.model_validate(
                {"name": "legacy", "value": 1.0, "weight": 0.25, "metadata": {"from": "metadata"}}
            ),
        )

    (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    info = {s["name"]: s["info"] for s in run.grade.raw["subscores"]}
    assert run.reward == pytest.approx(0.8125)
    assert info.keys() == {"scaled", "explained", "renamed", "legacy"}
    assert (info["explained"]["source"], info["renamed"]["reason"], info["legacy"]["from"]) == (
        "pair",
        "did it",
        "metadata",
    )


@pytest.mark.parametrize(
    ("command", "limit", "value", "info"),
    [
        ("echo oops >&2; false", None, 0.0, {"exit_code": 1, "stdout": "", "stderr": "oops\n"}),
        (
            "echo started; sleep 30 & exit 0",
            None,
            1.0,
            {"exit_code": 0, "stdout": "started\n", "stderr": ""},
        ),
        (
            "echo progress; echo stuck >&2; sleep 30",
            1,
            0.0,
            {"exit_code": None, "stdout": "progress\n", "stderr": "stuck\n", "timed_out": True},
        ),
        ("yes hud | head -c 500000", None, 1.0, {"exit_code": 0, "stdout": "hud\n" * 125_000}),
    ],
    ids=[
        "a-failing-command",
        "a-background-child-does-not-hold-the-grade",
        "an-overrun-keeps-its-output",
        "large-output-does-not-block",
    ],
)
async def test_the_bash_grader_scores_by_exit_code_and_reports_what_ran(
    command: str, limit: float | None, value: float, info: dict[str, Any]
) -> None:
    env = Environment("graders")

    @env.template()
    async def check():
        yield "go"
        yield await combine(BashGrader.grade(weight=1.0, command=command, timeout_seconds=limit))

    started = time.monotonic()
    (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    (subscore,) = run.grade.raw["subscores"]
    assert run.reward == value
    assert {key: subscore["info"][key] for key in info} == info
    assert time.monotonic() - started < 15


async def test_the_bash_grader_runs_in_the_directory_it_is_given(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("here")
    env = Environment("graders")

    @env.template()
    async def check():
        yield "go"
        yield await combine(
            BashGrader.grade(weight=1.0, command="cat marker.txt", cwd=str(tmp_path))
        )

    (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    assert (run.reward, run.grade.raw["subscores"][0]["info"]["stdout"]) == (1.0, "here")


VERDICTS = {
    "json": '{"criterion_status": "MET", "explanation": "plainly stated"}',
    "fenced": '```json\n{"criterion_status": "UNMET", "explanation": "no source"}\n```',
    "prose": "Verdict: UNMET, although the word MET appears here.",
    "prose-met": "The criterion is MET.",
}


@pytest.mark.parametrize(
    ("criteria", "verdicts", "reward", "children"),
    [
        (
            [["names Paris", 1.0], ["cites a source", 2.0], ["invents facts", -1.0]],
            {"names Paris": "json", "cites a source": "fenced", "invents facts": "prose"},
            1 / 3,
            [
                ("names Paris", 1.0, 1.0, "plainly stated"),
                ("cites a source", 2.0, 0.0, "no source"),
                ("invents facts", -1.0, 0.0, VERDICTS["prose"]),
            ],
        ),
        (
            [["invents facts", -1.0]],
            {"invents facts": "prose-met"},
            0.0,
            [("invents facts", -1.0, 1.0, "The criterion is MET.")],
        ),
        (
            [["invents facts", -1.0]],
            {"invents facts": "prose"},
            1.0,
            [("invents facts", -1.0, 0.0, VERDICTS["prose"])],
        ),
    ],
    ids=["json-fenced-and-prose-verdicts", "a-penalty-that-is-met", "a-penalty-that-is-not-met"],
)
async def test_the_judge_asks_once_per_criterion_and_weighs_each_verdict(
    criteria: list[list[Any]],
    verdicts: dict[str, str],
    reward: float,
    children: list[tuple[str, float, float, str]],
    models: Models,
    hud_env: HudEnv,
) -> None:
    hud_env.set(HUD_API_KEY="k")

    def verdict_for(request: ModelRequest) -> Turn:
        (criterion,) = [c for c in verdicts if c in json.dumps(request.body)]
        return Turn(text=VERDICTS[verdicts[criterion]])

    models.respond(verdict_for)
    env = Environment("graders")

    @env.template()
    async def check():
        answer = yield "Name the capital of France."
        judged = LLMJudgeGrader.grade(
            weight=1.0,
            answer=answer,
            criteria=[(requirement, weight) for requirement, weight in criteria],
            question="Name the capital of France.",
        )
        yield await combine(judged)

    (run,) = (await check().run(ScriptedAgent(ANSWER))).runs

    (judge,) = run.grade.raw["subscores"]
    assert run.reward == pytest.approx(reward)
    assert [
        (c["name"], c["weight"], c["value"], c["info"]["reason"]) for c in judge["children"]
    ] == children
    assert [request.credential for request in models.requests()] == ["k"] * len(criteria)
