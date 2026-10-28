"""Graders as task authors use them: a template grades the agent's answer with ``combine``,
the bash grader, the LLM judge or a grader of its own, and the run's reward and
evaluation show the result."""

from __future__ import annotations

import asyncio
import json
import threading
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
)
from tests.conftest import eventually
from tests.fixtures.agents import ScriptedAgent
from tests.fixtures.models import Turn
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tests.conftest import HudEnv
    from tests.fixtures.models import ModelRequest, Models
    from tests.fixtures.services import FakeServices, Request

ANSWER = "Paris is the capital."


def leaf(name: str, weight: float, value: float) -> dict[str, Any]:
    return {"name": name, "weight": weight, "value": value, "children": None}


@pytest.mark.parametrize("cleanup_raises", [False, True])
async def test_a_failed_grader_stops_its_siblings_before_the_environment_shuts_down(
    cleanup_raises: bool,
) -> None:
    env = Environment("graders")
    started = asyncio.Event()
    release = asyncio.Event()
    events = []

    class Waits(Grader):
        @classmethod
        async def compute_score(cls, **kwargs: Any) -> SubScore:
            started.set()
            try:
                await release.wait()
                return SubScore(name="waiting", value=1.0)
            finally:
                await asyncio.sleep(0)
                events.append("grader stopped")
                if cleanup_raises:
                    raise RuntimeError("cleanup failed")

    class Fails(Grader):
        @classmethod
        async def compute_score(cls, **kwargs: Any) -> SubScore:
            await started.wait()
            raise ValueError("grader failed")

    @env.shutdown
    async def down() -> None:
        events.append("environment stopped")

    @env.template()
    async def check():
        yield "go"
        yield await combine(Waits.grade(weight=0.5), Fails.grade(weight=0.5))

    try:
        job = await check().run(ScriptedAgent(ANSWER))
        (run,) = job.runs
        assert (run.trace.status, job.errors) == ("error", [run])
        assert "grader failed" in (run.trace.error or "")
        assert events == ["grader stopped", "environment stopped"]
    finally:
        release.set()
        await eventually(lambda: "grader stopped" in events)


@pytest.mark.parametrize("stop", ["caller", "caller-twice", "child"])
async def test_cancelling_parallel_grading_waits_for_a_prestarted_graders_cleanup(
    stop: str,
) -> None:
    started = asyncio.Event()
    cleaning_up = asyncio.Event()
    release = asyncio.Event()
    future = asyncio.get_running_loop().create_future()

    async def wait() -> SubScore:
        started.set()
        try:
            return await future
        finally:
            cleaning_up.set()
            await release.wait()

    grader = asyncio.create_task(wait())

    async def quick() -> SubScore:
        await started.wait()
        if stop == "child":
            raise asyncio.CancelledError
        return await asyncio.get_running_loop().create_future()

    combined = asyncio.create_task(combine(quick(), grader))
    try:
        await started.wait()
        if stop != "child":
            combined.cancel()
        await cleaning_up.wait()
        if stop == "caller-twice":
            combined.cancel()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(combined), timeout=0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await combined
        assert grader.cancelled() and future.cancelled()
    finally:
        release.set()
        combined.cancel()
        await asyncio.gather(combined, return_exceptions=True)


async def test_a_grader_coroutine_used_twice_runs_once_and_contributes_twice() -> None:
    completed = []

    async def score() -> SubScore:
        await asyncio.sleep(0)
        completed.append("scored")
        return SubScore(name="shared", value=0.75, weight=0.5)

    grade = score()
    result = await combine(grade, grade)

    assert completed == ["scored"]
    assert result.reward == 0.75
    assert result.subscores is not None
    assert [(s.value, s.weight) for s in result.subscores] == [(0.75, 0.5)] * 2


async def test_a_failed_judgment_closes_the_other_criterion_requests(
    services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    started = threading.Event()
    stopped = threading.Event()
    release = threading.Event()

    async def unfinished_response() -> AsyncIterator[bytes]:
        try:
            yield b'{"id":'
            started.set()
            await eventually(release.is_set)
            yield b'"unused"}'
        finally:
            stopped.set()

    async def respond(request: Request) -> Reply:
        if "waiting criterion" in json.dumps(request.json):
            return Reply(stream=unfinished_response(), content_type="application/json")
        await eventually(started.is_set)
        return Reply(json={"choices": [{"message": {"content": "no verdict"}}]})

    services.route("gateway", "POST", "/chat/completions", handler=respond)

    try:
        with pytest.raises(ValueError, match="verdict"):
            await LLMJudgeGrader.grade(
                weight=1.0,
                answer=ANSWER,
                criteria=["waiting criterion", "failing criterion"],
            )
        await eventually(stopped.is_set, within=2.0)
    finally:
        release.set()
        await eventually(stopped.is_set)


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


VERDICTS = {
    "json": '{"criterion_status": "MET", "explanation": "plainly stated"}',
    "fenced": '```json\n{"criterion_status": "UNMET", "explanation": "no source"}\n```',
    "prose": "Verdict: UNMET, although the word MET appears here.",
    "prose-met": "The criterion is MET.",
    "prose-not-met": "Criterion status: NOT MET",
    "no-verdict": "I'm sorry, something went wrong.",
    "unknown-status": '{"criterion_status": "MAYBE", "explanation": "unsure"}',
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
        (
            [["names Paris", 1.0]],
            {"names Paris": "prose-not-met"},
            0.0,
            [("names Paris", 1.0, 0.0, VERDICTS["prose-not-met"])],
        ),
        ([["names Paris", 1.0]], {"names Paris": "no-verdict"}, 0.0, None),
        ([["names Paris", 1.0]], {"names Paris": "unknown-status"}, 0.0, None),
    ],
    ids=[
        "json-fenced-and-prose-verdicts",
        "a-penalty-that-is-met",
        "a-penalty-that-is-not-met",
        "not-met-in-prose",
        "a-reply-without-a-verdict-is-a-grading-error",
        "an-unknown-status-is-a-grading-error",
    ],
)
async def test_the_judge_asks_once_per_criterion_and_weighs_each_verdict(
    criteria: list[list[Any]],
    verdicts: dict[str, str],
    reward: float,
    children: list[tuple[str, float, float, str]] | None,
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

    job = await check().run(ScriptedAgent(ANSWER))

    (run,) = job.runs
    assert (run.reward, run in job.errors) == (pytest.approx(reward), children is None)
    if children is not None:
        (judge,) = run.grade.raw["subscores"]
        assert [
            (c["name"], c["weight"], c["value"], c["info"]["reason"]) for c in judge["children"]
        ] == children
    assert [request.credential for request in models.requests()] == ["k"] * len(criteria)
