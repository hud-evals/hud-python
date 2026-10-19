"""Graders as task authors use them: text helpers, and graded templates in real rollouts.

Each grader family is one table: text comparisons, the bash grader, the LLM judge,
and combined grades. Authoring mistakes are a table of the refusals that fail the
grade.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import pytest

from hud import Environment
from hud.eval import LocalRuntime, Task, rollout
from hud.graders import (
    BashGrader,
    EvaluationResult,
    Grader,
    LLMJudgeGrader,
    SubScore,
    combine,
    combine_all,
    combine_any,
    contains,
    contains_all,
    contains_any,
    exact_match,
    f1_score,
    numeric_match,
)
from tests.support import ScriptedAgent, Turn, say

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path
    from typing import TypeAlias

    Grade: TypeAlias = Callable[[str], Awaitable[EvaluationResult]]

    from hud.eval import Run
    from tests.support import HudEnv, ModelRequest, Models

# ─── text comparisons ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        pytest.param(lambda: exact_match("The Answer!", "answer"), 1.0, id="exact-normalized"),
        pytest.param(
            lambda: exact_match("The Paris!", "Paris", normalize_text=False),
            0.0,
            id="exact-raw-differs",
        ),
        pytest.param(lambda: contains("The capital is Paris", "paris"), 1.0, id="contains"),
        pytest.param(
            lambda: contains("Paris", "paris", case_sensitive=True), 0.0, id="contains-case"
        ),
        pytest.param(
            lambda: contains_any("I like Toyota cars", ["toyota", "honda"]), 1.0, id="any-one"
        ),
        pytest.param(lambda: contains_any("anything", []), 0.0, id="any-empty"),
        pytest.param(
            lambda: contains_all("Toyota and Honda", ["toyota", "honda"]), 1.0, id="all-present"
        ),
        pytest.param(lambda: contains_all("anything", []), 1.0, id="all-empty"),
        pytest.param(lambda: numeric_match("The answer is 42", 42), 1.0, id="number-int"),
        pytest.param(
            lambda: numeric_match("It is 41.5", 42, tolerance=1.0), 1.0, id="number-tolerance"
        ),
        pytest.param(lambda: numeric_match("It is -5 degrees", -5), 1.0, id="number-negative"),
        pytest.param(lambda: numeric_match("3 items and 5 left", 3), 1.0, id="number-first"),
        pytest.param(lambda: f1_score("The capital is Paris, France", "Paris"), 0.4, id="f1-part"),
        pytest.param(lambda: f1_score("", "Paris"), 0.0, id="f1-empty-prediction"),
        pytest.param(lambda: f1_score("The PARIS!", "paris"), 1.0, id="f1-normalized"),
    ],
)
def test_text_comparisons_score_between_zero_and_one(
    score: Callable[[], float], expected: float
) -> None:
    assert score() == pytest.approx(expected)


# ─── graded templates ────────────────────────────────────────────────────────

ANSWER = "Paris is the capital."


class Measured(Grader):
    """A custom grader whose ``kind`` picks which result shape it returns."""

    name = "measured"

    @classmethod
    async def compute_score(cls, kind: str = "float", **kwargs: Any) -> Any:
        del kwargs
        if kind == "tuple":
            return 0.75, {"source": "tuple"}
        if kind == "subscore":
            return SubScore(name="ignored", value=1.0, info={"reason": "did it"})
        return 0.75


def graders_env(root: Path) -> Environment:
    env = Environment("graders")

    def graded(name: str) -> Callable[[Grade], Grade]:
        def register(grade: Grade) -> Grade:
            @env.template(id=name)
            async def template():
                answer = yield "Name the capital of France."
                yield await grade(answer)

            return grade

        return register

    # Combined grades.

    @graded("bash_pair")
    async def bash_pair(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(weight=0.5, command="echo pass"),
            BashGrader.grade(weight=0.5, command="echo oops >&2; false"),
        )

    @graded("any_tree")
    async def any_tree(answer: str) -> EvaluationResult:
        either = combine_any(
            weight=0.5,
            subscores=[
                await BashGrader.grade(weight=0.5, command="false"),
                await BashGrader.grade(weight=0.5, command="true"),
            ],
        )
        both = combine_all(
            weight=0.25,
            subscores=[SubScore(name="lint", value=1.0), SubScore(name="types", value=0.0)],
            name="checks",
        )
        return await combine(
            either, both, SubScore(name="format", value=exact_match(answer, ANSWER), weight=0.25)
        )

    @graded("weighted")
    async def weighted(answer: str) -> EvaluationResult:
        return await combine(
            SubScore(name="mentions", value=contains(answer, "paris"), weight=0.6),
            SubScore(name="numbers", value=numeric_match(answer, 7), weight=0.4),
            SubScore(name="penalty", value=1.0, weight=-0.2),
        )

    @graded("penalty_only_fails")
    async def penalty_only_fails(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            SubScore(name="correct", value=0.0, weight=1.0),
            SubScore(name="penalty", value=1.0, weight=-0.2),
        )

    @graded("suffixed")
    async def suffixed(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            SubScore(name="x-1", value=1.0, weight=0.3),
            SubScore(name="x", value=1.0, weight=0.3),
            SubScore(name="x", value=0.0, weight=0.4),
        )

    @graded("custom")
    async def custom(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            Measured.grade(weight=0.25, kind="float", payload=object()),
            Measured.grade(weight=0.25, kind="tuple"),
            Measured.grade(weight=0.25, kind="subscore", name="renamed"),
            SubScore.model_validate(
                {"name": "legacy", "value": 1.0, "weight": 0.25, "metadata": {"from": "metadata"}}
            ),
        )

    @graded("concurrent")
    async def concurrent(answer: str) -> EvaluationResult:
        del answer
        # Each command waits for the file the other writes, so they pass only in parallel.
        wait = "for _ in $(seq 50); do [ -e {mine} ] && exit 0; sleep 0.1; done; exit 1"
        return await combine(
            BashGrader.grade(
                weight=0.5, command=f"touch {root}/a; " + wait.format(mine=f"{root}/b")
            ),
            BashGrader.grade(
                weight=0.5, command=f"touch {root}/b; " + wait.format(mine=f"{root}/a")
            ),
        )

    @graded("misweighted")
    async def misweighted(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            SubScore(name="a", value=1.0, weight=0.5),
            SubScore(name="b", value=1.0, weight=0.3),
            SubScore(name="c", value=0.0, weight=0.3),
        )

    @graded("mismatched")
    async def mismatched(answer: str) -> EvaluationResult:
        del answer
        return EvaluationResult(reward=1.0, subscores=[SubScore(name="a", value=0.5, weight=1.0)])

    # The bash grader.

    @graded("failing_command")
    async def failing_command(answer: str) -> EvaluationResult:
        del answer
        return await combine(BashGrader.grade(weight=1.0, command="echo oops >&2; false"))

    @graded("in_cwd")
    async def in_cwd(answer: str) -> EvaluationResult:
        del answer
        (root / "marker.txt").write_text("here")
        return await combine(
            BashGrader.grade(weight=1.0, command="cat marker.txt", cwd=str(root)),
        )

    @graded("detached_child")
    async def detached_child(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(weight=1.0, command="echo started; sleep 30 & exit 0"),
        )

    @graded("overrun")
    async def overrun(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(
                weight=1.0, command="echo progress; echo stuck >&2; sleep 30", timeout_seconds=1
            ),
        )

    @graded("chatty")
    async def chatty(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(weight=1.0, command="yes hud | head -c 500000"),
        )

    # The LLM judge.

    @graded("judged")
    async def judged(answer: str) -> EvaluationResult:
        return await combine(
            LLMJudgeGrader.grade(
                weight=1.0,
                answer=answer,
                criteria=["names Paris", ("cites a source", 2.0), ("invents facts", -1.0)],
                question="Name the capital of France.",
            )
        )

    @graded("judged_negative_only")
    async def judged_negative_only(answer: str) -> EvaluationResult:
        return await combine(
            LLMJudgeGrader.grade(weight=1.0, answer=answer, criteria=[("invents facts", -1.0)])
        )

    @graded("judged_without_criteria")
    async def judged_without_criteria(answer: str) -> EvaluationResult:
        return await combine(LLMJudgeGrader.grade(weight=1.0, answer=answer, criteria=[]))

    # Authoring mistakes.

    @graded("no_positive_weight")
    async def no_positive_weight(answer: str) -> EvaluationResult:
        del answer
        return await combine(SubScore(name="p", value=1.0, weight=-1.0))

    @graded("not_a_subscore")
    async def not_a_subscore(answer: str) -> EvaluationResult:
        del answer
        mixed: list[Any] = [SubScore(name="a", value=1.0), 0.5]
        return await combine(*mixed)

    @graded("empty_all")
    async def empty_all(answer: str) -> EvaluationResult:
        del answer
        return await combine(combine_all(weight=1.0, subscores=[]))

    return env


async def grade(template: str, tmp_path: Path) -> Run:
    root = tmp_path / "grader"
    root.mkdir(exist_ok=True)
    task = Task(env="graders", id=template)
    return await rollout(task, ScriptedAgent(ANSWER), runtime=LocalRuntime(graders_env(root)))


def evaluated(run: Run) -> dict[str, Any]:
    """The evaluation the platform receives: the evaluate step's full grade."""
    step = run.trace.steps[-1]
    assert step.task_call is not None
    assert step.task_call.phase == "evaluate"
    result = step.task_call.result
    assert isinstance(result, dict)
    return result


# ─── combined grades ─────────────────────────────────────────────────────────


MISWEIGHTED = 0.5 + 0.3 + 0.3


def leaf(name: str, weight: float, value: float) -> dict[str, Any]:
    return {"name": name, "weight": weight, "value": value, "children": None}


@pytest.mark.parametrize(
    ("template", "reward", "summary", "warns", "info"),
    [
        pytest.param(
            "bash_pair",
            0.5,
            [leaf("BashGrader-1", 0.5, 1.0), leaf("BashGrader-2", 0.5, 0.0)],
            False,
            None,
            id="bash-pair",
        ),
        pytest.param(
            "any_tree",
            0.75,
            [
                {
                    "name": "any",
                    "weight": 0.5,
                    "value": 1.0,
                    "children": [leaf("BashGrader-1", 0.5, 0.0), leaf("BashGrader-2", 0.5, 1.0)],
                },
                {
                    "name": "checks",
                    "weight": 0.25,
                    "value": 0.0,
                    "children": [leaf("lint", 1.0, 1.0), leaf("types", 1.0, 0.0)],
                },
                leaf("format", 0.25, 1.0),
            ],
            False,
            None,
            id="any-all-tree",
        ),
        pytest.param(
            "weighted",
            0.4,
            [leaf("mentions", 0.6, 1.0), leaf("numbers", 0.4, 0.0), leaf("penalty", -0.2, 1.0)],
            False,
            None,
            id="penalty",
        ),
        pytest.param(
            "penalty_only_fails",
            -0.2,
            [leaf("correct", 1.0, 0.0), leaf("penalty", -0.2, 1.0)],
            False,
            None,
            id="negative-reward",
        ),
        pytest.param(
            "suffixed",
            0.6,
            [leaf("x-1", 0.3, 1.0), leaf("x-2", 0.3, 1.0), leaf("x-3", 0.4, 0.0)],
            False,
            None,
            id="name-collisions",
        ),
        pytest.param(
            "custom",
            0.875,
            [
                leaf("measured-1", 0.25, 0.75),
                leaf("measured-2", 0.25, 0.75),
                leaf("renamed", 0.25, 1.0),
                leaf("legacy", 0.25, 1.0),
            ],
            False,
            {
                "measured-1": {
                    "_parameters": {"kind": "float", "payload": "<object: not serializable>"}
                },
                "measured-2": {"source": "tuple", "_parameters": {"kind": "tuple"}},
                "renamed": {"reason": "did it", "_parameters": {"kind": "subscore"}},
                "legacy": {"from": "metadata"},
            },
            id="custom-grader-shapes",
        ),
        pytest.param(
            "concurrent",
            1.0,
            [leaf("BashGrader-1", 0.5, 1.0), leaf("BashGrader-2", 0.5, 1.0)],
            False,
            None,
            id="graders-run-concurrently",
        ),
        pytest.param(
            "misweighted",
            0.8 / MISWEIGHTED,
            [
                leaf("a", 0.5 / MISWEIGHTED, 1.0),
                leaf("b", 0.3 / MISWEIGHTED, 1.0),
                leaf("c", 0.3 / MISWEIGHTED, 0.0),
            ],
            True,
            None,
            id="weights-not-summing-to-one",
        ),
        pytest.param(
            "mismatched",
            1.0,
            [leaf("a", 1.0, 0.5)],
            True,
            None,
            id="subscores-not-matching-the-reward",
        ),
    ],
)
async def test_a_combined_grade_reaches_the_run_as_a_named_weighted_tree(
    tmp_path: Path,
    template: str,
    reward: float,
    summary: list[dict[str, Any]],
    warns: bool,
    info: dict[str, dict[str, Any]] | None,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = await grade(template, tmp_path)

    assert run.reward == pytest.approx(reward)
    assert run.evaluation["subscores"] == summary
    assert any(issubclass(warning.category, UserWarning) for warning in caught) is warns
    if info is not None:
        assert {s["name"]: s["info"] for s in evaluated(run)["subscores"]} == info


# ─── the bash grader ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("template", "value", "info"),
    [
        pytest.param(
            "failing_command",
            0.0,
            {"exit_code": 1, "stdout": "", "stderr": "oops\n"},
            id="nonzero-exit",
        ),
        pytest.param("in_cwd", 1.0, {"exit_code": 0, "stdout": "here", "stderr": ""}, id="cwd"),
        pytest.param(
            "detached_child",
            1.0,
            {"exit_code": 0, "stdout": "started\n", "stderr": ""},
            id="background-child-does-not-hold-the-grade",
        ),
        pytest.param(
            "overrun",
            0.0,
            {
                "exit_code": None,
                "stdout": "progress\n",
                "stderr": "stuck\n",
                "timed_out": True,
                "timeout": 1,
            },
            id="overrun-keeps-its-output",
        ),
        pytest.param(
            "chatty",
            1.0,
            {"exit_code": 0, "stdout": "hud\n" * 125_000},
            id="large-output-does-not-block",
        ),
    ],
)
async def test_bash_grader_scores_by_exit_code_and_reports_what_ran(
    tmp_path: Path, template: str, value: float, info: dict[str, Any]
) -> None:
    started = time.monotonic()

    run = await grade(template, tmp_path)

    (subscore,) = evaluated(run)["subscores"]
    assert run.reward == value
    assert {key: subscore["info"][key] for key in info} == info
    assert time.monotonic() - started < 15


# ─── the LLM judge ───────────────────────────────────────────────────────────

VERDICTS = {
    "json": '{"criterion_status": "MET", "explanation": "plainly stated"}',
    "fenced": '```json\n{"criterion_status": "UNMET", "explanation": "no source"}\n```',
    "prose": "Verdict: UNMET, although the word MET appears here.",
    "prose-met": "The criterion is MET.",
}


def section(prompt: str, tag: str) -> str:
    """The text between ``<tag>`` and ``</tag>`` in a judge prompt."""
    return prompt.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


def criterion(name: str, weight: float, value: float, reason: str) -> dict[str, Any]:
    return {**leaf(name, weight, value), "info": {"reason": reason}}


@pytest.mark.parametrize(
    ("template", "api_key", "verdicts", "reward", "children"),
    [
        pytest.param(
            "judged",
            "k",
            {"names Paris": "json", "cites a source": "fenced", "invents facts": "prose"},
            1 / 3,
            [
                criterion("names Paris", 1.0, 1.0, "plainly stated"),
                criterion("cites a source", 2.0, 0.0, "no source"),
                criterion("invents facts", -1.0, 0.0, VERDICTS["prose"]),
            ],
            id="json-fenced-prose",
        ),
        pytest.param(
            "judged_negative_only",
            "k",
            {"invents facts": "prose-met"},
            0.0,
            [criterion("invents facts", -1.0, 1.0, "The criterion is MET.")],
            id="negative-only-met",
        ),
        pytest.param(
            "judged_negative_only",
            "k",
            {"invents facts": "prose"},
            1.0,
            [criterion("invents facts", -1.0, 0.0, VERDICTS["prose"])],
            id="negative-only-unmet",
        ),
        pytest.param(
            "judged_without_criteria", "k", {}, 0.0, None, id="no-criteria-calls-no-model"
        ),
        pytest.param(
            "judged",
            None,
            {},
            None,
            None,
            id="no-hud-api-key-fails-the-grade",
        ),
    ],
)
async def test_the_judge_asks_once_per_criterion_and_scores_each_by_weight(
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
    template: str,
    api_key: str | None,
    verdicts: dict[str, str],
    reward: float | None,
    children: list[dict[str, Any]] | None,
) -> None:
    if api_key is not None:
        hud_env.set(HUD_API_KEY=api_key)

    def reply(request: ModelRequest) -> Turn:
        return say(VERDICTS[verdicts[section(request.body["messages"][1]["content"], "criterion")]])

    models.respond(reply)

    run = await grade(template, tmp_path)

    requests = models.requests()
    assert [request.headers["authorization"] for request in requests] == [
        f"Bearer {api_key}"
    ] * len(verdicts)
    assert sorted(
        section(request.body["messages"][1]["content"], "criterion") for request in requests
    ) == sorted(verdicts)
    if reward is None:
        assert run.trace.status == "error"
        assert "HUD_API_KEY is required for HUD gateway clients" in str(run.trace.error)
        return
    (judge,) = evaluated(run)["subscores"]
    assert run.reward == pytest.approx(reward)
    assert judge["children"] == children


# ─── authoring mistakes ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("template", "refusal"),
    [
        ("no_positive_weight", "subscores must include at least one positive weight"),
        ("not_a_subscore", "Expected SubScore or Awaitable[SubScore], got float"),
        ("empty_all", "subscores must not be empty"),
    ],
)
async def test_an_authoring_mistake_fails_the_grade_with_its_refusal(
    template: str, refusal: str, tmp_path: Path
) -> None:
    run = await grade(template, tmp_path)

    assert run.trace.status == "error"
    error = str(run.trace.error)
    assert error.startswith("[grading]")
    assert error.endswith(refusal)
