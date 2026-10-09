"""Graders as task authors use them: text helpers, and graded templates in real rollouts."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot

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
    normalize,
    numeric_match,
)
from tests.harness import ScriptedAgent, Turn, say

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.eval import Run
    from tests.harness import HudEnv, ModelRequest, Models

# ─── text helpers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello World", "hello world"),
        ("answer: 42!", "answer 42"),
        ("The answer is a test", "answer is test"),
        ("  lots   of    space  ", "lots of space"),
        (42, "42"),
    ],
)
def test_normalize_lowercases_and_strips_punctuation_articles_and_spaces(
    text: Any, expected: str
) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        pytest.param(lambda: exact_match("The Answer!", "answer"), 1.0, id="exact-normalized"),
        pytest.param(lambda: exact_match("Germany", "france"), 0.0, id="exact-differs"),
        pytest.param(lambda: exact_match("Paris.", "Paris"), 1.0, id="exact-punctuation"),
        pytest.param(
            lambda: exact_match("The capital is Paris", "capital is Paris"), 1.0, id="exact-article"
        ),
        pytest.param(
            lambda: exact_match("Paris", "Paris", normalize_text=False), 1.0, id="exact-raw-same"
        ),
        pytest.param(
            lambda: exact_match("The Paris!", "Paris", normalize_text=False),
            0.0,
            id="exact-raw-differs",
        ),
        pytest.param(lambda: exact_match(42, "42"), 1.0, id="exact-non-string"),
        pytest.param(lambda: exact_match("", ""), 1.0, id="exact-empty"),
        pytest.param(lambda: contains("The capital is Paris", "paris"), 1.0, id="contains"),
        pytest.param(lambda: contains("The capital is Paris", "berlin"), 0.0, id="contains-not"),
        pytest.param(
            lambda: contains("Paris", "paris", case_sensitive=True), 0.0, id="contains-case"
        ),
        pytest.param(
            lambda: contains("Paris", "Paris", case_sensitive=True), 1.0, id="contains-case-same"
        ),
        pytest.param(
            lambda: contains_any("I like Toyota cars", ["toyota", "honda"]), 1.0, id="any-one"
        ),
        pytest.param(
            lambda: contains_any("I like BMW cars", ["toyota", "honda"]), 0.0, id="any-none"
        ),
        pytest.param(lambda: contains_any("anything", []), 0.0, id="any-empty"),
        pytest.param(
            lambda: contains_any("Toyota", ["toyota"], case_sensitive=True), 0.0, id="any-case"
        ),
        pytest.param(
            lambda: contains_all("Toyota and Honda", ["toyota", "honda"]), 1.0, id="all-present"
        ),
        pytest.param(
            lambda: contains_all("Toyota is Japanese", ["toyota", "honda"]), 0.0, id="all-missing"
        ),
        pytest.param(lambda: contains_all("anything", []), 1.0, id="all-empty"),
        pytest.param(lambda: numeric_match("The answer is 42", 42), 1.0, id="number-int"),
        pytest.param(lambda: numeric_match("Result: 3.14", 3.14), 1.0, id="number-float"),
        pytest.param(lambda: numeric_match("no numbers", 42), 0.0, id="number-none"),
        pytest.param(lambda: numeric_match("The answer is 41", 42), 0.0, id="number-wrong"),
        pytest.param(
            lambda: numeric_match("It is 41.5", 42, tolerance=1.0), 1.0, id="number-tolerance"
        ),
        pytest.param(
            lambda: numeric_match("It is 40", 42, tolerance=1.0), 0.0, id="number-out-of-tolerance"
        ),
        pytest.param(lambda: numeric_match("It is -5 degrees", -5), 1.0, id="number-negative"),
        pytest.param(lambda: numeric_match("3 items and 5 left", 3), 1.0, id="number-first"),
        pytest.param(lambda: f1_score("Paris", "Paris"), 1.0, id="f1-exact"),
        pytest.param(lambda: f1_score("The capital is Paris, France", "Paris"), 0.4, id="f1-part"),
        pytest.param(lambda: f1_score("Berlin", "Paris"), 0.0, id="f1-none"),
        pytest.param(lambda: f1_score("", "Paris"), 0.0, id="f1-empty-prediction"),
        pytest.param(lambda: f1_score("Paris", ""), 0.0, id="f1-empty-reference"),
        pytest.param(lambda: f1_score("New York City", "New York City"), 1.0, id="f1-words"),
        pytest.param(lambda: f1_score("I think 42 degrees", "42 degrees"), 2 / 3, id="f1-superset"),
        pytest.param(lambda: f1_score("42", "42 degrees celsius"), 0.5, id="f1-subset"),
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


def _graders_env(root: Path) -> Environment:
    env = Environment("graders")

    def task(grade: Callable[[str], Any]) -> None:
        @env.template(id=grade.__name__)
        async def template():
            answer = yield "Name the capital of France."
            yield await grade(answer)

    async def bash_pair(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(weight=0.5, command="echo pass"),
            BashGrader.grade(weight=0.5, command="echo oops >&2; false"),
        )

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

    async def weighted(answer: str) -> EvaluationResult:
        return await combine(
            SubScore(name="mentions", value=contains(answer, "paris"), weight=0.6),
            SubScore(name="numbers", value=numeric_match(answer, 7), weight=0.4),
            SubScore(name="penalty", value=1.0, weight=-0.2),
        )

    async def penalty_only_fails(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            SubScore(name="correct", value=0.0, weight=1.0),
            SubScore(name="penalty", value=1.0, weight=-0.2),
        )

    async def suffixed(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            SubScore(name="x-1", value=1.0, weight=0.3),
            SubScore(name="x", value=1.0, weight=0.3),
            SubScore(name="x", value=0.0, weight=0.4),
        )

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

    async def in_cwd(answer: str) -> EvaluationResult:
        del answer
        (root / "marker.txt").write_text("here")
        return await combine(
            BashGrader.grade(weight=1.0, command="cat marker.txt", cwd=str(root)),
        )

    async def detached_child(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(weight=1.0, command="echo started; sleep 30 & exit 0"),
        )

    async def overrun(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(
                weight=1.0, command="echo progress; echo stuck >&2; sleep 30", timeout_seconds=1
            ),
        )

    async def chatty(answer: str) -> EvaluationResult:
        del answer
        return await combine(
            BashGrader.grade(weight=1.0, command="yes hud | head -c 500000"),
        )

    async def judged(answer: str) -> EvaluationResult:
        return await combine(
            LLMJudgeGrader.grade(
                weight=1.0,
                answer=answer,
                criteria=["names Paris", ("cites a source", 2.0), ("invents facts", -1.0)],
                question="Name the capital of France.",
            )
        )

    async def judged_negative_only(answer: str) -> EvaluationResult:
        return await combine(
            LLMJudgeGrader.grade(weight=1.0, answer=answer, criteria=[("invents facts", -1.0)])
        )

    async def judged_without_criteria(answer: str) -> EvaluationResult:
        return await combine(LLMJudgeGrader.grade(weight=1.0, answer=answer, criteria=[]))

    for grade in (
        bash_pair,
        any_tree,
        weighted,
        penalty_only_fails,
        suffixed,
        custom,
        concurrent,
        in_cwd,
        detached_child,
        overrun,
        chatty,
        judged,
        judged_negative_only,
        judged_without_criteria,
    ):
        task(grade)
    return env


async def _grade(template: str, tmp_path: Path, answer: str = ANSWER) -> Run:
    root = tmp_path / "grader"
    root.mkdir(exist_ok=True)
    task = Task(env="graders", id=template)
    return await rollout(task, ScriptedAgent(answer), runtime=LocalRuntime(_graders_env(root)))


def _evaluated(run: Run) -> dict[str, Any]:
    """The evaluation the platform receives: the evaluate step's full grade."""
    step = run.trace.steps[-1]
    assert step.task_call is not None
    assert step.task_call.phase == "evaluate"
    result = step.task_call.result
    assert isinstance(result, dict)
    return result


@pytest.mark.parametrize(
    ("template", "reward", "summary"),
    [
        pytest.param(
            "bash_pair",
            0.5,
            snapshot(
                [
                    {"name": "BashGrader-1", "weight": 0.5, "value": 1.0, "children": None},
                    {"name": "BashGrader-2", "weight": 0.5, "value": 0.0, "children": None},
                ]
            ),
            id="bash-pair",
        ),
        pytest.param(
            "any_tree",
            0.75,
            snapshot(
                [
                    {
                        "name": "any",
                        "weight": 0.5,
                        "value": 1.0,
                        "children": [
                            {"name": "BashGrader-1", "weight": 0.5, "value": 0.0, "children": None},
                            {"name": "BashGrader-2", "weight": 0.5, "value": 1.0, "children": None},
                        ],
                    },
                    {
                        "name": "checks",
                        "weight": 0.25,
                        "value": 0.0,
                        "children": [
                            {"name": "lint", "weight": 1.0, "value": 1.0, "children": None},
                            {"name": "types", "weight": 1.0, "value": 0.0, "children": None},
                        ],
                    },
                    {"name": "format", "weight": 0.25, "value": 1.0, "children": None},
                ]
            ),
            id="any-all-tree",
        ),
        pytest.param(
            "weighted",
            0.4,
            snapshot(
                [
                    {"name": "mentions", "weight": 0.6, "value": 1.0, "children": None},
                    {"name": "numbers", "weight": 0.4, "value": 0.0, "children": None},
                    {"name": "penalty", "weight": -0.2, "value": 1.0, "children": None},
                ]
            ),
            id="penalty",
        ),
        pytest.param(
            "penalty_only_fails",
            -0.2,
            snapshot(
                [
                    {"name": "correct", "weight": 1.0, "value": 0.0, "children": None},
                    {"name": "penalty", "weight": -0.2, "value": 1.0, "children": None},
                ]
            ),
            id="negative-reward",
        ),
        pytest.param(
            "suffixed",
            0.6,
            snapshot(
                [
                    {"name": "x-1", "weight": 0.3, "value": 1.0, "children": None},
                    {"name": "x-2", "weight": 0.3, "value": 1.0, "children": None},
                    {"name": "x-3", "weight": 0.4, "value": 0.0, "children": None},
                ]
            ),
            id="name-collisions",
        ),
        pytest.param(
            "custom",
            0.875,
            snapshot(
                [
                    {"name": "measured-1", "weight": 0.25, "value": 0.75, "children": None},
                    {"name": "measured-2", "weight": 0.25, "value": 0.75, "children": None},
                    {"name": "renamed", "weight": 0.25, "value": 1.0, "children": None},
                    {"name": "legacy", "weight": 0.25, "value": 1.0, "children": None},
                ]
            ),
            id="custom-grader-shapes",
        ),
    ],
)
async def test_a_combined_grade_reaches_the_run_as_a_named_weighted_tree(
    tmp_path: Path, template: str, reward: float, summary: list[dict[str, Any]]
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = await _grade(template, tmp_path)

    assert run.reward == pytest.approx(reward)
    assert run.evaluation["subscores"] == summary


async def test_the_evaluate_step_keeps_each_graders_details(tmp_path: Path) -> None:
    run = await _grade("bash_pair", tmp_path)

    assert [subscore["info"] for subscore in _evaluated(run)["subscores"]] == snapshot(
        [
            {
                "exit_code": 0,
                "stdout": "pass\n",
                "stderr": "",
                "_parameters": {"command": "echo pass"},
            },
            {
                "exit_code": 1,
                "stdout": "",
                "stderr": "oops\n",
                "_parameters": {"command": "echo oops >&2; false"},
            },
        ]
    )


async def test_custom_graders_record_their_parameters_and_info(tmp_path: Path) -> None:
    run = await _grade("custom", tmp_path)

    assert {s["name"]: s["info"] for s in _evaluated(run)["subscores"]} == snapshot(
        {
            "measured-1": {
                "_parameters": {"kind": "float", "payload": "<object: not serializable>"}
            },
            "measured-2": {"source": "tuple", "_parameters": {"kind": "tuple"}},
            "renamed": {"reason": "did it", "_parameters": {"kind": "subscore"}},
            "legacy": {"from": "metadata"},
        }
    )


@pytest.mark.parametrize(
    ("template", "value", "info"),
    [
        pytest.param(
            "in_cwd",
            1.0,
            {"exit_code": 0, "stdout": "here", "stderr": ""},
            id="cwd",
        ),
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
    ],
)
async def test_bash_grader_scores_by_exit_code_and_reports_what_ran(
    tmp_path: Path, template: str, value: float, info: dict[str, Any]
) -> None:
    started = time.monotonic()

    run = await _grade(template, tmp_path)

    (subscore,) = _evaluated(run)["subscores"]
    assert run.reward == value
    assert {key: subscore["info"][key] for key in info} == info
    assert time.monotonic() - started < 15


async def test_bash_grader_reads_a_large_output_without_blocking(tmp_path: Path) -> None:
    run = await _grade("chatty", tmp_path)

    (subscore,) = _evaluated(run)["subscores"]
    assert run.reward == 1.0
    assert len(subscore["info"]["stdout"]) == 500_000


async def test_combined_graders_run_concurrently(tmp_path: Path) -> None:
    run = await _grade("concurrent", tmp_path)

    assert run.reward == 1.0


# ─── authoring mistakes ──────────────────────────────────────────────────────


NOT_SUBSCORES: list[Any] = [0.5]


def _mistakes_env() -> Environment:
    env = Environment("mistakes")
    cases: dict[str, Callable[[], Any]] = {
        "empty_combine": lambda: combine(),
        "no_positive_weight": lambda: combine(SubScore(name="p", value=1.0, weight=-1.0)),
        "not_a_subscore": lambda: combine(*NOT_SUBSCORES),
        "missing_command": lambda: combine(BashGrader.grade(weight=1.0)),
        "empty_any": lambda: _awaitable(lambda: combine_any(weight=1.0, subscores=[])),
        "empty_all": lambda: _awaitable(lambda: combine_all(weight=1.0, subscores=[])),
    }
    for name, grade in cases.items():

        def register(grade: Callable[[], Any] = grade, name: str = name) -> None:
            @env.template(id=name)
            async def template():
                yield "go"
                yield await grade()

        register()
    return env


async def _awaitable(build: Callable[[], SubScore]) -> EvaluationResult:
    return await combine(build())


@pytest.mark.parametrize(
    ("template", "error"),
    [
        ("empty_combine", "subscores must not be empty"),
        ("no_positive_weight", "subscores must include at least one positive weight"),
        ("not_a_subscore", "Expected SubScore or Awaitable[SubScore], got float"),
        ("missing_command", "BashGrader requires command"),
        ("empty_any", "subscores must not be empty"),
        ("empty_all", "subscores must not be empty"),
    ],
)
async def test_an_authoring_mistake_fails_the_grade_with_its_message(
    template: str, error: str
) -> None:
    task = Task(env="mistakes", id=template)

    run = await rollout(task, ScriptedAgent("x"), runtime=LocalRuntime(_mistakes_env()))

    assert run.trace.status == "error"
    assert run.trace.error == IsStr(regex=rf"\[grading\] .*: {_escaped(error)}")


def _escaped(text: str) -> str:
    return "".join(f"\\{char}" if char in "[]()." else char for char in text)


def _warning_env() -> Environment:
    env = Environment("warnings")

    @env.template()
    async def misweighted():
        yield "go"
        yield await combine(
            SubScore(name="a", value=1.0, weight=0.5),
            SubScore(name="b", value=1.0, weight=0.3),
            SubScore(name="c", value=0.0, weight=0.3),
        )

    @env.template()
    async def mismatched():
        yield "go"
        yield EvaluationResult(reward=1.0, subscores=[SubScore(name="a", value=0.5, weight=1.0)])

    return env


@pytest.mark.parametrize(
    ("template", "warning", "reward"),
    [
        ("misweighted", "grader weights sum to 1.1000, not 1.0", 0.8 / 1.1),
        ("mismatched", "Subscores don't match reward", 1.0),
    ],
)
async def test_inconsistent_weights_warn_the_author(
    template: str, warning: str, reward: float
) -> None:
    task = Task(env="warnings", id=template)

    with pytest.warns(UserWarning, match=warning):
        run = await rollout(task, ScriptedAgent("x"), runtime=LocalRuntime(_warning_env()))

    assert run.reward == pytest.approx(reward)


# ─── the LLM judge ───────────────────────────────────────────────────────────

VERDICTS = {
    "json": '{"criterion_status": "MET", "explanation": "plainly stated"}',
    "fenced": '```json\n{"criterion_status": "UNMET", "explanation": "no source"}\n```',
    "prose": "Verdict: UNMET, although the word MET appears here.",
    "prose-met": "The criterion is MET.",
}


def _judge(verdicts: dict[str, str]) -> Callable[[ModelRequest], Turn]:
    def reply(request: ModelRequest) -> Turn:
        prompt = request.body["messages"][1]["content"]
        criterion = prompt.split("<criterion>\n", 1)[1].split("\n", 1)[0]
        return say(VERDICTS[verdicts[criterion]])

    return reply


@pytest.mark.parametrize(
    ("template", "verdicts", "reward", "children"),
    [
        pytest.param(
            "judged",
            {"names Paris": "json", "cites a source": "fenced", "invents facts": "prose"},
            1 / 3,
            snapshot(
                [
                    {
                        "name": "names Paris",
                        "weight": 1.0,
                        "value": 1.0,
                        "children": None,
                        "info": {"reason": "plainly stated"},
                    },
                    {
                        "name": "cites a source",
                        "weight": 2.0,
                        "value": 0.0,
                        "children": None,
                        "info": {"reason": "no source"},
                    },
                    {
                        "name": "invents facts",
                        "weight": -1.0,
                        "value": 0.0,
                        "children": None,
                        "info": {"reason": "Verdict: UNMET, although the word MET appears here."},
                    },
                ]
            ),
            id="json-fenced-prose",
        ),
        pytest.param(
            "judged_negative_only",
            {"invents facts": "prose-met"},
            0.0,
            snapshot(
                [
                    {
                        "name": "invents facts",
                        "weight": -1.0,
                        "value": 1.0,
                        "children": None,
                        "info": {"reason": "The criterion is MET."},
                    }
                ]
            ),
            id="negative-only-met",
        ),
        pytest.param(
            "judged_negative_only",
            {"invents facts": "prose"},
            1.0,
            snapshot(
                [
                    {
                        "name": "invents facts",
                        "weight": -1.0,
                        "value": 0.0,
                        "children": None,
                        "info": {"reason": "Verdict: UNMET, although the word MET appears here."},
                    }
                ]
            ),
            id="negative-only-unmet",
        ),
    ],
)
async def test_the_judge_scores_each_criterion_by_weight(
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
    template: str,
    verdicts: dict[str, str],
    reward: float,
    children: list[dict[str, Any]],
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(_judge(verdicts))

    run = await _grade(template, tmp_path)

    (judge,) = _evaluated(run)["subscores"]
    assert run.reward == pytest.approx(reward)
    assert judge["info"]["model"] == "claude-haiku-4-5"
    assert judge["children"] == children


async def test_the_judge_sends_one_request_per_criterion(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(
        _judge({"names Paris": "json", "cites a source": "json", "invents facts": "prose"})
    )

    await _grade("judged", tmp_path)

    requests = sorted(models.requests(), key=lambda request: request.body["messages"][1]["content"])
    assert [request.headers["authorization"] for request in requests] == ["Bearer k"] * 3
    assert {request.model for request in requests} == {"claude-haiku-4-5"}
    assert {request.body["messages"][0]["content"][:40] for request in requests} == {
        "You evaluate a response against a single"
    }
    assert [request.body["messages"][1]["content"] for request in requests] == snapshot(
        [
            """\
<criterion_type>
negative
</criterion_type>

<criterion>
invents facts
</criterion>

<query>
Name the capital of France.
</query>

<response>
Paris is the capital.
</response>\
""",
            """\
<criterion_type>
positive
</criterion_type>

<criterion>
cites a source
</criterion>

<query>
Name the capital of France.
</query>

<response>
Paris is the capital.
</response>\
""",
            """\
<criterion_type>
positive
</criterion_type>

<criterion>
names Paris
</criterion>

<query>
Name the capital of France.
</query>

<response>
Paris is the capital.
</response>\
""",
        ]
    )


async def test_the_judge_without_a_question_omits_the_query(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(_judge({"invents facts": "prose"}))

    await _grade("judged_negative_only", tmp_path)

    (request,) = models.requests()
    assert "<query>" not in request.body["messages"][1]["content"]


async def test_the_judge_without_criteria_scores_zero_without_calling_a_model(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")

    run = await _grade("judged_without_criteria", tmp_path)

    (judge,) = _evaluated(run)["subscores"]
    assert run.reward == 0.0
    assert judge["info"]["error"] == "no criteria provided"
    assert models.requests() == []


async def test_the_judge_without_a_hud_api_key_fails_the_grade(tmp_path: Path) -> None:
    run = await _grade("judged", tmp_path)

    assert run.trace.status == "error"
    assert run.trace.error == IsStr(regex=r".*HUD_API_KEY is required for HUD gateway clients")
