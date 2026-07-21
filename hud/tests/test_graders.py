"""Tests for hud.graders: comparison helpers, combine, graders."""

from __future__ import annotations

import json
import os
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

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


class TestResultShapes:
    def test_subscore_score_aliases_value(self) -> None:
        s = SubScore(name="acc", value=0.75, weight=1.0)
        assert s.score == 0.75

    def test_evaluation_result_from_float(self) -> None:
        r = EvaluationResult.from_float(0.25)
        assert r.reward == 0.25
        assert r.done is True

    def test_evaluation_result_warns_when_subscores_disagree_with_reward(self) -> None:
        with pytest.warns(UserWarning):
            EvaluationResult(reward=1.0, subscores=[SubScore(name="a", value=0.5, weight=1.0)])

    def test_subscore_omits_metadata_from_standard_serialization(self) -> None:
        node = SubScore(
            name="any",
            value=1.0,
            children=[SubScore(name="tests", value=1.0, metadata={"exit_code": 0})],
        )
        dumped = node.model_dump(mode="json")
        assert dumped["children"][0]["name"] == "tests"
        assert "metadata" not in dumped
        assert "metadata" not in dumped["children"][0]

    def test_evaluation_span_serializes_recursive_metadata(self) -> None:
        result = EvaluationResult(
            reward=1.0,
            subscores=[
                SubScore(
                    name="judge",
                    value=1.0,
                    children=[
                        SubScore(
                            name="Mentions Paris",
                            value=1.0,
                            metadata={"reason": "says Paris"},
                        )
                    ],
                    metadata={"model": "judge-model"},
                )
            ],
        )
        dumped = result.model_dump_for_span()
        assert dumped["subscores"][0]["metadata"] == {"model": "judge-model"}
        assert dumped["subscores"][0]["children"][0]["metadata"] == {"reason": "says Paris"}


class TestNormalize:
    def test_lowercases(self) -> None:
        assert normalize("Hello World") == "hello world"

    def test_strips_punctuation(self) -> None:
        assert normalize("answer: 42!") == "answer 42"

    def test_strips_articles(self) -> None:
        assert normalize("The answer is a test") == "answer is test"

    def test_collapses_whitespace(self) -> None:
        assert normalize("  lots   of    space  ") == "lots of space"

    def test_non_string_input(self) -> None:
        assert normalize(42) == "42"


class TestExactMatch:
    def test_match_normalized(self) -> None:
        assert exact_match("The Answer!", "answer") == 1.0

    def test_no_match(self) -> None:
        assert exact_match("Germany", "france") == 0.0

    def test_punctuation_stripped(self) -> None:
        assert exact_match("Paris.", "Paris") == 1.0

    def test_articles_stripped(self) -> None:
        assert exact_match("The capital is Paris", "capital is Paris") == 1.0

    def test_no_normalize(self) -> None:
        assert exact_match("Paris", "Paris", normalize_text=False) == 1.0
        assert exact_match("The Paris!", "Paris", normalize_text=False) == 0.0

    def test_non_string_answer(self) -> None:
        assert exact_match(42, "42") == 1.0

    def test_empty_strings(self) -> None:
        assert exact_match("", "") == 1.0

    def test_different_content(self) -> None:
        assert exact_match("hello", "world") == 0.0


class TestContains:
    def test_contains_substring(self) -> None:
        assert contains("The capital of France is Paris", "paris") == 1.0

    def test_not_contains(self) -> None:
        assert contains("The capital of France is Paris", "berlin") == 0.0

    def test_case_sensitive(self) -> None:
        assert contains("Paris", "paris", case_sensitive=True) == 0.0
        assert contains("Paris", "Paris", case_sensitive=True) == 1.0


class TestContainsAny:
    def test_matches_one(self) -> None:
        assert contains_any("I like Toyota cars", ["toyota", "honda", "nissan"]) == 1.0

    def test_matches_none(self) -> None:
        assert contains_any("I like BMW cars", ["toyota", "honda", "nissan"]) == 0.0

    def test_empty_list(self) -> None:
        assert contains_any("anything", []) == 0.0

    def test_case_sensitive(self) -> None:
        assert contains_any("Toyota", ["toyota"], case_sensitive=True) == 0.0


class TestContainsAll:
    def test_all_present(self) -> None:
        assert contains_all("Toyota and Honda are Japanese", ["toyota", "honda"]) == 1.0

    def test_some_missing(self) -> None:
        assert contains_all("Toyota is Japanese", ["toyota", "honda"]) == 0.0

    def test_empty_list(self) -> None:
        assert contains_all("anything", []) == 1.0


class TestNumericMatch:
    def test_exact_integer(self) -> None:
        assert numeric_match("The answer is 42", 42) == 1.0

    def test_exact_float(self) -> None:
        assert numeric_match("Result: 3.14", 3.14) == 1.0

    def test_no_number(self) -> None:
        assert numeric_match("no numbers", 42) == 0.0

    def test_wrong_number(self) -> None:
        assert numeric_match("The answer is 41", 42) == 0.0

    def test_tolerance(self) -> None:
        assert numeric_match("The answer is 41.5", 42, tolerance=1.0) == 1.0
        assert numeric_match("The answer is 40", 42, tolerance=1.0) == 0.0

    def test_negative_number(self) -> None:
        assert numeric_match("Temperature is -5 degrees", -5) == 1.0

    def test_extracts_first_number(self) -> None:
        assert numeric_match("There are 3 items and 5 left", 3) == 1.0


class TestF1Score:
    def test_exact_match(self) -> None:
        assert f1_score("Paris", "Paris") == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        # pred=["capital", "is", "paris", "france"], ref=["paris"]
        # common=1, precision=1/4, recall=1/1, f1=2*(0.25*1)/(0.25+1)=0.4
        assert f1_score("The capital is Paris, France", "Paris") == pytest.approx(0.4)

    def test_no_overlap(self) -> None:
        assert f1_score("Berlin", "Paris") == pytest.approx(0.0)

    def test_empty_prediction(self) -> None:
        assert f1_score("", "Paris") == pytest.approx(0.0)

    def test_empty_reference(self) -> None:
        assert f1_score("Paris", "") == pytest.approx(0.0)

    def test_multi_word_match(self) -> None:
        assert f1_score("New York City", "New York City") == pytest.approx(1.0)

    def test_superset_answer(self) -> None:
        # pred has all ref tokens plus extras -> recall=1.0, precision<1.0
        result = f1_score("I believe the answer is 42 degrees", "42 degrees")
        assert 0.0 < result < 1.0

    def test_subset_answer(self) -> None:
        # pred has fewer tokens than ref -> precision=1.0, recall<1.0
        result = f1_score("42", "42 degrees celsius")
        assert 0.0 < result < 1.0

    def test_normalization_applied(self) -> None:
        assert f1_score("The PARIS!", "paris") == pytest.approx(1.0)


class TestCombineParallelism:
    async def test_combine_sync_subscores(self) -> None:
        result = await combine(
            SubScore(name="a", value=1.0, weight=0.5),
            SubScore(name="b", value=0.0, weight=0.5),
        )
        assert result.reward == pytest.approx(0.5)

    async def test_combine_with_awaitables(self) -> None:
        import asyncio

        order: list[str] = []

        async def slow_check_a() -> SubScore:
            order.append("a_start")
            await asyncio.sleep(0.05)
            order.append("a_end")
            return SubScore(name="a", value=1.0, weight=0.5)

        async def slow_check_b() -> SubScore:
            order.append("b_start")
            await asyncio.sleep(0.05)
            order.append("b_end")
            return SubScore(name="b", value=0.0, weight=0.5)

        result = await combine(slow_check_a(), slow_check_b())
        assert result.reward == pytest.approx(0.5)
        assert order.index("b_start") < order.index("a_end")

    async def test_combine_mixed(self) -> None:
        import asyncio

        async def async_score() -> SubScore:
            await asyncio.sleep(0.01)
            return SubScore(name="async", value=1.0, weight=0.5)

        result = await combine(
            SubScore(name="sync", value=0.0, weight=0.5),
            async_score(),
        )
        assert result.reward == pytest.approx(0.5)


#: ``BashGrader`` shells out to ``/bin/bash``; skip its tests where it's absent (Windows).
_HAS_BASH = os.path.exists("/bin/bash")


class TestCombine:
    async def test_combine_returns_evaluation_result(self) -> None:
        result = await combine(SubScore(name="alpha", value=1.0, weight=1.0))
        assert isinstance(result, EvaluationResult)
        assert result.reward == 1.0
        assert result.done is True

    async def test_combine_normalizes_positive_weights(self) -> None:
        result = await combine(
            SubScore(name="alpha", value=1.0, weight=2.0),
            SubScore(name="beta", value=0.0, weight=1.0),
        )
        assert result.reward == pytest.approx(2.0 / 3.0)
        assert result.subscores is not None
        by_name = {subscore.name: subscore for subscore in result.subscores}
        assert by_name["alpha"].weight == pytest.approx(2.0 / 3.0)
        assert by_name["beta"].weight == pytest.approx(1.0 / 3.0)

    async def test_combine_preserves_negative_penalties(self) -> None:
        result = await combine(
            SubScore(name="correct", value=1.0, weight=1.0),
            SubScore(name="penalty", value=1.0, weight=-0.2),
        )
        assert result.reward == pytest.approx(0.8)
        assert result.subscores is not None
        by_name = {subscore.name: subscore for subscore in result.subscores}
        assert by_name["correct"].weight == pytest.approx(1.0)
        assert by_name["penalty"].weight == pytest.approx(-0.2)

    async def test_combine_duplicate_names_are_deduped(self) -> None:
        result = await combine(
            SubScore(name="same", value=1.0, weight=0.5),
            SubScore(name="same", value=0.0, weight=0.5),
        )
        assert result.subscores is not None
        assert [subscore.name for subscore in result.subscores] == ["same-1", "same-2"]

    async def test_combine_duplicate_names_avoid_existing_suffix_collisions(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await combine(
                SubScore(name="x-1", value=1.0, weight=0.3),
                SubScore(name="x", value=1.0, weight=0.4),
                SubScore(name="x", value=0.0, weight=0.6),
            )

        assert result.subscores is not None
        assert [subscore.name for subscore in result.subscores] == ["x-1", "x-2", "x-3"]
        assert set(result.info) == set()
        assert not [
            warning for warning in caught if "Duplicate subscore names" in str(warning.message)
        ]

    async def test_combine_propagates_metadata(self) -> None:
        metadata = {"stdout": "ok"}
        result = await combine(SubScore(name="grader", value=1.0, weight=1.0, metadata=metadata))
        assert result.info == {}
        assert result.subscores is not None
        assert result.subscores[0].metadata == metadata
        assert "metadata" not in result.model_dump(mode="json")["subscores"][0]

    async def test_combine_keeps_rubric_children_on_subscores(self) -> None:
        verdicts = [
            SubScore(name="Mentions Paris", value=1.0, metadata={"reason": "says Paris"}),
            SubScore(name="Names the river", value=0.0, metadata={"reason": "no river"}),
        ]
        result = await combine(
            SubScore(name="judge", value=0.5, weight=0.5, children=verdicts),
            SubScore(name="tests", value=0.0, weight=0.5),
        )
        assert result.subscores is not None
        by_name = {subscore.name: subscore for subscore in result.subscores}
        assert by_name["judge"].children == verdicts
        assert by_name["tests"].children is None

    async def test_combine_preserves_combinator_children(self) -> None:
        result = await combine(
            combine_any(
                weight=0.5,
                subscores=[
                    SubScore(name="pytest", value=1.0),
                    SubScore(name="make", value=0.0),
                ],
            ),
            SubScore(name="format", value=1.0, weight=0.5),
        )
        assert result.subscores is not None
        by_name = {subscore.name: subscore for subscore in result.subscores}
        any_node = by_name["any"]
        assert any_node.children is not None
        assert [child.name for child in any_node.children] == ["pytest", "make"]
        assert by_name["format"].children is None

    async def test_combine_preserves_negative_reward_without_validator_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await combine(
                SubScore(name="correct", value=0.0, weight=1.0),
                SubScore(name="penalty", value=1.0, weight=-0.2),
            )

        assert result.reward == pytest.approx(-0.2)
        assert not [
            warning for warning in caught if "Subscores don't match reward" in str(warning.message)
        ]


class TestGradeCompatShim:
    """v5 environments call ``Grade.gather`` / ``Grade.from_subscores`` via ``hud.native``."""

    async def test_gather_combines_like_combine(self) -> None:
        from hud.native import Grade  # pyright: ignore[reportAttributeAccessIssue]

        result = await Grade.gather(
            SubScore(name="alpha", value=1.0, weight=1.0),
            SubScore(name="beta", value=0.0, weight=1.0),
        )
        assert isinstance(result, EvaluationResult)
        assert result.reward == pytest.approx(0.5)

    def test_from_subscores_is_sync(self) -> None:
        from hud.native.graders import Grade

        result = Grade.from_subscores([SubScore(name="alpha", value=1.0, weight=1.0)])
        assert isinstance(result, EvaluationResult)
        assert result.reward == 1.0


class TestGrader:
    async def test_grade_wraps_float_and_stores_parameters(self) -> None:
        class DummyGrader(Grader):
            name = "DummyGrader"

            @classmethod
            async def compute_score(cls, **kwargs: object) -> Any:
                return 0.75

        subscore = await DummyGrader.grade(weight=0.4, marker="ok", payload=object())
        assert isinstance(subscore, SubScore)
        assert subscore.name == "DummyGrader"
        assert subscore.value == pytest.approx(0.75)
        assert subscore.weight == pytest.approx(0.4)
        assert subscore.metadata is not None
        assert subscore.metadata["_parameters"]["marker"] == "ok"
        assert subscore.metadata["_parameters"]["payload"] == "<object: not serializable>"

    async def test_grade_coerces_legacy_tuple_metadata(self) -> None:
        class TupleGrader(Grader):
            name = "tuple"

            @classmethod
            async def compute_score(cls, **kwargs: object) -> Any:
                return 0.75, {"source": "released-contract"}

        subscore = await TupleGrader.grade(weight=0.4, marker="ok")
        assert subscore.name == "tuple"
        assert subscore.value == pytest.approx(0.75)
        assert subscore.weight == pytest.approx(0.4)
        assert subscore.metadata is not None
        assert subscore.metadata["source"] == "released-contract"
        assert subscore.metadata["_parameters"]["marker"] == "ok"

    async def test_grade_uses_grader_name_without_override(self) -> None:
        class RubricGrader(Grader):
            name = "rubric"

            @classmethod
            async def compute_score(cls, **kwargs: object) -> SubScore:
                return SubScore(name="specific-rubric", value=1.0)

        subscore = await RubricGrader.grade(weight=0.7)
        assert subscore.name == "rubric"
        assert subscore.weight == pytest.approx(0.7)

    async def test_grade_stamps_name_and_weight_on_returned_subscore(self) -> None:
        class RubricGrader(Grader):
            name = "rubric"

            @classmethod
            async def compute_score(cls, **kwargs: object) -> SubScore:
                return SubScore(
                    name="ignored",
                    value=1.0,
                    metadata={"reason": "did the thing", "extra": "kept"},
                )

        subscore = await RubricGrader.grade(weight=0.7, name="my-rubric")
        assert subscore.name == "my-rubric"
        assert subscore.weight == pytest.approx(0.7)
        assert subscore.metadata is not None
        assert subscore.metadata["reason"] == "did the thing"
        assert subscore.metadata["extra"] == "kept"
        assert "_parameters" in subscore.metadata


class TestBooleanCombinators:
    def test_combine_any_picks_max(self) -> None:
        combined = combine_any(
            weight=1.0,
            subscores=[
                SubScore(name="a", value=1.0, weight=0.5),
                SubScore(name="b", value=0.0, weight=0.5),
            ],
        )
        assert combined.name == "any"
        assert combined.value == 1.0

    def test_combine_any_keeps_inputs_as_children(self) -> None:
        rubric = SubScore(
            name="judge",
            value=1.0,
            weight=0.5,
            children=[SubScore(name="c1", value=1.0, metadata={"reason": "met"})],
        )
        combined = combine_any(
            weight=1.0,
            subscores=[rubric, SubScore(name="tests", value=0.0, weight=0.5)],
        )
        assert combined.children is not None
        assert combined.children[0].children == rubric.children
        assert combined.children[1].children is None

    def test_combine_any_dedupes_child_names(self) -> None:
        combined = combine_any(
            weight=1.0,
            subscores=[
                SubScore(name="BashGrader", value=1.0, weight=0.5, metadata={"exit_code": 0}),
                SubScore(name="BashGrader", value=0.0, weight=0.5, metadata={"exit_code": 1}),
            ],
        )
        assert combined.children is not None
        assert [child.name for child in combined.children] == ["BashGrader-1", "BashGrader-2"]
        assert [child.metadata for child in combined.children] == [
            {"exit_code": 0},
            {"exit_code": 1},
        ]
        assert combined.metadata is None

    def test_combine_all_picks_min(self) -> None:
        combined = combine_all(
            weight=1.0,
            subscores=[
                SubScore(name="a", value=1.0, weight=0.5),
                SubScore(name="b", value=0.0, weight=0.5),
            ],
            name="tests_all",
        )
        assert combined.name == "tests_all"
        assert combined.value == 0.0

    def test_nested_combinators_build_a_tree(self) -> None:
        combined = combine_all(
            weight=1.0,
            subscores=[
                combine_any(
                    weight=1.0,
                    subscores=[
                        SubScore(name="pytest", value=0.0),
                        SubScore(name="make", value=1.0),
                    ],
                ),
                SubScore(name="lint", value=1.0),
            ],
        )
        assert combined.value == 1.0
        assert combined.children is not None
        inner = combined.children[0]
        assert inner.children is not None
        assert [child.name for child in inner.children] == ["pytest", "make"]


class _FakeGatewayClient:
    """Judges MET when the criterion text (inside the user prompt) contains 'met:'."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        user_prompt: str = kwargs["messages"][1]["content"]
        status = "MET" if "met:" in user_prompt else "UNMET"
        content = json.dumps({"criterion_status": status, "explanation": "because"})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class TestLLMJudgeGrader:
    async def test_compute_score_returns_rubric_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "hud.utils.gateway.build_gateway_client", lambda provider: _FakeGatewayClient()
        )
        subscore = await LLMJudgeGrader.compute_score(
            answer="some answer",
            criteria=["met: mentions Paris", ("names the river", 3.0)],
        )
        assert subscore.value == pytest.approx(0.25)
        assert subscore.children is not None
        assert [(c.name, c.value, c.weight, c.metadata) for c in subscore.children] == [
            ("met: mentions Paris", 1.0, 1.0, {"reason": "because"}),
            ("names the river", 0.0, 3.0, {"reason": "because"}),
        ]

    async def test_compute_score_applies_negative_criterion_penalty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "hud.utils.gateway.build_gateway_client", lambda provider: _FakeGatewayClient()
        )
        subscore = await LLMJudgeGrader.compute_score(
            answer="some answer",
            criteria=[("met: positive", 1.0), ("met: error", -0.5)],
        )
        assert subscore.value == pytest.approx(0.5)

    async def test_grade_puts_verdicts_on_children(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "hud.utils.gateway.build_gateway_client", lambda provider: _FakeGatewayClient()
        )
        subscore = await LLMJudgeGrader.grade(
            weight=1.0, answer="some answer", criteria=["met: correct"]
        )
        assert subscore.children == [
            SubScore(
                name="met: correct",
                value=1.0,
                weight=1.0,
                metadata={"reason": "because"},
            )
        ]
        assert subscore.metadata is not None
        assert subscore.metadata["model"] == "claude-haiku-4-5"


@pytest.mark.skipif(not _HAS_BASH, reason="/bin/bash not available (e.g. Windows)")
class TestBashGrader:
    async def test_compute_score_for_passing_command(self) -> None:
        subscore = await BashGrader.compute_score(command="echo hello")
        assert subscore.value == 1.0
        assert subscore.metadata is not None
        assert subscore.metadata["exit_code"] == 0
        assert "hello" in subscore.metadata["stdout"]

    async def test_compute_score_for_failing_command(self) -> None:
        subscore = await BashGrader.compute_score(command="echo oops >&2 && false")
        assert subscore.value == 0.0
        assert subscore.metadata is not None
        assert subscore.metadata["exit_code"] != 0
        assert "oops" in subscore.metadata["stderr"]

    async def test_compute_score_timeout(self) -> None:
        subscore = await BashGrader.compute_score(command="sleep 2", timeout_seconds=1)
        assert subscore.value == 0.0
        assert subscore.metadata is not None
        assert subscore.metadata["timed_out"] is True
        assert subscore.metadata["timeout"] == 1

    async def test_grade_and_combine_compose(self) -> None:
        result = await combine(
            BashGrader.grade(weight=0.5, command="true"),
            BashGrader.grade(weight=0.5, command="false"),
        )
        assert result.reward == pytest.approx(0.5)
        assert result.subscores is not None
        by_name = {subscore.name: subscore for subscore in result.subscores}
        assert by_name["BashGrader-1"].metadata is not None
        assert by_name["BashGrader-1"].metadata["exit_code"] == 0
        assert by_name["BashGrader-2"].metadata is not None
        assert by_name["BashGrader-2"].metadata["exit_code"] != 0
        assert result.info == {}
