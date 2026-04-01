"""Tests for hud.native.graders answer-comparison helpers."""

from __future__ import annotations

import pytest

from hud.native.graders import (
    contains,
    contains_all,
    contains_any,
    exact_match,
    f1_score,
    normalize,
    numeric_match,
)


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


class TestGradeGather:
    async def test_gather_sync_subscores(self) -> None:
        from hud.native.graders import Grade
        from hud.tools.types import SubScore

        result = await Grade.gather(
            SubScore(name="a", value=1.0, weight=0.5),
            SubScore(name="b", value=0.0, weight=0.5),
        )
        assert result.reward == pytest.approx(0.5)

    async def test_gather_with_awaitables(self) -> None:
        import asyncio

        from hud.native.graders import Grade
        from hud.tools.types import SubScore

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

        result = await Grade.gather(slow_check_a(), slow_check_b())
        assert result.reward == pytest.approx(0.5)
        assert order.index("b_start") < order.index("a_end")

    async def test_gather_mixed(self) -> None:
        import asyncio

        from hud.native.graders import Grade
        from hud.tools.types import SubScore

        async def async_score() -> SubScore:
            await asyncio.sleep(0.01)
            return SubScore(name="async", value=1.0, weight=0.5)

        result = await Grade.gather(
            SubScore(name="sync", value=0.0, weight=0.5),
            async_score(),
        )
        assert result.reward == pytest.approx(0.5)
