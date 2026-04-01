"""Tests for first-party HUD native graders."""

from __future__ import annotations

import warnings

import pytest

from hud.environment import Environment
from hud.native.graders import BashGrader, Grade, Grader
from hud.tools.types import EvaluationResult, SubScore


class TestGrade:
    def test_from_subscores_returns_evaluation_result(self) -> None:
        result = Grade.from_subscores([SubScore(name="alpha", value=1.0, weight=1.0)])
        assert isinstance(result, EvaluationResult)
        assert result.reward == 1.0
        assert result.done is True

    def test_from_subscores_normalizes_positive_weights(self) -> None:
        result = Grade.from_subscores(
            [
                SubScore(name="alpha", value=1.0, weight=2.0),
                SubScore(name="beta", value=0.0, weight=1.0),
            ]
        )
        assert result.reward == pytest.approx(2.0 / 3.0)
        assert result.subscores is not None
        by_name = {subscore.name: subscore for subscore in result.subscores}
        assert by_name["alpha"].weight == pytest.approx(2.0 / 3.0)
        assert by_name["beta"].weight == pytest.approx(1.0 / 3.0)

    def test_from_subscores_preserves_negative_penalties(self) -> None:
        result = Grade.from_subscores(
            [
                SubScore(name="correct", value=1.0, weight=1.0),
                SubScore(name="penalty", value=1.0, weight=-0.2),
            ]
        )
        assert result.reward == pytest.approx(0.8)
        assert result.subscores is not None
        by_name = {subscore.name: subscore for subscore in result.subscores}
        assert by_name["correct"].weight == pytest.approx(1.0)
        assert by_name["penalty"].weight == pytest.approx(-0.2)

    def test_from_subscores_duplicate_names_are_deduped(self) -> None:
        result = Grade.from_subscores(
            [
                SubScore(name="same", value=1.0, weight=0.5),
                SubScore(name="same", value=0.0, weight=0.5),
            ]
        )
        assert result.subscores is not None
        assert [subscore.name for subscore in result.subscores] == ["same-1", "same-2"]

    def test_from_subscores_duplicate_names_avoid_existing_suffix_collisions(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Grade.from_subscores(
                [
                    SubScore(name="x-1", value=1.0, weight=0.3),
                    SubScore(name="x", value=1.0, weight=0.4),
                    SubScore(name="x", value=0.0, weight=0.6),
                ]
            )

        assert result.subscores is not None
        assert [subscore.name for subscore in result.subscores] == ["x-1", "x-2", "x-3"]
        assert set(result.info) == set()
        assert not [
            warning for warning in caught if "Duplicate subscore names" in str(warning.message)
        ]

    def test_from_subscores_propagates_metadata(self) -> None:
        metadata = {"stdout": "ok"}
        result = Grade.from_subscores(
            [SubScore(name="grader", value=1.0, weight=1.0, metadata=metadata)]
        )
        assert result.info["grader"] == metadata
        assert result.subscores is not None
        assert result.subscores[0].metadata == metadata

    def test_from_subscores_preserves_negative_reward_without_validator_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Grade.from_subscores(
                [
                    SubScore(name="correct", value=0.0, weight=1.0),
                    SubScore(name="penalty", value=1.0, weight=-0.2),
                ]
            )

        assert result.reward == pytest.approx(-0.2)
        assert not [
            warning for warning in caught if "Subscores don't match reward" in str(warning.message)
        ]


class TestGrader:
<<<<<<< HEAD
    async def test_grade_returns_subscore_and_stores_parameters(self) -> None:
=======
    def test_grade_returns_subscore_and_stores_parameters(self) -> None:
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        class DummyGrader(Grader):
            name = "DummyGrader"

            @classmethod
<<<<<<< HEAD
            async def compute_score(cls, **kwargs: object) -> tuple[float, dict[str, object]]:
                return 0.75, {"source": "dummy", "kwargs_seen": sorted(kwargs)}

        subscore = await DummyGrader.grade(weight=0.4, marker="ok", payload=object())
=======
            def compute_score(cls, **kwargs: object) -> tuple[float, dict[str, object]]:
                return 0.75, {"source": "dummy", "kwargs_seen": sorted(kwargs)}

        subscore = DummyGrader.grade(weight=0.4, marker="ok", payload=object())
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        assert isinstance(subscore, SubScore)
        assert subscore.name == "DummyGrader"
        assert subscore.value == pytest.approx(0.75)
        assert subscore.weight == pytest.approx(0.4)
        assert subscore.metadata is not None
        assert subscore.metadata["source"] == "dummy"
        assert subscore.metadata["_parameters"]["marker"] == "ok"
        assert subscore.metadata["_parameters"]["payload"] == "<object: not serializable>"


class TestGraderCombinators:
    def test_any_picks_max(self) -> None:
        combined = Grader.any(
            weight=1.0,
            subscores=[
                SubScore(name="a", value=1.0, weight=0.5),
                SubScore(name="b", value=0.0, weight=0.5),
            ],
        )
        assert combined.name == "BaseGrader_any"
        assert combined.value == 1.0

    def test_any_preserves_metadata_for_duplicate_named_subscores(self) -> None:
        combined = Grader.any(
            weight=1.0,
            subscores=[
                SubScore(name="BashGrader", value=1.0, weight=0.5, metadata={"exit_code": 0}),
                SubScore(name="BashGrader", value=0.0, weight=0.5, metadata={"exit_code": 1}),
            ],
        )
        assert combined.metadata == {
            "subscores": ["BashGrader-1", "BashGrader-2"],
            "subscore_metadata": {
                "BashGrader-1": {"exit_code": 0},
                "BashGrader-2": {"exit_code": 1},
            },
        }

    def test_all_picks_min(self) -> None:
        combined = Grader.all(
            weight=1.0,
            subscores=[
                SubScore(name="a", value=1.0, weight=0.5),
                SubScore(name="b", value=0.0, weight=0.5),
            ],
        )
        assert combined.name == "BaseGrader_all"
        assert combined.value == 0.0

    def test_all_preserves_metadata_for_duplicate_named_subscores(self) -> None:
        combined = Grader.all(
            weight=1.0,
            subscores=[
                SubScore(name="BashGrader", value=1.0, weight=0.5, metadata={"exit_code": 0}),
                SubScore(name="BashGrader", value=0.0, weight=0.5, metadata={"exit_code": 1}),
            ],
        )
        assert combined.metadata == {
            "subscores": ["BashGrader-1", "BashGrader-2"],
            "subscore_metadata": {
                "BashGrader-1": {"exit_code": 0},
                "BashGrader-2": {"exit_code": 1},
            },
        }


class TestBashGrader:
<<<<<<< HEAD
    async def test_compute_score_for_passing_command(self) -> None:
        score, metadata = await BashGrader.compute_score(command="echo hello")
=======
    def test_compute_score_for_passing_command(self) -> None:
        score, metadata = BashGrader.compute_score(command="echo hello")
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        assert score == 1.0
        assert metadata["exit_code"] == 0
        assert "hello" in metadata["stdout"]

<<<<<<< HEAD
    async def test_compute_score_for_failing_command(self) -> None:
        score, metadata = await BashGrader.compute_score(command="echo oops >&2 && false")
=======
    def test_compute_score_for_failing_command(self) -> None:
        score, metadata = BashGrader.compute_score(command="echo oops >&2 && false")
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        assert score == 0.0
        assert metadata["exit_code"] != 0
        assert "oops" in metadata["stderr"]

<<<<<<< HEAD
    async def test_compute_score_timeout(self) -> None:
        score, metadata = await BashGrader.compute_score(command="sleep 2", timeout_seconds=1)
=======
    def test_compute_score_timeout(self) -> None:
        score, metadata = BashGrader.compute_score(command="sleep 2", timeout=1)
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        assert score == 0.0
        assert metadata["timed_out"] is True
        assert metadata["timeout"] == 1

<<<<<<< HEAD
    async def test_grade_and_from_subscores_compose(self) -> None:
        passing = await BashGrader.grade(weight=0.5, command="true")
        failing = await BashGrader.grade(weight=0.5, command="false")
=======
    def test_compute_score_invalid_cwd_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            BashGrader.compute_score(command="true", cwd=str(tmp_path / "missing"))

    def test_grade_and_from_subscores_compose(self) -> None:
        passing = BashGrader.grade(weight=0.5, command="true")
        failing = BashGrader.grade(weight=0.5, command="false")
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        result = Grade.from_subscores([passing, failing])
        assert result.reward == pytest.approx(0.5)
        assert result.info["BashGrader-1"]["exit_code"] == 0
        assert result.info["BashGrader-2"]["exit_code"] != 0

<<<<<<< HEAD
    async def test_grade_and_gather_compose(self) -> None:
        result = await Grade.gather(
            BashGrader.grade(weight=0.5, command="true"),
            BashGrader.grade(weight=0.5, command="false"),
        )
        assert result.reward == pytest.approx(0.5)


class TestScenarioIntegration:
    async def test_scenario_can_yield_grade_from_gather(self) -> None:
=======

class TestScenarioIntegration:
    @pytest.mark.asyncio
    async def test_scenario_can_yield_grade_from_subscores(self) -> None:
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20
        env = Environment("test-env")

        @env.scenario("bash-graded")
        async def bash_graded_scenario():
            yield "Run the verification"
<<<<<<< HEAD
            yield await Grade.gather(
                BashGrader.grade(weight=1.0, command="echo verified"),
            )
=======
            yield Grade.from_subscores([BashGrader.grade(weight=1.0, command="echo verified")])
>>>>>>> 3bae301c628a4de571c2531d1815203ff104de20

        prompt = await env.run_scenario_setup("bash-graded", {})
        assert prompt == "Run the verification"

        assert env._active_session is not None
        env._active_session.answer = "done"
        result = await env.run_scenario_evaluate("bash-graded")

        assert result is not None
        assert result.reward == 1.0
        assert result.subscores is not None
        assert result.subscores[0].name == "BashGrader"
        assert "verified" in result.info["BashGrader"]["stdout"]
