"""Tests for selected JUnit scoring."""

from pathlib import Path

import pytest

from grader import JUnitCase, JUnitGrader, parse_junit, score_tests


def test_scoring_tracks_fail_to_pass_and_pass_to_pass(tmp_path: Path):
    report = tmp_path / "junit.xml"
    report.write_text(
        """<testsuite>
  <testcase classname="tests.test_widget" name="test_fixed"><failure /></testcase>
  <testcase classname="tests.test_widget" name="test_regression" />
</testsuite>
""",
        encoding="utf-8",
    )

    result = score_tests(
        parse_junit(report),
        ["tests.test_widget::test_fixed"],
        ["tests.test_widget.test_regression"],
        binary=False,
    )

    assert result.value == 0.5
    assert result.info is not None
    assert result.info["f2p_passed"] == 0
    assert result.info["p2p_passed"] == 1


def test_binary_scoring_requires_every_selected_test():
    result = score_tests(
        [
            JUnitCase("tests.test_widget.test_fixed", passed=True, skipped=False),
            JUnitCase("tests.test_widget.test_regression", passed=False, skipped=False),
        ],
        ["tests.test_widget.test_fixed"],
        ["tests.test_widget.test_regression"],
        binary=True,
    )

    assert result.value == 0.0


def test_missing_selected_test_counts_as_failure():
    result = score_tests(
        [JUnitCase("tests.test_widget.test_present", passed=True, skipped=False)],
        ["tests.test_widget.test_missing"],
        ["tests.test_widget.test_present"],
        binary=False,
    )

    assert result.value == 0.5


def test_truncated_parameter_id_matches_all_reported_cases():
    result = score_tests(
        [
            JUnitCase(
                'tests.test_cli.test_locate_app[cliapp.factory-create_app2("foo", "bar")]',
                passed=True,
                skipped=False,
            )
        ],
        ['tests.test_cli.test_locate_app[cliapp.factory-create_app2("foo",'],
        [],
        binary=True,
    )

    assert result.value == 1.0


def test_scoring_rejects_empty_or_duplicate_ids():
    case = JUnitCase("tests.test_widget.test_duplicate", passed=True, skipped=False)

    with pytest.raises(ValueError, match="at least one"):
        score_tests([case], [], [], binary=False)
    with pytest.raises(ValueError, match="JUnit test case IDs"):
        score_tests([case, case], None, None, binary=False)


def test_parse_junit_rejects_malformed_report(tmp_path: Path):
    report = tmp_path / "junit.xml"
    report.write_text("<testsuite>", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JUnit XML"):
        parse_junit(report)


@pytest.mark.asyncio
async def test_junit_grader_reports_missing_output(tmp_path: Path):
    result = await JUnitGrader.compute_score(command="true {junit_path}", cwd=str(tmp_path))

    assert result.value == 0.0
    assert result.info is not None
    assert result.info["error"] == "test command did not write JUnit XML"
    assert result.info["exit_code"] == 0


@pytest.mark.asyncio
async def test_binary_junit_grader_rejects_passing_report_from_failed_command(tmp_path: Path):
    result = await JUnitGrader.compute_score(
        command=(
            'printf \'<testsuite><testcase classname="tests.test_widget" '
            'name="test_fixed" /></testsuite>\' > {junit_path}; false'
        ),
        cwd=str(tmp_path),
        binary=True,
    )

    assert result.value == 0.0
    assert result.info is not None
    assert result.info["exit_code"] == 1
    assert "all_testcases" not in result.info
