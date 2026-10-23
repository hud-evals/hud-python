"""Tests for selected JUnit scoring."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from grader import JUnitCase, grade_tests, parse_junit, score_tests


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


@pytest.mark.asyncio(loop_scope="session")
async def test_junit_grader_reports_missing_output(grading_workspace):
    result = await grade_tests(grading_workspace, "true {junit_path}")

    assert result.value == 0.0
    assert result.info is not None
    assert result.info["report_error"] == "test command did not write JUnit XML"
    assert "error" not in result.info
    assert result.info["exit_code"] == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_binary_junit_grader_rejects_passing_report_from_failed_command(grading_workspace):
    result = await grade_tests(
        grading_workspace,
        command=(
            'printf \'<testsuite><testcase classname="tests.test_widget" '
            'name="test_fixed" /></testsuite>\' > {junit_path}; false'
        ),
        binary=True,
    )

    assert result.value == 0.0
    assert result.info is not None
    assert result.info["exit_code"] == 1
    assert "error" not in result.info
    assert "all_testcases" not in result.info


@pytest.mark.asyncio(loop_scope="session")
async def test_grading_runs_as_agent_after_session_termination(grading_workspace):
    session = await grading_workspace.run(["/bin/bash", "-lc", "echo agent > agent.txt"])
    assert session.returncode == 0
    await grading_workspace.terminate_sessions()

    result = await grade_tests(
        grading_workspace,
        'test "$(id -u)" = 1000 && test "$(cat agent.txt)" = agent && '
        "grep -q localhost /etc/hosts && "
        'printf \'<testsuite><testcase classname="tests" name="passed" /></testsuite>\' > {junit_path}',
        binary=True,
    )

    assert result.value == 1.0
    assert result.info["exit_code"] == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_timed_out_grader_does_not_score_partial_report(grading_workspace):
    result = await grade_tests(
        grading_workspace,
        'printf \'<testsuite><testcase classname="tests" name="passed" /></testsuite>\' > {junit_path}; sleep 30',
        timeout_seconds=1,
    )

    assert result.value == 0.0
    assert result.info["timed_out"] is True


@pytest.mark.asyncio(loop_scope="session")
async def test_timed_out_grader_without_report_is_a_graded_failure(grading_workspace):
    # A hidden-test run killed at the timeout usually writes no report. It is the agent's
    # code that hung, so it must score 0 rather than be marked a grading error.
    result = await grade_tests(grading_workspace, "true {junit_path}; sleep 30", timeout_seconds=1)

    assert result.value == 0.0
    assert result.info["timed_out"] is True
    assert "error" not in result.info


class _LocalWorkspace:
    """Runs the grading command with local bash instead of the sandbox, so the report
    handling can be tested on any platform. Records every command it runs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.commands: list[list[str]] = []

    async def run(self, argv, cwd, env, max_wait, scope):
        self.commands.append(argv)
        try:
            done = subprocess.run(argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, timeout=max_wait)
        except subprocess.TimeoutExpired as expired:
            return SimpleNamespace(
                returncode=None,
                stdout=expired.stdout or b"",
                stderr=expired.stderr or b"",
                timed_out=True,
            )
        return SimpleNamespace(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr, timed_out=False)


@pytest.mark.asyncio(loop_scope="session")
async def test_timeout_is_scored_before_the_missing_report_check(tmp_path: Path):
    result = await grade_tests(_LocalWorkspace(tmp_path), "true {junit_path}; sleep 30", timeout_seconds=1)

    assert result.value == 0.0
    assert result.info["timed_out"] is True
    assert "error" not in result.info


@pytest.mark.parametrize(
    ("command", "report_error"),
    [
        # The agent's code ended the hidden test run before pytest wrote its report.
        ("true {junit_path}", "test command did not write JUnit XML"),
        # The agent's code overwrote the report.
        ("printf '<testsuite>' > {junit_path}", "invalid JUnit XML"),
    ],
    ids=["missing-report", "malformed-report"],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_unusable_report_is_a_graded_failure(tmp_path: Path, command: str, report_error: str):
    result = await grade_tests(_LocalWorkspace(tmp_path), command)

    assert result.value == 0.0
    assert report_error in result.info["report_error"]
    assert "error" not in result.info


@pytest.mark.asyncio(loop_scope="session")
async def test_invalid_selection_is_a_grading_error_before_any_test_runs(tmp_path: Path):
    workspace = _LocalWorkspace(tmp_path)

    result = await grade_tests(workspace, "pytest --junitxml={junit_path}", fail_to_pass=["tests.a", "tests.a"])

    assert result.value == 0.0
    assert result.info == {"error": "test node IDs must be unique"}
    assert workspace.commands == []
