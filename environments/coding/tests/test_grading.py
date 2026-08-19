"""Tests for JUnit scoring and diff-based grading primitives."""

from pathlib import Path

import pytest

from coding import repo as repo_lib
from coding.grading import JUnitCase, parse_junit, score_tests


def test_junit_scoring_tracks_fail_to_pass_and_pass_to_pass(tmp_path: Path):
    report = tmp_path / "junit.xml"
    report.write_text(
        """<?xml version="1.0"?>
<testsuite tests="3" failures="1">
  <testcase classname="tests.test_widget" name="test_fixed">
    <failure message="still broken" />
  </testcase>
  <testcase classname="tests.test_widget" name="test_regression" />
  <testcase classname="tests.test_widget" name="test_unselected" />
</testsuite>
""",
        encoding="utf-8",
    )

    result = score_tests(
        parse_junit(report),
        ["tests.test_widget::test_fixed"],
        ["tests.test_widget.test_regression"],
        use_binary_score=False,
    )

    assert result.reward == 0.5
    assert result.info["f2p_passed"] == 0
    assert result.info["p2p_passed"] == 1
    assert result.info["total"] == 2


def test_binary_junit_scoring_requires_every_selected_test():
    result = score_tests(
        [
            JUnitCase("tests.test_widget.test_fixed", passed=True, skipped=False),
            JUnitCase("tests.test_widget.test_regression", passed=False, skipped=False),
        ],
        ["tests.test_widget.test_fixed"],
        ["tests.test_widget.test_regression"],
        use_binary_score=True,
    )

    assert result.reward == 0.0


def test_missing_selected_junit_test_counts_as_failure():
    result = score_tests(
        [JUnitCase("tests.test_widget.test_present", passed=True, skipped=False)],
        ["tests.test_widget.test_missing"],
        ["tests.test_widget.test_present"],
        use_binary_score=False,
    )

    assert result.reward == 0.5
    assert result.info["f2p_passed"] == 0


def test_truncated_parameter_id_matches_unambiguous_junit_cases():
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
        use_binary_score=True,
    )

    assert result.reward == 1.0


def test_junit_scoring_rejects_empty_or_duplicate_ids():
    case = JUnitCase("tests.test_widget.test_duplicate", passed=True, skipped=False)

    with pytest.raises(ValueError, match="at least one"):
        score_tests([case], [], [], use_binary_score=False)
    with pytest.raises(ValueError, match="JUnit test case IDs"):
        score_tests([case, case], None, None, use_binary_score=False)


def test_parse_junit_rejects_malformed_report(tmp_path: Path):
    report = tmp_path / "junit.xml"
    report.write_text("<testsuite>", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JUnit XML"):
        parse_junit(report)


@pytest.mark.asyncio
async def test_binary_and_non_utf8_changes_survive_diff_round_trip(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    await repo_lib.git(repo, "init", "-q", "-b", "main")
    binary = repo / "asset.bin"
    binary.write_bytes(b"\x00before\xff")
    legacy_text = repo / "legacy.txt"
    legacy_text.write_bytes(b"caf\xe9 before\n")
    await repo_lib.git(repo, "add", "-A")
    await repo_lib.git(repo, "commit", "-qm", "baseline")
    _, setup_commit_bytes, _ = await repo_lib.git(repo, "rev-parse", "HEAD")
    setup_commit = setup_commit_bytes.decode().strip()

    binary.write_bytes(b"\x00after\xfe")
    legacy_text.write_bytes(b"caf\xe9 after\n")
    diff = await repo_lib.capture_agent_diff(repo, setup_commit)

    assert b"GIT binary patch" in diff
    assert b"caf\xe9" in diff
    await repo_lib.reset_worktree(repo, setup_commit)
    error = await repo_lib.apply_diff(repo, diff, tmp_path / "agent.diff")

    assert error is None
    assert binary.read_bytes() == b"\x00after\xfe"
    assert legacy_text.read_bytes() == b"caf\xe9 after\n"
