"""Offline tests for the SWE-bench Pro flavor's pure pieces: the resolution
criterion, dataset field parsing, and patch sanitization."""

import json
from pathlib import Path

import pytest

from coding import repo as repo_lib
from coding.swe_bench_pro import score, str_list, strip_binary_hunks

INSTANCE = json.loads((Path(__file__).parent / "fixtures" / "instance" / "instance.json").read_text("utf-8"))

F2P = [
    "tests/test_widgets.py | test_widgets_work",
    "tests/test_widgets.py | test_widgets_don't_break",
]
P2P = ["tests/test_widgets.py | test_existing_behavior"]


def _reported(names, status="PASSED"):
    return [{"name": name, "status": status} for name in names]


def test_resolved_when_all_required_tests_pass():
    result = score(INSTANCE, _reported(F2P + P2P + ["extra | irrelevant_test"]))
    assert result.reward == 1.0
    assert result.content == "resolved"


def test_unresolved_when_a_fail_to_pass_test_fails():
    reported = _reported(F2P[:1] + P2P) + _reported(F2P[1:], status="FAILED")
    result = score(INSTANCE, reported)
    assert result.reward == 0.0
    by_name = {s.name: s.value for s in result.subscores}
    assert by_name["fail_to_pass"] == 0.5
    assert by_name["pass_to_pass"] == 1.0
    assert F2P[1] in result.info["missing"]


def test_unresolved_on_pass_to_pass_regression():
    """Fixing the bug while breaking existing behavior is not resolved."""
    result = score(INSTANCE, _reported(F2P) + _reported(P2P, status="ERROR"))
    assert result.reward == 0.0
    by_name = {s.name: s.value for s in result.subscores}
    assert by_name["fail_to_pass"] == 1.0
    assert by_name["pass_to_pass"] == 0.0


def test_missing_report_scores_zero():
    result = score(INSTANCE, [])
    assert result.reward == 0.0


def test_str_list_parses_python_repr_fields():
    """Rows store lists as Python reprs (the official evaluator uses eval)."""
    assert str_list(INSTANCE["fail_to_pass"]) == F2P
    assert str_list(INSTANCE["selected_test_files_to_run"]) == ["tests/test_widgets.py"]
    with pytest.raises(ValueError):
        str_list("'not a list'")


def test_strip_binary_hunks_drops_only_binary_sections():
    text = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    binary = "diff --git a/img b/img\nBinary files a/img and b/img differ\n"
    assert strip_binary_hunks(text + binary) == text
    assert strip_binary_hunks("") == ""


@pytest.mark.asyncio
async def test_binary_agent_changes_survive_diff_round_trip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    await repo_lib.git(repo, "init", "-q", "-b", "main")
    binary = repo / "asset.bin"
    binary.write_bytes(b"\x00before\xff")
    await repo_lib.git(repo, "add", "-A")
    await repo_lib.git(repo, "commit", "-qm", "baseline")
    _, setup_commit, _ = await repo_lib.git(repo, "rev-parse", "HEAD")

    binary.write_bytes(b"\x00after\xfe")
    diff = await repo_lib.capture_agent_diff(repo, setup_commit.strip())

    assert "GIT binary patch" in diff
    await repo_lib.reset_worktree(repo, setup_commit.strip())
    error = await repo_lib.apply_diff(repo, diff, tmp_path / "agent.diff")

    assert error is None
    assert binary.read_bytes() == b"\x00after\xfe"
