"""Offline tests for the mock GitHub store's contract."""

import json

import pytest

from coding.github import MockGitHub

ISSUE = {"number": 42, "title": "Crash", "body": "It crashes.", "labels": ["bug"]}


def _seeded() -> MockGitHub:
    github = MockGitHub()
    github.seed([ISSUE])
    return github


def test_seeded_issues_are_listable_and_open():
    github = _seeded()
    (issue,) = github.list_issues()
    assert issue["number"] == 42
    assert issue["state"] == "open"
    assert github.get_issue(42)["title"] == "Crash"
    with pytest.raises(ValueError):
        github.get_issue(7)


def test_issue_workflow_updates_state_and_transcript():
    github = _seeded()
    github.comment_on_issue(42, "Investigating.")
    github.close_issue(42)
    issue = github.get_issue(42)
    assert issue["state"] == "closed"
    assert issue["comments"] == ["Investigating."]
    transcript = json.loads(github.transcript())
    assert [a["action"] for a in transcript["actions"]] == ["comment_on_issue", "close_issue"]


def test_pull_requests_number_sequentially_and_latest_wins():
    github = _seeded()
    assert github.latest_pull_request() is None
    github.create_pull_request("First", "body", head="fix-a", base="main")
    second = github.create_pull_request("Second", "body", head="fix-b", base="main")
    assert second["number"] == 2
    latest = github.latest_pull_request()
    assert latest is not None and latest.head == "fix-b"
    assert len(github.list_pull_requests()) == 2


def test_seed_resets_previous_task_state():
    github = _seeded()
    github.create_pull_request("Old", "body", head="old", base="main")
    github.seed([ISSUE])
    assert github.latest_pull_request() is None
    assert github.actions == []
