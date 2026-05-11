"""Tests for `hud tasks` CLI."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from hud.cli.tasks import tasks_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every CLI command checks for an API key — provide a fake one."""
    from hud.settings import settings

    monkeypatch.setattr(settings, "api_key", "test-api-key")
    monkeypatch.setattr(settings, "hud_api_url", "https://test.example.com")


def _mock_response(status_code: int = 200, json_data: dict[str, Any] | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = str(json_data or "")
    return resp


def test_list_falls_back_to_config_taskset(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hud.cli.tasks.get_taskset_id", lambda: "stored-evalset")
    captured: dict[str, str] = {}

    def fake_request(method: str, path: str, **_: Any) -> MagicMock:
        captured["method"] = method
        captured["path"] = path
        return _mock_response(200, {"tasks": [], "task_stats": []})

    with patch("hud.cli.tasks._request", side_effect=fake_request):
        result = runner.invoke(tasks_app, ["list"])
    assert result.exit_code == 0
    assert captured["path"] == "/tasks/evalsets/stored-evalset/tasks-with-stats"


def test_list_errors_when_no_taskset_anywhere(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hud.cli.tasks.get_taskset_id", lambda: None)
    result = runner.invoke(tasks_app, ["list"])
    assert result.exit_code == 1
    assert "No taskset" in result.output


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_filters_to_matching_slug(runner: CliRunner) -> None:
    payload = {
        "tasks": [
            {"task": {"external_id": "alpha"}, "version": {"id": "v-1"}},
            {"task": {"external_id": "beta"}, "version": {"id": "v-2"}},
        ],
        "task_stats": [
            {"task_version_id": "v-1", "pass_rate": 1.0},
            {"task_version_id": "v-2", "pass_rate": 0.5},
        ],
    }
    with patch("hud.cli.tasks._request", return_value=_mock_response(200, payload)):
        result = runner.invoke(tasks_app, ["show", "beta", "--taskset", "e-1"])
    assert result.exit_code == 0
    assert "beta" in result.output
    assert "alpha" not in result.output
    assert '"pass_rate": 0.5' in result.output


def test_show_404s_when_slug_absent(runner: CliRunner) -> None:
    payload = {"tasks": [{"task": {"external_id": "alpha"}, "version": {}}]}
    with patch("hud.cli.tasks._request", return_value=_mock_response(200, payload)):
        result = runner.invoke(tasks_app, ["show", "ghost", "-t", "e-1"])
    assert result.exit_code == 1
    assert "No task with slug 'ghost'" in result.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_sends_bulk_request_with_set(runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> MagicMock:
        captured["method"] = method
        captured["path"] = path
        captured["json"] = kwargs.get("json")
        return _mock_response(200, {"succeeded": ["t-1", "t-2"], "unchanged": [], "failures": []})

    with patch("hud.cli.tasks._request", side_effect=fake_request):
        result = runner.invoke(
            tasks_app,
            ["status", "alpha", "beta", "--set", "ready", "-t", "e-1"],
        )
    assert result.exit_code == 0
    assert captured["method"] == "PATCH"
    assert captured["path"] == "/tasks/status"
    assert captured["json"] == {
        "slugs": ["alpha", "beta"],
        "evalset_id": "e-1",
        "status": "ready",
    }
    assert "2 updated" in result.output


def test_status_sends_clear_when_clear_flag(runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> MagicMock:
        captured["json"] = kwargs.get("json")
        return _mock_response(200, {"succeeded": [], "unchanged": ["t-1"], "failures": []})

    with patch("hud.cli.tasks._request", side_effect=fake_request):
        result = runner.invoke(tasks_app, ["status", "alpha", "--clear", "-t", "e-1"])
    assert result.exit_code == 0
    assert captured["json"]["clear"] is True
    assert "status" not in captured["json"]
    assert "unchanged" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["status", "alpha", "--set", "ready", "--clear", "-t", "e-1"],
        ["status", "alpha", "-t", "e-1"],
    ],
)
def test_status_requires_exactly_one_mutation(args: list[str], runner: CliRunner) -> None:
    result = runner.invoke(tasks_app, args)
    assert result.exit_code == 1
    assert "exactly one" in result.output


def test_status_all_fetches_slugs_then_mutates(runner: CliRunner) -> None:
    """--all → GET tasks-with-stats, then PATCH /tasks/status with all slugs."""
    calls: list[dict[str, Any]] = []

    def fake_request(method: str, path: str, **kwargs: Any) -> MagicMock:
        calls.append({"method": method, "path": path, "json": kwargs.get("json")})
        if method == "GET":
            return _mock_response(
                200,
                {
                    "tasks": [
                        {"task": {"external_id": "alpha"}, "version": {"id": "v-1"}},
                        {"task": {"external_id": "beta"}, "version": {"id": "v-2"}},
                    ]
                },
            )
        return _mock_response(200, {"succeeded": ["t-1", "t-2"], "unchanged": [], "failures": []})

    with patch("hud.cli.tasks._request", side_effect=fake_request):
        result = runner.invoke(tasks_app, ["status", "--set", "ready", "--all", "-t", "e-1"])
    assert result.exit_code == 0, result.output
    assert calls[0]["method"] == "GET"
    assert calls[1]["method"] == "PATCH"
    assert calls[1]["json"]["slugs"] == ["alpha", "beta"]


@pytest.mark.parametrize(
    "args",
    [
        ["status", "alpha", "--set", "ready", "--all", "-t", "e-1"],
        ["status", "--set", "ready", "-t", "e-1"],
    ],
)
def test_status_requires_slugs_or_all(args: list[str], runner: CliRunner) -> None:
    result = runner.invoke(tasks_app, args)
    assert result.exit_code == 1
    assert "either task slugs or `--all`" in result.output


def test_status_renders_failures(runner: CliRunner) -> None:
    payload = {
        "succeeded": ["t-1"],
        "unchanged": [],
        "failures": [
            {
                "input": "ghost",
                "resolved_id": None,
                "reason": "Task not found",
                "code": "not_found",
            }
        ],
    }
    with patch("hud.cli.tasks._request", return_value=_mock_response(200, payload)):
        result = runner.invoke(
            tasks_app, ["status", "alpha", "ghost", "--set", "ready", "-t", "e-1"]
        )
    # Mixed outcome: some succeeded → exit 0, but failures rendered
    assert result.exit_code == 0
    assert "1 updated" in result.output
    assert "ghost" in result.output
    assert "Task not found" in result.output


# ---------------------------------------------------------------------------
# duplicate
# ---------------------------------------------------------------------------


def test_duplicate_in_place_hits_duplicate_endpoint(runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> MagicMock:
        captured["path"] = path
        captured["json"] = kwargs.get("json")
        return _mock_response(200, {"tasks": [{"id": "t-1-copy"}], "failed_ids": []})

    with patch("hud.cli.tasks._request", side_effect=fake_request):
        result = runner.invoke(tasks_app, ["duplicate", "alpha", "-t", "e-1"])
    assert result.exit_code == 0
    assert captured["path"] == "/tasks/duplicate"
    assert captured["json"] == {"slugs": ["alpha"], "evalset_id": "e-1"}


def test_duplicate_to_target_hits_duplicate_to_taskset(runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> MagicMock:
        captured["path"] = path
        captured["json"] = kwargs.get("json")
        return _mock_response(
            200,
            {"tasks_copied": 2, "versions_copied": 2, "traces_copied": 0},
        )

    with patch("hud.cli.tasks._request", side_effect=fake_request):
        result = runner.invoke(
            tasks_app,
            [
                "duplicate",
                "alpha",
                "beta",
                "--to",
                "e-2",
                "-t",
                "e-1",
                "--on-conflict",
                "lolwat",
            ],
        )
    assert result.exit_code == 0
    assert captured["path"] == "/tasks/duplicate-to-taskset"
    assert captured["json"]["target_taskset_id"] == "e-2"
    assert captured["json"]["slugs"] == ["alpha", "beta"]
    assert captured["json"]["conflict_strategy"] == "lolwat"


# ---------------------------------------------------------------------------
# move
# ---------------------------------------------------------------------------


def test_move_hits_move_to_taskset(runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_request(method: str, path: str, **kwargs: Any) -> MagicMock:
        captured["path"] = path
        captured["json"] = kwargs.get("json")
        return _mock_response(200, {"tasks_moved": 1})

    with patch("hud.cli.tasks._request", side_effect=fake_request):
        result = runner.invoke(
            tasks_app,
            [
                "move",
                "alpha",
                "--to",
                "e-2",
                "-t",
                "e-1",
                "--on-conflict",
                "lolwat",
            ],
        )
    assert result.exit_code == 0
    assert captured["path"] == "/tasks/move-to-taskset"
    assert captured["json"]["target_taskset_id"] == "e-2"
    assert captured["json"]["conflict_strategy"] == "lolwat"


def test_move_requires_to_flag(runner: CliRunner) -> None:
    result = runner.invoke(tasks_app, ["move", "alpha", "-t", "e-1"])
    assert result.exit_code != 0  # typer enforces required option


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_4xx_relays_server_detail_verbatim(runner: CliRunner) -> None:
    err_resp = _mock_response(403, {"detail": "Access denied to this task"})
    with patch("hud.cli.tasks._request", return_value=err_resp):
        result = runner.invoke(tasks_app, ["status", "alpha", "--set", "verified", "-t", "e-1"])
    assert result.exit_code == 1
    assert "Access denied to this task" in result.output
    assert "403" in result.output
