"""Tests for hud rl command."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from hud.cli.rl import (
    _check_scenarios,
    _extract_env_names,
    _extract_scenarios,
    _preflight_validate,
    _submit,
)
from hud.eval.task import Task


def _make_tasks(env_name: str = "test-env", scenario: str = "checkout") -> list[Task]:
    """Create real Task objects matching what the loader returns."""
    return [Task(env={"name": env_name}, scenario=scenario, args={"user": "alice"})]


def _mock_http(status: int = 200, json_data: dict[str, Any] | None = None):
    """Create a patched httpx.AsyncClient that returns a canned response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = json_data or {}
    mock_resp.text = str(json_data)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.post = AsyncMock(return_value=mock_resp)

    return patch("hud.cli.rl.httpx.AsyncClient", return_value=mock_client), mock_client


class TestPreflight:
    """Test preflight validation catches bad envs/scenarios before submission."""

    def test_missing_env_exits(self) -> None:
        http_patch, _ = _mock_http(status=404)
        with http_patch, pytest.raises(click.exceptions.Exit) as exc_info:
            asyncio.run(_preflight_validate(_make_tasks("nonexistent")))
        assert exc_info.value.exit_code == 1

    def test_valid_env_and_scenario_passes(self) -> None:
        http_patch, _ = _mock_http(200, {
            "mcp_config": {},
            "registry_id": "abc",
            "scenarios": ["checkout", "search"],
        })
        with http_patch:
            asyncio.run(_preflight_validate(_make_tasks("my-env", "checkout")))

    def test_scenario_mismatch_exits(self) -> None:
        http_patch, _ = _mock_http(200, {
            "mcp_config": {},
            "registry_id": "abc",
            "scenarios": ["checkout", "search"],
        })
        with http_patch, pytest.raises(click.exceptions.Exit) as exc_info:
            asyncio.run(_preflight_validate(_make_tasks("my-env", "nonexistent")))
        assert exc_info.value.exit_code == 1

    def test_no_scenarios_surface_warns_but_passes(self, capsys) -> None:
        http_patch, _ = _mock_http(200, {"mcp_config": {}, "registry_id": "abc"})
        with http_patch:
            asyncio.run(_preflight_validate(_make_tasks("my-env", "checkout")))
        captured = capsys.readouterr()
        assert "Cannot verify scenarios" in captured.err or "Cannot verify scenarios" in captured.out

    def test_no_envs_in_tasks_skips(self, capsys) -> None:
        asyncio.run(_preflight_validate([{"scenario": "test"}]))
        # No http calls should be made — just a warning
        captured = capsys.readouterr()
        assert "No environment names" in captured.err or "No environment names" in captured.out


class TestSubmit:
    """Test that submission builds correct payload and hits RL service."""

    def test_sends_correct_payload(self) -> None:
        tasks = _make_tasks()
        http_patch, mock_client = _mock_http(200, {
            "job_id": "job-123",
            "model": {"id": "model-456"},
        })
        with http_patch:
            asyncio.run(_submit(tasks, "model-id-123", "medium"))

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["model_id"] == "model-id-123"
            assert payload["config"]["parameters"]["reasoning_effort"] == "medium"
            assert len(payload["dataset"]["tasks"]) == 1
            # Verify task was serialized via model_dump, not passed as raw dict
            task_data = payload["dataset"]["tasks"][0]
            assert task_data["scenario"] == "checkout"
            assert task_data["args"] == {"user": "alice"}

    def test_failure_exits(self) -> None:
        http_patch, _ = _mock_http(400, {"detail": "bad request"})
        with http_patch, pytest.raises(click.exceptions.Exit) as exc_info:
            asyncio.run(_submit(_make_tasks(), "model-id-123", "medium"))
        assert exc_info.value.exit_code == 1


class TestExtractors:
    """Test task field extraction from real Task objects."""

    def test_env_names_from_tasks(self) -> None:
        tasks = [
            Task(env={"name": "a"}, scenario="s1", args={}),
            Task(env={"name": "b"}, scenario="s2", args={}),
            Task(env={"name": "a"}, scenario="s3", args={}),
        ]
        assert _extract_env_names(tasks) == {"a", "b"}

    def test_scenarios_from_tasks(self) -> None:
        tasks = [
            Task(env={"name": "a"}, scenario="s1", args={}),
            Task(env={"name": "a"}, scenario="s2", args={}),
            Task(env={"name": "b"}, scenario="s1", args={}),
        ]
        assert _extract_scenarios(tasks) == {"a": {"s1", "s2"}, "b": {"s1"}}

    def test_check_scenarios_mismatch_exits(self) -> None:
        with pytest.raises(click.exceptions.Exit) as exc_info:
            _check_scenarios("env", {"missing"}, {"scenarios": ["checkout"]})
        assert exc_info.value.exit_code == 1

    def test_check_scenarios_match(self) -> None:
        _check_scenarios("env", {"checkout"}, {"scenarios": ["checkout", "search"]})

    def test_check_scenarios_no_surface(self, capsys) -> None:
        _check_scenarios("env", {"checkout"}, {"mcp_config": {}})
        captured = capsys.readouterr()
        assert "Cannot verify" in captured.err or "Cannot verify" in captured.out
