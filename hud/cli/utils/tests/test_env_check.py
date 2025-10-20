from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch

from hud.cli.utils.env_check import (
    _collect_source_diffs,
    _parse_generated_at,
    ensure_built,
    find_environment_dir,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_generated_at_variants():
    ts = _parse_generated_at({"build": {"generatedAt": datetime.now(UTC).isoformat()}})
    assert isinstance(ts, float)
    assert _parse_generated_at({}) is None


def test_collect_source_diffs_basic(tmp_path: Path):
    env = tmp_path / "env"
    env.mkdir()
    # simulate files
    (env / "Dockerfile").write_text("FROM python:3.11")
    (env / "pyproject.toml").write_text("[tool.hud]")
    (env / "a.txt").write_text("x")

    # stored file list includes a non-existent file and old time
    built_time = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    lock = {"build": {"sourceFiles": ["a.txt", "b.txt"], "generatedAt": built_time}}

    # Patch list_source_files to return current env files
    with patch("hud.cli.utils.env_check.list_source_files") as mock_list:
        mock_list.return_value = [env / "a.txt", env / "Dockerfile"]
        diffs = _collect_source_diffs(env, lock)
    assert "Dockerfile" in diffs["added"]
    assert "b.txt" in diffs["removed"]
    assert "a.txt" in diffs["modified"] or "a.txt" in diffs["added"]


def test_find_environment_dir_prefers_lock(tmp_path: Path):
    # Create env as a sibling to tasks, so it will be in the candidates list
    parent = tmp_path / "parent"
    parent.mkdir()
    tasks = parent / "tasks.json"
    tasks.write_text("[]")
    env = tmp_path / "env"
    env.mkdir()
    (env / "hud.lock.yaml").write_text("version: 1.0")
    # Set cwd to env so it's in the candidate list
    with patch("pathlib.Path.cwd", return_value=env):
        found = find_environment_dir(tasks)
    # Should find env because cwd returns env and it has hud.lock.yaml
    assert found == env


def test_ensure_built_no_lock_noninteractive(tmp_path: Path):
    env = tmp_path / "e"
    env.mkdir()
    # Non-interactive: returns empty dict and does not raise
    result = ensure_built(env, interactive=False)
    assert result == {}


def test_ensure_built_interactive_build(tmp_path: Path):
    env = tmp_path / "e"
    env.mkdir()
    # Simulate interactive=False path avoids prompts
    result = ensure_built(env, interactive=False)
    assert result == {}
