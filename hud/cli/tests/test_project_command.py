"""CLI parsing for Project commands."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from hud.cli import project
from hud.cli.utils.project import Project

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def project_record() -> Project:
    return Project(
        id="22222222-2222-4222-8222-222222222222",
        name="browser-evals",
        is_default=False,
        can_create=True,
    )


def test_group_directory_is_inherited_by_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_record: Project,
) -> None:
    pinned: list[str] = []
    monkeypatch.setattr(project, "require_api_key", lambda _: None)
    monkeypatch.setattr(project, "resolve_project", lambda _platform, _ref: project_record)
    monkeypatch.setattr(
        project, "_pin", lambda _project, directory, _console: pinned.append(directory)
    )

    result = CliRunner().invoke(
        project.project_app,
        ["-C", str(tmp_path), "use", "browser-evals"],
    )

    assert result.exit_code == 0
    assert pinned == [str(tmp_path)]


def test_subcommand_directory_overrides_group_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_record: Project,
) -> None:
    pinned: list[str] = []
    override = tmp_path / "override"
    monkeypatch.setattr(project, "require_api_key", lambda _: None)
    monkeypatch.setattr(project, "resolve_project", lambda _platform, _ref: project_record)
    monkeypatch.setattr(
        project, "_pin", lambda _project, directory, _console: pinned.append(directory)
    )

    result = CliRunner().invoke(
        project.project_app,
        ["-C", str(tmp_path), "use", "browser-evals", "-C", str(override)],
    )

    assert result.exit_code == 0
    assert pinned == [str(override)]


def test_group_directory_is_inherited_by_create(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_record: Project,
) -> None:
    pinned: list[str] = []
    platform = MagicMock()
    platform.post.return_value = {
        "id": project_record.id,
        "name": project_record.name,
        "capabilities": {"create": True},
    }
    monkeypatch.setattr(project, "require_api_key", lambda _: None)
    monkeypatch.setattr(project.PlatformClient, "from_settings", lambda: platform)
    monkeypatch.setattr(
        project, "_pin", lambda _project, directory, _console: pinned.append(directory)
    )

    result = CliRunner().invoke(
        project.project_app,
        ["-C", str(tmp_path), "create", "browser-evals"],
    )

    assert result.exit_code == 0
    assert pinned == [str(tmp_path)]
