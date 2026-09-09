"""CLI parsing for Project commands."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from hud.cli import project
from hud.cli.utils.project import Project
from hud.utils.exceptions import HudRequestError

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


def _projects_disabled() -> HudRequestError:
    return HudRequestError(
        "Request failed: Projects are not enabled",
        status_code=403,
        response_json={"error": "forbidden", "detail": "Projects are not enabled"},
    )


def test_bare_project_reports_disabled_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = MagicMock()
    platform.get.side_effect = _projects_disabled()
    monkeypatch.setattr(project, "require_api_key", lambda _: None)
    monkeypatch.setattr(project.PlatformClient, "from_settings", lambda: platform)

    result = CliRunner().invoke(project.project_app)

    assert result.exit_code == 1
    assert "Projects are not enabled for your team" in result.output
    assert "Failed to reach" not in result.output
    platform.get.assert_called_once_with("/projects", params={"limit": 1})


def test_create_distinguishes_disabled_feature_from_admin_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = MagicMock()
    platform.post.side_effect = _projects_disabled()
    monkeypatch.setattr(project, "require_api_key", lambda _: None)
    monkeypatch.setattr(project.PlatformClient, "from_settings", lambda: platform)

    result = CliRunner().invoke(project.project_app, ["create", "browser-evals"])

    assert result.exit_code == 1
    assert "Projects are not enabled for your team" in result.output
    assert "Only team admins" not in result.output


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
