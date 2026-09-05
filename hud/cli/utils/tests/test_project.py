"""Project lookup and placement precedence for CLI create-and-link flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import typer

from hud.cli.utils.project import (
    Project,
    ProjectNotFound,
    ProjectSource,
    resolve_placement,
    resolve_project,
    resolve_writable_placement,
)
from hud.cli.utils.source import EnvironmentSource
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_ID = "11111111-1111-4111-8111-111111111111"
_BROWSER_ID = "22222222-2222-4222-8222-222222222222"
_READONLY_ID = "33333333-3333-4333-8333-333333333333"


def _record(
    project_id: str, name: str, *, is_default: bool = False, create: bool = True
) -> dict[str, Any]:
    return {
        "id": project_id,
        "name": name,
        "is_default": is_default,
        "capabilities": {"view": True, "create": create, "manage": False},
    }


@pytest.fixture
def calls() -> list[str]:
    """URLs the fake platform transport was asked for."""
    return []


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> PlatformClient:
    """A client whose ``GET /projects`` returns a fixed three-project team."""

    def fake_request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        return {
            "projects": [
                _record(_DEFAULT_ID, "default", is_default=True),
                _record(_BROWSER_ID, "browser-evals"),
                _record(_READONLY_ID, "locked-down", create=False),
            ]
        }

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)
    return PlatformClient("https://api.example", "key")


def _no_settings_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.settings.settings.project", None)


def test_resolve_matches_a_normalized_name(platform: PlatformClient) -> None:
    """A human-typed name resolves through the same normalization the platform applies."""
    assert resolve_project(platform, "Browser Evals").id == _BROWSER_ID
    assert resolve_project(platform, "browser-evals").id == _BROWSER_ID


def test_resolve_matches_an_id(platform: PlatformClient) -> None:
    assert resolve_project(platform, _BROWSER_ID).name == "browser-evals"
    assert resolve_project(platform, _BROWSER_ID.upper()).name == "browser-evals"


def test_resolve_reports_the_visible_alternatives(platform: PlatformClient) -> None:
    with pytest.raises(ProjectNotFound) as excinfo:
        resolve_project(platform, "nope")

    assert [p.name for p in excinfo.value.available] == [
        "default",
        "browser-evals",
        "locked-down",
    ]


def test_flag_outranks_directory_config(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_settings_project(monkeypatch)
    source = EnvironmentSource.open(tmp_path)
    source.save_config({"projectId": _DEFAULT_ID})

    placement = resolve_placement(platform, source, flag="browser-evals")

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source is ProjectSource.FLAG


def test_directory_config_outranks_the_machine_default(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placement is a property of the environment, not of who deploys it."""
    monkeypatch.setattr("hud.settings.settings.project", "default")
    source = EnvironmentSource.open(tmp_path)
    source.save_config({"projectId": _BROWSER_ID})

    placement = resolve_placement(platform, source, flag=None)

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source is ProjectSource.CONFIG


def test_machine_default_applies_to_an_unpinned_directory(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.settings.settings.project", "browser-evals")

    placement = resolve_placement(platform, EnvironmentSource.open(tmp_path), flag=None)

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source is ProjectSource.SETTINGS


def test_unconfigured_placement_sends_no_project_and_makes_no_call(
    platform: PlatformClient,
    calls: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-config path stays free: no project on the wire, no lookup."""
    _no_settings_project(monkeypatch)

    placement = resolve_placement(platform, EnvironmentSource.open(tmp_path), flag=None)

    assert placement.project_id is None
    assert placement.source is ProjectSource.TEAM_DEFAULT
    assert placement.label == "team default Project"
    assert calls == []


def test_placement_resolves_a_project_the_caller_cannot_create_in(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_settings_project(monkeypatch)

    source = EnvironmentSource.open(tmp_path)
    placement = resolve_placement(platform, source, flag="locked-down")
    assert placement.project is not None
    assert placement.project.id == _READONLY_ID

    with pytest.raises(typer.Exit):
        resolve_writable_placement(
            platform,
            source,
            flag="locked-down",
            console=HUDConsole(),
        )


def test_from_record_defaults_capabilities_to_read_only() -> None:
    """A response without capabilities is not assumed writable."""
    assert Project.from_record({"id": "x", "name": "y"}).can_create is False
