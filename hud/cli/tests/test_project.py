"""Project lookup and placement precedence for CLI create-and-link flows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from typer.testing import CliRunner

from hud.cli import AuthScope, CliError, DirectoryLink, DirectoryState
from hud.cli.__main__ import app
from hud.cli.project import (
    Project,
    list_projects,
    require_writable_placement,
    resolve_placement,
    resolve_project,
)
from hud.utils.exceptions import HudRequestError
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
        records = [
            _record(_DEFAULT_ID, "default", is_default=True),
            _record(_BROWSER_ID, "browser-evals"),
            _record(_READONLY_ID, "locked-down", create=False),
        ]
        project_id = url.rsplit("/", 1)[-1]
        if project_id in {_DEFAULT_ID, _BROWSER_ID, _READONLY_ID}:
            return next(record for record in records if record["id"] == project_id)
        search = (kwargs.get("params") or {}).get("search")
        if search:
            records = [record for record in records if search in record["name"]]
        return {"items": records, "total": len(records), "limit": 50, "offset": 0}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)
    return PlatformClient("https://api.example", "key")


def _no_global_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.settings.settings.default_project", None)


def test_list_reads_paginated_items(platform: PlatformClient) -> None:
    assert [project.name for project in list_projects(platform)] == [
        "default",
        "browser-evals",
        "locked-down",
    ]


def test_project_names_require_explicit_selection(platform: PlatformClient) -> None:
    with pytest.raises(ValueError, match="Project ID"):
        resolve_project(platform, "browser-evals")


def test_resolve_matches_an_id(platform: PlatformClient) -> None:
    assert resolve_project(platform, _BROWSER_ID).name == "browser-evals"
    assert resolve_project(platform, _BROWSER_ID.upper()).name == "browser-evals"


def test_flag_outranks_directory_config(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.settings.settings.default_project", "locked-down")
    source = DirectoryLink(project_id=UUID(_DEFAULT_ID))

    placement = resolve_placement(platform, source, flag=_BROWSER_ID)

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source == "--project"


def test_directory_config_applies_without_a_flag(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placement is a property of the environment, not of who deploys it."""
    monkeypatch.setattr("hud.settings.settings.default_project", "default")
    source = DirectoryLink(project_id=UUID(_BROWSER_ID))

    placement = resolve_placement(platform, source, flag=None)

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source == ".hud/config.json"


def test_global_default_applies_to_an_unpinned_directory(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.settings.settings.default_project", _BROWSER_ID)

    placement = resolve_placement(platform, DirectoryLink(), flag=None)

    assert placement.project is not None
    assert placement.project.id == _BROWSER_ID
    assert placement.source == "HUD_DEFAULT_PROJECT"


def test_unconfigured_placement_sends_no_project_and_makes_no_call(
    platform: PlatformClient,
    calls: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-config path stays free: no project on the wire, no lookup."""
    _no_global_default(monkeypatch)
    placement = resolve_placement(platform, DirectoryLink(), flag=None)

    assert placement.project_id is None
    assert placement.source == "team default"
    assert placement.label == "team default Project"
    assert calls == []


def test_placement_resolves_a_project_the_caller_cannot_create_in(
    platform: PlatformClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_global_default(monkeypatch)
    source = DirectoryLink()
    placement = resolve_placement(platform, source, flag=_READONLY_ID)
    assert placement.project is not None
    assert placement.project.id == _READONLY_ID

    with pytest.raises(CliError, match="permission") as info:
        require_writable_placement(placement)
    assert info.value.error == "permission_denied"


def test_from_record_defaults_capabilities_to_read_only() -> None:
    """A response without capabilities is not assumed writable."""
    assert Project.from_record({"id": "x", "name": "y"}).can_create is False


_COMMAND_PROJECT_ID = "22222222-2222-4222-8222-222222222222"
_COMMAND_SCOPE = AuthScope(
    origin="https://api.example",
    user_id="11111111-1111-4111-8111-111111111111",
    team_id=_COMMAND_PROJECT_ID,
)
_COMMAND_RECORD = {
    "id": _COMMAND_PROJECT_ID,
    "name": "browser-evals",
    "capabilities": {"create": True},
}


class TestProjectCommand:
    @pytest.fixture(autouse=True)
    def platform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
        monkeypatch.setattr("hud.settings.settings.hud_api_url", _COMMAND_SCOPE.origin)
        monkeypatch.setattr("hud.settings.settings.default_project", None)

        def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
            if url.endswith("/auth/me"):
                return _COMMAND_SCOPE.model_dump(mode="json")
            if url.endswith(f"/projects/{_COMMAND_PROJECT_ID}") or method == "POST":
                return _COMMAND_RECORD
            raise AssertionError((method, url))

        monkeypatch.setattr("hud.utils.platform.make_request_sync", request)

    @pytest.mark.parametrize("override", [False, True])
    def test_use_honors_directory_options(self, tmp_path: Path, override: bool) -> None:
        group = tmp_path / "group"
        target = tmp_path / "override" if override else group
        args = ["project", "-C", str(group), "use", _COMMAND_PROJECT_ID, "--json"]
        if override:
            args += ["-C", str(target)]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["id"] == _COMMAND_PROJECT_ID
        assert DirectoryState(_COMMAND_SCOPE, target).load().project_id == UUID(_COMMAND_PROJECT_ID)
        assert (target / ".hud" / "config.json").exists()
        if override:
            assert DirectoryState(_COMMAND_SCOPE, group).load().project_id is None

    def test_create_links_group_directory(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            app, ["project", "-C", str(tmp_path), "create", "browser-evals", "--json"]
        )
        assert result.exit_code == 0, result.output
        assert DirectoryState(_COMMAND_SCOPE, tmp_path).load().project_id == UUID(
            _COMMAND_PROJECT_ID
        )

    def test_use_dry_run_does_not_persist(self, tmp_path: Path) -> None:
        legacy = tmp_path / "env" / ".hud" / "deploy.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text('{"projectId":"old"}')
        result = CliRunner().invoke(
            app,
            [
                "project",
                "use",
                _COMMAND_PROJECT_ID,
                "-C",
                str(legacy.parent.parent),
                "--dry-run",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not (legacy.parent / "config.json").exists()
        assert legacy.read_text() == '{"projectId":"old"}'

    def test_create_permission_error_is_one_json_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def denied(*args: Any, **kwargs: Any) -> None:
            raise HudRequestError("Projects are not enabled", status_code=403)

        monkeypatch.setattr("hud.utils.platform.make_request_sync", denied)
        result = CliRunner().invoke(
            app, ["project", "create", "browser-evals", "--no-use", "--json"]
        )
        assert result.exit_code == 1
        assert json.loads(result.stdout)["error"] == "permission_denied"

    def test_list_reads_every_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        offsets: list[int] = []

        def page(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
            offset = int(parse_qs(urlsplit(url).query)["offset"][0])
            offsets.append(offset)
            records = [
                {**_COMMAND_RECORD, "id": str(UUID(int=i + 1))}
                for i in range(offset, min(offset + 50, 51))
            ]
            return {"items": records, "total": 51}

        monkeypatch.setattr("hud.utils.platform.make_request_sync", page)
        result = CliRunner().invoke(app, ["project", "list", "--json"])
        assert result.exit_code == 0, result.output
        assert len(json.loads(result.stdout)) == 51
        assert offsets == [0, 50]
