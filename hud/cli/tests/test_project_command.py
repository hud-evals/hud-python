"""Project commands use scoped IDs and the shared output boundary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from typer.testing import CliRunner

from hud.cli import app
from hud.cli.config import AuthScope, DirectoryState
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from pathlib import Path

PROJECT_ID = "22222222-2222-4222-8222-222222222222"
SCOPE = AuthScope(
    origin="https://api.example", user_id="11111111-1111-4111-8111-111111111111", team_id=PROJECT_ID
)
RECORD = {"id": PROJECT_ID, "name": "browser-evals", "capabilities": {"create": True}}


@pytest.fixture(autouse=True)
def platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.hud_api_url", SCOPE.origin)
    monkeypatch.setattr("hud.settings.settings.default_project", None)

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/auth/me"):
            return SCOPE.model_dump(mode="json")
        if url.endswith(f"/projects/{PROJECT_ID}") or method == "POST":
            return RECORD
        raise AssertionError((method, url))

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)


@pytest.mark.parametrize("override", [False, True])
def test_use_honors_directory_options(tmp_path: Path, override: bool) -> None:
    group = tmp_path / "group"
    target = tmp_path / "override" if override else group
    args = ["project", "-C", str(group), "use", PROJECT_ID, "--json"]
    if override:
        args += ["-C", str(target)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["id"] == PROJECT_ID
    assert DirectoryState(SCOPE, target).load().project_id == UUID(PROJECT_ID)
    assert (target / ".hud" / "config.json").exists()
    if override:
        assert DirectoryState(SCOPE, group).load().project_id is None


def test_create_links_group_directory(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["project", "-C", str(tmp_path), "create", "browser-evals", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert DirectoryState(SCOPE, tmp_path).load().project_id == UUID(PROJECT_ID)


def test_use_dry_run_does_not_persist(tmp_path: Path) -> None:
    legacy = tmp_path / "env" / ".hud" / "deploy.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"projectId":"old"}')
    result = CliRunner().invoke(
        app, ["project", "use", PROJECT_ID, "-C", str(legacy.parent.parent), "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert not (legacy.parent / "config.json").exists()
    assert legacy.read_text() == '{"projectId":"old"}'


def test_create_permission_error_is_one_json_document(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: Any, **kwargs: Any) -> None:
        raise HudRequestError("Projects are not enabled", status_code=403)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", denied)
    result = CliRunner().invoke(app, ["project", "create", "browser-evals", "--no-use", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "permission_denied"


def test_list_reads_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    offsets: list[int] = []

    def page(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        from urllib.parse import parse_qs, urlsplit

        offset = int(parse_qs(urlsplit(url).query)["offset"][0])
        offsets.append(offset)
        records = [
            {**RECORD, "id": str(UUID(int=i + 1))} for i in range(offset, min(offset + 50, 51))
        ]
        return {"items": records, "total": 51}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", page)
    result = CliRunner().invoke(app, ["project", "list", "--json"])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)) == 51
    assert offsets == [0, 50]
