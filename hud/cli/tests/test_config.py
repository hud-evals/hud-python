from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from dotenv import dotenv_values

from hud.cli.app import (
    AuthScope,
    CliError,
    DirectoryLink,
    DirectoryState,
    set_env_values,
)
from hud.cli.deploy import EnvironmentSource
from hud.utils.platform import PlatformClient


def test_set_env_values_merges_and_preserves_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    secret = "key # with 'quotes' and \\slashes\nsecond line"
    path = set_env_values({"HUD_API_KEY": secret, "C": '"quoted"'})
    assert path == tmp_path / ".hud" / ".env"
    set_env_values({"HUD_DEFAULT_PROJECT": "example"})
    loaded = dotenv_values(path, interpolate=False)
    assert loaded["HUD_API_KEY"] == secret
    assert loaded["C"] == '"quoted"'
    assert loaded["HUD_DEFAULT_PROJECT"] == "example"


SCOPE = {
    "origin": "https://api.example",
    "user_id": "11111111-1111-4111-8111-111111111111",
    "team_id": "22222222-2222-4222-8222-222222222222",
}


def test_scoped_links_do_not_cross_origins_users_teams_or_directories(tmp_path: Path) -> None:
    scope = AuthScope.model_validate(SCOPE)
    directory = tmp_path / "environment"
    state = DirectoryState(scope, directory)
    registry = UUID(int=10)
    assert state.update(DirectoryLink(registry_id=registry)) is True
    before = state.path.read_text()
    assert state.update(DirectoryLink(registry_id=registry)) is False
    assert state.path.read_text() == before
    assert state.load().registry_id == registry
    assert DirectoryState(scope, tmp_path / "worktree").load().registry_id is None
    for field, value in [
        ("origin", "https://other.example"),
        ("user_id", str(UUID(int=30))),
        ("team_id", str(UUID(int=40))),
    ]:
        other = AuthScope.model_validate({**SCOPE, field: value})
        with pytest.raises(CliError, match=r"different HUD credentials|was linked against"):
            DirectoryState(other, directory).load()
    assert state.path == directory / ".hud" / "config.json"


def test_legacy_sync_env_is_ignored_and_dropped_on_write(tmp_path: Path) -> None:
    scope = AuthScope.model_validate(SCOPE)
    directory = tmp_path / "environment"
    path = directory / ".hud" / "config.json"
    path.parent.mkdir(parents=True)
    registry = UUID(int=10)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "scope": SCOPE,
                "registry_id": str(registry),
                "sync_env": {str(registry): True},
            }
        )
    )
    state = DirectoryState(scope, directory)
    link = state.load()
    assert link.registry_id == registry
    assert link.scope == scope
    assert state.update(DirectoryLink(taskset_id=UUID(int=11))) is True
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "sync_env" not in stored
    assert stored["taskset_id"] == str(UUID(int=11))


def test_read_leaves_legacy_files_and_home_untouched(tmp_path: Path) -> None:
    directory = tmp_path / "environment"
    legacy = directory / ".hud" / "deploy.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"registryId":"old"}')
    state = DirectoryState(AuthScope.model_validate(SCOPE), directory)
    assert state.load().registry_id is None
    assert EnvironmentSource.open(directory).dockerfile is None
    assert legacy.read_text() == '{"registryId":"old"}'
    assert not (legacy.parent / "config.json").exists()


def test_corrupt_config_is_not_overwritten(tmp_path: Path) -> None:
    state = DirectoryState(AuthScope.model_validate(SCOPE), tmp_path)
    path = state.path
    path.parent.mkdir(parents=True)
    path.write_text('{"broken":')
    with pytest.raises(CliError, match="not a valid HUD workspace config"):
        state.load()
    with pytest.raises(CliError, match="not a valid HUD workspace config"):
        state.update(DirectoryLink())
    assert path.read_text() == '{"broken":'


def test_api_scope_uses_authoritative_identity_not_credentials(monkeypatch) -> None:
    def request(method, url, **kwargs):
        assert method == "GET"
        assert url.endswith("/v2/auth/me")
        return SCOPE

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    first = AuthScope.resolve(PlatformClient("https://API.EXAMPLE:443/", "first-secret"))
    second = AuthScope.resolve(PlatformClient("https://api.example", "rotated-secret"))
    assert first == second == AuthScope.model_validate(SCOPE)
    assert "secret" not in first.model_dump_json()
