from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from dotenv import dotenv_values

from hud.cli.config import (
    AuthScope,
    DirectoryLink,
    DirectoryState,
    ensure_config_dir,
    get_config_dir,
    get_user_env_path,
    load_env_file,
    parse_env_file,
    render_env_file,
    save_env_file,
    set_env_values,
)
from hud.cli.source import EnvironmentSource
from hud.utils.platform import PlatformClient


def test_parse_env_file_basic():
    contents = """
# comment
KEY=VALUE
EMPTY=
NOEQ
 SPACED = v 
"""  # noqa: W291
    data = parse_env_file(contents)
    assert data["KEY"] == "VALUE"
    assert data["EMPTY"] == ""
    assert data["SPACED"] == "v"
    assert "NOEQ" not in data


def test_render_and_load_roundtrip(tmp_path: Path):
    env = {"A": "1", "B": "a # secret's \\ value\nnext line", "C": '"quoted"'}
    file_path = tmp_path / ".env"
    rendered = render_env_file(env)
    file_path.write_text(rendered, encoding="utf-8")
    loaded = load_env_file(file_path)
    assert loaded == env


def test_set_preserves_credentials_when_another_setting_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    secret = "key # with 'quotes' and \\slashes\nsecond line"
    path = set_env_values({"HUD_API_KEY": secret})
    set_env_values({"HUD_DEFAULT_PROJECT": "example"})
    assert dotenv_values(path)["HUD_API_KEY"] == secret
    assert load_env_file(path)["HUD_DEFAULT_PROJECT"] == "example"


def test_get_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cfg = get_config_dir()
    assert str(cfg).replace("\\", "/").endswith("/.hud")
    assert str(get_user_env_path()).replace("\\", "/").endswith("/.hud/.env")


def test_ensure_and_save(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = ensure_config_dir()
    assert cfg.exists()
    out = save_env_file({"K": "V"})
    assert out.exists()
    assert load_env_file(out) == {"K": "V"}


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
        with pytest.raises(ValueError, match="not the current credentials"):
            DirectoryState(other, directory).load()
    assert state.path == directory / ".hud" / "config.json"


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
    with pytest.raises(ValueError, match="Invalid HUD workspace link"):
        state.load()
    with pytest.raises(ValueError, match="Invalid HUD workspace link"):
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
