from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.utils.config import (
    ensure_config_dir,
    get_config_dir,
    get_user_env_path,
    load_env_file,
    parse_env_file,
    render_env_file,
    save_env_file,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    env = {"A": "1", "B": "2"}
    file_path = tmp_path / ".env"
    rendered = render_env_file(env)
    file_path.write_text(rendered, encoding="utf-8")
    loaded = load_env_file(file_path)
    assert loaded == env


def test_get_paths(monkeypatch, tmp_path: Path):
    from pathlib import Path as _Path

    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
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
